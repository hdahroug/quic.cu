"""End-to-end parity verification comparing QUIC generation against Hugging Face reference."""

import pytest
import torch

import model as quic_model
import model_hf as hf_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_logits_max_absolute_difference(model_paths):
    fp16_path = model_paths["fp16"]
    quic_net, tokenizer = quic_model.load(fp16_path)
    hf_net, _ = hf_model.load(fp16_path)

    prompt = "Large language models require efficient GPU kernel fusion and continuous batching."
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device="cuda")

    with torch.no_grad():
        quic_logits = quic_net(input_ids).logits
        hf_logits = hf_net(input_ids).logits

    max_abs_diff = torch.max(torch.abs(quic_logits - hf_logits)).item()
    mean_abs_diff = torch.mean(torch.abs(quic_logits - hf_logits)).item()

    print(f"\n[Parity Test] Logits Max Absolute Difference: {max_abs_diff:.6f}")
    print(f"[Parity Test] Logits Mean Absolute Difference: {mean_abs_diff:.6f}")

    assert mean_abs_diff < 0.05, f"Mean logits divergence too large: {mean_abs_diff}"
    assert torch.equal(quic_logits[:, -1, :].argmax(dim=-1), hf_logits[:, -1, :].argmax(dim=-1)), "Next-token argmax mismatch!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_first_100_tokens_word_for_word_parity(model_paths):
    fp16_path = model_paths["fp16"]
    quic_net, tokenizer = quic_model.load(fp16_path)
    hf_net, _ = hf_model.load(fp16_path)

    prompt = "Explain the fundamental principles of physics and how gravity works:"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device="cuda")
    target_tokens = 100

    quic_generated = []
    with torch.no_grad():
        out = quic_net(input_ids)
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        quic_generated.append(next_token.item())
        cache = out.past_key_values

        curr = next_token
        for _ in range(target_tokens - 1):
            out = quic_net(curr, past_key_values=cache)
            curr = out.logits[:, -1:, :].argmax(dim=-1)
            quic_generated.append(curr.item())

    with torch.no_grad():
        hf_out = hf_net.generate(
            input_ids,
            max_new_tokens=target_tokens,
            do_sample=False,
            use_cache=True,
        )
        hf_generated = hf_out[0, input_ids.shape[1] :].tolist()

    quic_text = tokenizer.decode(quic_generated, skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_generated, skip_special_tokens=True)

    print("\n--- QUIC Output (100 tokens) ---")
    print(quic_text)
    print("\n--- HF Output (100 tokens) ---")
    print(hf_text)

    assert (
        quic_generated == hf_generated
    ), f"Token mismatch! First diff at token {next(i for i, (q, h) in enumerate(zip(quic_generated, hf_generated)) if q != h)}"

    assert quic_text == hf_text, "Decoded text does not match word-for-word!"
