"""QUIC Command-Line Generation Entrypoint."""

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def run_local(args):
    import torch
    import model as quic_model
    import model_hf as hf_model

    fp16_path = args.model_path or os.environ.get("QWEN25_FP16_PATH", "/weights/Qwen2.5-0.5B")
    awq_path = (
        args.awq_path
        or (args.model_path if args.path in ("quic_quant", "hf_quant") else None)
        or os.environ.get("QWEN25_AWQ_PATH", "/weights/Qwen2.5-0.5B-AWQ")
    )
    tok_path = args.tokenizer_path or fp16_path
    fp16_path = os.path.expanduser(fp16_path)
    awq_path = os.path.expanduser(awq_path)
    tok_path = os.path.expanduser(tok_path)

    print(f"[QUIC] Loading model for path: '{args.path}'...")
    if args.path == "quic":
        net, tokenizer = quic_model.load(fp16_path, prefill_backend="pytorch", tokenizer_path=tok_path)
    elif args.path == "quic_quant":
        net, tokenizer = quic_model.load(awq_path, use_marlin=True, tokenizer_path=tok_path)
    elif args.path == "hf":
        net, tokenizer = hf_model.load(fp16_path, tokenizer_path=tok_path)
        net = net.to("cuda").eval()
    elif args.path == "hf_quant":
        net, tokenizer = hf_model.load(awq_path, tokenizer_path=tok_path)
        net = net.to("cuda").eval()

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device="cuda")

    print(f"\n[Prompt] {args.prompt}")
    generated = []

    with torch.no_grad():
        out = net(input_ids)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())
        cache = out.past_key_values

        curr = next_tok
        for _ in range(args.max_tokens - 1):
            if next_tok.item() == tokenizer.eos_token_id:
                break
            out = net(curr, past_key_values=cache)
            curr = out.logits[:, -1:, :].argmax(dim=-1)
            generated.append(curr.item())

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"[Generated ({len(generated)} tokens)]\n{output_text}\n")


try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    volume = modal.Volume.from_name("qwen2-weights-volume")
    image = (
        modal.Image.from_registry("nvidia/cuda:12.6.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("git", "build-essential", "ninja-build")
        .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu126")
        .pip_install(
            "transformers<4.45.0",
            "accelerate",
            "safetensors",
            "tokenizers",
            "numpy>=1.26.0",
            "ninja",
        )
        .add_local_dir(ROOT_DIR, remote_path="/root/quic", copy=False)
    )

    app = modal.App("quic_cli")


    @app.function(
        image=image,
        gpu="L4",
        timeout=600,
        volumes={"/weights": volume},
    )
    def remote_generate(
        path: str,
        prompt: str,
        max_tokens: int,
        model_path: str | None = None,
        awq_path: str | None = None,
        tokenizer_path: str | None = None,
    ):
        import subprocess

        env = dict(
            os.environ,
            PYTHONPATH="/root/quic/src:/root/quic",
            QWEN25_FP16_PATH="/weights/Qwen2.5-0.5B",
            QWEN25_AWQ_PATH="/weights/Qwen2.5-0.5B-AWQ",
            TORCH_EXTENSIONS_DIR="/weights/_torchcache_serving",
        )
        cmd = [
            sys.executable,
            "run.py",
            "--local",
            "--path", path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
        ]
        if model_path:
            cmd.extend(["--model-path", model_path])
        if awq_path:
            cmd.extend(["--awq-path", awq_path])
        if tokenizer_path:
            cmd.extend(["--tokenizer-path", tokenizer_path])
        res = subprocess.run(cmd, cwd="/root/quic", env=env, capture_output=True, text=True)
        return res.stdout, res.stderr


def main():
    parser = argparse.ArgumentParser(description="QUIC Generation CLI")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt text")
    parser.add_argument("--path", type=str, choices=["quic", "quic_quant", "hf", "hf_quant"], default="quic", help="Execution path")
    parser.add_argument("--max-tokens", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FP16 model weights directory")
    parser.add_argument("--awq-path", type=str, default=None, help="Path to AWQ 4-bit model weights directory")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to tokenizer directory (defaults to model-path)")
    parser.add_argument("--local", action="store_true", help="Run generation on local GPU")
    parser.add_argument("--modal", action="store_true", help="Run generation on Modal cloud GPU")
    parser.add_argument("--gpu", type=str, default="L4", help="Modal GPU type (L4, H100, etc.)")
    args = parser.parse_args()

    if not args.local and not args.modal:
        try:
            import torch

            if torch.cuda.is_available():
                args.local = True
            else:
                args.modal = True
        except ImportError:
            args.modal = True

    if args.modal:
        if not HAS_MODAL:
            print("Error: 'modal' is not installed. Install it with 'pip install modal'.", file=sys.stderr)
            sys.exit(1)
        print(f"[QUIC] Executing on Modal ({args.gpu})...")
        with app.run():
            stdout, stderr = remote_generate.with_options(gpu=args.gpu).remote(
                args.path,
                args.prompt,
                args.max_tokens,
                args.model_path,
                args.awq_path,
                args.tokenizer_path,
            )
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
    else:
        run_local(args)


if __name__ == "__main__":
    main()

