"""Unit tests verifying prefill and Split-K decode attention kernels against PyTorch SDPA."""

import pytest
import torch
import torch.nn.functional as F

import kernels

BATCH, HEADS, HEAD_DIM = 4, 14, 64
SCALE = HEAD_DIM**-0.5
ATOL = 1e-2


def _prefill_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, pad_lens: torch.Tensor
) -> torch.Tensor:
    prefill = kernels.flash_attn_prefill()
    if kernels.is_hopper():
        return prefill.forward(q, k, v, True, scale, pad_lens)
    return prefill.forward(q.float(), k.float(), v.float(), True, scale, pad_lens).to(q.dtype)


def random_qkv(seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seq_len)
    shape = (BATCH, HEADS, seq_len, HEAD_DIM)
    q, k, v = (
        torch.randn(shape, device="cuda", generator=generator).half() for _ in range(3)
    )
    return q, k, v


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
@pytest.mark.parametrize("seq_len", [2, 64, 128])
def test_prefill_matches_sdpa_with_left_padding(seq_len: int):
    q, k, v = random_qkv(seq_len)
    pad_lens = torch.tensor([0, 1, seq_len // 2, seq_len - 1], device="cuda", dtype=torch.int32)
    out = _prefill_attn(q, k, v, SCALE, pad_lens)

    pos = torch.arange(seq_len, device="cuda")
    causal = pos[None, :] <= pos[:, None]
    not_pad = pos[None, :] >= pad_lens[:, None]
    mask = (causal[None, None] & not_pad[:, None, None, :]) | torch.eye(
        seq_len, dtype=torch.bool, device="cuda"
    )
    expected = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=mask, scale=SCALE
    )

    real_rows = not_pad[:, None, :, None].expand_as(out)
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()
    assert torch.allclose(out[real_rows].float(), expected[real_rows], atol=ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
@pytest.mark.parametrize("kv_len", [16, 64, 256])
def test_decode_matches_sdpa_with_left_padding(kv_len: int):
    q_shape = (BATCH, HEADS, 1, HEAD_DIM)
    kv_shape = (BATCH, 2, kv_len, HEAD_DIM)

    generator = torch.Generator(device="cuda").manual_seed(kv_len)
    q = torch.randn(q_shape, device="cuda", generator=generator).half()
    k = torch.randn(kv_shape, device="cuda", generator=generator).half()
    v = torch.randn(kv_shape, device="cuda", generator=generator).half()

    pad_lens = torch.tensor([0, 2, kv_len // 4, kv_len - 1], device="cuda", dtype=torch.int32)
    decode_mod = kernels.flash_attn_decode()
    out = decode_mod.forward(q, k, v, SCALE, pad_lens)

    k_rep = k.repeat_interleave(7, dim=1)
    v_rep = v.repeat_interleave(7, dim=1)
    pos = torch.arange(kv_len, device="cuda")
    not_pad = pos[None, :] >= pad_lens[:, None]
    mask = not_pad[:, None, None, :].expand(BATCH, HEADS, 1, kv_len)

    expected = F.scaled_dot_product_attention(
        q.float(), k_rep.float(), v_rep.float(), attn_mask=mask, scale=SCALE
    ).to(q.dtype)

    assert out.shape == (BATCH, HEADS, 1, HEAD_DIM)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), expected.float(), atol=ATOL)
