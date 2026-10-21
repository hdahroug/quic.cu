"""Unit tests verifying standalone CUDA kernels against PyTorch reference implementations."""

import pytest
import torch
import torch.nn.functional as F

import kernels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_rmsnorm():
    hidden = 896
    x = torch.randn(8, hidden, device="cuda", dtype=torch.float16)
    w = torch.ones(hidden, device="cuda", dtype=torch.float16)
    eps = 1e-6

    out = kernels.rmsnorm().forward(x, w, eps)

    variance = x.float().pow(2).mean(-1, keepdim=True)
    expected = (x.float() * torch.rsqrt(variance + eps) * w.float()).half()

    assert torch.allclose(out, expected, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_swiglu():
    rows, intermediate = 16, 4864
    gate = torch.randn(rows, intermediate, device="cuda", dtype=torch.float16)
    up = torch.randn(rows, intermediate, device="cuda", dtype=torch.float16)

    out = kernels.swiglu().forward(gate, up)
    expected = (F.silu(gate.float()) * up.float()).half()

    assert torch.allclose(out, expected, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_gemm():
    m, k, n = 8, 896, 4864
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn(n, device="cuda", dtype=torch.float16) * 0.1

    out = kernels.gemm().forward(x, w, b)
    expected = F.linear(x, w, b)

    assert torch.allclose(out, expected, rtol=1e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_sampler():
    b, v = 4, 151936
    logits = torch.randn(b, 1, v, device="cuda", dtype=torch.float16)

    out = kernels.sampler().forward(logits)
    expected = torch.argmax(logits, dim=-1)

    assert torch.equal(out.view(-1), expected.view(-1))
