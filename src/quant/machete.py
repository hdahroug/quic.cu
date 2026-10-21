"""Machete W4A16 prepacking and CUTLASS mixed-precision GEMM execution."""

import torch

import kernels

from .checkpoint import packed_prefixes
from .marlin import unpack_packed_n


class MacheteWeight:
    __slots__ = ("b_q", "group_scales", "group_zeros", "bias",
                 "group_size", "size_k", "size_n")

    def __init__(self, b_q, group_scales, group_zeros, bias,
                 group_size, size_k, size_n):
        self.b_q = b_q
        self.group_scales = group_scales
        self.group_zeros = group_zeros
        self.bias = bias
        self.group_size = group_size
        self.size_k = size_k
        self.size_n = size_n


def prepack_layer(packed, scale, zp_packed, group_size, bias=None,
                  b_q=None):
    size_n, k8 = packed.shape
    size_k = k8 * 8
    if b_q is None:
        bq = packed.contiguous().t()
        b_q = kernels.machete().machete_prepack_B(bq)

    group_scales = scale.transpose(0, 1).contiguous().to(torch.float16)
    z_raw = unpack_packed_n(zp_packed, size_n).to(torch.float16)
    group_zeros = (-group_scales * z_raw).contiguous()

    b_bias = bias.to(torch.float16).contiguous() if bias is not None else None
    return MacheteWeight(b_q, group_scales, group_zeros, b_bias,
                         group_size, size_k, size_n)


def select_layers(weights: dict, group_size: int) -> dict:
    out = {}
    if group_size not in (64, 128):
        return out
    for prefix in packed_prefixes(weights):
        packed = weights[f'{prefix}.weight_packed']
        size_n, k8 = packed.shape
        if (k8 * 8) % 64 or size_n % 128:
            continue
        out[prefix] = (
            packed,
            weights[f'{prefix}.weight_scale'],
            weights[f'{prefix}.weight_zero_point'],
            weights.get(f'{prefix}.bias'),
        )
    return out


def attach(engine, machete_src: dict, group_size: int) -> None:
    if not machete_src:
        return
    device = engine.lm_head.weight.device
    for name, mod in engine.named_modules():
        stage = machete_src.get(name)
        if stage is None:
            continue
        packed, scale, zp, bias = stage
        mod._machete = prepack_layer(
            packed.to(device), scale.to(device), zp.to(device), group_size,
            bias=bias.to(device) if bias is not None else None,
        )


def machete_linear(module, x: torch.Tensor) -> torch.Tensor:
    mw = module._machete
    x2 = x.contiguous().reshape(-1, mw.size_k)
    out = kernels.machete().machete_mm(
        x2, mw.b_q, mw.group_scales, mw.group_zeros, mw.group_size,
    )
    if mw.bias is not None:
        out = out + mw.bias
    return out.reshape(*x.shape[:-1], mw.size_n)
