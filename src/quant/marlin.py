"""Marlin W4A16 prepacking and mixed-precision tensor core GEMM execution."""

import torch

import kernels

from .checkpoint import packed_prefixes


def _scale_perm() -> torch.Tensor:
    perm = []
    for i in range(8):
        for j in range(8):
            perm.append(i + 8 * j)
    return torch.tensor(perm, dtype=torch.int64)


_INTERLEAVE4 = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64)


def unpack_packed_n(weight_zero_point: torch.Tensor, size_n: int) -> torch.Tensor:
    num_words, groups = weight_zero_point.shape
    n = torch.arange(size_n, device=weight_zero_point.device)
    word_idx = n // 8
    nibble = n % 8
    vals = (weight_zero_point[word_idx] >> (4 * nibble).unsqueeze(1)) & 0xF
    return vals.transpose(0, 1).contiguous()


def pack_cols(zp: torch.Tensor, num_bits: int) -> torch.Tensor:
    size_k, size_n = zp.shape
    pack_factor = 32 // num_bits
    cols = zp.view(size_k, size_n // pack_factor, pack_factor).transpose(1, 2)
    shifts = (num_bits * torch.arange(pack_factor, device=zp.device)).view(1, pack_factor, 1)
    words = (cols.to(torch.int64) << shifts).sum(1).to(torch.int32)
    return words.contiguous()


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int, group_size: int):
    scale_perm = _scale_perm()
    s = s.reshape(-1, len(scale_perm))[:, scale_perm]
    return s.reshape(-1, size_n).contiguous()


def marlin_zero_points(zp: torch.Tensor, size_n: int, size_k: int, num_bits: int):
    scale_perm = _scale_perm()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]
    zp = zp.reshape((-1, len(_INTERLEAVE4)))[:, _INTERLEAVE4.to(zp.device)].reshape(-1)
    zp = zp.reshape((-1, size_n)).contiguous()
    return pack_cols(zp, num_bits)


def gptq_repack(b_q_weight: torch.Tensor) -> torch.Tensor:
    k_words, n = b_q_weight.shape
    k = k_words * 8
    assert k % 16 == 0 and n % 64 == 0

    kt = torch.arange(k // 16, device=b_q_weight.device)

    o = torch.arange(128, device=b_q_weight.device)
    th_id = o // 4
    warp = o % 4
    tc_col = th_id // 4
    tc_row = (th_id % 4) * 2
    n_a = warp * 16 + tc_col

    pk = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=b_q_weight.device)
    p = torch.arange(8, device=b_q_weight.device)
    offsets = torch.tensor([0, 1, 8, 9], dtype=torch.int64, device=b_q_weight.device)
    low = pk < 4
    ksel = offsets[torch.where(low, pk, pk - 4)]
    n_off = (~low).to(torch.int64) * 8

    k_loc = tc_row.unsqueeze(1) + ksel
    n_loc = n_a.unsqueeze(1) + n_off

    kidx = kt.unsqueeze(1).unsqueeze(2) * 16 + k_loc
    shifts = (p * 4).unsqueeze(0).unsqueeze(0)

    out = torch.empty((k // 16, n * 2), dtype=torch.int32, device=b_q_weight.device)

    for nt_id in range(n // 64):
        nidx = nt_id * 64 + n_loc
        vals = b_q_weight[kidx // 8, nidx]
        vals = ((vals >> (4 * (kidx % 8))) & 0xF).to(torch.int64)
        words = (vals << shifts).sum(-1).to(torch.int32)
        nt0 = nt_id * 128
        out[:, nt0:nt0 + 128] = words
    return out


def _scale_perm_single() -> torch.Tensor:
    perm: list[int] = []
    for i in range(4):
        perm.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return torch.tensor(perm, dtype=torch.int64)


def marlin_permute_bias(bias: torch.Tensor) -> torch.Tensor:
    perm = _scale_perm_single()
    origin_shape = bias.shape
    return bias.reshape((-1, len(perm)))[:, perm].reshape(*origin_shape).contiguous()


def prepack_layer(packed: torch.Tensor, scale: torch.Tensor, zp_packed: torch.Tensor,
                  group_size: int = 32, bias: torch.Tensor | None = None):
    size_n, k8 = packed.shape
    size_k = k8 * 8
    num_groups = scale.shape[1]
    assert size_n % 64 == 0 and size_k % 16 == 0
    assert num_groups == size_k // group_size

    bq = packed.t().contiguous()
    b_q_weight = gptq_repack(bq)

    s = scale.transpose(0, 1).contiguous().to(torch.float16)
    b_scales = marlin_permute_scales(s, size_k, size_n, group_size)

    zp_u4 = unpack_packed_n(zp_packed, size_n)
    b_zeros = marlin_zero_points(zp_u4, size_n, num_groups, 4)

    b_bias_orig = bias.to(torch.float16).contiguous() if bias is not None and bias.numel() else None
    b_bias = marlin_permute_bias(b_bias_orig) if b_bias_orig is not None else None

    return b_q_weight, b_scales, b_zeros, b_bias, size_k, size_n, num_groups


class MarlinWeight:
    __slots__ = ("b_q_weight", "b_scales", "b_zeros", "bias", "size_k", "size_n",
                 "num_groups", "workspace", "b_type_id")

    def __init__(self, b_q_weight, b_scales, b_zeros, bias, size_k, size_n, num_groups):
        self.b_q_weight = b_q_weight
        self.b_scales = b_scales
        self.b_zeros = b_zeros
        self.bias = bias
        self.size_k = size_k
        self.size_n = size_n
        self.num_groups = num_groups
        self.workspace = None
        self.b_type_id = None

    @classmethod
    def from_checkpoint_layer(cls, packed, scale, zp_packed, bias=None, group_size=32,
                              b_type_id=None, sms=16):
        bq, bs, bz, bb, k, n, g = prepack_layer(packed, scale, zp_packed, group_size, bias)
        w = cls(bq, bs, bz, bb, k, n, g)
        w.b_type_id = b_type_id
        w.workspace = torch.zeros(sms, dtype=torch.int, device="cuda")
        return w


def prepack_all(weights: dict, group_size: int = 32, names: set = None) -> dict:
    out = {}
    for prefix in (packed_prefixes(weights) if names is None else names):
        try:
            bq, bs, bz, bb, k, n, g = prepack_layer(
                weights[f'{prefix}.weight_packed'],
                weights[f'{prefix}.weight_scale'],
                weights[f'{prefix}.weight_zero_point'],
                group_size,
                bias=weights.get(f'{prefix}.bias'),
            )
        except Exception:
            continue
        out[prefix] = MarlinWeight(bq, bs, bz, bb, k, n, g)
    return out


def attach(engine, marlin_map: dict) -> None:
    if not marlin_map:
        return
    marlin = kernels.marlin()
    device = engine.lm_head.weight.device
    b_type_id = marlin.k_u4()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    for name, mod in engine.named_modules():
        mw = marlin_map.get(name)
        if mw is None:
            continue
        mw.b_q_weight = mw.b_q_weight.to(device)
        mw.b_scales = mw.b_scales.to(device)
        mw.b_zeros = mw.b_zeros.to(device)
        mw.bias = mw.bias.to(device) if mw.bias is not None else None
        mw.workspace = torch.zeros(sms, dtype=torch.int, device=device)
        mw.b_type_id = b_type_id
        mod._marlin = mw


def marlin_linear(module, x: torch.Tensor) -> torch.Tensor:
    mw = module._marlin
    marlin = kernels.marlin()
    x2 = x.contiguous().reshape(-1, mw.size_k)
    out = marlin.forward(
        a=x2, c=None, b_q_weight=mw.b_q_weight, b_bias=mw.bias,
        b_scales=mw.b_scales, a_scales=None, global_scale=None,
        b_zeros=mw.b_zeros, workspace=mw.workspace, b_type_id=mw.b_type_id,
        size_m=x2.size(0), size_n=mw.size_n, size_k=mw.size_k,
        use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
    )
    return out.reshape(*x.shape[:-1], mw.size_n)
