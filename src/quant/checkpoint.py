"""Checkpoint inspection and weight dequantization utilities for packed formats."""

import torch

QUANT_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_zero_point", ".weight_shape")


def packed_prefixes(weights: dict) -> set:
    return {k[:-len('.weight_packed')] for k in weights if k.endswith('.weight_packed')}


def detect_group_size(raw_config: dict, default: int = 32) -> int:
    qc = raw_config.get("quantization_config") or {}
    if isinstance(qc.get("group_size"), int):
        return qc["group_size"]
    for group in (qc.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        if isinstance(weights.get("group_size"), int):
            return weights["group_size"]
    return default


def dequantize(weights: dict, prefix: str, group_size: int) -> torch.Tensor:
    packed = weights[f"{prefix}.weight_packed"]
    scale = weights[f"{prefix}.weight_scale"].float()
    zero_packed = weights[f"{prefix}.weight_zero_point"]
    out_features = packed.shape[0]

    nibble_shifts = 4 * torch.arange(8, device=packed.device)
    q = ((packed[:, :, None] >> nibble_shifts) & 0xF).reshape(out_features, -1)
    rows = torch.arange(out_features, device=packed.device)
    zero = (zero_packed[rows // 8] >> (4 * (rows % 8))[:, None]) & 0xF

    zero = zero.repeat_interleave(group_size, dim=1)
    scale = scale.repeat_interleave(group_size, dim=1)
    return (q - zero).float() * scale


def dequantized_state_dict(weights: dict, group_size: int) -> dict:
    state = {k: v for k, v in weights.items() if not k.endswith(QUANT_SUFFIXES)}
    for prefix in packed_prefixes(weights):
        state[f"{prefix}.weight"] = dequantize(weights, prefix, group_size)
    return state
