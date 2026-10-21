"""PyTorch SDPA prefill attention implementation with causal masking and left-padding support."""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    softmax_scale: float = -1.0,
    pad_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, nh, N, D = q.shape
    scale = 1.0 / math.sqrt(D) if softmax_scale <= 0.0 else softmax_scale

    if pad_lens is not None and (pad_lens > 0).any():
        pos = torch.arange(N, device=q.device)
        causal = pos[None, :] <= pos[:, None]
        not_pad = pos[None, :] >= pad_lens[:, None]
        mask = (causal[None, None, :, :] & not_pad[:, None, None, :]) | torch.eye(
            N, dtype=torch.bool, device=q.device
        )
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale
        )

    return F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, scale=scale
    )
