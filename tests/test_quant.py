"""Unit tests verifying W4A16 AWQ quantized linear module binding and execution."""

import os
import pytest
import torch

from quant.checkpoint import dequantize, detect_group_size, packed_prefixes
import model as quic_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_marlin_linear_binding(model_paths):
    awq_path = model_paths["awq"]
    if not os.path.exists(os.path.join(awq_path, "model.safetensors")):
        pytest.skip(f"AWQ weights not found at {awq_path}")

    net, _ = quic_model.load(awq_path, use_marlin=True)

    marlin_count = 0
    for name, module in net.named_modules():
        if getattr(module, "_marlin", None) is not None:
            marlin_count += 1

    assert marlin_count == 168, f"Expected 168 Marlin modules, found {marlin_count}"
