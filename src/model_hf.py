"""Hugging Face reference baseline wrapper for FP16 and AWQ weights."""

import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast, Qwen2ForCausalLM

from config import MODEL_ROOT
from quant.checkpoint import dequantized_state_dict, detect_group_size, packed_prefixes


def load(
    path: str = MODEL_ROOT,
    tokenizer_path: str | None = None,
) -> tuple[Qwen2ForCausalLM, PreTrainedTokenizerFast]:
    path = os.path.expanduser(path)
    if not os.path.exists(os.path.join(path, "config.json")) and os.path.isdir(os.path.join(path, "Qwen2.5-0.5B")):
        path = os.path.join(path, "Qwen2.5-0.5B")
    tok_dir = tokenizer_path if tokenizer_path and os.path.exists(tokenizer_path) else path
    if tokenizer_path is None and not os.path.exists(os.path.join(tok_dir, "tokenizer.json")) and os.path.isdir(os.path.join(tok_dir, "Qwen2.5-0.5B")):
        tok_dir = os.path.join(tok_dir, "Qwen2.5-0.5B")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    except Exception:
        fallback_path = os.environ.get("QWEN25_FP16_PATH", MODEL_ROOT)
        tokenizer = AutoTokenizer.from_pretrained(fallback_path)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    weights = load_file(os.path.join(path, "model.safetensors"))
    if not packed_prefixes(weights):
        del weights
        return Qwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.float16), tokenizer

    with open(os.path.join(path, "config.json")) as handle:
        group_size = detect_group_size(json.load(handle))
    model = Qwen2ForCausalLM(AutoConfig.from_pretrained(path)).to(torch.float16)
    missing, unexpected = model.load_state_dict(
        dequantized_state_dict(weights, group_size), strict=False
    )
    if set(missing) - {"lm_head.weight"} or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    model.tie_weights()
    return model.eval(), tokenizer
