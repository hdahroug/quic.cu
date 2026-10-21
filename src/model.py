"""Qwen2.5-0.5B causal language model graph with fused CUDA kernel operators."""

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DynamicCache,
    PreTrainedTokenizerFast,
    Qwen2Config,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import kernels
from kernels import flash_attn_prefill_pytorch
from config import MODEL_ROOT
from quant import machete_prepack, marlin_prepack
from quant.checkpoint import QUANT_SUFFIXES, detect_group_size, packed_prefixes
from quant.machete import machete_linear
from quant.marlin import marlin_linear

DTYPE = torch.float16

_HIDDEN = 896
_INTERMEDIATE = 4864
_HEAD_DIM = 64
_SWIGLU_MAX_ROWS = 65535


def _linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    if getattr(module, "_machete", None) is not None:
        return machete_linear(module, x)
    if getattr(module, "_marlin", None) is not None:
        return marlin_linear(module, x)
    return kernels.gemm().forward(x, module.weight, module.bias)


def _decode_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    pad_lens: torch.Tensor,
) -> torch.Tensor:
    return kernels.flash_attn_decode().forward(query, key, value, scale, pad_lens)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None]
        .expand(batch, kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, kv_heads * n_rep, seq_len, head_dim)
    )


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=DTYPE))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernels.rmsnorm().forward(x.contiguous(), self.weight, self.eps)


class _Rotary(nn.Module):
    def __init__(self, head_dim: int, theta: float, max_positions: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.max_positions = max_positions
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cos is None or self._sin is None:
            exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            inv_freq = (1.0 / (self.theta**exponent)).to(position_ids.device).contiguous()
            self._cos, self._sin = kernels.rope().init_table(inv_freq, self.max_positions)
        return self._cos[position_ids], self._sin[position_ids]


class _MLP(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=DTYPE)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=DTYPE)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = _linear(self.gate_proj, x)
        up = _linear(self.up_proj, x)
        rows = gate.numel() // _INTERMEDIATE
        if rows <= _SWIGLU_MAX_ROWS:
            act = kernels.swiglu().forward(gate.view(-1, _INTERMEDIATE), up.view(-1, _INTERMEDIATE))
            act = act.view(gate.shape)
        else:
            act = F.silu(gate) * up
        return _linear(self.down_proj, act)


class _BatchMeta:
    def __init__(
        self,
        pad_lens: torch.Tensor,
        use_pytorch_prefill: bool,
        use_custom_prefill: bool,
        sdpa_mask: torch.Tensor | None,
    ) -> None:
        self.pad_lens = pad_lens
        self.use_pytorch_prefill = use_pytorch_prefill
        self.use_custom_prefill = use_custom_prefill
        self.sdpa_mask = sdpa_mask


class _Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=True, dtype=DTYPE)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True, dtype=DTYPE)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True, dtype=DTYPE)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False, dtype=DTYPE)

    def _heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, n_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: DynamicCache,
        meta: _BatchMeta,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        query = self._heads(_linear(self.q_proj, hidden_states), self.num_heads)
        key = self._heads(_linear(self.k_proj, hidden_states), self.num_kv_heads)
        value = self._heads(_linear(self.v_proj, hidden_states), self.num_kv_heads)

        rope = kernels.rope()
        query = rope.forward(query, cos, sin)
        key = rope.forward(key, cos, sin)

        key, value = cache.update(key, value, self.layer_idx)

        if seq_len == 1:
            attn = _decode_attn(query, key, value, self.scaling, meta.pad_lens)
        elif meta.use_pytorch_prefill:
            attn = flash_attn_prefill_pytorch.forward(
                query,
                _repeat_kv(key, self.num_kv_groups),
                _repeat_kv(value, self.num_kv_groups),
                is_causal=True,
                softmax_scale=self.scaling,
                pad_lens=meta.pad_lens,
            )
        elif meta.use_custom_prefill:
            prefill_mod = kernels.flash_attn_prefill()
            if kernels.is_hopper():
                attn = prefill_mod.forward(
                    query,
                    _repeat_kv(key, self.num_kv_groups),
                    _repeat_kv(value, self.num_kv_groups),
                    True,
                    self.scaling,
                    meta.pad_lens,
                )
            else:
                q_f = query.float()
                k_f = _repeat_kv(key, self.num_kv_groups).float()
                v_f = _repeat_kv(value, self.num_kv_groups).float()
                attn = prefill_mod.forward(
                    q_f, k_f, v_f, True, self.scaling, meta.pad_lens
                ).to(query.dtype)
        else:
            attn = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=meta.sdpa_mask,
                scale=self.scaling,
                enable_gqa=True,
            )
        return _linear(self.o_proj, attn.transpose(1, 2).reshape(batch, seq_len, -1))


class _DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = _Attention(config, layer_idx)
        self.mlp = _MLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: DynamicCache,
        meta: _BatchMeta,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.input_layernorm(x), cos, sin, cache, meta)
        x = residual + x
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        return residual + x


class _Model(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=DTYPE)
        self.layers = nn.ModuleList([_DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        if hasattr(config, "rope_parameters") and config.rope_parameters and "rope_theta" in config.rope_parameters:
            theta = float(config.rope_parameters["rope_theta"])
        elif hasattr(config, "rope_theta"):
            theta = float(config.rope_theta)
        else:
            theta = 1000000.0
        self.rotary = _Rotary(head_dim, theta, config.max_position_embeddings)


class QUICForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = _Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=DTYPE)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.prefill_backend = "pytorch"

    def _batch_meta(self, attention_mask: torch.Tensor, seq_len: int, kv_len: int) -> _BatchMeta:
        pad_lens = (attention_mask == 0).sum(dim=1).to(dtype=torch.int32, device="cuda")
        if seq_len == 1:
            return _BatchMeta(pad_lens, False, False, None)

        if self.prefill_backend == "pytorch":
            return _BatchMeta(pad_lens, True, False, None)
        if self.prefill_backend == "quic":
            return _BatchMeta(pad_lens, False, True, None)

        batch = attention_mask.shape[0]
        q_pos = torch.arange(seq_len, device="cuda").unsqueeze(1)
        k_pos = torch.arange(kv_len, device="cuda").unsqueeze(0)
        causal = k_pos <= q_pos
        padding = attention_mask.bool().unsqueeze(1)
        mask = (causal & padding).unsqueeze(1).expand(batch, 1, seq_len, kv_len)
        return _BatchMeta(pad_lens, False, False, mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> CausalLMOutputWithPast:
        batch, seq_len = input_ids.shape
        cache = past_key_values if past_key_values is not None else DynamicCache()
        past_len = cache.get_seq_length()
        kv_len = past_len + seq_len

        if attention_mask is None:
            attention_mask = torch.ones(batch, kv_len, device=input_ids.device)
        if attention_mask.shape != (batch, kv_len):
            raise ValueError(f"attention_mask {tuple(attention_mask.shape)} does not match (batch={batch}, kv_len={kv_len})")
        if position_ids is None:
            position_ids = torch.arange(past_len, kv_len, device=input_ids.device).expand(batch, -1)

        meta = self._batch_meta(attention_mask, seq_len, kv_len)
        cos, sin = self.model.rotary(position_ids)

        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, cos, sin, cache, meta)
        hidden_states = self.model.norm(hidden_states)

        return CausalLMOutputWithPast(
            logits=_linear(self.lm_head, hidden_states),
            past_key_values=cache if use_cache else None,
        )


def load(
    path: str = MODEL_ROOT,
    prefill_backend: str = "pytorch",
    use_marlin: bool = False,
    use_machete: bool = False,
    tokenizer_path: str | None = None,
) -> tuple[QUICForCausalLM, PreTrainedTokenizerFast]:
    path = os.path.expanduser(path)
    if not os.path.exists(os.path.join(path, "config.json")) and os.path.isdir(os.path.join(path, "Qwen2.5-0.5B")):
        path = os.path.join(path, "Qwen2.5-0.5B")
    tok_path = tokenizer_path or path
    if tokenizer_path is None and not os.path.exists(os.path.join(tok_path, "tokenizer.json")) and os.path.isdir(os.path.join(tok_path, "Qwen2.5-0.5B")):
        tok_path = os.path.join(tok_path, "Qwen2.5-0.5B")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
    except Exception:
        fallback = os.environ.get("QWEN25_FP16_PATH", MODEL_ROOT)
        tokenizer = AutoTokenizer.from_pretrained(fallback)
    config = AutoConfig.from_pretrained(path)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert isinstance(config, Qwen2Config)

    with torch.device("cuda"):
        model = QUICForCausalLM(config)
    if prefill_backend is not None:
        model.prefill_backend = prefill_backend
    weights = load_file(os.path.join(path, "model.safetensors"))

    prefixes = packed_prefixes(weights)
    marlin_map, machete_src = {}, {}
    if prefixes:
        with open(os.path.join(path, "config.json")) as handle:
            group_size = detect_group_size(json.load(handle))
        if use_machete and kernels.is_hopper():
            machete_src = machete_prepack.select_layers(weights, group_size)
        with torch.device("cpu"):
            marlin_map = marlin_prepack.prepack_all(
                weights, group_size, names=prefixes - set(machete_src)
            )
        uncovered = prefixes - set(marlin_map) - set(machete_src)
        if uncovered:
            raise RuntimeError(f"no W4A16 kernel for layers: {sorted(uncovered)}")

    missing, unexpected = model.load_state_dict(weights, strict=False)
    allowed_missing = {f"{prefix}.weight" for prefix in prefixes}
    if config.tie_word_embeddings:
        allowed_missing.add("lm_head.weight")
    bad_missing = sorted(set(missing) - allowed_missing)
    bad_unexpected = sorted(k for k in unexpected if not (prefixes and k.endswith(QUANT_SUFFIXES)))
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"weight mismatch: missing={bad_missing} unexpected={bad_unexpected}")

    if prefixes:
        marlin_prepack.attach(model, marlin_map)
        machete_prepack.attach(model, machete_src, group_size)
        for name, module in model.named_modules():
            if name in marlin_map or name in machete_src:
                module.weight = nn.Parameter(torch.empty(0), requires_grad=False)

    return model.eval(), tokenizer
