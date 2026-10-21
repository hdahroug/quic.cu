"""Continuous and static batching serving engine with dynamic KV-cache slot allocation."""

import itertools
from collections import deque
from typing import cast

import torch
import torch.nn.functional as F
from transformers import BatchEncoding, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import EngineConfig
from sequence import SequenceState
from metrics import MetricsRegistry, Span
from scheduler import ContinuousAdmissionPolicy, StaticAdmissionPolicy
import model as quic_model

torch.set_default_device("cuda")

BATCHING_POLICIES = {
    "static": StaticAdmissionPolicy,
    "continuous": ContinuousAdmissionPolicy,
}


class Engine:
    def __init__(
        self,
        registry: MetricsRegistry | None = None,
        max_batch_size: int = 8,
        config: EngineConfig | None = None,
        model_module=None,
    ) -> None:
        if config is None:
            config = EngineConfig()
        self.max_batch_size = max_batch_size
        self.running: list[SequenceState] = []
        self.waiting: deque[SequenceState] = deque()

        load_options: dict[str, str | bool] = {}
        if config.model_path is not None:
            load_options["path"] = config.model_path
        if config.tokenizer_path is not None:
            load_options["tokenizer_path"] = config.tokenizer_path
        if config.use_machete:
            load_options["use_machete"] = True
        if getattr(config, "prefill_backend", None) is not None:
            load_options["prefill_backend"] = config.prefill_backend

        active_module = model_module or quic_model
        self.model, self.tokenizer = active_module.load(**load_options)
        self.max_tokens = 256
        self.registry = registry
        self.cache = DynamicCache()
        self.attention_mask: torch.Tensor = torch.Tensor()
        self._request_id_generator = itertools.count(1)
        self.batching_policy = BATCHING_POLICIES[config.batching]()

    def add_request(self, prompt: str):
        encoding = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        assert isinstance(encoding, BatchEncoding)
        input_ids = cast(torch.Tensor, encoding["input_ids"])
        assert isinstance(input_ids, torch.Tensor)
        self.waiting.append(
            SequenceState(
                prompt_tokens=input_ids,
                request_id=next(self._request_id_generator),
                max_tokens=self.max_tokens,
            )
        )

    @torch.inference_mode()
    def step(self):
        with Span("step", self.registry):
            with Span("prepare_decode", self.registry):
                self._prepare_decode_step()
                next_tokens = torch.tensor(
                    [sequence.generated_tokens[-1] for sequence in self.running]
                ).view(-1, 1)
            with Span("forward_pass", self.registry):
                outputs = cast(
                    CausalLMOutputWithPast,
                    self.model(
                        next_tokens,
                        past_key_values=self.cache,
                        use_cache=True,
                        attention_mask=self.attention_mask,
                        position_ids=(
                            self.attention_mask.long().cumsum(dim=-1) - 1
                        ).clamp(min=0)[:, -next_tokens.shape[1] :],
                    ),
                )
                torch.cuda.synchronize()

            with Span("sample_and_update", self.registry):
                assert isinstance(outputs.logits, torch.Tensor)
                token_ids = cast(
                    list[int],
                    torch.argmax(outputs.logits[:, -1, :], dim=-1).tolist(),
                )
                for sequence, token_id in zip(self.running, token_ids):
                    self._update_sequence_with_token(sequence, token_id)

    @torch.inference_mode()
    def run(self):
        outputs: dict[int, str] = {}
        with Span("request run", self.registry):
            while True:
                with Span("scheduling", self.registry):
                    batching_decision = self.batching_policy.decide(
                        self.running, self.waiting, self.max_batch_size
                    )
                    released_ids = {seq.request_id for seq in batching_decision.release}
                    keep = [
                        i
                        for i, seq in enumerate(self.running)
                        if seq.request_id not in released_ids
                    ]

                    for seq in batching_decision.release:
                        output = self.tokenizer.decode(
                            seq.generated_tokens, skip_special_tokens=True
                        )
                        assert isinstance(output, str)
                        outputs[seq.request_id] = output

                    if len(keep) == 0:
                        self.running.clear()
                    elif len(keep) != len(self.running):
                        self.running = [self.running[i] for i in keep]
                        keep_tensor = torch.tensor(keep, dtype=torch.long)
                        self.attention_mask = self.attention_mask[keep_tensor]
                        self.cache.batch_select_indices(keep_tensor)

                    is_batch_empty = len(self.running) == 0
                    torch.cuda.synchronize()

                if batching_decision.admit:
                    with Span("prefill", self.registry):
                        if is_batch_empty:
                            for sequence in batching_decision.admit:
                                self.running.append(sequence)
                                _ = self.waiting.popleft()
                            self.attention_mask, self.cache = self._prefill_sequences(
                                self.running
                            )
                        else:
                            new_sequences: list[SequenceState] = []
                            for _ in batching_decision.admit:
                                sequence = self.waiting.popleft()
                                new_sequences.append(sequence)

                            attention_mask, cache = self._prefill_sequences(new_sequences)
                            self.running.extend(new_sequences)
                            self._splice_sequences(attention_mask, cache)
                        torch.cuda.synchronize()

                if not self.running:
                    break

                self.step()

        return [outputs[request_id] for request_id in sorted(outputs)]

    def _prepare_decode_step(self):
        unfinished_sequences_mask = (
            torch.tensor([not sequence.is_finished for sequence in self.running])
            .to(self.attention_mask.dtype)
            .view(-1, 1)
        )
        self.attention_mask = torch.cat(
            [self.attention_mask, unfinished_sequences_mask], dim=1
        )

    def _update_sequence_with_token(self, sequence: SequenceState, token: int):
        if not sequence.is_finished:
            sequence.generated_tokens.append(token)
        if token == self.tokenizer.eos_token_id and not sequence.is_finished:
            sequence.finished_at = len(sequence.generated_tokens) - 1

    @torch.inference_mode()
    def _prefill_sequences(self, sequences: list[SequenceState]):
        largest_sequence_size = max([seq.prompt_tokens.shape[1] for seq in sequences])
        next_tokens = torch.empty(
            len(sequences),
            largest_sequence_size,
            dtype=sequences[0].prompt_tokens.dtype,
        )
        attention_mask = torch.empty(len(sequences), largest_sequence_size)
        cache = DynamicCache()
        for i, sequence in enumerate(sequences):
            padding_mask = torch.zeros((1, largest_sequence_size))
            padding_mask[:, -sequence.prompt_tokens.shape[1] :] = 1
            attention_mask[i] = padding_mask

            prompt_tokens = torch.zeros((1, largest_sequence_size))
            prompt_tokens[:, -sequence.prompt_tokens.shape[1] :] = sequence.prompt_tokens
            next_tokens[i] = prompt_tokens

        outputs = cast(
            CausalLMOutputWithPast,
            self.model(
                next_tokens,
                past_key_values=cache,
                use_cache=True,
                attention_mask=attention_mask,
                position_ids=(attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0),
            ),
        )
        assert isinstance(outputs.logits, torch.Tensor)
        token_ids = cast(
            list[int],
            torch.argmax(outputs.logits[:, -1, :], dim=-1).tolist(),
        )
        for sequence, token_id in zip(sequences, token_ids):
            self._update_sequence_with_token(sequence, token_id)
        return attention_mask, cache

    @torch.inference_mode()
    def _splice_sequences(
        self,
        new_attention_mask: torch.Tensor,
        new_cache: DynamicCache,
    ):
        cache_len = self.cache.get_seq_length()
        new_cache_len = new_cache.get_seq_length()
        assert self.attention_mask.shape[1] == cache_len
        assert new_attention_mask.shape[1] == new_cache_len

        if hasattr(self.cache, "key_cache") and len(self.cache.key_cache) > 0:
            for i in range(len(self.cache.key_cache)):
                k1 = self.cache.key_cache[i]
                v1 = self.cache.value_cache[i]
                k2 = new_cache.key_cache[i]
                v2 = new_cache.value_cache[i]

                if new_cache_len < cache_len:
                    k2 = F.pad(k2, (0, 0, cache_len - new_cache_len, 0), value=0)
                    v2 = F.pad(v2, (0, 0, cache_len - new_cache_len, 0), value=0)
                elif new_cache_len > cache_len:
                    k1 = F.pad(k1, (0, 0, new_cache_len - cache_len, 0), value=0)
                    v1 = F.pad(v1, (0, 0, new_cache_len - cache_len, 0), value=0)

                self.cache.key_cache[i] = torch.cat([k1, k2], dim=0)
                self.cache.value_cache[i] = torch.cat([v1, v2], dim=0)
        elif hasattr(self.cache, "layers"):
            for cache_layer, new_cache_layer in zip(self.cache.layers, new_cache.layers):
                k1, v1 = cache_layer.keys, cache_layer.values
                k2, v2 = new_cache_layer.keys, new_cache_layer.values

                if new_cache_len < cache_len:
                    k2 = F.pad(k2, (0, 0, cache_len - new_cache_len, 0), value=0)
                    v2 = F.pad(v2, (0, 0, cache_len - new_cache_len, 0), value=0)
                elif new_cache_len > cache_len:
                    k1 = F.pad(k1, (0, 0, new_cache_len - cache_len, 0), value=0)
                    v1 = F.pad(v1, (0, 0, new_cache_len - cache_len, 0), value=0)

                cache_layer.keys = torch.cat([k1, k2], dim=0)
                cache_layer.values = torch.cat([v1, v2], dim=0)

        if new_cache_len > cache_len:
            self.attention_mask = F.pad(
                self.attention_mask, (new_cache_len - cache_len, 0), value=0
            )
        elif new_cache_len < cache_len:
            new_attention_mask = F.pad(
                new_attention_mask, (cache_len - new_cache_len, 0), value=0
            )

        self.attention_mask = torch.cat(
            [self.attention_mask, new_attention_mask], dim=0
        )
