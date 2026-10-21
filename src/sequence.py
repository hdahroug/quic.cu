"""Sequence state representation for tracking generation progress and tokens."""
from dataclasses import dataclass, field
import torch


@dataclass
class SequenceState:
    prompt_tokens: torch.Tensor
    request_id: int
    max_tokens: int
    generated_tokens: list[int] = field(default_factory=list)
    finished_at: int | None = None

    @property
    def is_finished(self) -> bool:
        return (
            self.finished_at is not None
            or len(self.generated_tokens) >= self.max_tokens
        )

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, SequenceState):
            return False
        return (
            torch.equal(self.prompt_tokens, value.prompt_tokens)
            and self.request_id == value.request_id
            and self.max_tokens == value.max_tokens
            and self.generated_tokens == value.generated_tokens
            and self.finished_at == value.finished_at
        )
