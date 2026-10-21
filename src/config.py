"""Engine configuration parameters and default model paths."""
import os
from dataclasses import dataclass
from typing import Literal

MODEL_ROOT = os.environ.get(
    "QWEN25_FP16_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Qwen2.5-0.5B")),
)


@dataclass(frozen=True)
class EngineConfig:
    batching: Literal["static", "continuous"] = "continuous"
    model_path: str | None = None
    tokenizer_path: str | None = None
    use_machete: bool = False
    prefill_backend: Literal["pytorch", "quic"] = "pytorch"
