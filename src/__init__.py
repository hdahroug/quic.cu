"""QUIC: Qwen Unified Inference on CUDA."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import EngineConfig
from engine import Engine
from metrics import MetricsRegistry, Span, Timer
from model import QUICForCausalLM, load as load_quic_model
from model_hf import load as load_hf_model
from sequence import SequenceState
from tokenizer import FastTokenizer

__all__ = [
    "Engine",
    "EngineConfig",
    "QUICForCausalLM",
    "SequenceState",
    "MetricsRegistry",
    "Span",
    "Timer",
    "FastTokenizer",
    "load_quic_model",
    "load_hf_model",
]
