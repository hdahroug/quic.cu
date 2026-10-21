"""Unit tests verifying engine execution across static and continuous batching policies."""

import pytest
import torch

from config import EngineConfig
from engine import Engine
from sequence import SequenceState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_engine_empty_run():
    engine = Engine()
    outputs = engine.run()
    assert outputs == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
@pytest.mark.parametrize("batching", ["static", "continuous"])
def test_engine_multirequest_generation(batching: str):
    config = EngineConfig(batching=batching)
    engine = Engine(config=config, max_batch_size=4)
    engine.max_tokens = 16

    prompts = [
        "What is 2 + 2?",
        "Name two planets in the solar system.",
        "What is the capital of France?",
    ]

    for p in prompts:
        engine.add_request(p)

    outputs = engine.run()

    assert len(outputs) == len(prompts)
    for out in outputs:
        assert isinstance(out, str)
        assert len(out.strip()) > 0
