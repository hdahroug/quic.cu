"""PyTest fixtures and environment configuration for QUIC test suite."""

import os
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import pytest
import torch

FP16_PATH = os.environ.get("QWEN25_FP16_PATH", "/weights/Qwen2.5-0.5B")
AWQ_PATH = os.environ.get("QWEN25_AWQ_PATH", "/weights/Qwen2.5-0.5B-AWQ")


@pytest.fixture(scope="session")
def model_paths():
    return {
        "fp16": FP16_PATH,
        "awq": AWQ_PATH,
    }


@pytest.fixture(autouse=True)
def cuda_device():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
