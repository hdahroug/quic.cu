"""Unified test suite execution orchestration for local CUDA and Modal cloud environments."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_local(pytest_args: list[str], model_path: str | None = None, awq_path: str | None = None) -> int:
    env = dict(
        os.environ,
        PYTHONPATH=f"{ROOT_DIR}/src:{ROOT_DIR}",
    )
    if model_path:
        env["QWEN25_FP16_PATH"] = str(Path(model_path).resolve())
    if awq_path:
        env["QWEN25_AWQ_PATH"] = str(Path(awq_path).resolve())
    cmd = [sys.executable, "-m", "pytest", "tests"] + pytest_args
    print(f"[QUIC Test Runner] Executing locally: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT_DIR), env=env)


import modal

volume = modal.Volume.from_name("qwen2-weights-volume")
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers<4.45.0",
        "accelerate",
        "safetensors",
        "tokenizers",
        "numpy>=1.26.0",
        "ninja",
        "pytest",
    )
    .add_local_dir(ROOT_DIR, remote_path="/root/quic", copy=False)
)

app = modal.App("quic_test_runner")


@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    volumes={"/weights": volume},
)
def remote_test(args):
    import subprocess
    import sys

    env = dict(
        os.environ,
        PYTHONPATH="/root/quic/src:/root/quic",
        QWEN25_FP16_PATH="/weights/Qwen2.5-0.5B",
        QWEN25_AWQ_PATH="/weights/Qwen2.5-0.5B-AWQ",
        TORCH_EXTENSIONS_DIR="/weights/_torchcache_serving",
    )
    cmd = [sys.executable, "-m", "pytest", "tests"] + args
    res = subprocess.run(cmd, cwd="/root/quic", env=env, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def run_modal(gpu: str, pytest_args: list[str]) -> int:
    print(f"[QUIC Test Runner] Launching tests on Modal ({gpu})...")
    with app.run():
        retcode, stdout, stderr = remote_test.with_options(gpu=gpu).remote(pytest_args)
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    return retcode


def main():
    parser = argparse.ArgumentParser(description="QUIC Test Suite Runner")
    parser.add_argument("--local", action="store_true", help="Run tests on local CUDA GPU")
    parser.add_argument("--modal", action="store_true", help="Run tests on Modal cloud GPU")
    parser.add_argument("--gpu", type=str, default="L4", help="Modal GPU type (L4, H100, etc.)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FP16 model weights directory")
    parser.add_argument("--awq-path", type=str, default=None, help="Path to AWQ 4-bit model weights directory")
    args, extra_pytest_args = parser.parse_known_args()

    if not args.local and not args.modal:
        import torch

        if torch.cuda.is_available():
            args.local = True
        else:
            args.modal = True

    if args.modal:
        ret = run_modal(args.gpu, extra_pytest_args)
    else:
        ret = run_local(extra_pytest_args, model_path=args.model_path, awq_path=args.awq_path)

    sys.exit(ret)


if __name__ == "__main__":
    main()
