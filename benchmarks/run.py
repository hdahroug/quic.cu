"""Benchmark execution orchestration across local CUDA and Modal cloud environments."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_local(args) -> int:
    env = dict(
        os.environ,
        PYTHONPATH=f"{ROOT_DIR}/src:{ROOT_DIR}",
    )
    if args.model_path:
        env["QWEN25_FP16_PATH"] = str(Path(args.model_path).resolve())
    if args.awq_path:
        env["QWEN25_AWQ_PATH"] = str(Path(args.awq_path).resolve())
    cmd = [
        sys.executable,
        "benchmarks/collect.py",
        "--paths", args.paths,
        "--batch-sizes", args.batch_sizes,
        "--context-lengths", args.context_lengths,
        "--batching", args.batching,
        "--out", args.out,
    ]
    print(f"[QUIC Benchmark Runner] Executing locally: {' '.join(cmd)}")
    ret = subprocess.call(cmd, cwd=str(ROOT_DIR), env=env)
    if ret == 0:
        subprocess.call([sys.executable, "benchmarks/report.py", f"{args.out}/benchmark_results.json", f"{args.out}/REPORT.md"], cwd=str(ROOT_DIR), env=env)
    return ret


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
    )
    .add_local_dir(ROOT_DIR, remote_path="/root/quic", copy=False)
)

app = modal.App("quic_benchmark_runner")


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    volumes={"/weights": volume},
)
def remote_benchmark(paths, batch_sizes, context_lengths, batching: str = "none"):
    import subprocess

    env = dict(
        os.environ,
        PYTHONPATH="/root/quic/src:/root/quic",
        QWEN25_FP16_PATH="/weights/Qwen2.5-0.5B",
        QWEN25_AWQ_PATH="/weights/Qwen2.5-0.5B-AWQ",
        TORCH_EXTENSIONS_DIR="/weights/_torchcache_serving",
    )
    cmd = [
        "python3",
        "benchmarks/collect.py",
        "--paths", paths,
        "--batch-sizes", batch_sizes,
        "--context-lengths", context_lengths,
        "--batching", batching,
        "--out", "/tmp/results",
    ]
    ret = subprocess.call(cmd, cwd="/root/quic", env=env)
    if ret == 0:
        subprocess.call(["python3", "benchmarks/report.py", "/tmp/results/benchmark_results.json", "/tmp/results/REPORT.md"], cwd="/root/quic", env=env)
    
    files = {}
    for p in Path("/tmp/results").glob("*"):
        if p.is_file():
            files[p.name] = p.read_text()
    return files


def run_modal(args) -> int:
    print(f"[QUIC Benchmark Runner] Launching on Modal ({args.gpu})...")
    with app.run():
        results = remote_benchmark.with_options(gpu=args.gpu).remote(
            args.paths, args.batch_sizes, args.context_lengths, args.batching
        )

    out_dir = ROOT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in results.items():
        (out_dir / name).write_text(content)
        print(f"Downloaded: {out_dir / name}")

    report_file = out_dir / "REPORT.md"
    if report_file.exists():
        print("\n" + report_file.read_text())
    return 0


def main():
    parser = argparse.ArgumentParser(description="QUIC Benchmark Launcher")
    parser.add_argument("--local", action="store_true", help="Run benchmark locally")
    parser.add_argument("--modal", action="store_true", help="Run benchmark on Modal GPU")
    parser.add_argument("--gpu", type=str, default="L4", help="Modal GPU type (L4, H100, etc.)")
    parser.add_argument("--paths", type=str, default="all", help="quic,quic_quant,hf,hf_quant or all or none")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16", help="Batch sizes")
    parser.add_argument("--context-lengths", type=str, default="128,512,1024,2048", help="Context lengths")
    parser.add_argument("--batching", type=str, choices=["none", "static", "continuous", "both"], default="none", help="Benchmark serving engine batching")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FP16 model weights directory")
    parser.add_argument("--awq-path", type=str, default=None, help="Path to AWQ 4-bit model weights directory")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    if not args.local and not args.modal:
        import torch
        if torch.cuda.is_available():
            args.local = True
        else:
            args.modal = True

    if args.modal:
        ret = run_modal(args)
    else:
        ret = run_local(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
