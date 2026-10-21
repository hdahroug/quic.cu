"""Benchmark collector measuring TTFT, decode throughput, and VRAM across model execution paths."""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, DynamicCache

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import model as quic_model
import model_hf as hf_model
from config import EngineConfig
from engine import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_prompt(tokenizer, target_len: int) -> torch.Tensor:
    sample = (
        "In artificial intelligence and deep learning, large language models are trained on diverse datasets "
        "to perform natural language processing tasks. The transformer architecture relies on multi-head attention "
        "mechanisms to capture long-range contextual dependencies across sequences. Efficient inference optimization "
        "requires careful memory hierarchy management, kernel fusion, and hardware-specific tensor operations. "
    )
    tokens = tokenizer.encode(sample)
    repeat_count = (target_len // len(tokens)) + 2
    long_tokens = (tokens * repeat_count)[:target_len]
    return torch.tensor([long_tokens], dtype=torch.long, device="cuda")


def benchmark_model_sweep(
    model: torch.nn.Module,
    tokenizer,
    batch_size: int,
    ctx_len: int,
    gen_tokens: int = 32,
    trials: int = 3,
) -> Dict[str, float]:
    try:
        base_prompt = generate_prompt(tokenizer, ctx_len)
        input_ids = base_prompt.repeat(batch_size, 1)

        cache = DynamicCache()
        with torch.no_grad():
            out = model(input_ids, past_key_values=cache)
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            for _ in range(2):
                out = model(next_tok, past_key_values=cache)
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()

        ttft_list, decode_lat_list, vram_list = [], [], []

        for _ in range(trials):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cache = DynamicCache()

            start_prefill = torch.cuda.Event(enable_timing=True)
            end_prefill = torch.cuda.Event(enable_timing=True)
            start_decode = torch.cuda.Event(enable_timing=True)
            end_decode = torch.cuda.Event(enable_timing=True)

            with torch.no_grad():
                start_prefill.record()
                out = model(input_ids, past_key_values=cache)
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                end_prefill.record()

                start_decode.record()
                curr = next_tok
                for _ in range(gen_tokens - 1):
                    out = model(curr, past_key_values=cache)
                    curr = out.logits[:, -1:, :].argmax(dim=-1)
                end_decode.record()

            torch.cuda.synchronize()

            ttft_list.append(start_prefill.elapsed_time(end_prefill))
            decode_lat_list.append(start_decode.elapsed_time(end_decode))
            vram_list.append(torch.cuda.max_memory_allocated() / (1024 * 1024))

        mean_ttft = float(np.mean(ttft_list))
        mean_decode = float(np.mean(decode_lat_list))
        mean_vram = float(np.mean(vram_list))

        total_decode_tokens = batch_size * gen_tokens
        decode_tps = total_decode_tokens / (mean_decode / 1000.0) if mean_decode > 0 else 0.0

        return {
            "ttft_ms": mean_ttft,
            "decode_tps": decode_tps,
            "peak_vram_mb": mean_vram,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"OOM on B={batch_size}, Ctx={ctx_len}")
            gc.collect()
            torch.cuda.empty_cache()
            return {"ttft_ms": float("nan"), "decode_tps": 0.0, "peak_vram_mb": float("nan")}
        raise


def benchmark_engine_batching(
    policy: str,
    max_batch_size: int = 4,
    num_requests: int = 8,
    max_tokens: int = 32,
    model_path: str | None = None,
) -> Dict[str, float]:
    logger.info(f"[Engine Benchmark] Evaluating policy='{policy}', max_batch_size={max_batch_size}, requests={num_requests}...")
    config = EngineConfig(batching=policy, model_path=model_path)
    engine = Engine(config=config, max_batch_size=max_batch_size)
    engine.max_tokens = max_tokens

    prompts = [
        "In artificial intelligence and deep learning, large language models are trained on diverse datasets.",
        "The transformer architecture relies on multi-head attention mechanisms for sequence modeling.",
        "Efficient inference optimization requires memory hierarchy management and kernel fusion.",
        "GPU computing accelerates matrix multiplication through specialized tensor core hardware.",
        "Continuous batching improves serving throughput by dynamically scheduling sequence slots.",
        "Quantization reduces memory footprint while preserving task accuracy across layers.",
        "Split-K decode attention parallelizes across sequence length for low latency generation.",
        "RMSNorm normalizes activation variance without computing mean offsets for high bandwidth.",
    ][:num_requests]

    for p in prompts:
        engine.add_request(p)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    outputs = engine.run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    total_tokens = sum(len(engine.tokenizer.encode(o)) for o in outputs)
    tps = total_tokens / elapsed if elapsed > 0 else 0.0

    logger.info(
        f"  [{policy}] Elapsed: {elapsed:.2f}s | Throughput: {tps:.1f} tok/s | Peak VRAM: {peak_vram:.1f} MB | Tokens: {total_tokens}"
    )

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "policy": policy,
        "max_batch_size": max_batch_size,
        "num_requests": len(outputs),
        "total_tokens": total_tokens,
        "elapsed_s": float(elapsed),
        "throughput_tps": float(tps),
        "peak_vram_mb": float(peak_vram),
    }


def run_benchmarks(
    paths: List[str],
    batch_sizes: List[int],
    ctx_lens: List[int],
    gen_tokens: int = 32,
    batching: str = "none",
    model_path: str | None = None,
    awq_path: str | None = None,
    output_dir: str = "results",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    fp16_path = model_path or os.environ.get("QWEN25_FP16_PATH", "/weights/Qwen2.5-0.5B")
    awq_path = awq_path or os.environ.get("QWEN25_AWQ_PATH", "/weights/Qwen2.5-0.5B-AWQ")
    fp16_path = os.path.expanduser(fp16_path)
    awq_path = os.path.expanduser(awq_path)

    all_data = {}

    if paths:
        tokenizer = AutoTokenizer.from_pretrained(fp16_path)

        for path_name in paths:
            logger.info(f"\n=======================================================")
            logger.info(f"Loading Model for Path: {path_name}")
            logger.info(f"=======================================================")

            if path_name == "quic":
                model, _ = quic_model.load(fp16_path, prefill_backend="pytorch")
            elif path_name == "quic_quant":
                model, _ = quic_model.load(awq_path, use_marlin=True, tokenizer_path=fp16_path)
            elif path_name == "hf":
                model, _ = hf_model.load(fp16_path)
                model = model.to("cuda").eval()
            elif path_name == "hf_quant":
                model, _ = hf_model.load(awq_path, tokenizer_path=fp16_path)
                model = model.to("cuda").eval()
            else:
                raise ValueError(f"Unknown path: {path_name}")

            path_results = {}
            for b in batch_sizes:
                for ctx in ctx_lens:
                    logger.info(f"[{path_name}] Benchmarking B={b}, Ctx={ctx}...")
                    res = benchmark_model_sweep(model, tokenizer, b, ctx, gen_tokens)
                    path_results[f"B{b}_C{ctx}"] = res
                    logger.info(
                        f"  -> TTFT: {res['ttft_ms']:.2f} ms | Dec TPS: {res['decode_tps']:.1f} | VRAM: {res['peak_vram_mb']:.1f} MB"
                    )

            all_data[path_name] = path_results
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if batching in ("static", "continuous", "both"):
        policies = ["static", "continuous"] if batching == "both" else [batching]
        batching_results = {}
        for pol in policies:
            res = benchmark_engine_batching(pol, max_batch_size=4, num_requests=8, model_path=fp16_path)
            batching_results[pol] = res
        all_data["engine_batching"] = batching_results

    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    logger.info(f"Saved results to {json_path}")
    return all_data


def main():
    parser = argparse.ArgumentParser(description="QUIC Benchmark Collector")
    parser.add_argument("--paths", type=str, default="all", help="Comma-separated: quic,quic_quant,hf,hf_quant or all or none")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16", help="Batch sizes list")
    parser.add_argument("--context-lengths", type=str, default="128,512,1024,2048", help="Context lengths list")
    parser.add_argument("--batching", type=str, choices=["none", "static", "continuous", "both"], default="none", help="Engine batching mode")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FP16 model weights directory")
    parser.add_argument("--awq-path", type=str, default=None, help="Path to AWQ 4-bit model weights directory")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    paths = []
    if args.paths != "none":
        paths = ["quic", "quic_quant", "hf", "hf_quant"] if args.paths == "all" else args.paths.split(",")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    ctx_lens = [int(x) for x in args.context_lengths.split(",")]

    run_benchmarks(
        paths,
        batch_sizes,
        ctx_lens,
        batching=args.batching,
        model_path=args.model_path,
        awq_path=args.awq_path,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
