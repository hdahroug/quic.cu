"""Markdown benchmark report generator for model paths and serving policies."""

import json
import os
import sys
from pathlib import Path


def generate_markdown_report(json_path: str, out_md_path: str):
    with open(json_path) as f:
        data = json.load(f)

    sections = []

    paths = [p for p in data.keys() if p != "engine_batching"]
    if paths:
        first_path = paths[0]
        keys = list(data[first_path].keys())

        lines = [
            "# QUIC 4-Path Benchmark Summary Report\n",
            "| Batch | Context | "
            + " | ".join(f"{p} TTFT (ms)" for p in paths)
            + " | "
            + " | ".join(f"{p} Dec TPS" for p in paths)
            + " | "
            + " | ".join(f"{p} VRAM (MB)" for p in paths)
            + " |",
            "| :---: | :---: | "
            + " | ".join(":---:" for _ in paths)
            + " | "
            + " | ".join(":---:" for _ in paths)
            + " | "
            + " | ".join(":---:" for _ in paths)
            + " |",
        ]

        for k in keys:
            b = k.split("_")[0][1:]
            c = k.split("_")[1][1:]

            ttft_vals = [
                f"{data[p][k]['ttft_ms']:.1f}"
                if data[p][k]["ttft_ms"] == data[p][k]["ttft_ms"]
                else "OOM"
                for p in paths
            ]
            tps_vals = [
                f"{data[p][k]['decode_tps']:.1f}"
                if data[p][k]["decode_tps"] > 0
                else "OOM"
                for p in paths
            ]
            vram_vals = [
                f"{data[p][k]['peak_vram_mb']:.1f}"
                if data[p][k].get("peak_vram_mb") and data[p][k]["peak_vram_mb"] == data[p][k]["peak_vram_mb"]
                else "OOM"
                for p in paths
            ]

            row = (
                f"| {b} | {c} | "
                + " | ".join(ttft_vals)
                + " | "
                + " | ".join(tps_vals)
                + " | "
                + " | ".join(vram_vals)
                + " |"
            )
            lines.append(row)
        sections.append("\n".join(lines))

    if "engine_batching" in data:
        b_data = data["engine_batching"]
        b_lines = [
            "## Serving Engine Batching Policies (Static vs Continuous)\n",
            "| Policy | Max Batch Size | Requests | Total Tokens | Latency (s) | Throughput (tok/s) | Peak VRAM (MB) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for pol, info in b_data.items():
            b_lines.append(
                f"| {info['policy']} | {info['max_batch_size']} | {info['num_requests']} | {info['total_tokens']} | {info['elapsed_s']:.2f} | {info['throughput_tps']:.1f} | {info['peak_vram_mb']:.1f} |"
            )
        sections.append("\n".join(b_lines))

    md_content = "\n\n".join(sections) + "\n"
    with open(out_md_path, "w") as f:
        f.write(md_content)
    print(f"Report written to {out_md_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    json_file = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.json"
    md_file = sys.argv[2] if len(sys.argv) > 2 else "results/REPORT.md"
    if os.path.exists(json_file):
        generate_markdown_report(json_file, md_file)
    else:
        print(f"No results found at {json_file}")
