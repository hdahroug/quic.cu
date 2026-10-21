"""JIT compiler and Python loader interface for QUIC CUDA kernels."""

import functools
import os
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

_CUDA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
_MACHETE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "machete")
_BASE_CFLAGS = ["-O3", "--use_fast_math"]
_HOPPER_CFLAGS = [
    "--std=c++20",
    "-DENABLE_HOPPER_TMA_WGMMA=1",
    "-gencode=arch=compute_90a,code=sm_90a",
]


def is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9


def _build(
    name: str,
    source: str,
    cflags: list[str] | None = None,
    ldflags: list[str] | None = None,
    include_paths: list[str] | None = None,
    base_cflags: list[str] = _BASE_CFLAGS,
) -> ModuleType:
    return load(
        name=f"quic_{name}",
        sources=[os.path.join(_CUDA_DIR, source)],
        extra_cuda_cflags=base_cflags + (cflags or []),
        extra_ldflags=ldflags or [],
        extra_include_paths=include_paths or [],
        verbose=False,
    )


@functools.cache
def gemm() -> ModuleType:
    return _build("gemm", "gemm.cu")


@functools.cache
def rmsnorm() -> ModuleType:
    return _build("rmsnorm", "rmsnorm.cu")


@functools.cache
def rope() -> ModuleType:
    return _build("rope", "rope.cu")


@functools.cache
def swiglu() -> ModuleType:
    return _build("swiglu", "swiglu.cu")


@functools.cache
def sampler() -> ModuleType:
    return _build("sampler", "sampler.cu")


@functools.cache
def flash_attn_decode() -> ModuleType:
    if not is_hopper():
        return _build("flash_attn_decode", "flash_attn_decode.cu", cflags=["-std=c++20"])
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    return _build("flash_attn_decode", "flash_attn_decode.cu", cflags=_HOPPER_CFLAGS)


@functools.cache
def flash_attn_prefill() -> ModuleType:
    if not is_hopper():
        return _build("flash_attn_prefill", "flash_attn_prefill.cu")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    return _build(
        "flash_attn_prefill",
        "flash_attn_prefill.cu",
        cflags=_HOPPER_CFLAGS,
        ldflags=["-lcuda"],
    )


@functools.cache
def marlin() -> ModuleType:
    major, minor = torch.cuda.get_device_capability(0)
    return _build(
        "marlin",
        "marlin.cu",
        cflags=["-std=c++20", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
    )


@functools.cache
def machete() -> ModuleType:
    if not is_hopper():
        raise RuntimeError("machete requires a Hopper (sm_90) GPU")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    cutlass = os.environ.get("VLLM_CUTLASS_DIR", "/opt/cutlass")
    return _build(
        "machete",
        "machete.cu",
        cflags=["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
        include_paths=[
            _MACHETE_DIR,
            os.path.join(_MACHETE_DIR, "quantization", "machete"),
            os.path.join(cutlass, "include"),
            os.path.join(cutlass, "tools", "util", "include"),
        ],
        base_cflags=["-O3"],
    )
