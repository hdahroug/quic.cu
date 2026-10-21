// Fused SwiGLU activation CUDA kernel computing SiLU(gate) * up.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#define QWEN_INTERMEDIATE 4864   
#define THREADS 256

__global__ void swiglu_naive_half_kernel(const __half* __restrict__ gate,
                                         const __half* __restrict__ up,
                                         __half* __restrict__ y, int numRows) {
  const int row = blockIdx.y;

#ifdef _WIN32
  const long long base = (long long)row * QWEN_INTERMEDIATE;
#else
  const long base = (long)row * QWEN_INTERMEDIATE;
#endif

  for (int col = blockIdx.x * blockDim.x + threadIdx.x;
       col < QWEN_INTERMEDIATE; col += gridDim.x * blockDim.x) {
#ifdef _WIN32
    const long long idx = base + col;
#else
    const long idx = base + col;
#endif
    const float g  = __half2float(gate[idx]);
    const float u  = __half2float(up[idx]);
    y[idx] = __float2half((g / (1.0f + expf(-g))) * u);   
  }
}

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "gate/up must be CUDA tensors");
  TORCH_CHECK(gate.scalar_type() == at::kHalf, "gate must be fp16");
  TORCH_CHECK(gate.dim() == 2 && gate.size(1) == QWEN_INTERMEDIATE,
              "gate must be [B, 4864]");
  TORCH_CHECK(up.sizes() == gate.sizes(), "gate/up shape mismatch");

  auto y = torch::empty_like(gate);

  const int numRows = (int)gate.size(0) * (int)gate.size(1) / QWEN_INTERMEDIATE;
  dim3 block(THREADS);
  dim3 grid((QWEN_INTERMEDIATE + THREADS - 1) / THREADS, numRows);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  swiglu_naive_half_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(up.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()), numRows);

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &swiglu_forward, "Qwen2 SwiGLU Naive CUDA");
}
