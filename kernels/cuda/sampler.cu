// Greedy argmax sampling CUDA kernel for vocabulary token selection.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace sampler_d {

constexpr int kThreads = 256;

__global__ void argmax_kernel(const __half* __restrict__ logits,
                              int64_t V,
                              int64_t* __restrict__ out) {
  const int64_t row = blockIdx.x;
  const __half* base = logits + row * V;

  float best_val = -FLT_MAX;
  int64_t best_idx = 0;

  for (int64_t i = threadIdx.x; i < V; i += kThreads) {
    const float v = __half2float(base[i]);
    if (v > best_val) {
      best_val = v;
      best_idx = i;
    }
  }

  __shared__ float s_val[kThreads];
  __shared__ int64_t s_idx[kThreads];
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();

  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float v = s_val[threadIdx.x + s];
      const int64_t i = s_idx[threadIdx.x + s];
      if (v > s_val[threadIdx.x] ||
          (v == s_val[threadIdx.x] && i < s_idx[threadIdx.x])) {
        s_val[threadIdx.x] = v;
        s_idx[threadIdx.x] = i;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[row] = s_idx[0];
  }
}

torch::Tensor forward(torch::Tensor logits) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kHalf,
              "sampler_d: fp16 CUDA tensor required");
  TORCH_CHECK(logits.dim() == 3 && logits.size(1) == 1,
              "sampler_d: expected [B,1,V]");

  const int64_t B = logits.size(0);
  const int64_t V = logits.size(2);
  auto logits_c = logits.contiguous();
  auto idx = torch::empty({B, 1}, logits.options().dtype(at::kLong));

  const auto stream = at::cuda::getCurrentCUDAStream();
  argmax_kernel<<<B, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(logits_c.data_ptr<at::Half>()),
      V,
      idx.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return idx;
}

}  

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sampler_d::forward, "greedy argmax fp16 [B,1,V] -> int64 [B,1]");
}
