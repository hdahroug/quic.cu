// cuBLASLt high-performance FP16 GEMM with optional bias addition.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION < 11000
#error "gemm_d requires cuBLAS 11+ for cublasGemmEx"
#endif

#define CUBLAS_CHECK(x)                                                            \
  do {                                                                             \
    cublasStatus_t __s = (x);                                                      \
    TORCH_CHECK(__s == CUBLAS_STATUS_SUCCESS, "cuBLAS error ", (int)__s, " at ",   \
                __FILE__, ":", __LINE__);                                          \
  } while (0)

namespace gemm_d {

cublasHandle_t handle() {
  static cublasHandle_t h = [] {
    cublasHandle_t x;
    CUBLAS_CHECK(cublasCreate(&x));
    return x;
  }();
  return h;
}

__global__ void add_bias_kernel(__half* __restrict__ y, const __half* __restrict__ bias,
                                int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int c = idx % N;
  y[idx] = __float2half(__half2float(y[idx]) + __half2float(bias[c]));
}

torch::Tensor forward(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "gemm_d: CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf,
              "gemm_d: fp16 tensors required");
  TORCH_CHECK(w.dim() == 2, "gemm_d: weight must be [N,K]");
  TORCH_CHECK(x.dim() >= 2 && x.size(-1) == w.size(1), "gemm_d: inner dim mismatch");

  const int N = w.size(0);
  const int K = w.size(1);

  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto x2 = xc.reshape({-1, K});
  const int M = x2.size(0);

  auto y2 = torch::empty({M, N}, x.options());

  const float fal = 1.0f;
  const float fbe = 0.0f;

  cublasGemmEx(
      handle(), CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K,
      &fal,
      wc.data_ptr<at::Half>(), CUDA_R_16F, K,
      x2.data_ptr<at::Half>(), CUDA_R_16F, K,
      &fbe,
      y2.data_ptr<at::Half>(), CUDA_R_16F, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  const bool has_bias = bias.has_value() && bias->defined() && bias->numel() > 0;
  if (has_bias) {
    auto bc = bias->contiguous();
    const __half* bias_ptr = reinterpret_cast<const __half*>(bc.const_data_ptr<at::Half>());
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    const auto stream = at::cuda::getCurrentCUDAStream();
    add_bias_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(y2.data_ptr<at::Half>()), bias_ptr, M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end() - 1);
  out_shape.push_back(N);
  return y2.reshape(out_shape);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gemm_d::forward, "cuBLAS fp16 linear y[...,N] = x[...,K] @ W[N,K]^T (+bias)",
        py::arg("x"), py::arg("w"), py::arg("bias") = py::none());
}
