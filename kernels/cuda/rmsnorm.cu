// Vectorized 16-byte load RMSNorm CUDA kernel for hidden dimension 896.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define WARP_SIZE 32
#define QWEN_HIDDEN_DIM 896
#define QWEN_THREADS 128

  struct alignas(16) Half8 {
        half2 h0, h1, h2, h3;
  };

__device__ float warpReduce(float val) {
  #pragma unroll
  for (int activeThreads = WARP_SIZE >> 1; activeThreads > 0;
       activeThreads >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, activeThreads);
  }
  return val;
}

template <int hiddenDim, int threadsPerBlock>
__global__ void qwen2_rmsnorm_kernel(const Half8 *x, const Half8 *w, float eps,
                                     Half8 *y) {

  __shared__ Half8 xShared[112];
  __shared__ float sumPerWarp[WARP_SIZE];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int laneId = tid & 31;
  const int warpId = tid >> 5;
  const int warpsPerBlock = threadsPerBlock >> 5;

  const int bid = blockIdx.x;
  float sum = 0.0f;

  if (tid < 112) {

    Half8 x_val= x[bid*112+tid];
    xShared [tid] = x_val;

    float2 f0 = __half22float2(x_val.h0);
    float2 f1 = __half22float2(x_val.h1);
    float2 f2 = __half22float2(x_val.h2);
    float2 f3 = __half22float2(x_val.h3);

    sum += (f0.x*f0.x)+(f0.y*f0.y)+(f1.x*f1.x)+(f1.y*f1.y)
          +(f2.x*f2.x)+(f2.y*f2.y)+(f3.x*f3.x)+(f3.y*f3.y);

  }


  float warpSum = warpReduce(sum);
  if (laneId == 0) {
    sumPerWarp[warpId] = warpSum;
  }
  __syncthreads();

  if (tid < WARP_SIZE) {
    sumPerWarp[tid] = warpReduce(tid < warpsPerBlock ? sumPerWarp[tid] : 0);
    if (tid == 0) {
      rms_ = rsqrtf(sumPerWarp[tid] / hiddenDim + eps);
    }
  }
  __syncthreads();


  if(tid < 112) {
    Half8 w_val = w[tid];
    Half8 x_cached = xShared[tid];
    Half8 out_val ;

    float2 x0 = __half22float2(x_cached.h0); float2 w0 = __half22float2(w_val.h0);
    float2 x1 = __half22float2(x_cached.h1); float2 w1 = __half22float2(w_val.h1);
    float2 x2 = __half22float2(x_cached.h2); float2 w2 = __half22float2(w_val.h2);
    float2 x3 = __half22float2(x_cached.h3); float2 w3 = __half22float2(w_val.h3);

    out_val.h0 = __floats2half2_rn(x0.x * rms_ * w0.x, x0.y * rms_ * w0.y);
    out_val.h1 = __floats2half2_rn(x1.x * rms_ * w1.x, x1.y * rms_ * w1.y);
    out_val.h2 = __floats2half2_rn(x2.x * rms_ * w2.x, x2.y * rms_ * w2.y);
    out_val.h3 = __floats2half2_rn(x3.x * rms_ * w3.x, x3.y * rms_ * w3.y);
    
    
    y[bid * 112 + tid] = out_val;
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormWarpFloat4(float *x, float *w, float eps, float *y) {
  const Half8 *x_ = reinterpret_cast<const Half8 *>(x);
  const Half8 *w_ = reinterpret_cast<const Half8 *>(w);
  Half8 *y_ = reinterpret_cast<Half8*>(y);

  qwen2_rmsnorm_kernel<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x_, w_, eps, y_);
}

torch::Tensor qwen2_rmsnorm_forward(torch::Tensor x, torch::Tensor weight, float eps = 1e-6f) {
    auto y = torch::empty_like(x);
    int numTokens = x.numel() / QWEN_HIDDEN_DIM;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    qwen2_rmsnorm_kernel<QWEN_HIDDEN_DIM, QWEN_THREADS><<<numTokens, QWEN_THREADS, 0, stream>>>(
        reinterpret_cast<const Half8*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const Half8*>(weight.data_ptr<at::Half>()),
        eps,
        reinterpret_cast<Half8*>(y.data_ptr<at::Half>())
    );
    return y;
}
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("forward", &qwen2_rmsnorm_forward, "Qwen2 RMSNorm CUDA Forward");
    }