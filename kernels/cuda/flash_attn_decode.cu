// Split-K GQA decode attention CUDA kernel with left-padding clamping.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FAD_HEAD_DIM 64
#define FAD_TILE 64
#define FAD_MAX_SPLITS 2048

__global__ void fad_decode_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ Opart,
    float* __restrict__ Mpart,
    float* __restrict__ Lpart,
    int S, int n_kv, int nh, int groups, int num_splits, int chunk,
    float softmax_scale,
    const int* __restrict__ pad_lens) {
  const int D = FAD_HEAD_DIM;
  const int TILE = FAD_TILE;

  __shared__ __align__(16) half Ksh[TILE * FAD_HEAD_DIM];
  __shared__ __align__(16) half Vsh[TILE * FAD_HEAD_DIM];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  const int split = blockIdx.x;
  const int kv = blockIdx.y;
  const int b = blockIdx.z;

  const int pad = pad_lens ? pad_lens[b] : 0;
  const int token_start = max(split * chunk, pad);
  const int token_end = min(split * chunk + chunk, S);
  if (token_start >= token_end) return;

  const int qh = kv * groups + warp;

  float2 qf = make_float2(0.0f, 0.0f);
  if (warp < groups) {
    const half2* qsrc = reinterpret_cast<const half2*>(Q + ((size_t)(b * nh + qh)) * D);
    qf = __half22float2(qsrc[lane]);
  }

  const size_t kv_base = ((size_t)(b * n_kv + kv)) * S * D;

  float m = -INFINITY;
  float l = 0.0f;
  float o0 = 0.0f;
  float o1 = 0.0f;

  for (int tile_start = token_start; tile_start < token_end; tile_start += TILE) {
    const int len = min(TILE, token_end - tile_start);

    const uint4* ksrc = reinterpret_cast<const uint4*>(K + kv_base + (size_t)tile_start * D);
    const uint4* vsrc = reinterpret_cast<const uint4*>(V + kv_base + (size_t)tile_start * D);
    uint4* kdst = reinterpret_cast<uint4*>(Ksh);
    uint4* vdst = reinterpret_cast<uint4*>(Vsh);
    const int nvec = len * (D * sizeof(half) / sizeof(uint4));
    for (int i = tid; i < nvec; i += blockDim.x) {
      kdst[i] = ksrc[i];
      vdst[i] = vsrc[i];
    }
    __syncthreads();

    if (warp < groups) {
      const half2* ksh = reinterpret_cast<const half2*>(Ksh);
      const half2* vsh = reinterpret_cast<const half2*>(Vsh);
      for (int t = 0; t < len; ++t) {
        const float2 kf = __half22float2(ksh[t * 32 + lane]);
        float score = qf.x * kf.x + qf.y * kf.y;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          score += __shfl_xor_sync(0xffffffffu, score, off);
        }
        score *= softmax_scale;

        const float m_new = fmaxf(m, score);
        const float alpha = __expf(m - m_new);
        const float p = __expf(score - m_new);
        l = l * alpha + p;

        const float2 vf = __half22float2(vsh[t * 32 + lane]);
        o0 = o0 * alpha + p * vf.x;
        o1 = o1 * alpha + p * vf.y;
        m = m_new;
      }
    }
    __syncthreads();
  }

  if (warp < groups) {
    const size_t row = (size_t)(b * nh + qh) * num_splits + split;
    reinterpret_cast<float2*>(Opart + row * D)[lane] = make_float2(o0, o1);
    Mpart[row] = m;
    Lpart[row] = l;
  }
}

__global__ void fad_combine_kernel(
    const float* __restrict__ Opart,
    const float* __restrict__ Mpart,
    const float* __restrict__ Lpart,
    half* __restrict__ out,
    int num_splits, int nh, int D) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int d = threadIdx.x;
  const size_t base = ((size_t)(b * nh + h)) * num_splits;

  float m = -INFINITY;
  for (int s = 0; s < num_splits; ++s) m = fmaxf(m, Mpart[base + s]);

  if (m == -INFINITY) {
    out[((size_t)(b * nh + h)) * D + d] = __float2half(0.0f);
    return;
  }

  float num = 0.0f;
  float den = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float w = __expf(Mpart[base + s] - m);
    den += w * Lpart[base + s];
    num += w * Opart[(base + s) * D + d];
  }
  out[((size_t)(b * nh + h)) * D + d] = __float2half(num / den);
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      float softmax_scale = -1.0f,
                      c10::optional<torch::Tensor> pad_lens = c10::nullopt) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "FAD_d: CUDA tensors required");
  TORCH_CHECK(Q.scalar_type() == at::kHalf && K.scalar_type() == at::kHalf &&
                  V.scalar_type() == at::kHalf,
              "FAD_d: fp16 tensors required");
  TORCH_CHECK(Q.dim() == 4 && Q.size(2) == 1, "FAD_d: Q must be [B, nh, 1, D]");
  TORCH_CHECK(Q.size(3) == FAD_HEAD_DIM, "FAD_d: head_dim must be 64");

  const int B = Q.size(0);
  const int nh = Q.size(1);
  const int D = Q.size(3);
  const int n_kv = K.size(1);
  const int S = K.size(2);
  TORCH_CHECK(K.size(0) == B && V.size(0) == B, "FAD_d: batch mismatch");
  TORCH_CHECK(V.size(1) == n_kv && V.size(2) == S, "FAD_d: K/V length mismatch");
  TORCH_CHECK(K.size(3) == D && V.size(3) == D, "FAD_d: K/V head_dim mismatch");
  TORCH_CHECK(nh % n_kv == 0, "FAD_d: nh must be divisible by n_kv");
  const int groups = nh / n_kv;
  TORCH_CHECK(groups <= 32, "FAD_d: groups too large");

  if (softmax_scale <= 0.0f) softmax_scale = 1.0f / sqrtf((float)D);

  const int* pad_ptr = nullptr;
  if (pad_lens.has_value() && pad_lens->defined()) {
    TORCH_CHECK(pad_lens->is_cuda() && pad_lens->scalar_type() == at::kInt &&
                    pad_lens->dim() == 1 && pad_lens->size(0) == B && pad_lens->is_contiguous(),
                "FAD_d: pad_lens must be a contiguous CUDA int32 tensor of shape [B]");
    pad_ptr = pad_lens->const_data_ptr<int>();
  }

  auto Qc = Q.contiguous();
  auto Kc = K.contiguous();
  auto Vc = V.contiguous();

  int num_splits = (S + FAD_TILE - 1) / FAD_TILE;
  if (num_splits < 1) num_splits = 1;
  if (num_splits > FAD_MAX_SPLITS) num_splits = FAD_MAX_SPLITS;
  const int chunk = (S + num_splits - 1) / num_splits;

  auto fopts = Qc.options().dtype(at::kFloat);
  auto Opart = torch::zeros({B, nh, num_splits, D}, fopts);
  auto Mpart = torch::full({B, nh, num_splits}, -INFINITY, fopts);
  auto Lpart = torch::zeros({B, nh, num_splits}, fopts);
  auto out = torch::empty({B, nh, 1, D}, Qc.options());

  const int nwarps = groups > 8 ? groups : 8;
  const dim3 grid(num_splits, n_kv, B);
  const dim3 block(nwarps * 32);
  auto stream = at::cuda::getCurrentCUDAStream();

  fad_decode_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(Qc.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Kc.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Vc.data_ptr<at::Half>()),
      Opart.data_ptr<float>(), Mpart.data_ptr<float>(), Lpart.data_ptr<float>(),
      S, n_kv, nh, groups, num_splits, chunk, softmax_scale, pad_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fad_combine_kernel<<<dim3(nh, B), dim3(D), 0, stream>>>(
      Opart.data_ptr<float>(), Mpart.data_ptr<float>(), Lpart.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), num_splits, nh, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "FAD_d GQA split-K decode attention",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("softmax_scale") = -1.0f,
        py::arg("pad_lens") = py::none());
}
