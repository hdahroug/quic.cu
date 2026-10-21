// Precomputed trigonometric rotary position embedding (RoPE) CUDA kernel.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

#define HEAD_DIM 64
#define HALF_DIM 32
#define WARP_SIZE 32
#define HEADS_PER_BLOCK 4
#define THREADS_PER_BLOCK (HEADS_PER_BLOCK * WARP_SIZE) 

__global__ void qwen2_rope_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    const half* __restrict__ cos_table,
    const half* __restrict__ sin_table,
    int seq_len,
    int num_heads,
    int head_dim,
    int trig_batch_stride
) {
    __shared__ half s_cos[HALF_DIM];
    __shared__ half s_sin[HALF_DIM];

    int tid = threadIdx.x;
    int lane_id = tid & 31;          
    int warp_id = tid >> 5;

    int token_idx = blockIdx.x; 
    int head_block = blockIdx.y;  
    int head_idx    = head_block * HEADS_PER_BLOCK + warp_id;
    int batch_idx = blockIdx.z; 

    if (tid < HALF_DIM) {
        int trig_offset = batch_idx * trig_batch_stride + token_idx * HALF_DIM + tid;
        s_cos[tid] = cos_table[trig_offset];
        s_sin[tid] = sin_table[trig_offset];
    }
    __syncthreads();

    if (head_idx >= num_heads) return;

    int stride_b = num_heads * seq_len * head_dim;
    int stride_h = seq_len * head_dim;
    int stride_s = head_dim;

    int offset = batch_idx * stride_b + head_idx * stride_h + token_idx * stride_s;
    int idx0 = offset + lane_id;                 
    int idx1 = offset + lane_id + HALF_DIM; 

    float cos_val = __half2float(s_cos[lane_id]);
    float sin_val = __half2float(s_sin[lane_id]);

    float v0 = __half2float(src[idx0]);
    float v1 = __half2float(src[idx1]);

    float out0 = v0 * cos_val - v1 * sin_val;
    float out1 = v0 * sin_val + v1 * cos_val;
    dst[idx0]=__float2half(out0);
    dst[idx1]=__float2half(out1);
}

__global__ void qwen2_rope_table_init_kernel(
    const float* __restrict__ inv_freq,   
    half* __restrict__ cos_table,         
    half* __restrict__ sin_table,          
    int max_seq_len
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= max_seq_len) return;

    half2* cos2 = reinterpret_cast<half2*>(cos_table + (size_t)t * HALF_DIM);
    half2* sin2 = reinterpret_cast<half2*>(sin_table + (size_t)t * HALF_DIM);

    #pragma unroll
    for (int c = 0; c < HALF_DIM; c += 2) {
        float a0 = (float)t * inv_freq[c];
        float a1 = (float)t * inv_freq[c + 1];
        cos2[c >> 1] = __floats2half2_rn(__cosf(a0), __cosf(a1));
        sin2[c >> 1] = __floats2half2_rn(__sinf(a0), __sinf(a1));
    }
}

torch::Tensor qwen2_rope_forward(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "Input must be Float16");

    auto out = torch::empty_like(x);
    int batch_size = x.size(0);
    int num_heads  = x.size(1);
    int seq_len    = x.size(2);
    int head_dim   = x.size(3);

    TORCH_CHECK(x.dim() == 4 && x.is_contiguous(), "x must be contiguous [B, H, S, D]");
    TORCH_CHECK(head_dim == HEAD_DIM, "head_dim must be ", HEAD_DIM);
    TORCH_CHECK(cos.scalar_type() == torch::kHalf && sin.scalar_type() == torch::kHalf,
                "cos/sin must be Float16");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos/sin must be contiguous");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos/sin shape mismatch");
    int trig_batch_stride = 0;
    if (cos.dim() == 3) {
        TORCH_CHECK(cos.size(0) == batch_size && cos.size(1) == seq_len && cos.size(2) == HALF_DIM,
                    "per-row cos/sin must be [B, S, ", HALF_DIM, "]");
        trig_batch_stride = seq_len * HALF_DIM;
    } else {
        TORCH_CHECK(cos.dim() == 2 && cos.size(0) == seq_len && cos.size(1) == HALF_DIM,
                    "shared cos/sin must be [S, ", HALF_DIM, "]");
    }

    int num_head_blocks = (num_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;
    dim3 grid(seq_len, num_head_blocks, batch_size);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    qwen2_rope_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        seq_len, num_heads, head_dim, trig_batch_stride
    );
    return out;
}

std::vector<torch::Tensor> qwen2_rope_table_init(torch::Tensor inv_freq, int64_t max_seq_len) {
    TORCH_CHECK(inv_freq.is_cuda(), "inv_freq must be a CUDA tensor");
    TORCH_CHECK(inv_freq.scalar_type() == torch::kFloat, "inv_freq must be fp32");
    TORCH_CHECK(inv_freq.numel() == HALF_DIM, "inv_freq must have exactly HALF_DIM elements");

    auto opts = inv_freq.options().dtype(torch::kHalf);
    auto cos = torch::empty({max_seq_len, HALF_DIM}, opts);
    auto sin = torch::empty({max_seq_len, HALF_DIM}, opts);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int threads = 256;
    int blocks = (int)((max_seq_len + threads - 1) / threads);
    qwen2_rope_table_init_kernel<<<blocks, threads, 0, stream>>>(
        inv_freq.data_ptr<float>(),
        reinterpret_cast<half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<half*>(sin.data_ptr<at::Half>()),
        (int)max_seq_len
    );
    return {cos, sin};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_table", &qwen2_rope_table_init, "Qwen2 RoPE cos/sin tables init");
    m.def("forward", &qwen2_rope_forward, "Qwen2 RoPE Forward");
}
