// Hopper TMA and WGMMA FlashAttention prefill CUDA kernel.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#if defined(ENABLE_HOPPER_TMA_WGMMA) && ENABLE_HOPPER_TMA_WGMMA
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cassert>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t make_smem_desc(half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], half* sA, half* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

void create_tensor_map(CUtensorMap *tma_map, half* gmem_ptr, int total_rows, int d, CUtensorMapSwizzle swizzle) {
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {(uint64_t)d, (uint64_t)total_rows, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(half), sizeof(half) * d, 0, 0, 0};
    uint32_t smem_box_shape[5] = {uint32_t(d), uint32_t(64), 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

__device__ static inline int swizzle_idx(int r, int c) {
    return r * 64 + ((((c >> 3) ^ (r & 7)) << 3) | (c & 7));
}

__global__ void tma_fa2_pingpong_kernel(
    int N, int d, int Tc, int Tr, float softmax_scale, bool is_causal,

    const int* __restrict__ pad_lens,
    half* O,
    const __grid_constant__ CUtensorMap mapQ,
    const __grid_constant__ CUtensorMap mapK,
    const __grid_constant__ CUtensorMap mapV
) {
    int tx = threadIdx.x;
    int i  = blockIdx.x;
    int h  = blockIdx.y;
    int b  = blockIdx.z;
    int head_offset = (b * gridDim.y + h) * N;

    const int pad = pad_lens ? pad_lens[b] : 0;

    __shared__ alignas(128) half Qi[64 * 64];
    __shared__ alignas(128) half Kj[2][64 * 64];
    __shared__ alignas(128) half Vj[2][64 * 64];
    __shared__ alignas(128) half sP[64 * 64];
    __shared__ alignas(128) float S[64 * 64];
    __shared__ alignas(128) float s_alpha[64];
    __shared__ alignas(128) float s_l[64];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barQ;
    __shared__ barrier barK[2];
    __shared__ barrier barV[2];

    if (tx == 0) {
        init(&barQ, blockDim.x);
        init(&barK[0], blockDim.x);
        init(&barK[1], blockDim.x);
        init(&barV[0], blockDim.x);
        init(&barV[1], blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int q_row = head_offset + i * 64;
    barrier::arrival_token tokenQ;
    if (tx == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&Qi[0], &mapQ, 0, q_row, barQ);
        tokenQ = cuda::device::barrier_arrive_tx(barQ, 1, sizeof(Qi));
    } else {
        tokenQ = barQ.arrive();
    }
    barQ.wait(std::move(tokenQ));
    __syncthreads();

    int lane = tx % 32;
    int warp = tx / 32;
    uint32_t row1 = warp * 16 + lane / 4;
    uint32_t row2 = row1 + 8;

    float acc_o[4][8];
    memset(acc_o, 0, sizeof(acc_o));

    float row_m = -INFINITY;
    float row_l = 0.0f;
    int row_idx = (64 * i) + tx;

    int max_j = is_causal ? i : (Tc - 1);

    barrier::arrival_token tokenK[2];
    barrier::arrival_token tokenV[2];

    if (max_j >= 0) {
        int k_row_0 = head_offset + 0 * 64;
        if (tx == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&Kj[0][0], &mapK, 0, k_row_0, barK[0]);
            tokenK[0] = cuda::device::barrier_arrive_tx(barK[0], 1, sizeof(Kj[0]));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&Vj[0][0], &mapV, 0, k_row_0, barV[0]);
            tokenV[0] = cuda::device::barrier_arrive_tx(barV[0], 1, sizeof(Vj[0]));
        } else {
            tokenK[0] = barK[0].arrive();
            tokenV[0] = barV[0].arrive();
        }
    }

    for (int j = 0; j <= max_j; j++) {
        int curr_buf = j % 2;
        int next_buf = (j + 1) % 2;

        if (j + 1 <= max_j) {
            int next_k_row = head_offset + (j + 1) * 64;
            if (tx == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&Kj[next_buf][0], &mapK, 0, next_k_row, barK[next_buf]);
                tokenK[next_buf] = cuda::device::barrier_arrive_tx(barK[next_buf], 1, sizeof(Kj[next_buf]));
                cde::cp_async_bulk_tensor_2d_global_to_shared(&Vj[next_buf][0], &mapV, 0, next_k_row, barV[next_buf]);
                tokenV[next_buf] = cuda::device::barrier_arrive_tx(barV[next_buf], 1, sizeof(Vj[next_buf]));
            } else {
                tokenK[next_buf] = barK[next_buf].arrive();
                tokenV[next_buf] = barV[next_buf].arrive();
            }
        }

        barK[curr_buf].wait(std::move(tokenK[curr_buf]));
        __syncthreads();

        float acc_s[4][8];
        memset(acc_s, 0, sizeof(acc_s));

        warpgroup_arrive();
        wgmma64<1, 1, 1, 0, 0>(acc_s, &Qi[0], &Kj[curr_buf][0]);
        wgmma64<1, 1, 1, 0, 0>(acc_s, &Qi[16], &Kj[curr_buf][16]);
        wgmma64<1, 1, 1, 0, 0>(acc_s, &Qi[32], &Kj[curr_buf][32]);
        wgmma64<1, 1, 1, 0, 0>(acc_s, &Qi[48], &Kj[curr_buf][48]);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        for (int w = 0; w < 4; ++w) {
            int col = 16 * w + 2 * (tx % 4);

            S[row1 * 64 + col]           = acc_s[w][0];
            S[row1 * 64 + col + 1]       = acc_s[w][1];
            S[row2 * 64 + col]           = acc_s[w][2];
            S[row2 * 64 + col + 1]       = acc_s[w][3];

            S[row1 * 64 + col + 8]       = acc_s[w][4];
            S[row1 * 64 + col + 9]       = acc_s[w][5];
            S[row2 * 64 + col + 8]       = acc_s[w][6];
            S[row2 * 64 + col + 9]       = acc_s[w][7];
        }
        __syncthreads();

        float m_tile = -INFINITY;
        if (tx < 64 && row_idx < N) {
            for (int y = 0; y < 64; y++) {
                int curr_col = (64 * j) + y;
                float sum = S[(64 * tx) + y];
                if (is_causal && curr_col > row_idx) {
                    sum = -INFINITY;
                } else if (curr_col < N && curr_col >= pad) {
                    sum *= softmax_scale;
                } else {
                    sum = -INFINITY;
                }

                S[(64 * tx) + y] = sum;
                if (sum > m_tile) m_tile = sum;
            }
        }

        float m_prev = row_m;
        float m_new = max(m_prev, m_tile);
        float alpha = (m_prev == -INFINITY) ? 0.0f : __expf(m_prev - m_new);

        float sum_p = 0.0f;
        if (tx < 64) {
            if (row_idx < N && m_tile > -INFINITY) {
                for (int y = 0; y < 64; y++) {
                    if (S[(64 * tx) + y] > -INFINITY) {
                        float p_val = __expf(S[(64 * tx) + y] - m_new);
                        sP[swizzle_idx(tx, y)] = __float2half(p_val);
                        sum_p += p_val;
                    } else {
                        sP[swizzle_idx(tx, y)] = __float2half(0.0f);
                    }
                }
            } else {
                for (int y = 0; y < 64; y++) {
                    sP[swizzle_idx(tx, y)] = __float2half(0.0f);
                }
            }
            row_l = (alpha * row_l) + sum_p;
            row_m = m_new;
            s_alpha[tx] = alpha;
        }
        __syncthreads();

        float a1 = (row1 < 64) ? s_alpha[row1] : 0.0f;
        float a2 = (row2 < 64) ? s_alpha[row2] : 0.0f;

        for (int w = 0; w < 4; ++w) {
            acc_o[w][0] *= a1;
            acc_o[w][1] *= a1;
            acc_o[w][4] *= a1;
            acc_o[w][5] *= a1;

            acc_o[w][2] *= a2;
            acc_o[w][3] *= a2;
            acc_o[w][6] *= a2;
            acc_o[w][7] *= a2;
        }

        barV[curr_buf].wait(std::move(tokenV[curr_buf]));
        __syncthreads();

        warpgroup_arrive();
        wgmma64<1, 1, 1, 0, 1>(acc_o, &sP[0],  &Vj[curr_buf][0]);
        wgmma64<1, 1, 1, 0, 1>(acc_o, &sP[16], &Vj[curr_buf][1024]);
        wgmma64<1, 1, 1, 0, 1>(acc_o, &sP[32], &Vj[curr_buf][2048]);
        wgmma64<1, 1, 1, 0, 1>(acc_o, &sP[48], &Vj[curr_buf][3072]);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        __syncthreads();
    }

    if (tx < 64) {
        s_l[tx] = row_l;
    }
    __syncthreads();

    float inv_l1 = (row1 < 64 && s_l[row1] > 0.0f) ? (1.0f / s_l[row1]) : 0.0f;
    float inv_l2 = (row2 < 64 && s_l[row2] > 0.0f) ? (1.0f / s_l[row2]) : 0.0f;

    for (int w = 0; w < 4; ++w) {
        acc_o[w][0] *= inv_l1;
        acc_o[w][1] *= inv_l1;
        acc_o[w][4] *= inv_l1;
        acc_o[w][5] *= inv_l1;

        acc_o[w][2] *= inv_l2;
        acc_o[w][3] *= inv_l2;
        acc_o[w][6] *= inv_l2;
        acc_o[w][7] *= inv_l2;
    }

    for (int w = 0; w < 4; ++w) {
        int col = 16 * w + 2 * (tx % 4);

        S[row1 * 64 + col]           = acc_o[w][0];
        S[row1 * 64 + col + 1]       = acc_o[w][1];
        S[row2 * 64 + col]           = acc_o[w][2];
        S[row2 * 64 + col + 1]       = acc_o[w][3];

        S[row1 * 64 + col + 8]       = acc_o[w][4];
        S[row1 * 64 + col + 9]       = acc_o[w][5];
        S[row2 * 64 + col + 8]       = acc_o[w][6];
        S[row2 * 64 + col + 9]       = acc_o[w][7];
    }
    __syncthreads();

    for (int idx = tx; idx < 64 * d; idx += blockDim.x) {
        int r = idx / d;
        int c = idx % d;
        int g_r = i * 64 + r;
        if (g_r < N) {
            O[(head_offset + g_r) * d + c] = __float2half(S[idx]);
        }
    }
}

torch::Tensor forward_sm90_tma_wgmma(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                     bool is_causal, float softmax_scale,
                                     const int* pad_ptr ) {

    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "FA_d: Q/K/V must be contiguous");
    TORCH_CHECK(K.sizes() == Q.sizes() && V.sizes() == Q.sizes(),
                "FA_d: Q/K/V must have identical [B, nh, N, d] shapes");
    int B = Q.size(0);
    int nh = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    int Tc = (N + 63) / 64;
    int Tr = (N + 63) / 64;

    auto O = torch::zeros_like(Q);

    int total_rows = B * nh * N;
    CUtensorMap hostMapQ, hostMapK, hostMapV;
    create_tensor_map(&hostMapQ, reinterpret_cast<half*>(Q.data_ptr<at::Half>()), total_rows, d, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tensor_map(&hostMapK, reinterpret_cast<half*>(K.data_ptr<at::Half>()), total_rows, d, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tensor_map(&hostMapV, reinterpret_cast<half*>(V.data_ptr<at::Half>()), total_rows, d, CU_TENSOR_MAP_SWIZZLE_128B);

    dim3 grid(Tr, nh, B);
    dim3 block(128);

    tma_fa2_pingpong_kernel<<<grid, block>>>(
        N, d, Tc, Tr, softmax_scale, is_causal, pad_ptr,
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        hostMapQ, hostMapK, hostMapV
    );

    return O;
}
#endif

__device__ __forceinline__ void wgmma_64x64x16_emulated(float* S_out, const float* A_tile, const float* B_tile) {
    int tid = threadIdx.x;
    for (int i = tid; i < 64 * 64; i += blockDim.x) {
        int r = i / 64;
        int c = i % 64;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            acc += A_tile[r * 64 + k] * B_tile[c * 64 + k];
        }
        S_out[i] += acc;
    }
}

__global__
void forward_kernel_fallback(const float* Q, const float* K, const float* V, const int N, const int d,
                             const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                             const bool is_causal,

                             const int* pad_lens,
                             float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;

    const int pad = pad_lens ? pad_lens[bx] : 0;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {
        if (tx < Bc) {
            int col_idx = (Bc * j) + tx;
            for (int x = 0; x < d; x++) {
                if (col_idx < N) {
                    Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                    Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
                } else {
                    Kj[(tx * d) + x] = 0.0f;
                    Vj[(tx * d) + x] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int i = (is_causal ? j : 0); i < Tr; i++) {
            int row_idx = (Br * i) + tx;
            if (tx < Br) {
                for (int x = 0; x < d; x++) {
                    if (row_idx < N) {
                        Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                    } else {
                        Qi[(tx * d) + x] = 0.0f;
                    }
                }
            }

            for (int idx = tx; idx < Bc * Br; idx += blockDim.x) {
                S[idx] = 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (int k_step = 0; k_step < 64; k_step += 16) {
                wgmma_64x64x16_emulated(S, Qi + k_step, Kj + k_step);
            }
            __syncthreads();

            float row_m_prev = (tx < Br && row_idx < N) ? m[lm_offset + (Br * i) + tx] : -INFINITY;
            float row_l_prev = (tx < Br && row_idx < N) ? l[lm_offset + (Br * i) + tx] : 0.0f;

            float row_m = -INFINITY;
            if (tx < Br) {
                for (int y = 0; y < Bc; y++) {
                    int curr_col = (Bc * j) + y;
                    float sum = S[(Bc * tx) + y];

                    if (is_causal && curr_col > row_idx) {
                        sum = -INFINITY;
                    } else if (row_idx < N && curr_col < N && curr_col >= pad) {

                        sum *= softmax_scale;
                    } else {
                        sum = -INFINITY;
                    }
                    S[(Bc * tx) + y] = sum;

                    if (sum > row_m)
                        row_m = sum;
                }
            }

            float row_l = 0.0f;
            float row_m_new = row_m_prev;
            float row_l_new = row_l_prev;

            if (tx < Br) {
                if (row_idx < N && row_m > -INFINITY) {
                    for (int y = 0; y < Bc; y++) {
                        if (S[(Bc * tx) + y] > -INFINITY) {
                            S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                            row_l += S[(Bc * tx) + y];
                        } else {
                            S[(Bc * tx) + y] = 0.0f;
                        }
                    }
                    row_m_new = max(row_m_prev, row_m);
                    row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);
                } else {
                    for (int y = 0; y < Bc; y++) {
                        S[(Bc * tx) + y] = 0.0f;
                    }
                }
            }
            __syncthreads();

            float* PV_tile = Qi;
            for (int idx = tx; idx < Br * d; idx += blockDim.x) {
                int r = idx / d;
                int x = idx % d;
                float acc = 0.0f;
                #pragma unroll 16
                for (int y = 0; y < Bc; y++) {
                    acc += S[(r * Bc) + y] * Vj[(y * d) + x];
                }
                PV_tile[idx] = acc;
            }
            __syncthreads();

            if (tx < Br && row_idx < N && row_m > -INFINITY) {
                for (int x = 0; x < d; x++) {
                    float pv = PV_tile[(tx * d) + x];
                    int out_idx = qkv_offset + (tile_size * i) + (tx * d) + x;
                    float prev_term = (row_l_prev > 0.0f) ? (row_l_prev * __expf(row_m_prev - row_m_new) * O[out_idx]) : 0.0f;
                    O[out_idx] = (1.0f / row_l_new) * (prev_term + (__expf(row_m - row_m_new) * pv));
                }
                m[lm_offset + (Br * i) + tx] = row_m_new;
                l[lm_offset + (Br * i) + tx] = row_l_new;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      bool is_causal = true, float softmax_scale = -1.0f,

                      c10::optional<torch::Tensor> pad_lens = c10::nullopt) {
    const int d = Q.size(3);
    if (softmax_scale <= 0.0f) {
        softmax_scale = 1.0f / sqrtf((float)d);
    }

    const int* pad_ptr = nullptr;
    const bool has_pad = pad_lens.has_value() && pad_lens->defined();
    if (has_pad) {
        TORCH_CHECK(pad_lens->is_cuda() && pad_lens->scalar_type() == at::kInt &&
                        pad_lens->dim() == 1 && pad_lens->size(0) == Q.size(0) &&
                        pad_lens->is_contiguous(),
                    "FA_d: pad_lens must be a contiguous CUDA int32 tensor of shape [B]");
        pad_ptr = pad_lens->const_data_ptr<int>();
    }

#if defined(ENABLE_HOPPER_TMA_WGMMA) && ENABLE_HOPPER_TMA_WGMMA
    if (Q.dtype() == torch::kFloat16 && d == 64) {
        return forward_sm90_tma_wgmma(Q, K, V, is_causal, softmax_scale, pad_ptr);
    }
#endif

    TORCH_CHECK(Q.scalar_type() == at::kFloat && K.scalar_type() == at::kFloat &&
                    V.scalar_type() == at::kFloat,
                "FA_d: the portable fallback is fp32-only; cast Q/K/V to float");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "FA_d: Q/K/V must be contiguous");
    TORCH_CHECK(K.sizes() == Q.sizes() && V.sizes() == Q.sizes(),
                "FA_d: Q/K/V must have identical [B, nh, N, d] shapes");

    const int Bc = 64; const int Br = 64;
    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int N  = Q.size(2);
    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N}, Q.options());
    auto m = torch::full({B, nh, N}, -INFINITY, Q.options());

    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc * 2);

    cudaFuncSetAttribute(forward_kernel_fallback, cudaFuncAttributeMaxDynamicSharedMemorySize, sram_size);

    forward_kernel_fallback<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        is_causal, pad_ptr,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CUDA FlashAttention forward",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("is_causal") = true, py::arg("softmax_scale") = -1.0f,
          py::arg("pad_lens") = py::none());
}
