// WMMA Tensor Core GEMM kernels for Qwen3-0.6B prefill matmuls
// Designed for IREE executable substitution on Jetson AGX Orin (sm_87)
//
// Each kernel: C[M,N] = A[M,K] @ B[K,N], all f16, M dynamic
// Some kernels also fuse: output = residual + matmul(input, weight)
//
// WMMA config: 16×16×16 f16 fragments
// Block: 128 threads = 4 warps in 2×2 layout
// Tile: 32×32 output per block, loop over K with TILE_K=16
// Grid: (ceil(M/32), ceil(N/32), 1)

#include <mma.h>
#include <cuda_fp16.h>
#include <stdint.h>
using namespace nvcuda;

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define WARPS_M 2
#define WARPS_N 2
#define WARP_SIZE 32
#define NTHREADS 128  // WARPS_M * WARPS_N * WARP_SIZE

// ============================================================
// Core GEMM + optional residual add
// ============================================================
// A[M, K] @ B[K, N] -> C[M, N]  (+ optional residual[M, N])
template<int N_DIM, int K_DIM>
__device__ void gemm_core(
    const __half* __restrict__ A,   // [M, K]
    const __half* __restrict__ B,   // [K, N]  (single layer slice)
    __half* __restrict__ C,         // [M, N]
    const __half* residual,         // [M, N] or nullptr
    int M
) {
    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_m = warp_id / WARPS_N;  // 0 or 1
    int warp_n = warp_id % WARPS_N;  // 0 or 1

    // Shared memory for input tiles
    __shared__ __half smem_A[TILE_M * TILE_K];  // 32×16 = 1024 bytes
    __shared__ __half smem_B[TILE_K * TILE_N];  // 16×32 = 1024 bytes

    // Accumulator
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    // Main K loop
    for (int k = 0; k < K_DIM; k += TILE_K) {
        // Load A tile [TILE_M × TILE_K] cooperatively
        for (int i = threadIdx.x; i < TILE_M * TILE_K; i += NTHREADS) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gr = block_m + row;
            int gc = k + col;
            smem_A[i] = (gr < M) ? A[gr * K_DIM + gc] : __float2half(0.0f);
        }

        // Load B tile [TILE_K × TILE_N] cooperatively
        for (int i = threadIdx.x; i < TILE_K * TILE_N; i += NTHREADS) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int gc = block_n + col;
            smem_B[i] = (gc < N_DIM) ? B[(k + row) * N_DIM + gc] : __float2half(0.0f);
        }

        __syncthreads();

        // WMMA: each warp computes its 16×16 tile
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, &smem_A[warp_m * 16 * TILE_K], TILE_K);
        wmma::load_matrix_sync(b_frag, &smem_B[warp_n * 16], TILE_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store result via shared memory (for bounds checking)
    __shared__ __half smem_out[TILE_M * TILE_N];  // 32×32 = 2048 bytes
    wmma::store_matrix_sync(
        &smem_out[warp_m * 16 * TILE_N + warp_n * 16],
        c_frag, TILE_N, wmma::mem_row_major);

    __syncthreads();

    // Write to global memory with bounds check (+ optional residual)
    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += NTHREADS) {
        int row = i / TILE_N;
        int col = i % TILE_N;
        int gr = block_m + row;
        int gc = block_n + col;
        if (gr < M && gc < N_DIM) {
            __half val = smem_out[i];
            if (residual) {
                __half res = residual[gr * N_DIM + gc];
                val = __hadd(val, res);
            }
            C[gr * N_DIM + gc] = val;
        }
    }
}

// Helper: decode i64 from two i32 push constants
__device__ __forceinline__ long long decode_i64(uint32_t lo, uint32_t hi) {
    return (long long)lo | ((long long)hi << 32);
}


// ============================================================
// dispatch_3: attn_q  [M, 1024] @ [1024, 2048] -> [M, 2048]
// 8 push constants, 3 bindings
// c0:1 = input offset, c2:3 = output offset, c4:5 = layer, c6:7 = M
// ============================================================
extern "C" __global__ __launch_bounds__(128)
void prefill_dispatch_3_matmul_Dx2048x1024_f16(
    const __half* __restrict__ weights_base,  // binding(0) [28,1024,2048]
    const char* __restrict__ b1_raw,          // binding(1) input
    char* __restrict__ b2_raw,                // binding(2) output
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t c4, uint32_t c5, uint32_t c6, uint32_t c7
) {
    const __half* A = (const __half*)(b1_raw + decode_i64(c0, c1));
    __half* C = (__half*)(b2_raw + decode_i64(c2, c3));
    long long layer = decode_i64(c4, c5);
    int M = (int)decode_i64(c6, c7);
    const __half* B = weights_base + layer * (1024LL * 2048);
    gemm_core<2048, 1024>(A, B, C, nullptr, M);
}


// ============================================================
// dispatch_4: attn_kv [M, 1024] @ [1024, 1024] -> [M, 1024]
// 8 push constants, 3 bindings (same layout as dispatch_3)
// ============================================================
extern "C" __global__ __launch_bounds__(128)
void prefill_dispatch_4_matmul_Dx1024x1024_f16(
    const __half* __restrict__ weights_base,  // binding(0) [28,1024,1024]
    const char* __restrict__ b1_raw,          // binding(1) input
    char* __restrict__ b2_raw,                // binding(2) output
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t c4, uint32_t c5, uint32_t c6, uint32_t c7
) {
    const __half* A = (const __half*)(b1_raw + decode_i64(c0, c1));
    __half* C = (__half*)(b2_raw + decode_i64(c2, c3));
    long long layer = decode_i64(c4, c5);
    int M = (int)decode_i64(c6, c7);
    const __half* B = weights_base + layer * (1024LL * 1024);
    gemm_core<1024, 1024>(A, B, C, nullptr, M);
}


// ============================================================
// dispatch_13: attn_o [M, 2048] @ [2048, 1024] -> [M, 1024] + residual
// 10 push constants, 3 bindings (binding(1) used for input AND residual)
// c0:1 = input offset, c2:3 = residual offset (same binding),
// c4:5 = output offset, c6:7 = layer, c8:9 = M
// ============================================================
extern "C" __global__ __launch_bounds__(128)
void prefill_dispatch_13_matmul_Dx1024x2048_f16(
    const __half* __restrict__ weights_base,  // binding(0) [28,2048,1024]
    const char* __restrict__ b1_raw,          // binding(1) input + residual
    char* __restrict__ b2_raw,                // binding(2) output
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t c4, uint32_t c5, uint32_t c6, uint32_t c7,
    uint32_t c8, uint32_t c9
) {
    const __half* A = (const __half*)(b1_raw + decode_i64(c0, c1));        // attn_flat [M,2048]
    const __half* residual = (const __half*)(b1_raw + decode_i64(c2, c3)); // residual [M,1024]
    __half* C = (__half*)(b2_raw + decode_i64(c4, c5));                     // output [M,1024]
    long long layer = decode_i64(c6, c7);
    int M = (int)decode_i64(c8, c9);
    const __half* B = weights_base + layer * (2048LL * 1024);
    gemm_core<1024, 2048>(A, B, C, residual, M);
}


// ============================================================
// dispatch_17: gate_up [M, 1024] @ [1024, 6144] -> [M, 6144]
// 6 push constants, 3 bindings
// c0:1 = output offset, c2:3 = layer, c4:5 = M
// binding(1) offset = 0
// ============================================================
extern "C" __global__ __launch_bounds__(128)
void prefill_dispatch_17_matmul_Dx6144x1024_f16(
    const __half* __restrict__ weights_base,  // binding(0) [28,1024,6144]
    const char* __restrict__ b1_raw,          // binding(1) input (offset 0)
    char* __restrict__ b2_raw,                // binding(2) output
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t c4, uint32_t c5
) {
    const __half* A = (const __half*)b1_raw;  // no offset
    __half* C = (__half*)(b2_raw + decode_i64(c0, c1));
    long long layer = decode_i64(c2, c3);
    int M = (int)decode_i64(c4, c5);
    const __half* B = weights_base + layer * (1024LL * 6144);
    gemm_core<6144, 1024>(A, B, C, nullptr, M);
}


// ============================================================
// dispatch_19: ffn_down [M, 3072] @ [3072, 1024] -> [M, 1024] + residual
// 8 push constants, 4 bindings
// c0:1 = input offset (b1), c2:3 = residual offset (b2),
// c4:5 = layer, c6:7 = M
// binding(3) offset = 0 (output)
// ============================================================
extern "C" __global__ __launch_bounds__(128)
void prefill_dispatch_19_matmul_Dx1024x3072_f16(
    const __half* __restrict__ weights_base,  // binding(0) [28,3072,1024]
    const char* __restrict__ b1_raw,          // binding(1) swiglu input
    const char* __restrict__ b2_raw,          // binding(2) residual
    __half* __restrict__ b3_out,              // binding(3) output (offset 0)
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t c4, uint32_t c5, uint32_t c6, uint32_t c7
) {
    const __half* A = (const __half*)(b1_raw + decode_i64(c0, c1));
    const __half* residual = (const __half*)(b2_raw + decode_i64(c2, c3));
    long long layer = decode_i64(c4, c5);
    int M = (int)decode_i64(c6, c7);
    const __half* B = weights_base + layer * (3072LL * 1024);
    gemm_core<1024, 3072>(A, B, b3_out, residual, M);
}
