#include <mma.h>
#include <cuda_fp16.h>
#include <stdint.h>
using namespace nvcuda;

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define NTHREADS 128
#define WARP_SIZE 32
#define WARPS_N 2

__device__ __forceinline__ long long decode_i64(uint32_t lo, uint32_t hi) {
    return (long long)lo | ((long long)hi << 32);
}

template<int N_DIM, int K_DIM>
__device__ void gemm_core(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    const __half* residual,
    int M
) {
    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    __shared__ __half smem_A[TILE_M * TILE_K];
    __shared__ __half smem_B[TILE_K * TILE_N];

    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    for (int k = 0; k < K_DIM; k += TILE_K) {
        for (int i = threadIdx.x; i < TILE_M * TILE_K; i += NTHREADS) {
            int row = i / TILE_K, col = i % TILE_K;
            int gr = block_m + row;
            smem_A[i] = (gr < M) ? A[gr * K_DIM + (k + col)] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < TILE_K * TILE_N; i += NTHREADS) {
            int row = i / TILE_N, col = i % TILE_N;
            int gc = block_n + col;
            smem_B[i] = (gc < N_DIM) ? B[(k + row) * N_DIM + gc] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, &smem_A[warp_m * 16 * TILE_K], TILE_K);
        wmma::load_matrix_sync(b_frag, &smem_B[warp_n * 16], TILE_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    __shared__ __half smem_out[TILE_M * TILE_N];
    wmma::store_matrix_sync(&smem_out[warp_m * 16 * TILE_N + warp_n * 16], c_frag, TILE_N, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += NTHREADS) {
        int row = i / TILE_N, col = i % TILE_N;
        int gr = block_m + row, gc = block_n + col;
        if (gr < M && gc < N_DIM) {
            __half val = smem_out[i];
            if (residual) val = __hadd(val, residual[gr * N_DIM + gc]);
            C[gr * N_DIM + gc] = val;
        }
    }
}
