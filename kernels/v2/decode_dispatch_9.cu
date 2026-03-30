// Dispatch 9: matmul [2048, 1024] @ [1024, 1] -> [2048, 1]  (f16)
// Layout: 0 constants, 2 bindings (Indirect)
//   binding(0) ReadOnly|Indirect: weight [2048x1024xf16] at offset 0
//                                 input  [1024x1xf16]    at offset 31459392
//   binding(1) Indirect:          output [2048x1xf16]    at offset 31461440
//
// Grid: (N/8, 1, 1) = (256, 1, 1),  Block: (256, 1, 1)
// 8 rows per block, 8 warps of 32 threads each
// Each warp computes one output row by reducing over K=1024

#include <cuda_fp16.h>
#include <stdint.h>

#define N_ROWS 2048
#define K_DIM  1024
#define ROWS_PER_BLOCK 8
#define WARP_SIZE 32

extern "C"
__global__ __launch_bounds__(256)
void decode_dispatch_9_matmul_2048x1x1024_f16(
    const char* __restrict__ binding0,
    char* __restrict__ binding1
) {
    // Hardcoded offsets from dispatch source
    const __half* weight = (const __half*)(binding0);                  // offset 0
    const __half* input  = (const __half*)(binding0 + 31459392);       // offset 31459392
    __half* output       = (__half*)(binding1 + 31461440);             // offset 31461440

    // Load input vector into shared memory
    __shared__ __half shmem[K_DIM];
    for (int i = threadIdx.x; i < K_DIM; i += blockDim.x)
        shmem[i] = input[i];
    __syncthreads();

    // 8 rows per block, 1 warp per row
    int row  = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= N_ROWS) return;

    // Each thread processes K_DIM/WARP_SIZE = 32 elements
    const __half* w_row = weight + (long long)row * K_DIM;
    float acc = 0.0f;

    #pragma unroll
    for (int j = 0; j < K_DIM / WARP_SIZE; j++) {
        int k = lane * (K_DIM / WARP_SIZE) + j;
        acc += __half2float(w_row[k]) * __half2float(shmem[k]);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        output[row] = __float2half(acc);
}
