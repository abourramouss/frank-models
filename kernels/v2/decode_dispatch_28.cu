// Dispatch 28: matmul [151936, 1024] @ [1024, 1] -> [151936, 1]  (f16)
// Layout: 1 constant, 3 bindings (Indirect)
//   push_const[0] = byte offset for output in binding(2)
//   binding(0) ReadOnly:          weight [151936x1024xf16] at offset 0
//   binding(1) ReadOnly|Indirect: input  [1024x1xf16]     at offset 64
//   binding(2) Indirect:          output [151936x1xf16]    at push_const[0]
//
// Grid: (ceil(N/8), 1, 1) = (18992, 1, 1),  Block: (256, 1, 1)
// Note: 151936 / 8 = 18992 exactly

#include <cuda_fp16.h>
#include <stdint.h>

#define N_ROWS 151936
#define K_DIM  1024
#define ROWS_PER_BLOCK 8
#define WARP_SIZE 32

extern "C"
__global__ __launch_bounds__(256)
void decode_dispatch_28_matmul_151936x1x1024_f16(
    const char* __restrict__ binding0,
    const char* __restrict__ binding1,
    char* __restrict__ binding2,
    uint32_t push_const_0
) {
    // Offsets
    const __half* weight = (const __half*)(binding0);           // offset 0
    const __half* input  = (const __half*)(binding1 + 64);      // offset 64
    __half* output       = (__half*)(binding2 + push_const_0);  // push_const[0]

    // Load input vector into shared memory
    __shared__ __half shmem[K_DIM];
    for (int i = threadIdx.x; i < K_DIM; i += blockDim.x)
        shmem[i] = input[i];
    __syncthreads();

    int row  = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= N_ROWS) return;

    const __half* w_row = weight + (long long)row * K_DIM;
    float acc = 0.0f;

    // K=1024, each warp lane processes 1024/32 = 32 elements
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
