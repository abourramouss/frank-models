// Dispatch 25: matmul [1024, 3072] @ [3072, 1] -> [1024, 1] + residual add  (f16)
// Layout: 2 constants, 2 bindings (Indirect)
//   push_const[0] = byte offset for input vector in binding(0)
//   push_const[1] = byte offset for residual in binding(0)
//   binding(0) ReadOnly|Indirect: weight   [1024x3072xf16] at fixed offset 25165824
//                                 input    [3072x1xf16]    at push_const[0]
//                                 residual [1024x1xf16]    at push_const[1]
//   binding(1) Indirect:          output   [1024x1xf16]    at fixed offset 0
//
// Fused: output = matmul_result + residual
// Grid: (N/8, 1, 1) = (128, 1, 1),  Block: (256, 1, 1)

#include <cuda_fp16.h>
#include <stdint.h>

#define N_ROWS 1024
#define K_DIM  3072
#define ROWS_PER_BLOCK 8
#define WARP_SIZE 32

extern "C"
__global__ __launch_bounds__(256)
void decode_dispatch_25_matmul_1024x1x3072_f16(
    const char* __restrict__ binding0,
    char* __restrict__ binding1,
    uint32_t push_const_0,
    uint32_t push_const_1
) {
    // Offsets
    const __half* weight   = (const __half*)(binding0 + 25165824);
    const __half* input    = (const __half*)(binding0 + push_const_0);
    const __half* residual = (const __half*)(binding0 + push_const_1);
    __half* output         = (__half*)(binding1);  // offset 0

    // Load input vector into shared memory
    // K=3072 * 2 bytes = 6144 bytes shared memory
    __shared__ __half shmem[K_DIM];
    for (int i = threadIdx.x; i < K_DIM; i += blockDim.x)
        shmem[i] = input[i];
    __syncthreads();

    int row  = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= N_ROWS) return;

    const __half* w_row = weight + (long long)row * K_DIM;
    float acc = 0.0f;

    // K=3072, each warp lane processes 3072/32 = 96 elements
    #pragma unroll 4
    for (int j = 0; j < K_DIM / WARP_SIZE; j++) {
        int k = lane * (K_DIM / WARP_SIZE) + j;
        acc += __half2float(w_row[k]) * __half2float(shmem[k]);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    // Fused residual addition
    if (lane == 0) {
        float res = __half2float(residual[row]);
        output[row] = __float2half(acc + res);
    }
}
