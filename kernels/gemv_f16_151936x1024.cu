#include <cuda_fp16.h>
#include <stdint.h>

extern "C"
__global__ __launch_bounds__(256)
void decode_dispatch_21_matvec_like_151936x1024_f16(
    const __half* __restrict__ weights,
    const char* __restrict__ binding1_raw,
    char* __restrict__ binding2_raw,
    uint32_t output_offset_bytes
) {
    // Binding offsets from dispatch source (must be added manually):
    // binding(0) offset=0    -> weights: tensor<151936x1024xf16> (plain f16, ReadOnly)
    // binding(1) offset=64   -> input vector (1024 f16)
    // binding(2) offset=variable (0 or 2112) -> output vector (151936 f16)
    const __half* input = (const __half*)(binding1_raw + 64);
    __half* output = (__half*)(binding2_raw + output_offset_bytes);

    const int N = 151936, K = 1024;

    // Shared memory for input vector
    __shared__ __half shmem[K];
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        shmem[i] = input[i];
    __syncthreads();

    // 8 rows per block, 32 threads (1 warp) per row
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= N) return;

    // Plain f16 weights: row-major, each row is 1024 f16 values
    const __half* row_data = weights + (long long)row * K;

    // Each thread reads 32 consecutive f16 elements (K=1024, 32 threads, 32 each)
    float acc = 0.0f;
    int k_base = lane * 32;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        acc += __half2float(row_data[k_base + j]) * __half2float(shmem[k_base + j]);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        output[row] = __float2half(acc);
}
