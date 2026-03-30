// Custom GEMV: Q8_0 dequant + matvec for [6144, 1024]
// Drop-in replacement for IREE's decode_dispatch_16_matvec_like_6144x1024_f16
//
// Interface (must match IREE's generated PTX exactly):
//   param_0: binding(0) = Q8_0 weights [28, 6684672] i8 (base pointer)
//   param_1: binding(1) = input vector [1024] f16
//   param_2: binding(2) = output vector [6144] f16
//   param_3: push constant lo (u32) — layer index low bits
//   param_4: push constant hi (u32) — layer index high bits
//
// Layer offset: row_ptr = base + layer_idx * 6684672
//
// Strategy: 256 threads/block, 8 rows/block, shared memory for input vector

#include <cuda_fp16.h>
#include <stdint.h>

extern "C"
__global__ __launch_bounds__(256)
void decode_dispatch_16_matvec_like_6144x1024_f16(
    const int8_t* __restrict__ q8_base,   // param_0
    const __half* __restrict__ input,      // param_1
    __half* __restrict__ output,           // param_2
    uint32_t layer_lo,                     // param_3
    uint32_t layer_hi                      // param_4
) {
    const int N = 6144;
    const int K = 1024;
    const int ROWS_PER_BLOCK = 8;
    const int THREADS_PER_ROW = 32;
    const int Q8_BLOCK_SIZE = 32;
    const int Q8_BYTES_PER_BLOCK = 34;
    const int BYTES_PER_ROW = (K / Q8_BLOCK_SIZE) * Q8_BYTES_PER_BLOCK; // 1088
    const long long ROW_STRIDE = 6684672LL; // bytes per layer in stacked tensor

    // Compute layer base pointer (same as IREE: base + layer_idx * 6684672)
    long long layer_idx = (long long)layer_lo | ((long long)layer_hi << 32);
    const int8_t* layer_base = q8_base + layer_idx * ROW_STRIDE;

    // Load input vector into shared memory (2KB)
    __shared__ __half shmem[K];
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        shmem[i] = input[i];
    }
    __syncthreads();

    // Each block handles 8 output rows, 32 threads (1 warp) per row
    int local_row = threadIdx.x / THREADS_PER_ROW;
    int lane = threadIdx.x % THREADS_PER_ROW;
    int row = blockIdx.x * ROWS_PER_BLOCK + local_row;
    if (row >= N) return;

    // Pointer to this row's Q8_0 data
    const uint8_t* row_data = (const uint8_t*)(layer_base + (long long)row * BYTES_PER_ROW);

    // Each warp of 32 threads reduces 1024 elements
    // 1024 / 32 = 32 elements per thread = 1 Q8 block per thread
    float acc = 0.0f;

    int block_idx = lane; // thread i handles Q8 block i
    int byte_off = block_idx * Q8_BYTES_PER_BLOCK;

    // Read f16 scale (first 2 bytes of Q8 block)
    uint16_t scale_bits = row_data[byte_off] | ((uint16_t)row_data[byte_off + 1] << 8);
    float scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));

    // Read 32 i8 values and accumulate with input from shared memory
    int k_base = block_idx * Q8_BLOCK_SIZE;
    const int8_t* qvals = (const int8_t*)(row_data + byte_off + 2);

    // Unrolled inner loop (32 iterations)
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float w = scale * (float)qvals[j];
        float x = __half2float(shmem[k_base + j]);
        acc += w * x;
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    // Lane 0 writes result
    if (lane == 0) {
        output[row] = __float2half(acc);
    }
}
