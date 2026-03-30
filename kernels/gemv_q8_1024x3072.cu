#include <cuda_fp16.h>
#include <stdint.h>

extern "C"
__global__ __launch_bounds__(256)
void decode_dispatch_18_matvec_like_1024x3072_f16(
    const int8_t* __restrict__ q8_base,
    const char* __restrict__ binding1_raw,
    __half* __restrict__ output_raw,
    uint32_t layer_lo,
    uint32_t layer_hi
) {
    // Binding offsets from dispatch source (must be added manually):
    // binding(0) offset=0      -> q8_base (28x3342336 weight tensor)
    // binding(1) offset=18496  -> input vector (3072 f16 values)
    // binding(1) offset=4160   -> residual vector (1024 f16 values)
    // binding(2) offset=0      -> output (no offset needed!)
    //
    // This dispatch computes: output = residual + matmul(q8_weights, input)
    const __half* input    = (const __half*)(binding1_raw + 18496);
    const __half* residual = (const __half*)(binding1_raw + 4160);

    const int N = 1024, K = 3072;
    const long long BYTES_PER_LAYER = 3342336LL;
    const int BYTES_PER_ROW = 3264; // 96 Q8 blocks * 34 bytes
    const int BLOCKS_PER_ROW = 96;  // K/32 = 3072/32

    long long layer_idx = (long long)layer_lo | ((long long)layer_hi << 32);
    const uint8_t* layer_data = (const uint8_t*)(q8_base + layer_idx * BYTES_PER_LAYER);

    // Shared memory for input vector (3072 f16 = 6144 bytes)
    __shared__ __half shmem[K];
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        shmem[i] = input[i];
    __syncthreads();

    // 8 rows per block, 32 threads (1 warp) per row
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= N) return;

    const uint8_t* row_data = layer_data + (long long)row * BYTES_PER_ROW;

    // Each thread processes multiple Q8 blocks (96 blocks / 32 threads = 3 per thread)
    float acc = 0.0f;
    for (int b = lane; b < BLOCKS_PER_ROW; b += 32) {
        int byte_off = b * 34;
        uint16_t scale_bits = row_data[byte_off] | ((uint16_t)row_data[byte_off + 1] << 8);
        float scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));

        int k_base = b * 32;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            int8_t qv = ((const int8_t*)row_data)[byte_off + 2 + j];
            acc += scale * (float)qv * __half2float(shmem[k_base + j]);
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0) {
        // Add residual (fused add from the dispatch)
        float res = __half2float(residual[row]);
        output_raw[row] = __float2half(acc + res);
    }
}
