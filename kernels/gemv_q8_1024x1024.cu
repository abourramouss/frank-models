#include <cuda_fp16.h>
#include <stdint.h>

extern "C"
__global__ __launch_bounds__(256)
void decode_dispatch_4_matvec_like_1024x1024_f16(
    const int8_t* __restrict__ q8_base,
    const char* __restrict__ binding1_raw,
    char* __restrict__ binding2_raw,
    uint32_t output_offset_bytes,
    uint32_t layer_lo,
    uint32_t layer_hi
) {
    // Binding offsets from dispatch source (must be added manually):
    // binding(0) offset=0     -> q8_base (28x1114112 weight tensor)
    // binding(1) offset=2112  -> input vector (1024 f16)
    // binding(2) offset=variable (8256 or 10304) -> output (push constant 0)
    const __half* input = (const __half*)(binding1_raw + 2112);
    __half* output = (__half*)(binding2_raw + output_offset_bytes);

    const int N = 1024, K = 1024;
    const long long BYTES_PER_LAYER = 1114112LL;
    const int BYTES_PER_ROW = 1088; // 32 Q8 blocks * 34 bytes

    long long layer_idx = (long long)layer_lo | ((long long)layer_hi << 32);
    const uint8_t* layer_data = (const uint8_t*)(q8_base + layer_idx * BYTES_PER_LAYER);

    // Shared memory for input vector
    __shared__ __half shmem[K];
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        shmem[i] = input[i];
    __syncthreads();

    // 8 rows per block, 32 threads (1 warp) per row
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= N) return;

    const uint8_t* row_data = layer_data + (long long)row * BYTES_PER_ROW;

    // Each thread reads 1 Q8 block (32 elements), accumulates dot product
    float acc = 0.0f;
    int byte_off = lane * 34;
    uint16_t scale_bits = row_data[byte_off] | ((uint16_t)row_data[byte_off + 1] << 8);
    float scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));

    int k_base = lane * 32;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int8_t qv = ((const int8_t*)row_data)[byte_off + 2 + j];
        acc += scale * (float)qv * __half2float(shmem[k_base + j]);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        output[row] = __float2half(acc);
}
