#include <cuda_fp16.h>
#include <stdint.h>

extern "C"
__global__ __launch_bounds__(256)
void decode_dispatch_16_matvec_like_6144x1024_f16(
    const int8_t* __restrict__ q8_base,
    const __half* __restrict__ input,
    __half* __restrict__ output_raw,   // NOTE: needs +6208 byte offset!
    uint32_t layer_lo,
    uint32_t layer_hi
) {
    // Apply the binding(2) offset that IREE doesn't pre-apply for externalized kernels
    __half* output = (__half*)((char*)output_raw + 6208);
    
    const int N = 6144, K = 1024;
    const long long BYTES_PER_LAYER = 6684672LL;
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
