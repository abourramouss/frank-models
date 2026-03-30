// Q8_0 fused dequant+GEMV kernels for Qwen3-0.6B decode
// output[row] = sum_k( dequant(q8_weight[row][k]) * input[k] )
//
// Q8_0 format: blocks of 32 values, each 34 bytes = 2B f16 scale + 32B i8 values
// Weight layout: GGUF native [N, K] stored as Q8_0 blocks row-major
//
// Kernel: 256 threads = 8 warps, 8 rows per block, 1 warp per row
// Each warp lane handles one Q8_0 block (32 elements), then warp shuffle reduce
// For K=1024: 32 Q8 blocks/row, 32 lanes/warp → 1 block per lane (perfect)
// For K=2048: 64 Q8 blocks/row, 32 lanes/warp → 2 blocks per lane
// For K=3072: 96 Q8 blocks/row, 32 lanes/warp → 3 blocks per lane
//
// Interface: IREE dispatch with stacked Q8 weights + input vector + output vector
// Push constants encode the layer index (i64 as 2×i32)

#include <cuda_fp16.h>
#include <stdint.h>

#define ROWS_PER_BLOCK 8
#define WARP_SIZE 32
#define NTHREADS 256  // 8 warps

// Generic Q8_0 fused GEMV: output[0..N-1] = Q8_weight[N,K] @ input[0..K-1]
template<int N, int K>
__device__ void q8_gemv_core(
    const uint8_t* __restrict__ layer_data,  // Q8_0 raw bytes for this layer [N*K/32*34]
    const __half* __restrict__ input,         // [K]
    __half* __restrict__ output               // [N]
) {
    const int Q8_BLOCK = 32;
    const int Q8_BYTES = 34;
    const int BYTES_PER_ROW = (K / Q8_BLOCK) * Q8_BYTES;
    const int BLOCKS_PER_ROW = K / Q8_BLOCK;

    // Load input vector to shared memory
    __shared__ __half shmem[K];
    for (int i = threadIdx.x; i < K; i += NTHREADS)
        shmem[i] = input[i];
    __syncthreads();

    // Each warp processes one row
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= N) return;

    const uint8_t* row_data = layer_data + (long long)row * BYTES_PER_ROW;

    float acc = 0.0f;

    // Each lane processes BLOCKS_PER_ROW/32 Q8 blocks
    // For K=1024: 1 block per lane, K=2048: 2, K=3072: 3
    for (int b = lane; b < BLOCKS_PER_ROW; b += WARP_SIZE) {
        int byte_off = b * Q8_BYTES;
        // Read scale (2 bytes as f16)
        uint16_t scale_bits = row_data[byte_off] | ((uint16_t)row_data[byte_off + 1] << 8);
        float scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));

        // Dot product for this block's 32 elements
        int k_base = b * Q8_BLOCK;
        #pragma unroll
        for (int j = 0; j < Q8_BLOCK; j++) {
            int8_t qv = ((const int8_t*)row_data)[byte_off + 2 + j];
            acc += scale * (float)qv * __half2float(shmem[k_base + j]);
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        output[row] = __float2half(acc);
}


// Helper: decode i64 from two i32 push constants
__device__ __forceinline__ long long decode_i64(uint32_t lo, uint32_t hi) {
    return (long long)lo | ((long long)hi << 32);
}


// ============================================================
// Dispatch kernels
// All take: Q8 stacked weights (binding 0), input vector (binding 1),
// output vector (binding 2), layer index as 2 push constants
// ============================================================

// attn_q: [2048, 1024] — 32 Q8 blocks/row, 1 per lane
extern "C" __global__ __launch_bounds__(256)
void q8_gemv_2048x1024(
    const uint8_t* __restrict__ q8_stacked,  // [28, 2228224] i8
    const __half* __restrict__ input,         // [1024]
    __half* __restrict__ output,              // [2048]
    uint32_t layer_lo, uint32_t layer_hi
) {
    long long layer = decode_i64(layer_lo, layer_hi);
    q8_gemv_core<2048, 1024>(q8_stacked + layer * 2228224LL, input, output);
}

// attn_k: [1024, 1024] — 32 Q8 blocks/row, 1 per lane
extern "C" __global__ __launch_bounds__(256)
void q8_gemv_1024x1024(
    const uint8_t* __restrict__ q8_stacked,
    const __half* __restrict__ input,
    __half* __restrict__ output,
    uint32_t layer_lo, uint32_t layer_hi
) {
    long long layer = decode_i64(layer_lo, layer_hi);
    q8_gemv_core<1024, 1024>(q8_stacked + layer * 1114112LL, input, output);
}

// attn_v: same as attn_k (reuses q8_gemv_1024x1024)

// attn_o: [1024, 2048] — 64 Q8 blocks/row, 2 per lane
extern "C" __global__ __launch_bounds__(256)
void q8_gemv_1024x2048(
    const uint8_t* __restrict__ q8_stacked,
    const __half* __restrict__ input,
    __half* __restrict__ output,
    uint32_t layer_lo, uint32_t layer_hi
) {
    long long layer = decode_i64(layer_lo, layer_hi);
    q8_gemv_core<1024, 2048>(q8_stacked + layer * 2228224LL, input, output);
}

// gate_up: [6144, 1024] — 32 Q8 blocks/row, 1 per lane
extern "C" __global__ __launch_bounds__(256)
void q8_gemv_6144x1024(
    const uint8_t* __restrict__ q8_stacked,
    const __half* __restrict__ input,
    __half* __restrict__ output,
    uint32_t layer_lo, uint32_t layer_hi
) {
    long long layer = decode_i64(layer_lo, layer_hi);
    q8_gemv_core<6144, 1024>(q8_stacked + layer * 6684672LL, input, output);
}

// ffn_down: [1024, 3072] — 96 Q8 blocks/row, 3 per lane
extern "C" __global__ __launch_bounds__(256)
void q8_gemv_1024x3072(
    const uint8_t* __restrict__ q8_stacked,
    const __half* __restrict__ input,
    __half* __restrict__ output,
    uint32_t layer_lo, uint32_t layer_hi
) {
    long long layer = decode_i64(layer_lo, layer_hi);
    q8_gemv_core<1024, 3072>(q8_stacked + layer * 3342336LL, input, output);
}

// vocab (f16, not Q8): [151936, 1024] @ [1024, 1]
// Output weight is f16 (not quantized), stored transposed as [151936, 1024]
extern "C" __global__ __launch_bounds__(256)
void f16_gemv_151936x1024(
    const __half* __restrict__ weight,   // [151936, 1024] f16
    const __half* __restrict__ input,    // [1024]
    __half* __restrict__ output          // [151936]
) {
    const int N = 151936, K = 1024;

    __shared__ __half shmem[K];
    for (int i = threadIdx.x; i < K; i += NTHREADS)
        shmem[i] = input[i];
    __syncthreads();

    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= N) return;

    const __half* row_data = weight + (long long)row * K;
    float acc = 0.0f;

    // Each lane handles K/32 = 32 elements
    for (int k = lane; k < K; k += WARP_SIZE) {
        acc += __half2float(row_data[k]) * __half2float(shmem[k]);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        output[row] = __float2half(acc);
}
