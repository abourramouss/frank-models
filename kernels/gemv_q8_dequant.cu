// Custom GEMV kernel: Q8_0 dequant + matvec fused
// Replaces IREE's generated matvec_like dispatches for batch=1 decode.
//
// Q8_0 format: blocks of 32 values, each block = 2 bytes f16 scale + 32 bytes i8 values
// Weight layout: [K, N] stored as Q8_0 blocks row-major (K*N/32 blocks of 34 bytes)
//
// Compute: output[n] = sum_k( dequant(weight[k,n]) * input[k] ) for all n
//
// Strategy:
//   - 256 threads per block, 4 output elements per block
//   - Load input vector into shared memory (reuse across 4 outputs)
//   - Each group of 64 threads reduces over K dimension for one output
//   - Vectorized Q8_0 block reads

#include <cuda_fp16.h>

extern "C" {

// Generic Q8 dequant + GEMV: output[N] = Q8_weights[K, N] @ input[K]
// weight_q8: raw Q8_0 bytes, layout [K, N] with K*N/32 blocks of 34 bytes
// input: f16[K]
// output: f16[N]
// K, N: dimensions
__global__ void gemv_q8_fused(
    const int8_t* __restrict__ weight_q8,
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int K, int N
) {
    // 4 outputs per block, 64 threads per output
    const int OUTPUTS_PER_BLOCK = 4;
    const int THREADS_PER_OUTPUT = 64;

    int block_output_base = blockIdx.x * OUTPUTS_PER_BLOCK;
    int local_output = threadIdx.x / THREADS_PER_OUTPUT;  // 0..3
    int lane = threadIdx.x % THREADS_PER_OUTPUT;           // 0..63
    int n = block_output_base + local_output;

    if (n >= N) return;

    // Load input vector into shared memory
    __shared__ __half shared_input[4096];  // max K=4096
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    // Each thread accumulates partial sum over a chunk of K
    float acc = 0.0f;

    // Total elements per output row: K
    // Total Q8 blocks per row: K / 32
    int blocks_per_row = K / 32;
    int total_elements = K;  // = blocks_per_row * 32

    // Each of 64 threads handles K/64 elements
    // Process in Q8 block granularity (32 elements per block)
    int blocks_per_thread = (blocks_per_row + THREADS_PER_OUTPUT - 1) / THREADS_PER_OUTPUT;

    for (int bi = 0; bi < blocks_per_thread; bi++) {
        int block_idx_in_row = lane + bi * THREADS_PER_OUTPUT;
        if (block_idx_in_row >= blocks_per_row) break;

        // Global linear block index: row n has blocks_per_row blocks
        // But weight is stored as [K, N] = K rows of N elements
        // Wait - we need to think about the layout carefully.
        //
        // Weight is [K, N] in Q8_0. The Q8_0 encoding blocks along the
        // flattened K*N dimension. Block b corresponds to elements
        // [b*32 .. b*32+31] in the flat array.
        // Flat index for (k, n) = k * N + n
        //
        // For a single output n, we need k=0..K-1:
        //   flat indices: 0*N+n, 1*N+n, 2*N+n, ..., (K-1)*N+n
        //   These are NOT contiguous - they're strided by N!
        //
        // Q8 blocks don't align to single-column access patterns.
        // We'd need to read individual elements from different blocks.
        //
        // This is the fundamental issue: Q8_0 blocks are row-major,
        // but GEMV needs column access. Each Q8 block contains 32
        // consecutive elements from ONE row, but we need one element
        // from each of K rows.
        //
        // For column access, we must do scalar Q8 dequant:
        int k_base = block_idx_in_row * 32;

        for (int ki = 0; ki < 32 && k_base + ki < K; ki++) {
            int k = k_base + ki;
            // Flat index in the weight matrix
            int flat = k * N + n;
            int qblock = flat / 32;
            int qelem = flat % 32;
            int byte_offset = qblock * 34;

            // Read scale (2 bytes at block start)
            uint8_t s0 = ((const uint8_t*)weight_q8)[byte_offset];
            uint8_t s1 = ((const uint8_t*)weight_q8)[byte_offset + 1];
            uint16_t scale_bits = s0 | (s1 << 8);
            float scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));

            // Read quantized value
            int8_t qval = weight_q8[byte_offset + 2 + qelem];
            float w = scale * (float)qval;

            float x = __half2float(shared_input[k]);
            acc += w * x;
        }
    }

    // Warp-level reduction (64 threads → need 2 warps to cooperate)
    // First reduce within each warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float warp_results[8];  // max 8 warps
    int warp_id = threadIdx.x / 32;
    int lane_in_warp = threadIdx.x % 32;

    if (lane_in_warp == 0) {
        warp_results[warp_id] = acc;
    }
    __syncthreads();

    // Thread 0 of each output group does final reduction
    if (lane == 0) {
        float final_acc = warp_results[local_output * 2] + warp_results[local_output * 2 + 1];
        output[n] = __float2half(final_acc);
    }
}

}  // extern "C"
