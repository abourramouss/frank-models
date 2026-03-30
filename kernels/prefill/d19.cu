#include "gemm_core.cuh"
extern "C" __global__ __launch_bounds__(128)
void prefill_dispatch_19_matmul_Dx1024x3072_f16(
    const __half* w, const char* b1, const char* b2, __half* b3,
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t c4, uint32_t c5, uint32_t c6, uint32_t c7) {
    gemm_core<1024, 3072>(
        (const __half*)(b1 + decode_i64(c0,c1)),
        w + decode_i64(c4,c5) * (3072LL*1024),
        b3,
        (const __half*)(b2 + decode_i64(c2,c3)),
        (int)decode_i64(c6,c7));
}
