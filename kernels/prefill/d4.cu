#include "gemm_core.cuh"
extern "C" __global__ __launch_bounds__(128)
void prefill_dispatch_4_matmul_Dx1024x1024_f16(
    const __half* w, const char* b1, char* b2,
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t c4, uint32_t c5, uint32_t c6, uint32_t c7) {
    gemm_core<1024, 1024>(
        (const __half*)(b1 + decode_i64(c0,c1)),
        w + decode_i64(c4,c5) * (1024LL*1024),
        (__half*)(b2 + decode_i64(c2,c3)),
        nullptr, (int)decode_i64(c6,c7));
}
