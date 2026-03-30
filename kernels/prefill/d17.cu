#include "gemm_core.cuh"
extern "C" __global__ __launch_bounds__(128)
void prefill_dispatch_17_matmul_Dx6144x1024_f16(
    const __half* w, const char* b1, char* b2,
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t c4, uint32_t c5) {
    gemm_core<6144, 1024>(
        (const __half*)b1,
        w + decode_i64(c2,c3) * (1024LL*6144),
        (__half*)(b2 + decode_i64(c0,c1)),
        nullptr, (int)decode_i64(c4,c5));
}
