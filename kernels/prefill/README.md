# WMMA Tensor Core GEMM Kernels for Qwen3-0.6B Prefill

Custom CUDA WMMA (Tensor Core) kernels replacing IREE's default codegen for
prefill matmul dispatches. Integrated via `--iree-hal-substitute-executable-source`.

## Why

IREE's codegen picks `LLVMGPUVectorDistribute` for dynamic-M matmuls, generating
**1 output element per block** — catastrophically bad for prefill GEMM. With
static M, IREE correctly picks `LLVMGPUTileAndFuse` with MMA tensor cores,
but our prefill has dynamic `seq_len`.

These kernels use `nvcuda::wmma::mma_sync` (16x16x16 f16 tensor core fragments)
with 32x32 output tiling and 128 threads (4 warps, 2x2 layout).

## Results (Jetson AGX Orin, sm_87)

GPU kernel time for all prefill matmuls across 28 layers (seq_len=9):

| Dispatch | Operation | Baseline | WMMA TC | Speedup |
|----------|-----------|----------|---------|---------|
| 3  | attn_q [M,1024]@[1024,2048] | 178ms | 6.8ms | 26x |
| 4  | attn_kv [M,1024]@[1024,1024] | 63ms | 11.3ms | 5.6x |
| 13 | attn_o [M,2048]@[2048,1024]+res | 101ms | 11.2ms | 9x |
| 17 | gate_up [M,1024]@[1024,6144] | 529ms | 12.4ms | 43x |
| 19 | ffn_down [M,3072]@[3072,1024]+res | 147ms | 18.3ms | 8x |
| **Total** | | **1018ms** | **60ms** | **17x** |

Note: First-invocation module loading (~4s) masks the speedup.
In a persistent server, prefill goes from ~1s to ~60ms.

## Files

- `gemm_core.cuh` — Shared WMMA GEMM template (32x32 tiling, 4 warps)
- `d{3,4,13,17,19}.cu` — Per-dispatch kernel with IREE interface decoding
- `dispatch_{3,4,13,17,19}.cubin` — Pre-compiled sm_87 SASS binaries
- `prefill_dispatch_{3,4,13,17,19}.mlir` — IREE hal.executable wrappers

## Compile

```bash
# Build all cubins
for n in 3 4 13 17 19; do
  nvcc -arch=sm_87 -cubin d${n}.cu -o dispatch_${n}.cubin
done

# Compile model with substitutions
iree-compile model.mlir \
  --iree-hal-executable-object-search-path=kernels/prefill/ \
  --iree-hal-substitute-executable-sources-from=kernels/prefill/ \
  -o output.vmfb
```
