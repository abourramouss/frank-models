# Q8_0 Fused Dequant+GEMV Kernels

Custom CUDA kernels that fuse Q8_0 dequantization with matrix-vector multiply.
Reads Q8_0 raw bytes directly, avoiding the separate dequant→f16→GEMV pipeline
that doubles memory traffic.

## Kernel Design
- 256 threads = 8 warps, 8 rows per block
- 1 warp per row: each lane handles 1+ Q8_0 blocks (32 elements)
- Shared memory for input vector
- Warp shuffle reduction for dot product
- f32 accumulation → f16 output

## Integration
Requires MLIR model change: wrap Q8_0 dequant + GEMV in `flow.dispatch.region`
to force single-dispatch fusion, then substitute with these kernels.

## Files
- `gemv_q8_fused.cu` — All 6 CUDA kernels (5 Q8_0 + 1 f16 vocab)
- `gemv_q8_fused.ptx` — Compiled for sm_87

## Status
- [x] CUDA kernels written and compiled
- [ ] MLIR model modified with flow.dispatch.region fusion
- [ ] MLIR wrappers created for executable substitution
- [ ] End-to-end testing
