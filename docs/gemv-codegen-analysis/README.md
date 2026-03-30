# GEMV Codegen Analysis: why IREE is 65x slower than cuBLAS

**Benchmark:** `[1, 1024] × [1024, 2048]` f16 GEMV on Jetson AGX Orin (sm_87)
- IREE: **1.29ms**
- cuBLAS (via llama.cpp): **~0.02ms**
- Ratio: **~65x**

## Files (compilation pipeline order)

| File | Pass | What to look for |
|------|------|-----------------|
| `00_input.mlir` | Input | `linalg.matmul` M=1 |
| `00_dispatch.mlir` | After dispatch formation | Collapsed to `matvec_like` generic |
| `01_lowering_strategy.mlir` | **LLVMGPUSelectLoweringStrategy** | **The critical decision:** `workgroup_size=[32,1,1]`, `workgroup=[1,0,0]` — 1 output per block, 32 threads |
| `02_tile_distribute.mlir` | Tile & distribute | `scf.forall in (2048)` — 2048 workgroups, each computes 1 output |
| `03_gpu_tiling.mlir` | GPU tiling levels | Reduction tiling: 256 elements per iteration |
| `04_vectorization.mlir` | Vectorization | `vector.contract <32x8> × <32x8x1> → <1x32>` — warp-level reduction |
| `05_vector_distribute.mlir` | Vector distribute | Maps vector ops to GPU threads |
| `06_vector_lowering.mlir` | Vector lowering | Scalarized to individual f16 ops |
| `07_nvvm.mlir` | NVVM dialect | `nvvm.shfl.sync` for warp reduction |
| `08_final.ptx` | **Generated PTX** | `.maxntid 32,1,1`, scalar `ld.global.nc.b16`, f16→f32→f16 per FMA |

## Root cause

`LLVMGPUSelectLoweringStrategyPass` (file 01) picks `LLVMGPUVectorDistribute` with:

```
workgroup_size = [32, 1, 1]   ← only 1 warp, 3% occupancy
workgroup = [1, 0, 0]         ← 1 output element per block
thread = [0, 1, 8]            ← each thread does 8-element chunks
partial_reduction = [0, 32, 0] ← warp shuffle reduction
```

This means:
1. **2048 blocks × 32 threads** = terrible occupancy (32/1024 = 3%)
2. **Input vector reloaded by every block** — no shared memory
3. **Scalar f16 loads** from weight matrix — should be vectorized 128-bit
4. **f16→f32→f16 round-trip per FMA** instead of f32 accumulation

## What cuBLAS does

- 256+ threads per block, 4-8 outputs per block
- Input vector in shared memory (load once, reuse)
- Vectorized 128-bit loads (`float4`) from weights
- f32 accumulation, single conversion at the end

## Fix options

1. **Override lowering config** with `#iree_codegen.compilation_info` annotation on the matmul — set larger workgroups, multiple outputs per block
2. **Custom CUDA kernel** via `hal.executable.export` — write a proper GEMV kernel
3. **cuBLAS** via IREE external dispatch mechanism
4. **File an IREE issue** — the M=1 matmul/matvec heuristic in `LLVMGPUSelectLoweringStrategy` is clearly suboptimal for sm_87

## Optimization progress so far

| Change | tok/s | vs original |
|--------|-------|-------------|
| Original (paged, O0, f16) | 2.5 | 1.0x |
| + O1 fix (concat→insert_slice) | 2.6 | 1.04x |
| + Direct KV cache | 2.7 | 1.08x |
| + Static shapes, O3 | 3.2 | 1.28x |
| + Tensor-threaded cache | 4.4 | 1.76x |
| + Q8_0 runtime dequant | **6.1** | **2.44x** |
| llama.cpp FP32 cuBLAS | 49.4 | 19.8x |
| llama.cpp Q8_0 cuBLAS | 113.2 | 45.3x |
