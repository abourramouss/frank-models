# Qwen3-0.6B Decode Optimization: 2.5 → 25 tok/s (10x)

**Platform:** Jetson AGX Orin (sm_87, 16 SMs, 64GB unified memory)
**Model:** Qwen3-0.6B, 28 layers, f16 weights, Q8_0 quantized matmul weights
**Compiler:** IREE (custom aarch64 build)
**Baseline comparison:** llama.cpp cuBLAS = 49 tok/s (FP32), 113 tok/s (Q8_0)

## The killer optimization: Transposed GEMV

IREE's GPU codegen for `[1, K] @ [K, N]` (M=1 matmul) is catastrophically bad.
It generates **one CUDA block per output element** with **32 threads** (one warp),
achieving 1.6% SM occupancy. For the vocab projection `[1, 1024] @ [1024, 151936]`,
this means 151,936 blocks of 32 threads — each independently loading the full
input vector from DRAM.

The fix: **transpose the matmul** to `[N, K] @ [K, 1] → [N, 1]`. Now the M
dimension is N (large), and IREE tiles it across workgroups properly with
multiple outputs per block. Same math, completely different codegen.

**Impact per matmul size:**

| Matmul | Original | Transposed | Speedup |
|--------|----------|------------|---------|
| vocab [151936, 1024] | 58.0 ms | 2.7 ms | **21x** |
| gate_up [6144, 1024] | 2.68 ms | 0.81 ms | **3.3x** |
| attn_q [2048, 1024] | 1.26 ms | 0.76 ms | 1.7x |
| ffn_down [1024, 3072] | 1.25 ms | 0.79 ms | 1.6x |
| attn_k/v [1024, 1024] | 1.08 ms | 0.86 ms | 1.3x |

The vocab projection alone was **48% of all GPU time** at 58ms/token. Transposing
it to 2.7ms saved 55ms per token — more than all other optimizations combined.

### Why IREE generates bad GEMV

From the codegen analysis (`docs/gemv-codegen-analysis/`):

`LLVMGPUSelectLoweringStrategyPass` picks `LLVMGPUVectorDistribute` with:
```
workgroup_size = [32, 1, 1]       // 1 warp, 1.6% occupancy
workgroup = [1, 0, 0]             // 1 output per block
thread = [0, 1, 8]                // 8 elements per thread
partial_reduction = [0, 32, 0]    // warp shuffle reduction
```

The heuristic treats M=1 matmul as a **reduction problem** (each output is an
independent dot product), assigning one warp per output. It doesn't recognize
that GEMV has massive **input reuse** across outputs — the same input vector
is needed by every output element.

cuBLAS handles this with 256 threads per block, shared memory for the input
vector, vectorized 128-bit loads, and multiple outputs per block.

The transposed approach sidesteps the bad heuristic entirely: with M=N (large)
and N=1, IREE's standard matmul tiling distributes M across workgroups, which
is exactly what we want.

### Weight storage

The transposed matmuls use weights in `[N, K]` layout, which is the GGUF native
storage order. This means:
- No transpose needed in the GGUF → IRPA conversion for Q8_0 weights
- Raw Q8_0 blocks are stored contiguously along rows, maximizing cache efficiency
  when reading a full row for one output element

## All optimizations (cumulative)

| # | Change | tok/s | Speedup | What it fixed |
|---|--------|-------|---------|---------------|
| 0 | Original (paged cache, O0, f16) | 2.5 | 1.0x | Baseline |
| 1 | O1+ compilation fix | 2.6 | 1.04x | `tensor.concat` → `tensor.insert_slice` to avoid illegal `tensor.cast` in canonicalization |
| 2 | Direct KV cache | 2.7 | 1.08x | Replaced paged block_tables with flat contiguous cache. Eliminated gather/scatter indirection |
| 3 | Static shapes + O3 | 3.2 | 1.28x | All known dims as compile-time constants. Only seq_len dynamic. Eliminated `tensor.dim` dispatches |
| 4 | Tensor-threaded layer cache | 4.4 | 1.76x | K/V as `tensor` iter_args through the 28-layer `scf.for`, not `!util.list<?>`. One import/export per token instead of per layer |
| 5 | Q8_0 runtime dequant | 6.7 | 2.68x | Weights stored as Q8_0 raw bytes, dequantized to f16 in `linalg.generic` before matmul. ~1.4x smaller params |
| 6 | Transposed vocab projection | ~9 | 3.6x | `[151936,1024]@[1024,1]` instead of `[1,1024]@[1024,151936]`. 21x faster for vocab matmul alone |
| 7 | Tiled argmax | ~10 | 4.0x | 2-phase parallel reduction: [594×256] chunk-local, then 594 final. Replaced single-thread 151K scan |
| 8 | **Transposed ALL matmuls** | **18** | **7.2x** | Same trick applied to Q/K/V/O projections and FFN gate_up/down. 1.3-3.3x per matmul |
| 9 | Tensor-loop generate | **25** | **10x** | Position as GPU tensor + CPU index counter in generate loop. Reduced per-token `cuStreamSynchronize` |

## Remaining gap to llama.cpp

| Metric | IREE (ours) | llama.cpp cuBLAS |
|--------|-------------|------------------|
| tok/s | 25 | 49 |
| GPU compute/tok | ~12ms | ~18ms |
| CPU overhead/tok | ~28ms | ~2ms |
| Syncs/tok | ~54 | ~2 |

**GPU compute is competitive** (12ms vs 18ms — IREE is actually faster because
of f16 vs llama.cpp's f32). The remaining 2x gap is entirely **IREE runtime
overhead**: `cuStreamSynchronize` calls from the VM bytecode interpreter,
`cuMemAllocFromPoolAsync` for temporary buffers, and `hal.tensor.import/export`
for the KV cache buffer views.

### What would close the gap

1. **Global KV cache tensors** — replace `!util.list<?>` with `util.global` mutable
   tensors to eliminate `hal.tensor.import/export` syncs (attempted, needs debugging)
2. **CUDA graph capture** — replay the decode loop as a graph without per-dispatch sync
3. **IREE runtime fix** — reduce `cuStreamSynchronize` frequency in loops with
   scalar readback (`tensor.extract`)

## Profiling methodology

All profiling done with `nsys profile` (NVIDIA Nsight Systems 2023.2.4).
Key metrics: `cuda_api_sum` for host-side overhead, `cuda_gpu_kern_sum` for
kernel execution times. IREE's built-in `DISPATCH TIMING` reports dispatch
launch overhead (consistently ~6μs/dispatch, not a bottleneck).

## Files

- `architectures/llm/qwen3/qwen3_full.mlir` — optimized model (current best)
- `architectures/llm/qwen3/qwen3_full_paged.mlir` — original paged cache version
- `architectures/llm/qwen3/qwen3_full_direct_dynamic.mlir` — direct cache, dynamic shapes
- `scripts/create_q8_irpa.py` — GGUF Q8_0 → stacked IRPA converter
- `scripts/convert_gguf_to_irpa.py` — GGUF FP32 → stacked f16 IRPA converter
- `docs/gemv-codegen-analysis/` — full IR dump at every compilation phase for GEMV
- `kernels/gemv_q8_dequant.cu` — custom CUDA GEMV kernel (prototype, not integrated)
