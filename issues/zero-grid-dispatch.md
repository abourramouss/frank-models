# CUDA crash on zero-workgroup dispatch

## Status: FIXED (compiler canonicalization + runtime defense)

## Summary

`cuLaunchKernel` rejects `grid=(0,...)` with `CUDA_ERROR_INVALID_VALUE`. This happens when dynamic shapes produce zero-sized outputs — e.g. KV cache gather with `max_context_len=0` on the first decode step (no cached context yet).

## Root cause

The compiler correctly computes `grid=(0,1,1)` for a dispatch over a zero-length dimension. But CUDA's API treats zero grid size as an error rather than a no-op. On CPU backends this is fine (zero-iteration loops), but CUDA crashes.

## Fix (commit 35974a9d62)

Two-level fix:

### 1. Compiler: `ElideZeroWorkgroupDispatch` canonicalization (proper fix)

Added to `HALOpFolders.cpp` — a canonicalization pattern for `hal.command_buffer.dispatch` that erases the op when any workgroup dimension is a known zero constant. This is the proper fix: zero-workgroup dispatches are no-ops by definition (no kernel invocations, no side effects, no results to replace). The pattern runs during standard canonicalization, so it catches all statically-known zero dispatches before they reach the runtime.

This is correct and complete for static cases. A dispatch with `workgroups([%c0, %c1, %c1])` is simply erased.

### 2. Runtime: skip in `stream_command_buffer.c` (defense-in-depth)

For dynamically-computed workgroup counts that are zero at runtime but not provably zero at compile time, the CUDA HAL skips the `cuLaunchKernel` call instead of crashing. This is a defense-in-depth measure, not a workaround — the compiler fix handles the common case, and the runtime guard handles the edge case of dynamic zeros.

## Reproduction

The KV cache gather with `max_context_len=0` is the primary trigger. This happens on the first decode step of autoregressive inference when no tokens have been cached yet.

```bash
# Without fix: CUDA_ERROR_INVALID_VALUE in cuLaunchKernel
# With fix: dispatch elided at compile time, or skipped at runtime
python3 scripts/chat_qwen3.py --stream  # works on CUDA
```

## Files

- Compiler fix: `iree/compiler/src/iree/compiler/Dialect/HAL/IR/HALOpFolders.cpp`
- Runtime guard: `iree/runtime/src/iree/hal/drivers/cuda/stream_command_buffer.c`
