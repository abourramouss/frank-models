# Shared function affinity contamination in multi-device compilation

## Summary

When a utility function is called from two public entry points with different device affinities, the compiler propagates the "wrong" affinity to callers that should stay on a single device. This causes either:
- Runtime: `HAL device does not support executable format` (CPU device gets CUDA executable)
- Jetson unified memory: `CUDA_ERROR_ILLEGAL_ADDRESS` (GPU kernel accesses CPU-allocated buffer)

## Reproduction

```mlir
// repro.mlir
module @repro attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_gpu -> @device_cpu = {unified_memory = true}),
    (@device_cpu -> @device_gpu = {unified_memory = true})
  ]>
} {

  // Shared utility function
  util.func private @normalize(%input: tensor<?x?xf32>, %weight: tensor<?xf32>) -> tensor<?x?xf32> {
    // ... RMS normalization ...
  }

  // GPU-only: calls @normalize (should stay entirely on GPU)
  util.func public @compute_gpu(%input: tensor<?x?xf32>, ...) -> tensor<?x?xf32> {
    %normed = util.call @normalize(%input, %weight) ...  // <-- gets contaminated
    %result = linalg.matmul ...
    util.return %result
  }

  // Hetero: calls @normalize on CPU via flow.tensor.transfer
  util.func public @compute_hetero(%input: tensor<?x?xf32>, ...) -> tensor<?x?xf32> {
    %on_cpu = flow.tensor.transfer %gpu_result to #hal.device.promise<@device_cpu>
    %normed = util.call @normalize(%on_cpu, %wn_cpu) ...  // <-- introduces CPU affinity
    util.return ...
  }
}
```

Compile:
```bash
iree-compile repro.mlir \
  --iree-hal-target-device=device_gpu=cuda[0] \
  --iree-hal-target-device=device_cpu=local[0] \
  --iree-hal-default-device=device_gpu \
  --iree-hal-local-target-device-backends=llvm-cpu \
  -o repro.vmfb
```

Runtime error:
```
UNAVAILABLE; HAL device `device_cpu` does not support any variant of
executable `repro_linked`; available formats: [cuda-nvptx-fb]
```

## Root Cause

The compiler's affinity analysis (likely in `AnnotateAffinities` or during inlining) propagates `@device_cpu` affinity from `compute_hetero`'s call to `@normalize` into the shared function body. Since `compute_gpu` also calls `@normalize`, the contaminated affinity affects `compute_gpu`'s code generation.

The expected behavior: `@normalize` should be **cloned per-affinity** — one copy compiled for CUDA (used by `compute_gpu`), one for llvm-cpu (used by `compute_hetero`).

## Impact

This blocks single-VMFB heterogeneous execution on Jetson AGX Orin. We can work around it with two separate VMFBs (GPU and CPU compiled independently), but that doubles HAL overhead and prevents the compiler from optimizing cross-device transfers.

## Context

- Platform: Jetson AGX Orin (unified memory, CUDA 11.4, sm_87)
- Use case: OLMoE-1B-7B inference with GPU transformer layers + CPU output post-processing
- The `ElideAsyncTransfersPass` correctly elides transfers on `unified_memory` topologies, but the affinity contamination happens before elision
- Related: #20851 (heterogeneous MVP), #20853 (elide transfers)

## Full repro file

See `affinity-contamination-repro.mlir` in this directory.
