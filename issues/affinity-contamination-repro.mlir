// Minimal reproduction: shared function affinity contamination in multi-device compilation.
//
// When a shared utility function (normalize) is called from two public functions
// with different device affinities, the multi-device compiler propagates the
// CPU affinity into the GPU-only function, causing CUDA_ERROR_ILLEGAL_ADDRESS.
//
// Compile (multi-device, triggers the bug):
//   iree-compile repro.mlir \
//     --iree-hal-target-device=device_gpu=cuda[0] \
//     --iree-hal-target-device=device_cpu=local[0] \
//     --iree-hal-default-device=device_gpu \
//     --iree-hal-local-target-device-backends=llvm-cpu \
//     --iree-cuda-target=sm_87 --iree-cuda-target-features=+ptx74 \
//     -o repro_multi.vmfb
//
// Compile (single-device, works correctly):
//   iree-compile repro.mlir \
//     --iree-hal-target-backends=cuda \
//     --iree-cuda-target=sm_87 --iree-cuda-target-features=+ptx74 \
//     -o repro_single.vmfb
//
// The single-device compile fails because @device_cpu is undeclared.
// This is expected — the point is that the multi-device compile produces
// a VMFB where @compute_gpu generates CUDA_ERROR_ILLEGAL_ADDRESS after
// 2-3 invocations on Jetson AGX Orin (unified memory).
//
// Root cause: the compiler's affinity analysis propagates @device_cpu
// affinity from @compute_hetero into the shared @normalize function,
// which then contaminates @compute_gpu's code generation.

module @affinity_contamination_repro attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_gpu -> @device_cpu = {transparent_access = true, unified_memory = true}),
    (@device_cpu -> @device_gpu = {transparent_access = true, unified_memory = true})
  ]>
} {

  // Shared utility function — called by BOTH compute_gpu and compute_hetero.
  // This is the function that gets "contaminated" with CPU affinity.
  util.func private @normalize(
      %input: tensor<?x?xf32>,
      %weight: tensor<?xf32>
  ) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim0 = tensor.dim %input, %c0 : tensor<?x?xf32>
    %dim1 = tensor.dim %input, %c1 : tensor<?x?xf32>

    // Sum of squares (reduction)
    %init_sum = tensor.empty(%dim0) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero : f32) outs(%init_sum : tensor<?xf32>) -> tensor<?xf32>
    %sum_sq = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%input : tensor<?x?xf32>) outs(%sum_init : tensor<?xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sq = arith.mulf %in, %in : f32
      %sum = arith.addf %acc, %sq : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>

    // Normalize and scale
    %eps = arith.constant 1.0e-5 : f32
    %output_init = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %sum_sq, %weight : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
      outs(%output_init : tensor<?x?xf32>) {
    ^bb0(%x: f32, %ss: f32, %w: f32, %out: f32):
      %rms = math.sqrt %ss : f32
      %rms_eps = arith.addf %rms, %eps : f32
      %norm = arith.divf %x, %rms_eps : f32
      %scaled = arith.mulf %norm, %w : f32
      linalg.yield %scaled : f32
    } -> tensor<?x?xf32>

    util.return %output : tensor<?x?xf32>
  }

  // ========================================================================
  // GPU-only function: normalize + matmul, entirely on GPU.
  // BUG: after multi-device compilation, this function gets contaminated
  // with @device_cpu affinity from compute_hetero via shared @normalize.
  // ========================================================================
  util.func public @compute_gpu(
      %input: tensor<?x?xf32>,
      %weight_norm: tensor<?xf32>,
      %weight_proj: tensor<?x?xf32>
  ) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Normalize (on GPU — but gets contaminated)
    %normed = util.call @normalize(%input, %weight_norm)
        : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>

    // Matmul projection
    %batch = tensor.dim %input, %c0 : tensor<?x?xf32>
    %out_dim = tensor.dim %weight_proj, %c1 : tensor<?x?xf32>
    %proj_init = tensor.empty(%batch, %out_dim) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %proj_fill = linalg.fill ins(%zero : f32) outs(%proj_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %result = linalg.matmul ins(%normed, %weight_proj : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%proj_fill : tensor<?x?xf32>) -> tensor<?x?xf32>

    util.return %result : tensor<?x?xf32>
  }

  // ========================================================================
  // Heterogeneous function: GPU matmul, then transfer to CPU for normalize.
  // This is the function that introduces @device_cpu affinity into @normalize.
  // ========================================================================
  util.func public @compute_hetero(
      %input: tensor<?x?xf32>,
      %weight_norm: tensor<?xf32>,
      %weight_proj: tensor<?x?xf32>
  ) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Matmul on GPU
    %batch = tensor.dim %input, %c0 : tensor<?x?xf32>
    %out_dim = tensor.dim %weight_proj, %c1 : tensor<?x?xf32>
    %proj_init = tensor.empty(%batch, %out_dim) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %proj_fill = linalg.fill ins(%zero : f32) outs(%proj_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %matmul_result = linalg.matmul ins(%input, %weight_proj : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%proj_fill : tensor<?x?xf32>) -> tensor<?x?xf32>

    // Transfer to CPU
    %on_cpu = flow.tensor.transfer %matmul_result : tensor<?x?xf32>{%batch, %out_dim}
        to #hal.device.promise<@device_cpu>

    // Normalize on CPU — THIS contaminates @normalize with CPU affinity
    %wn_dim = tensor.dim %weight_norm, %c0 : tensor<?xf32>
    %wn_cpu = flow.tensor.transfer %weight_norm : tensor<?xf32>{%wn_dim}
        to #hal.device.promise<@device_cpu>
    %normed = util.call @normalize(%on_cpu, %wn_cpu)
        : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>

    // Transfer back to GPU
    %result = flow.tensor.transfer %normed : tensor<?x?xf32>{%batch, %out_dim}
        to #hal.device.promise<@device_gpu>

    util.return %result : tensor<?x?xf32>
  }

}
