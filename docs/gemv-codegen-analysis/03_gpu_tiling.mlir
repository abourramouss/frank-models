// -----// IR Dump After GPUApplyTilingLevelPass (iree-codegen-gpu-apply-tiling-level) //-----
 //
func.func @gemv_f16_dispatch_0_matvec_like_2048x1024_f16() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [32, 1, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024xf16, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<2048xf16, #hal.descriptor_type<storage_buffer>>
  %3 = iree_codegen.load_from_buffer %0 : memref<1024xf16, #hal.descriptor_type<storage_buffer>> -> tensor<1024xf16>
  %4 = iree_codegen.load_from_buffer %1 : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>> -> tensor<1024x2048xf16>
  %5 = tensor.empty() : tensor<2048xf16>
  %6 = scf.forall (%arg0) in (2048) shared_outs(%arg1 = %5) -> (tensor<2048xf16>) {
    %7 = tensor.empty() : tensor<1xf16>
    %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<1xf16>) -> tensor<1xf16>
    %9 = tensor.empty() : tensor<1x32xf16>
    %10 = linalg.fill ins(%cst : f16) outs(%9 : tensor<1x32xf16>) -> tensor<1x32xf16>
    %11 = scf.for %arg2 = %c0 to %c128 step %c32 iter_args(%arg3 = %10) -> (tensor<1x32xf16>) {
      %12 = affine.linearize_index disjoint [%arg2, %c0] by (128, 8) : index
      %extracted_slice = tensor.extract_slice %3[%12] [256] [1] : tensor<1024xf16> to tensor<256xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [32, 8] : tensor<256xf16> into tensor<32x8xf16>
      %13 = tensor.empty() : tensor<32x8xf16>
      %14 = linalg.copy ins(%expanded : tensor<32x8xf16>) outs(%13 : tensor<32x8xf16>) -> tensor<32x8xf16>
      %extracted_slice_0 = tensor.extract_slice %4[%12, %arg0] [256, 1] [1, 1] : tensor<1024x2048xf16> to tensor<256x1xf16>
      %expanded_1 = tensor.expand_shape %extracted_slice_0 [[0, 1], [2]] output_shape [32, 8, 1] : tensor<256x1xf16> into tensor<32x8x1xf16>
      %15 = tensor.empty() : tensor<32x8x1xf16>
      %16 = linalg.copy ins(%expanded_1 : tensor<32x8x1xf16>) outs(%15 : tensor<32x8x1xf16>) -> tensor<32x8x1xf16>
      %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14, %16 : tensor<32x8xf16>, tensor<32x8x1xf16>) outs(%arg3 : tensor<1x32xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{expand_dims = #iree_gpu.expand_dims<[[0], [1, 2]], output_shape = [?, ?, 8]>, lane_basis = [[1, 32, 1], [0, 1, 2]], partial_reduction = [0, 32, 0], subgroup_basis = [[1, 1, 1], [0, 1, 2]], thread = [0, 1, 8], workgroup = [1, 0, 0]}>} {
      ^bb0(%in: f16, %in_2: f16, %out: f16):
        %18 = arith.mulf %in, %in_2 : f16
        %19 = arith.addf %out, %18 : f16
        linalg.yield %19 : f16
      } -> tensor<1x32xf16>
      scf.yield %17 : tensor<1x32xf16>
    }
    %reduced = linalg.reduce ins(%11 : tensor<1x32xf16>) outs(%8 : tensor<1xf16>) dimensions = [1] 
      (%in: f16, %init: f16) {
        %12 = arith.addf %in, %init : f16
        linalg.yield %12 : f16
      }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %reduced into %arg1[%arg0] [1] [1] : tensor<1xf16> into tensor<2048xf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %6, %2 : tensor<2048xf16> into memref<2048xf16, #hal.descriptor_type<storage_buffer>>
  return
}

