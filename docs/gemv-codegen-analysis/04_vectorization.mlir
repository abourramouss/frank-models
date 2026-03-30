// -----// IR Dump After GenericVectorizationPass (iree-codegen-generic-vectorization) //-----
 //
func.func @gemv_f16_dispatch_0_matvec_like_2048x1024_f16() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [32, 1, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant dense<0.000000e+00> : vector<1x32xf16>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
  %0 = ub.poison : f16
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024xf16, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<2048xf16, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %1 : memref<1024xf16, #hal.descriptor_type<storage_buffer>> -> tensor<1024xf16>
  %5 = iree_codegen.load_from_buffer %2 : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>> -> tensor<1024x2048xf16>
  %6 = tensor.empty() : tensor<2048xf16>
  %7 = scf.forall (%arg0) in (2048) shared_outs(%arg1 = %6) -> (tensor<2048xf16>) {
    %8 = tensor.empty() : tensor<1xf16>
    %9 = vector.transfer_write %cst_0, %8[%c0] {in_bounds = [true]} : vector<1xf16>, tensor<1xf16>
    %10 = tensor.empty() : tensor<1x32xf16>
    %11 = vector.transfer_write %cst, %10[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf16>, tensor<1x32xf16>
    %12 = scf.for %arg2 = %c0 to %c128 step %c32 iter_args(%arg3 = %11) -> (tensor<1x32xf16>) {
      %16 = affine.linearize_index disjoint [%arg2, %c0] by (128, 8) : index
      %extracted_slice = tensor.extract_slice %5[%16, %arg0] [256, 1] [1, 1] : tensor<1024x2048xf16> to tensor<256x1xf16>
      %extracted_slice_1 = tensor.extract_slice %4[%16] [256] [1] : tensor<1024xf16> to tensor<256xf16>
      %expanded = tensor.expand_shape %extracted_slice_1 [[0, 1]] output_shape [32, 8] : tensor<256xf16> into tensor<32x8xf16>
      %17 = vector.transfer_read %expanded[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<32x8xf16>, vector<32x8xf16>
      %expanded_2 = tensor.expand_shape %extracted_slice [[0, 1], [2]] output_shape [32, 8, 1] : tensor<256x1xf16> into tensor<32x8x1xf16>
      %18 = vector.transfer_read %expanded_2[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : tensor<32x8x1xf16>, vector<32x8x1xf16>
      %19 = iree_vector_ext.to_layout %17 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [32, 1], element_tile = [1, 8], subgroup_strides = [0, 0], thread_strides = [1, 0]>) : vector<32x8xf16>
      %20 = iree_vector_ext.to_layout %18 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1], batch_tile = [1, 1, 1], outer_tile = [1, 1, 1], thread_tile = [32, 1, 1], element_tile = [1, 8, 1], subgroup_strides = [0, 0, 0], thread_strides = [1, 0, 0]>) : vector<32x8x1xf16>
      %21 = vector.transfer_read %arg3[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x32xf16>, vector<1x32xf16>
      %22 = iree_vector_ext.to_layout %21 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [1, 32], element_tile = [1, 1], subgroup_strides = [0, 0], thread_strides = [0, 1]>) : vector<1x32xf16>
      %23 = vector.transfer_write %22, %arg3[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf16>, tensor<1x32xf16>
      %24 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %19, %20, %22 : vector<32x8xf16>, vector<32x8x1xf16> into vector<1x32xf16>
      %25 = vector.transfer_write %24, %23[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf16>, tensor<1x32xf16>
      %26 = iree_vector_ext.to_layout %24 to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [1, 32], element_tile = [1, 1], subgroup_strides = [0, 0], thread_strides = [0, 1]>) : vector<1x32xf16>
      %27 = vector.transfer_write %26, %25[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf16>, tensor<1x32xf16>
      scf.yield %27 : tensor<1x32xf16>
    }
    %13 = vector.transfer_read %12[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x32xf16>, vector<1x32xf16>
    %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<1x32xf16> to vector<1xf16>
    %15 = vector.transfer_write %14, %9[%c0] {in_bounds = [true]} : vector<1xf16>, tensor<1xf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %15 into %arg1[%arg0] [1] [1] : tensor<1xf16> into tensor<2048xf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %7, %3 : tensor<2048xf16> into memref<2048xf16, #hal.descriptor_type<storage_buffer>>
  return
}

