// -----// IR Dump After TileAndDistributeToWorkgroupsUsingForallOpPass (iree-codegen-tile-and-distribute-to-workgroups-using-forall-op) //-----
 //
func.func @gemv_f16_dispatch_0_matvec_like_2048x1024_f16() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [32, 1, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024xf16, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<2048xf16, #hal.descriptor_type<storage_buffer>>
  %3 = iree_codegen.load_from_buffer %0 : memref<1024xf16, #hal.descriptor_type<storage_buffer>> -> tensor<1024xf16>
  %4 = iree_codegen.load_from_buffer %1 : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>> -> tensor<1024x2048xf16>
  %5 = tensor.empty() : tensor<2048xf16>
  %6 = scf.forall (%arg0) in (2048) shared_outs(%arg1 = %5) -> (tensor<2048xf16>) {
    %extracted_slice = tensor.extract_slice %4[0, %arg0] [1024, 1] [1, 1] : tensor<1024x2048xf16> to tensor<1024x1xf16>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0] [1] [1] : tensor<2048xf16> to tensor<1xf16>
    %7 = linalg.fill ins(%cst : f16) outs(%extracted_slice_0 : tensor<1xf16>) -> tensor<1xf16>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%3, %extracted_slice : tensor<1024xf16>, tensor<1024x1xf16>) outs(%7 : tensor<1xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{expand_dims = #iree_gpu.expand_dims<[[0], [1, 2]], output_shape = [?, ?, 8]>, lane_basis = [[1, 32, 1], [0, 1, 2]], partial_reduction = [0, 32, 0], subgroup_basis = [[1, 1, 1], [0, 1, 2]], thread = [0, 1, 8], workgroup = [1, 0, 0]}>} {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %9 = arith.mulf %in, %in_1 : f16
      %10 = arith.addf %out, %9 : f16
      linalg.yield %10 : f16
    } -> tensor<1xf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg1[%arg0] [1] [1] : tensor<1xf16> into tensor<2048xf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %6, %2 : tensor<2048xf16> into memref<2048xf16, #hal.descriptor_type<storage_buffer>>
  return
}

