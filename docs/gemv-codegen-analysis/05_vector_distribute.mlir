// -----// IR Dump After LLVMGPUVectorDistributePass (iree-llvmgpu-vector-distribute) //-----
 //
func.func @gemv_f16_dispatch_0_matvec_like_2048x1024_f16() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [32, 1, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf16>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x1x1x1x1x1x1x8x1xf16>
  %cst_1 = arith.constant dense<0.000000e+00> : vector<1x1x1x1x1x8xf16>
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %0 = ub.poison : f16
  %cst_2 = arith.constant dense<0.000000e+00> : vector<1x1x1xf16>
  %cst_3 = arith.constant dense<0.000000e+00> : vector<1x1x1x1x1x1xf16>
  %thread_id_z = gpu.thread_id z
  %thread_id_y = gpu.thread_id y
  %thread_id_x = gpu.thread_id x
  %1 = affine.linearize_index disjoint [%thread_id_z, %thread_id_y, %thread_id_x] by (1, 1, 32) : index
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024xf16, #hal.descriptor_type<storage_buffer>>
  %assume_align = memref.assume_alignment %2, 64 : memref<1024xf16, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>>
  %assume_align_4 = memref.assume_alignment %3, 64 : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<2048xf16, #hal.descriptor_type<storage_buffer>>
  %assume_align_5 = memref.assume_alignment %4, 64 : memref<2048xf16, #hal.descriptor_type<storage_buffer>>
  scf.forall (%arg0) in (2048) {
    %subview = memref.subview %assume_align_5[%arg0] [1] [1] : memref<2048xf16, #hal.descriptor_type<storage_buffer>> to memref<1xf16, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
    %5 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %cst_3) -> (vector<1x1x1x1x1x1xf16>) {
      %16 = affine.linearize_index disjoint [%arg1, %c0] by (128, 8) : index
      %subview_6 = memref.subview %assume_align[%16] [256] [1] : memref<1024xf16, #hal.descriptor_type<storage_buffer>> to memref<256xf16, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
      %subview_7 = memref.subview %assume_align_4[%16, %arg0] [256, 1] [1, 1] : memref<1024x2048xf16, #hal.descriptor_type<storage_buffer>> to memref<256x1xf16, strided<[2048, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
      %expand_shape = memref.expand_shape %subview_6 [[0, 1]] output_shape [32, 8] : memref<256xf16, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>> into memref<32x8xf16, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
      %17:4 = affine.delinearize_index %1 into (1, 1, 32) : index, index, index, index
      %18:3 = affine.delinearize_index %1 into (32, 1) : index, index, index
      %19 = affine.linearize_index disjoint [%17#2, %c0, %c0, %18#1, %c0] by (1, 1, 1, 32, 1) : index
      %20 = affine.linearize_index disjoint [%17#1, %c0, %c0, %18#2, %c0] by (1, 1, 1, 1, 8) : index
      %21 = vector.transfer_read %expand_shape[%19, %20], %0 {in_bounds = [true, true]} : memref<32x8xf16, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
      %22 = vector.insert_strided_slice %21, %cst_1 {offsets = [0, 0, 0, 0, 0, 0], strides = [1, 1]} : vector<1x8xf16> into vector<1x1x1x1x1x8xf16>
      %expand_shape_8 = memref.expand_shape %subview_7 [[0, 1], [2]] output_shape [32, 8, 1] : memref<256x1xf16, strided<[2048, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> into memref<32x8x1xf16, strided<[16384, 2048, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
      %23:5 = affine.delinearize_index %1 into (1, 1, 1, 32) : index, index, index, index, index
      %24:4 = affine.delinearize_index %1 into (32, 1, 1) : index, index, index, index
      %25 = affine.linearize_index disjoint [%23#3, %c0, %c0, %24#1, %c0] by (1, 1, 1, 32, 1) : index
      %26 = affine.linearize_index disjoint [%23#2, %c0, %c0, %24#3, %c0] by (1, 1, 1, 1, 8) : index
      %27 = affine.linearize_index disjoint [%23#1, %c0, %c0, %24#2, %c0] by (1, 1, 1, 1, 1) : index
      %28 = vector.transfer_read %expand_shape_8[%25, %26, %27], %0 {in_bounds = [true, true, true]} : memref<32x8x1xf16, strided<[16384, 2048, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<1x8x1xf16>
      %29 = vector.insert_strided_slice %28, %cst_0 {offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : vector<1x8x1xf16> into vector<1x1x1x1x1x1x1x8x1xf16>
      %30 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d4, d5, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d0, d4, d5, d3, d7, d8, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d3, d4, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %29, %cst_3 : vector<1x1x1x1x1x8xf16>, vector<1x1x1x1x1x1x1x8x1xf16> into vector<1x1x1x1x1x1xf16>
      %31 = vector.shape_cast %30 : vector<1x1x1x1x1x1xf16> to vector<1x1x1x1x1x1x1x1x1xf16>
      %32 = vector.multi_reduction <add>, %31, %cst_3 [2, 5, 8] : vector<1x1x1x1x1x1x1x1x1xf16> to vector<1x1x1x1x1x1xf16>
      %33 = arith.addf %32, %arg2 : vector<1x1x1x1x1x1xf16>
      scf.yield %33 : vector<1x1x1x1x1x1xf16>
    }
    %6 = vector.multi_reduction <add>, %5, %cst_2 [1, 3, 5] : vector<1x1x1x1x1x1xf16> to vector<1x1x1xf16>
    %7 = vector.extract %6[0, 0, 0] : f16 from vector<1x1x1xf16>
    %8 = gpu.subgroup_reduce add %7 cluster(size = 32) : (f16) -> f16
    %9 = vector.insert %8, %cst [0] : f16 into vector<1xf16>
    %10 = vector.shape_cast %9 : vector<1xf16> to vector<1x1x1xf16>
    %11 = arith.addf %10, %cst_2 : vector<1x1x1xf16>
    %12:3 = affine.delinearize_index %1 into (1, 32) : index, index, index
    %13:2 = affine.delinearize_index %1 into (1) : index, index
    %14:3 = affine.delinearize_index %1 into (32, 1, 1) : index, index, index
    %15 = arith.cmpi eq, %14#0, %c0 : index
    scf.if %15 {
      %16 = affine.linearize_index disjoint [%12#1, %c0, %c0, %13#1, %c0] by (1, 1, 1, 1, 1) : index
      %17 = vector.extract %11[0, 0] : vector<1xf16> from vector<1x1x1xf16>
      vector.transfer_write %17, %subview[%16] {in_bounds = [true]} : vector<1xf16>, memref<1xf16, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  return
}

