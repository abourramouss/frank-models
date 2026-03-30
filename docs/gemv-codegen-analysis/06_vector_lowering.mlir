// -----// IR Dump After LLVMGPUVectorLoweringPass (iree-llvmgpu-vector-lowering) //-----
 //
func.func @gemv_f16_dispatch_0_matvec_like_2048x1024_f16() {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf16>
  %0 = ub.poison : vector<1x1x1x1x1x1xf16>
  %1 = ub.poison : vector<1x1x1x1x1x1x1x1x8xf16>
  %2 = ub.poison : vector<1x1x1x1x1x1x1x8xf16>
  %3 = ub.poison : vector<1x1x1x1x1x1x8xf16>
  %4 = ub.poison : vector<1x8xf16>
  %c7_i32 = arith.constant 7 : i32
  %c6_i32 = arith.constant 6 : i32
  %c5_i32 = arith.constant 5 : i32
  %c3_i32 = arith.constant 3 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c4_i32 = arith.constant 4 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x1x1x1x1x1x1x8x1xf16>
  %cst_1 = arith.constant dense<0.000000e+00> : vector<1x1x1x1x1x8xf16>
  %c0 = arith.constant 0 : index
  %cst_2 = arith.constant dense<0.000000e+00> : vector<1x1x1x1x1x1xf16>
  %thread_id_x = gpu.thread_id x upper_bound 32
  %5 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024xf16, #gpu.address_space<global>>
  %assume_align = memref.assume_alignment %5, 64 : memref<1024xf16, #gpu.address_space<global>>
  %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x2048xf16, #gpu.address_space<global>>
  %assume_align_3 = memref.assume_alignment %6, 64 : memref<1024x2048xf16, #gpu.address_space<global>>
  %7 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<2048xf16, #gpu.address_space<global>>
  %assume_align_4 = memref.assume_alignment %7, 64 : memref<2048xf16, #gpu.address_space<global>>
  %workgroup_id_x = hal.interface.workgroup.id[0] upper_bound 2048 : index
  %8 = arith.index_castui %thread_id_x : index to i32
  %9 = arith.muli %8, %c8_i32 overflow<nsw> : i32
  %10 = scf.for %arg0 = %c0_i32 to %c128_i32 step %c32_i32 iter_args(%arg1 = %cst_2) -> (vector<1x1x1x1x1x1xf16>)  : i32 {
    %42 = arith.muli %arg0, %c8_i32 overflow<nsw> : i32
    %43 = arith.addi %9, %42 : i32
    %44 = arith.index_castui %43 : i32 to index
    %45 = vector.load %assume_align[%44] : memref<1024xf16, #gpu.address_space<global>>, vector<8xf16>
    %46 = vector.insert %45, %4 [0] : vector<8xf16> into vector<1x8xf16>
    %47 = vector.insert_strided_slice %46, %cst_1 {offsets = [0, 0, 0, 0, 0, 0], strides = [1, 1]} : vector<1x8xf16> into vector<1x1x1x1x1x8xf16>
    %48 = vector.load %assume_align_3[%44, %workgroup_id_x] : memref<1024x2048xf16, #gpu.address_space<global>>, vector<1xf16>
    %49 = arith.addi %43, %c1_i32 overflow<nsw> : i32
    %50 = arith.index_castui %49 : i32 to index
    %51 = vector.load %assume_align_3[%50, %workgroup_id_x] : memref<1024x2048xf16, #gpu.address_space<global>>, vector<1xf16>
    %52 = arith.addi %43, %c2_i32 overflow<nsw> : i32
    %53 = arith.index_castui %52 : i32 to index
    %54 = vector.load %assume_align_3[%53, %workgroup_id_x] : memref<1024x2048xf16, #gpu.address_space<global>>, vector<1xf16>
    %55 = arith.addi %43, %c3_i32 overflow<nsw> : i32
    %56 = arith.index_castui %55 : i32 to index
    %57 = vector.load %assume_align_3[%56, %workgroup_id_x] : memref<1024x2048xf16, #gpu.address_space<global>>, vector<1xf16>
    %58 = arith.addi %43, %c4_i32 overflow<nsw> : i32
    %59 = arith.index_castui %58 : i32 to index
    %60 = vector.load %assume_align_3[%59, %workgroup_id_x] : memref<1024x2048xf16, #gpu.address_space<global>>, vector<1xf16>
    %61 = arith.addi %43, %c5_i32 overflow<nsw> : i32
    %62 = arith.index_castui %61 : i32 to index
    %63 = vector.load %assume_align_3[%62, %workgroup_id_x] : memref<1024x2048xf16, #gpu.address_space<global>>, vector<1xf16>
    %64 = arith.addi %43, %c6_i32 overflow<nsw> : i32
    %65 = arith.index_castui %64 : i32 to index
    %66 = vector.load %assume_align_3[%65, %workgroup_id_x] : memref<1024x2048xf16, #gpu.address_space<global>>, vector<1xf16>
    %67 = arith.addi %43, %c7_i32 overflow<nsw> : i32
    %68 = arith.index_castui %67 : i32 to index
    %69 = vector.load %assume_align_3[%68, %workgroup_id_x] : memref<1024x2048xf16, #gpu.address_space<global>>, vector<1xf16>
    %70 = vector.to_elements %48 : vector<1xf16>
    %71 = vector.to_elements %51 : vector<1xf16>
    %72 = vector.to_elements %54 : vector<1xf16>
    %73 = vector.to_elements %57 : vector<1xf16>
    %74 = vector.to_elements %60 : vector<1xf16>
    %75 = vector.to_elements %63 : vector<1xf16>
    %76 = vector.to_elements %66 : vector<1xf16>
    %77 = vector.to_elements %69 : vector<1xf16>
    %78 = vector.from_elements %70, %71, %72, %73, %74, %75, %76, %77 : vector<1x8x1xf16>
    %79 = vector.insert_strided_slice %78, %cst_0 {offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : vector<1x8x1xf16> into vector<1x1x1x1x1x1x1x8x1xf16>
    %80 = vector.insert %47, %3 [0] : vector<1x1x1x1x1x8xf16> into vector<1x1x1x1x1x1x8xf16>
    %81 = vector.insert %80, %2 [0] : vector<1x1x1x1x1x1x8xf16> into vector<1x1x1x1x1x1x1x8xf16>
    %82 = vector.insert %81, %1 [0] : vector<1x1x1x1x1x1x1x8xf16> into vector<1x1x1x1x1x1x1x1x8xf16>
    %83 = vector.transpose %82, [4, 6, 8, 0, 3, 1, 5, 2, 7] : vector<1x1x1x1x1x1x1x1x8xf16> to vector<1x1x8x1x1x1x1x1x1xf16>
    %84 = vector.transpose %79, [1, 4, 7, 2, 0, 5, 3, 8, 6] : vector<1x1x1x1x1x1x1x8x1xf16> to vector<1x1x8x1x1x1x1x1x1xf16>
    %85 = vector.shape_cast %arg1 : vector<1x1x1x1x1x1xf16> to vector<1xf16>
    %86 = vector.extract %83[0, 0, 7, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %87 = vector.extract %84[0, 0, 7, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %88 = math.fma %86, %87, %85 : vector<1xf16>
    %89 = vector.extract %83[0, 0, 6, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %90 = vector.extract %84[0, 0, 6, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %91 = math.fma %89, %90, %88 : vector<1xf16>
    %92 = vector.extract %83[0, 0, 5, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %93 = vector.extract %84[0, 0, 5, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %94 = math.fma %92, %93, %91 : vector<1xf16>
    %95 = vector.extract %83[0, 0, 4, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %96 = vector.extract %84[0, 0, 4, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %97 = math.fma %95, %96, %94 : vector<1xf16>
    %98 = vector.extract %83[0, 0, 3, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %99 = vector.extract %84[0, 0, 3, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %100 = math.fma %98, %99, %97 : vector<1xf16>
    %101 = vector.extract %83[0, 0, 2, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %102 = vector.extract %84[0, 0, 2, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %103 = math.fma %101, %102, %100 : vector<1xf16>
    %104 = vector.extract %83[0, 0, 1, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %105 = vector.extract %84[0, 0, 1, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %106 = math.fma %104, %105, %103 : vector<1xf16>
    %107 = vector.extract %83[0, 0, 0, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %108 = vector.extract %84[0, 0, 0, 0, 0, 0, 0, 0] : vector<1xf16> from vector<1x1x8x1x1x1x1x1x1xf16>
    %109 = math.fma %107, %108, %106 : vector<1xf16>
    %110 = vector.insert %109, %0 [0, 0, 0, 0, 0] : vector<1xf16> into vector<1x1x1x1x1x1xf16>
    scf.yield %110 : vector<1x1x1x1x1x1xf16>
  }
  %11 = vector.shape_cast %10 : vector<1x1x1x1x1x1xf16> to vector<1xf16>
  %12 = arith.addf %11, %cst : vector<1xf16>
  %13 = vector.extract %12[0] : f16 from vector<1xf16>
  %14 = arith.bitcast %13 : f16 to i16
  %15 = arith.extui %14 : i16 to i32
  %shuffleResult, %valid = gpu.shuffle xor %15, %c1_i32, %c32_i32 : i32
  %16 = arith.trunci %shuffleResult : i32 to i16
  %17 = arith.bitcast %16 : i16 to f16
  %18 = arith.addf %13, %17 : f16
  %19 = arith.bitcast %18 : f16 to i16
  %20 = arith.extui %19 : i16 to i32
  %shuffleResult_5, %valid_6 = gpu.shuffle xor %20, %c2_i32, %c32_i32 : i32
  %21 = arith.trunci %shuffleResult_5 : i32 to i16
  %22 = arith.bitcast %21 : i16 to f16
  %23 = arith.addf %18, %22 : f16
  %24 = arith.bitcast %23 : f16 to i16
  %25 = arith.extui %24 : i16 to i32
  %shuffleResult_7, %valid_8 = gpu.shuffle xor %25, %c4_i32, %c32_i32 : i32
  %26 = arith.trunci %shuffleResult_7 : i32 to i16
  %27 = arith.bitcast %26 : i16 to f16
  %28 = arith.addf %23, %27 : f16
  %29 = arith.bitcast %28 : f16 to i16
  %30 = arith.extui %29 : i16 to i32
  %shuffleResult_9, %valid_10 = gpu.shuffle xor %30, %c8_i32, %c32_i32 : i32
  %31 = arith.trunci %shuffleResult_9 : i32 to i16
  %32 = arith.bitcast %31 : i16 to f16
  %33 = arith.addf %28, %32 : f16
  %34 = arith.bitcast %33 : f16 to i16
  %35 = arith.extui %34 : i16 to i32
  %shuffleResult_11, %valid_12 = gpu.shuffle xor %35, %c16_i32, %c32_i32 : i32
  %36 = arith.trunci %shuffleResult_11 : i32 to i16
  %37 = arith.bitcast %36 : i16 to f16
  %38 = arith.addf %33, %37 : f16
  %39 = vector.broadcast %38 : f16 to vector<1xf16>
  %40 = arith.addf %39, %cst : vector<1xf16>
  %41 = arith.cmpi eq, %8, %c0_i32 : i32
  scf.if %41 {
    vector.store %40, %assume_align_4[%workgroup_id_x] : memref<2048xf16, #gpu.address_space<global>>, vector<1xf16>
  }
  return
}

