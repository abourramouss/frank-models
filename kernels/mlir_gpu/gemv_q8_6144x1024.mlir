// Q8_0 fused dequant+GEMV [6144, 1024] — GPU-dialect MLIR kernel
// 256 threads (8 warps), 8 rows/block, shared memory, warp shuffle reduction
// Compiles through IREE's GPU backend to PTX — no external CUDA needed.

module attributes {transform.with_named_sequence} {
  stream.executable private @q8_gemv_6144x1024 {
    stream.executable.export public @q8_gemv_6144x1024 workgroups() -> (index, index, index) {
      %c768 = arith.constant 768 : index
      %c1 = arith.constant 1 : index
      stream.return %c768, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @q8_gemv_6144x1024(
          %q8_arg: !stream.binding,
          %input_arg: !stream.binding,
          %output_arg: !stream.binding
      ) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c34 = arith.constant 34 : index
        %c256 = arith.constant 256 : index
        %c1024 = arith.constant 1024 : index
        %c1088 = arith.constant 1088 : index
        %c6144 = arith.constant 6144 : index
        %c8_i16 = arith.constant 8 : i16
        %cst_zero = arith.constant 0.000000e+00 : f32
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c16_i32 = arith.constant 16 : i32
        %c32_i32 = arith.constant 32 : i32

        // Bind buffers as flat memref
        %q8_mem = stream.binding.subspan %q8_arg[%c0] : !stream.binding -> memref<6684672xi8>
        %input_mem = stream.binding.subspan %input_arg[%c0] : !stream.binding -> memref<1024xf16>
        %output_mem = stream.binding.subspan %output_arg[%c0] : !stream.binding -> memref<6144xf16>

        // Thread/block IDs
        %tid = gpu.thread_id x upper_bound 256
        %bid = gpu.block_id x upper_bound 768

        // Allocate shared memory for input vector
        %smem_raw = memref.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
        %smem = memref.view %smem_raw[%c0][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xf16, #gpu.address_space<workgroup>>

        // Cooperative load: 256 threads, 1024 elements, 4 per thread
        scf.for %i = %tid to %c1024 step %c256 {
          %val = memref.load %input_mem[%i] : memref<1024xf16>
          memref.store %val, %smem[%i] : memref<1024xf16, #gpu.address_space<workgroup>>
        }
        gpu.barrier

        // Row = bid * 8 + tid / 32
        %warp_id = arith.divui %tid, %c32 : index
        %lane = arith.remui %tid, %c32 : index
        %row_base = arith.muli %bid, %c8 : index
        %row = arith.addi %row_base, %warp_id : index

        %in_bounds = arith.cmpi ult, %row, %c6144 : index
        scf.if %in_bounds {
          // Byte offset into Q8 data for this row and lane's block
          %row_off = arith.muli %row, %c1088 : index
          %blk_off = arith.muli %lane, %c34 : index
          %byte_off = arith.addi %row_off, %blk_off : index

          // Load scale
          %s0_i8 = memref.load %q8_mem[%byte_off] : memref<6684672xi8>
          %s1_idx = arith.addi %byte_off, %c1 : index
          %s1_i8 = memref.load %q8_mem[%s1_idx] : memref<6684672xi8>
          %s0_i16 = arith.extui %s0_i8 : i8 to i16
          %s1_i16 = arith.extui %s1_i8 : i8 to i16
          %s1_sh = arith.shli %s1_i16, %c8_i16 : i16
          %scale_i16 = arith.ori %s0_i16, %s1_sh : i16
          %scale_f16 = arith.bitcast %scale_i16 : i16 to f16
          %scale = arith.extf %scale_f16 : f16 to f32

          // Dot product over 32 Q8 values
          %val_base = arith.addi %byte_off, %c2 : index
          %k_base = arith.muli %lane, %c32 : index
          %dot = scf.for %j = %c0 to %c32 step %c1 iter_args(%sum = %cst_zero) -> (f32) {
            %j_off = arith.addi %val_base, %j : index
            %qv_i8 = memref.load %q8_mem[%j_off] : memref<6684672xi8>
            %qv_f32 = arith.sitofp %qv_i8 : i8 to f32
            %k_idx = arith.addi %k_base, %j : index
            %inp_f16 = memref.load %smem[%k_idx] : memref<1024xf16, #gpu.address_space<workgroup>>
            %inp_f32 = arith.extf %inp_f16 : f16 to f32
            %prod = arith.mulf %qv_f32, %inp_f32 : f32
            %new_sum = arith.addf %sum, %prod : f32
            scf.yield %new_sum : f32
          }
          %scaled = arith.mulf %scale, %dot : f32

          // Warp shuffle reduction (5 rounds for 32 lanes)
          %r1_v, %r1_ok = gpu.shuffle xor %scaled, %c16_i32, %c32_i32 : f32
          %r1 = arith.addf %scaled, %r1_v : f32
          %r2_v, %r2_ok = gpu.shuffle xor %r1, %c8_i32, %c32_i32 : f32
          %r2 = arith.addf %r1, %r2_v : f32
          %r3_v, %r3_ok = gpu.shuffle xor %r2, %c4_i32, %c32_i32 : f32
          %r3 = arith.addf %r2, %r3_v : f32
          %r4_v, %r4_ok = gpu.shuffle xor %r3, %c2_i32, %c32_i32 : f32
          %r4 = arith.addf %r3, %r4_v : f32
          %r5_v, %r5_ok = gpu.shuffle xor %r4, %c1_i32, %c32_i32 : f32
          %r5 = arith.addf %r4, %r5_v : f32

          // Lane 0 writes
          %is_lane0 = arith.cmpi eq, %lane, %c0 : index
          scf.if %is_lane0 {
            %out_f16 = arith.truncf %r5 : f32 to f16
            memref.store %out_f16, %output_mem[%row] : memref<6144xf16>
          }
        }
        return
      }
    }
  }

  // Benchmark entry: just dispatches the stream executable
  func.func @bench(%q8: !hal.buffer_view, %input: !hal.buffer_view) -> !hal.buffer_view {
    %c6684672 = arith.constant 6684672 : index
    %c2048 = arith.constant 2048 : index
    %c12288 = arith.constant 12288 : index
    %q8_size = hal.buffer_view.dim<%q8 : !hal.buffer_view>[0] : index
    %q8_t = hal.tensor.import %q8 : !hal.buffer_view -> tensor<?xi8>{%q8_size}
    %inp_t = hal.tensor.import %input : !hal.buffer_view -> tensor<1024xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %out_init = tensor.empty() : tensor<6144xf16>
    %out = linalg.fill ins(%cst : f16) outs(%out_init : tensor<6144xf16>) -> tensor<6144xf16>
    %result = stream.tensor.dispatch @q8_gemv_6144x1024::@q8_gemv_6144x1024(%q8_t, %inp_t, %out)
        : (tensor<?xi8>{%q8_size}, tensor<1024xf16>, tensor<6144xf16>) -> tensor<6144xf16>
    %bv = hal.tensor.export %result : tensor<6144xf16> -> !hal.buffer_view
    return %bv : !hal.buffer_view
  }
}
