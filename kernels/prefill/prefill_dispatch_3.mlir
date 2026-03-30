// attn_q: [M, 1024] @ [1024, 2048] -> [M, 2048]
// 8 push constants, 3 bindings
hal.executable public @prefill_dispatch_3 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) objects([
    #hal.executable.object<{path = "dispatch_3.cubin"}>
  ]) {
    hal.executable.export public @prefill_dispatch_3_matmul_Dx2048x1024_f16 ordinal(0) layout(#hal.pipeline.layout<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      // arg1 = layer index (workload ordinal 0), arg2 = M/seq_len (workload ordinal 1)
      // grid = (ceil(M/32), ceil(2048/32), 1) = (ceil(M/32), 64, 1)
      %c32 = arith.constant 32 : index
      %c31 = arith.constant 31 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %m_up = arith.addi %arg2, %c31 : index
      %grid_x = arith.divui %m_up, %c32 : index
      hal.return %grid_x, %c64, %c1 : index, index, index
    } attributes {workgroup_size = [128 : index, 1 : index, 1 : index]}
  }
}
