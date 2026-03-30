// gate_up: [M, 1024] @ [1024, 6144] -> [M, 6144]
// 6 push constants, 3 bindings
hal.executable public @prefill_dispatch_17 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) objects([
    #hal.executable.object<{path = "dispatch_17.cubin"}>
  ]) {
    hal.executable.export public @prefill_dispatch_17_matmul_Dx6144x1024_f16 ordinal(0) layout(#hal.pipeline.layout<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      // arg1 = layer index, arg2 = M
      %c32 = arith.constant 32 : index
      %c31 = arith.constant 31 : index
      %c192 = arith.constant 192 : index
      %c1 = arith.constant 1 : index
      %m_up = arith.addi %arg2, %c31 : index
      %grid_x = arith.divui %m_up, %c32 : index
      hal.return %grid_x, %c192, %c1 : index, index, index
    } attributes {workgroup_size = [128 : index, 1 : index, 1 : index]}
  }
}
