// attn_o + residual: [M, 2048] @ [2048, 1024] + residual -> [M, 1024]
// 10 push constants, 3 bindings (binding(1) dual-use: input + residual)
hal.executable public @prefill_dispatch_13 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) objects([
    #hal.executable.object<{path = "dispatch_13.cubin"}>
  ]) {
    hal.executable.export public @prefill_dispatch_13_matmul_Dx1024x2048_f16 ordinal(0) layout(#hal.pipeline.layout<constants = 10, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      // arg1 = layer index, arg2 = M
      %c32 = arith.constant 32 : index
      %c31 = arith.constant 31 : index
      %c1 = arith.constant 1 : index
      %m_up = arith.addi %arg2, %c31 : index
      %grid_x = arith.divui %m_up, %c32 : index
      hal.return %grid_x, %c32, %c1 : index, index, index
    } attributes {workgroup_size = [128 : index, 1 : index, 1 : index]}
  }
}
