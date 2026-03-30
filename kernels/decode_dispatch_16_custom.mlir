hal.executable public @decode_dispatch_16 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) objects([
    #hal.executable.object<{path = "gemv_q8_6144x1024.ptx"}>
  ]) {
    hal.executable.export public @decode_dispatch_16_matvec_like_6144x1024_f16
      ordinal(0)
      layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>)
      attributes {workgroup_size = [256 : index, 1 : index, 1 : index]}
  }
}
