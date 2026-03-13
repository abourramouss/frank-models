module @llm_inference attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>]> : !hal.device
  util.func public @allocate_kv_cache(%arg0: index, %arg1: index) -> !util.list<?> attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @allocate_kv_cache(%input0: index, %input1: index) -> (%output0: !util.list<?>)"}} {
    %c8192 = arith.constant 8192 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c-1_i64 = arith.constant -1 : i64
    %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %0 = arith.muli %arg0, %arg1 : index
    %1 = arith.muli %0, %c8192 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%1}
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    %view = hal.buffer_view.create buffer(%buffer : !hal.buffer)[%c0, %1] shape([%arg0, %arg1, %c16, %c128]) type(%element_type_f32) encoding(%dense_row_major) : !hal.buffer_view
    %buffer_0 = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%1}
    %view_1 = hal.buffer_view.create buffer(%buffer_0 : !hal.buffer)[%c0, %1] shape([%arg0, %arg1, %c16, %c128]) type(%element_type_f32) encoding(%dense_row_major) : !hal.buffer_view
    %2 = util.list.create %c2 : !util.list<?>
    util.list.resize %2, %c2 : !util.list<?>
    util.list.set %2[%c0], %view : !hal.buffer_view -> !util.list<?>
    util.list.set %2[%c1], %view_1 : !hal.buffer_view -> !util.list<?>
    util.return %2 : !util.list<?>
  }
  flow.executable private @prefill_dispatch_0 {
    flow.executable.export public @prefill_dispatch_0_gather_50304x2048xf32_dispatch_tensor_store workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_0_gather_50304x2048xf32_dispatch_tensor_store(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<50304x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>, %arg2: index, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>{%0}
        %3 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%1}
        %4 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [50304, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<50304x2048xf32>> -> tensor<50304x2048xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [%0], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>{%0} -> tensor<?xi64>
        %6 = tensor.empty(%1) : tensor<?x2048xf32>
        %7 = iree_linalg_ext.gather dimension_map = [0] ins(%4, %5 : tensor<50304x2048xf32>, tensor<?xi64>) outs(%6 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %3, offsets = [0, 0], sizes = [%1, 2048], strides = [1, 1] : tensor<?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%1}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_1 {
    flow.executable.export public @prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%arg0: i1, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [2048], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>> -> tensor<2048xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [2048], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>> -> tensor<2048xf32>
        %2 = tensor.empty() : tensor<2048xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %0, %1 : i1, tensor<2048xf32>, tensor<2048xf32>) outs(%2 : tensor<2048xf32>) {
        ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
          %4 = arith.select %in, %in_0, %in_1 : f32
          linalg.yield %4 : f32
        } -> tensor<2048xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0], sizes = [2048], strides = [1] : tensor<2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf32>>
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_3 {
    flow.executable.export public @prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%arg0: i1, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4194304xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4194304xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4194304xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [4194304], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4194304xf32>> -> tensor<4194304xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [4194304], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4194304xf32>> -> tensor<4194304xf32>
        %2 = tensor.empty() : tensor<4194304xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %0, %1 : i1, tensor<4194304xf32>, tensor<4194304xf32>) outs(%2 : tensor<4194304xf32>) {
        ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
          %4 = arith.select %in, %in_0, %in_1 : f32
          linalg.yield %4 : f32
        } -> tensor<4194304xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0], sizes = [4194304], strides = [1] : tensor<4194304xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4194304xf32>>
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_9 {
    flow.executable.export public @prefill_dispatch_9_elementwise_131072_i1xf32xf32xf32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_9_elementwise_131072_i1xf32xf32xf32(%arg0: i1, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<131072xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [131072], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072xf32>> -> tensor<131072xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [131072], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072xf32>> -> tensor<131072xf32>
        %2 = tensor.empty() : tensor<131072xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %0, %1 : i1, tensor<131072xf32>, tensor<131072xf32>) outs(%2 : tensor<131072xf32>) {
        ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
          %4 = arith.select %in, %in_0, %in_1 : f32
          linalg.yield %4 : f32
        } -> tensor<131072xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0], sizes = [131072], strides = [1] : tensor<131072xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<131072xf32>>
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_10 {
    flow.executable.export public @prefill_dispatch_10_elementwise_134217728_i1xf32xf32xf32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_10_elementwise_134217728_i1xf32xf32xf32(%arg0: i1, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<134217728xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<134217728xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<134217728xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [134217728], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<134217728xf32>> -> tensor<134217728xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [134217728], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<134217728xf32>> -> tensor<134217728xf32>
        %2 = tensor.empty() : tensor<134217728xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %0, %1 : i1, tensor<134217728xf32>, tensor<134217728xf32>) outs(%2 : tensor<134217728xf32>) {
        ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
          %4 = arith.select %in, %in_0, %in_1 : f32
          linalg.yield %4 : f32
        } -> tensor<134217728xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0], sizes = [134217728], strides = [1] : tensor<134217728xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<134217728xf32>>
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_13 {
    flow.executable.export public @prefill_dispatch_13_reduction_Dx2048_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_13_reduction_Dx2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>) {
        %cst = arith.constant 9.99999974E-6 : f32
        %cst_0 = arith.constant 2.048000e+03 : f32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [2048], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>> -> tensor<2048xf32>
        %5 = tensor.empty(%0) : tensor<?x2048xf32>
        %6 = tensor.empty(%0) : tensor<?xf32>
        %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<?xf32>) -> tensor<?xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<?x2048xf32>) outs(%7 : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %10 = arith.mulf %in, %in : f32
          %11 = arith.addf %out, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<?xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %8, %4 : tensor<?x2048xf32>, tensor<?xf32>, tensor<2048xf32>) outs(%5 : tensor<?x2048xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
          %10 = arith.divf %in_2, %cst_0 : f32
          %11 = arith.addf %10, %cst : f32
          %12 = math.sqrt %11 : f32
          %13 = arith.divf %in, %12 : f32
          %14 = arith.mulf %13, %in_3 : f32
          linalg.yield %14 : f32
        } -> tensor<?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %2, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : tensor<?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_14 {
    flow.executable.export public @prefill_dispatch_14_elementwise_broadcast_Dx4194304_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_14_elementwise_broadcast_Dx4194304_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4194304xf32>>, %arg1: index, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x4194304xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x4194304xf32>>{%0}
        %2 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [4194304], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4194304xf32>> -> tensor<4194304xf32>
        %3 = tensor.empty(%0) : tensor<?x4194304xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<4194304xf32>) outs(%3 : tensor<?x4194304xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x4194304xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [%0, 4194304], strides = [1, 1] : tensor<?x4194304xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x4194304xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_17 {
    flow.executable.export public @prefill_dispatch_17_batch_matmul_DxDx2048x2048_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_17_batch_matmul_DxDx2048x2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x2048xf32>>, %arg2: index, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x2048xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x2048xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x2048xf32>>{%0, %1}
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%0, %1, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%0, %1} -> tensor<?x?x2048xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [%0, 2048, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x2048xf32>>{%0} -> tensor<?x2048x2048xf32>
        %7 = tensor.empty(%0, %1) : tensor<?x?x2048xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x?x2048xf32>) -> tensor<?x?x2048xf32>
        %9 = linalg.batch_matmul ins(%5, %6 : tensor<?x?x2048xf32>, tensor<?x2048x2048xf32>) outs(%8 : tensor<?x?x2048xf32>) -> tensor<?x?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0, 0, 0], sizes = [%0, %1, 2048], strides = [1, 1, 1] : tensor<?x?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x2048xf32>>{%0, %1}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_22 {
    flow.executable.export public @prefill_dispatch_22_elementwise_broadcast_D_f32 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_22_elementwise_broadcast_D_f32(%arg0: index, %arg1: index, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
        %c16 = arith.constant 16 : index
        %c128 = arith.constant 128 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 2 : index
        %3 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%2, %1}
        %4 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [%2, %1, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%2, %1} -> tensor<?x?x2048xf32>
        %6 = tensor.empty(%0) : tensor<?xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%6 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %8 = linalg.index 0 : index
          %9 = arith.remsi %8, %c128 : index
          %10 = arith.divsi %8, %c128 : index
          %11 = arith.remsi %10, %c16 : index
          %12 = arith.divsi %10, %c16 : index
          %13 = arith.remsi %12, %1 : index
          %14 = arith.divsi %12, %1 : index
          %15 = arith.muli %11, %c128 : index
          %16 = arith.addi %15, %9 : index
          %extracted = tensor.extract %5[%14, %13, %16] : tensor<?x?x2048xf32>
          linalg.yield %extracted : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %4, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_24 {
    flow.executable.export public @prefill_dispatch_24_elementwise_broadcast_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_24_elementwise_broadcast_64_f32(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>) {
        %cst = arith.constant 2.000000e+00 : f32
        %cst_0 = arith.constant 1.280000e+02 : f32
        %cst_1 = arith.constant 1.000000e+04 : f32
        %0 = tensor.empty() : tensor<64xf32>
        %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%0 : tensor<64xf32>) {
        ^bb0(%out: f32):
          %2 = linalg.index 0 : index
          %3 = arith.index_cast %2 : index to i32
          %4 = arith.sitofp %3 : i32 to f32
          %5 = arith.mulf %4, %cst : f32
          %6 = arith.divf %5, %cst_0 : f32
          %7 = arith.negf %6 : f32
          %8 = math.powf %cst_1, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %1, %arg0, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_25 {
    flow.executable.export public @prefill_dispatch_25_elementwise_broadcast_D_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_25_elementwise_broadcast_D_f32(%arg0: index, %arg1: index, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi64>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>, %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x128xf32>>, %arg5: index, %arg6: index, %arg7: index, %arg8: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
        %c16 = arith.constant 16 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg6, 3 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg7, 4 : index
        %5 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi64>>{%2, %3}
        %6 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x128xf32>>{%4, %1}
        %7 = flow.dispatch.tie_shape %arg8 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        %8 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi64>>{%2, %3} -> tensor<?x?xi64>
        %9 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %6, offsets = [0, 0, 0, 0], sizes = [%4, %1, 16, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x128xf32>>{%4, %1} -> tensor<?x?x16x128xf32>
        %11 = tensor.empty(%0) : tensor<?xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%11 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %13 = linalg.index 0 : index
          %14 = arith.remsi %13, %c128 : index
          %15 = arith.divsi %13, %c128 : index
          %16 = arith.remsi %15, %1 : index
          %17 = arith.divsi %15, %1 : index
          %18 = arith.remsi %17, %c16 : index
          %19 = arith.divsi %17, %c16 : index
          %extracted = tensor.extract %8[%19, %16] : tensor<?x?xi64>
          %20 = arith.trunci %extracted : i64 to i32
          %21 = arith.sitofp %20 : i32 to f32
          %22 = arith.cmpi slt, %14, %c64 : index
          %23 = scf.if %22 -> (f32) {
            %extracted_0 = tensor.extract %9[%14] : tensor<64xf32>
            %24 = arith.mulf %21, %extracted_0 : f32
            %25 = math.cos %24 : f32
            %26 = math.sin %24 : f32
            %extracted_1 = tensor.extract %10[%19, %16, %18, %14] : tensor<?x?x16x128xf32>
            %27 = arith.addi %14, %c64 : index
            %extracted_2 = tensor.extract %10[%19, %16, %18, %27] : tensor<?x?x16x128xf32>
            %28 = arith.mulf %extracted_1, %25 : f32
            %29 = arith.mulf %extracted_2, %26 : f32
            %30 = arith.subf %28, %29 : f32
            scf.yield %30 : f32
          } else {
            %24 = arith.subi %14, %c64 : index
            %extracted_0 = tensor.extract %9[%24] : tensor<64xf32>
            %25 = arith.mulf %21, %extracted_0 : f32
            %26 = math.cos %25 : f32
            %27 = math.sin %25 : f32
            %extracted_1 = tensor.extract %10[%19, %16, %18, %14] : tensor<?x?x16x128xf32>
            %extracted_2 = tensor.extract %10[%19, %16, %18, %24] : tensor<?x?x16x128xf32>
            %28 = arith.mulf %extracted_2, %27 : f32
            %29 = arith.mulf %extracted_1, %26 : f32
            %30 = arith.addf %28, %29 : f32
            scf.yield %30 : f32
          }
          linalg.yield %23 : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %7, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_27 {
    flow.executable.export public @prefill_dispatch_27_elementwise_broadcast_D_f32 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_27_elementwise_broadcast_D_f32(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>, %arg2: index, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
        %c16 = arith.constant 16 : index
        %c128 = arith.constant 128 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 2 : index
        %3 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%1, %2}
        %4 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [%1, %2, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%1, %2} -> tensor<?x?x2048xf32>
        %6 = tensor.empty(%0) : tensor<?xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%6 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %8 = linalg.index 0 : index
          %9 = arith.remsi %8, %2 : index
          %10 = arith.divsi %8, %2 : index
          %11 = arith.remsi %10, %c128 : index
          %12 = arith.divsi %10, %c128 : index
          %13 = arith.remsi %12, %c16 : index
          %14 = arith.divsi %12, %c16 : index
          %15 = arith.muli %13, %c128 : index
          %16 = arith.addi %15, %11 : index
          %extracted = tensor.extract %5[%14, %9, %16] : tensor<?x?x2048xf32>
          linalg.yield %extracted : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %4, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_28 {
    flow.executable.export public @prefill_dispatch_28_attention_DxDx128xDx128 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_28_attention_DxDx128xDx128(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128x?xf32>>, %arg4: index, %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x128xf32>>) {
        %cst = arith.constant 0.0883883461 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 1 : index
        %2 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>{%0, %1}
        %4 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128x?xf32>>{%0, %1}
        %5 = flow.dispatch.tie_shape %arg5 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x128xf32>>{%0, %1}
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%0, %1, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>{%0, %1} -> tensor<?x?x128xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [%0, %1, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>{%0, %1} -> tensor<?x?x128xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [%0, 128, %1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128x?xf32>>{%0, %1} -> tensor<?x128x?xf32>
        %9 = tensor.empty(%0, %1) : tensor<?x?x128xf32>
        %10 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%6, %7, %8, %cst : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x128x?xf32>, f32) outs(%9 : tensor<?x?x128xf32>) {
        ^bb0(%arg6: f32):
          iree_linalg_ext.yield %arg6 : f32
        } -> tensor<?x?x128xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %5, offsets = [0, 0, 0], sizes = [%0, %1, 128], strides = [1, 1, 1] : tensor<?x?x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x128xf32>>{%0, %1}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_29 {
    flow.executable.export public @prefill_dispatch_29_elementwise_broadcast_D_f32 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_29_elementwise_broadcast_D_f32(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>, %arg2: index, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
        %c128 = arith.constant 128 : index
        %c16 = arith.constant 16 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 2 : index
        %3 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>{%1, %2}
        %4 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [%1, %2, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>{%1, %2} -> tensor<?x?x128xf32>
        %6 = tensor.empty(%0) : tensor<?xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%6 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %8 = linalg.index 0 : index
          %9 = arith.remsi %8, %c128 : index
          %10 = arith.divsi %8, %c128 : index
          %11 = arith.remsi %10, %c16 : index
          %12 = arith.divsi %10, %c16 : index
          %13 = arith.remsi %12, %2 : index
          %14 = arith.divsi %12, %2 : index
          %15 = arith.muli %14, %c16 : index
          %16 = arith.addi %15, %11 : index
          %extracted = tensor.extract %5[%16, %13, %9] : tensor<?x?x128xf32>
          linalg.yield %extracted : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %4, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_31 {
    flow.executable.export public @prefill_dispatch_31_batch_matmul_DxDx2048x2048_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_31_batch_matmul_DxDx2048x2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x2048xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>, %arg3: index, %arg4: index, %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x2048xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x2048xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%0, %1}
        %5 = flow.dispatch.tie_shape %arg5 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x2048xf32>>{%0, %1}
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%0, %1, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%0, %1} -> tensor<?x?x2048xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [%0, 2048, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x2048xf32>>{%0} -> tensor<?x2048x2048xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [%0, %1, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%0, %1} -> tensor<?x?x2048xf32>
        %9 = tensor.empty(%0, %1) : tensor<?x?x2048xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?x2048xf32>) -> tensor<?x?x2048xf32>
        %11 = linalg.batch_matmul ins(%6, %7 : tensor<?x?x2048xf32>, tensor<?x2048x2048xf32>) outs(%10 : tensor<?x?x2048xf32>) -> tensor<?x?x2048xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %11 : tensor<?x?x2048xf32>, tensor<?x?x2048xf32>) outs(%9 : tensor<?x?x2048xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %13 = arith.addf %in, %in_0 : f32
          linalg.yield %13 : f32
        } -> tensor<?x?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %5, offsets = [0, 0, 0], sizes = [%0, %1, 2048], strides = [1, 1, 1] : tensor<?x?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x2048xf32>>{%0, %1}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_33 {
    flow.executable.export public @prefill_dispatch_33_matmul_64xDx2048_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_33_matmul_64xDx2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x?xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x?xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [64, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x2048xf32>> -> tensor<64x2048xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %5 = tensor.empty(%0) : tensor<64x?xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64x?xf32>) -> tensor<64x?xf32>
        %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%3, %4 : tensor<64x2048xf32>, tensor<?x2048xf32>) outs(%6 : tensor<64x?xf32>) -> tensor<64x?xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, %0], strides = [1, 1] : tensor<64x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_34 {
    flow.executable.export public @prefill_dispatch_34_softmax_64xDxf32_dispatch_tensor_store workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_34_softmax_64xDxf32_dispatch_tensor_store(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x?xf32>>, %arg1: index, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x?xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x?xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x?xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, %0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x?xf32>>{%0} -> tensor<64x?xf32>
        %4 = tensor.empty(%0) : tensor<64x?xf32>
        %5 = linalg.softmax dimension(0) ins(%3 : tensor<64x?xf32>) outs(%4 : tensor<64x?xf32>) -> tensor<64x?xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %2, offsets = [0, 0], sizes = [64, %0], strides = [1, 1] : tensor<64x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_35 {
    flow.executable.export public @prefill_dispatch_35_topk_64xDxf32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_35_topk_64xDxf32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x?xf32>>, %arg1: index, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xi32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x?xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xf32>>{%0}
        %3 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xi32>>{%0}
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, %0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x?xf32>>{%0} -> tensor<64x?xf32>
        %5 = tensor.empty(%0) : tensor<8x?xi32>
        %6 = tensor.empty(%0) : tensor<8x?xf32>
        %7:2 = iree_linalg_ext.topk dimension(0) ins(%4 : tensor<64x?xf32>) outs(%6, %5 : tensor<8x?xf32>, tensor<8x?xi32>) {
        ^bb0(%arg4: f32, %arg5: f32):
          %8 = arith.cmpf ogt, %arg4, %arg5 : f32
          iree_linalg_ext.yield %8 : i1
        } -> tensor<8x?xf32>, tensor<8x?xi32>
        iree_tensor_ext.dispatch.tensor.store %7#0, %2, offsets = [0, 0], sizes = [8, %0], strides = [1, 1] : tensor<8x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xf32>>{%0}
        iree_tensor_ext.dispatch.tensor.store %7#1, %3, offsets = [0, 0], sizes = [8, %0], strides = [1, 1] : tensor<8x?xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xi32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_36 {
    flow.executable.export public @prefill_dispatch_36_transpose_2097152x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_36_transpose_2097152x64_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2097152x64xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x2097152xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2097152, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2097152x64xf32>> -> tensor<2097152x64xf32>
        %1 = tensor.empty() : tensor<64x2097152xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<2097152x64xf32>) outs(%1 : tensor<64x2097152xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<64x2097152xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [64, 2097152], strides = [1, 1] : tensor<64x2097152xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x2097152xf32>>
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_37 {
    flow.executable.export public @prefill_dispatch_37_gather_8xDx1024x2048xf32_dispatch_tensor_store workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_37_gather_8xDx1024x2048xf32_dispatch_tensor_store(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x1024x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xi32>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?x1024x2048xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xi32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?x1024x2048xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 1024, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x1024x2048xf32>> -> tensor<64x1024x2048xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, %0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xi32>>{%0} -> tensor<8x?xi32>
        %5 = tensor.empty(%0) : tensor<8x?x1024x2048xf32>
        %6 = iree_linalg_ext.gather dimension_map = [0] ins(%3, %4 : tensor<64x1024x2048xf32>, tensor<8x?xi32>) outs(%5 : tensor<8x?x1024x2048xf32>) -> tensor<8x?x1024x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [8, %0, 1024, 2048], strides = [1, 1, 1, 1] : tensor<8x?x1024x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?x1024x2048xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_38 {
    flow.executable.export public @prefill_dispatch_38_elementwise_broadcast_8xD_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_38_elementwise_broadcast_8xD_f32(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [%0], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%0} -> tensor<?xf32>
        %4 = tensor.empty(%0) : tensor<8x?xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<?xf32>) outs(%4 : tensor<8x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<8x?xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %2, offsets = [0, 0], sizes = [8, %0], strides = [1, 1] : tensor<8x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_39 {
    flow.executable.export public @prefill_dispatch_39_batch_matmul_Dx1024x1x2048_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_39_batch_matmul_Dx1024x1x2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x1xf32>>, %arg2: index, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024x1xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024x2048xf32>>{%0}
        %3 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x1xf32>>{%1}
        %4 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024x1xf32>>{%1}
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%0, 1024, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024x2048xf32>>{%0} -> tensor<?x1024x2048xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [%1, 2048, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x1xf32>>{%1} -> tensor<?x2048x1xf32>
        %7 = tensor.empty(%1) : tensor<?x1024x1xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x1024x1xf32>) -> tensor<?x1024x1xf32>
        %9 = linalg.batch_matmul ins(%5, %6 : tensor<?x1024x2048xf32>, tensor<?x2048x1xf32>) outs(%8 : tensor<?x1024x1xf32>) -> tensor<?x1024x1xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0, 0, 0], sizes = [%1, 1024, 1], strides = [1, 1, 1] : tensor<?x1024x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024x1xf32>>{%1}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_44 {
    flow.executable.export public @prefill_dispatch_44_gather_8xDx2048x1024xf32_dispatch_tensor_store workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_44_gather_8xDx2048x1024xf32_dispatch_tensor_store(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x2048x1024xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xi32>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?x2048x1024xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xi32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?x2048x1024xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 2048, 1024], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x2048x1024xf32>> -> tensor<64x2048x1024xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, %0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xi32>>{%0} -> tensor<8x?xi32>
        %5 = tensor.empty(%0) : tensor<8x?x2048x1024xf32>
        %6 = iree_linalg_ext.gather dimension_map = [0] ins(%3, %4 : tensor<64x2048x1024xf32>, tensor<8x?xi32>) outs(%5 : tensor<8x?x2048x1024xf32>) -> tensor<8x?x2048x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [8, %0, 2048, 1024], strides = [1, 1, 1, 1] : tensor<8x?x2048x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x?x2048x1024xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_45 {
    flow.executable.export public @prefill_dispatch_45_elementwise_D_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_45_elementwise_D_f32(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
        %cst = arith.constant 1.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%0}
        %3 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [%0], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%0} -> tensor<?xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [%0], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%0} -> tensor<?xf32>
        %6 = tensor.empty(%0) : tensor<?xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<?xf32>, tensor<?xf32>) outs(%6 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.negf %in : f32
          %9 = math.exp %8 : f32
          %10 = arith.addf %9, %cst : f32
          %11 = arith.divf %cst, %10 : f32
          %12 = arith.mulf %in, %11 : f32
          %13 = arith.mulf %12, %in_0 : f32
          linalg.yield %13 : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %3, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_46 {
    flow.executable.export public @prefill_dispatch_46_batch_matmul_Dx2048x1x1024_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_46_batch_matmul_Dx2048x1x1024_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x1024xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024x1xf32>>, %arg2: index, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048x1xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x1024xf32>>{%0}
        %3 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024x1xf32>>{%1}
        %4 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048x1xf32>>{%1}
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%0, 2048, 1024], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x1024xf32>>{%0} -> tensor<?x2048x1024xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [%1, 1024, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024x1xf32>>{%1} -> tensor<?x1024x1xf32>
        %7 = tensor.empty(%1) : tensor<?x2048x1xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x2048x1xf32>) -> tensor<?x2048x1xf32>
        %9 = linalg.batch_matmul ins(%5, %6 : tensor<?x2048x1024xf32>, tensor<?x1024x1xf32>) outs(%8 : tensor<?x2048x1xf32>) -> tensor<?x2048x1xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0, 0, 0], sizes = [%1, 2048, 1], strides = [1, 1, 1] : tensor<?x2048x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048x1xf32>>{%1}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_47 {
    flow.executable.export public @prefill_dispatch_47_matvec_like_Dx2048x8_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_47_matvec_like_Dx2048x8_f32(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?x2048xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?x2048xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xf32>>{%0}
        %3 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [8, %0, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?x2048xf32>>{%0} -> tensor<8x?x2048xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [8, %0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xf32>>{%0} -> tensor<8x?xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %8 = tensor.empty(%0) : tensor<?x2048xf32>
        %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0, d1)>, affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %6 : tensor<8x?x2048xf32>, tensor<8x?xf32>) outs(%9 : tensor<?x2048xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %12 = arith.mulf %in, %in_0 : f32
          %13 = arith.addf %12, %out : f32
          linalg.yield %13 : f32
        } -> tensor<?x2048xf32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7, %10 : tensor<?x2048xf32>, tensor<?x2048xf32>) outs(%8 : tensor<?x2048xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %12 = arith.addf %in, %in_0 : f32
          linalg.yield %12 : f32
        } -> tensor<?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %11, %4, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : tensor<?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_48 {
    flow.executable.export public @prefill_dispatch_48_elementwise_broadcast_2048_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_48_elementwise_broadcast_2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi64>>, %arg1: index, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>, %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x128xf32>>, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf32>>) {
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 2 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg6, 3 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg7, 4 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg8, 5 : index
        %4 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi64>>{%0, %1}
        %5 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x128xf32>>{%2, %3}
        %6 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %7 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %8 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi64>>{%0, %1} -> tensor<?x?xi64>
        %9 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%2, %3, 16, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x128xf32>>{%2, %3} -> tensor<?x?x16x128xf32>
        %11 = tensor.empty() : tensor<2048xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%11 : tensor<2048xf32>) {
        ^bb0(%out: f32):
          %13 = linalg.index 0 : index
          %14 = arith.remsi %13, %c128 : index
          %15 = arith.divsi %13, %c128 : index
          %extracted = tensor.extract %8[%6, %7] : tensor<?x?xi64>
          %16 = arith.trunci %extracted : i64 to i32
          %17 = arith.sitofp %16 : i32 to f32
          %18 = arith.cmpi slt, %14, %c64 : index
          %19 = scf.if %18 -> (f32) {
            %extracted_0 = tensor.extract %9[%14] : tensor<64xf32>
            %20 = arith.mulf %17, %extracted_0 : f32
            %21 = math.cos %20 : f32
            %22 = math.sin %20 : f32
            %extracted_1 = tensor.extract %10[%6, %7, %15, %14] : tensor<?x?x16x128xf32>
            %23 = arith.addi %14, %c64 : index
            %extracted_2 = tensor.extract %10[%6, %7, %15, %23] : tensor<?x?x16x128xf32>
            %24 = arith.mulf %extracted_1, %21 : f32
            %25 = arith.mulf %extracted_2, %22 : f32
            %26 = arith.subf %24, %25 : f32
            scf.yield %26 : f32
          } else {
            %20 = arith.subi %14, %c64 : index
            %extracted_0 = tensor.extract %9[%20] : tensor<64xf32>
            %21 = arith.mulf %17, %extracted_0 : f32
            %22 = math.cos %21 : f32
            %23 = math.sin %21 : f32
            %extracted_1 = tensor.extract %10[%6, %7, %15, %14] : tensor<?x?x16x128xf32>
            %extracted_2 = tensor.extract %10[%6, %7, %15, %20] : tensor<?x?x16x128xf32>
            %24 = arith.mulf %extracted_2, %23 : f32
            %25 = arith.mulf %extracted_1, %22 : f32
            %26 = arith.addf %24, %25 : f32
            scf.yield %26 : f32
          }
          linalg.yield %19 : f32
        } -> tensor<2048xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %arg9, offsets = [0], sizes = [2048], strides = [1] : tensor<2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf32>>
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_49 {
    flow.executable.export public @prefill_dispatch_49_slow_memcpy workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_49_slow_memcpy(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x128xf32>>, %arg6: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index, %arg14: index) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg8, 5 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg9, 6 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg10, 7 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg11, 8 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg12, 9 : index
        %5 = iree_tensor_ext.dispatch.workload.ordinal %arg13, 10 : index
        %6 = iree_tensor_ext.dispatch.workload.ordinal %arg14, 11 : index
        %7 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%0, %1, %2}
        %8 = flow.dispatch.tie_shape %arg6 : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%3, %4, %5, %6}
        %9 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %10 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %11 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 2 : index
        %12 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 3 : index
        %13 = iree_tensor_ext.dispatch.workload.ordinal %arg7, 4 : index
        %14 = iree_tensor_ext.dispatch.tensor.load %arg5, offsets = [0, 0], sizes = [16, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x128xf32>> -> tensor<16x128xf32>
        %15 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [%9, 0, 0], sizes = [1, %10, %2], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%0, %1, %2} -> tensor<?x?xi32>
        %extracted = tensor.extract %15[%11, %12] : tensor<?x?xi32>
        %16 = arith.index_cast %extracted : i32 to index
        iree_tensor_ext.dispatch.tensor.store %14, %8, offsets = [%16, %13, 0, 0], sizes = [1, 1, 16, 128], strides = [1, 1, 1, 1] : tensor<16x128xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%3, %4, %5, %6}
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_50 {
    flow.executable.export public @prefill_dispatch_50_elementwise_broadcast_2048_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_50_elementwise_broadcast_2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf32>>) {
        %c128 = arith.constant 128 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 2 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 3 : index
        %2 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%0, %1}
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%0, %1, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2048xf32>>{%0, %1} -> tensor<?x?x2048xf32>
        %6 = tensor.empty() : tensor<2048xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%6 : tensor<2048xf32>) {
        ^bb0(%out: f32):
          %8 = linalg.index 0 : index
          %9 = arith.remsi %8, %c128 : index
          %10 = arith.divsi %8, %c128 : index
          %11 = arith.muli %10, %c128 : index
          %12 = arith.addi %11, %9 : index
          %extracted = tensor.extract %5[%3, %4, %12] : tensor<?x?x2048xf32>
          linalg.yield %extracted : f32
        } -> tensor<2048xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg5, offsets = [0], sizes = [2048], strides = [1] : tensor<2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf32>>
        return
      }
    }
  }
  flow.executable private @prefill_dispatch_53 {
    flow.executable.export public @prefill_dispatch_53_matmul_Dx50304x2048_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_dispatch_53_matmul_Dx50304x2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x50304xf32>>, %arg2: index, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x50304xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %3 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x50304xf32>>{%1}
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2048, 50304], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x50304xf32>> -> tensor<2048x50304xf32>
        %6 = tensor.empty(%1) : tensor<?x50304xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?x50304xf32>) -> tensor<?x50304xf32>
        %8 = linalg.matmul ins(%4, %5 : tensor<?x2048xf32>, tensor<2048x50304xf32>) outs(%7 : tensor<?x50304xf32>) -> tensor<?x50304xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [%1, 50304], strides = [1, 1] : tensor<?x50304xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x50304xf32>>{%1}
        return
      }
    }
  }
  util.global private @__parameter_model_token_embd_weight_tensor_50304x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"token_embd.weight"> : tensor<50304x2048xf32>
  util.global private @__parameter_model_blk_0_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_1_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_2_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_3_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_4_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_5_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_6_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_7_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_8_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_9_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_10_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_11_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_12_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_13_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_14_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_15_ffn_down_exps_weight_tensor_2048x1024x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.ffn_down_exps.weight"> : tensor<2048x1024x64xf32>
  util.global private @__parameter_model_blk_0_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_1_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_2_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_3_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_4_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_5_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_6_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_7_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_8_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_9_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_10_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_11_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_12_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_13_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_14_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_15_ffn_gate_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.ffn_gate_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_0_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_1_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_2_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_3_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_4_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_5_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_6_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_7_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_8_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_9_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_10_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_11_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_12_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_13_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_14_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_15_ffn_up_exps_weight_tensor_1024x2048x64xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.ffn_up_exps.weight"> : tensor<1024x2048x64xf32>
  util.global private @__parameter_model_blk_0_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_1_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_2_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_3_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_4_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_5_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_6_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_7_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_8_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_9_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_10_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_11_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_12_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_13_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_14_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_15_ffn_gate_inp_weight_tensor_64x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.ffn_gate_inp.weight"> : tensor<64x2048xf32>
  util.global private @__parameter_model_blk_0_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_1_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_2_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_3_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_4_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_5_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_6_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_7_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_8_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_9_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_10_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_11_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_12_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_13_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_14_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_15_attn_k_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.attn_k_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_0_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_1_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_2_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_3_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_4_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_5_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_6_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_7_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_8_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_9_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_10_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_11_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_12_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_13_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_14_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_15_attn_q_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.attn_q_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_0_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_1_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_2_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_3_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_4_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_5_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_6_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_7_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_8_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_9_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_10_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_11_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_12_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_13_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_14_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_15_attn_output_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.attn_output.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_0_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_1_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_2_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_3_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_4_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_5_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_6_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_7_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_8_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_9_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_10_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_11_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_12_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_13_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_14_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_15_attn_v_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.attn_v.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_0_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_1_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_2_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_3_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_4_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_5_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_6_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_7_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_8_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_9_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_10_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_11_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_12_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_13_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_14_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_15_attn_k_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.attn_k.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_0_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_1_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_2_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_3_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_4_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_5_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_6_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_7_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_8_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_9_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_10_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_11_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_12_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_13_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_14_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_15_attn_q_weight_tensor_2048x2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.attn_q.weight"> : tensor<2048x2048xf32>
  util.global private @__parameter_model_blk_0_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_1_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_2_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_3_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_4_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_5_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_6_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_7_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_8_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_9_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_10_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_11_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_12_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_13_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_14_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_15_ffn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.ffn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_0_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.0.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_1_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.1.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_2_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.2.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_3_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.3.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_4_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.4.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_5_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.5.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_6_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.6.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_7_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.7.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_8_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.8.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_9_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.9.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_10_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.10.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_11_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.11.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_12_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.12.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_13_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.13.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_14_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.14.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_blk_15_attn_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"blk.15.attn_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_output_norm_weight_tensor_2048xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"output_norm.weight"> : tensor<2048xf32>
  util.global private @__parameter_model_output_weight_tensor_2048x50304xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@__device_0>} = #flow.parameter.named<"model"::"output.weight"> : tensor<2048x50304xf32>
  util.func public @prefill(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !util.list<?>, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: index) -> (!hal.buffer_view, !util.list<?>) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @prefill(%input0: tensor<?x?xi64>, %input1: tensor<?x?xi64>, %input2: !util.list<?>, %input3: tensor<?x?x?xi32>, %input4: tensor<?xi32>, %input5: index) -> (%output0: tensor<?x?x?xf32>, %output1: !util.list<?>)"}} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %c11_i32 = arith.constant 11 : i32
    %c12_i32 = arith.constant 12 : i32
    %c13_i32 = arith.constant 13 : i32
    %c8192 = arith.constant 8192 : index
    %c2048 = arith.constant 2048 : index
    %c14_i32 = arith.constant 14 : i32
    %c50304 = arith.constant 50304 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %__parameter_model_token_embd_weight_tensor_50304x2048xf32 = util.global.load immutable @__parameter_model_token_embd_weight_tensor_50304x2048xf32 : tensor<50304x2048xf32>
    %__parameter_model_blk_0_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_1_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_2_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_3_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_4_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_5_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_6_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_7_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_8_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_9_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_10_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_11_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_12_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_13_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_14_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_15_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_0_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_1_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_2_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_3_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_4_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_5_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_6_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_7_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_8_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_9_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_10_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_11_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_12_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_13_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_14_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_15_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_0_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_1_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_2_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_3_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_4_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_5_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_6_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_7_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_8_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_9_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_10_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_11_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_12_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_13_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_14_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_15_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_0_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_1_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_2_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_3_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_4_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_5_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_6_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_7_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_8_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_9_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_10_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_11_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_12_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_13_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_14_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_15_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_0_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_1_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_2_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_3_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_4_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_5_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_6_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_7_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_8_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_9_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_10_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_11_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_12_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_13_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_14_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_15_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_0_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_1_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_2_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_3_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_4_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_5_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_6_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_7_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_8_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_9_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_10_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_11_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_12_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_13_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_14_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_15_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_0_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_1_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_2_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_3_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_4_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_5_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_6_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_7_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_8_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_9_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_10_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_11_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_12_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_13_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_14_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_15_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_0_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_1_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_2_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_3_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_4_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_5_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_6_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_7_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_8_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_9_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_10_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_11_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_12_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_13_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_14_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_15_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_0_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_1_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_2_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_3_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_4_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_5_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_6_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_7_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_8_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_9_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_10_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_11_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_12_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_13_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_14_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_15_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_0_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_1_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_2_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_3_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_4_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_5_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_6_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_7_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_8_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_9_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_10_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_11_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_12_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_13_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_14_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_15_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_0_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_1_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_2_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_3_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_4_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_5_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_6_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_7_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_8_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_9_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_10_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_11_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_12_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_13_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_14_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_15_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_0_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_1_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_2_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_3_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_4_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_5_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_6_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_7_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_8_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_9_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_10_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_11_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_12_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_13_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_14_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_15_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_output_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_output_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_output_weight_tensor_2048x50304xf32 = util.global.load immutable @__parameter_model_output_weight_tensor_2048x50304xf32 : tensor<2048x50304xf32>
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %2 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<?x?xi64>{%0, %1}
    %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %5 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<?x?xi64>{%3, %4}
    %6 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[0] : index
    %7 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[1] : index
    %8 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[2] : index
    %9 = hal.tensor.import %arg3 "input3" : !hal.buffer_view -> tensor<?x?x?xi32>{%6, %7, %8}
    %10 = hal.buffer_view.dim<%arg4 : !hal.buffer_view>[0] : index
    %11 = hal.tensor.import %arg4 "input4" : !hal.buffer_view -> tensor<?xi32>{%10}
    %12 = arith.muli %0, %1 : index
    %13 = arith.muli %0, %1 overflow<nsw> : index
    %14 = flow.tensor.reshape %2 : tensor<?x?xi64>{%0, %1} -> tensor<?xi64>{%13}
    %15 = flow.dispatch @prefill_dispatch_0::@prefill_dispatch_0_gather_50304x2048xf32_dispatch_tensor_store[%13, %12](%__parameter_model_token_embd_weight_tensor_50304x2048xf32, %14, %13, %12) : (tensor<50304x2048xf32>, tensor<?xi64>{%13}, index, index) -> tensor<?x2048xf32>{%12}
    %16 = flow.tensor.reshape %15 : tensor<?x2048xf32>{%12} -> tensor<?x?x2048xf32>{%0, %1}
    %17 = scf.for %arg6 = %c0 to %c16 step %c1 iter_args(%arg7 = %16) -> (tensor<?x?x2048xf32>) {
      %23 = arith.index_castui %arg6 : index to i32
      %24 = arith.cmpi eq, %23, %c0_i32 : i32
      %25:12 = scf.if %24 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
        scf.yield %__parameter_model_blk_0_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_0_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_0_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_0_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_0_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_0_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_0_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_0_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_0_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_0_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_0_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_0_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
      } else {
        %113 = arith.cmpi eq, %23, %c1_i32 : i32
        %114:12 = scf.if %113 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
          scf.yield %__parameter_model_blk_1_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_1_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_1_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_1_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_1_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_1_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_1_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_1_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_1_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_1_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_1_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_1_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
        } else {
          %115 = arith.cmpi eq, %23, %c2_i32 : i32
          %116:12 = scf.if %115 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
            scf.yield %__parameter_model_blk_2_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_2_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_2_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_2_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_2_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_2_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_2_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_2_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_2_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_2_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_2_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_2_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
          } else {
            %117 = arith.cmpi eq, %23, %c3_i32 : i32
            %118:12 = scf.if %117 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
              scf.yield %__parameter_model_blk_3_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_3_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_3_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_3_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_3_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_3_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_3_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_3_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_3_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_3_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_3_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_3_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
            } else {
              %119 = arith.cmpi eq, %23, %c4_i32 : i32
              %120:12 = scf.if %119 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                scf.yield %__parameter_model_blk_4_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_4_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_4_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_4_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_4_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_4_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_4_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_4_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_4_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_4_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_4_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_4_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
              } else {
                %121 = arith.cmpi eq, %23, %c5_i32 : i32
                %122:12 = scf.if %121 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                  scf.yield %__parameter_model_blk_5_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_5_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_5_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_5_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_5_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_5_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_5_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_5_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_5_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_5_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_5_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_5_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                } else {
                  %123 = arith.cmpi eq, %23, %c6_i32 : i32
                  %124:12 = scf.if %123 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                    scf.yield %__parameter_model_blk_6_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_6_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_6_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_6_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_6_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_6_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_6_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_6_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_6_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_6_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_6_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_6_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                  } else {
                    %125 = arith.cmpi eq, %23, %c7_i32 : i32
                    %126:12 = scf.if %125 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                      scf.yield %__parameter_model_blk_7_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_7_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_7_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_7_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_7_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_7_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_7_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_7_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_7_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_7_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_7_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_7_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                    } else {
                      %127 = arith.cmpi eq, %23, %c8_i32 : i32
                      %128:12 = scf.if %127 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                        scf.yield %__parameter_model_blk_8_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_8_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_8_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_8_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_8_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_8_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_8_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_8_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_8_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_8_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_8_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_8_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                      } else {
                        %129 = arith.cmpi eq, %23, %c9_i32 : i32
                        %130:12 = scf.if %129 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                          scf.yield %__parameter_model_blk_9_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_9_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_9_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_9_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_9_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_9_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_9_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_9_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_9_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_9_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_9_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_9_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                        } else {
                          %131 = arith.cmpi eq, %23, %c10_i32 : i32
                          %132:12 = scf.if %131 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                            scf.yield %__parameter_model_blk_10_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_10_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_10_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_10_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_10_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_10_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_10_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_10_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_10_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_10_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_10_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_10_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                          } else {
                            %133 = arith.cmpi eq, %23, %c11_i32 : i32
                            %134:12 = scf.if %133 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                              scf.yield %__parameter_model_blk_11_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_11_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_11_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_11_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_11_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_11_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_11_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_11_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_11_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_11_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_11_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_11_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                            } else {
                              %135 = arith.cmpi eq, %23, %c12_i32 : i32
                              %136:12 = scf.if %135 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                                scf.yield %__parameter_model_blk_12_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_12_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_12_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_12_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_12_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_12_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_12_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_12_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_12_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_12_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_12_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_12_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                              } else {
                                %137 = arith.cmpi eq, %23, %c13_i32 : i32
                                %138:12 = scf.if %137 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                                  scf.yield %__parameter_model_blk_13_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_13_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_13_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_13_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_13_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_13_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_13_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_13_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_13_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_13_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_13_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_13_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                                } else {
                                  %139 = arith.cmpi eq, %23, %c14_i32 : i32
                                  %140 = flow.dispatch @prefill_dispatch_1::@prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%139, %__parameter_model_blk_14_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_15_attn_norm_weight_tensor_2048xf32) : (i1, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
                                  %141 = flow.dispatch @prefill_dispatch_1::@prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%139, %__parameter_model_blk_14_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_15_ffn_norm_weight_tensor_2048xf32) : (i1, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
                                  %142 = flow.tensor.reshape %__parameter_model_blk_14_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %143 = flow.tensor.reshape %__parameter_model_blk_15_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %144 = flow.dispatch @prefill_dispatch_3::@prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%139, %142, %143) : (i1, tensor<4194304xf32>, tensor<4194304xf32>) -> tensor<4194304xf32>
                                  %145 = flow.tensor.reshape %144 : tensor<4194304xf32> -> tensor<2048x2048xf32>
                                  %146 = flow.tensor.reshape %__parameter_model_blk_14_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %147 = flow.tensor.reshape %__parameter_model_blk_15_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %148 = flow.dispatch @prefill_dispatch_3::@prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%139, %146, %147) : (i1, tensor<4194304xf32>, tensor<4194304xf32>) -> tensor<4194304xf32>
                                  %149 = flow.tensor.reshape %148 : tensor<4194304xf32> -> tensor<2048x2048xf32>
                                  %150 = flow.tensor.reshape %__parameter_model_blk_14_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %151 = flow.tensor.reshape %__parameter_model_blk_15_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %152 = flow.dispatch @prefill_dispatch_3::@prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%139, %150, %151) : (i1, tensor<4194304xf32>, tensor<4194304xf32>) -> tensor<4194304xf32>
                                  %153 = flow.tensor.reshape %152 : tensor<4194304xf32> -> tensor<2048x2048xf32>
                                  %154 = flow.tensor.reshape %__parameter_model_blk_14_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %155 = flow.tensor.reshape %__parameter_model_blk_15_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %156 = flow.dispatch @prefill_dispatch_3::@prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%139, %154, %155) : (i1, tensor<4194304xf32>, tensor<4194304xf32>) -> tensor<4194304xf32>
                                  %157 = flow.tensor.reshape %156 : tensor<4194304xf32> -> tensor<2048x2048xf32>
                                  %158 = flow.dispatch @prefill_dispatch_1::@prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%139, %__parameter_model_blk_14_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_15_attn_q_norm_weight_tensor_2048xf32) : (i1, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
                                  %159 = flow.dispatch @prefill_dispatch_1::@prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%139, %__parameter_model_blk_14_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_15_attn_k_norm_weight_tensor_2048xf32) : (i1, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
                                  %160 = flow.tensor.reshape %__parameter_model_blk_14_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32> -> tensor<131072xf32>
                                  %161 = flow.tensor.reshape %__parameter_model_blk_15_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32> -> tensor<131072xf32>
                                  %162 = flow.dispatch @prefill_dispatch_9::@prefill_dispatch_9_elementwise_131072_i1xf32xf32xf32(%139, %160, %161) : (i1, tensor<131072xf32>, tensor<131072xf32>) -> tensor<131072xf32>
                                  %163 = flow.tensor.reshape %162 : tensor<131072xf32> -> tensor<64x2048xf32>
                                  %164 = flow.tensor.reshape %__parameter_model_blk_14_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32> -> tensor<134217728xf32>
                                  %165 = flow.tensor.reshape %__parameter_model_blk_15_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32> -> tensor<134217728xf32>
                                  %166 = flow.dispatch @prefill_dispatch_10::@prefill_dispatch_10_elementwise_134217728_i1xf32xf32xf32(%139, %164, %165) : (i1, tensor<134217728xf32>, tensor<134217728xf32>) -> tensor<134217728xf32>
                                  %167 = flow.tensor.reshape %166 : tensor<134217728xf32> -> tensor<1024x2048x64xf32>
                                  %168 = flow.tensor.reshape %__parameter_model_blk_14_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32> -> tensor<134217728xf32>
                                  %169 = flow.tensor.reshape %__parameter_model_blk_15_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32> -> tensor<134217728xf32>
                                  %170 = flow.dispatch @prefill_dispatch_10::@prefill_dispatch_10_elementwise_134217728_i1xf32xf32xf32(%139, %168, %169) : (i1, tensor<134217728xf32>, tensor<134217728xf32>) -> tensor<134217728xf32>
                                  %171 = flow.tensor.reshape %170 : tensor<134217728xf32> -> tensor<1024x2048x64xf32>
                                  %172 = flow.tensor.reshape %__parameter_model_blk_14_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32> -> tensor<134217728xf32>
                                  %173 = flow.tensor.reshape %__parameter_model_blk_15_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32> -> tensor<134217728xf32>
                                  %174 = flow.dispatch @prefill_dispatch_10::@prefill_dispatch_10_elementwise_134217728_i1xf32xf32xf32(%139, %172, %173) : (i1, tensor<134217728xf32>, tensor<134217728xf32>) -> tensor<134217728xf32>
                                  %175 = flow.tensor.reshape %174 : tensor<134217728xf32> -> tensor<2048x1024x64xf32>
                                  scf.yield %140, %141, %145, %149, %153, %157, %158, %159, %163, %167, %171, %175 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                                }
                                scf.yield %138#0, %138#1, %138#2, %138#3, %138#4, %138#5, %138#6, %138#7, %138#8, %138#9, %138#10, %138#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                              }
                              scf.yield %136#0, %136#1, %136#2, %136#3, %136#4, %136#5, %136#6, %136#7, %136#8, %136#9, %136#10, %136#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                            }
                            scf.yield %134#0, %134#1, %134#2, %134#3, %134#4, %134#5, %134#6, %134#7, %134#8, %134#9, %134#10, %134#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                          }
                          scf.yield %132#0, %132#1, %132#2, %132#3, %132#4, %132#5, %132#6, %132#7, %132#8, %132#9, %132#10, %132#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                        }
                        scf.yield %130#0, %130#1, %130#2, %130#3, %130#4, %130#5, %130#6, %130#7, %130#8, %130#9, %130#10, %130#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                      }
                      scf.yield %128#0, %128#1, %128#2, %128#3, %128#4, %128#5, %128#6, %128#7, %128#8, %128#9, %128#10, %128#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                    }
                    scf.yield %126#0, %126#1, %126#2, %126#3, %126#4, %126#5, %126#6, %126#7, %126#8, %126#9, %126#10, %126#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                  }
                  scf.yield %124#0, %124#1, %124#2, %124#3, %124#4, %124#5, %124#6, %124#7, %124#8, %124#9, %124#10, %124#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                }
                scf.yield %122#0, %122#1, %122#2, %122#3, %122#4, %122#5, %122#6, %122#7, %122#8, %122#9, %122#10, %122#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
              }
              scf.yield %120#0, %120#1, %120#2, %120#3, %120#4, %120#5, %120#6, %120#7, %120#8, %120#9, %120#10, %120#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
            }
            scf.yield %118#0, %118#1, %118#2, %118#3, %118#4, %118#5, %118#6, %118#7, %118#8, %118#9, %118#10, %118#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
          }
          scf.yield %116#0, %116#1, %116#2, %116#3, %116#4, %116#5, %116#6, %116#7, %116#8, %116#9, %116#10, %116#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
        }
        scf.yield %114#0, %114#1, %114#2, %114#3, %114#4, %114#5, %114#6, %114#7, %114#8, %114#9, %114#10, %114#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
      }
      %26 = flow.tensor.reshape %arg7 : tensor<?x?x2048xf32>{%0, %1} -> tensor<?x2048xf32>{%13}
      %27 = flow.dispatch @prefill_dispatch_13::@prefill_dispatch_13_reduction_Dx2048_f32[%13](%26, %25#0, %13) : (tensor<?x2048xf32>{%13}, tensor<2048xf32>, index) -> tensor<?x2048xf32>{%13}
      %28 = flow.tensor.reshape %27 : tensor<?x2048xf32>{%13} -> tensor<?x?x2048xf32>{%0, %1}
      %29 = flow.tensor.reshape %25#2 : tensor<2048x2048xf32> -> tensor<4194304xf32>
      %30 = flow.dispatch @prefill_dispatch_14::@prefill_dispatch_14_elementwise_broadcast_Dx4194304_f32[%0](%29, %0) : (tensor<4194304xf32>, index) -> tensor<?x4194304xf32>{%0}
      %31 = flow.tensor.reshape %30 : tensor<?x4194304xf32>{%0} -> tensor<?x2048x2048xf32>{%0}
      %32 = flow.tensor.reshape %25#3 : tensor<2048x2048xf32> -> tensor<4194304xf32>
      %33 = flow.dispatch @prefill_dispatch_14::@prefill_dispatch_14_elementwise_broadcast_Dx4194304_f32[%0](%32, %0) : (tensor<4194304xf32>, index) -> tensor<?x4194304xf32>{%0}
      %34 = flow.tensor.reshape %33 : tensor<?x4194304xf32>{%0} -> tensor<?x2048x2048xf32>{%0}
      %35 = flow.tensor.reshape %25#4 : tensor<2048x2048xf32> -> tensor<4194304xf32>
      %36 = flow.dispatch @prefill_dispatch_14::@prefill_dispatch_14_elementwise_broadcast_Dx4194304_f32[%0](%35, %0) : (tensor<4194304xf32>, index) -> tensor<?x4194304xf32>{%0}
      %37 = flow.tensor.reshape %36 : tensor<?x4194304xf32>{%0} -> tensor<?x2048x2048xf32>{%0}
      %38 = flow.dispatch @prefill_dispatch_17::@prefill_dispatch_17_batch_matmul_DxDx2048x2048_f32[%0, %1](%28, %31, %0, %1) : (tensor<?x?x2048xf32>{%0, %1}, tensor<?x2048x2048xf32>{%0}, index, index) -> tensor<?x?x2048xf32>{%0, %1}
      %39 = flow.dispatch @prefill_dispatch_17::@prefill_dispatch_17_batch_matmul_DxDx2048x2048_f32[%0, %1](%28, %34, %0, %1) : (tensor<?x?x2048xf32>{%0, %1}, tensor<?x2048x2048xf32>{%0}, index, index) -> tensor<?x?x2048xf32>{%0, %1}
      %40 = flow.dispatch @prefill_dispatch_17::@prefill_dispatch_17_batch_matmul_DxDx2048x2048_f32[%0, %1](%28, %37, %0, %1) : (tensor<?x?x2048xf32>{%0, %1}, tensor<?x2048x2048xf32>{%0}, index, index) -> tensor<?x?x2048xf32>{%0, %1}
      %41 = flow.tensor.reshape %38 : tensor<?x?x2048xf32>{%0, %1} -> tensor<?x2048xf32>{%13}
      %42 = flow.dispatch @prefill_dispatch_13::@prefill_dispatch_13_reduction_Dx2048_f32[%13](%41, %25#6, %13) : (tensor<?x2048xf32>{%13}, tensor<2048xf32>, index) -> tensor<?x2048xf32>{%13}
      %43 = flow.tensor.reshape %42 : tensor<?x2048xf32>{%13} -> tensor<?x?x2048xf32>{%0, %1}
      %44 = flow.tensor.reshape %39 : tensor<?x?x2048xf32>{%0, %1} -> tensor<?x2048xf32>{%13}
      %45 = flow.dispatch @prefill_dispatch_13::@prefill_dispatch_13_reduction_Dx2048_f32[%13](%44, %25#7, %13) : (tensor<?x2048xf32>{%13}, tensor<2048xf32>, index) -> tensor<?x2048xf32>{%13}
      %46 = flow.tensor.reshape %45 : tensor<?x2048xf32>{%13} -> tensor<?x?x2048xf32>{%0, %1}
      %47 = arith.muli %13, %c2048 overflow<nsw> : index
      %48 = flow.dispatch @prefill_dispatch_22::@prefill_dispatch_22_elementwise_broadcast_D_f32[%47, %1, %0](%47, %1, %43, %0) : (index, index, tensor<?x?x2048xf32>{%0, %1}, index) -> tensor<?xf32>{%47}
      %49 = flow.tensor.reshape %48 : tensor<?xf32>{%47} -> tensor<?x?x16x128xf32>{%0, %1}
      %50 = flow.dispatch @prefill_dispatch_22::@prefill_dispatch_22_elementwise_broadcast_D_f32[%47, %1, %0](%47, %1, %46, %0) : (index, index, tensor<?x?x2048xf32>{%0, %1}, index) -> tensor<?xf32>{%47}
      %51 = flow.tensor.reshape %50 : tensor<?xf32>{%47} -> tensor<?x?x16x128xf32>{%0, %1}
      %52 = flow.dispatch @prefill_dispatch_24::@prefill_dispatch_24_elementwise_broadcast_64_f32() : () -> tensor<64xf32>
      %53 = flow.dispatch @prefill_dispatch_25::@prefill_dispatch_25_elementwise_broadcast_D_f32[%47, %1, %3, %4, %0](%47, %1, %5, %52, %49, %3, %4, %0) : (index, index, tensor<?x?xi64>{%3, %4}, tensor<64xf32>, tensor<?x?x16x128xf32>{%0, %1}, index, index, index) -> tensor<?xf32>{%47}
      %54 = flow.dispatch @prefill_dispatch_25::@prefill_dispatch_25_elementwise_broadcast_D_f32[%47, %1, %3, %4, %0](%47, %1, %5, %52, %51, %3, %4, %0) : (index, index, tensor<?x?xi64>{%3, %4}, tensor<64xf32>, tensor<?x?x16x128xf32>{%0, %1}, index, index, index) -> tensor<?xf32>{%47}
      %55 = flow.dispatch @prefill_dispatch_27::@prefill_dispatch_27_elementwise_broadcast_D_f32[%47, %0, %1](%47, %40, %0, %1) : (index, tensor<?x?x2048xf32>{%0, %1}, index, index) -> tensor<?xf32>{%47}
      %56 = arith.muli %0, %c16 overflow<nsw> : index
      %57 = flow.tensor.reshape %53 : tensor<?xf32>{%47} -> tensor<?x?x128xf32>{%56, %1}
      %58 = flow.tensor.reshape %54 : tensor<?xf32>{%47} -> tensor<?x?x128xf32>{%56, %1}
      %59 = flow.tensor.reshape %55 : tensor<?xf32>{%47} -> tensor<?x128x?xf32>{%56, %1}
      %60 = flow.dispatch @prefill_dispatch_28::@prefill_dispatch_28_attention_DxDx128xDx128[%56, %1](%56, %57, %58, %59, %1) : (index, tensor<?x?x128xf32>{%56, %1}, tensor<?x?x128xf32>{%56, %1}, tensor<?x128x?xf32>{%56, %1}, index) -> tensor<?x?x128xf32>{%56, %1}
      %61 = flow.dispatch @prefill_dispatch_29::@prefill_dispatch_29_elementwise_broadcast_D_f32[%47, %56, %1](%47, %60, %56, %1) : (index, tensor<?x?x128xf32>{%56, %1}, index, index) -> tensor<?xf32>{%47}
      %62 = flow.tensor.reshape %61 : tensor<?xf32>{%47} -> tensor<?x?x2048xf32>{%0, %1}
      %63 = flow.tensor.reshape %25#5 : tensor<2048x2048xf32> -> tensor<4194304xf32>
      %64 = flow.dispatch @prefill_dispatch_14::@prefill_dispatch_14_elementwise_broadcast_Dx4194304_f32[%0](%63, %0) : (tensor<4194304xf32>, index) -> tensor<?x4194304xf32>{%0}
      %65 = flow.tensor.reshape %64 : tensor<?x4194304xf32>{%0} -> tensor<?x2048x2048xf32>{%0}
      %66 = flow.dispatch @prefill_dispatch_31::@prefill_dispatch_31_batch_matmul_DxDx2048x2048_f32[%0, %1](%62, %65, %arg7, %0, %1) : (tensor<?x?x2048xf32>{%0, %1}, tensor<?x2048x2048xf32>{%0}, tensor<?x?x2048xf32>{%0, %1}, index, index) -> tensor<?x?x2048xf32>{%0, %1}
      %67 = flow.tensor.reshape %66 : tensor<?x?x2048xf32>{%0, %1} -> tensor<?x2048xf32>{%13}
      %68 = flow.dispatch @prefill_dispatch_13::@prefill_dispatch_13_reduction_Dx2048_f32[%13](%67, %25#1, %13) : (tensor<?x2048xf32>{%13}, tensor<2048xf32>, index) -> tensor<?x2048xf32>{%13}
      %69 = flow.dispatch @prefill_dispatch_33::@prefill_dispatch_33_matmul_64xDx2048_f32[%13](%25#8, %68, %13) : (tensor<64x2048xf32>, tensor<?x2048xf32>{%13}, index) -> tensor<64x?xf32>{%13}
      %70 = flow.dispatch @prefill_dispatch_34::@prefill_dispatch_34_softmax_64xDxf32_dispatch_tensor_store[%13](%69, %13) : (tensor<64x?xf32>{%13}, index) -> tensor<64x?xf32>{%13}
      %71:2 = flow.dispatch @prefill_dispatch_35::@prefill_dispatch_35_topk_64xDxf32[%13](%70, %13) : (tensor<64x?xf32>{%13}, index) -> (tensor<8x?xf32>{%13}, tensor<8x?xi32>{%13})
      %72 = flow.tensor.reshape %25#9 : tensor<1024x2048x64xf32> -> tensor<2097152x64xf32>
      %73 = flow.dispatch @prefill_dispatch_36::@prefill_dispatch_36_transpose_2097152x64_f32(%72) : (tensor<2097152x64xf32>) -> tensor<64x2097152xf32>
      %74 = flow.tensor.reshape %73 : tensor<64x2097152xf32> -> tensor<64x1024x2048xf32>
      %75 = arith.muli %13, %c8 : index
      %76 = flow.dispatch @prefill_dispatch_37::@prefill_dispatch_37_gather_8xDx1024x2048xf32_dispatch_tensor_store[%13](%74, %71#1, %13) : (tensor<64x1024x2048xf32>, tensor<8x?xi32>{%13}, index) -> tensor<8x?x1024x2048xf32>{%13}
      %77 = arith.muli %13, %c8 overflow<nsw> : index
      %78 = flow.tensor.reshape %76 : tensor<8x?x1024x2048xf32>{%13} -> tensor<?x1024x2048xf32>{%77}
      %79 = flow.tensor.reshape %68 : tensor<?x2048xf32>{%13} -> tensor<?xf32>{%47}
      %80 = flow.dispatch @prefill_dispatch_38::@prefill_dispatch_38_elementwise_broadcast_8xD_f32[%47](%47, %79) : (index, tensor<?xf32>{%47}) -> tensor<8x?xf32>{%47}
      %81 = flow.tensor.reshape %80 : tensor<8x?xf32>{%47} -> tensor<?x2048x1xf32>{%75}
      %82 = flow.dispatch @prefill_dispatch_39::@prefill_dispatch_39_batch_matmul_Dx1024x1x2048_f32[%77, %75](%78, %81, %77, %75) : (tensor<?x1024x2048xf32>{%77}, tensor<?x2048x1xf32>{%75}, index, index) -> tensor<?x1024x1xf32>{%75}
      %83 = flow.tensor.reshape %25#10 : tensor<1024x2048x64xf32> -> tensor<2097152x64xf32>
      %84 = flow.dispatch @prefill_dispatch_36::@prefill_dispatch_36_transpose_2097152x64_f32(%83) : (tensor<2097152x64xf32>) -> tensor<64x2097152xf32>
      %85 = flow.tensor.reshape %84 : tensor<64x2097152xf32> -> tensor<64x1024x2048xf32>
      %86 = flow.dispatch @prefill_dispatch_37::@prefill_dispatch_37_gather_8xDx1024x2048xf32_dispatch_tensor_store[%13](%85, %71#1, %13) : (tensor<64x1024x2048xf32>, tensor<8x?xi32>{%13}, index) -> tensor<8x?x1024x2048xf32>{%13}
      %87 = flow.tensor.reshape %86 : tensor<8x?x1024x2048xf32>{%13} -> tensor<?x1024x2048xf32>{%77}
      %88 = flow.dispatch @prefill_dispatch_39::@prefill_dispatch_39_batch_matmul_Dx1024x1x2048_f32[%77, %75](%87, %81, %77, %75) : (tensor<?x1024x2048xf32>{%77}, tensor<?x2048x1xf32>{%75}, index, index) -> tensor<?x1024x1xf32>{%75}
      %89 = flow.tensor.reshape %25#11 : tensor<2048x1024x64xf32> -> tensor<2097152x64xf32>
      %90 = flow.dispatch @prefill_dispatch_36::@prefill_dispatch_36_transpose_2097152x64_f32(%89) : (tensor<2097152x64xf32>) -> tensor<64x2097152xf32>
      %91 = flow.tensor.reshape %90 : tensor<64x2097152xf32> -> tensor<64x2048x1024xf32>
      %92 = flow.dispatch @prefill_dispatch_44::@prefill_dispatch_44_gather_8xDx2048x1024xf32_dispatch_tensor_store[%13](%91, %71#1, %13) : (tensor<64x2048x1024xf32>, tensor<8x?xi32>{%13}, index) -> tensor<8x?x2048x1024xf32>{%13}
      %93 = flow.tensor.reshape %92 : tensor<8x?x2048x1024xf32>{%13} -> tensor<?x2048x1024xf32>{%77}
      %94 = arith.muli %13, %c8192 overflow<nsw> : index
      %95 = flow.tensor.reshape %88 : tensor<?x1024x1xf32>{%75} -> tensor<?xf32>{%94}
      %96 = flow.tensor.reshape %82 : tensor<?x1024x1xf32>{%75} -> tensor<?xf32>{%94}
      %97 = flow.dispatch @prefill_dispatch_45::@prefill_dispatch_45_elementwise_D_f32[%94](%94, %95, %96) : (index, tensor<?xf32>{%94}, tensor<?xf32>{%94}) -> tensor<?xf32>{%94}
      %98 = flow.tensor.reshape %97 : tensor<?xf32>{%94} -> tensor<?x1024x1xf32>{%75}
      %99 = flow.dispatch @prefill_dispatch_46::@prefill_dispatch_46_batch_matmul_Dx2048x1x1024_f32[%77, %75](%93, %98, %77, %75) : (tensor<?x2048x1024xf32>{%77}, tensor<?x1024x1xf32>{%75}, index, index) -> tensor<?x2048x1xf32>{%75}
      %100 = flow.tensor.reshape %99 : tensor<?x2048x1xf32>{%75} -> tensor<8x?x2048xf32>{%13}
      %101 = flow.dispatch @prefill_dispatch_47::@prefill_dispatch_47_matvec_like_Dx2048x8_f32[%13](%13, %100, %71#0, %67) : (index, tensor<8x?x2048xf32>{%13}, tensor<8x?xf32>{%13}, tensor<?x2048xf32>{%13}) -> tensor<?x2048xf32>{%13}
      %102 = flow.tensor.reshape %101 : tensor<?x2048xf32>{%13} -> tensor<?x?x2048xf32>{%0, %1}
      %103 = util.list.get %arg2[%c0] : !util.list<?> -> !hal.buffer_view
      %104 = hal.buffer_view.dim<%103 : !hal.buffer_view>[0] : index
      %105 = hal.buffer_view.dim<%103 : !hal.buffer_view>[1] : index
      %106 = hal.tensor.import %103 : !hal.buffer_view -> tensor<?x?x?x?xf32>{%104, %105, %c16, %c128}
      %107 = scf.for %arg8 = %c0 to %12 step %c1 iter_args(%arg9 = %106) -> (tensor<?x?x?x?xf32>) {
        %113 = arith.divui %arg8, %1 : index
        %114 = arith.remui %arg8, %1 : index
        %115 = flow.tensor.load %11[%113] : tensor<?xi32>{%10}
        %116 = arith.index_cast %115 : i32 to index
        %117 = arith.addi %116, %114 : index
        %118 = arith.divui %117, %arg5 : index
        %119 = arith.remui %117, %arg5 : index
        %120 = flow.dispatch @prefill_dispatch_48::@prefill_dispatch_48_elementwise_broadcast_2048_f32[%113, %114, %3, %4, %0, %1](%5, %113, %114, %52, %51, %3, %4, %0, %1) : (tensor<?x?xi64>{%3, %4}, index, index, tensor<64xf32>, tensor<?x?x16x128xf32>{%0, %1}, index, index, index, index) -> tensor<2048xf32>
        %121 = flow.tensor.reshape %120 : tensor<2048xf32> -> tensor<16x128xf32>
        %122 = flow.dispatch @prefill_dispatch_49::@prefill_dispatch_49_slow_memcpy[%arg6, %0, %113, %118, %119, %6, %7, %8, %104, %105, %c16, %c128](%9, %arg6, %0, %113, %118, %121, %arg9, %119, %6, %7, %8, %104, %105, %c16, %c128) : (tensor<?x?x?xi32>{%6, %7, %8}, index, index, index, index, tensor<16x128xf32>, tensor<?x?x?x?xf32>{%104, %105, %c16, %c128}, index, index, index, index, index, index, index, index) -> %arg9{%104, %105, %c16, %c128}
        scf.yield %122 : tensor<?x?x?x?xf32>
      }
      %108 = hal.tensor.export %107 : tensor<?x?x?x?xf32>{%104, %105, %c16, %c128} -> !hal.buffer_view
      %109 = util.list.get %arg2[%c1] : !util.list<?> -> !hal.buffer_view
      %110 = hal.tensor.import %109 : !hal.buffer_view -> tensor<?x?x?x?xf32>{%104, %105, %c16, %c128}
      %111 = scf.for %arg8 = %c0 to %12 step %c1 iter_args(%arg9 = %110) -> (tensor<?x?x?x?xf32>) {
        %113 = arith.divui %arg8, %1 : index
        %114 = arith.remui %arg8, %1 : index
        %115 = flow.tensor.load %11[%113] : tensor<?xi32>{%10}
        %116 = arith.index_cast %115 : i32 to index
        %117 = arith.addi %116, %114 : index
        %118 = arith.divui %117, %arg5 : index
        %119 = arith.remui %117, %arg5 : index
        %120 = flow.dispatch @prefill_dispatch_50::@prefill_dispatch_50_elementwise_broadcast_2048_f32[%113, %114, %0, %1](%40, %113, %114, %0, %1) : (tensor<?x?x2048xf32>{%0, %1}, index, index, index, index) -> tensor<2048xf32>
        %121 = flow.tensor.reshape %120 : tensor<2048xf32> -> tensor<16x128xf32>
        %122 = flow.dispatch @prefill_dispatch_49::@prefill_dispatch_49_slow_memcpy[%arg6, %0, %113, %118, %119, %6, %7, %8, %104, %105, %c16, %c128](%9, %arg6, %0, %113, %118, %121, %arg9, %119, %6, %7, %8, %104, %105, %c16, %c128) : (tensor<?x?x?xi32>{%6, %7, %8}, index, index, index, index, tensor<16x128xf32>, tensor<?x?x?x?xf32>{%104, %105, %c16, %c128}, index, index, index, index, index, index, index, index) -> %arg9{%104, %105, %c16, %c128}
        scf.yield %122 : tensor<?x?x?x?xf32>
      }
      %112 = hal.tensor.export %111 : tensor<?x?x?x?xf32>{%104, %105, %c16, %c128} -> !hal.buffer_view
      util.list.set %arg2[%c0], %108 : !hal.buffer_view -> !util.list<?>
      util.list.set %arg2[%c1], %112 : !hal.buffer_view -> !util.list<?>
      scf.yield %102 : tensor<?x?x2048xf32>
    }
    %18 = flow.tensor.reshape %17 : tensor<?x?x2048xf32>{%0, %1} -> tensor<?x2048xf32>{%13}
    %19 = flow.dispatch @prefill_dispatch_13::@prefill_dispatch_13_reduction_Dx2048_f32[%13](%18, %__parameter_model_output_norm_weight_tensor_2048xf32, %13) : (tensor<?x2048xf32>{%13}, tensor<2048xf32>, index) -> tensor<?x2048xf32>{%13}
    %20 = flow.dispatch @prefill_dispatch_53::@prefill_dispatch_53_matmul_Dx50304x2048_f32[%13, %12](%19, %__parameter_model_output_weight_tensor_2048x50304xf32, %13, %12) : (tensor<?x2048xf32>{%13}, tensor<2048x50304xf32>, index, index) -> tensor<?x50304xf32>{%12}
    %21 = flow.tensor.reshape %20 : tensor<?x50304xf32>{%12} -> tensor<?x?x?xf32>{%0, %1, %c50304}
    %22 = hal.tensor.export %21 "output0" : tensor<?x?x?xf32>{%0, %1, %c50304} -> !hal.buffer_view
    util.return %22, %arg2 : !hal.buffer_view, !util.list<?>
  }
  flow.executable private @decode_dispatch_0 {
    flow.executable.export public @decode_dispatch_0_gather_50304x2048xf32_dispatch_tensor_store workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_0_gather_50304x2048xf32_dispatch_tensor_store(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<50304x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [50304, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<50304x2048xf32>> -> tensor<50304x2048xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [%0], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>{%0} -> tensor<?xi64>
        %5 = tensor.empty(%0) : tensor<?x2048xf32>
        %6 = iree_linalg_ext.gather dimension_map = [0] ins(%3, %4 : tensor<50304x2048xf32>, tensor<?xi64>) outs(%5 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : tensor<?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_13 {
    flow.executable.export public @decode_dispatch_13_elementwise_Dx2048_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_13_elementwise_Dx2048_f32(%arg0: i32, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>>, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>) {
        %cst = arith.constant 9.99999974E-6 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 1 : index
        %2 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg6 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [2048], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>> -> tensor<2048xf32>
        %8 = tensor.empty(%0) : tensor<?x2048xf32>
        %9 = tensor.empty(%0) : tensor<?xf32>
        %10 = arith.sitofp %arg0 : i32 to f32
        %11 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<?xf32>) -> tensor<?xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<?x?xf32>) outs(%11 : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %14 = arith.mulf %in, %in : f32
          %15 = arith.addf %out, %14 : f32
          linalg.yield %15 : f32
        } -> tensor<?xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %12, %7 : tensor<?x2048xf32>, tensor<?xf32>, tensor<2048xf32>) outs(%8 : tensor<?x2048xf32>) {
        ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
          %14 = arith.divf %in_1, %10 : f32
          %15 = arith.addf %14, %cst : f32
          %16 = math.sqrt %15 : f32
          %17 = arith.divf %in, %16 : f32
          %18 = arith.mulf %17, %in_2 : f32
          linalg.yield %18 : f32
        } -> tensor<?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %13, %4, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : tensor<?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_14 {
    flow.executable.export public @decode_dispatch_14_elementwise_broadcast_DxD_i32xf32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_14_elementwise_broadcast_DxD_i32xf32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi32>>, %arg1: index, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index, %arg14: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 2 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg6, 3 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg7, 4 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg8, 5 : index
        %5 = iree_tensor_ext.dispatch.workload.ordinal %arg9, 6 : index
        %6 = iree_tensor_ext.dispatch.workload.ordinal %arg10, 7 : index
        %7 = iree_tensor_ext.dispatch.workload.ordinal %arg11, 8 : index
        %8 = iree_tensor_ext.dispatch.workload.ordinal %arg12, 9 : index
        %9 = iree_tensor_ext.dispatch.workload.ordinal %arg13, 10 : index
        %10 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi32>>{%1, %2}
        %11 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%3, %9, %4}
        %12 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%5, %6, %7, %8}
        %13 = flow.dispatch.tie_shape %arg14 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%9, %0}
        %14 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %15 = iree_tensor_ext.dispatch.tensor.load %12, offsets = [0, 0, 0, 0], sizes = [%5, %6, %7, %8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%5, %6, %7, %8} -> tensor<?x?x?x?xf32>
        %16 = iree_tensor_ext.dispatch.tensor.load %10, offsets = [%14, 0], sizes = [1, %9], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi32>>{%1, %2} -> tensor<?xi32>
        %17 = iree_tensor_ext.dispatch.tensor.load %11, offsets = [%14, 0, 0], sizes = [1, %9, %4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%3, %9, %4} -> tensor<?x?xi32>
        %18 = tensor.empty(%9, %0) : tensor<?x?xf32>
        %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<?xi32>) outs(%18 : tensor<?x?xf32>) {
        ^bb0(%in: i32, %out: f32):
          %20 = linalg.index 0 : index
          %21 = linalg.index 1 : index
          %22 = arith.remsi %21, %8 : index
          %23 = arith.divsi %21, %8 : index
          %24 = arith.remsi %23, %7 : index
          %25 = arith.divsi %23, %7 : index
          %26 = arith.divui %25, %6 : index
          %27 = arith.remui %25, %6 : index
          %extracted = tensor.extract %17[%20, %26] : tensor<?x?xi32>
          %28 = arith.index_cast %extracted : i32 to index
          %extracted_0 = tensor.extract %15[%28, %27, %24, %22] : tensor<?x?x?x?xf32>
          %29 = arith.index_cast %in : i32 to index
          %30 = arith.cmpi ult, %25, %29 : index
          %31 = arith.select %30, %extracted_0, %cst : f32
          linalg.yield %31 : f32
        } -> tensor<?x?xf32>
        iree_tensor_ext.dispatch.tensor.store %19, %13, offsets = [0, 0], sizes = [%9, %0], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%9, %0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_16 {
    flow.executable.export public @decode_dispatch_16_matmul_Dx2048x2048_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_16_matmul_Dx2048x2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf32>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf32>> -> tensor<2048x2048xf32>
        %5 = tensor.empty(%0) : tensor<?x2048xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
        %7 = linalg.matmul ins(%3, %4 : tensor<?x2048xf32>, tensor<2048x2048xf32>) outs(%6 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : tensor<?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_21 {
    flow.executable.export public @decode_dispatch_21_elementwise_broadcast_D_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_21_elementwise_broadcast_D_f32(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
        %c16 = arith.constant 16 : index
        %c128 = arith.constant 128 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %2 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%1}
        %3 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%1, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%1} -> tensor<?x2048xf32>
        %5 = tensor.empty(%0) : tensor<?xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%5 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %7 = linalg.index 0 : index
          %8 = arith.remsi %7, %c128 : index
          %9 = arith.divsi %7, %c128 : index
          %10 = arith.remsi %9, %c16 : index
          %11 = arith.divsi %9, %c16 : index
          %12 = arith.muli %10, %c128 : index
          %13 = arith.addi %12, %8 : index
          %extracted = tensor.extract %4[%11, %13] : tensor<?x2048xf32>
          linalg.yield %extracted : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %3, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_25 {
    flow.executable.export public @decode_dispatch_25_elementwise_broadcast_D_f32 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_25_elementwise_broadcast_D_f32(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16x128xf32>>, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
        %c16 = arith.constant 16 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 2 : index
        %3 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>{%1}
        %4 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16x128xf32>>{%2}
        %5 = flow.dispatch.tie_shape %arg6 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [%1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>{%1} -> tensor<?xi64>
        %7 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [%2, 16, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16x128xf32>>{%2} -> tensor<?x16x128xf32>
        %9 = tensor.empty(%0) : tensor<?xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%9 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %11 = linalg.index 0 : index
          %12 = arith.remsi %11, %c128 : index
          %13 = arith.divsi %11, %c128 : index
          %14 = arith.remsi %13, %c16 : index
          %15 = arith.divsi %13, %c16 : index
          %extracted = tensor.extract %6[%15] : tensor<?xi64>
          %16 = arith.trunci %extracted : i64 to i32
          %17 = arith.sitofp %16 : i32 to f32
          %18 = arith.cmpi slt, %12, %c64 : index
          %19 = scf.if %18 -> (f32) {
            %extracted_0 = tensor.extract %7[%12] : tensor<64xf32>
            %20 = arith.mulf %17, %extracted_0 : f32
            %21 = math.cos %20 : f32
            %22 = math.sin %20 : f32
            %extracted_1 = tensor.extract %8[%15, %14, %12] : tensor<?x16x128xf32>
            %23 = arith.addi %12, %c64 : index
            %extracted_2 = tensor.extract %8[%15, %14, %23] : tensor<?x16x128xf32>
            %24 = arith.mulf %extracted_1, %21 : f32
            %25 = arith.mulf %extracted_2, %22 : f32
            %26 = arith.subf %24, %25 : f32
            scf.yield %26 : f32
          } else {
            %20 = arith.subi %12, %c64 : index
            %extracted_0 = tensor.extract %7[%20] : tensor<64xf32>
            %21 = arith.mulf %17, %extracted_0 : f32
            %22 = math.cos %21 : f32
            %23 = math.sin %21 : f32
            %extracted_1 = tensor.extract %8[%15, %14, %12] : tensor<?x16x128xf32>
            %extracted_2 = tensor.extract %8[%15, %14, %20] : tensor<?x16x128xf32>
            %24 = arith.mulf %extracted_2, %23 : f32
            %25 = arith.mulf %extracted_1, %22 : f32
            %26 = arith.addf %24, %25 : f32
            scf.yield %26 : f32
          }
          linalg.yield %19 : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %5, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_27 {
    flow.executable.export public @decode_dispatch_27_slow_memcpy workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_27_slow_memcpy(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 3 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg6, 4 : index
        %5 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %6 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%0, %4, %2, %3}
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %6, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%0, %4, %2, %3}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_28 {
    flow.executable.export public @decode_dispatch_28_slow_memcpy workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_28_slow_memcpy(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1x16x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 2 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 3 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg6, 4 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg7, 5 : index
        %5 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1x16x128xf32>>{%0}
        %6 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%1, %2, %3, %4}
        %7 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %8 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, 1, 16, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1x16x128xf32>>{%0} -> tensor<?x1x16x128xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %6, offsets = [0, %7, 0, 0], sizes = [%0, 1, 16, 128], strides = [1, 1, 1, 1] : tensor<?x1x16x128xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%1, %2, %3, %4}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_31 {
    flow.executable.export public @decode_dispatch_31_elementwise_broadcast_DxDxDxDx128_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_31_elementwise_broadcast_DxDxDxDx128_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x128xf32>>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x128xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 3 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 4 : index
        %5 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x128xf32>>{%0, %1, %2}
        %6 = flow.dispatch.tie_shape %arg6 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x128xf32>>{%3, %2, %4, %1}
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x128xf32>>{%0, %1, %2} -> tensor<?x?x?x128xf32>
        %8 = tensor.empty(%3, %2, %4, %1) : tensor<?x?x?x?x128xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<?x?x?x128xf32>) outs(%8 : tensor<?x?x?x?x128xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x?x?x?x128xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %6, offsets = [0, 0, 0, 0, 0], sizes = [%3, %2, %4, %1, 128], strides = [1, 1, 1, 1, 1] : tensor<?x?x?x?x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?x128xf32>>{%3, %2, %4, %1}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_32 {
    flow.executable.export public @decode_dispatch_32_elementwise_broadcast_DxDxDx128xD_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_32_elementwise_broadcast_DxDxDx128xD_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x128xf32>>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x128x?xf32>>) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 3 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 4 : index
        %5 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x128xf32>>{%0, %1, %2}
        %6 = flow.dispatch.tie_shape %arg6 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x128x?xf32>>{%3, %2, %4, %1}
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x128xf32>>{%0, %1, %2} -> tensor<?x?x?x128xf32>
        %8 = tensor.empty(%3, %2, %4, %1) : tensor<?x?x?x128x?xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<?x?x?x128xf32>) outs(%8 : tensor<?x?x?x128x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x?x?x128x?xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %6, offsets = [0, 0, 0, 0, 0], sizes = [%3, %2, %4, 128, %1], strides = [1, 1, 1, 1, 1] : tensor<?x?x?x128x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x128x?xf32>>{%3, %2, %4, %1}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_33 {
    flow.executable.export public @decode_dispatch_33_attention_Dx128xDx128 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_33_attention_Dx128xDx128(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128x?xf32>>, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x128xf32>>) {
        %cst = arith.constant 0.0883883461 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 2 : index
        %3 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>{%1, %2}
        %5 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128x?xf32>>{%1, %2}
        %6 = flow.dispatch.tie_shape %arg6 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x128xf32>>{%0}
        %7 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128xf32>>{%0} -> tensor<?x128xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [%1, %2, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x128xf32>>{%1, %2} -> tensor<?x?x128xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [%1, 128, %2], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128x?xf32>>{%1, %2} -> tensor<?x128x?xf32>
        %10 = tensor.empty(%0) : tensor<?x128xf32>
        %11 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>]} ins(%7, %8, %9, %cst : tensor<?x128xf32>, tensor<?x?x128xf32>, tensor<?x128x?xf32>, f32) outs(%10 : tensor<?x128xf32>) {
        ^bb0(%arg7: f32):
          iree_linalg_ext.yield %arg7 : f32
        } -> tensor<?x128xf32>
        iree_tensor_ext.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%0, 128], strides = [1, 1] : tensor<?x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x128xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_34 {
    flow.executable.export public @decode_dispatch_34_elementwise_broadcast_D_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_34_elementwise_broadcast_D_f32(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128xf32>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
        %c128 = arith.constant 128 : index
        %c16 = arith.constant 16 : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %2 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128xf32>>{%1}
        %3 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%1, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128xf32>>{%1} -> tensor<?x128xf32>
        %5 = tensor.empty(%0) : tensor<?xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%5 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %7 = linalg.index 0 : index
          %8 = arith.remsi %7, %c128 : index
          %9 = arith.divsi %7, %c128 : index
          %10 = arith.remsi %9, %c16 : index
          %11 = arith.divsi %9, %c16 : index
          %12 = arith.muli %11, %c16 : index
          %13 = arith.addi %12, %10 : index
          %extracted = tensor.extract %4[%13, %8] : tensor<?x128xf32>
          linalg.yield %extracted : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %3, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_35 {
    flow.executable.export public @decode_dispatch_35_matmul_Dx2048x2048_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_35_matmul_Dx2048x2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %3 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf32>> -> tensor<2048x2048xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %7 = tensor.empty(%0) : tensor<?x2048xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
        %9 = linalg.matmul ins(%4, %5 : tensor<?x2048xf32>, tensor<2048x2048xf32>) outs(%8 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %9 : tensor<?x2048xf32>, tensor<?x2048xf32>) outs(%7 : tensor<?x2048xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          linalg.yield %11 : f32
        } -> tensor<?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : tensor<?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_36 {
    flow.executable.export public @decode_dispatch_36_slow_memcpy workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_36_slow_memcpy(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>, %arg1: index, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16x128xf32>>, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index, %arg14: index) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg8, 5 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %arg9, 6 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %arg10, 7 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %arg11, 8 : index
        %5 = iree_tensor_ext.dispatch.workload.ordinal %arg12, 9 : index
        %6 = iree_tensor_ext.dispatch.workload.ordinal %arg13, 10 : index
        %7 = iree_tensor_ext.dispatch.workload.ordinal %arg14, 11 : index
        %8 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%1, %2, %3}
        %9 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16x128xf32>>{%0}
        %10 = flow.dispatch.tie_shape %arg6 : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%4, %5, %6, %7}
        %11 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
        %12 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 2 : index
        %13 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 3 : index
        %14 = iree_tensor_ext.dispatch.workload.ordinal %arg7, 4 : index
        %15 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [%11, 0, 0], sizes = [1, %0, %3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%1, %2, %3} -> tensor<?x?xi32>
        %16 = iree_tensor_ext.dispatch.tensor.load %9, offsets = [%12, 0, 0], sizes = [1, 16, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16x128xf32>>{%0} -> tensor<16x128xf32>
        %extracted = tensor.extract %15[%12, %13] : tensor<?x?xi32>
        %17 = arith.index_cast %extracted : i32 to index
        iree_tensor_ext.dispatch.tensor.store %16, %10, offsets = [%17, %14, 0, 0], sizes = [1, 1, 16, 128], strides = [1, 1, 1, 1] : tensor<16x128xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%4, %5, %6, %7}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_53 {
    flow.executable.export public @decode_dispatch_53_matvec_like_Dx2048x8_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_53_matvec_like_Dx2048x8_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?x2048xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xf32>>{%0}
        %3 = flow.dispatch.tie_shape %arg2 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg4 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [8, %0, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?x2048xf32>>{%0} -> tensor<8x?x2048xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [8, %0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xf32>>{%0} -> tensor<8x?xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %8 = tensor.empty(%0) : tensor<?x2048xf32>
        %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0, d1)>, affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %6 : tensor<8x?x2048xf32>, tensor<8x?xf32>) outs(%9 : tensor<?x2048xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %12 = arith.mulf %in, %in_0 : f32
          %13 = arith.addf %12, %out : f32
          linalg.yield %13 : f32
        } -> tensor<?x2048xf32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7, %10 : tensor<?x2048xf32>, tensor<?x2048xf32>) outs(%8 : tensor<?x2048xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %12 = arith.addf %in, %in_0 : f32
          linalg.yield %12 : f32
        } -> tensor<?x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %11, %4, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : tensor<?x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @decode_dispatch_55 {
    flow.executable.export public @decode_dispatch_55_matmul_Dx50304x2048_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @decode_dispatch_55_matmul_Dx50304x2048_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x50304xf32>>, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x50304xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x50304xf32>>{%0}
        %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%0, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32>>{%0} -> tensor<?x2048xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2048, 50304], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x50304xf32>> -> tensor<2048x50304xf32>
        %5 = tensor.empty(%0) : tensor<?x50304xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<?x50304xf32>) -> tensor<?x50304xf32>
        %7 = linalg.matmul ins(%3, %4 : tensor<?x2048xf32>, tensor<2048x50304xf32>) outs(%6 : tensor<?x50304xf32>) -> tensor<?x50304xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [%0, 50304], strides = [1, 1] : tensor<?x50304xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x50304xf32>>{%0}
        return
      }
    }
  }
  util.func public @decode(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !util.list<?>, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: index) -> (!hal.buffer_view, !util.list<?>) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @decode(%input0: tensor<?xi64>, %input1: tensor<?xi64>, %input2: !util.list<?>, %input3: tensor<?x?x?xi32>, %input4: tensor<?x?xi32>, %input5: index) -> (%output0: tensor<?x?xf32>, %output1: !util.list<?>)"}} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %c11_i32 = arith.constant 11 : i32
    %c12_i32 = arith.constant 12 : i32
    %c13_i32 = arith.constant 13 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c8192 = arith.constant 8192 : index
    %c14_i32 = arith.constant 14 : i32
    %c2048 = arith.constant 2048 : index
    %c50304 = arith.constant 50304 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %__parameter_model_token_embd_weight_tensor_50304x2048xf32 = util.global.load immutable @__parameter_model_token_embd_weight_tensor_50304x2048xf32 : tensor<50304x2048xf32>
    %__parameter_model_blk_0_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_1_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_2_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_3_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_4_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_5_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_6_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_7_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_8_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_9_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_10_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_11_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_12_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_13_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_14_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_15_ffn_down_exps_weight_tensor_2048x1024x64xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32>
    %__parameter_model_blk_0_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_1_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_2_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_3_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_4_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_5_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_6_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_7_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_8_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_9_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_10_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_11_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_12_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_13_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_14_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_15_ffn_gate_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_0_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_1_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_2_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_3_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_4_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_5_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_6_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_7_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_8_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_9_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_10_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_11_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_12_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_13_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_14_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_15_ffn_up_exps_weight_tensor_1024x2048x64xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32>
    %__parameter_model_blk_0_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_1_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_2_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_3_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_4_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_5_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_6_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_7_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_8_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_9_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_10_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_11_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_12_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_13_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_14_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_15_ffn_gate_inp_weight_tensor_64x2048xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32>
    %__parameter_model_blk_0_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_1_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_2_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_3_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_4_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_5_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_6_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_7_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_8_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_9_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_10_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_11_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_12_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_13_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_14_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_15_attn_k_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_k_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_0_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_1_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_2_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_3_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_4_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_5_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_6_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_7_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_8_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_9_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_10_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_11_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_12_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_13_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_14_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_15_attn_q_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_q_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_0_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_1_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_2_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_3_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_4_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_5_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_6_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_7_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_8_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_9_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_10_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_11_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_12_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_13_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_14_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_15_attn_output_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_0_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_1_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_2_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_3_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_4_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_5_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_6_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_7_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_8_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_9_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_10_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_11_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_12_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_13_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_14_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_15_attn_v_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_0_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_1_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_2_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_3_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_4_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_5_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_6_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_7_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_8_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_9_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_10_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_11_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_12_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_13_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_14_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_15_attn_k_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_0_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_1_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_2_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_3_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_4_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_5_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_6_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_7_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_8_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_9_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_10_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_11_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_12_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_13_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_14_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_15_attn_q_weight_tensor_2048x2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32>
    %__parameter_model_blk_0_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_0_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_1_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_1_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_2_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_2_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_3_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_3_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_4_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_4_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_5_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_5_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_6_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_6_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_7_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_7_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_8_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_8_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_9_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_9_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_10_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_10_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_11_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_11_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_12_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_12_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_13_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_13_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_14_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_14_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_15_ffn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_15_ffn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_0_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_0_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_1_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_1_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_2_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_2_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_3_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_3_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_4_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_4_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_5_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_5_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_6_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_6_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_7_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_7_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_8_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_8_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_9_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_9_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_10_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_10_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_11_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_11_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_12_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_12_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_13_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_13_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_14_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_14_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_blk_15_attn_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_blk_15_attn_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_output_norm_weight_tensor_2048xf32 = util.global.load immutable @__parameter_model_output_norm_weight_tensor_2048xf32 : tensor<2048xf32>
    %__parameter_model_output_weight_tensor_2048x50304xf32 = util.global.load immutable @__parameter_model_output_weight_tensor_2048x50304xf32 : tensor<2048x50304xf32>
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<?xi64>{%0}
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %3 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<?xi64>{%2}
    %4 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[0] : index
    %5 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[1] : index
    %6 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[2] : index
    %7 = hal.tensor.import %arg3 "input3" : !hal.buffer_view -> tensor<?x?x?xi32>{%4, %5, %6}
    %8 = hal.buffer_view.dim<%arg4 : !hal.buffer_view>[0] : index
    %9 = hal.buffer_view.dim<%arg4 : !hal.buffer_view>[1] : index
    %10 = hal.tensor.import %arg4 "input4" : !hal.buffer_view -> tensor<?x?xi32>{%8, %9}
    %11 = flow.dispatch @decode_dispatch_0::@decode_dispatch_0_gather_50304x2048xf32_dispatch_tensor_store[%0](%__parameter_model_token_embd_weight_tensor_50304x2048xf32, %1, %0) : (tensor<50304x2048xf32>, tensor<?xi64>{%0}, index) -> tensor<?x2048xf32>{%0}
    %12 = flow.tensor.reshape %11 : tensor<?x2048xf32>{%0} -> tensor<?x?xf32>{%0, %c2048}
    %13 = scf.for %arg6 = %c0 to %c16 step %c1 iter_args(%arg7 = %12) -> (tensor<?x?xf32>) {
      %19 = arith.index_castui %arg6 : index to i32
      %20 = arith.cmpi eq, %19, %c0_i32 : i32
      %21:12 = scf.if %20 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
        scf.yield %__parameter_model_blk_0_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_0_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_0_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_0_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_0_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_0_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_0_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_0_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_0_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_0_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_0_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_0_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
      } else {
        %118 = arith.cmpi eq, %19, %c1_i32 : i32
        %119:12 = scf.if %118 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
          scf.yield %__parameter_model_blk_1_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_1_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_1_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_1_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_1_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_1_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_1_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_1_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_1_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_1_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_1_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_1_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
        } else {
          %120 = arith.cmpi eq, %19, %c2_i32 : i32
          %121:12 = scf.if %120 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
            scf.yield %__parameter_model_blk_2_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_2_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_2_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_2_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_2_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_2_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_2_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_2_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_2_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_2_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_2_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_2_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
          } else {
            %122 = arith.cmpi eq, %19, %c3_i32 : i32
            %123:12 = scf.if %122 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
              scf.yield %__parameter_model_blk_3_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_3_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_3_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_3_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_3_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_3_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_3_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_3_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_3_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_3_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_3_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_3_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
            } else {
              %124 = arith.cmpi eq, %19, %c4_i32 : i32
              %125:12 = scf.if %124 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                scf.yield %__parameter_model_blk_4_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_4_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_4_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_4_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_4_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_4_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_4_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_4_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_4_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_4_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_4_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_4_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
              } else {
                %126 = arith.cmpi eq, %19, %c5_i32 : i32
                %127:12 = scf.if %126 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                  scf.yield %__parameter_model_blk_5_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_5_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_5_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_5_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_5_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_5_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_5_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_5_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_5_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_5_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_5_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_5_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                } else {
                  %128 = arith.cmpi eq, %19, %c6_i32 : i32
                  %129:12 = scf.if %128 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                    scf.yield %__parameter_model_blk_6_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_6_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_6_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_6_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_6_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_6_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_6_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_6_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_6_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_6_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_6_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_6_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                  } else {
                    %130 = arith.cmpi eq, %19, %c7_i32 : i32
                    %131:12 = scf.if %130 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                      scf.yield %__parameter_model_blk_7_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_7_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_7_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_7_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_7_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_7_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_7_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_7_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_7_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_7_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_7_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_7_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                    } else {
                      %132 = arith.cmpi eq, %19, %c8_i32 : i32
                      %133:12 = scf.if %132 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                        scf.yield %__parameter_model_blk_8_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_8_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_8_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_8_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_8_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_8_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_8_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_8_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_8_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_8_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_8_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_8_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                      } else {
                        %134 = arith.cmpi eq, %19, %c9_i32 : i32
                        %135:12 = scf.if %134 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                          scf.yield %__parameter_model_blk_9_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_9_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_9_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_9_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_9_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_9_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_9_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_9_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_9_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_9_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_9_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_9_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                        } else {
                          %136 = arith.cmpi eq, %19, %c10_i32 : i32
                          %137:12 = scf.if %136 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                            scf.yield %__parameter_model_blk_10_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_10_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_10_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_10_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_10_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_10_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_10_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_10_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_10_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_10_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_10_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_10_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                          } else {
                            %138 = arith.cmpi eq, %19, %c11_i32 : i32
                            %139:12 = scf.if %138 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                              scf.yield %__parameter_model_blk_11_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_11_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_11_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_11_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_11_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_11_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_11_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_11_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_11_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_11_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_11_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_11_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                            } else {
                              %140 = arith.cmpi eq, %19, %c12_i32 : i32
                              %141:12 = scf.if %140 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                                scf.yield %__parameter_model_blk_12_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_12_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_12_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_12_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_12_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_12_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_12_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_12_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_12_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_12_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_12_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_12_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                              } else {
                                %142 = arith.cmpi eq, %19, %c13_i32 : i32
                                %143:12 = scf.if %142 -> (tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>) {
                                  scf.yield %__parameter_model_blk_13_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_13_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_13_attn_q_weight_tensor_2048x2048xf32, %__parameter_model_blk_13_attn_k_weight_tensor_2048x2048xf32, %__parameter_model_blk_13_attn_v_weight_tensor_2048x2048xf32, %__parameter_model_blk_13_attn_output_weight_tensor_2048x2048xf32, %__parameter_model_blk_13_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_13_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_13_ffn_gate_inp_weight_tensor_64x2048xf32, %__parameter_model_blk_13_ffn_up_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_13_ffn_gate_exps_weight_tensor_1024x2048x64xf32, %__parameter_model_blk_13_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                                } else {
                                  %144 = arith.cmpi eq, %19, %c14_i32 : i32
                                  %145 = flow.dispatch @prefill_dispatch_1::@prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%144, %__parameter_model_blk_14_attn_norm_weight_tensor_2048xf32, %__parameter_model_blk_15_attn_norm_weight_tensor_2048xf32) : (i1, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
                                  %146 = flow.dispatch @prefill_dispatch_1::@prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%144, %__parameter_model_blk_14_ffn_norm_weight_tensor_2048xf32, %__parameter_model_blk_15_ffn_norm_weight_tensor_2048xf32) : (i1, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
                                  %147 = flow.tensor.reshape %__parameter_model_blk_14_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %148 = flow.tensor.reshape %__parameter_model_blk_15_attn_q_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %149 = flow.dispatch @prefill_dispatch_3::@prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%144, %147, %148) : (i1, tensor<4194304xf32>, tensor<4194304xf32>) -> tensor<4194304xf32>
                                  %150 = flow.tensor.reshape %149 : tensor<4194304xf32> -> tensor<2048x2048xf32>
                                  %151 = flow.tensor.reshape %__parameter_model_blk_14_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %152 = flow.tensor.reshape %__parameter_model_blk_15_attn_k_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %153 = flow.dispatch @prefill_dispatch_3::@prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%144, %151, %152) : (i1, tensor<4194304xf32>, tensor<4194304xf32>) -> tensor<4194304xf32>
                                  %154 = flow.tensor.reshape %153 : tensor<4194304xf32> -> tensor<2048x2048xf32>
                                  %155 = flow.tensor.reshape %__parameter_model_blk_14_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %156 = flow.tensor.reshape %__parameter_model_blk_15_attn_v_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %157 = flow.dispatch @prefill_dispatch_3::@prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%144, %155, %156) : (i1, tensor<4194304xf32>, tensor<4194304xf32>) -> tensor<4194304xf32>
                                  %158 = flow.tensor.reshape %157 : tensor<4194304xf32> -> tensor<2048x2048xf32>
                                  %159 = flow.tensor.reshape %__parameter_model_blk_14_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %160 = flow.tensor.reshape %__parameter_model_blk_15_attn_output_weight_tensor_2048x2048xf32 : tensor<2048x2048xf32> -> tensor<4194304xf32>
                                  %161 = flow.dispatch @prefill_dispatch_3::@prefill_dispatch_3_elementwise_4194304_i1xf32xf32xf32(%144, %159, %160) : (i1, tensor<4194304xf32>, tensor<4194304xf32>) -> tensor<4194304xf32>
                                  %162 = flow.tensor.reshape %161 : tensor<4194304xf32> -> tensor<2048x2048xf32>
                                  %163 = flow.dispatch @prefill_dispatch_1::@prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%144, %__parameter_model_blk_14_attn_q_norm_weight_tensor_2048xf32, %__parameter_model_blk_15_attn_q_norm_weight_tensor_2048xf32) : (i1, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
                                  %164 = flow.dispatch @prefill_dispatch_1::@prefill_dispatch_1_elementwise_2048_i1xf32xf32xf32(%144, %__parameter_model_blk_14_attn_k_norm_weight_tensor_2048xf32, %__parameter_model_blk_15_attn_k_norm_weight_tensor_2048xf32) : (i1, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
                                  %165 = flow.tensor.reshape %__parameter_model_blk_14_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32> -> tensor<131072xf32>
                                  %166 = flow.tensor.reshape %__parameter_model_blk_15_ffn_gate_inp_weight_tensor_64x2048xf32 : tensor<64x2048xf32> -> tensor<131072xf32>
                                  %167 = flow.dispatch @prefill_dispatch_9::@prefill_dispatch_9_elementwise_131072_i1xf32xf32xf32(%144, %165, %166) : (i1, tensor<131072xf32>, tensor<131072xf32>) -> tensor<131072xf32>
                                  %168 = flow.tensor.reshape %167 : tensor<131072xf32> -> tensor<64x2048xf32>
                                  %169 = flow.tensor.reshape %__parameter_model_blk_14_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32> -> tensor<134217728xf32>
                                  %170 = flow.tensor.reshape %__parameter_model_blk_15_ffn_up_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32> -> tensor<134217728xf32>
                                  %171 = flow.dispatch @prefill_dispatch_10::@prefill_dispatch_10_elementwise_134217728_i1xf32xf32xf32(%144, %169, %170) : (i1, tensor<134217728xf32>, tensor<134217728xf32>) -> tensor<134217728xf32>
                                  %172 = flow.tensor.reshape %171 : tensor<134217728xf32> -> tensor<1024x2048x64xf32>
                                  %173 = flow.tensor.reshape %__parameter_model_blk_14_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32> -> tensor<134217728xf32>
                                  %174 = flow.tensor.reshape %__parameter_model_blk_15_ffn_gate_exps_weight_tensor_1024x2048x64xf32 : tensor<1024x2048x64xf32> -> tensor<134217728xf32>
                                  %175 = flow.dispatch @prefill_dispatch_10::@prefill_dispatch_10_elementwise_134217728_i1xf32xf32xf32(%144, %173, %174) : (i1, tensor<134217728xf32>, tensor<134217728xf32>) -> tensor<134217728xf32>
                                  %176 = flow.tensor.reshape %175 : tensor<134217728xf32> -> tensor<1024x2048x64xf32>
                                  %177 = flow.tensor.reshape %__parameter_model_blk_14_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32> -> tensor<134217728xf32>
                                  %178 = flow.tensor.reshape %__parameter_model_blk_15_ffn_down_exps_weight_tensor_2048x1024x64xf32 : tensor<2048x1024x64xf32> -> tensor<134217728xf32>
                                  %179 = flow.dispatch @prefill_dispatch_10::@prefill_dispatch_10_elementwise_134217728_i1xf32xf32xf32(%144, %177, %178) : (i1, tensor<134217728xf32>, tensor<134217728xf32>) -> tensor<134217728xf32>
                                  %180 = flow.tensor.reshape %179 : tensor<134217728xf32> -> tensor<2048x1024x64xf32>
                                  scf.yield %145, %146, %150, %154, %158, %162, %163, %164, %168, %172, %176, %180 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                                }
                                scf.yield %143#0, %143#1, %143#2, %143#3, %143#4, %143#5, %143#6, %143#7, %143#8, %143#9, %143#10, %143#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                              }
                              scf.yield %141#0, %141#1, %141#2, %141#3, %141#4, %141#5, %141#6, %141#7, %141#8, %141#9, %141#10, %141#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                            }
                            scf.yield %139#0, %139#1, %139#2, %139#3, %139#4, %139#5, %139#6, %139#7, %139#8, %139#9, %139#10, %139#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                          }
                          scf.yield %137#0, %137#1, %137#2, %137#3, %137#4, %137#5, %137#6, %137#7, %137#8, %137#9, %137#10, %137#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                        }
                        scf.yield %135#0, %135#1, %135#2, %135#3, %135#4, %135#5, %135#6, %135#7, %135#8, %135#9, %135#10, %135#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                      }
                      scf.yield %133#0, %133#1, %133#2, %133#3, %133#4, %133#5, %133#6, %133#7, %133#8, %133#9, %133#10, %133#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                    }
                    scf.yield %131#0, %131#1, %131#2, %131#3, %131#4, %131#5, %131#6, %131#7, %131#8, %131#9, %131#10, %131#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                  }
                  scf.yield %129#0, %129#1, %129#2, %129#3, %129#4, %129#5, %129#6, %129#7, %129#8, %129#9, %129#10, %129#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
                }
                scf.yield %127#0, %127#1, %127#2, %127#3, %127#4, %127#5, %127#6, %127#7, %127#8, %127#9, %127#10, %127#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
              }
              scf.yield %125#0, %125#1, %125#2, %125#3, %125#4, %125#5, %125#6, %125#7, %125#8, %125#9, %125#10, %125#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
            }
            scf.yield %123#0, %123#1, %123#2, %123#3, %123#4, %123#5, %123#6, %123#7, %123#8, %123#9, %123#10, %123#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
          }
          scf.yield %121#0, %121#1, %121#2, %121#3, %121#4, %121#5, %121#6, %121#7, %121#8, %121#9, %121#10, %121#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
        }
        scf.yield %119#0, %119#1, %119#2, %119#3, %119#4, %119#5, %119#6, %119#7, %119#8, %119#9, %119#10, %119#11 : tensor<2048xf32>, tensor<2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048xf32>, tensor<1024x2048x64xf32>, tensor<1024x2048x64xf32>, tensor<2048x1024x64xf32>
      }
      %22 = flow.tensor.reshape %arg7 : tensor<?x?xf32>{%0, %c2048} -> tensor<?x2048xf32>{%0}
      %23 = flow.dispatch @decode_dispatch_13::@decode_dispatch_13_elementwise_Dx2048_f32[%0, %c2048](%c2048_i32, %arg7, %22, %21#0, %0, %c2048) : (i32, tensor<?x?xf32>{%0, %c2048}, tensor<?x2048xf32>{%0}, tensor<2048xf32>, index, index) -> tensor<?x2048xf32>{%0}
      %24 = util.list.get %arg2[%c0] : !util.list<?> -> !hal.buffer_view
      %25 = hal.buffer_view.dim<%24 : !hal.buffer_view>[0] : index
      %26 = hal.buffer_view.dim<%24 : !hal.buffer_view>[1] : index
      %27 = hal.buffer_view.dim<%24 : !hal.buffer_view>[2] : index
      %28 = hal.buffer_view.dim<%24 : !hal.buffer_view>[3] : index
      %29 = hal.tensor.import %24 : !hal.buffer_view -> tensor<?x?x?x?xf32>{%25, %26, %27, %28}
      %30 = util.list.get %arg2[%c1] : !util.list<?> -> !hal.buffer_view
      %31 = hal.tensor.import %30 : !hal.buffer_view -> tensor<?x?x?x?xf32>{%25, %26, %27, %28}
      %32 = arith.muli %arg5, %27 overflow<nsw> : index
      %33 = arith.muli %32, %28 overflow<nsw> : index
      %34 = flow.dispatch @decode_dispatch_14::@decode_dispatch_14_elementwise_broadcast_DxD_i32xf32[%arg6, %33, %8, %9, %4, %6, %25, %26, %27, %28, %5](%10, %arg6, %7, %33, %29, %8, %9, %4, %6, %25, %26, %27, %28, %5) : (tensor<?x?xi32>{%8, %9}, index, tensor<?x?x?xi32>{%4, %5, %6}, index, tensor<?x?x?x?xf32>{%25, %26, %27, %28}, index, index, index, index, index, index, index, index, index) -> tensor<?x?xf32>{%5, %33}
      %35 = flow.tensor.reshape %34 : tensor<?x?xf32>{%5, %33} -> tensor<?x?x?x?xf32>{%5, %arg5, %27, %28}
      %36 = flow.dispatch @decode_dispatch_14::@decode_dispatch_14_elementwise_broadcast_DxD_i32xf32[%arg6, %33, %8, %9, %4, %6, %25, %26, %27, %28, %5](%10, %arg6, %7, %33, %31, %8, %9, %4, %6, %25, %26, %27, %28, %5) : (tensor<?x?xi32>{%8, %9}, index, tensor<?x?x?xi32>{%4, %5, %6}, index, tensor<?x?x?x?xf32>{%25, %26, %27, %28}, index, index, index, index, index, index, index, index, index) -> tensor<?x?xf32>{%5, %33}
      %37 = flow.tensor.reshape %36 : tensor<?x?xf32>{%5, %33} -> tensor<?x?x?x?xf32>{%5, %arg5, %27, %28}
      %38 = flow.dispatch @decode_dispatch_16::@decode_dispatch_16_matmul_Dx2048x2048_f32[%0](%23, %21#2, %0) : (tensor<?x2048xf32>{%0}, tensor<2048x2048xf32>, index) -> tensor<?x2048xf32>{%0}
      %39 = flow.dispatch @decode_dispatch_16::@decode_dispatch_16_matmul_Dx2048x2048_f32[%0](%23, %21#3, %0) : (tensor<?x2048xf32>{%0}, tensor<2048x2048xf32>, index) -> tensor<?x2048xf32>{%0}
      %40 = flow.dispatch @decode_dispatch_16::@decode_dispatch_16_matmul_Dx2048x2048_f32[%0](%23, %21#4, %0) : (tensor<?x2048xf32>{%0}, tensor<2048x2048xf32>, index) -> tensor<?x2048xf32>{%0}
      %41 = flow.dispatch @prefill_dispatch_13::@prefill_dispatch_13_reduction_Dx2048_f32[%0](%38, %21#6, %0) : (tensor<?x2048xf32>{%0}, tensor<2048xf32>, index) -> tensor<?x2048xf32>{%0}
      %42 = flow.dispatch @prefill_dispatch_13::@prefill_dispatch_13_reduction_Dx2048_f32[%0](%39, %21#7, %0) : (tensor<?x2048xf32>{%0}, tensor<2048xf32>, index) -> tensor<?x2048xf32>{%0}
      %43 = arith.muli %0, %c2048 overflow<nsw> : index
      %44 = flow.dispatch @decode_dispatch_21::@decode_dispatch_21_elementwise_broadcast_D_f32[%43, %0](%43, %41, %0) : (index, tensor<?x2048xf32>{%0}, index) -> tensor<?xf32>{%43}
      %45 = flow.tensor.reshape %44 : tensor<?xf32>{%43} -> tensor<?x16x128xf32>{%0}
      %46 = flow.dispatch @decode_dispatch_21::@decode_dispatch_21_elementwise_broadcast_D_f32[%43, %0](%43, %42, %0) : (index, tensor<?x2048xf32>{%0}, index) -> tensor<?xf32>{%43}
      %47 = flow.tensor.reshape %46 : tensor<?xf32>{%43} -> tensor<?x16x128xf32>{%0}
      %48 = flow.dispatch @decode_dispatch_21::@decode_dispatch_21_elementwise_broadcast_D_f32[%43, %0](%43, %40, %0) : (index, tensor<?x2048xf32>{%0}, index) -> tensor<?xf32>{%43}
      %49 = flow.tensor.reshape %48 : tensor<?xf32>{%43} -> tensor<?x16x128xf32>{%0}
      %50 = flow.dispatch @prefill_dispatch_24::@prefill_dispatch_24_elementwise_broadcast_64_f32() : () -> tensor<64xf32>
      %51 = flow.dispatch @decode_dispatch_25::@decode_dispatch_25_elementwise_broadcast_D_f32[%43, %2, %0](%43, %3, %50, %45, %2, %0) : (index, tensor<?xi64>{%2}, tensor<64xf32>, tensor<?x16x128xf32>{%0}, index, index) -> tensor<?xf32>{%43}
      %52 = flow.dispatch @decode_dispatch_25::@decode_dispatch_25_elementwise_broadcast_D_f32[%43, %2, %0](%43, %3, %50, %47, %2, %0) : (index, tensor<?xi64>{%2}, tensor<64xf32>, tensor<?x16x128xf32>{%0}, index, index) -> tensor<?xf32>{%43}
      %53 = flow.tensor.reshape %52 : tensor<?xf32>{%43} -> tensor<?x16x128xf32>{%0}
      %54 = flow.tensor.reshape %52 : tensor<?xf32>{%43} -> tensor<?x1x16x128xf32>{%0}
      %55 = arith.addi %arg5, %c1 : index
      %56 = flow.tensor.empty : tensor<?x?x?x?xf32>{%5, %55, %27, %28}
      %57 = flow.dispatch @decode_dispatch_27::@decode_dispatch_27_slow_memcpy[%5, %arg5, %27, %28, %55](%35, %56, %5, %arg5, %27, %28, %55) : (tensor<?x?x?x?xf32>{%5, %arg5, %27, %28}, tensor<?x?x?x?xf32>{%5, %55, %27, %28}, index, index, index, index, index) -> %56{%5, %55, %27, %28}
      %58 = flow.dispatch @decode_dispatch_28::@decode_dispatch_28_slow_memcpy[%arg5, %0, %5, %55, %27, %28](%54, %57, %arg5, %0, %5, %55, %27, %28) : (tensor<?x1x16x128xf32>{%0}, tensor<?x?x?x?xf32>{%5, %55, %27, %28}, index, index, index, index, index, index) -> %57{%5, %55, %27, %28}
      %59 = flow.tensor.reshape %48 : tensor<?xf32>{%43} -> tensor<?x1x16x128xf32>{%0}
      %60 = flow.dispatch @decode_dispatch_27::@decode_dispatch_27_slow_memcpy[%5, %arg5, %27, %28, %55](%37, %56, %5, %arg5, %27, %28, %55) : (tensor<?x?x?x?xf32>{%5, %arg5, %27, %28}, tensor<?x?x?x?xf32>{%5, %55, %27, %28}, index, index, index, index, index) -> %56{%5, %55, %27, %28}
      %61 = flow.dispatch @decode_dispatch_28::@decode_dispatch_28_slow_memcpy[%arg5, %0, %5, %55, %27, %28](%59, %60, %arg5, %0, %5, %55, %27, %28) : (tensor<?x1x16x128xf32>{%0}, tensor<?x?x?x?xf32>{%5, %55, %27, %28}, index, index, index, index, index, index) -> %60{%5, %55, %27, %28}
      %62 = flow.tensor.reshape %58 : tensor<?x?x?x?xf32>{%5, %55, %27, %c128} -> tensor<?x?x?x128xf32>{%5, %55, %27}
      %63 = flow.tensor.reshape %61 : tensor<?x?x?x?xf32>{%5, %55, %27, %c128} -> tensor<?x?x?x128xf32>{%5, %55, %27}
      %64 = arith.divui %c16, %27 : index
      %65 = flow.dispatch @decode_dispatch_31::@decode_dispatch_31_elementwise_broadcast_DxDxDxDx128_f32[%5, %55, %27, %0, %64](%62, %5, %55, %27, %0, %64) : (tensor<?x?x?x128xf32>{%5, %55, %27}, index, index, index, index, index) -> tensor<?x?x?x?x128xf32>{%0, %27, %64, %55}
      %66 = arith.muli %0, %27 overflow<nsw> : index
      %67 = arith.muli %66, %64 overflow<nsw> : index
      %68 = flow.tensor.reshape %65 : tensor<?x?x?x?x128xf32>{%0, %27, %64, %55} -> tensor<?x?x128xf32>{%67, %55}
      %69 = flow.dispatch @decode_dispatch_32::@decode_dispatch_32_elementwise_broadcast_DxDxDx128xD_f32[%5, %55, %27, %0, %64](%63, %5, %55, %27, %0, %64) : (tensor<?x?x?x128xf32>{%5, %55, %27}, index, index, index, index, index) -> tensor<?x?x?x128x?xf32>{%0, %27, %64, %55}
      %70 = flow.tensor.reshape %69 : tensor<?x?x?x128x?xf32>{%0, %27, %64, %55} -> tensor<?x128x?xf32>{%67, %55}
      %71 = arith.muli %0, %c16 overflow<nsw> : index
      %72 = flow.tensor.reshape %51 : tensor<?xf32>{%43} -> tensor<?x128xf32>{%71}
      %73 = flow.dispatch @decode_dispatch_33::@decode_dispatch_33_attention_Dx128xDx128[%71, %67, %55](%71, %72, %68, %70, %67, %55) : (index, tensor<?x128xf32>{%71}, tensor<?x?x128xf32>{%67, %55}, tensor<?x128x?xf32>{%67, %55}, index, index) -> tensor<?x128xf32>{%71}
      %74 = flow.dispatch @decode_dispatch_34::@decode_dispatch_34_elementwise_broadcast_D_f32[%43, %71](%43, %73, %71) : (index, tensor<?x128xf32>{%71}, index) -> tensor<?xf32>{%43}
      %75 = flow.tensor.reshape %74 : tensor<?xf32>{%43} -> tensor<?x2048xf32>{%0}
      %76 = flow.dispatch @decode_dispatch_35::@decode_dispatch_35_matmul_Dx2048x2048_f32[%0](%75, %21#5, %22, %0) : (tensor<?x2048xf32>{%0}, tensor<2048x2048xf32>, tensor<?x2048xf32>{%0}, index) -> tensor<?x2048xf32>{%0}
      %77 = hal.tensor.import %24 : !hal.buffer_view -> tensor<?x?x?x?xf32>{%25, %26, %c16, %c128}
      %78 = scf.for %arg8 = %c0 to %0 step %c1 iter_args(%arg9 = %77) -> (tensor<?x?x?x?xf32>) {
        %118 = flow.tensor.load %3[%arg8] : tensor<?xi64>{%2}
        %119 = arith.index_cast %118 : i64 to index
        %120 = arith.divui %119, %26 : index
        %121 = arith.remui %119, %26 : index
        %122 = flow.dispatch @decode_dispatch_36::@decode_dispatch_36_slow_memcpy[%arg6, %0, %arg8, %120, %121, %4, %5, %6, %25, %26, %c16, %c128](%7, %arg6, %0, %53, %arg8, %120, %arg9, %121, %4, %5, %6, %25, %26, %c16, %c128) : (tensor<?x?x?xi32>{%4, %5, %6}, index, index, tensor<?x16x128xf32>{%0}, index, index, tensor<?x?x?x?xf32>{%25, %26, %c16, %c128}, index, index, index, index, index, index, index, index) -> %arg9{%25, %26, %c16, %c128}
        scf.yield %122 : tensor<?x?x?x?xf32>
      }
      %79 = hal.tensor.export %78 : tensor<?x?x?x?xf32>{%25, %26, %c16, %c128} -> !hal.buffer_view
      %80 = hal.tensor.import %30 : !hal.buffer_view -> tensor<?x?x?x?xf32>{%25, %26, %c16, %c128}
      %81 = scf.for %arg8 = %c0 to %0 step %c1 iter_args(%arg9 = %80) -> (tensor<?x?x?x?xf32>) {
        %118 = flow.tensor.load %3[%arg8] : tensor<?xi64>{%2}
        %119 = arith.index_cast %118 : i64 to index
        %120 = arith.divui %119, %26 : index
        %121 = arith.remui %119, %26 : index
        %122 = flow.dispatch @decode_dispatch_36::@decode_dispatch_36_slow_memcpy[%arg6, %0, %arg8, %120, %121, %4, %5, %6, %25, %26, %c16, %c128](%7, %arg6, %0, %49, %arg8, %120, %arg9, %121, %4, %5, %6, %25, %26, %c16, %c128) : (tensor<?x?x?xi32>{%4, %5, %6}, index, index, tensor<?x16x128xf32>{%0}, index, index, tensor<?x?x?x?xf32>{%25, %26, %c16, %c128}, index, index, index, index, index, index, index, index) -> %arg9{%25, %26, %c16, %c128}
        scf.yield %122 : tensor<?x?x?x?xf32>
      }
      %82 = hal.tensor.export %81 : tensor<?x?x?x?xf32>{%25, %26, %c16, %c128} -> !hal.buffer_view
      util.list.set %arg2[%c0], %79 : !hal.buffer_view -> !util.list<?>
      util.list.set %arg2[%c1], %82 : !hal.buffer_view -> !util.list<?>
      %83 = flow.dispatch @prefill_dispatch_13::@prefill_dispatch_13_reduction_Dx2048_f32[%0](%76, %21#1, %0) : (tensor<?x2048xf32>{%0}, tensor<2048xf32>, index) -> tensor<?x2048xf32>{%0}
      %84 = flow.dispatch @prefill_dispatch_33::@prefill_dispatch_33_matmul_64xDx2048_f32[%0](%21#8, %83, %0) : (tensor<64x2048xf32>, tensor<?x2048xf32>{%0}, index) -> tensor<64x?xf32>{%0}
      %85 = flow.dispatch @prefill_dispatch_34::@prefill_dispatch_34_softmax_64xDxf32_dispatch_tensor_store[%0](%84, %0) : (tensor<64x?xf32>{%0}, index) -> tensor<64x?xf32>{%0}
      %86:2 = flow.dispatch @prefill_dispatch_35::@prefill_dispatch_35_topk_64xDxf32[%0](%85, %0) : (tensor<64x?xf32>{%0}, index) -> (tensor<8x?xf32>{%0}, tensor<8x?xi32>{%0})
      %87 = flow.tensor.reshape %21#9 : tensor<1024x2048x64xf32> -> tensor<2097152x64xf32>
      %88 = flow.dispatch @prefill_dispatch_36::@prefill_dispatch_36_transpose_2097152x64_f32(%87) : (tensor<2097152x64xf32>) -> tensor<64x2097152xf32>
      %89 = flow.tensor.reshape %88 : tensor<64x2097152xf32> -> tensor<64x1024x2048xf32>
      %90 = arith.muli %0, %c8 : index
      %91 = flow.dispatch @prefill_dispatch_37::@prefill_dispatch_37_gather_8xDx1024x2048xf32_dispatch_tensor_store[%0](%89, %86#1, %0) : (tensor<64x1024x2048xf32>, tensor<8x?xi32>{%0}, index) -> tensor<8x?x1024x2048xf32>{%0}
      %92 = arith.muli %0, %c8 overflow<nsw> : index
      %93 = flow.tensor.reshape %91 : tensor<8x?x1024x2048xf32>{%0} -> tensor<?x1024x2048xf32>{%92}
      %94 = flow.tensor.reshape %83 : tensor<?x2048xf32>{%0} -> tensor<?xf32>{%43}
      %95 = flow.dispatch @prefill_dispatch_38::@prefill_dispatch_38_elementwise_broadcast_8xD_f32[%43](%43, %94) : (index, tensor<?xf32>{%43}) -> tensor<8x?xf32>{%43}
      %96 = flow.tensor.reshape %95 : tensor<8x?xf32>{%43} -> tensor<?x2048x1xf32>{%90}
      %97 = flow.dispatch @prefill_dispatch_39::@prefill_dispatch_39_batch_matmul_Dx1024x1x2048_f32[%92, %90](%93, %96, %92, %90) : (tensor<?x1024x2048xf32>{%92}, tensor<?x2048x1xf32>{%90}, index, index) -> tensor<?x1024x1xf32>{%90}
      %98 = flow.tensor.reshape %21#10 : tensor<1024x2048x64xf32> -> tensor<2097152x64xf32>
      %99 = flow.dispatch @prefill_dispatch_36::@prefill_dispatch_36_transpose_2097152x64_f32(%98) : (tensor<2097152x64xf32>) -> tensor<64x2097152xf32>
      %100 = flow.tensor.reshape %99 : tensor<64x2097152xf32> -> tensor<64x1024x2048xf32>
      %101 = flow.dispatch @prefill_dispatch_37::@prefill_dispatch_37_gather_8xDx1024x2048xf32_dispatch_tensor_store[%0](%100, %86#1, %0) : (tensor<64x1024x2048xf32>, tensor<8x?xi32>{%0}, index) -> tensor<8x?x1024x2048xf32>{%0}
      %102 = flow.tensor.reshape %101 : tensor<8x?x1024x2048xf32>{%0} -> tensor<?x1024x2048xf32>{%92}
      %103 = flow.dispatch @prefill_dispatch_39::@prefill_dispatch_39_batch_matmul_Dx1024x1x2048_f32[%92, %90](%102, %96, %92, %90) : (tensor<?x1024x2048xf32>{%92}, tensor<?x2048x1xf32>{%90}, index, index) -> tensor<?x1024x1xf32>{%90}
      %104 = flow.tensor.reshape %21#11 : tensor<2048x1024x64xf32> -> tensor<2097152x64xf32>
      %105 = flow.dispatch @prefill_dispatch_36::@prefill_dispatch_36_transpose_2097152x64_f32(%104) : (tensor<2097152x64xf32>) -> tensor<64x2097152xf32>
      %106 = flow.tensor.reshape %105 : tensor<64x2097152xf32> -> tensor<64x2048x1024xf32>
      %107 = flow.dispatch @prefill_dispatch_44::@prefill_dispatch_44_gather_8xDx2048x1024xf32_dispatch_tensor_store[%0](%106, %86#1, %0) : (tensor<64x2048x1024xf32>, tensor<8x?xi32>{%0}, index) -> tensor<8x?x2048x1024xf32>{%0}
      %108 = flow.tensor.reshape %107 : tensor<8x?x2048x1024xf32>{%0} -> tensor<?x2048x1024xf32>{%92}
      %109 = arith.muli %0, %c8192 overflow<nsw> : index
      %110 = flow.tensor.reshape %103 : tensor<?x1024x1xf32>{%90} -> tensor<?xf32>{%109}
      %111 = flow.tensor.reshape %97 : tensor<?x1024x1xf32>{%90} -> tensor<?xf32>{%109}
      %112 = flow.dispatch @prefill_dispatch_45::@prefill_dispatch_45_elementwise_D_f32[%109](%109, %110, %111) : (index, tensor<?xf32>{%109}, tensor<?xf32>{%109}) -> tensor<?xf32>{%109}
      %113 = flow.tensor.reshape %112 : tensor<?xf32>{%109} -> tensor<?x1024x1xf32>{%90}
      %114 = flow.dispatch @prefill_dispatch_46::@prefill_dispatch_46_batch_matmul_Dx2048x1x1024_f32[%92, %90](%108, %113, %92, %90) : (tensor<?x2048x1024xf32>{%92}, tensor<?x1024x1xf32>{%90}, index, index) -> tensor<?x2048x1xf32>{%90}
      %115 = flow.tensor.reshape %114 : tensor<?x2048x1xf32>{%90} -> tensor<8x?x2048xf32>{%0}
      %116 = flow.dispatch @decode_dispatch_53::@decode_dispatch_53_matvec_like_Dx2048x8_f32[%0](%115, %86#0, %76, %0) : (tensor<8x?x2048xf32>{%0}, tensor<8x?xf32>{%0}, tensor<?x2048xf32>{%0}, index) -> tensor<?x2048xf32>{%0}
      %117 = flow.tensor.reshape %116 : tensor<?x2048xf32>{%0} -> tensor<?x?xf32>{%0, %c2048}
      scf.yield %117 : tensor<?x?xf32>
    }
    %14 = flow.tensor.reshape %13 : tensor<?x?xf32>{%0, %c2048} -> tensor<?x2048xf32>{%0}
    %15 = flow.dispatch @decode_dispatch_13::@decode_dispatch_13_elementwise_Dx2048_f32[%0, %c2048](%c2048_i32, %13, %14, %__parameter_model_output_norm_weight_tensor_2048xf32, %0, %c2048) : (i32, tensor<?x?xf32>{%0, %c2048}, tensor<?x2048xf32>{%0}, tensor<2048xf32>, index, index) -> tensor<?x2048xf32>{%0}
    %16 = flow.dispatch @decode_dispatch_55::@decode_dispatch_55_matmul_Dx50304x2048_f32[%0](%15, %__parameter_model_output_weight_tensor_2048x50304xf32, %0) : (tensor<?x2048xf32>{%0}, tensor<2048x50304xf32>, index) -> tensor<?x50304xf32>{%0}
    %17 = flow.tensor.reshape %16 : tensor<?x50304xf32>{%0} -> tensor<?x?xf32>{%0, %c50304}
    %18 = hal.tensor.export %17 "output0" : tensor<?x?xf32>{%0, %c50304} -> !hal.buffer_view
    util.return %18, %arg2 : !hal.buffer_view, !util.list<?>
  }
}