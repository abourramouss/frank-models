// Repro: scf.while with KV cache via util.list + hal.tensor.import/export.
// Each iteration imports buffer views, gathers with growing size,
// scatters back, and exports. Crashes at ~31 iterations.

!elem_t = f32

module @dynamic_loop_repro {

  util.func public @run_loop(%max_iters: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : !elem_t
    %true = arith.constant true
    %affinity = arith.constant -1 : i64

    // Allocate K and V cache buffers [512] each, store in util.list
    %elem_sz = util.sizeof !elem_t
    %byte_size = arith.muli %c512, %elem_sz : index

    %device = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %mem_type = hal.memory_type<"DeviceLocal"> : i32
    %buf_usage = hal.buffer_usage<"TransferSource|TransferTarget|DispatchStorageRead|DispatchStorageWrite"> : i32
    %elem_type = hal.element_type<!elem_t> : i32
    %enc_type = hal.encoding_type<dense_row_major> : i32

    %k_buf = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%mem_type) usage(%buf_usage) : !hal.buffer{%byte_size}
    %k_bv = hal.buffer_view.create buffer(%k_buf : !hal.buffer)[%c0, %byte_size]
        shape([%c512]) type(%elem_type) encoding(%enc_type) : !hal.buffer_view

    %v_buf = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%mem_type) usage(%buf_usage) : !hal.buffer{%byte_size}
    %v_bv = hal.buffer_view.create buffer(%v_buf : !hal.buffer)[%c0, %byte_size]
        shape([%c512]) type(%elem_type) encoding(%enc_type) : !hal.buffer_view

    %cache = util.list.create %c2 : !util.list<?>
    util.list.resize %cache, %c2 : !util.list<?>
    util.list.set %cache[%c0], %k_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %v_bv : !hal.buffer_view -> !util.list<?>

    // Loop: each iteration imports cache, gathers [0:pos], computes,
    // scatters new value at pos, exports back.
    %result_val, %result_pos, %result_cache, %result_cont = scf.while (
        %acc = %zero,
        %pos = %c1,
        %cache_w = %cache,
        %cont = %true
    ) : (!elem_t, index, !util.list<?>, i1) -> (!elem_t, index, !util.list<?>, i1) {
      %under = arith.cmpi ult, %pos, %max_iters : index
      %go = arith.andi %cont, %under : i1
      scf.condition(%go) %acc, %pos, %cache_w, %cont : !elem_t, index, !util.list<?>, i1
    } do {
    ^bb0(%acc_d: !elem_t, %pos_d: index, %cache_d: !util.list<?>, %cont_d: i1):

      // Import K cache
      %k_bv_in = util.list.get %cache_d[%c0] : !util.list<?> -> !hal.buffer_view
      %k_tensor = hal.tensor.import %k_bv_in : !hal.buffer_view -> tensor<?x!elem_t>{%c512}

      // Gather: read [0:pos] using tensor.extract (breaks provenance)
      %gathered_init = tensor.empty(%pos_d) : tensor<?x!elem_t>
      %gathered = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } outs(%gathered_init : tensor<?x!elem_t>) {
      ^bb0(%out: !elem_t):
        %i = linalg.index 0 : index
        %val = tensor.extract %k_tensor[%i] : tensor<?x!elem_t>
        linalg.yield %val : !elem_t
      } -> tensor<?x!elem_t>

      // Reduce gathered values
      %red_init = tensor.empty() : tensor<!elem_t>
      %red_fill = linalg.fill ins(%zero : !elem_t) outs(%red_init : tensor<!elem_t>) -> tensor<!elem_t>
      %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]
      } ins(%gathered : tensor<?x!elem_t>) outs(%red_fill : tensor<!elem_t>) {
      ^bb0(%in: !elem_t, %a: !elem_t):
        %s = arith.addf %in, %a : !elem_t
        linalg.yield %s : !elem_t
      } -> tensor<!elem_t>
      %sum_val = tensor.extract %reduced[] : tensor<!elem_t>

      // Scatter: write sum at position pos
      %new_val_tensor = tensor.from_elements %sum_val : tensor<1x!elem_t>
      %new_val_dyn = tensor.cast %new_val_tensor : tensor<1x!elem_t> to tensor<?x!elem_t>
      %k_updated = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } outs(%k_tensor : tensor<?x!elem_t>) {
      ^bb0(%existing: !elem_t):
        %i = linalg.index 0 : index
        %is_pos = arith.cmpi eq, %i, %pos_d : index
        %new = tensor.extract %new_val_dyn[%c0] : tensor<?x!elem_t>
        %result = arith.select %is_pos, %new, %existing : !elem_t
        linalg.yield %result : !elem_t
      } -> tensor<?x!elem_t>

      // Export back
      %k_bv_out = hal.tensor.export %k_updated : tensor<?x!elem_t>{%c512} -> !hal.buffer_view
      util.list.set %cache_d[%c0], %k_bv_out : !hal.buffer_view -> !util.list<?>

      %new_acc = arith.addf %acc_d, %sum_val : !elem_t
      %new_pos = arith.addi %pos_d, %c1 : index

      scf.yield %new_acc, %new_pos, %cache_d, %cont_d : !elem_t, index, !util.list<?>, i1
    }

    util.return %result_val : !elem_t
  }
}
