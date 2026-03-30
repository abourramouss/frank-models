// Direct contiguous KV cache — zero-copy like the CUDA version.
//
// Cache layout: [total_slots, n_head_kv, head_dim] where
//   total_slots = n_layers * max_seq_len
//   Layer l at position p → slot = l * max_seq_len + p
//
// No paging, no block tables, no gather/scatter.
// Read: extract_slice from [slot_start..slot_start+ctx_len]
// Write: insert_slice at [slot]

!elem_t = f16

module @kvcache_direct_components {

  // Allocate flat K and V buffers.
  // total_slots = n_layers * max_seq_len
  util.func public @allocate(
      %total_slots: index,
      %n_head_kv: index,
      %head_dim: index
  ) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %element_size = util.sizeof !elem_t
    %affinity = arith.constant -1 : i64

    %d0 = arith.muli %total_slots, %n_head_kv : index
    %d1 = arith.muli %d0, %head_dim : index
    %byte_size = arith.muli %d1, %element_size : index

    %device = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %memory_type = hal.memory_type<"DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|DispatchStorageRead|DispatchStorageWrite"> : i32
    %element_type = hal.element_type<!elem_t> : i32
    %encoding_type = hal.encoding_type<dense_row_major> : i32

    %k_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%byte_size}
    %k_bv = hal.buffer_view.create buffer(%k_buffer : !hal.buffer)[%c0, %byte_size]
        shape([%total_slots, %n_head_kv, %head_dim])
        type(%element_type)
        encoding(%encoding_type) : !hal.buffer_view

    %v_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%byte_size}
    %v_bv = hal.buffer_view.create buffer(%v_buffer : !hal.buffer)[%c0, %byte_size]
        shape([%total_slots, %n_head_kv, %head_dim])
        type(%element_type)
        encoding(%encoding_type) : !hal.buffer_view

    %list = util.list.create %c2 : !util.list<?>
    util.list.resize %list, %c2 : !util.list<?>
    util.list.set %list[%c0], %k_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %list[%c1], %v_bv : !hal.buffer_view -> !util.list<?>
    util.return %list : !util.list<?>
  }

  // Read K/V for layer at [slot_start..slot_start+ctx_len).
  // slot_start = layer * max_seq_len
  // Returns [1, ctx_len, n_head_kv, head_dim] tensors.
  util.func public @read(
      %cache: !util.list<?>,
      %slot_start: index,
      %ctx_len: index
  ) -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view

    %total_slots = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %n_head_kv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %head_dim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index

    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total_slots, %n_head_kv, %head_dim}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total_slots, %n_head_kv, %head_dim}

    // Direct slice — no indirection
    %k_slice = tensor.extract_slice %k_full[%slot_start, 0, 0] [%ctx_len, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?x?xf16> to tensor<?x?x?xf16>
    %v_slice = tensor.extract_slice %v_full[%slot_start, 0, 0] [%ctx_len, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?x?xf16> to tensor<?x?x?xf16>

    // Add batch dim: [ctx, heads, dim] -> [1, ctx, heads, dim]
    %k_out = tensor.expand_shape %k_slice [[0, 1], [2], [3]]
        output_shape [%c1, %ctx_len, %n_head_kv, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %v_out = tensor.expand_shape %v_slice [[0, 1], [2], [3]]
        output_shape [%c1, %ctx_len, %n_head_kv, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    util.return %k_out, %v_out : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
  }

  // Write new K/V at a single position.
  // slot = layer * max_seq_len + pos
  // new_k, new_v: [batch=1, n_head_kv, head_dim] (collapsed from [1, 1, heads, dim])
  util.func public @write(
      %cache: !util.list<?>,
      %slot: index,
      %new_k: tensor<?x?xf16>,
      %new_v: tensor<?x?xf16>
  ) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view

    %total_slots = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %n_head_kv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %head_dim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index

    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total_slots, %n_head_kv, %head_dim}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total_slots, %n_head_kv, %head_dim}

    // Direct insert — just pointer arithmetic equivalent
    %k_updated = tensor.insert_slice %new_k into %k_full[%slot, 0, 0] [1, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %v_updated = tensor.insert_slice %new_v into %v_full[%slot, 0, 0] [1, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    %k_out_bv = hal.tensor.export %k_updated : tensor<?x?x?xf16>{%total_slots, %n_head_kv, %head_dim} -> !hal.buffer_view
    %v_out_bv = hal.tensor.export %v_updated : tensor<?x?x?xf16>{%total_slots, %n_head_kv, %head_dim} -> !hal.buffer_view

    util.list.set %cache[%c0], %k_out_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %v_out_bv : !hal.buffer_view -> !util.list<?>

    util.return %cache : !util.list<?>
  }

}
