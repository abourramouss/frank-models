// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tensor-based Paged KV Cache — no HAL imports/exports, no sync barriers.
//
// K and V caches flow as regular tensors through the computation graph.
// This eliminates the !util.list<?> / hal.tensor.import/export sync points
// that were the main performance bottleneck.
//
// Physical cache layout:
//   K/V: [n_blocks, block_size, n_head_kv, head_dim]
//
// Interface:
//   allocate(n_blocks, block_size, n_head_kv, head_dim) -> (K_cache, V_cache)
//   gather(K_cache, V_cache, layer, block_tables, context_lens, max_context_len) -> (K, V)
//   scatter_decode(K_cache, V_cache, layer, new_k, new_v, block_tables, positions) -> (K_cache, V_cache)
//   scatter_prefill(K_cache, V_cache, layer, new_k, new_v, block_tables, start_positions, block_size) -> (K_cache, V_cache)

!elem_t = f16

module @kvcache_components {

  // Allocate KV cache as two zero-initialized tensors.
  util.func public @allocate(
      %n_blocks: index,
      %block_size: index,
      %n_head_kv: index,
      %head_dim: index
  ) -> (tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>) {
    %zero = arith.constant 0.0 : !elem_t
    %k_init = tensor.empty(%n_blocks, %block_size, %n_head_kv, %head_dim) : tensor<?x?x?x?x!elem_t>
    %k_cache = linalg.fill ins(%zero : !elem_t) outs(%k_init : tensor<?x?x?x?x!elem_t>) -> tensor<?x?x?x?x!elem_t>
    %v_init = tensor.empty(%n_blocks, %block_size, %n_head_kv, %head_dim) : tensor<?x?x?x?x!elem_t>
    %v_cache = linalg.fill ins(%zero : !elem_t) outs(%v_init : tensor<?x?x?x?x!elem_t>) -> tensor<?x?x?x?x!elem_t>
    util.return %k_cache, %v_cache : tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>
  }

  // Gather K and V for attention.
  util.func public @gather(
      %k_cache: tensor<?x?x?x?x!elem_t>,   // [n_blocks, block_size, n_head_kv, head_dim]
      %v_cache: tensor<?x?x?x?x!elem_t>,
      %layer: index,
      %block_tables: tensor<?x?x?xi32>,
      %context_lens: tensor<?x?xi32>,
      %max_context_len: index
  ) -> (tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.0 : !elem_t

    %batch = tensor.dim %block_tables, %c1 : tensor<?x?x?xi32>
    %max_blocks = tensor.dim %block_tables, %c2 : tensor<?x?x?xi32>
    %block_size = tensor.dim %k_cache, %c1 : tensor<?x?x?x?x!elem_t>
    %n_head_kv = tensor.dim %k_cache, %c2 : tensor<?x?x?x?x!elem_t>
    %head_dim = tensor.dim %k_cache, %c0 : tensor<?x?x?x?x!elem_t>
    // head_dim is actually dim 3
    %head_dim_actual = tensor.dim %k_cache, %c2 : tensor<?x?x?x?x!elem_t>

    // Slice block_tables for this layer
    %block_tables_layer = tensor.extract_slice %block_tables[%layer, 0, 0] [1, %batch, %max_blocks] [1, 1, 1]
      : tensor<?x?x?xi32> to tensor<1x?x?xi32>
    %block_tables_2d = tensor.collapse_shape %block_tables_layer [[0, 1], [2]]
      : tensor<1x?x?xi32> into tensor<?x?xi32>

    %context_lens_layer = tensor.extract_slice %context_lens[%layer, 0] [1, %batch] [1, 1]
      : tensor<?x?xi32> to tensor<1x?xi32>
    %context_lens_1d = tensor.collapse_shape %context_lens_layer [[0, 1]]
      : tensor<1x?xi32> into tensor<?xi32>

    // Re-derive correct dims from cache shape
    %n_head_kv_c = tensor.dim %k_cache, %c2 : tensor<?x?x?x?x!elem_t>
    %c3 = arith.constant 3 : index
    %head_dim_c = tensor.dim %k_cache, %c3 : tensor<?x?x?x?x!elem_t>

    %k_init = tensor.empty(%batch, %max_context_len, %n_head_kv_c, %head_dim_c) : tensor<?x?x?x?x!elem_t>
    %v_init = tensor.empty(%batch, %max_context_len, %n_head_kv_c, %head_dim_c) : tensor<?x?x?x?x!elem_t>

    // Gather K
    %k_gathered = linalg.generic {
      indexing_maps = [
        affine_map<(b, ctx, head, dim) -> (b)>,
        affine_map<(b, ctx, head, dim) -> (b, ctx, head, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%context_lens_1d : tensor<?xi32>)
      outs(%k_init : tensor<?x?x?x?x!elem_t>) {
    ^bb0(%ctx_len_i32: i32, %out: !elem_t):
      %b_idx = linalg.index 0 : index
      %ctx_idx = linalg.index 1 : index
      %head_idx = linalg.index 2 : index
      %dim_idx = linalg.index 3 : index

      %logical_block = arith.divui %ctx_idx, %block_size : index
      %pos_in_block = arith.remui %ctx_idx, %block_size : index

      %physical_block_i32 = tensor.extract %block_tables_2d[%b_idx, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      %k_val = tensor.extract %k_cache[%physical_block, %pos_in_block, %head_idx, %dim_idx]
        : tensor<?x?x?x?x!elem_t>

      %ctx_len = arith.index_cast %ctx_len_i32 : i32 to index
      %in_range = arith.cmpi ult, %ctx_idx, %ctx_len : index
      %result = arith.select %in_range, %k_val, %zero : !elem_t

      linalg.yield %result : !elem_t
    } -> tensor<?x?x?x?x!elem_t>

    // Gather V
    %v_gathered = linalg.generic {
      indexing_maps = [
        affine_map<(b, ctx, head, dim) -> (b)>,
        affine_map<(b, ctx, head, dim) -> (b, ctx, head, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%context_lens_1d : tensor<?xi32>)
      outs(%v_init : tensor<?x?x?x?x!elem_t>) {
    ^bb0(%ctx_len_i32: i32, %out: !elem_t):
      %b_idx = linalg.index 0 : index
      %ctx_idx = linalg.index 1 : index
      %head_idx = linalg.index 2 : index
      %dim_idx = linalg.index 3 : index

      %logical_block = arith.divui %ctx_idx, %block_size : index
      %pos_in_block = arith.remui %ctx_idx, %block_size : index

      %physical_block_i32 = tensor.extract %block_tables_2d[%b_idx, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      %v_val = tensor.extract %v_cache[%physical_block, %pos_in_block, %head_idx, %dim_idx]
        : tensor<?x?x?x?x!elem_t>

      %ctx_len = arith.index_cast %ctx_len_i32 : i32 to index
      %in_range = arith.cmpi ult, %ctx_idx, %ctx_len : index
      %result = arith.select %in_range, %v_val, %zero : !elem_t

      linalg.yield %result : !elem_t
    } -> tensor<?x?x?x?x!elem_t>

    util.return %k_gathered, %v_gathered : tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>
  }

  // Scatter one new token per sequence (decode phase).
  util.func public @scatter_decode(
      %k_cache: tensor<?x?x?x?x!elem_t>,
      %v_cache: tensor<?x?x?x?x!elem_t>,
      %layer: index,
      %new_k: tensor<?x?x?x!elem_t>,       // [batch, n_head_kv, head_dim]
      %new_v: tensor<?x?x?x!elem_t>,
      %block_tables: tensor<?x?x?xi32>,
      %positions: tensor<?xi64>
  ) -> (tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %batch_size = tensor.dim %new_k, %c0 : tensor<?x?x?x!elem_t>
    %n_head_kv = tensor.dim %new_k, %c1 : tensor<?x?x?x!elem_t>
    %head_dim = tensor.dim %new_k, %c2 : tensor<?x?x?x!elem_t>

    %max_blocks = tensor.dim %block_tables, %c2 : tensor<?x?x?xi32>
    %block_tables_layer = tensor.extract_slice %block_tables[%layer, 0, 0] [1, %batch_size, %max_blocks] [1, 1, 1]
      : tensor<?x?x?xi32> to tensor<1x?x?xi32>
    %block_tables_2d = tensor.collapse_shape %block_tables_layer [[0, 1], [2]]
      : tensor<1x?x?xi32> into tensor<?x?xi32>

    %n_blocks = tensor.dim %k_cache, %c0 : tensor<?x?x?x?x!elem_t>
    %block_size = tensor.dim %k_cache, %c1 : tensor<?x?x?x?x!elem_t>

    // Scatter K
    %k_updated = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%cache_k = %k_cache) -> (tensor<?x?x?x?x!elem_t>) {
      %abs_pos_i64 = tensor.extract %positions[%i] : tensor<?xi64>
      %abs_pos = arith.index_cast %abs_pos_i64 : i64 to index
      %logical_block = arith.divui %abs_pos, %block_size : index
      %pos_in_block = arith.remui %abs_pos, %block_size : index

      %physical_block_i32 = tensor.extract %block_tables_2d[%i, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      %new_k_slice = tensor.extract_slice %new_k[%i, 0, 0] [1, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?x?x!elem_t> to tensor<1x?x?x!elem_t>

      %updated = tensor.insert_slice %new_k_slice into %cache_k[%physical_block, %pos_in_block, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %updated : tensor<?x?x?x?x!elem_t>
    }

    %k_result = flow.tensor.tie_shape %k_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Scatter V
    %v_updated = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%cache_v = %v_cache) -> (tensor<?x?x?x?x!elem_t>) {
      %abs_pos_i64 = tensor.extract %positions[%i] : tensor<?xi64>
      %abs_pos = arith.index_cast %abs_pos_i64 : i64 to index
      %logical_block = arith.divui %abs_pos, %block_size : index
      %pos_in_block = arith.remui %abs_pos, %block_size : index

      %physical_block_i32 = tensor.extract %block_tables_2d[%i, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      %new_v_slice = tensor.extract_slice %new_v[%i, 0, 0] [1, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?x?x!elem_t> to tensor<1x?x?x!elem_t>

      %updated = tensor.insert_slice %new_v_slice into %cache_v[%physical_block, %pos_in_block, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %updated : tensor<?x?x?x?x!elem_t>
    }

    %v_result = flow.tensor.tie_shape %v_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    util.return %k_result, %v_result : tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>
  }

  // Scatter multiple tokens (prefill phase).
  util.func public @scatter_prefill(
      %k_cache: tensor<?x?x?x?x!elem_t>,
      %v_cache: tensor<?x?x?x?x!elem_t>,
      %layer: index,
      %new_k: tensor<?x?x?x?x!elem_t>,    // [batch, seq_len, n_head_kv, head_dim]
      %new_v: tensor<?x?x?x?x!elem_t>,
      %block_tables: tensor<?x?x?xi32>,
      %start_positions: tensor<?xi32>,
      %block_size: index
  ) -> (tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %batch = tensor.dim %new_k, %c0 : tensor<?x?x?x?x!elem_t>
    %seq_len = tensor.dim %new_k, %c1 : tensor<?x?x?x?x!elem_t>
    %n_head_kv = tensor.dim %new_k, %c2 : tensor<?x?x?x?x!elem_t>
    %head_dim = tensor.dim %new_k, %c3 : tensor<?x?x?x?x!elem_t>

    %max_blocks = tensor.dim %block_tables, %c2 : tensor<?x?x?xi32>
    %block_tables_layer = tensor.extract_slice %block_tables[%layer, 0, 0] [1, %batch, %max_blocks] [1, 1, 1]
      : tensor<?x?x?xi32> to tensor<1x?x?xi32>
    %block_tables_2d = tensor.collapse_shape %block_tables_layer [[0, 1], [2]]
      : tensor<1x?x?xi32> into tensor<?x?xi32>

    %n_blocks = tensor.dim %k_cache, %c0 : tensor<?x?x?x?x!elem_t>
    %cache_block_size = tensor.dim %k_cache, %c1 : tensor<?x?x?x?x!elem_t>
    %total_positions = arith.muli %batch, %seq_len : index

    // Scatter K
    %k_updated = scf.for %flat_idx = %c0 to %total_positions step %c1
        iter_args(%cache_k = %k_cache) -> (tensor<?x?x?x?x!elem_t>) {
      %b = arith.divui %flat_idx, %seq_len : index
      %s = arith.remui %flat_idx, %seq_len : index

      %start_pos_i32 = tensor.extract %start_positions[%b] : tensor<?xi32>
      %start_pos = arith.index_cast %start_pos_i32 : i32 to index
      %abs_pos = arith.addi %start_pos, %s : index
      %logical_block = arith.divui %abs_pos, %block_size : index
      %pos_in_block = arith.remui %abs_pos, %block_size : index

      %physical_block_i32 = tensor.extract %block_tables_2d[%b, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      %new_k_slice = tensor.extract_slice %new_k[%b, %s, 0, 0] [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<?x?x?x?x!elem_t> to tensor<1x1x?x?x!elem_t>
      %new_k_3d = tensor.collapse_shape %new_k_slice [[0, 1], [2], [3]]
        : tensor<1x1x?x?x!elem_t> into tensor<1x?x?x!elem_t>

      %result = tensor.insert_slice %new_k_3d into %cache_k[%physical_block, %pos_in_block, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %result : tensor<?x?x?x?x!elem_t>
    }

    %k_result = flow.tensor.tie_shape %k_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim}

    // Scatter V
    %v_updated = scf.for %flat_idx = %c0 to %total_positions step %c1
        iter_args(%cache_v = %v_cache) -> (tensor<?x?x?x?x!elem_t>) {
      %b = arith.divui %flat_idx, %seq_len : index
      %s = arith.remui %flat_idx, %seq_len : index

      %start_pos_i32 = tensor.extract %start_positions[%b] : tensor<?xi32>
      %start_pos = arith.index_cast %start_pos_i32 : i32 to index
      %abs_pos = arith.addi %start_pos, %s : index
      %logical_block = arith.divui %abs_pos, %block_size : index
      %pos_in_block = arith.remui %abs_pos, %block_size : index

      %physical_block_i32 = tensor.extract %block_tables_2d[%b, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      %new_v_slice = tensor.extract_slice %new_v[%b, %s, 0, 0] [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<?x?x?x?x!elem_t> to tensor<1x1x?x?x!elem_t>
      %new_v_3d = tensor.collapse_shape %new_v_slice [[0, 1], [2], [3]]
        : tensor<1x1x?x?x!elem_t> into tensor<1x?x?x!elem_t>

      %result = tensor.insert_slice %new_v_3d into %cache_v[%physical_block, %pos_in_block, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %result : tensor<?x?x?x?x!elem_t>
    }

    %v_result = flow.tensor.tie_shape %v_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim}

    util.return %k_result, %v_result : tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>
  }

}
