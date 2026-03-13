// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Multi-Head Attention using iree_linalg_ext.attention.
// Supports GQA: n_head_kv may differ from n_head (Q heads).
// KV heads are repeated to match Q heads via the attention indexing maps.
//
// TODO(#4): This component lacks causal masking, making it unsuitable for
// autoregressive (decoder-only) models. The model will produce incorrect
// outputs because each position can attend to future positions.
// See: https://github.com/stellaraccident/frank-models/issues/4
//
// Usage:
//   %output = call @attention_gqa(%query, %key, %value, %scale)
//       : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32)
//       -> tensor<?x?x?x?xf32>

module @attention_components {

  util.func public @attention_gqa(
      %query: tensor<?x?x?x?xf32>,   // [batch, seq_len, n_head, head_dim]
      %key: tensor<?x?x?x?xf32>,     // [batch, seq_len, n_head, head_dim]
      %value: tensor<?x?x?x?xf32>,   // [batch, seq_len, n_head, head_dim]
      %scale: f32                     // 1.0 / sqrt(head_dim)
  ) -> tensor<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %batch = tensor.dim %query, %c0 : tensor<?x?x?x?xf32>
    %seq_len = tensor.dim %query, %c1 : tensor<?x?x?x?xf32>
    %n_head = tensor.dim %query, %c2 : tensor<?x?x?x?xf32>
    %head_dim = tensor.dim %query, %c3 : tensor<?x?x?x?xf32>
    %n_head_kv = tensor.dim %key, %c2 : tensor<?x?x?x?xf32>
    // K/V may have a different sequence length than Q (decode: seq_q=1, seq_kv=ctx+1).
    %seq_len_kv = tensor.dim %key, %c1 : tensor<?x?x?x?xf32>

    // Transpose Q/K/V from [batch, seq, n_head, head_dim] to [batch, n_head, seq, head_dim]
    %q_transposed_init = tensor.empty(%batch, %n_head, %seq_len, %head_dim) : tensor<?x?x?x?xf32>
    %q_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%query : tensor<?x?x?x?xf32>) outs(%q_transposed_init : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?xf32>

    %k_transposed_init = tensor.empty(%batch, %n_head_kv, %seq_len_kv, %head_dim) : tensor<?x?x?x?xf32>
    %k_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%key : tensor<?x?x?x?xf32>) outs(%k_transposed_init : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?xf32>

    %v_transposed_init = tensor.empty(%batch, %n_head_kv, %seq_len_kv, %head_dim) : tensor<?x?x?x?xf32>
    %v_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%value : tensor<?x?x?x?xf32>) outs(%v_transposed_init : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?xf32>

    // GQA: Repeat K/V heads to match Q heads.
    // Strategy: broadcast to [batch, n_head_kv, repeat_factor, seq, head_dim],
    // then collapse [1,2] to get [batch, n_head, seq, head_dim].
    // This ensures each KV head is repeated together: [H0, H0, H1, H1] not [H0, H1, H0, H1].
    %repeat_factor = arith.divui %n_head, %n_head_kv : index

    // Allocate 5D output for K broadcast: [batch, n_head_kv, repeat_factor, seq, head_dim].
    %k_broadcast_init = tensor.empty(%batch, %n_head_kv, %repeat_factor, %seq_len_kv, %head_dim) : tensor<?x?x?x?x?xf32>

    // Broadcast K along repeat_factor dimension via linalg.generic.
    // Input map omits d2 (repeat_factor), causing each K[batch, kv_head, seq, head_dim]
    // to be broadcast across all repeat_factor positions.
    %k_broadcast = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,      // Input: omits d2
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>   // Output: has d2
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%k_transposed : tensor<?x?x?x?xf32>) outs(%k_broadcast_init : tensor<?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?x?xf32>

    // Collapse n_head_kv and repeat_factor into n_head dimension.
    // This is a zero-cost metadata operation that fuses with surrounding ops.
    %k_repeated = tensor.collapse_shape %k_broadcast [[0], [1, 2], [3], [4]]
        : tensor<?x?x?x?x?xf32> into tensor<?x?x?x?xf32>

    // Repeat V using the same broadcast+collapse pattern.
    %v_broadcast_init = tensor.empty(%batch, %n_head_kv, %repeat_factor, %seq_len_kv, %head_dim) : tensor<?x?x?x?x?xf32>
    %v_broadcast = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%v_transposed : tensor<?x?x?x?xf32>) outs(%v_broadcast_init : tensor<?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?x?xf32>
    %v_repeated = tensor.collapse_shape %v_broadcast [[0], [1, 2], [3], [4]]
        : tensor<?x?x?x?x?xf32> into tensor<?x?x?x?xf32>

    // Collapse [batch, n_head, seq, head_dim] to [batch * n_head, seq, head_dim]
    %batch_heads = arith.muli %batch, %n_head : index
    %q_3d = tensor.collapse_shape %q_transposed [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
    %k_3d = tensor.collapse_shape %k_repeated [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
    %v_3d = tensor.collapse_shape %v_repeated [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>

    // Run attention: [batch * n_head, seq, head_dim]
    %output_3d_init = tensor.empty(%batch_heads, %seq_len, %head_dim) : tensor<?x?x?xf32>
    %output_3d = iree_linalg_ext.attention {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,  // Query
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,  // Key
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,  // Value
        affine_map<(d0, d1, d2, d3, d4) -> ()>,            // Scale
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>   // Output
      ]
    } ins(%q_3d, %k_3d, %v_3d, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32)
      outs(%output_3d_init : tensor<?x?x?xf32>) {
    ^bb0(%arg0: f32):
      iree_linalg_ext.yield %arg0 : f32
    } -> tensor<?x?x?xf32>

    // Reshape [batch*n_head, seq, head_dim] → [batch, seq, n_head, head_dim].
    // Combines un-collapsing batch*n_head and the n_head↔seq transpose into one generic.
    // Uses linalg.index + tensor.extract instead of tensor.expand_shape to avoid an IREE
    // GlobalOpt bug: expanding dynamic dim 0 (batch*n_head) into [batch, n_head=static]
    // produces static_output_shape = array<i64> (empty) → verifier crash.
    %output_init = tensor.empty(%batch, %seq_len, %n_head, %head_dim) : tensor<?x?x?x?xf32>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%output_init : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      %i0 = linalg.index 0 : index  // batch
      %i1 = linalg.index 1 : index  // seq
      %i2 = linalg.index 2 : index  // n_head
      %i3 = linalg.index 3 : index  // head_dim
      // Flat batch-head index: i0 * n_head + i2
      %flat_head = arith.muli %i0, %n_head : index
      %flat_idx = arith.addi %flat_head, %i2 : index
      // output_3d is [batch*n_head, seq, head_dim]
      %val = tensor.extract %output_3d[%flat_idx, %i1, %i3] : tensor<?x?x?xf32>
      linalg.yield %val : f32
    } -> tensor<?x?x?x?xf32>

    util.return %output : tensor<?x?x?x?xf32>
  }

}
