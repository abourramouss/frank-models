// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Multi-Head Attention using iree_linalg_ext.attention.
// Supports GQA: n_head_kv may differ from n_head (Q heads).
// KV heads are repeated to match Q heads via the attention indexing maps.
//
// Causal masking: attend if key_pos <= query_pos + (seq_kv - seq_q).
// Prefill (seq_q == seq_kv): lower-triangular mask.
// Decode (seq_q == 1, seq_kv == context_len): all-true (all past positions).
//
// Usage:
//   %output = call @attention_gqa(%query, %key, %value, %scale)
//       : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f32)
//       -> tensor<?x?x?x?xf16>

module @attention_components {

  util.func public @attention_gqa(
      %query: tensor<?x?x?x?xf16>,   // [batch, seq_len, n_head, head_dim]
      %key: tensor<?x?x?x?xf16>,     // [batch, seq_len, n_head, head_dim]
      %value: tensor<?x?x?x?xf16>,   // [batch, seq_len, n_head, head_dim]
      %scale: f32                     // 1.0 / sqrt(head_dim)
  ) -> tensor<?x?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %batch = tensor.dim %query, %c0 : tensor<?x?x?x?xf16>
    %seq_len = tensor.dim %query, %c1 : tensor<?x?x?x?xf16>
    %n_head = tensor.dim %query, %c2 : tensor<?x?x?x?xf16>
    %head_dim = tensor.dim %query, %c3 : tensor<?x?x?x?xf16>
    %n_head_kv = tensor.dim %key, %c2 : tensor<?x?x?x?xf16>
    %seq_len_kv = tensor.dim %key, %c1 : tensor<?x?x?x?xf16>

    // Transpose Q/K/V from [batch, seq, n_head, head_dim] to [batch, n_head, seq, head_dim]
    %q_transposed_init = tensor.empty(%batch, %n_head, %seq_len, %head_dim) : tensor<?x?x?x?xf16>
    %q_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%query : tensor<?x?x?x?xf16>) outs(%q_transposed_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %k_transposed_init = tensor.empty(%batch, %n_head_kv, %seq_len_kv, %head_dim) : tensor<?x?x?x?xf16>
    %k_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%key : tensor<?x?x?x?xf16>) outs(%k_transposed_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %v_transposed_init = tensor.empty(%batch, %n_head_kv, %seq_len_kv, %head_dim) : tensor<?x?x?x?xf16>
    %v_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%value : tensor<?x?x?x?xf16>) outs(%v_transposed_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // GQA: Repeat K/V heads to match Q heads.
    %repeat_factor = arith.divui %n_head, %n_head_kv : index

    %k_broadcast_init = tensor.empty(%batch, %n_head_kv, %repeat_factor, %seq_len_kv, %head_dim) : tensor<?x?x?x?x?xf16>
    %k_broadcast = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%k_transposed : tensor<?x?x?x?xf16>) outs(%k_broadcast_init : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %k_repeated = tensor.collapse_shape %k_broadcast [[0], [1, 2], [3], [4]]
        : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>

    %v_broadcast_init = tensor.empty(%batch, %n_head_kv, %repeat_factor, %seq_len_kv, %head_dim) : tensor<?x?x?x?x?xf16>
    %v_broadcast = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%v_transposed : tensor<?x?x?x?xf16>) outs(%v_broadcast_init : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %v_repeated = tensor.collapse_shape %v_broadcast [[0], [1, 2], [3], [4]]
        : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>

    // Collapse [batch, n_head, seq, head_dim] to [batch * n_head, seq, head_dim]
    %batch_heads = arith.muli %batch, %n_head : index
    %q_3d = tensor.collapse_shape %q_transposed [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %k_3d = tensor.collapse_shape %k_repeated [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %v_3d = tensor.collapse_shape %v_repeated [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>

    // Generate causal mask
    %offset = arith.subi %seq_len_kv, %seq_len : index
    %mask_init = tensor.empty(%batch_heads, %seq_len, %seq_len_kv) : tensor<?x?x?xi1>
    %causal_mask = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%mask_init : tensor<?x?x?xi1>) {
    ^bb0(%out: i1):
      %qi = linalg.index 1 : index
      %kj = linalg.index 2 : index
      %qi_shifted = arith.addi %qi, %offset : index
      %in_bounds = arith.cmpi ule, %kj, %qi_shifted : index
      linalg.yield %in_bounds : i1
    } -> tensor<?x?x?xi1>

    // Run attention with causal mask — f16 Q/K/V, f32 internal accumulation
    %output_3d_init = tensor.empty(%batch_heads, %seq_len, %head_dim) : tensor<?x?x?xf16>
    %output_3d = iree_linalg_ext.attention {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4) -> ()>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
      ]
    } ins(%q_3d, %k_3d, %v_3d, %scale, %causal_mask : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f32, tensor<?x?x?xi1>)
      outs(%output_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%arg0: f32):
      iree_linalg_ext.yield %arg0 : f32
    } -> tensor<?x?x?xf16>

    // Reshape [batch*n_head, seq, head_dim] -> [batch, n_head, seq, head_dim]
    %output_4d = tensor.expand_shape %output_3d [[0, 1], [2], [3]]
        output_shape [%batch, %n_head, %seq_len, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    // Transpose [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim]
    %output_init = tensor.empty(%batch, %seq_len, %n_head, %head_dim) : tensor<?x?x?x?xf16>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%output_4d : tensor<?x?x?x?xf16>) outs(%output_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    util.return %output : tensor<?x?x?x?xf16>
  }

}
