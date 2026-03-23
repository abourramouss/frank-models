// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Decode Attention Block: QKV projection, RoPE, attention with cached K/V, output projection.
// Processes a single token per sequence using cached K/V from previous tokens.
//
// This is the decode variant that processes one new token and uses cached K/V
// from the KV cache. Used during autoregressive generation.
//
// Logical shapes:
//   input:      [batch, n_embd]               - single token hidden state per sequence
//   positions:  [batch]                       - single position per sequence
//   k_cached:   [batch, ctx_len, n_head_kv, head_dim] - gathered past K (with RoPE)
//   v_cached:   [batch, ctx_len, n_head_kv, head_dim] - gathered past V
//   wqkv:       [n_embd, n_embd + 2*n_embd_kv] - fused QKV projection weight
//   wo:         same as prefill
//
// Returns:
//   output:    [batch, n_embd]                - attention output for single token
//   k_new:     [batch, n_head_kv, head_dim]   - new K with RoPE (for scatter to cache)
//   v_new:     [batch, n_head_kv, head_dim]   - new V (for scatter to cache)

module @attention_block_decode_qwen_components {

  // External dependencies resolved by iree-link.
  util.func private @position_components.rope(
      tensor<?x?x?x?xf16>,   // [batch, seq_len, n_head, head_dim]
      tensor<?x?xi64>,        // [batch, seq_len]
      f32,                    // freq_base
      f32                     // freq_scale
  ) -> tensor<?x?x?x?xf16>

  util.func private @attention_components.attention_gqa(
      tensor<?x?x?x?xf16>,   // Q: [batch, seq_len, n_head, head_dim]
      tensor<?x?x?x?xf16>,   // K: [batch, seq_len, n_head_kv, head_dim]
      tensor<?x?x?x?xf16>,   // V: [batch, seq_len, n_head_kv, head_dim]
      f32                     // scale
  ) -> tensor<?x?x?x?xf16>

  util.func public @attention_block_decode_qwen(
      %input: tensor<?x?xf16>,              // [batch, n_embd]
      %positions: tensor<?xi64>,             // [batch] - single position per sequence
      %k_cached: tensor<?x?x?x?xf16>,       // [batch, ctx_len, n_head_kv, head_dim]
      %v_cached: tensor<?x?x?x?xf16>,       // [batch, ctx_len, n_head_kv, head_dim]
      %wq: tensor<?x?xf16>,                  // [n_embd, n_embd_q]
      %wk: tensor<?x?xf16>,                  // [n_embd, n_embd_kv]
      %wv: tensor<?x?xf16>,                  // [n_embd, n_embd_kv]
      %wo: tensor<?x?xf16>,                  // [n_embd_q, n_embd]
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %rope_freq_base: f32,
      %rope_freq_scale: f32,
      %q_norm_weight: tensor<?xf16>,         // [n_embd] - QK norm weight
      %k_norm_weight: tensor<?xf16>,         // [n_embd_kv] - QK norm weight
      %rms_eps: f32                          // epsilon for QK norm
  ) -> (tensor<?x?xf16>,                     // output: [batch, n_embd]
        tensor<?x?x?xf16>,                   // k_new: [batch, n_head_kv, head_dim]
        tensor<?x?x?xf16>) {                 // v_new: [batch, n_head_kv, head_dim]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?xf16>

    // Get context length from cached K/V.
    %ctx_len = tensor.dim %k_cached, %c1 : tensor<?x?x?x?xf16>

    // Derive head_dim from Q weight: wq is [n_embd, n_embd_q], head_dim = n_embd_q / n_head
    %n_embd_q = tensor.dim %wq, %c1 : tensor<?x?xf16>
    %head_dim = arith.divui %n_embd_q, %n_head : index
    %n_embd_kv = arith.muli %n_head_kv, %head_dim : index

    // seq_len=1 for decode (single token)
    %seq_len_1 = arith.constant 1 : index

    // Expand input from [batch, n_embd] to [batch, 1, n_embd].
    %input_3d = tensor.expand_shape %input [[0], [1, 2]]
        output_shape [%batch, %seq_len_1, %n_embd]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    // Expand positions from [batch] to [batch, 1].
    %positions_2d = tensor.expand_shape %positions [[0, 1]]
        output_shape [%batch, %seq_len_1]
        : tensor<?xi64> into tensor<?x?xi64>

    // Separate Q, K, V projections — no extract_slice = no slow_memcpy dispatches
    %cst_zero = arith.constant 0.0 : f16

    %q_proj_init = tensor.empty(%batch, %n_embd_q) : tensor<?x?xf16>
    %q_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%q_proj_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %q_proj = linalg.matmul ins(%input, %wq : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%q_proj_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    %k_proj_init = tensor.empty(%batch, %n_embd_kv) : tensor<?x?xf16>
    %k_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%k_proj_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %k_proj = linalg.matmul ins(%input, %wk : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%k_proj_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    %v_proj_init = tensor.empty(%batch, %n_embd_kv) : tensor<?x?xf16>
    %v_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%v_proj_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %v_proj = linalg.matmul ins(%input, %wv : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%v_proj_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // Reshape Q/K to 3D: [batch, n_embd_q] -> [batch, n_head, head_dim]
    %q_3d = tensor.expand_shape %q_proj [[0], [1, 2]]
        output_shape [%batch, %n_head, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %k_3d = tensor.expand_shape %k_proj [[0], [1, 2]]
        output_shape [%batch, %n_head_kv, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %v_3d = tensor.expand_shape %v_proj [[0], [1, 2]]
        output_shape [%batch, %n_head_kv, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    // QK norm via flow.dispatch.region: forces reduction+elementwise into 1 dispatch.
    // Without this, canonicalization pushes expand_shape between them, breaking fusion.
    %q_batch_heads = arith.muli %batch, %n_head : index
    %q_flat = tensor.collapse_shape %q_3d [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>

    %q_normed_flat = flow.dispatch.region[] -> (tensor<?x?xf16>{%q_batch_heads, %head_dim}) {
      %cst_zero_f32 = arith.constant 0.0 : f32
      %hd_i32 = arith.index_cast %head_dim : index to i32
      %hd_f32 = arith.sitofp %hd_i32 : i32 to f32
      %q_sum_init = tensor.empty(%q_batch_heads) : tensor<?xf32>
      %q_sum_zero = linalg.fill ins(%cst_zero_f32 : f32) outs(%q_sum_init : tensor<?xf32>) -> tensor<?xf32>
      %q_sum_sq = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%q_flat : tensor<?x?xf16>) outs(%q_sum_zero : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %e = arith.extf %in : f16 to f32
        %sq = arith.mulf %e, %e : f32
        %s = arith.addf %out, %sq : f32
        linalg.yield %s : f32
      } -> tensor<?xf32>
      %q_out_init = tensor.empty(%q_batch_heads, %head_dim) : tensor<?x?xf16>
      %q_normed = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%q_flat, %q_sum_sq, %q_norm_weight : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>)
        outs(%q_out_init : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %mean = arith.divf %ss, %hd_f32 : f32
        %eps_add = arith.addf %mean, %rms_eps : f32
        %rms_val = math.sqrt %eps_add : f32
        %x = arith.extf %in : f16 to f32
        %wf = arith.extf %w : f16 to f32
        %n = arith.divf %x, %rms_val : f32
        %sc = arith.mulf %n, %wf : f32
        %r = arith.truncf %sc : f32 to f16
        linalg.yield %r : f16
      } -> tensor<?x?xf16>
      flow.return %q_normed : tensor<?x?xf16>
    }

    %k_batch_heads = arith.muli %batch, %n_head_kv : index
    %k_flat = tensor.collapse_shape %k_3d [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>

    %k_normed_flat = flow.dispatch.region[] -> (tensor<?x?xf16>{%k_batch_heads, %head_dim}) {
      %cst_zero_f32 = arith.constant 0.0 : f32
      %hd_i32 = arith.index_cast %head_dim : index to i32
      %hd_f32 = arith.sitofp %hd_i32 : i32 to f32
      %k_sum_init = tensor.empty(%k_batch_heads) : tensor<?xf32>
      %k_sum_zero = linalg.fill ins(%cst_zero_f32 : f32) outs(%k_sum_init : tensor<?xf32>) -> tensor<?xf32>
      %k_sum_sq = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%k_flat : tensor<?x?xf16>) outs(%k_sum_zero : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %e = arith.extf %in : f16 to f32
        %sq = arith.mulf %e, %e : f32
        %s = arith.addf %out, %sq : f32
        linalg.yield %s : f32
      } -> tensor<?xf32>
      %k_out_init = tensor.empty(%k_batch_heads, %head_dim) : tensor<?x?xf16>
      %k_normed = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%k_flat, %k_sum_sq, %k_norm_weight : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>)
        outs(%k_out_init : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %mean = arith.divf %ss, %hd_f32 : f32
        %eps_add = arith.addf %mean, %rms_eps : f32
        %rms_val = math.sqrt %eps_add : f32
        %x = arith.extf %in : f16 to f32
        %wf = arith.extf %w : f16 to f32
        %n = arith.divf %x, %rms_val : f32
        %sc = arith.mulf %n, %wf : f32
        %r = arith.truncf %sc : f32 to f16
        linalg.yield %r : f16
      } -> tensor<?x?xf16>
      flow.return %k_normed : tensor<?x?xf16>
    }

    // Reshape back: [batch*n_head, head_dim] -> [batch, n_head, head_dim] -> [batch, 1, n_head, head_dim]
    %q_reshaped_3d = tensor.expand_shape %q_normed_flat [[0, 1], [2]]
        output_shape [%batch, %n_head, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %q_reshaped_4d = tensor.expand_shape %q_reshaped_3d [[0, 1], [2], [3]]
        output_shape [%batch, %seq_len_1, %n_head, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    %k_reshaped_3d = tensor.expand_shape %k_normed_flat [[0, 1], [2]]
        output_shape [%batch, %n_head_kv, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %k_reshaped_4d = tensor.expand_shape %k_reshaped_3d [[0, 1], [2], [3]]
        output_shape [%batch, %seq_len_1, %n_head_kv, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    // V is already [batch, n_head_kv, head_dim] from the reshape
    %v_reshaped_3d = tensor.cast %v_3d : tensor<?x?x?xf16> to tensor<?x?x?xf16>

    // Apply RoPE to query and key (new token only).
    %q_rope_4d = util.call @position_components.rope(%q_reshaped_4d, %positions_2d, %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>
    %k_rope_4d = util.call @position_components.rope(%k_reshaped_4d, %positions_2d, %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>

    // Concat new K/V with cached K/V: [batch, ctx_len, ...] + [batch, 1, ...] -> [batch, ctx_len+1, ...]
    // K_full: [batch, ctx_len+1, n_head_kv, head_dim]
    %ctx_len_plus_1 = arith.addi %ctx_len, %c1 : index
    %k_full = tensor.concat dim(1) %k_cached, %k_rope_4d
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>

    // V_full: [batch, ctx_len+1, n_head_kv, head_dim]
    %v_reshaped_4d = tensor.expand_shape %v_reshaped_3d [[0, 1], [2], [3]]
        output_shape [%batch, %seq_len_1, %n_head_kv, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %v_full = tensor.concat dim(1) %v_cached, %v_reshaped_4d
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>

    // Compute attention: Q[batch, 1, ...] against K/V[batch, ctx_len+1, ...]
    // scale = 1/sqrt(head_dim)
    %head_dim_i32 = arith.index_cast %head_dim : index to i32
    %head_dim_f32 = arith.sitofp %head_dim_i32 : i32 to f32
    %scale = math.rsqrt %head_dim_f32 : f32

    %attn_out_4d = util.call @attention_components.attention_gqa(%q_rope_4d, %k_full, %v_full, %scale)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f32) -> tensor<?x?x?x?xf16>

    // Reshape attention output: [batch, 1, n_head, head_dim] -> [batch, n_embd]
    %attn_out_3d = tensor.collapse_shape %attn_out_4d [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %attn_flat = tensor.collapse_shape %attn_out_3d [[0], [1, 2]] : tensor<?x?x?xf16> into tensor<?x?xf16>

    // Output projection: [batch, n_embd] @ [n_embd, n_embd] -> [batch, n_embd]
    %output_proj_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf16>
    %output_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%output_proj_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %output_proj = linalg.matmul ins(%attn_flat, %wo : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%output_proj_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // OLMoE: no output bias. Use output_proj directly.

    // Extract new K/V for cache update: [batch, 1, n_head_kv, head_dim] -> [batch, n_head_kv, head_dim]
    %k_new = tensor.collapse_shape %k_rope_4d [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %v_new = tensor.collapse_shape %v_reshaped_4d [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>

    util.return %output_proj, %k_new, %v_new : tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>
  }

}
