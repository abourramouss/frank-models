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

module @attention_block_decode_components {

  // External dependencies resolved by iree-link.
  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf16>,    // [n_tokens, hidden_dim]
      tensor<?xf16>,       // [hidden_dim]
      f32                  // epsilon
  ) -> tensor<?x?xf16>

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

  util.func public @attention_block_decode(
      %input: tensor<?x?xf16>,              // [batch, n_embd]
      %positions: tensor<?xi64>,             // [batch] - single position per sequence
      %k_cached: tensor<?x?x?x?xf16>,       // [batch, ctx_len, n_head_kv, head_dim]
      %v_cached: tensor<?x?x?x?xf16>,       // [batch, ctx_len, n_head_kv, head_dim]
      %wqkv: tensor<?x?xf16>,                // [n_embd, n_embd + 2*n_embd_kv] (fused QKV)
      %wo: tensor<?x?xf16>,                  // [n_embd, n_embd]
      %bq: tensor<?xf16>,                    // [n_embd] - may be dummy if not used
      %bk: tensor<?xf16>,                    // [n_embd_kv]
      %bv: tensor<?xf16>,                    // [n_embd_kv]
      %bo: tensor<?xf16>,                    // [n_embd]
      %use_bias: i1,                         // flag to enable/disable biases
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %rope_freq_base: f32,
      %rope_freq_scale: f32,
      %use_qk_norm: i1,                      // flag to enable/disable QK norm
      %q_norm_weight: tensor<?xf16>,         // [n_embd] - may be dummy if not used
      %k_norm_weight: tensor<?xf16>,         // [n_embd_kv] - may be dummy if not used
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

    // Get dimensions from fused QKV weight.
    // wqkv shape: [n_embd, qkv_out_dim] where qkv_out_dim = n_embd + 2*n_embd_kv
    %qkv_out_dim = tensor.dim %wqkv, %c1 : tensor<?x?xf16>
    // n_embd_kv = (qkv_out_dim - n_embd) / 2
    %kv_total = arith.subi %qkv_out_dim, %n_embd : index
    %n_embd_kv = arith.divsi %kv_total, %c2 : index

    // Compute head dimensions.
    %head_dim = arith.divsi %n_embd, %n_head : index
    %head_dim_kv = arith.divsi %n_embd_kv, %n_head_kv : index

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

    // Fused QKV projection: [batch, n_embd] @ [n_embd, qkv_out_dim] -> [batch, qkv_out_dim]
    %cst_zero = arith.constant 0.0 : f16

    %qkv_proj_init = tensor.empty(%batch, %qkv_out_dim) : tensor<?x?xf16>
    %qkv_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%qkv_proj_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %qkv_proj = linalg.matmul ins(%input, %wqkv : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%qkv_proj_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // Split fused QKV output: [batch, qkv_out_dim] -> Q[batch, n_embd], K[batch, n_embd_kv], V[batch, n_embd_kv]
    %q_proj = tensor.extract_slice %qkv_proj[0, 0] [%batch, %n_embd] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %k_proj = tensor.extract_slice %qkv_proj[0, %n_embd] [%batch, %n_embd_kv] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %v_offset = arith.addi %n_embd, %n_embd_kv : index
    %v_proj = tensor.extract_slice %qkv_proj[0, %v_offset] [%batch, %n_embd_kv] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>

    // Conditionally add biases if enabled.
    // bias_add: out[b, i] = proj[b, i] + bias[i]
    %q_bias_out_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf16>
    // OLMoE: no bias (use_bias=false), always QK norm (use_qk_norm=true).
    // Removing scf.if conditionals to eliminate fusion barriers.

    // QK norm directly on projections (no bias add).
    %q_normed = util.call @rms_norm_components.rms_norm_linalg(
        %q_proj, %q_norm_weight, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    %k_normed = util.call @rms_norm_components.rms_norm_linalg(
        %k_proj, %k_norm_weight, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // Reshape for multi-head: [batch, n_embd] -> [batch, 1, n_head, head_dim]
    // Two expand_shapes: [batch, n_embd] -> [batch, n_head, head_dim] -> [batch, 1, n_head, head_dim]
    %q_reshaped_3d = tensor.expand_shape %q_normed [[0], [1, 2]]
        output_shape [%batch, %n_head, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %q_reshaped_4d = tensor.expand_shape %q_reshaped_3d [[0, 1], [2], [3]]
        output_shape [%batch, %seq_len_1, %n_head, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    %k_reshaped_3d = tensor.expand_shape %k_normed [[0], [1, 2]]
        output_shape [%batch, %n_head_kv, %head_dim_kv]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %k_reshaped_4d = tensor.expand_shape %k_reshaped_3d [[0, 1], [2], [3]]
        output_shape [%batch, %seq_len_1, %n_head_kv, %head_dim_kv]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    %v_reshaped_3d = tensor.expand_shape %v_proj [[0], [1, 2]]
        output_shape [%batch, %n_head_kv, %head_dim_kv]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

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
        output_shape [%batch, %seq_len_1, %n_head_kv, %head_dim_kv]
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
