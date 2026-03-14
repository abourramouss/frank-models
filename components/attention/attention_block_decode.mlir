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
//   wq/wk/wv/wo: same as prefill
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
      %wq: tensor<?x?xf16>,                  // [n_embd, n_embd]
      %wk: tensor<?x?xf16>,                  // [n_embd, n_embd_kv]
      %wv: tensor<?x?xf16>,                  // [n_embd, n_embd_kv]
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

    // Get dimensions from weights.
    %n_embd_kv = tensor.dim %wk, %c1 : tensor<?x?xf16>

    // Compute head dimensions.
    %head_dim = arith.divsi %n_embd, %n_head : index
    %head_dim_kv = arith.divsi %n_embd_kv, %n_head_kv : index

    // seq_len=1 for decode (single token)
    %seq_len_1 = arith.constant 1 : index

    // Expand input from [batch, n_embd] to [batch, 1, n_embd] for matmul.
    // Use linalg.generic broadcast (d1 absent from input map) to avoid tensor.expand_shape
    // splitting the static n_embd dim — see IREE GlobalOpt bug note in attention_block_prefill.
    %input_3d_init = tensor.empty(%batch, %seq_len_1, %n_embd) : tensor<?x?x?xf16>
    %input_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<?x?xf16>) outs(%input_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?xf16>

    // Expand positions from [batch] to [batch, 1] for RoPE.
    // Use linalg.generic broadcast to avoid any expand_shape static-dim issues.
    %positions_2d_init = tensor.empty(%batch, %seq_len_1) : tensor<?x?xi64>
    %positions_2d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%positions : tensor<?xi64>) outs(%positions_2d_init : tensor<?x?xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<?x?xi64>

    // QKV projections: [batch, 1, n_embd] @ [n_embd, n_out] -> [batch, 1, n_out]
    %cst_zero = arith.constant 0.0 : f16

    // Q projection.
    %q_proj_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf16>
    %q_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%q_proj_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %q_proj = linalg.matmul ins(%input, %wq : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%q_proj_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // K projection.
    %k_proj_init = tensor.empty(%batch, %n_embd_kv) : tensor<?x?xf16>
    %k_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%k_proj_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %k_proj = linalg.matmul ins(%input, %wk : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%k_proj_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // V projection.
    %v_proj_init = tensor.empty(%batch, %n_embd_kv) : tensor<?x?xf16>
    %v_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%v_proj_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %v_proj = linalg.matmul ins(%input, %wv : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%v_proj_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // Conditionally add biases if enabled.
    // bias_add: out[b, i] = proj[b, i] + bias[i]
    %q_final = scf.if %use_bias -> (tensor<?x?xf16>) {
      %q_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,  // proj: [batch, n_embd]
          affine_map<(d0, d1) -> (d1)>,       // bias: [n_embd]
          affine_map<(d0, d1) -> (d0, d1)>   // out: [batch, n_embd]
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%q_proj, %bq : tensor<?x?xf16>, tensor<?xf16>) outs(%q_proj_init : tensor<?x?xf16>) {
      ^bb0(%proj: f16, %bias: f16, %out: f16):
        %sum = arith.addf %proj, %bias : f16
        linalg.yield %sum : f16
      } -> tensor<?x?xf16>
      scf.yield %q_biased : tensor<?x?xf16>
    } else {
      scf.yield %q_proj : tensor<?x?xf16>
    }

    %k_final = scf.if %use_bias -> (tensor<?x?xf16>) {
      %k_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%k_proj, %bk : tensor<?x?xf16>, tensor<?xf16>) outs(%k_proj_init : tensor<?x?xf16>) {
      ^bb0(%proj: f16, %bias: f16, %out: f16):
        %sum = arith.addf %proj, %bias : f16
        linalg.yield %sum : f16
      } -> tensor<?x?xf16>
      scf.yield %k_biased : tensor<?x?xf16>
    } else {
      scf.yield %k_proj : tensor<?x?xf16>
    }

    %v_final = scf.if %use_bias -> (tensor<?x?xf16>) {
      %v_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%v_proj, %bv : tensor<?x?xf16>, tensor<?xf16>) outs(%v_proj_init : tensor<?x?xf16>) {
      ^bb0(%proj: f16, %bias: f16, %out: f16):
        %sum = arith.addf %proj, %bias : f16
        linalg.yield %sum : f16
      } -> tensor<?x?xf16>
      scf.yield %v_biased : tensor<?x?xf16>
    } else {
      scf.yield %v_proj : tensor<?x?xf16>
    }

    // Conditionally apply QK norm (RMS norm on flat Q/K projections before reshape).
    // Applied to [batch, n_embd] (Q) or [batch, n_embd_kv] (K).
    %q_normed = scf.if %use_qk_norm -> (tensor<?x?xf16>) {
      %q_normed_2d = util.call @rms_norm_components.rms_norm_linalg(
          %q_final, %q_norm_weight, %rms_eps)
          : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
      scf.yield %q_normed_2d : tensor<?x?xf16>
    } else {
      scf.yield %q_final : tensor<?x?xf16>
    }

    %k_normed = scf.if %use_qk_norm -> (tensor<?x?xf16>) {
      %k_normed_2d = util.call @rms_norm_components.rms_norm_linalg(
          %k_final, %k_norm_weight, %rms_eps)
          : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
      scf.yield %k_normed_2d : tensor<?x?xf16>
    } else {
      scf.yield %k_final : tensor<?x?xf16>
    }

    // Reshape for multi-head: [batch, n_embd] -> [batch, n_head, head_dim]
    // Use linalg.generic + linalg.index to avoid IREE expand_shape bug when n_head/head_dim
    // are constant-folded to static values by GlobalOpt.
    %q_reshaped_3d_init = tensor.empty(%batch, %n_head, %head_dim) : tensor<?x?x?xf16>
    %q_reshaped_3d = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%q_reshaped_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%out: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %flat = arith.muli %i1, %head_dim : index
      %flat_idx = arith.addi %flat, %i2 : index
      %val = tensor.extract %q_normed[%i0, %flat_idx] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?x?xf16>

    %k_reshaped_3d_init = tensor.empty(%batch, %n_head_kv, %head_dim_kv) : tensor<?x?x?xf16>
    %k_reshaped_3d = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%k_reshaped_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%out: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %flat = arith.muli %i1, %head_dim_kv : index
      %flat_idx = arith.addi %flat, %i2 : index
      %val = tensor.extract %k_normed[%i0, %flat_idx] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?x?xf16>

    %v_reshaped_3d_init = tensor.empty(%batch, %n_head_kv, %head_dim_kv) : tensor<?x?x?xf16>
    %v_reshaped_3d = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%v_reshaped_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%out: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %flat = arith.muli %i1, %head_dim_kv : index
      %flat_idx = arith.addi %flat, %i2 : index
      %val = tensor.extract %v_final[%i0, %flat_idx] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?x?xf16>

    // Add seq_len=1 dimension for RoPE: [batch, n_head, head_dim] -> [batch, 1, n_head, head_dim]
    // Use linalg.generic broadcast (d1 absent from input map) to avoid static-dim split bug.
    %q_reshaped_4d_init = tensor.empty(%batch, %seq_len_1, %n_head, %head_dim) : tensor<?x?x?x?xf16>
    %q_reshaped_4d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%q_reshaped_3d : tensor<?x?x?xf16>) outs(%q_reshaped_4d_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %k_reshaped_4d_init = tensor.empty(%batch, %seq_len_1, %n_head_kv, %head_dim_kv) : tensor<?x?x?x?xf16>
    %k_reshaped_4d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%k_reshaped_3d : tensor<?x?x?xf16>) outs(%k_reshaped_4d_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

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
    // Note: v_reshaped_3d is [batch, n_head_kv, head_dim], need to expand to [batch, 1, n_head_kv, head_dim]
    // Use linalg.generic broadcast to avoid the IREE expand_shape static-dim-split bug.
    %v_reshaped_4d_init = tensor.empty(%batch, %seq_len_1, %n_head_kv, %head_dim_kv) : tensor<?x?x?x?xf16>
    %v_reshaped_4d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%v_reshaped_3d : tensor<?x?x?xf16>) outs(%v_reshaped_4d_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
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

    // Conditionally add output bias.
    %output = scf.if %use_bias -> (tensor<?x?xf16>) {
      %output_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%output_proj, %bo : tensor<?x?xf16>, tensor<?xf16>) outs(%output_proj_init : tensor<?x?xf16>) {
      ^bb0(%proj: f16, %bias: f16, %out: f16):
        %sum = arith.addf %proj, %bias : f16
        linalg.yield %sum : f16
      } -> tensor<?x?xf16>
      scf.yield %output_biased : tensor<?x?xf16>
    } else {
      scf.yield %output_proj : tensor<?x?xf16>
    }

    // Extract new K/V for cache update: [batch, 1, n_head_kv, head_dim] -> [batch, n_head_kv, head_dim]
    %k_new = tensor.collapse_shape %k_rope_4d [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %v_new = tensor.collapse_shape %v_reshaped_4d [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>

    util.return %output, %k_new, %v_new : tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>
  }

}
