// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prefill Attention Block: QKV projection, RoPE, GQA attention, output projection.
// Returns K and V (with RoPE applied to K) for cache storage.
//
// This is the prefill variant that processes a full input sequence and returns
// the K/V tensors needed for KV cache population. Used during prompt processing.
//
// Logical shapes:
//   input:     [batch, seq_len, n_embd]      - input hidden states
//   positions: [batch, seq_len]              - position indices for RoPE
//   wq:        [n_embd, n_embd]              - query projection weights
//   wk:        [n_embd, n_kv_embd]           - key projection (n_kv_embd = n_head_kv * head_dim)
//   wv:        [n_embd, n_kv_embd]           - value projection
//   wo:        [n_embd, n_embd]              - output projection
//   bq/bk/bv/bo: [n_embd] or [n_kv_embd]     - optional biases
//
// Returns:
//   output:    [batch, seq_len, n_embd]      - attention output
//   k_out:     [batch, seq_len, n_head_kv, head_dim] - K with RoPE applied (for cache)
//   v_out:     [batch, seq_len, n_head_kv, head_dim] - V (for cache)

module @attention_block_prefill_components {

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

  util.func public @attention_block_prefill(
      %input: tensor<?x?x?xf16>,        // [batch, seq_len, n_embd]
      %positions: tensor<?x?xi64>,       // [batch, seq_len]
      %wq: tensor<?x?xf16>,              // [n_embd, n_embd]
      %wk: tensor<?x?xf16>,              // [n_embd, n_embd_kv]
      %wv: tensor<?x?xf16>,              // [n_embd, n_embd_kv]
      %wo: tensor<?x?xf16>,              // [n_embd, n_embd]
      %bq: tensor<?xf16>,                // [n_embd] - may be dummy if not used
      %bk: tensor<?xf16>,                // [n_embd_kv]
      %bv: tensor<?xf16>,                // [n_embd_kv]
      %bo: tensor<?xf16>,                // [n_embd]
      %use_bias: i1,                     // flag to enable/disable biases
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %rope_freq_base: f32,
      %rope_freq_scale: f32,
      %use_qk_norm: i1,                  // flag to enable/disable QK norm
      %q_norm_weight: tensor<?xf16>,     // [n_embd] - may be dummy if not used
      %k_norm_weight: tensor<?xf16>,     // [n_embd_kv] - may be dummy if not used
      %rms_eps: f32                      // epsilon for QK norm
  ) -> (tensor<?x?x?xf16>,               // output: [batch, seq_len, n_embd]
        tensor<?x?x?x?xf16>,             // k_out: [batch, seq_len, n_head_kv, head_dim]
        tensor<?x?x?x?xf16>) {           // v_out: [batch, seq_len, n_head_kv, head_dim]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?x?xf16>
    %seq_len = tensor.dim %input, %c1 : tensor<?x?x?xf16>

    // Get dimensions from weights.
    %n_embd_kv = tensor.dim %wk, %c1 : tensor<?x?xf16>
    %wq_k = tensor.dim %wq, %c0 : tensor<?x?xf16>
    %wk_k = tensor.dim %wk, %c0 : tensor<?x?xf16>
    %wv_k = tensor.dim %wv, %c0 : tensor<?x?xf16>
    %wo_k = tensor.dim %wo, %c0 : tensor<?x?xf16>

    // Broadcast weights from [K, N] to [batch, K, N] for batch_matmul.
    // Q weights: [n_embd, n_embd] -> [batch, n_embd, n_embd]
    %wq_3d_init = tensor.empty(%batch, %wq_k, %n_embd) : tensor<?x?x?xf16>
    %wq_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,     // input: [K, N]
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>  // output: [batch, K, N]
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%wq : tensor<?x?xf16>) outs(%wq_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?xf16>

    // K weights: [n_embd, n_embd_kv] -> [batch, n_embd, n_embd_kv]
    %wk_3d_init = tensor.empty(%batch, %wk_k, %n_embd_kv) : tensor<?x?x?xf16>
    %wk_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%wk : tensor<?x?xf16>) outs(%wk_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?xf16>

    // V weights: [n_embd, n_embd_kv] -> [batch, n_embd, n_embd_kv]
    %wv_3d_init = tensor.empty(%batch, %wv_k, %n_embd_kv) : tensor<?x?x?xf16>
    %wv_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%wv : tensor<?x?xf16>) outs(%wv_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?xf16>

    // QKV projections: [batch, seq, n_embd] @ [batch, n_embd, n_out] -> [batch, seq, n_out]
    %cst_zero = arith.constant 0.0 : f16
    %q_proj_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf16>
    %q_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%q_proj_init : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %q_proj = linalg.batch_matmul ins(%input, %wq_3d : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
        outs(%q_proj_zero : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

    %k_proj_init = tensor.empty(%batch, %seq_len, %n_embd_kv) : tensor<?x?x?xf16>
    %k_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%k_proj_init : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %k_proj = linalg.batch_matmul ins(%input, %wk_3d : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
        outs(%k_proj_zero : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

    %v_proj_init = tensor.empty(%batch, %seq_len, %n_embd_kv) : tensor<?x?x?xf16>
    %v_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%v_proj_init : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %v_proj = linalg.batch_matmul ins(%input, %wv_3d : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
        outs(%v_proj_zero : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

    // Conditionally add biases if enabled.
    %q_final = scf.if %use_bias -> (tensor<?x?x?xf16>) {
      %q_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,  // proj
          affine_map<(d0, d1, d2) -> (d2)>,          // bias
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>   // out
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%q_proj, %bq : tensor<?x?x?xf16>, tensor<?xf16>) outs(%q_proj_init : tensor<?x?x?xf16>) {
      ^bb0(%proj: f16, %bias: f16, %out: f16):
        %sum = arith.addf %proj, %bias : f16
        linalg.yield %sum : f16
      } -> tensor<?x?x?xf16>
      scf.yield %q_biased : tensor<?x?x?xf16>
    } else {
      scf.yield %q_proj : tensor<?x?x?xf16>
    }

    %k_final = scf.if %use_bias -> (tensor<?x?x?xf16>) {
      %k_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%k_proj, %bk : tensor<?x?x?xf16>, tensor<?xf16>) outs(%k_proj_init : tensor<?x?x?xf16>) {
      ^bb0(%proj: f16, %bias: f16, %out: f16):
        %sum = arith.addf %proj, %bias : f16
        linalg.yield %sum : f16
      } -> tensor<?x?x?xf16>
      scf.yield %k_biased : tensor<?x?x?xf16>
    } else {
      scf.yield %k_proj : tensor<?x?x?xf16>
    }

    %v_final = scf.if %use_bias -> (tensor<?x?x?xf16>) {
      %v_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%v_proj, %bv : tensor<?x?x?xf16>, tensor<?xf16>) outs(%v_proj_init : tensor<?x?x?xf16>) {
      ^bb0(%proj: f16, %bias: f16, %out: f16):
        %sum = arith.addf %proj, %bias : f16
        linalg.yield %sum : f16
      } -> tensor<?x?x?xf16>
      scf.yield %v_biased : tensor<?x?x?xf16>
    } else {
      scf.yield %v_proj : tensor<?x?x?xf16>
    }

    // Conditionally apply QK norm (RMS norm on flat Q/K projections before reshape).
    // Applied to [batch, seq_len, n_embd] (Q) or [batch, seq_len, n_embd_kv] (K).
    %q_normed = scf.if %use_qk_norm -> (tensor<?x?x?xf16>) {
      %q_flat = tensor.collapse_shape %q_final [[0, 1], [2]]
          : tensor<?x?x?xf16> into tensor<?x?xf16>
      %q_normed_flat = util.call @rms_norm_components.rms_norm_linalg(
          %q_flat, %q_norm_weight, %rms_eps)
          : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
      %q_normed_3d = tensor.expand_shape %q_normed_flat [[0, 1], [2]]
          output_shape [%batch, %seq_len, %n_embd]
          : tensor<?x?xf16> into tensor<?x?x?xf16>
      scf.yield %q_normed_3d : tensor<?x?x?xf16>
    } else {
      scf.yield %q_final : tensor<?x?x?xf16>
    }

    %k_normed = scf.if %use_qk_norm -> (tensor<?x?x?xf16>) {
      %k_flat = tensor.collapse_shape %k_final [[0, 1], [2]]
          : tensor<?x?x?xf16> into tensor<?x?xf16>
      %k_normed_flat = util.call @rms_norm_components.rms_norm_linalg(
          %k_flat, %k_norm_weight, %rms_eps)
          : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
      %k_normed_3d = tensor.expand_shape %k_normed_flat [[0, 1], [2]]
          output_shape [%batch, %seq_len, %n_embd_kv]
          : tensor<?x?xf16> into tensor<?x?x?xf16>
      scf.yield %k_normed_3d : tensor<?x?x?xf16>
    } else {
      scf.yield %k_final : tensor<?x?x?xf16>
    }

    // Reshape for multi-head: [batch, seq_len, n_embd] -> [batch, seq_len, n_head, head_dim]
    // Use linalg.generic + linalg.index instead of tensor.expand_shape to avoid an IREE
    // GlobalOpt bug: when n_head and head_dim are constant-folded to static values, IREE
    // produces expand_shape with static_output_shape = array<i64> (empty) → verifier crash.
    %head_dim = arith.divsi %n_embd, %n_head : index
    %head_dim_kv = arith.divsi %n_embd_kv, %n_head_kv : index

    %q_reshaped_init = tensor.empty(%batch, %seq_len, %n_head, %head_dim) : tensor<?x?x?x?xf16>
    %q_reshaped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%q_reshaped_init : tensor<?x?x?x?xf16>) {
    ^bb0(%out: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %i3 = linalg.index 3 : index
      %flat = arith.muli %i2, %head_dim : index
      %flat_idx = arith.addi %flat, %i3 : index
      %val = tensor.extract %q_normed[%i0, %i1, %flat_idx] : tensor<?x?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?x?x?xf16>

    %k_reshaped_init = tensor.empty(%batch, %seq_len, %n_head_kv, %head_dim_kv) : tensor<?x?x?x?xf16>
    %k_reshaped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%k_reshaped_init : tensor<?x?x?x?xf16>) {
    ^bb0(%out: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %i3 = linalg.index 3 : index
      %flat = arith.muli %i2, %head_dim_kv : index
      %flat_idx = arith.addi %flat, %i3 : index
      %val = tensor.extract %k_normed[%i0, %i1, %flat_idx] : tensor<?x?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?x?x?xf16>

    %v_reshaped_init = tensor.empty(%batch, %seq_len, %n_head_kv, %head_dim_kv) : tensor<?x?x?x?xf16>
    %v_reshaped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%v_reshaped_init : tensor<?x?x?x?xf16>) {
    ^bb0(%out: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %i3 = linalg.index 3 : index
      %flat = arith.muli %i2, %head_dim_kv : index
      %flat_idx = arith.addi %flat, %i3 : index
      %val = tensor.extract %v_final[%i0, %i1, %flat_idx] : tensor<?x?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?x?x?xf16>

    // Apply RoPE to query and key.
    // K with RoPE will be stored in cache.
    %q_rope = util.call @position_components.rope(%q_reshaped, %positions, %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>
    %k_rope = util.call @position_components.rope(%k_reshaped, %positions, %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>

    // Compute attention with GQA.
    // scale = 1/sqrt(head_dim)
    %head_dim_i32 = arith.index_cast %head_dim : index to i32
    %head_dim_f32 = arith.sitofp %head_dim_i32 : i32 to f32
    %scale = math.rsqrt %head_dim_f32 : f32

    %attn_out = util.call @attention_components.attention_gqa(%q_rope, %k_rope, %v_reshaped, %scale)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f32) -> tensor<?x?x?x?xf16>

    // Reshape back: [batch, seq_len, n_head, head_dim] -> [batch, seq_len, n_embd]
    %attn_flat = tensor.collapse_shape %attn_out [[0], [1], [2, 3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>

    // Output projection.
    %wo_3d_init = tensor.empty(%batch, %wo_k, %n_embd) : tensor<?x?x?xf16>
    %wo_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%wo : tensor<?x?xf16>) outs(%wo_3d_init : tensor<?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?xf16>

    %output_proj_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf16>
    %output_proj_zero = linalg.fill ins(%cst_zero : f16) outs(%output_proj_init : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %output_proj = linalg.batch_matmul ins(%attn_flat, %wo_3d : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
        outs(%output_proj_zero : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

    // Conditionally add output bias.
    %output = scf.if %use_bias -> (tensor<?x?x?xf16>) {
      %output_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%output_proj, %bo : tensor<?x?x?xf16>, tensor<?xf16>) outs(%output_proj_init : tensor<?x?x?xf16>) {
      ^bb0(%proj: f16, %bias: f16, %out: f16):
        %sum = arith.addf %proj, %bias : f16
        linalg.yield %sum : f16
      } -> tensor<?x?x?xf16>
      scf.yield %output_biased : tensor<?x?x?xf16>
    } else {
      scf.yield %output_proj : tensor<?x?x?xf16>
    }

    // Return output, K (with RoPE), V (without RoPE - applied during attention)
    // K_rope is stored in cache; V is stored as-is
    util.return %output, %k_rope, %v_reshaped : tensor<?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
  }

}
