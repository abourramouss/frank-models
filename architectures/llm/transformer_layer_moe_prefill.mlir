// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MoE Transformer Layer (Prefill): pre-norm attention + MoE FFN with residual connections.
// Processes full input sequence and writes K/V to cache via scatter_prefill.
//
// Architecture (Mixtral-style):
//   input → attn_norm → attention_block_prefill → scatter_prefill(cache) → +residual
//         → ffn_norm  → moe_ffn_block          → +residual → output
//
// Integrates with unified paged KV cache. K/V from attention is scattered to cache
// before returning.
//
// Logical shapes:
//   input:           [batch, seq_len, n_embd]           - Input hidden states
//   positions:       [batch, seq_len]                   - Position indices for RoPE
//   k_cache:         tensor<?x?x?x?xf16>               - K cache [n_blocks, block_size, n_head_kv, head_dim]
//   v_cache:         tensor<?x?x?x?xf16>               - V cache [n_blocks, block_size, n_head_kv, head_dim]
//   block_tables:    [n_layers, batch, max_blocks]      - Block indirection
//   start_positions: [batch]                            - Where to start writing (usually 0)
//   block_size:      index                              - Tokens per block
//
// Returns:
//   output:          [batch, seq_len, n_embd]           - Output hidden states
//   k_cache_out:     tensor<?x?x?x?xf16>               - Updated K cache
//   v_cache_out:     tensor<?x?x?x?xf16>               - Updated V cache
//
// Reference: transformer_layer_moe.mlir, attention_block_prefill.mlir, kvcache.mlir

module @transformer_layer_moe_prefill_components {

  // ===== Parameter accessor imports (model_params module provides these) =====
  // Each takes a layer index and returns the parameter tensor for that layer.

  // Normalization weights
  util.func private @model_params.attn_norm_weight(i32) -> tensor<?xf16>
  util.func private @model_params.ffn_norm_weight(i32) -> tensor<?xf16>

  // Attention projection weights
  util.func private @model_params.attn_q_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.attn_k_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.attn_v_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.attn_output_weight(i32) -> tensor<?x?xf16>

  // Attention biases (may be dummy zeros if use_bias=false)
  util.func private @model_params.attn_q_bias(i32) -> tensor<?xf16>
  util.func private @model_params.attn_k_bias(i32) -> tensor<?xf16>
  util.func private @model_params.attn_v_bias(i32) -> tensor<?xf16>
  util.func private @model_params.attn_output_bias(i32) -> tensor<?xf16>

  // QK norm weights (may be dummy if use_qk_norm=false)
  util.func private @model_params.attn_q_norm_weight(i32) -> tensor<?xf16>
  util.func private @model_params.attn_k_norm_weight(i32) -> tensor<?xf16>

  // MoE weights
  util.func private @model_params.ffn_gate_inp_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.ffn_up_exps_weight(i32) -> tensor<?x?x?xf16>
  util.func private @model_params.ffn_gate_exps_weight(i32) -> tensor<?x?x?xf16>
  util.func private @model_params.ffn_down_exps_weight(i32) -> tensor<?x?x?xf16>

  // ===== Component imports (resolved by iree-link) =====

  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf16>,    // [n_tokens, hidden_dim]
      tensor<?xf16>,       // [hidden_dim]
      f32                  // epsilon
  ) -> tensor<?x?xf16>

  // Prefill attention: returns output + K/V for cache storage
  util.func private @attention_block_prefill_components.attention_block_prefill(
      tensor<?x?x?xf16>,   // [batch, seq_len, n_embd]
      tensor<?x?xi64>,     // [batch, seq_len]
      tensor<?x?xf16>,     // wq
      tensor<?x?xf16>,     // wk
      tensor<?x?xf16>,     // wv
      tensor<?x?xf16>,     // wo
      tensor<?xf16>,       // bq
      tensor<?xf16>,       // bk
      tensor<?xf16>,       // bv
      tensor<?xf16>,       // bo
      i1,                  // use_bias
      index,               // n_head
      index,               // n_head_kv
      index,               // n_embd
      f32,                 // rope_freq_base
      f32,                 // rope_freq_scale
      i1,                  // use_qk_norm
      tensor<?xf16>,       // q_norm_weight [head_dim]
      tensor<?xf16>,       // k_norm_weight [head_dim]
      f32                  // rms_eps (for QK norm)
  ) -> (tensor<?x?x?xf16>,     // output: [batch, seq_len, n_embd]
        tensor<?x?x?x?xf16>,   // k_out: [batch, seq_len, n_head_kv, head_dim]
        tensor<?x?x?x?xf16>)   // v_out: [batch, seq_len, n_head_kv, head_dim]

  util.func private @moe_ffn_components.moe_ffn_block(
      tensor<?x?xf16>,       // [n_tokens, n_embd]
      tensor<?x?xf16>,       // gate_inp_weight
      tensor<?x?x?xf16>,     // up_exps_weight
      tensor<?x?x?xf16>,     // gate_exps_weight
      tensor<?x?x?xf16>,     // down_exps_weight
      index,                 // n_expert
      index,                 // n_expert_used
      index,                 // n_embd
      index,                 // n_ff
      i1                     // normalize_weights
  ) -> tensor<?x?xf16>

  // KV cache scatter for prefill
  util.func private @kvcache_components.scatter_prefill(
      tensor<?x?x?x?xf16>,      // k_cache
      tensor<?x?x?x?xf16>,      // v_cache
      index,                     // layer
      tensor<?x?x?x?xf16>,      // new_k: [batch, seq_len, n_head_kv, head_dim]
      tensor<?x?x?x?xf16>,      // new_v: [batch, seq_len, n_head_kv, head_dim]
      tensor<?x?x?xi32>,        // block_tables: [n_layers, batch, max_blocks]
      tensor<?xi32>,             // start_positions: [batch]
      index                     // block_size
  ) -> (tensor<?x?x?x?xf16>,   // k_cache_out
        tensor<?x?x?x?xf16>)   // v_cache_out

  // ===== Layer function =====

  util.func public @transformer_layer_moe_prefill(
      %input: tensor<?x?x?xf16>,        // [batch, seq_len, n_embd]
      %positions: tensor<?x?xi64>,       // [batch, seq_len]
      %k_cache: tensor<?x?x?x?xf16>,    // K cache
      %v_cache: tensor<?x?x?x?xf16>,    // V cache
      %block_tables: tensor<?x?x?xi32>,  // [n_layers, batch, max_blocks]
      %start_positions: tensor<?xi32>,   // [batch] - where to start writing (usually 0)
      %block_size: index,
      %layer_idx: i32,
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %n_ff: index,
      %n_expert: index,
      %n_expert_used: index,
      %rms_eps: f32,
      %rope_freq_base: f32,
      %rope_freq_scale: f32,
      %use_bias: i1,
      %normalize_weights: i1,
      %use_qk_norm: i1
  ) -> (tensor<?x?x?xf16>,               // output: [batch, seq_len, n_embd]
        tensor<?x?x?x?xf16>,             // k_cache_out
        tensor<?x?x?x?xf16>) {           // v_cache_out
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?x?xf16>
    %seq_len = tensor.dim %input, %c1 : tensor<?x?x?xf16>

    // ---- Load all parameters for this layer ----

    %attn_norm_w = util.call @model_params.attn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %ffn_norm_w = util.call @model_params.ffn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>

    %wq = util.call @model_params.attn_q_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %wk = util.call @model_params.attn_k_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %wv = util.call @model_params.attn_v_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %wo = util.call @model_params.attn_output_weight(%layer_idx) : (i32) -> tensor<?x?xf16>

    %bq = util.call @model_params.attn_q_bias(%layer_idx) : (i32) -> tensor<?xf16>
    %bk = util.call @model_params.attn_k_bias(%layer_idx) : (i32) -> tensor<?xf16>
    %bv = util.call @model_params.attn_v_bias(%layer_idx) : (i32) -> tensor<?xf16>
    %bo = util.call @model_params.attn_output_bias(%layer_idx) : (i32) -> tensor<?xf16>

    %q_norm_w = util.call @model_params.attn_q_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %k_norm_w = util.call @model_params.attn_k_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>

    %gate_inp_w = util.call @model_params.ffn_gate_inp_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %up_exps_w = util.call @model_params.ffn_up_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf16>
    %gate_exps_w = util.call @model_params.ffn_gate_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf16>
    %down_exps_w = util.call @model_params.ffn_down_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf16>

    // ---- Attention sub-layer ----

    // Flatten [batch, seq_len, n_embd] → [n_tokens, n_embd] for rms_norm (2D).
    %input_2d = tensor.collapse_shape %input [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>

    %attn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %input_2d, %attn_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // Unflatten back to [batch, seq_len, n_embd] for attention_block_prefill (3D).
    %attn_normed_3d = tensor.expand_shape %attn_normed [[0, 1], [2]]
        output_shape [%batch, %seq_len, %n_embd]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    // Call prefill attention: returns output + K/V for cache storage.
    %attn_out, %k_out, %v_out = util.call @attention_block_prefill_components.attention_block_prefill(
        %attn_normed_3d, %positions,
        %wq, %wk, %wv, %wo,
        %bq, %bk, %bv, %bo,
        %use_bias, %n_head, %n_head_kv, %n_embd,
        %rope_freq_base, %rope_freq_scale,
        %use_qk_norm, %q_norm_w, %k_norm_w, %rms_eps)
        : (tensor<?x?x?xf16>, tensor<?x?xi64>,
           tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>,
           tensor<?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>,
           i1, index, index, index, f32, f32,
           i1, tensor<?xf16>, tensor<?xf16>, f32)
        -> (tensor<?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)

    // Residual connection: input + attn_out.
    %residual1_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf16>
    %residual1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %attn_out : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
      outs(%residual1_init : tensor<?x?x?xf16>) {
    ^bb0(%a: f16, %b: f16, %out: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    } -> tensor<?x?x?xf16>

    // ---- MoE FFN sub-layer ----

    // Flatten [batch, seq_len, n_embd] → [n_tokens, n_embd] for rms_norm + moe_ffn_block.
    %residual1_2d = tensor.collapse_shape %residual1 [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>

    %ffn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %residual1_2d, %ffn_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // MoE FFN block operates on [n_tokens, n_embd].
    %moe_out = util.call @moe_ffn_components.moe_ffn_block(
        %ffn_normed, %gate_inp_w,
        %up_exps_w, %gate_exps_w, %down_exps_w,
        %n_expert, %n_expert_used, %n_embd, %n_ff,
        %normalize_weights)
        : (tensor<?x?xf16>, tensor<?x?xf16>,
           tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>,
           index, index, index, index, i1) -> tensor<?x?xf16>

    // Unflatten MoE output back to [batch, seq_len, n_embd].
    %moe_out_3d = tensor.expand_shape %moe_out [[0, 1], [2]]
        output_shape [%batch, %seq_len, %n_embd]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    // Residual connection: residual1 + moe_out.
    %output_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf16>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%residual1, %moe_out_3d : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
      outs(%output_init : tensor<?x?x?xf16>) {
    ^bb0(%a: f16, %b: f16, %out: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    } -> tensor<?x?x?xf16>

    // Scatter K/V to cache for this layer.
    %layer = arith.index_cast %layer_idx : i32 to index
    %k_cache_updated, %v_cache_updated = util.call @kvcache_components.scatter_prefill(
        %k_cache, %v_cache, %layer, %k_out, %v_out,
        %block_tables, %start_positions, %block_size)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, index, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
           tensor<?x?x?xi32>, tensor<?xi32>, index)
        -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)

    util.return %output, %k_cache_updated, %v_cache_updated : tensor<?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
  }

}
