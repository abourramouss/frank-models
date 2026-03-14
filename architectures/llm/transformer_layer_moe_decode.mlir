// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MoE Transformer Layer (Decode): pre-norm attention with cached K/V + MoE FFN.
// Processes a single token per sequence using cached K/V from previous tokens.
//
// Architecture (Mixtral-style):
//   input → attn_norm → gather(cache) → attention_block_decode → scatter_decode(cache)
//         → +residual → ffn_norm → moe_ffn_block → +residual → output
//
// Integrates with unified paged KV cache. All layers share the same physical block pool.
// Layer-aware metadata (block_tables, context_lens) has layer dimension sliced internally
// by gather. Scatter uses scalar indices (logical_block, pos_in_block) passed from the
// entry point to avoid device->host staging transfers.
//
// Logical shapes:
//   input:              [batch, n_embd]               - Single token hidden state per sequence
//   positions:          [batch]                       - Single position per sequence
//   cache:              !util.list<?>                 - Unified KV cache (K_blocks, V_blocks)
//   block_tables:       [n_layers, batch, max_blocks] - Block indirection (gather only)
//   context_lens:       [n_layers, batch]             - Current context length per layer/seq
//   max_context_len:    index                         - Max context for gather output shape
//   logical_block:      index                         - cur_pos // block_size
//   pos_in_block:       index                         - cur_pos % block_size
//   max_blocks_per_seq: index                         - for physical block offset computation
//
// Returns:
//   output:          [batch, n_embd]               - Output hidden state
//   cache_out:       !util.list<?>                 - Cache with new K/V written
//
// Reference: transformer_layer_moe.mlir, attention_block_decode.mlir, kvcache.mlir

module @transformer_layer_moe_decode_components {

  // ===== Parameter accessor imports (model_params module provides these) =====

  // Normalization weights
  util.func private @model_params.attn_norm_weight(i32) -> tensor<?xf16>
  util.func private @model_params.ffn_norm_weight(i32) -> tensor<?xf16>

  // Attention projection weights
  util.func private @model_params.attn_qkv_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.attn_output_weight(i32) -> tensor<?x?xf16>

  // QK norm weights
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

  // KV cache: gather K/V for a specific layer
  util.func private @kvcache_components.gather(
      !util.list<?>,           // cache
      index,                   // layer
      tensor<?x?x?xi32>,       // block_tables [n_layers, batch, max_blocks]
      tensor<?x?xi32>,         // context_lens [n_layers, batch]
      index                    // max_context_len
  ) -> (tensor<?x?x?x?xf16>,   // k_gathered: [batch, max_ctx, n_head_kv, head_dim]
        tensor<?x?x?x?xf16>)   // v_gathered: [batch, max_ctx, n_head_kv, head_dim]

  // KV cache: scatter new K/V for a specific layer (decode: single token)
  // Scalar scatter: takes precomputed physical block and position-in-block
  // instead of device tensors, eliminating staging transfers.
  util.func private @kvcache_components.scatter_decode(
      !util.list<?>,           // cache
      index,                   // layer
      tensor<?x?x?xf16>,       // new_k: [batch, n_head_kv, head_dim]
      tensor<?x?x?xf16>,       // new_v: [batch, n_head_kv, head_dim]
      index,                   // target_block (physical block index)
      index                    // pos_in_block
  ) -> !util.list<?>

  // Decode attention: process single token with cached K/V (fused QKV)
  util.func private @attention_block_decode_components.attention_block_decode(
      tensor<?x?xf16>,         // [batch, n_embd]
      tensor<?xi64>,           // [batch]
      tensor<?x?x?x?xf16>,     // k_cached: [batch, ctx_len, n_head_kv, head_dim]
      tensor<?x?x?x?xf16>,     // v_cached: [batch, ctx_len, n_head_kv, head_dim]
      tensor<?x?xf16>,         // wqkv (fused QKV weight)
      tensor<?x?xf16>,         // wo
      index,                   // n_head
      index,                   // n_head_kv
      index,                   // n_embd
      f32,                     // rope_freq_base
      f32,                     // rope_freq_scale
      tensor<?xf16>,           // q_norm_weight [head_dim]
      tensor<?xf16>,           // k_norm_weight [head_dim]
      f32                      // rms_eps (for QK norm)
  ) -> (tensor<?x?xf16>,       // output: [batch, n_embd]
        tensor<?x?x?xf16>,     // k_new: [batch, n_head_kv, head_dim]
        tensor<?x?x?xf16>)     // v_new: [batch, n_head_kv, head_dim]

  util.func private @moe_ffn_components.moe_ffn_block(
      tensor<?x?xf16>,       // [n_tokens, n_embd]
      tensor<?x?xf16>,       // gate_inp_weight
      tensor<?x?x?xf16>,     // up_exps_weight
      tensor<?x?x?xf16>,     // gate_exps_weight
      tensor<?x?x?xf16>,     // down_exps_weight
      index,                 // n_expert
      index,                 // n_expert_used
      index,                 // n_embd
      index                  // n_ff
  ) -> tensor<?x?xf16>

  // ===== Layer function =====

  util.func public @transformer_layer_moe_decode(
      %input: tensor<?x?xf16>,             // [batch, n_embd]
      %positions: tensor<?xi64>,            // [batch]
      %cache: !util.list<?>,                // Unified KV cache
      %block_tables: tensor<?x?x?xi32>,     // [n_layers, batch, max_blocks]
      %context_lens: tensor<?x?xi32>,       // [n_layers, batch]
      %max_context_len: index,
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
      %logical_block: index,               // logical block index (cur_pos // block_size)
      %pos_in_block: index,                 // position within block (cur_pos % block_size)
      %max_blocks_per_seq: index            // for computing physical block offset
  ) -> (tensor<?x?xf16>,                    // output: [batch, n_embd]
        !util.list<?>) {                    // cache_out with new K/V written
    %c0 = arith.constant 0 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?xf16>

    // Convert layer_idx to index for kvcache calls.
    %layer = arith.index_cast %layer_idx : i32 to index

    // ---- Load all parameters for this layer ----

    %attn_norm_w = util.call @model_params.attn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %ffn_norm_w = util.call @model_params.ffn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>

    %wqkv = util.call @model_params.attn_qkv_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %wo = util.call @model_params.attn_output_weight(%layer_idx) : (i32) -> tensor<?x?xf16>

    %q_norm_w = util.call @model_params.attn_q_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %k_norm_w = util.call @model_params.attn_k_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>

    %gate_inp_w = util.call @model_params.ffn_gate_inp_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %up_exps_w = util.call @model_params.ffn_up_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf16>
    %gate_exps_w = util.call @model_params.ffn_gate_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf16>
    %down_exps_w = util.call @model_params.ffn_down_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf16>

    // ---- Attention sub-layer ----

    // RMS norm on input: [batch, n_embd].
    %attn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %input, %attn_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // Gather cached K/V for this layer: [batch, max_ctx, n_head_kv, head_dim].
    %k_cached, %v_cached = util.call @kvcache_components.gather(
        %cache, %layer, %block_tables, %context_lens, %max_context_len)
        : (!util.list<?>, index, tensor<?x?x?xi32>, tensor<?x?xi32>, index)
        -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)

    // Decode attention with cached K/V (fused QKV).
    %attn_out, %k_new, %v_new = util.call @attention_block_decode_components.attention_block_decode(
        %attn_normed, %positions,
        %k_cached, %v_cached,
        %wqkv, %wo,
        %n_head, %n_head_kv, %n_embd,
        %rope_freq_base, %rope_freq_scale,
        %q_norm_w, %k_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xi64>,
           tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
           tensor<?x?xf16>, tensor<?x?xf16>,
           index, index, index, f32, f32,
           tensor<?xf16>, tensor<?xf16>, f32)
        -> (tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)

    // Compute physical block: layer * max_blocks_per_seq + logical_block
    %layer_offset = arith.muli %layer, %max_blocks_per_seq : index
    %physical_block = arith.addi %layer_offset, %logical_block : index

    // Scatter new K/V to cache (scalar indices, no staging transfers).
    %cache_updated = util.call @kvcache_components.scatter_decode(
        %cache, %layer, %k_new, %v_new, %physical_block, %pos_in_block)
        : (!util.list<?>, index, tensor<?x?x?xf16>, tensor<?x?x?xf16>,
           index, index) -> !util.list<?>

    // Residual connection: input + attn_out.
    %residual1_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf16>
    %residual1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %attn_out : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%residual1_init : tensor<?x?xf16>) {
    ^bb0(%a: f16, %b: f16, %out: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    // ---- MoE FFN sub-layer ----

    // RMS norm on residual: [batch, n_embd].
    %ffn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %residual1, %ffn_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // MoE FFN block operates on [batch, n_embd] (batch = n_tokens for decode).
    %moe_out = util.call @moe_ffn_components.moe_ffn_block(
        %ffn_normed, %gate_inp_w,
        %up_exps_w, %gate_exps_w, %down_exps_w,
        %n_expert, %n_expert_used, %n_embd, %n_ff)
        : (tensor<?x?xf16>, tensor<?x?xf16>,
           tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>,
           index, index, index, index) -> tensor<?x?xf16>

    // Residual connection: residual1 + moe_out.
    %output_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf16>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%residual1, %moe_out : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%output_init : tensor<?x?xf16>) {
    ^bb0(%a: f16, %b: f16, %out: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    util.return %output, %cache_updated : tensor<?x?xf16>, !util.list<?>
  }

}
