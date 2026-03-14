// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MoE LLM Architecture with KV Cache integration.
// Separate @prefill and @decode entry points for efficient autoregressive generation.
//
// This is an architecture-generic module that gets linked with model-specific
// hparams.mlir and params.mlir to produce a complete model.
//
// Unified paged KV cache: single block pool shared across all layers.
// Layer-aware metadata: block_tables[n_layers, batch, max_blocks], context_lens[n_layers, batch].
//
// Entry points:
//   @allocate_kv_cache(n_blocks, block_size) -> cache
//   @prefill(tokens, positions, cache, ...) -> (logits, cache_out)
//   @decode(tokens, positions, cache, ...) -> (logits, cache_out)
//
// Linked modules required:
//   @hparams - Scalar hyperparameters (vocab_size, block_count, etc.)
//   @model_params - Parameter accessors (token_embd_weight, attn_qkv_weight, etc.)
//   Components: embedding, rms_norm, kvcache, transformer_layer_moe_prefill/decode

module @llm_inference {

  // ===== Hyperparameter imports (hparams module provides these) =====
  util.func private @hparams.vocab_size() -> i64
  util.func private @hparams.block_count() -> i64
  util.func private @hparams.embedding_length() -> i64
  util.func private @hparams.attention_head_count() -> i64
  util.func private @hparams.attention_head_count_kv() -> i64
  util.func private @hparams.feed_forward_length() -> i64
  util.func private @hparams.expert_count() -> i64
  util.func private @hparams.expert_used_count() -> i64
  util.func private @hparams.rope_freq_base() -> f32
  util.func private @hparams.layer_norm_rms_epsilon() -> f32
  // Architecture variant flags
  util.func private @hparams.use_attention_bias() -> i1
  util.func private @hparams.normalize_expert_weights() -> i1
  util.func private @hparams.use_qk_norm() -> i1

  // ===== Model-level parameter imports (model_params module provides these) =====
  util.func private @model_params.token_embd_weight() -> tensor<?x?xf16>
  util.func private @model_params.output_norm_weight() -> tensor<?xf16>
  util.func private @model_params.output_weight() -> tensor<?x?xf16>

  // ===== Component imports (resolved by iree-link) =====
  util.func private @embedding_components.embedding_lookup(
      tensor<?x?xf16>,   // [vocab_size, n_embd]
      tensor<?x?xi64>    // [batch, seq_len]
  ) -> tensor<?x?x?xf16> // [batch, seq_len, n_embd]

  util.func private @embedding_components.embedding_lookup_1d(
      tensor<?x?xf16>,   // [vocab_size, n_embd]
      tensor<?xi64>      // [batch]
  ) -> tensor<?x?xf16>   // [batch, n_embd]

  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf16>,   // [n_tokens, hidden_dim]
      tensor<?xf16>,      // [hidden_dim]
      f32                 // epsilon
  ) -> tensor<?x?xf16>

  // KV cache
  util.func private @kvcache_components.allocate(
      index,   // n_blocks
      index,   // block_size
      index,   // n_head_kv
      index    // head_dim
  ) -> !util.list<?>

  // Prefill transformer layer (scatters K/V to cache internally)
  util.func private @transformer_layer_moe_prefill_components.transformer_layer_moe_prefill(
      tensor<?x?x?xf16>,   // input: [batch, seq_len, n_embd]
      tensor<?x?xi64>,     // positions: [batch, seq_len]
      !util.list<?>,       // cache
      tensor<?x?x?xi32>,   // block_tables: [n_layers, batch, max_blocks]
      tensor<?xi32>,       // start_positions: [batch]
      index,               // block_size
      i32,                 // layer_idx
      index,               // n_head
      index,               // n_head_kv
      index,               // n_embd
      index,               // n_ff
      index,               // n_expert
      index,               // n_expert_used
      f32,                 // rms_eps
      f32,                 // rope_freq_base
      f32,                 // rope_freq_scale
      i1,                  // use_bias
      i1                   // use_qk_norm
  ) -> (tensor<?x?x?xf16>,     // output: [batch, seq_len, n_embd]
        !util.list<?>)         // cache_out with K/V written

  // Decode transformer layer
  util.func private @transformer_layer_moe_decode_components.transformer_layer_moe_decode(
      tensor<?x?xf16>,         // input: [batch, n_embd]
      tensor<?xi64>,           // positions: [batch]
      !util.list<?>,           // cache
      tensor<?x?x?xi32>,       // block_tables: [n_layers, batch, max_blocks]
      tensor<?x?xi32>,         // context_lens: [n_layers, batch]
      index,                   // max_context_len
      i32,                     // layer_idx
      index,                   // n_head
      index,                   // n_head_kv
      index,                   // n_embd
      index,                   // n_ff
      index,                   // n_expert
      index,                   // n_expert_used
      f32,                     // rms_eps
      f32,                     // rope_freq_base
      f32                      // rope_freq_scale
  ) -> (tensor<?x?xf16>,       // output: [batch, n_embd]
        !util.list<?>)         // cache_out

  // ===== KV Cache Allocation =====
  // Derives n_head_kv and head_dim from hparams.

  util.func public @allocate_kv_cache(
      %n_blocks: index,
      %block_size: index
  ) -> !util.list<?> {
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64

    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %head_dim = arith.divui %n_embd, %n_head : index

    %cache = util.call @kvcache_components.allocate(
        %n_blocks, %block_size, %n_head_kv, %head_dim)
        : (index, index, index, index) -> !util.list<?>
    util.return %cache : !util.list<?>
  }

  // ===== Prefill Entry Point =====
  // Process initial tokens, scatter K/V to cache, return logits and updated cache.

  util.func public @prefill(
      %tokens: tensor<?x?xi64>,           // [batch, seq_len]
      %positions: tensor<?x?xi64>,        // [batch, seq_len]
      %cache: !util.list<?>,              // Unified KV cache
      %block_tables: tensor<?x?x?xi32>,   // [n_layers, batch, max_blocks]
      %start_positions: tensor<?xi32>,    // [batch]
      %block_size: index
  ) -> (tensor<?x?x?xf16>,                // logits: [batch, seq_len, vocab_size]
        !util.list<?>) {                  // cache_out
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Load hyperparameters.
    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_ff_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %n_expert_i64 = util.call @hparams.expert_count() : () -> i64
    %n_expert_used_i64 = util.call @hparams.expert_used_count() : () -> i64
    %rope_freq_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32
    %use_bias = util.call @hparams.use_attention_bias() : () -> i1
    %normalize_weights = util.call @hparams.normalize_expert_weights() : () -> i1
    %use_qk_norm = util.call @hparams.use_qk_norm() : () -> i1

    // Convert to index.
    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %n_layer = arith.index_cast %n_layer_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_ff = arith.index_cast %n_ff_i64 : i64 to index
    %n_expert = arith.index_cast %n_expert_i64 : i64 to index
    %n_expert_used = arith.index_cast %n_expert_used_i64 : i64 to index

    %batch = tensor.dim %tokens, %c0 : tensor<?x?xi64>
    %seq_len = tensor.dim %tokens, %c1 : tensor<?x?xi64>

    // Token embedding lookup.
    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %embeddings = util.call @embedding_components.embedding_lookup(%tok_embd_weight, %tokens)
        : (tensor<?x?xf16>, tensor<?x?xi64>) -> tensor<?x?x?xf16>

    // Transformer layers loop (prefill variant with cache threading).
    %rope_freq_scale = arith.constant 1.0 : f32
    %final_hidden, %final_cache = scf.for %layer_idx = %c0 to %n_layer step %c1
        iter_args(%hidden = %embeddings, %cache_iter = %cache) -> (tensor<?x?x?xf16>, !util.list<?>) {
      %layer_idx_i32 = arith.index_cast %layer_idx : index to i32

      %layer_out, %cache_updated = util.call @transformer_layer_moe_prefill_components.transformer_layer_moe_prefill(
          %hidden, %positions, %cache_iter,
          %block_tables, %start_positions, %block_size,
          %layer_idx_i32,
          %n_head, %n_head_kv, %n_embd, %n_ff,
          %n_expert, %n_expert_used,
          %rms_eps, %rope_freq_base, %rope_freq_scale,
          %use_bias, %use_qk_norm)
          : (tensor<?x?x?xf16>, tensor<?x?xi64>, !util.list<?>,
             tensor<?x?x?xi32>, tensor<?xi32>, index,
             i32,
             index, index, index, index, index, index,
             f32, f32, f32, i1, i1)
          -> (tensor<?x?x?xf16>, !util.list<?>)

      scf.yield %layer_out, %cache_updated : tensor<?x?x?xf16>, !util.list<?>
    }

    // Output normalization (operates on 2D: [batch*seq_len, n_embd]).
    %final_hidden_2d = tensor.collapse_shape %final_hidden [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %normalized_2d = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden_2d, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // LM head projection: 2D matmul [batch*seq_len, n_embd] @ [n_embd, vocab] -> [batch*seq_len, vocab].
    // Using 2D matmul avoids shape inference issues with expand_shape on dynamic tensors.
    %output_weight = util.call @model_params.output_weight() : () -> tensor<?x?xf16>
    %n_tokens = arith.muli %batch, %seq_len : index
    %logits_2d_empty = tensor.empty(%n_tokens, %n_vocab) : tensor<?x?xf16>
    %zero = arith.constant 0.0 : f16
    %logits_2d_init = linalg.fill ins(%zero : f16) outs(%logits_2d_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
    %logits_2d = linalg.matmul ins(%normalized_2d, %output_weight : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%logits_2d_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    // Reshape logits to [batch, seq_len, vocab].
    %logits = tensor.expand_shape %logits_2d [[0, 1], [2]]
        output_shape [%batch, %seq_len, %n_vocab]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    util.return %logits, %final_cache : tensor<?x?x?xf16>, !util.list<?>
  }

  // ===== Decode Entry Point =====
  // Process single token per sequence using cached K/V.

  util.func public @decode(
      %tokens: tensor<?xi64>,               // [batch]
      %positions: tensor<?xi64>,            // [batch]
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>,     // [n_layers, batch, max_blocks]
      %context_lens: tensor<?x?xi32>,       // [n_layers, batch]
      %max_context_len: index
  ) -> (tensor<?x?xf16>,                    // logits: [batch, vocab_size]
        !util.list<?>) attributes {inlining_policy = #util.inline.never} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Load hyperparameters.
    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_ff_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %n_expert_i64 = util.call @hparams.expert_count() : () -> i64
    %n_expert_used_i64 = util.call @hparams.expert_used_count() : () -> i64
    %rope_freq_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32
    // normalize_weights removed from decode path (OLMoE: always false)

    // Convert to index.
    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %n_layer = arith.index_cast %n_layer_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_ff = arith.index_cast %n_ff_i64 : i64 to index
    %n_expert = arith.index_cast %n_expert_i64 : i64 to index
    %n_expert_used = arith.index_cast %n_expert_used_i64 : i64 to index

    %batch = tensor.dim %tokens, %c0 : tensor<?xi64>

    // Token embedding lookup (1D variant for decode).
    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %embeddings = util.call @embedding_components.embedding_lookup_1d(%tok_embd_weight, %tokens)
        : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

    // Transformer layers loop (decode variant with cache threading).
    %rope_freq_scale = arith.constant 1.0 : f32
    %final_hidden, %final_cache = scf.for %layer_idx = %c0 to %n_layer step %c1
        iter_args(%hidden = %embeddings, %cache_iter = %cache) -> (tensor<?x?xf16>, !util.list<?>) {
      %layer_idx_i32 = arith.index_cast %layer_idx : index to i32

      %layer_out, %cache_updated = util.call @transformer_layer_moe_decode_components.transformer_layer_moe_decode(
          %hidden, %positions, %cache_iter,
          %block_tables, %context_lens, %max_context_len,
          %layer_idx_i32,
          %n_head, %n_head_kv, %n_embd, %n_ff,
          %n_expert, %n_expert_used,
          %rms_eps, %rope_freq_base, %rope_freq_scale)
          : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>,
             tensor<?x?x?xi32>, tensor<?x?xi32>, index,
             i32,
             index, index, index, index, index, index,
             f32, f32, f32)
          -> (tensor<?x?xf16>, !util.list<?>)

      scf.yield %layer_out, %cache_updated : tensor<?x?xf16>, !util.list<?>
    }

    // Output normalization.
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %normalized = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // LM head projection: [batch, n_embd] @ [n_embd, vocab] -> [batch, vocab].
    %output_weight = util.call @model_params.output_weight() : () -> tensor<?x?xf16>
    %logits_empty = tensor.empty(%batch, %n_vocab) : tensor<?x?xf16>
    %zero = arith.constant 0.0 : f16
    %logits_init = linalg.fill ins(%zero : f16) outs(%logits_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
    %logits = linalg.matmul ins(%normalized, %output_weight : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    util.return %logits, %final_cache : tensor<?x?xf16>, !util.list<?>
  }

  // ===== Generate Step: one decode + argmax =====
  // Kept as separate function to prevent inlining into scf.while,
  // which breaks GPU distribution of the fused mul_mat_id generic.

  util.func public @generate_step(
      %token: tensor<?xi64>,              // [1] current token
      %position: tensor<?xi64>,           // [1] current position
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>,
      %context_len: index
  ) -> (i64, !util.list<?>) attributes {inlining_policy = #util.inline.never} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n_layers_i64 = util.call @hparams.block_count() : () -> i64
    %n_layers = arith.index_cast %n_layers_i64 : i64 to index
    %context_len_i32 = arith.index_cast %context_len : index to i32

    // Build context_lens tensor
    %cl_init = tensor.empty(%n_layers, %c1) : tensor<?x?xi32>
    %ctx_lens = linalg.fill ins(%context_len_i32 : i32) outs(%cl_init : tensor<?x?xi32>) -> tensor<?x?xi32>

    // Decode
    %logits, %cache_out = util.call @decode(
        %token, %position, %cache,
        %block_tables, %ctx_lens, %context_len)
        : (tensor<?xi64>, tensor<?xi64>, !util.list<?>,
           tensor<?x?x?xi32>, tensor<?x?xi32>, index)
        -> (tensor<?x?xf16>, !util.list<?>)

    // Argmax: cast to f32, find max via topk(k=1)
    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %logits_1d = tensor.collapse_shape %logits [[0, 1]]
        : tensor<?x?xf16> into tensor<?xf16>
    %logits_f32_init = tensor.empty(%n_vocab) : tensor<?xf32>
    %logits_f32 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%logits_1d : tensor<?xf16>) outs(%logits_f32_init : tensor<?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %v = arith.extf %in : f16 to f32
      linalg.yield %v : f32
    } -> tensor<?xf32>

    %tk_val = tensor.empty(%c1) : tensor<?xf32>
    %tk_idx = tensor.empty(%c1) : tensor<?xi32>
    %tv, %ti = iree_linalg_ext.topk
        dimension(0)
        ins(%logits_f32 : tensor<?xf32>)
        outs(%tk_val, %tk_idx : tensor<?xf32>, tensor<?xi32>) {
      ^bb0(%lhs: f32, %rhs: f32):
        %cmp = arith.cmpf ogt, %lhs, %rhs : f32
        iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xf32>, tensor<?xi32>

    %next_token_i32 = tensor.extract %ti[%c0] : tensor<?xi32>
    %next_token = arith.extsi %next_token_i32 : i32 to i64

    util.return %next_token, %cache_out : i64, !util.list<?>
  }

  // ===== Self-contained Generate Entry Point =====
  // Does the entire autoregressive loop: prefill → decode → argmax → repeat.
  // No Python needed — takes input token IDs, returns generated token IDs.
  //
  // Usage via iree-run-module:
  //   iree-run-module --function=generate --input=5xi64=[tok0,tok1,...] --input=20
  //
  // Args:
  //   input_tokens: [seq_len] i64 — prompt token IDs
  //   max_new_tokens: index — maximum tokens to generate
  // Returns:
  //   output_tokens: [max_new_tokens] i64 — generated token IDs (padded with 0)
  //   num_generated: i64 — actual number of tokens generated


  util.func public @generate(
      %input_tokens: tensor<?xi64>,
      %max_new_tokens: index
  ) -> (tensor<?xi64>, i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %eos_token = arith.constant 50279 : i64
    %zero_i64 = arith.constant 0 : i64

    %seq_len = tensor.dim %input_tokens, %c0 : tensor<?xi64>

    // Compute cache size
    %total_len = arith.addi %seq_len, %max_new_tokens : index
    %total_plus = arith.addi %total_len, %c16 : index
    %total_minus = arith.subi %total_plus, %c1 : index
    %max_blocks = arith.divui %total_minus, %c16 : index
    %n_layers_i64 = util.call @hparams.block_count() : () -> i64
    %n_layers = arith.index_cast %n_layers_i64 : i64 to index
    %n_blocks = arith.muli %n_layers, %max_blocks : index

    // Allocate KV cache
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %head_dim = arith.divui %n_embd, %n_head : index
    %cache_init = util.call @kvcache_components.allocate(%n_blocks, %c16, %n_head_kv, %head_dim)
        : (index, index, index, index) -> !util.list<?>

    // Build block tables [n_layers, 1, max_blocks]
    %bt_init = tensor.empty(%n_layers, %c1, %max_blocks) : tensor<?x?x?xi32>
    %block_tables = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%bt_init : tensor<?x?x?xi32>) {
    ^bb0(%out: i32):
      %layer = linalg.index 0 : index
      %blk = linalg.index 2 : index
      %off = arith.muli %layer, %max_blocks : index
      %idx = arith.addi %off, %blk : index
      %idx_i32 = arith.index_cast %idx : index to i32
      linalg.yield %idx_i32 : i32
    } -> tensor<?x?x?xi32>

    // Start positions
    %sp_init = tensor.empty(%c1) : tensor<?xi32>
    %zero_i32 = arith.constant 0 : i32
    %start_positions = linalg.fill ins(%zero_i32 : i32) outs(%sp_init : tensor<?xi32>) -> tensor<?xi32>

    // Prefill first token
    %first_tok_val = tensor.extract %input_tokens[%c0] : tensor<?xi64>
    %first_tok_t = tensor.from_elements %first_tok_val : tensor<1x1xi64>
    %first_tok = tensor.cast %first_tok_t : tensor<1x1xi64> to tensor<?x?xi64>
    %pos_zero_val = arith.constant 0 : i64
    %pos_zero_t = tensor.from_elements %pos_zero_val : tensor<1x1xi64>
    %pos_zero = tensor.cast %pos_zero_t : tensor<1x1xi64> to tensor<?x?xi64>

    %prefill_logits, %cache_pf = util.call @prefill(
        %first_tok, %pos_zero, %cache_init,
        %block_tables, %start_positions, %c16)
        : (tensor<?x?xi64>, tensor<?x?xi64>, !util.list<?>,
           tensor<?x?x?xi32>, tensor<?xi32>, index)
        -> (tensor<?x?x?xf16>, !util.list<?>)

    // Process remaining prompt tokens through generate_step
    %prompt_result:2 = scf.for %pi = %c1 to %seq_len step %c1
        iter_args(%cache_p = %cache_pf, %dummy = %zero_i64)
        -> (!util.list<?>, i64) {
      %tok_val = tensor.extract %input_tokens[%pi] : tensor<?xi64>
      %tok_t = tensor.from_elements %tok_val : tensor<1xi64>
      %tok_dyn = tensor.cast %tok_t : tensor<1xi64> to tensor<?xi64>
      %pi_i64 = arith.index_cast %pi : index to i64
      %pos_t = tensor.from_elements %pi_i64 : tensor<1xi64>
      %pos_dyn = tensor.cast %pos_t : tensor<1xi64> to tensor<?xi64>

      %next_tok, %cache_new = util.call @generate_step(
          %tok_dyn, %pos_dyn, %cache_p, %block_tables, %pi)
          : (tensor<?xi64>, tensor<?xi64>, !util.list<?>,
             tensor<?x?x?xi32>, index)
          -> (i64, !util.list<?>)

      scf.yield %cache_new, %next_tok : !util.list<?>, i64
    }

    // Get first generated token from prefill logits (argmax)
    // Use generate_step with the last prompt token's output
    // Actually we already have prompt_result#1 as the last next_token
    // But for the first step, we need the argmax of prefill if seq_len==1
    // Let's use generate_step for uniformity:
    // The first generated token comes from the last prompt processing step

    // Output buffer
    %output_init = tensor.empty(%max_new_tokens) : tensor<?xi64>
    %output_zeros = linalg.fill ins(%zero_i64 : i64) outs(%output_init : tensor<?xi64>) -> tensor<?xi64>

    // Autoregressive decode loop using generate_step
    %init_step = arith.constant 0 : index
    %gen:4 = scf.while (
        %step = %init_step,
        %next_tok = %prompt_result#1,
        %cache_w = %prompt_result#0,
        %output_w = %output_zeros
    ) : (index, i64, !util.list<?>, tensor<?xi64>) -> (index, i64, !util.list<?>, tensor<?xi64>) {
      %not_done = arith.cmpi ult, %step, %max_new_tokens : index
      %not_eos = arith.cmpi ne, %next_tok, %eos_token : i64
      %continue = arith.andi %not_done, %not_eos : i1
      scf.condition(%continue) %step, %next_tok, %cache_w, %output_w
          : index, i64, !util.list<?>, tensor<?xi64>
    } do {
    ^bb0(%step_: index, %next_tok_: i64, %cache_: !util.list<?>, %output_: tensor<?xi64>):
      // Store token
      %output_updated = tensor.insert %next_tok_ into %output_[%step_] : tensor<?xi64>

      // Compute position
      %cur_pos = arith.addi %seq_len, %step_ : index
      %cur_pos_i64 = arith.index_cast %cur_pos : index to i64
      %pos_t_ = tensor.from_elements %cur_pos_i64 : tensor<1xi64>
      %pos_dyn_ = tensor.cast %pos_t_ : tensor<1xi64> to tensor<?xi64>
      %tok_t_ = tensor.from_elements %next_tok_ : tensor<1xi64>
      %tok_dyn_ = tensor.cast %tok_t_ : tensor<1xi64> to tensor<?xi64>

      // Call generate_step (separate function = won't be inlined)
      %new_tok, %cache_new = util.call @generate_step(
          %tok_dyn_, %pos_dyn_, %cache_, %block_tables, %cur_pos)
          : (tensor<?xi64>, tensor<?xi64>, !util.list<?>,
             tensor<?x?x?xi32>, index)
          -> (i64, !util.list<?>)

      %next_step = arith.addi %step_, %c1 : index
      scf.yield %next_step, %new_tok, %cache_new, %output_updated
          : index, i64, !util.list<?>, tensor<?xi64>
    }

    %num_gen = arith.index_cast %gen#0 : index to i64
    util.return %gen#3, %num_gen : tensor<?xi64>, i64
  }

}
