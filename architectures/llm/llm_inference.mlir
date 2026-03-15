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

module @llm_inference attributes {
  // Jetson AGX Orin: GPU + CPU share unified memory — zero-copy transfers.
  stream.topology = #hal.device.topology<links = [
    (@device_gpu -> @device_cpu = {transparent_access = true, unified_memory = true}),
    (@device_cpu -> @device_gpu = {transparent_access = true, unified_memory = true})
  ]>
} {

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
        !util.list<?>) {                    // cache_out
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

  // ===== Decode Body Entry Point =====
  // Same as decode but returns hidden state BEFORE output norm + LM head.
  // Post-processing (RMSNorm + matmul + argmax) runs on CPU for:
  //   1. Eliminating 2-3 GPU dispatches (~12-18ms HAL overhead saved)
  //   2. Overlapping CPU post-processing with next GPU decode_body call

  util.func public @decode_body(
      %tokens: tensor<?xi64>,               // [batch]
      %positions: tensor<?xi64>,            // [batch]
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>,     // [n_layers, batch, max_blocks]
      %context_lens: tensor<?x?xi32>,       // [n_layers, batch]
      %max_context_len: index
  ) -> (tensor<?x?xf16>,                    // hidden: [batch, n_embd]
        !util.list<?>) {                    // cache_out
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Load hyperparameters.
    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_ff_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %n_expert_i64 = util.call @hparams.expert_count() : () -> i64
    %n_expert_used_i64 = util.call @hparams.expert_used_count() : () -> i64
    %rope_freq_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    // Convert to index.
    %n_layer = arith.index_cast %n_layer_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_ff = arith.index_cast %n_ff_i64 : i64 to index
    %n_expert = arith.index_cast %n_expert_i64 : i64 to index
    %n_expert_used = arith.index_cast %n_expert_used_i64 : i64 to index

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

    // Return hidden state directly — output norm + LM head done on CPU.
    util.return %final_hidden, %final_cache : tensor<?x?xf16>, !util.list<?>
  }

  // ===== Heterogeneous Decode Entry Point =====
  // Full decode with IREE device affinities:
  //   - Transformer layers run on @device_gpu
  //   - Output norm + LM head run on @device_cpu
  // On Jetson unified memory, flow.tensor.transfer is zero-copy.
  //
  // Compile with:
  //   --iree-execution-model=async-external
  //   --iree-hal-target-device=device_gpu=cuda[0]
  //   --iree-hal-target-device=device_cpu=local[0]
  //   --iree-stream-partitioning-favor=max-concurrency

  util.func public @decode_hetero(
      %tokens: tensor<?xi64>,               // [batch]
      %positions: tensor<?xi64>,            // [batch]
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>,     // [n_layers, batch, max_blocks]
      %context_lens: tensor<?x?xi32>,       // [n_layers, batch]
      %max_context_len: index
  ) -> (tensor<?x?xf16>,                    // logits: [batch, vocab_size]
        !util.list<?>) {                    // cache_out
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

    // ===== GPU: Transformer layers =====
    // Token embedding lookup on GPU.
    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %embeddings = util.call @embedding_components.embedding_lookup_1d(%tok_embd_weight, %tokens)
        : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

    // Transformer layers loop on GPU.
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

    // ===== Transfer GPU → CPU (zero-copy on Jetson unified memory) =====
    %hidden_cpu = flow.tensor.transfer %final_hidden : tensor<?x?xf16>{%batch, %n_embd}
        to #hal.device.promise<@device_cpu>

    // ===== CPU: Output normalization (INLINED — no shared function call) =====
    // This avoids affinity contamination of the shared @rms_norm_linalg function.
    %output_norm_w_raw = flow.tensor.constant #flow.parameter.named<"model"::"output_norm.weight"> : tensor<2048xf16>
    %output_norm_w = tensor.cast %output_norm_w_raw : tensor<2048xf16> to tensor<?xf16>
    %norm_w_cpu = flow.tensor.transfer %output_norm_w : tensor<?xf16>{%n_embd}
        to #hal.device.promise<@device_cpu>

    // Inlined RMS norm (f32 accumulation, f16 I/O)
    %init_sum = tensor.empty(%batch) : tensor<?xf32>
    %zero_f32 = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_f32 : f32) outs(%init_sum : tensor<?xf32>) -> tensor<?xf32>
    %sum_sq = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%hidden_cpu : tensor<?x?xf16>) outs(%sum_init : tensor<?xf32>) {
    ^bb0(%in: f16, %acc: f32):
      %in_f32 = arith.extf %in : f16 to f32
      %sq = arith.mulf %in_f32, %in_f32 : f32
      %sum = arith.addf %acc, %sq : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>

    %n_embd_i32 = arith.index_cast %n_embd : index to i32
    %n_embd_f32 = arith.sitofp %n_embd_i32 : i32 to f32
    %rms_init = tensor.empty(%batch) : tensor<?xf32>
    %rms = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%sum_sq : tensor<?xf32>) outs(%rms_init : tensor<?xf32>) {
    ^bb0(%s: f32, %out: f32):
      %mean = arith.divf %s, %n_embd_f32 : f32
      %with_eps = arith.addf %mean, %rms_eps : f32
      %rms_val = math.sqrt %with_eps : f32
      linalg.yield %rms_val : f32
    } -> tensor<?xf32>

    %norm_out_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf16>
    %normalized = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%hidden_cpu, %rms, %norm_w_cpu : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>)
      outs(%norm_out_init : tensor<?x?xf16>) {
    ^bb0(%x: f16, %rms_val: f32, %w: f16, %out: f16):
      %x_f32 = arith.extf %x : f16 to f32
      %w_f32 = arith.extf %w : f16 to f32
      %norm_val = arith.divf %x_f32, %rms_val : f32
      %scaled = arith.mulf %norm_val, %w_f32 : f32
      %result = arith.truncf %scaled : f32 to f16
      linalg.yield %result : f16
    } -> tensor<?x?xf16>

    // ===== CPU: LM head projection (INLINED) =====
    %output_weight_raw = flow.tensor.constant #flow.parameter.named<"model"::"output.weight"> : tensor<2048x50304xf16>
    %output_weight = tensor.cast %output_weight_raw : tensor<2048x50304xf16> to tensor<?x?xf16>
    %out_w_cpu = flow.tensor.transfer %output_weight : tensor<?x?xf16>{%n_embd, %n_vocab}
        to #hal.device.promise<@device_cpu>
    %logits_empty = tensor.empty(%batch, %n_vocab) : tensor<?x?xf16>
    %zero = arith.constant 0.0 : f16
    %logits_init = linalg.fill ins(%zero : f16) outs(%logits_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
    %logits_cpu = linalg.matmul ins(%normalized, %out_w_cpu : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    // Transfer result back to GPU (for consistent output device).
    %logits = flow.tensor.transfer %logits_cpu : tensor<?x?xf16>{%batch, %n_vocab}
        to #hal.device.promise<@device_gpu>

    util.return %logits, %final_cache : tensor<?x?xf16>, !util.list<?>
  }

}
