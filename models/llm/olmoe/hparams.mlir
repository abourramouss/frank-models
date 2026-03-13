// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// OLMoE-1B-7B hyperparameters.
//
// Architecture summary:
//   vocab_size: 50304
//   block_count: 16 layers
//   embedding_length: 2048 (16 heads x 128 head_dim)
//   attention: 16 heads, 16 KV heads (MHA, no GQA)
//   feed_forward_length: 1024 (per-expert)
//   experts: 64 total, top-8 routing
//
// Reference: https://huggingface.co/allenai/OLMoE-1B-7B-0924

module @hparams {

  util.func public @vocab_size() -> i64 {
    %v = arith.constant 50304 : i64
    util.return %v : i64
  }

  util.func public @block_count() -> i64 {
    %v = arith.constant 16 : i64
    util.return %v : i64
  }

  util.func public @embedding_length() -> i64 {
    %v = arith.constant 2048 : i64
    util.return %v : i64
  }

  util.func public @attention_head_count() -> i64 {
    %v = arith.constant 16 : i64
    util.return %v : i64
  }

  util.func public @attention_head_count_kv() -> i64 {
    %v = arith.constant 16 : i64
    util.return %v : i64
  }

  util.func public @feed_forward_length() -> i64 {
    %v = arith.constant 1024 : i64
    util.return %v : i64
  }

  util.func public @expert_count() -> i64 {
    %v = arith.constant 64 : i64
    util.return %v : i64
  }

  util.func public @expert_used_count() -> i64 {
    %v = arith.constant 8 : i64
    util.return %v : i64
  }

  util.func public @rope_freq_base() -> f32 {
    %v = arith.constant 10000.0 : f32
    util.return %v : f32
  }

  util.func public @layer_norm_rms_epsilon() -> f32 {
    %v = arith.constant 1.0e-5 : f32
    util.return %v : f32
  }

  // OLMoE uses no attention bias and no expert weight normalization.
  util.func public @use_attention_bias() -> i1 {
    %v = arith.constant false
    util.return %v : i1
  }

  util.func public @normalize_expert_weights() -> i1 {
    %v = arith.constant false
    util.return %v : i1
  }

  // OLMoE uses per-head QK norm (RMSNorm on Q and K after projection, before RoPE).
  util.func public @use_qk_norm() -> i1 {
    %v = arith.constant true
    util.return %v : i1
  }

}
