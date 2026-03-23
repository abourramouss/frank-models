// Qwen3-0.6B hyperparameters.
// hidden=1024, heads=16(Q)/8(KV), head_dim=128, FFN=3072, 28 layers, vocab=151936
// RMSNorm with learned weights, QK norm per-head-dim, tied embeddings

module @hparams {
  util.func public @vocab_size() -> i64 { %v = arith.constant 151936 : i64 util.return %v : i64 }
  util.func public @block_count() -> i64 { %v = arith.constant 28 : i64 util.return %v : i64 }
  util.func public @embedding_length() -> i64 { %v = arith.constant 1024 : i64 util.return %v : i64 }
  util.func public @attention_head_count() -> i64 { %v = arith.constant 16 : i64 util.return %v : i64 }
  util.func public @attention_head_count_kv() -> i64 { %v = arith.constant 8 : i64 util.return %v : i64 }
  util.func public @feed_forward_length() -> i64 { %v = arith.constant 3072 : i64 util.return %v : i64 }
  util.func public @expert_count() -> i64 { %v = arith.constant 1 : i64 util.return %v : i64 }
  util.func public @expert_used_count() -> i64 { %v = arith.constant 1 : i64 util.return %v : i64 }
  util.func public @rope_freq_base() -> f32 { %v = arith.constant 1000000.0 : f32 util.return %v : f32 }
  util.func public @layer_norm_rms_epsilon() -> f32 { %v = arith.constant 1.0e-6 : f32 util.return %v : f32 }
  util.func public @use_attention_bias() -> i1 { %v = arith.constant false util.return %v : i1 }
  util.func public @normalize_expert_weights() -> i1 { %v = arith.constant false util.return %v : i1 }
  util.func public @use_qk_norm() -> i1 { %v = arith.constant true util.return %v : i1 }
  util.func public @head_dim() -> i64 { %v = arith.constant 128 : i64 util.return %v : i64 }
}
