// Qwen3-0.6B parameter accessors.
// hidden=1024, n_head=16, n_head_kv=8, head_dim=128, n_ff=3072, layers=28, vocab=151936
// n_embd_q = 16*128 = 2048, n_embd_kv = 8*128 = 1024
// QKV fused: [1024, 2048+1024+1024] = [1024, 4096], flat=4194304
// O proj: [2048, 1024], flat=2097152
// FFN gate_up: [1024, 6144], flat=6291456
// FFN down: [3072, 1024], flat=3145728
// QK norm: [128] per head_dim

module @model_params {

  util.func public @token_embd_weight() -> tensor<?x?xf16> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"token_embd.weight"> : tensor<151936x1024xf16>
    %d = tensor.cast %w : tensor<151936x1024xf16> to tensor<?x?xf16>
    util.return %d : tensor<?x?xf16>
  }

  util.func public @output_norm_weight() -> tensor<?xf16> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"output_norm.weight"> : tensor<1024xf16>
    %d = tensor.cast %w : tensor<1024xf16> to tensor<?xf16>
    util.return %d : tensor<?xf16>
  }

  util.func public @output_weight() -> tensor<?x?xf16> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"output.weight"> : tensor<1024x151936xf16>
    %d = tensor.cast %w : tensor<1024x151936xf16> to tensor<?x?xf16>
    util.return %d : tensor<?x?xf16>
  }

  // --- RMSNorm weights [28, 1024] ---
  util.func public @attn_norm_weight(%layer: i32) -> tensor<?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_norm.weight"> : tensor<28x1024xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 1024] [1, 1] : tensor<28x1024xf16> to tensor<1x1024xf16>
    %r = tensor.collapse_shape %sl [[0, 1]] : tensor<1x1024xf16> into tensor<1024xf16>
    %d = tensor.cast %r : tensor<1024xf16> to tensor<?xf16>
    util.return %d : tensor<?xf16>
  }

  util.func public @ffn_norm_weight(%layer: i32) -> tensor<?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_norm.weight"> : tensor<28x1024xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 1024] [1, 1] : tensor<28x1024xf16> to tensor<1x1024xf16>
    %r = tensor.collapse_shape %sl [[0, 1]] : tensor<1x1024xf16> into tensor<1024xf16>
    %d = tensor.cast %r : tensor<1024xf16> to tensor<?xf16>
    util.return %d : tensor<?xf16>
  }

  // --- Q weight [28, 2097152] -> [1024, 2048] ---
  util.func public @attn_q_weight(%layer: i32) -> tensor<?x?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q.weight"> : tensor<28x2097152xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 2097152] [1, 1] : tensor<28x2097152xf16> to tensor<1x2097152xf16>
    %f = tensor.collapse_shape %sl [[0, 1]] : tensor<1x2097152xf16> into tensor<2097152xf16>
    %r = tensor.expand_shape %f [[0, 1]] output_shape [1024, 2048] : tensor<2097152xf16> into tensor<1024x2048xf16>
    %d = tensor.cast %r : tensor<1024x2048xf16> to tensor<?x?xf16>
    util.return %d : tensor<?x?xf16>
  }

  // --- K weight [28, 1048576] -> [1024, 1024] ---
  util.func public @attn_k_weight(%layer: i32) -> tensor<?x?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k.weight"> : tensor<28x1048576xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 1048576] [1, 1] : tensor<28x1048576xf16> to tensor<1x1048576xf16>
    %f = tensor.collapse_shape %sl [[0, 1]] : tensor<1x1048576xf16> into tensor<1048576xf16>
    %r = tensor.expand_shape %f [[0, 1]] output_shape [1024, 1024] : tensor<1048576xf16> into tensor<1024x1024xf16>
    %d = tensor.cast %r : tensor<1024x1024xf16> to tensor<?x?xf16>
    util.return %d : tensor<?x?xf16>
  }

  // --- V weight [28, 1048576] -> [1024, 1024] ---
  util.func public @attn_v_weight(%layer: i32) -> tensor<?x?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_v.weight"> : tensor<28x1048576xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 1048576] [1, 1] : tensor<28x1048576xf16> to tensor<1x1048576xf16>
    %f = tensor.collapse_shape %sl [[0, 1]] : tensor<1x1048576xf16> into tensor<1048576xf16>
    %r = tensor.expand_shape %f [[0, 1]] output_shape [1024, 1024] : tensor<1048576xf16> into tensor<1024x1024xf16>
    %d = tensor.cast %r : tensor<1024x1024xf16> to tensor<?x?xf16>
    util.return %d : tensor<?x?xf16>
  }

  // --- O proj [28, 2097152] -> [2048, 1024] ---
  util.func public @attn_output_weight(%layer: i32) -> tensor<?x?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_output.weight"> : tensor<28x2097152xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 2097152] [1, 1] : tensor<28x2097152xf16> to tensor<1x2097152xf16>
    %f = tensor.collapse_shape %sl [[0, 1]] : tensor<1x2097152xf16> into tensor<2097152xf16>
    %r = tensor.expand_shape %f [[0, 1]] output_shape [2048, 1024] : tensor<2097152xf16> into tensor<2048x1024xf16>
    %d = tensor.cast %r : tensor<2048x1024xf16> to tensor<?x?xf16>
    util.return %d : tensor<?x?xf16>
  }

  // --- QK norm weights [28, 128] -> [128] (per head_dim) ---
  util.func public @attn_q_norm_weight(%layer: i32) -> tensor<?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q_norm.weight"> : tensor<28x128xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 128] [1, 1] : tensor<28x128xf16> to tensor<1x128xf16>
    %r = tensor.collapse_shape %sl [[0, 1]] : tensor<1x128xf16> into tensor<128xf16>
    %d = tensor.cast %r : tensor<128xf16> to tensor<?xf16>
    util.return %d : tensor<?xf16>
  }

  util.func public @attn_k_norm_weight(%layer: i32) -> tensor<?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k_norm.weight"> : tensor<28x128xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 128] [1, 1] : tensor<28x128xf16> to tensor<1x128xf16>
    %r = tensor.collapse_shape %sl [[0, 1]] : tensor<1x128xf16> into tensor<128xf16>
    %d = tensor.cast %r : tensor<128xf16> to tensor<?xf16>
    util.return %d : tensor<?xf16>
  }

  // --- FFN gate_up fused [28, 6291456] -> [1024, 6144] ---
  util.func public @ffn_gate_up_weight(%layer: i32) -> tensor<?x?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_gate_up.weight"> : tensor<28x6291456xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 6291456] [1, 1] : tensor<28x6291456xf16> to tensor<1x6291456xf16>
    %f = tensor.collapse_shape %sl [[0, 1]] : tensor<1x6291456xf16> into tensor<6291456xf16>
    %r = tensor.expand_shape %f [[0, 1]] output_shape [1024, 6144] : tensor<6291456xf16> into tensor<1024x6144xf16>
    %d = tensor.cast %r : tensor<1024x6144xf16> to tensor<?x?xf16>
    util.return %d : tensor<?x?xf16>
  }

  // --- FFN down [28, 3145728] -> [3072, 1024] ---
  util.func public @ffn_down_weight(%layer: i32) -> tensor<?x?xf16> {
    %s = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_down.weight"> : tensor<28x3145728xf16>
    %i = arith.index_cast %layer : i32 to index
    %sl = tensor.extract_slice %s[%i, 0] [1, 3145728] [1, 1] : tensor<28x3145728xf16> to tensor<1x3145728xf16>
    %f = tensor.collapse_shape %sl [[0, 1]] : tensor<1x3145728xf16> into tensor<3145728xf16>
    %r = tensor.expand_shape %f [[0, 1]] output_shape [3072, 1024] : tensor<3145728xf16> into tensor<3072x1024xf16>
    %d = tensor.cast %r : tensor<3072x1024xf16> to tensor<?x?xf16>
    util.return %d : tensor<?x?xf16>
  }

}
