// CPU-side post-processing: output norm + LM head projection.
// Compiled separately with --iree-hal-target-backends=llvm-cpu
// and loaded alongside the GPU VMFB at runtime.
//
// This replaces 2-3 GPU dispatches (~12-18ms HAL overhead) with
// a single CPU module call (~11ms total).

module @cpu_postprocess {

  // Import the RMS norm implementation
  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

  // Parameters loaded from same .irpa archive
  util.func private @model_params.output_norm_weight() -> tensor<?xf16>
  util.func private @model_params.output_weight() -> tensor<?x?xf16>

  // Hyperparameters
  util.func private @hparams.vocab_size() -> i64
  util.func private @hparams.layer_norm_rms_epsilon() -> f32

  /// Output norm + LM head + argmax on CPU.
  /// Input: hidden [batch, n_embd] from GPU decode_body
  /// Output: logits [batch, vocab_size]
  util.func public @postprocess(
      %hidden: tensor<?x?xf16>   // [batch, n_embd]
  ) -> tensor<?x?xf16> {         // logits: [batch, vocab_size]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    %batch = tensor.dim %hidden, %c0 : tensor<?x?xf16>

    // Output normalization
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %normalized = util.call @rms_norm_components.rms_norm_linalg(
        %hidden, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // LM head projection: [batch, n_embd] @ [n_embd, vocab] -> [batch, vocab]
    %output_weight = util.call @model_params.output_weight() : () -> tensor<?x?xf16>
    %logits_empty = tensor.empty(%batch, %n_vocab) : tensor<?x?xf16>
    %zero = arith.constant 0.0 : f16
    %logits_init = linalg.fill ins(%zero : f16) outs(%logits_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
    %logits = linalg.matmul ins(%normalized, %output_weight : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    util.return %logits : tensor<?x?xf16>
  }
}
