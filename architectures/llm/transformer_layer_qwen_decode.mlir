// Dense Transformer Layer (Decode): pre-norm attention + dense SwiGLU FFN.
// Replaces MoE FFN with a standard dense FFN: gate_proj + up_proj + SwiGLU + down_proj.
//
// Architecture (OLMo-style):
//   input -> attn_norm -> gather(cache) -> attention_block_decode -> scatter_decode(cache)
//         -> +residual -> ffn_norm -> dense_ffn(gate, up, SwiGLU, down) -> +residual -> output

module @transformer_layer_qwen_decode_components {

  // ===== Parameter accessor imports =====

  util.func private @model_params.attn_norm_weight(i32) -> tensor<?xf16>
  util.func private @model_params.ffn_norm_weight(i32) -> tensor<?xf16>
  util.func private @model_params.attn_q_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.attn_k_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.attn_v_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.attn_output_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.attn_q_norm_weight(i32) -> tensor<?xf16>
  util.func private @model_params.attn_k_norm_weight(i32) -> tensor<?xf16>

  // Dense FFN weights (no MoE)
  // gate_up is fused: [n_embd, 2*n_ff] = concat(gate, up) along output dim
  util.func private @model_params.ffn_gate_up_weight(i32) -> tensor<?x?xf16>
  util.func private @model_params.ffn_down_weight(i32) -> tensor<?x?xf16>

  // ===== Component imports =====

  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

  util.func private @layer_norm_components.layer_norm(
      tensor<?x?xf16>, f32) -> tensor<?x?xf16>

  util.func private @kvcache_components.gather(
      !util.list<?>, index, tensor<?x?x?xi32>, tensor<?x?xi32>, index
  ) -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)

  util.func private @kvcache_components.scatter_decode(
      !util.list<?>, index, tensor<?x?x?xf16>, tensor<?x?x?xf16>,
      tensor<?x?x?xi32>, tensor<?xi64>
  ) -> !util.list<?>

  util.func private @attention_block_decode_qwen_components.attention_block_decode_qwen(
      tensor<?x?xf16>, tensor<?xi64>,
      tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
      tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>,
      index, index, index, f32, f32,
      tensor<?xf16>, tensor<?xf16>, f32
  ) -> (tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)

  // ===== Dense transformer layer =====

  util.func public @transformer_layer_qwen_decode(
      %input: tensor<?x?xf16>,
      %positions: tensor<?xi64>,
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>,
      %context_lens: tensor<?x?xi32>,
      %max_context_len: index,
      %layer_idx: i32,
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %n_ff: index,
      %rms_eps: f32,
      %rope_freq_base: f32,
      %rope_freq_scale: f32
  ) -> (tensor<?x?xf16>, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?xf16>
    %layer = arith.index_cast %layer_idx : i32 to index

    // ---- Load parameters ----
    %attn_norm_w = util.call @model_params.attn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %ffn_norm_w = util.call @model_params.ffn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %wq = util.call @model_params.attn_q_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %wk = util.call @model_params.attn_k_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %wv = util.call @model_params.attn_v_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %wo = util.call @model_params.attn_output_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %q_norm_w = util.call @model_params.attn_q_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %k_norm_w = util.call @model_params.attn_k_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %w_gate_up = util.call @model_params.ffn_gate_up_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %w_down = util.call @model_params.ffn_down_weight(%layer_idx) : (i32) -> tensor<?x?xf16>

    // ---- Attention sub-layer ----

    %attn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %input, %attn_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    %k_cached, %v_cached = util.call @kvcache_components.gather(
        %cache, %layer, %block_tables, %context_lens, %max_context_len)
        : (!util.list<?>, index, tensor<?x?x?xi32>, tensor<?x?xi32>, index)
        -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)

    %attn_out, %k_new, %v_new = util.call @attention_block_decode_qwen_components.attention_block_decode_qwen(
        %attn_normed, %positions,
        %k_cached, %v_cached,
        %wq, %wk, %wv, %wo,
        %n_head, %n_head_kv, %n_embd,
        %rope_freq_base, %rope_freq_scale,
        %q_norm_w, %k_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xi64>,
           tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
           tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>,
           index, index, index, f32, f32,
           tensor<?xf16>, tensor<?xf16>, f32)
        -> (tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)

    %cache_updated = util.call @kvcache_components.scatter_decode(
        %cache, %layer, %k_new, %v_new, %block_tables, %positions)
        : (!util.list<?>, index, tensor<?x?x?xf16>, tensor<?x?x?xf16>,
           tensor<?x?x?xi32>, tensor<?xi64>) -> !util.list<?>

    // Residual 1
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

    // ---- Dense FFN sub-layer ----

    %ffn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %residual1, %ffn_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    %zero = arith.constant 0.0 : f16
    %n_ff_2 = arith.muli %n_ff, %c2 : index
    %gate_up_init = tensor.empty(%batch, %n_ff_2) : tensor<?x?xf16>
    %gate_up_filled = linalg.fill ins(%zero : f16) outs(%gate_up_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %gate_up = linalg.matmul ins(%ffn_normed, %w_gate_up : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%gate_up_filled : tensor<?x?xf16>) -> tensor<?x?xf16>

    %gate = tensor.extract_slice %gate_up[0, 0] [%batch, %n_ff] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %up = tensor.extract_slice %gate_up[0, %n_ff] [%batch, %n_ff] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>

    // SwiGLU: silu(gate) * up
    %swiglu_init = tensor.empty(%batch, %n_ff) : tensor<?x?xf16>
    %swiglu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%gate, %up : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%swiglu_init : tensor<?x?xf16>) {
    ^bb0(%g: f16, %u: f16, %out: f16):
      %neg_g = arith.negf %g : f16
      %exp_neg = math.exp %neg_g : f16
      %one = arith.constant 1.0 : f16
      %denom = arith.addf %one, %exp_neg : f16
      %sigmoid = arith.divf %one, %denom : f16
      %silu = arith.mulf %g, %sigmoid : f16
      %result = arith.mulf %silu, %u : f16
      linalg.yield %result : f16
    } -> tensor<?x?xf16>

    %down_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf16>
    %down_filled = linalg.fill ins(%zero : f16) outs(%down_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %ffn_out = linalg.matmul ins(%swiglu, %w_down : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%down_filled : tensor<?x?xf16>) -> tensor<?x?xf16>

    // Residual 2
    %output_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf16>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%residual1, %ffn_out : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%output_init : tensor<?x?xf16>) {
    ^bb0(%a: f16, %b: f16, %out: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    util.return %output, %cache_updated : tensor<?x?xf16>, !util.list<?>
  }

}
