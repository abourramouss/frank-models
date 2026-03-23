// Dense LLM inference with fully-compiled autoregressive generation loop.
//
// Unlike the MoE variant which needs Python orchestration, this module
// contains a @generate entry point that runs the entire decode loop
// inside a single VMFB via scf.while.
//
// Entry points:
//   @allocate_kv_cache(n_blocks, block_size) -> cache
//   @decode(token, position, cache, ...) -> (logits, cache_out)
//   @generate(token, cache, block_tables, max_steps, eos_token) -> (output_tokens, n_generated, cache_out)

module @llm_inference_qwen {

  // ===== Hyperparameter imports =====
  util.func private @hparams.vocab_size() -> i64
  util.func private @hparams.block_count() -> i64
  util.func private @hparams.embedding_length() -> i64
  util.func private @hparams.attention_head_count() -> i64
  util.func private @hparams.attention_head_count_kv() -> i64
  util.func private @hparams.feed_forward_length() -> i64
  util.func private @hparams.head_dim() -> i64
  util.func private @hparams.rope_freq_base() -> f32
  util.func private @hparams.layer_norm_rms_epsilon() -> f32

  // ===== Parameter imports =====
  util.func private @model_params.token_embd_weight() -> tensor<?x?xf16>
  util.func private @model_params.output_norm_weight() -> tensor<?xf16>
  util.func private @model_params.output_weight() -> tensor<?x?xf16>

  // ===== Component imports =====
  util.func private @embedding_components.embedding_lookup_1d(
      tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

  util.func private @layer_norm_components.layer_norm(
      tensor<?x?xf16>, f32) -> tensor<?x?xf16>

  util.func private @kvcache_components.allocate(
      index, index, index, index) -> !util.list<?>

  // Dense decode layer
  util.func private @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(
      tensor<?x?xf16>, tensor<?xi64>, !util.list<?>,
      tensor<?x?x?xi32>, tensor<?x?xi32>, index,
      i32, index, index, index, index, f32, f32, f32
  ) -> (tensor<?x?xf16>, !util.list<?>)

  // ===== KV Cache Allocation =====

  util.func public @allocate_kv_cache(
      %n_blocks: index, %block_size: index
  ) -> !util.list<?> {
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %head_dim_i64 = util.call @hparams.head_dim() : () -> i64
    %head_dim = arith.index_cast %head_dim_i64 : i64 to index

    %cache = util.call @kvcache_components.allocate(
        %n_blocks, %block_size, %n_head_kv, %head_dim)
        : (index, index, index, index) -> !util.list<?>
    util.return %cache : !util.list<?>
  }

  // ===== Prefill: process all prompt tokens in one invoke =====
  // Returns logits for the last token + updated cache.

  util.func public @prefill_all(
      %prompt_tokens: tensor<?xi64>,    // [prompt_len]
      %prompt_len: index,
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>
  ) -> (tensor<?x?xf16>, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_f16 = arith.constant 0.0 : f16

    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_ff_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %rope_freq_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %n_layer = arith.index_cast %n_layer_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_ff = arith.index_cast %n_ff_i64 : i64 to index
    %rope_freq_scale = arith.constant 1.0 : f32
    %c1_idx = arith.constant 1 : index

    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %output_weight = util.call @model_params.output_weight() : () -> tensor<?x?xf16>

    // Process all prompt tokens in scf.for
    %final_cache, %final_hidden = scf.for %pi = %c0 to %prompt_len step %c1
        iter_args(%pc = %cache, %dummy_h = %tok_embd_weight)
        -> (!util.list<?>, tensor<?x?xf16>) {
      %tok_i64 = tensor.extract %prompt_tokens[%pi] : tensor<?xi64>
      %tok_t = tensor.from_elements %tok_i64 : tensor<1xi64>
      %tok_dyn = tensor.cast %tok_t : tensor<1xi64> to tensor<?xi64>

      %pi_i64 = arith.index_cast %pi : index to i64
      %pos_t = tensor.from_elements %pi_i64 : tensor<1xi64>
      %pos_dyn = tensor.cast %pos_t : tensor<1xi64> to tensor<?xi64>

      %pi_i32 = arith.index_cast %pi : index to i32
      %ctx_init = tensor.empty(%n_layer, %c1_idx) : tensor<?x?xi32>
      %ctx_lens = linalg.fill ins(%pi_i32 : i32) outs(%ctx_init : tensor<?x?xi32>) -> tensor<?x?xi32>
      %pi_gt0 = arith.cmpi ugt, %pi, %c0 : index
      %max_ctx = arith.select %pi_gt0, %pi, %c1 : index

      %emb = util.call @embedding_components.embedding_lookup_1d(%tok_embd_weight, %tok_dyn)
          : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

      %hidden, %lc = scf.for %li = %c0 to %n_layer step %c1
          iter_args(%h = %emb, %c = %pc) -> (tensor<?x?xf16>, !util.list<?>) {
        %li_i32 = arith.index_cast %li : index to i32
        %lo, %cu = util.call @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(
            %h, %pos_dyn, %c, %block_tables, %ctx_lens, %max_ctx,
            %li_i32, %n_head, %n_head_kv, %n_embd, %n_ff,
            %rms_eps, %rope_freq_base, %rope_freq_scale)
            : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>,
               tensor<?x?x?xi32>, tensor<?x?xi32>, index,
               i32, index, index, index, index, f32, f32, f32)
            -> (tensor<?x?xf16>, !util.list<?>)
        scf.yield %lo, %cu : tensor<?x?xf16>, !util.list<?>
      }
      scf.yield %lc, %hidden : !util.list<?>, tensor<?x?xf16>
    }

    // Output norm + LM head on last token's hidden state
    %normalized = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    %logits_empty = tensor.empty(%c1_idx, %n_vocab) : tensor<?x?xf16>
    %logits_init = linalg.fill ins(%zero_f16 : f16) outs(%logits_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
    %logits = linalg.matmul ins(%normalized, %output_weight : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    util.return %logits, %final_cache : tensor<?x?xf16>, !util.list<?>
  }

  // ===== Single decode step (for external use / testing) =====

  util.func public @decode(
      %tokens: tensor<?xi64>,
      %positions: tensor<?xi64>,
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>,
      %context_lens: tensor<?x?xi32>,
      %max_context_len: index
  ) -> (tensor<?x?xf16>, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_ff_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %rope_freq_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %n_layer = arith.index_cast %n_layer_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_ff = arith.index_cast %n_ff_i64 : i64 to index

    %batch = tensor.dim %tokens, %c0 : tensor<?xi64>

    // Embedding
    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %embeddings = util.call @embedding_components.embedding_lookup_1d(%tok_embd_weight, %tokens)
        : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

    // Transformer layers
    %rope_freq_scale = arith.constant 1.0 : f32
    %final_hidden, %final_cache = scf.for %layer_idx = %c0 to %n_layer step %c1
        iter_args(%hidden = %embeddings, %cache_iter = %cache) -> (tensor<?x?xf16>, !util.list<?>) {
      %layer_idx_i32 = arith.index_cast %layer_idx : index to i32

      %layer_out, %cache_updated = util.call @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(
          %hidden, %positions, %cache_iter,
          %block_tables, %context_lens, %max_context_len,
          %layer_idx_i32,
          %n_head, %n_head_kv, %n_embd, %n_ff,
          %rms_eps, %rope_freq_base, %rope_freq_scale)
          : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>,
             tensor<?x?x?xi32>, tensor<?x?xi32>, index,
             i32, index, index, index, index, f32, f32, f32)
          -> (tensor<?x?xf16>, !util.list<?>)

      scf.yield %layer_out, %cache_updated : tensor<?x?xf16>, !util.list<?>
    }

    // Output norm + LM head
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %normalized = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    %output_weight = util.call @model_params.output_weight() : () -> tensor<?x?xf16>
    %logits_empty = tensor.empty(%batch, %n_vocab) : tensor<?x?xf16>
    %zero = arith.constant 0.0 : f16
    %logits_init = linalg.fill ins(%zero : f16) outs(%logits_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
    %logits = linalg.matmul ins(%normalized, %output_weight : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    util.return %logits, %final_cache : tensor<?x?xf16>, !util.list<?>
  }

  // Autoregressive generation from a seed token. Prompt processing must be
  // done externally via @decode calls before invoking this.
  util.func public @generate(
      %seed_token: i64,               // first token to start generation
      %cache: !util.list<?>,          // pre-allocated KV cache
      %block_tables: tensor<?x?x?xi32>,
      %max_steps: index,              // maximum tokens to generate
      %eos_token: i64,                // EOS token ID for early stopping
      %start_pos: i64                 // position to start from (= prompt_len)
  ) -> (tensor<?xi64>,                // generated token IDs [max_steps]
        index,                        // number of tokens actually generated
        !util.list<?>) {              // final cache state
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant true
    %zero_f16 = arith.constant 0.0 : f16

    // Load hparams
    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_ff_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %rope_freq_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %n_layer = arith.index_cast %n_layer_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_ff = arith.index_cast %n_ff_i64 : i64 to index
    %rope_freq_scale = arith.constant 1.0 : f32

    // Pre-load weights (outside the loop)
    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %output_weight = util.call @model_params.output_weight() : () -> tensor<?x?xf16>

    // Output buffer for generated tokens
    %output_buf_init = tensor.empty(%max_steps) : tensor<?xi64>
    %output_buf = linalg.fill ins(%c0_i64 : i64) outs(%output_buf_init : tensor<?xi64>) -> tensor<?xi64>

    // Batch = 1 for autoregressive generation
    %c1_idx = arith.constant 1 : index
    %n_layer_i32 = arith.index_cast %n_layer : index to i32

    // ---- Autoregressive loop via scf.while ----
    // Carry: (current_token, current_pos, cache, output_buf, step, continue_flag)
    %r_token, %r_pos, %result_cache, %result_token_buf, %result_n, %r_cont = scf.while (
        %token = %seed_token,
        %pos = %start_pos,
        %cache_w = %cache,
        %out_buf = %output_buf,
        %step = %c0,
        %cont = %true
    ) : (i64, i64, !util.list<?>, tensor<?xi64>, index, i1)
      -> (i64, i64, !util.list<?>, tensor<?xi64>, index, i1) {
      // Condition: continue && step < max_steps
      %under_limit = arith.cmpi ult, %step, %max_steps : index
      %keep_going = arith.andi %cont, %under_limit : i1
      scf.condition(%keep_going)
          %token, %pos, %cache_w, %out_buf, %step, %cont
          : i64, i64, !util.list<?>, tensor<?xi64>, index, i1
    } do {
    ^bb0(%token_d: i64, %pos_d: i64, %cache_d: !util.list<?>,
         %out_buf_d: tensor<?xi64>, %step_d: index, %cont_d: i1):

      // Build input tensors (batch=1)
      %token_tensor = tensor.from_elements %token_d : tensor<1xi64>
      %token_dyn = tensor.cast %token_tensor : tensor<1xi64> to tensor<?xi64>

      %pos_tensor = tensor.from_elements %pos_d : tensor<1xi64>
      %pos_dyn = tensor.cast %pos_tensor : tensor<1xi64> to tensor<?xi64>

      // context_lens: [n_layers, batch=1] = all set to current pos
      %pos_i32 = arith.trunci %pos_d : i64 to i32
      %ctx_lens_init = tensor.empty(%n_layer, %c1_idx) : tensor<?x?xi32>
      %ctx_lens = linalg.fill ins(%pos_i32 : i32) outs(%ctx_lens_init : tensor<?x?xi32>) -> tensor<?x?xi32>

      %pos_idx = arith.index_cast %pos_d : i64 to index

      // ---- Embedding ----
      %embeddings = util.call @embedding_components.embedding_lookup_1d(
          %tok_embd_weight, %token_dyn)
          : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

      // ---- Transformer layers ----
      %final_hidden, %loop_cache = scf.for %li = %c0 to %n_layer step %c1
          iter_args(%h = %embeddings, %c = %cache_d) -> (tensor<?x?xf16>, !util.list<?>) {
        %li_i32 = arith.index_cast %li : index to i32
        %lo, %cu = util.call @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(
            %h, %pos_dyn, %c,
            %block_tables, %ctx_lens, %pos_idx,
            %li_i32,
            %n_head, %n_head_kv, %n_embd, %n_ff,
            %rms_eps, %rope_freq_base, %rope_freq_scale)
            : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>,
               tensor<?x?x?xi32>, tensor<?x?xi32>, index,
               i32, index, index, index, index, f32, f32, f32)
            -> (tensor<?x?xf16>, !util.list<?>)
        scf.yield %lo, %cu : tensor<?x?xf16>, !util.list<?>
      }

      // ---- Output norm ----
      %normalized = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

      // ---- LM head: [1, n_embd] @ [n_embd, vocab] -> [1, vocab] ----
      %logits_empty = tensor.empty(%c1_idx, %n_vocab) : tensor<?x?xf16>
      %logits_init = linalg.fill ins(%zero_f16 : f16) outs(%logits_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
      %logits = linalg.matmul ins(%normalized, %output_weight : tensor<?x?xf16>, tensor<?x?xf16>)
          outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>

      // ---- Argmax over vocab (greedy decoding) ----
      // Find the index of the maximum logit in logits[0, :]
      %neg_inf = arith.constant 0xFC00 : f16  // -inf in f16
      %neg_one = arith.constant -1 : i64

      // Reduce to find argmax
      %argmax_init = tensor.empty() : tensor<f16>
      %argmax_idx_init = tensor.empty() : tensor<i64>
      %max_val_filled = linalg.fill ins(%neg_inf : f16) outs(%argmax_init : tensor<f16>) -> tensor<f16>
      %max_idx_filled = linalg.fill ins(%neg_one : i64) outs(%argmax_idx_init : tensor<i64>) -> tensor<i64>

      // Collapse logits to 1D [vocab] (take row 0)
      %logits_1d = tensor.extract_slice %logits[0, 0] [1, %n_vocab] [1, 1]
          : tensor<?x?xf16> to tensor<?xf16>

      %max_val_result, %max_idx_result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> ()>,
          affine_map<(d0) -> ()>
        ],
        iterator_types = ["reduction"]
      } ins(%logits_1d : tensor<?xf16>)
        outs(%max_val_filled, %max_idx_filled : tensor<f16>, tensor<i64>) {
      ^bb0(%val: f16, %cur_max: f16, %cur_idx: i64):
        %idx = linalg.index 0 : index
        %idx_i64 = arith.index_cast %idx : index to i64
        %is_greater = arith.cmpf ogt, %val, %cur_max : f16
        %new_max = arith.select %is_greater, %val, %cur_max : f16
        %new_idx = arith.select %is_greater, %idx_i64, %cur_idx : i64
        linalg.yield %new_max, %new_idx : f16, i64
      } -> (tensor<f16>, tensor<i64>)

      %next_token = tensor.extract %max_idx_result[] : tensor<i64>

      // Store token in output buffer
      %out_buf_updated = tensor.insert %next_token into %out_buf_d[%step_d] : tensor<?xi64>

      // Check EOS
      %is_eos = arith.cmpi eq, %next_token, %eos_token : i64
      %not_eos = arith.xori %is_eos, %true : i1

      // Advance position and step
      %next_pos = arith.addi %pos_d, %c1_i64 : i64
      %next_step = arith.addi %step_d, %c1 : index

      scf.yield %next_token, %next_pos, %loop_cache, %out_buf_updated,
                %next_step, %not_eos
          : i64, i64, !util.list<?>, tensor<?xi64>, index, i1
    }

    util.return %result_token_buf, %result_n, %result_cache
        : tensor<?xi64>, index, !util.list<?>
  }

  // FIXME: generate_from_prompt blocked by ConvertToStreamPass bug:
  // scf.while with !util.list<?> + nested scf.for drops operand count.
  // Tracked in issues/stream-conversion-while-list.md
  util.func private @generate_from_prompt_DISABLED(
      %prompt_tokens: tensor<?xi64>,      // [prompt_len] token IDs
      %prompt_len: index,                 // number of prompt tokens
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>,
      %max_new_tokens: index,
      %eos_token: i64
  ) -> (tensor<?xi64>, index) {            // (generated_tokens, n_generated)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant true
    %zero_f16 = arith.constant 0.0 : f16
    %c1_idx = arith.constant 1 : index

    // Load hparams
    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_ff_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %rope_freq_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %n_layer = arith.index_cast %n_layer_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_ff = arith.index_cast %n_ff_i64 : i64 to index
    %rope_freq_scale = arith.constant 1.0 : f32

    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %output_weight = util.call @model_params.output_weight() : () -> tensor<?x?xf16>

    // ---- Phase 1: Process prompt tokens via scf.for ----
    %prompt_cache = scf.for %pi = %c0 to %prompt_len step %c1
        iter_args(%pc = %cache) -> (!util.list<?>) {
      // Extract token at position pi
      %tok_i64 = tensor.extract %prompt_tokens[%pi] : tensor<?xi64>
      %tok_tensor = tensor.from_elements %tok_i64 : tensor<1xi64>
      %tok_dyn = tensor.cast %tok_tensor : tensor<1xi64> to tensor<?xi64>

      %pi_i64 = arith.index_cast %pi : index to i64
      %pos_tensor = tensor.from_elements %pi_i64 : tensor<1xi64>
      %pos_dyn = tensor.cast %pos_tensor : tensor<1xi64> to tensor<?xi64>

      %pi_i32 = arith.index_cast %pi : index to i32
      %ctx_lens_init = tensor.empty(%n_layer, %c1_idx) : tensor<?x?xi32>
      %ctx_lens = linalg.fill ins(%pi_i32 : i32) outs(%ctx_lens_init : tensor<?x?xi32>) -> tensor<?x?xi32>
      // max_context_len = max(1, pi)
      %pi_gt0 = arith.cmpi ugt, %pi, %c0 : index
      %max_ctx = arith.select %pi_gt0, %pi, %c1 : index

      // Embedding
      %embeddings = util.call @embedding_components.embedding_lookup_1d(%tok_embd_weight, %tok_dyn)
          : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

      // Transformer layers
      %hidden, %lc = scf.for %li = %c0 to %n_layer step %c1
          iter_args(%h = %embeddings, %c = %pc) -> (tensor<?x?xf16>, !util.list<?>) {
        %li_i32 = arith.index_cast %li : index to i32
        %lo, %cu = util.call @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(
            %h, %pos_dyn, %c, %block_tables, %ctx_lens, %max_ctx,
            %li_i32, %n_head, %n_head_kv, %n_embd, %n_ff,
            %rms_eps, %rope_freq_base, %rope_freq_scale)
            : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>,
               tensor<?x?x?xi32>, tensor<?x?xi32>, index,
               i32, index, index, index, index, f32, f32, f32)
            -> (tensor<?x?xf16>, !util.list<?>)
        scf.yield %lo, %cu : tensor<?x?xf16>, !util.list<?>
      }
      scf.yield %lc : !util.list<?>
    }

    // Get first generated token: run one more decode on last prompt position
    %last_idx = arith.subi %prompt_len, %c1 : index
    %last_tok_i64 = tensor.extract %prompt_tokens[%last_idx] : tensor<?xi64>
    %last_tok_t = tensor.from_elements %last_tok_i64 : tensor<1xi64>
    %last_tok_dyn = tensor.cast %last_tok_t : tensor<1xi64> to tensor<?xi64>
    %last_pos_i64 = arith.index_cast %last_idx : index to i64
    %last_pos_t = tensor.from_elements %last_pos_i64 : tensor<1xi64>
    %last_pos_dyn = tensor.cast %last_pos_t : tensor<1xi64> to tensor<?xi64>
    %last_pi_i32 = arith.index_cast %last_idx : index to i32
    %last_ctx_init = tensor.empty(%n_layer, %c1_idx) : tensor<?x?xi32>
    %last_ctx_lens = linalg.fill ins(%last_pi_i32 : i32) outs(%last_ctx_init : tensor<?x?xi32>) -> tensor<?x?xi32>
    %last_max_ctx = arith.select %true, %prompt_len, %c1 : index

    %last_emb = util.call @embedding_components.embedding_lookup_1d(%tok_embd_weight, %last_tok_dyn)
        : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>
    %last_hidden, %last_cache = scf.for %li = %c0 to %n_layer step %c1
        iter_args(%h = %last_emb, %c = %prompt_cache) -> (tensor<?x?xf16>, !util.list<?>) {
      %li_i32 = arith.index_cast %li : index to i32
      %lo, %cu = util.call @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(
          %h, %last_pos_dyn, %c, %block_tables, %last_ctx_lens, %last_max_ctx,
          %li_i32, %n_head, %n_head_kv, %n_embd, %n_ff,
          %rms_eps, %rope_freq_base, %rope_freq_scale)
          : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>,
             tensor<?x?x?xi32>, tensor<?x?xi32>, index,
             i32, index, index, index, index, f32, f32, f32)
          -> (tensor<?x?xf16>, !util.list<?>)
      scf.yield %lo, %cu : tensor<?x?xf16>, !util.list<?>
    }

    // Output norm + LM head + argmax for first token
    %last_normed = util.call @rms_norm_components.rms_norm_linalg(
        %last_hidden, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
    %lm_empty = tensor.empty(%c1_idx, %n_vocab) : tensor<?x?xf16>
    %lm_init = linalg.fill ins(%zero_f16 : f16) outs(%lm_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
    %lm_logits = linalg.matmul ins(%last_normed, %output_weight : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%lm_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    %neg_inf = arith.constant 0xFC00 : f16
    %neg_one = arith.constant -1 : i64
    %am_init = tensor.empty() : tensor<f16>
    %ami_init = tensor.empty() : tensor<i64>
    %am_fill = linalg.fill ins(%neg_inf : f16) outs(%am_init : tensor<f16>) -> tensor<f16>
    %ami_fill = linalg.fill ins(%neg_one : i64) outs(%ami_init : tensor<i64>) -> tensor<i64>
    %first_logits_1d = tensor.extract_slice %lm_logits[0, 0] [1, %n_vocab] [1, 1]
        : tensor<?x?xf16> to tensor<?xf16>
    %first_mv, %first_mi = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%first_logits_1d : tensor<?xf16>)
      outs(%am_fill, %ami_fill : tensor<f16>, tensor<i64>) {
    ^bb0(%val: f16, %cm: f16, %ci: i64):
      %idx = linalg.index 0 : index
      %idx_i64 = arith.index_cast %idx : index to i64
      %gt = arith.cmpf ogt, %val, %cm : f16
      %nm = arith.select %gt, %val, %cm : f16
      %ni = arith.select %gt, %idx_i64, %ci : i64
      linalg.yield %nm, %ni : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    %first_token = tensor.extract %first_mi[] : tensor<i64>

    // ---- Phase 2: Autoregressive generation via scf.while ----
    %output_buf_init = tensor.empty(%max_new_tokens) : tensor<?xi64>
    %output_buf = linalg.fill ins(%c0_i64 : i64) outs(%output_buf_init : tensor<?xi64>) -> tensor<?xi64>
    // Store first token
    %output_buf_1 = tensor.insert %first_token into %output_buf[%c0] : tensor<?xi64>

    %prompt_len_i64 = arith.index_cast %prompt_len : index to i64
    %is_first_eos = arith.cmpi eq, %first_token, %eos_token : i64
    %not_first_eos = arith.xori %is_first_eos, %true : i1

    %r_tok, %r_pos, %r_cache, %r_buf, %r_n, %r_cont = scf.while (
        %token = %first_token,
        %pos = %prompt_len_i64,
        %cache_w = %last_cache,
        %out_buf = %output_buf_1,
        %step = %c1,
        %cont = %not_first_eos
    ) : (i64, i64, !util.list<?>, tensor<?xi64>, index, i1)
      -> (i64, i64, !util.list<?>, tensor<?xi64>, index, i1) {
      %under_limit = arith.cmpi ult, %step, %max_new_tokens : index
      %keep_going = arith.andi %cont, %under_limit : i1
      scf.condition(%keep_going)
          %token, %pos, %cache_w, %out_buf, %step, %cont
          : i64, i64, !util.list<?>, tensor<?xi64>, index, i1
    } do {
    ^bb0(%token_d: i64, %pos_d: i64, %cache_d: !util.list<?>,
         %out_buf_d: tensor<?xi64>, %step_d: index, %cont_d: i1):

      %token_tensor = tensor.from_elements %token_d : tensor<1xi64>
      %token_dyn = tensor.cast %token_tensor : tensor<1xi64> to tensor<?xi64>
      %pos_tensor = tensor.from_elements %pos_d : tensor<1xi64>
      %pos_dyn = tensor.cast %pos_tensor : tensor<1xi64> to tensor<?xi64>

      %pos_i32 = arith.trunci %pos_d : i64 to i32
      %ctx_lens_init_w = tensor.empty(%n_layer, %c1_idx) : tensor<?x?xi32>
      %ctx_lens_w = linalg.fill ins(%pos_i32 : i32) outs(%ctx_lens_init_w : tensor<?x?xi32>) -> tensor<?x?xi32>
      %pos_idx = arith.index_cast %pos_d : i64 to index

      %emb_w = util.call @embedding_components.embedding_lookup_1d(%tok_embd_weight, %token_dyn)
          : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

      %final_hidden, %loop_cache = scf.for %li = %c0 to %n_layer step %c1
          iter_args(%h = %emb_w, %c = %cache_d) -> (tensor<?x?xf16>, !util.list<?>) {
        %li_i32 = arith.index_cast %li : index to i32
        %lo, %cu = util.call @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(
            %h, %pos_dyn, %c, %block_tables, %ctx_lens_w, %pos_idx,
            %li_i32, %n_head, %n_head_kv, %n_embd, %n_ff,
            %rms_eps, %rope_freq_base, %rope_freq_scale)
            : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>,
               tensor<?x?x?xi32>, tensor<?x?xi32>, index,
               i32, index, index, index, index, f32, f32, f32)
            -> (tensor<?x?xf16>, !util.list<?>)
        scf.yield %lo, %cu : tensor<?x?xf16>, !util.list<?>
      }

      %norm_w = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

      %lg_empty = tensor.empty(%c1_idx, %n_vocab) : tensor<?x?xf16>
      %lg_init = linalg.fill ins(%zero_f16 : f16) outs(%lg_empty : tensor<?x?xf16>) -> tensor<?x?xf16>
      %lg = linalg.matmul ins(%norm_w, %output_weight : tensor<?x?xf16>, tensor<?x?xf16>)
          outs(%lg_init : tensor<?x?xf16>) -> tensor<?x?xf16>

      %lg_1d = tensor.extract_slice %lg[0, 0] [1, %n_vocab] [1, 1]
          : tensor<?x?xf16> to tensor<?xf16>
      %mv_init = linalg.fill ins(%neg_inf : f16) outs(%am_init : tensor<f16>) -> tensor<f16>
      %mi_init = linalg.fill ins(%neg_one : i64) outs(%ami_init : tensor<i64>) -> tensor<i64>
      %mv_r, %mi_r = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]
      } ins(%lg_1d : tensor<?xf16>)
        outs(%mv_init, %mi_init : tensor<f16>, tensor<i64>) {
      ^bb0(%val: f16, %cm: f16, %ci: i64):
        %idx = linalg.index 0 : index
        %idx_i64 = arith.index_cast %idx : index to i64
        %gt = arith.cmpf ogt, %val, %cm : f16
        %nm = arith.select %gt, %val, %cm : f16
        %ni = arith.select %gt, %idx_i64, %ci : i64
        linalg.yield %nm, %ni : f16, i64
      } -> (tensor<f16>, tensor<i64>)
      %next_token = tensor.extract %mi_r[] : tensor<i64>

      %out_buf_upd = tensor.insert %next_token into %out_buf_d[%step_d] : tensor<?xi64>
      %is_eos = arith.cmpi eq, %next_token, %eos_token : i64
      %not_eos = arith.xori %is_eos, %true : i1
      %next_pos = arith.addi %pos_d, %c1_i64 : i64
      %next_step = arith.addi %step_d, %c1 : index

      scf.yield %next_token, %next_pos, %loop_cache, %out_buf_upd,
                %next_step, %not_eos
          : i64, i64, !util.list<?>, tensor<?xi64>, index, i1
    }

    util.return %r_buf, %r_n : tensor<?xi64>, index
  }

  // ===== Single entry point: prompt → tokens in one invoke =====
  // Python calls this once. Everything else is MLIR.

  util.func public @run(
      %prompt_tokens: tensor<?xi64>,   // [prompt_len]
      %prompt_len: index,
      %max_new_tokens: index,
      %eos_token: i64
  ) -> (tensor<?xi64>, index) {        // (generated_tokens, n_generated)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_layer = arith.index_cast %n_layer_i64 : i64 to index

    // Compute cache size
    %total_len = arith.addi %prompt_len, %max_new_tokens : index
    %total_plus = arith.addi %total_len, %c16 : index
    %c1_sub = arith.subi %total_plus, %c1 : index
    %max_blocks_per_seq = arith.divui %c1_sub, %c16 : index
    %n_blocks = arith.muli %n_layer, %max_blocks_per_seq : index

    // Allocate cache
    %cache = util.call @allocate_kv_cache(%n_blocks, %c16)
        : (index, index) -> !util.list<?>

    // Build block_tables [n_layers, 1, max_blocks_per_seq]
    %bt_init = tensor.empty(%n_layer, %c1, %max_blocks_per_seq) : tensor<?x?x?xi32>
    %block_tables = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%bt_init : tensor<?x?x?xi32>) {
    ^bb0(%out: i32):
      %i0 = linalg.index 0 : index
      %i2 = linalg.index 2 : index
      %offset = arith.muli %i0, %max_blocks_per_seq : index
      %block_idx = arith.addi %offset, %i2 : index
      %block_i32 = arith.index_cast %block_idx : index to i32
      linalg.yield %block_i32 : i32
    } -> tensor<?x?x?xi32>

    // Prefill prompt
    %logits, %prefill_cache = util.call @prefill_all(
        %prompt_tokens, %prompt_len, %cache, %block_tables)
        : (tensor<?xi64>, index, !util.list<?>, tensor<?x?x?xi32>)
        -> (tensor<?x?xf16>, !util.list<?>)

    // Argmax on prefill logits
    %neg_inf = arith.constant 0xFC00 : f16
    %neg_one = arith.constant -1 : i64
    %am_init = tensor.empty() : tensor<f16>
    %ami_init = tensor.empty() : tensor<i64>
    %am_fill = linalg.fill ins(%neg_inf : f16) outs(%am_init : tensor<f16>) -> tensor<f16>
    %ami_fill = linalg.fill ins(%neg_one : i64) outs(%ami_init : tensor<i64>) -> tensor<i64>
    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %logits_1d = tensor.extract_slice %logits[0, 0] [1, %n_vocab] [1, 1]
        : tensor<?x?xf16> to tensor<?xf16>
    %mv, %mi = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%logits_1d : tensor<?xf16>)
      outs(%am_fill, %ami_fill : tensor<f16>, tensor<i64>) {
    ^bb0(%val: f16, %cm: f16, %ci: i64):
      %idx = linalg.index 0 : index
      %idx_i64 = arith.index_cast %idx : index to i64
      %gt = arith.cmpf ogt, %val, %cm : f16
      %nm = arith.select %gt, %val, %cm : f16
      %ni = arith.select %gt, %idx_i64, %ci : i64
      linalg.yield %nm, %ni : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    %first_token = tensor.extract %mi[] : tensor<i64>

    // Generate
    %prompt_len_i64 = arith.index_cast %prompt_len : index to i64
    %gen_tokens, %n_gen, %gen_cache = util.call @generate(
        %first_token, %prefill_cache, %block_tables, %max_new_tokens, %eos_token, %prompt_len_i64)
        : (i64, !util.list<?>, tensor<?x?x?xi32>, index, i64, i64)
        -> (tensor<?xi64>, index, !util.list<?>)

    util.return %gen_tokens, %n_gen : tensor<?xi64>, index
  }

}
