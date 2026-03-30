#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map16 = affine_map<(d0) -> ()>
module @llm_inference_qwen {
  // ---- Persistent state for multi-turn chat ----
  util.global private mutable @kv_cache = #util.uninitialized : !util.list<?>
  util.global private mutable @max_seq_len = 0 : index
  util.global private mutable @current_pos = 0 : index
  util.global private mutable @is_initialized = 0 : i1

  util.func private @hparams.vocab_size() -> i64 {
    %c151936_i64 = arith.constant 151936 : i64
    util.return %c151936_i64 : i64
  }
  util.func private @hparams.block_count() -> i64 {
    %c28_i64 = arith.constant 28 : i64
    util.return %c28_i64 : i64
  }
  util.func private @hparams.embedding_length() -> i64 {
    %c1024_i64 = arith.constant 1024 : i64
    util.return %c1024_i64 : i64
  }
  util.func private @hparams.attention_head_count() -> i64 {
    %c16_i64 = arith.constant 16 : i64
    util.return %c16_i64 : i64
  }
  util.func private @hparams.attention_head_count_kv() -> i64 {
    %c8_i64 = arith.constant 8 : i64
    util.return %c8_i64 : i64
  }
  util.func private @hparams.feed_forward_length() -> i64 {
    %c3072_i64 = arith.constant 3072 : i64
    util.return %c3072_i64 : i64
  }
  util.func private @hparams.head_dim() -> i64 {
    %c128_i64 = arith.constant 128 : i64
    util.return %c128_i64 : i64
  }
  util.func private @hparams.rope_freq_base() -> f32 {
    %cst = arith.constant 1.000000e+06 : f32
    util.return %cst : f32
  }
  util.func private @hparams.layer_norm_rms_epsilon() -> f32 {
    %cst = arith.constant 9.99999997E-7 : f32
    util.return %cst : f32
  }
  util.func private @model_params.token_embd_weight() -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"token_embd.weight"> : tensor<151936x1024xf16>
    %cast = tensor.cast %0 : tensor<151936x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.output_norm_weight() -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"output_norm.weight"> : tensor<1024xf16>
    %cast = tensor.cast %0 : tensor<1024xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  util.func private @model_params.output_weight() -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"output.weight"> : tensor<1024x151936xf16>
    %cast = tensor.cast %0 : tensor<1024x151936xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @embedding_components.embedding_lookup_1d(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi64>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?xi64>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %1 = iree_linalg_ext.gather dimension_map = [0] ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?xi64>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
    util.return %1 : tensor<?x?xf16>
  }
  util.func private @rms_norm_components.rms_norm_linalg(%arg0: tensor<?x?xf16>, %arg1: tensor<?xf16>, %arg2: f32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf16>) outs(%1 : tensor<?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %9 = arith.extf %in : f16 to f32
      %10 = arith.mulf %9, %9 : f32
      %11 = arith.addf %out, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<?xf32>
    %3 = arith.index_cast %dim_0 : index to i32
    %4 = arith.sitofp %3 : i32 to f32
    %5 = tensor.empty(%dim) : tensor<?xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%2 : tensor<?xf32>) outs(%5 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.divf %in, %4 : f32
      %10 = arith.addf %9, %arg2 : f32
      %11 = math.sqrt %10 : f32
      linalg.yield %11 : f32
    } -> tensor<?xf32>
    %7 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %6, %arg1 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%7 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f32, %in_2: f16, %out: f16):
      %9 = arith.extf %in : f16 to f32
      %10 = arith.extf %in_2 : f16 to f32
      %11 = arith.divf %9, %in_1 : f32
      %12 = arith.mulf %11, %10 : f32
      %13 = arith.truncf %12 : f32 to f16
      linalg.yield %13 : f16
    } -> tensor<?x?xf16>
    util.return %8 : tensor<?x?xf16>
  }
  util.func private @layer_norm_components.layer_norm(%arg0: tensor<?x?xf16>, %arg1: f32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %dim_0 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %2 = tensor.empty(%dim) : tensor<?xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf16>) outs(%3 : tensor<?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %12 = arith.extf %in : f16 to f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<?xf32>
    %5 = tensor.empty(%dim) : tensor<?xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%4 : tensor<?xf32>) outs(%5 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.divf %in, %1 : f32
      linalg.yield %12 : f32
    } -> tensor<?xf32>
    %7 = tensor.empty(%dim) : tensor<?xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?xf32>) -> tensor<?xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %6 : tensor<?x?xf16>, tensor<?xf32>) outs(%8 : tensor<?xf32>) {
    ^bb0(%in: f16, %in_1: f32, %out: f32):
      %12 = arith.extf %in : f16 to f32
      %13 = arith.subf %12, %in_1 : f32
      %14 = arith.mulf %13, %13 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<?xf32>
    %10 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %6, %9 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>) outs(%10 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f32, %in_2: f32, %out: f16):
      %12 = arith.extf %in : f16 to f32
      %13 = arith.subf %12, %in_1 : f32
      %14 = arith.divf %in_2, %1 : f32
      %15 = arith.addf %14, %arg1 : f32
      %16 = math.sqrt %15 : f32
      %17 = arith.divf %13, %16 : f32
      %18 = arith.truncf %17 : f32 to f16
      linalg.yield %18 : f16
    } -> tensor<?x?xf16>
    util.return %11 : tensor<?x?xf16>
  }
  // ---- Direct (flat) KV cache: 3D [total_slots, n_head_kv, head_dim] ----
  util.func private @kvcache_components.allocate(%arg0: index, %arg1: index, %arg2: index) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f16
    %5 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %7 = hal.tensor.export %6 : tensor<?x?x?xf16>{%arg0, %arg1, %arg2} -> !hal.buffer_view
    %cst_v = arith.constant 0.000000e+00 : f16
    %8 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?xf16>
    %9 = linalg.fill ins(%cst_v : f16) outs(%8 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %10 = hal.tensor.export %9 : tensor<?x?x?xf16>{%arg0, %arg1, %arg2} -> !hal.buffer_view
    %11 = util.list.create %c2 : !util.list<?>
    util.list.resize %11, %c2 : !util.list<?>
    util.list.set %11[%c0], %7 : !hal.buffer_view -> !util.list<?>
    util.list.set %11[%c1], %10 : !hal.buffer_view -> !util.list<?>
    util.return %11 : !util.list<?>
  }
  // ---- Direct KV cache read: extract_slice for a layer's context ----
  util.func private @kvcache_direct_read(%cache: !util.list<?>, %slot_start: index, %ctx_len: index) -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %total = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %nkv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %hdim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %ks = tensor.extract_slice %k_full[%slot_start, 0, 0] [%ctx_len, %nkv, %hdim] [1, 1, 1] : tensor<?x?x?xf16> to tensor<?x?x?xf16>
    %vs = tensor.extract_slice %v_full[%slot_start, 0, 0] [%ctx_len, %nkv, %hdim] [1, 1, 1] : tensor<?x?x?xf16> to tensor<?x?x?xf16>
    %ko = tensor.expand_shape %ks [[0, 1], [2], [3]] output_shape [%c1, %ctx_len, %nkv, %hdim] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %vo = tensor.expand_shape %vs [[0, 1], [2], [3]] output_shape [%c1, %ctx_len, %nkv, %hdim] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    util.return %ko, %vo : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
  }
  // ---- Direct KV cache write: insert_slice for a single token ----
  util.func private @kvcache_direct_write(%cache: !util.list<?>, %slot: index, %new_k: tensor<?x?xf16>, %new_v: tensor<?x?xf16>) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %total = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %nkv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %hdim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %ku = tensor.insert_slice %new_k into %k_full[%slot, 0, 0] [1, %nkv, %hdim] [1, 1, 1] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %vu = tensor.insert_slice %new_v into %v_full[%slot, 0, 0] [1, %nkv, %hdim] [1, 1, 1] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %kbv = hal.tensor.export %ku : tensor<?x?x?xf16>{%total, %nkv, %hdim} -> !hal.buffer_view
    %vbv = hal.tensor.export %vu : tensor<?x?x?xf16>{%total, %nkv, %hdim} -> !hal.buffer_view
    util.list.set %cache[%c0], %kbv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %vbv : !hal.buffer_view -> !util.list<?>
    util.return %cache : !util.list<?>
  }
  // ---- Direct KV cache scatter for prefill: insert_slice for a batch of tokens ----
  util.func private @kvcache_direct_scatter_prefill(%cache: !util.list<?>, %layer_idx: index, %new_k: tensor<?x?x?xf16>, %new_v: tensor<?x?x?xf16>, %max_seq_len_val: index, %start_pos: index) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %seq_len = tensor.dim %new_k, %c0 : tensor<?x?x?xf16>
    %slot_start_base = arith.muli %layer_idx, %max_seq_len_val : index
    %slot_start = arith.addi %slot_start_base, %start_pos : index
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %total = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %nkv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %hdim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %ku = tensor.insert_slice %new_k into %k_full[%slot_start, 0, 0] [%seq_len, %nkv, %hdim] [1, 1, 1] : tensor<?x?x?xf16> into tensor<?x?x?xf16>
    %vu = tensor.insert_slice %new_v into %v_full[%slot_start, 0, 0] [%seq_len, %nkv, %hdim] [1, 1, 1] : tensor<?x?x?xf16> into tensor<?x?x?xf16>
    %kbv = hal.tensor.export %ku : tensor<?x?x?xf16>{%total, %nkv, %hdim} -> !hal.buffer_view
    %vbv = hal.tensor.export %vu : tensor<?x?x?xf16>{%total, %nkv, %hdim} -> !hal.buffer_view
    util.list.set %cache[%c0], %kbv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %vbv : !hal.buffer_view -> !util.list<?>
    util.return %cache : !util.list<?>
  }
  // ---- Transformer layer decode with direct cache ----
  // Removed: %block_tables (arg3), %context_lens (arg4), %ctx_len_scalar (arg5)
  // Added:   %max_seq_len (arg3), %ctx_len (arg4)
  util.func private @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi64>, %arg2: !util.list<?>, %max_seq_len_arg: index, %ctx_len: index, %arg6: i32, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: f32, %arg12: f32, %arg13: f32) -> (tensor<?x?xf16>, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = util.call @model_params.attn_norm_weight(%arg6) : (i32) -> tensor<?xf16>
    %2 = util.call @model_params.ffn_norm_weight(%arg6) : (i32) -> tensor<?xf16>
    %3 = util.call @model_params.attn_q_weight(%arg6) : (i32) -> tensor<?x?xf16>
    %4 = util.call @model_params.attn_k_weight(%arg6) : (i32) -> tensor<?x?xf16>
    %5 = util.call @model_params.attn_v_weight(%arg6) : (i32) -> tensor<?x?xf16>
    %6 = util.call @model_params.attn_output_weight(%arg6) : (i32) -> tensor<?x?xf16>
    %7 = util.call @model_params.attn_q_norm_weight(%arg6) : (i32) -> tensor<?xf16>
    %8 = util.call @model_params.attn_k_norm_weight(%arg6) : (i32) -> tensor<?xf16>
    %9 = util.call @model_params.ffn_gate_up_weight(%arg6) : (i32) -> tensor<?x?xf16>
    %10 = util.call @model_params.ffn_down_weight(%arg6) : (i32) -> tensor<?x?xf16>
    %11 = util.call @rms_norm_components.rms_norm_linalg(%arg0, %1, %arg11) : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
    // Direct cache read: slot_start = layer_idx * max_seq_len
    %layer_idx = arith.index_cast %arg6 : i32 to index
    %slot_start = arith.muli %layer_idx, %max_seq_len_arg : index
    %12:2 = util.call @kvcache_direct_read(%arg2, %slot_start, %ctx_len) : (!util.list<?>, index, index) -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)
    %13:3 = util.call @attention_block_decode_qwen_components.attention_block_decode_qwen(%11, %arg1, %12#0, %12#1, %3, %4, %5, %6, %arg7, %arg8, %arg9, %arg12, %arg13, %7, %8, %arg11) : (tensor<?x?xf16>, tensor<?xi64>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, index, index, index, f32, f32, tensor<?xf16>, tensor<?xf16>, f32) -> (tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)
    // Direct cache write: slot = slot_start + pos
    %pos_i64 = tensor.extract %arg1[%c0] : tensor<?xi64>
    %pos_idx = arith.index_cast %pos_i64 : i64 to index
    %write_slot = arith.addi %slot_start, %pos_idx : index
    %new_k_2d = tensor.collapse_shape %13#1 [[0, 1], [2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
    %new_v_2d = tensor.collapse_shape %13#2 [[0, 1], [2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
    %14 = util.call @kvcache_direct_write(%arg2, %write_slot, %new_k_2d, %new_v_2d) : (!util.list<?>, index, tensor<?x?xf16>, tensor<?x?xf16>) -> !util.list<?>
    %15 = tensor.empty(%dim, %arg9) : tensor<?x?xf16>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %13#0 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%15 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %29 = arith.addf %in, %in_1 : f16
      linalg.yield %29 : f16
    } -> tensor<?x?xf16>
    %17 = util.call @rms_norm_components.rms_norm_linalg(%16, %2, %arg11) : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %18 = arith.muli %arg10, %c2 : index
    %19 = tensor.empty(%dim, %18) : tensor<?x?xf16>
    %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %21 = linalg.matmul ins(%17, %9 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%20 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %extracted_slice = tensor.extract_slice %21[0, 0] [%dim, %arg10] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
    %extracted_slice_0 = tensor.extract_slice %21[0, %arg10] [%dim, %arg10] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
    %22 = tensor.empty(%dim, %arg10) : tensor<?x?xf16>
    %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%22 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %29 = arith.negf %in : f16
      %30 = math.exp %29 : f16
      %cst_2 = arith.constant 1.000000e+00 : f16
      %31 = arith.addf %cst_2, %30 : f16
      %32 = arith.divf %cst_2, %31 : f16
      %33 = arith.mulf %in, %32 : f16
      %34 = arith.mulf %33, %in_1 : f16
      linalg.yield %34 : f16
    } -> tensor<?x?xf16>
    %24 = tensor.empty(%dim, %arg9) : tensor<?x?xf16>
    %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %26 = linalg.matmul ins(%23, %10 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %27 = tensor.empty(%dim, %arg9) : tensor<?x?xf16>
    %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%16, %26 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%27 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %29 = arith.addf %in, %in_1 : f16
      linalg.yield %29 : f16
    } -> tensor<?x?xf16>
    util.return %28, %14 : tensor<?x?xf16>, !util.list<?>
  }
  util.func private @attention_block_decode_qwen_components.attention_block_decode_qwen(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi64>, %arg2: tensor<?x?x?x?xf16>, %arg3: tensor<?x?x?x?xf16>, %arg4: tensor<?x?xf16>, %arg5: tensor<?x?xf16>, %arg6: tensor<?x?xf16>, %arg7: tensor<?x?xf16>, %arg8: index, %arg9: index, %arg10: index, %arg11: f32, %arg12: f32, %arg13: tensor<?xf16>, %arg14: tensor<?xf16>, %arg15: f32) -> (tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg2, %c1 : tensor<?x?x?x?xf16>
    %dim_1 = tensor.dim %arg4, %c1 : tensor<?x?xf16>
    %0 = arith.divui %dim_1, %arg8 : index
    %1 = arith.muli %arg9, %0 : index
    %c1_2 = arith.constant 1 : index
    // dyn_1: opaque "1" to prevent O1 from inserting illegal tensor.cast
    // on the concat path (tensor<?x1x?x?xf16> -> tensor<?x?x?x?xf16>).
    %dyn_1 = util.optimization_barrier %c1_2 : index
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [%dim, %c1_2, %arg10] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_3 = tensor.expand_shape %arg1 [[0, 1]] output_shape [%dim, %c1_2] : tensor<?xi64> into tensor<?x?xi64>
    %cst = arith.constant 0.000000e+00 : f16
    %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %4 = linalg.matmul ins(%arg0, %arg4 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %5 = tensor.empty(%dim, %1) : tensor<?x?xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %7 = linalg.matmul ins(%arg0, %arg5 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%6 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %8 = tensor.empty(%dim, %1) : tensor<?x?xf16>
    %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %10 = linalg.matmul ins(%arg0, %arg6 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%9 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %expanded_4 = tensor.expand_shape %4 [[0], [1, 2]] output_shape [%dim, %arg8, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_5 = tensor.expand_shape %7 [[0], [1, 2]] output_shape [%dim, %arg9, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_6 = tensor.expand_shape %10 [[0], [1, 2]] output_shape [%dim, %arg9, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %collapsed = tensor.collapse_shape %expanded_4 [[0, 1], [2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
    %collapsed_7 = tensor.collapse_shape %expanded_5 [[0, 1], [2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
    // Inline fused QK norm via flow.dispatch.region (forces reduction+elementwise into 1 dispatch)
    %q_bh = arith.muli %dim, %arg8 : index
    %11 = flow.dispatch.region[] -> (tensor<?x?xf16>{%q_bh, %0}) {
      %_cst0 = arith.constant 0.000000e+00 : f32
      %_hdi = arith.index_cast %0 : index to i32
      %_hdf = arith.sitofp %_hdi : i32 to f32
      %_si = tensor.empty(%q_bh) : tensor<?xf32>
      %_sz = linalg.fill ins(%_cst0 : f32) outs(%_si : tensor<?xf32>) -> tensor<?xf32>
      %_ss = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%collapsed : tensor<?x?xf16>) outs(%_sz : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<?xf32>
      %_oi = tensor.empty(%q_bh, %0) : tensor<?x?xf16>
      %_on = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %_ss, %arg13 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%_oi : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf : f32
        %b = arith.addf %a, %arg15 : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<?x?xf16>
      flow.return %_on : tensor<?x?xf16>
    }
    %k_bh = arith.muli %dim, %arg9 : index
    %12 = flow.dispatch.region[] -> (tensor<?x?xf16>{%k_bh, %0}) {
      %_cst0 = arith.constant 0.000000e+00 : f32
      %_hdi = arith.index_cast %0 : index to i32
      %_hdf = arith.sitofp %_hdi : i32 to f32
      %_si = tensor.empty(%k_bh) : tensor<?xf32>
      %_sz = linalg.fill ins(%_cst0 : f32) outs(%_si : tensor<?xf32>) -> tensor<?xf32>
      %_ss = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%collapsed_7 : tensor<?x?xf16>) outs(%_sz : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<?xf32>
      %_oi = tensor.empty(%k_bh, %0) : tensor<?x?xf16>
      %_on = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_7, %_ss, %arg14 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%_oi : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf : f32
        %b = arith.addf %a, %arg15 : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<?x?xf16>
      flow.return %_on : tensor<?x?xf16>
    }
    %expanded_8 = tensor.expand_shape %11 [[0, 1], [2]] output_shape [%dim, %arg8, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_9 = tensor.expand_shape %expanded_8 [[0, 1], [2], [3]] output_shape [%dim, %dyn_1, %arg8, %0] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %expanded_10 = tensor.expand_shape %12 [[0, 1], [2]] output_shape [%dim, %arg9, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_11 = tensor.expand_shape %expanded_10 [[0, 1], [2], [3]] output_shape [%dim, %dyn_1, %arg9, %0] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %cast = tensor.cast %expanded_6 : tensor<?x?x?xf16> to tensor<?x?x?xf16>
    %13 = util.call @position_components.rope(%expanded_9, %expanded_3, %arg11, %arg12) : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>
    %14 = util.call @position_components.rope(%expanded_11, %expanded_3, %arg11, %arg12) : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>
    %15 = arith.addi %dim_0, %c1 : index
    // Replace tensor.concat with insert_slice to avoid O1 tensor.cast legality issue.
    // Read actual dims from the cache tensors to avoid head-count mismatches.
    %k_dim2 = tensor.dim %arg2, %c2 : tensor<?x?x?x?xf16>
    %k_dim3 = tensor.dim %arg2, %c3 : tensor<?x?x?x?xf16>
    %v_dim2 = tensor.dim %arg3, %c2 : tensor<?x?x?x?xf16>
    %v_dim3 = tensor.dim %arg3, %c3 : tensor<?x?x?x?xf16>
    // concat K: [batch, ctx, kv_heads, head_dim] ++ [batch, 1, kv_heads, head_dim]
    %k_out = tensor.empty(%dim, %15, %k_dim2, %k_dim3) : tensor<?x?x?x?xf16>
    %k_ins0 = tensor.insert_slice %arg2 into %k_out[0, 0, 0, 0] [%dim, %dim_0, %k_dim2, %k_dim3] [1, 1, 1, 1] : tensor<?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %concat = tensor.insert_slice %14 into %k_ins0[0, %dim_0, 0, 0] [%dim, %dyn_1, %k_dim2, %k_dim3] [1, 1, 1, 1] : tensor<?x?x?x?xf16> into tensor<?x?x?x?xf16>
    // concat V: same pattern
    %expanded_12 = tensor.expand_shape %cast [[0, 1], [2], [3]] output_shape [%dim, %dyn_1, %v_dim2, %v_dim3] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %v_out = tensor.empty(%dim, %15, %v_dim2, %v_dim3) : tensor<?x?x?x?xf16>
    %v_ins0 = tensor.insert_slice %arg3 into %v_out[0, 0, 0, 0] [%dim, %dim_0, %v_dim2, %v_dim3] [1, 1, 1, 1] : tensor<?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %concat_13 = tensor.insert_slice %expanded_12 into %v_ins0[0, %dim_0, 0, 0] [%dim, %dyn_1, %v_dim2, %v_dim3] [1, 1, 1, 1] : tensor<?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %16 = arith.index_cast %0 : index to i32
    %17 = arith.sitofp %16 : i32 to f32
    %18 = math.rsqrt %17 : f32
    %19 = util.call @attention_components.attention_gqa(%13, %concat, %concat_13, %18) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f32) -> tensor<?x?x?x?xf16>
    %collapsed_14 = tensor.collapse_shape %19 [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %collapsed_15 = tensor.collapse_shape %collapsed_14 [[0], [1, 2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
    %20 = tensor.empty(%dim, %arg10) : tensor<?x?xf16>
    %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %22 = linalg.matmul ins(%collapsed_15, %arg7 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%21 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %collapsed_16 = tensor.collapse_shape %14 [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %collapsed_17 = tensor.collapse_shape %expanded_12 [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    util.return %22, %collapsed_16, %collapsed_17 : tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>
  }
  util.func private @attention_components.attention_gqa(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?x?x?xf16>, %arg2: tensor<?x?x?x?xf16>, %arg3: f32) -> tensor<?x?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf16>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf16>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf16>
    %dim_3 = tensor.dim %arg1, %c2 : tensor<?x?x?x?xf16>
    %dim_4 = tensor.dim %arg1, %c1 : tensor<?x?x?x?xf16>
    %0 = tensor.empty(%dim, %dim_1, %dim_0, %dim_2) : tensor<?x?x?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?x?xf16>) outs(%0 : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    %2 = tensor.empty(%dim, %dim_3, %dim_4, %dim_2) : tensor<?x?x?x?xf16>
    %3 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<?x?x?x?xf16>) outs(%2 : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    %4 = tensor.empty(%dim, %dim_3, %dim_4, %dim_2) : tensor<?x?x?x?xf16>
    %5 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<?x?x?x?xf16>) outs(%4 : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    %6 = arith.divui %dim_1, %dim_3 : index
    %7 = tensor.empty(%dim, %dim_3, %6, %dim_4, %dim_2) : tensor<?x?x?x?x?xf16>
    %8 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<?x?x?x?xf16>) outs(%7 : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %collapsed = tensor.collapse_shape %8 [[0], [1, 2], [3], [4]] : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %9 = tensor.empty(%dim, %dim_3, %6, %dim_4, %dim_2) : tensor<?x?x?x?x?xf16>
    %10 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<?x?x?x?xf16>) outs(%9 : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %collapsed_5 = tensor.collapse_shape %10 [[0], [1, 2], [3], [4]] : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %11 = arith.muli %dim, %dim_1 : index
    %collapsed_6 = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %collapsed_7 = tensor.collapse_shape %collapsed [[0, 1], [2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %collapsed_8 = tensor.collapse_shape %collapsed_5 [[0, 1], [2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %12 = arith.subi %dim_4, %dim_0 : index
    %13 = tensor.empty(%11, %dim_0, %dim_4) : tensor<?x?x?xi1>
    %14 = linalg.generic {indexing_maps = [#map8], iterator_types = ["parallel", "parallel", "parallel"]} outs(%13 : tensor<?x?x?xi1>) {
    ^bb0(%out: i1):
      %19 = linalg.index 1 : index
      %20 = linalg.index 2 : index
      %21 = arith.addi %19, %12 : index
      %22 = arith.cmpi ule, %20, %21 : index
      linalg.yield %22 : i1
    } -> tensor<?x?x?xi1>
    %15 = tensor.empty(%11, %dim_0, %dim_2) : tensor<?x?x?xf16>
    %16 = iree_linalg_ext.attention {indexing_maps = [#map9, #map10, #map11, #map12, #map13, #map14]} ins(%collapsed_6, %collapsed_7, %collapsed_8, %arg3, %14 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f32, tensor<?x?x?xi1>) outs(%15 : tensor<?x?x?xf16>) {
    ^bb0(%arg4: f32):
      iree_linalg_ext.yield %arg4 : f32
    } -> tensor<?x?x?xf16>
    %expanded = tensor.expand_shape %16 [[0, 1], [2], [3]] output_shape [%dim, %dim_1, %dim_0, %dim_2] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %17 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf16>
    %18 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<?x?x?x?xf16>) outs(%17 : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    util.return %18 : tensor<?x?x?x?xf16>
  }
  util.func private @position_components.rope(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?xi64>, %arg2: f32, %arg3: f32) -> tensor<?x?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf16>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf16>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf16>
    %0 = arith.divsi %dim_2, %c2 : index
    %1 = arith.index_cast %dim_2 : index to i32
    %2 = arith.sitofp %1 : i32 to f32
    %3 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf16>
    %4 = linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<?x?x?x?xf16>) {
    ^bb0(%out: f16):
      %5 = linalg.index 0 : index
      %6 = linalg.index 1 : index
      %7 = linalg.index 2 : index
      %8 = linalg.index 3 : index
      %extracted = tensor.extract %arg1[%5, %6] : tensor<?x?xi64>
      %9 = arith.trunci %extracted : i64 to i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.cmpi slt, %8, %0 : index
      %12 = scf.if %11 -> (f16) {
        %13 = arith.index_cast %8 : index to i32
        %14 = arith.sitofp %13 : i32 to f32
        %cst = arith.constant 2.000000e+00 : f32
        %15 = arith.mulf %cst, %14 : f32
        %16 = arith.divf %15, %2 : f32
        %17 = arith.negf %16 : f32
        %18 = math.powf %arg2, %17 : f32
        %19 = arith.mulf %18, %arg3 : f32
        %20 = arith.mulf %10, %19 : f32
        %21 = math.cos %20 : f32
        %22 = math.sin %20 : f32
        %extracted_3 = tensor.extract %arg0[%5, %6, %7, %8] : tensor<?x?x?x?xf16>
        %23 = arith.addi %8, %0 : index
        %extracted_4 = tensor.extract %arg0[%5, %6, %7, %23] : tensor<?x?x?x?xf16>
        %24 = arith.extf %extracted_3 : f16 to f32
        %25 = arith.extf %extracted_4 : f16 to f32
        %26 = arith.mulf %24, %21 : f32
        %27 = arith.mulf %25, %22 : f32
        %28 = arith.subf %26, %27 : f32
        %29 = arith.truncf %28 : f32 to f16
        scf.yield %29 : f16
      } else {
        %13 = arith.subi %8, %0 : index
        %14 = arith.index_cast %13 : index to i32
        %15 = arith.sitofp %14 : i32 to f32
        %cst = arith.constant 2.000000e+00 : f32
        %16 = arith.mulf %cst, %15 : f32
        %17 = arith.divf %16, %2 : f32
        %18 = arith.negf %17 : f32
        %19 = math.powf %arg2, %18 : f32
        %20 = arith.mulf %19, %arg3 : f32
        %21 = arith.mulf %10, %20 : f32
        %22 = math.cos %21 : f32
        %23 = math.sin %21 : f32
        %extracted_3 = tensor.extract %arg0[%5, %6, %7, %8] : tensor<?x?x?x?xf16>
        %extracted_4 = tensor.extract %arg0[%5, %6, %7, %13] : tensor<?x?x?x?xf16>
        %24 = arith.extf %extracted_4 : f16 to f32
        %25 = arith.extf %extracted_3 : f16 to f32
        %26 = arith.mulf %24, %23 : f32
        %27 = arith.mulf %25, %22 : f32
        %28 = arith.addf %26, %27 : f32
        %29 = arith.truncf %28 : f32 to f16
        scf.yield %29 : f16
      }
      linalg.yield %12 : f16
    } -> tensor<?x?x?x?xf16>
    util.return %4 : tensor<?x?x?x?xf16>
  }
  util.func private @model_params.ffn_down_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_down.weight"> : tensor<28x3145728xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 3145728] [1, 1] : tensor<28x3145728xf16> to tensor<1x3145728xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x3145728xf16> into tensor<3145728xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [3072, 1024] : tensor<3145728xf16> into tensor<3072x1024xf16>
    %cast = tensor.cast %expanded : tensor<3072x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.ffn_gate_up_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_gate_up.weight"> : tensor<28x6291456xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 6291456] [1, 1] : tensor<28x6291456xf16> to tensor<1x6291456xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x6291456xf16> into tensor<6291456xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1024, 6144] : tensor<6291456xf16> into tensor<1024x6144xf16>
    %cast = tensor.cast %expanded : tensor<1024x6144xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.attn_k_norm_weight(%arg0: i32) -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k_norm.weight"> : tensor<28x128xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 128] [1, 1] : tensor<28x128xf16> to tensor<1x128xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x128xf16> into tensor<128xf16>
    %cast = tensor.cast %collapsed : tensor<128xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  util.func private @model_params.attn_q_norm_weight(%arg0: i32) -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q_norm.weight"> : tensor<28x128xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 128] [1, 1] : tensor<28x128xf16> to tensor<1x128xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x128xf16> into tensor<128xf16>
    %cast = tensor.cast %collapsed : tensor<128xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  util.func private @model_params.attn_output_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_output.weight"> : tensor<28x2097152xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 2097152] [1, 1] : tensor<28x2097152xf16> to tensor<1x2097152xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x2097152xf16> into tensor<2097152xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [2048, 1024] : tensor<2097152xf16> into tensor<2048x1024xf16>
    %cast = tensor.cast %expanded : tensor<2048x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.attn_v_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_v.weight"> : tensor<28x1048576xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 1048576] [1, 1] : tensor<28x1048576xf16> to tensor<1x1048576xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1048576xf16> into tensor<1048576xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1024, 1024] : tensor<1048576xf16> into tensor<1024x1024xf16>
    %cast = tensor.cast %expanded : tensor<1024x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.attn_k_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k.weight"> : tensor<28x1048576xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 1048576] [1, 1] : tensor<28x1048576xf16> to tensor<1x1048576xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1048576xf16> into tensor<1048576xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1024, 1024] : tensor<1048576xf16> into tensor<1024x1024xf16>
    %cast = tensor.cast %expanded : tensor<1024x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.attn_q_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q.weight"> : tensor<28x2097152xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 2097152] [1, 1] : tensor<28x2097152xf16> to tensor<1x2097152xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x2097152xf16> into tensor<2097152xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1024, 2048] : tensor<2097152xf16> into tensor<1024x2048xf16>
    %cast = tensor.cast %expanded : tensor<1024x2048xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.ffn_norm_weight(%arg0: i32) -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_norm.weight"> : tensor<28x1024xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 1024] [1, 1] : tensor<28x1024xf16> to tensor<1x1024xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1024xf16> into tensor<1024xf16>
    %cast = tensor.cast %collapsed : tensor<1024xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  util.func private @model_params.attn_norm_weight(%arg0: i32) -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_norm.weight"> : tensor<28x1024xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 1024] [1, 1] : tensor<28x1024xf16> to tensor<1x1024xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1024xf16> into tensor<1024xf16>
    %cast = tensor.cast %collapsed : tensor<1024xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  // ---- allocate_kv_cache: direct flat cache ----
  util.func public @allocate_kv_cache(%max_seq_len_val: index) -> !util.list<?> {
    %0 = util.call @hparams.attention_head_count_kv() : () -> i64
    %1 = util.call @hparams.block_count() : () -> i64
    %3 = arith.index_cast %0 : i64 to index
    %n_layers = arith.index_cast %1 : i64 to index
    %6 = util.call @hparams.head_dim() : () -> i64
    %7 = arith.index_cast %6 : i64 to index
    %total_slots = arith.muli %n_layers, %max_seq_len_val : index
    %8 = util.call @kvcache_components.allocate(%total_slots, %3, %7) : (index, index, index) -> !util.list<?>
    util.return %8 : !util.list<?>
  }
  // ---- decode: direct cache (no block_tables, no context_lens) ----
  util.func public @decode(%arg0: tensor<?xi64>, %arg1: tensor<?xi64>, %arg2: !util.list<?>, %max_seq_len_val: index, %ctx_len: index) -> (tensor<?x?xf16>, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = util.call @hparams.vocab_size() : () -> i64
    %1 = util.call @hparams.block_count() : () -> i64
    %2 = util.call @hparams.embedding_length() : () -> i64
    %3 = util.call @hparams.attention_head_count() : () -> i64
    %4 = util.call @hparams.attention_head_count_kv() : () -> i64
    %5 = util.call @hparams.feed_forward_length() : () -> i64
    %6 = util.call @hparams.rope_freq_base() : () -> f32
    %7 = util.call @hparams.layer_norm_rms_epsilon() : () -> f32
    %8 = arith.index_cast %0 : i64 to index
    %9 = arith.index_cast %1 : i64 to index
    %10 = arith.index_cast %2 : i64 to index
    %11 = arith.index_cast %3 : i64 to index
    %12 = arith.index_cast %4 : i64 to index
    %13 = arith.index_cast %5 : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?xi64>
    %14 = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %15 = util.call @embedding_components.embedding_lookup_1d(%14, %arg0) : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>
    %cst = arith.constant 1.000000e+00 : f32
    %16:2 = scf.for %arg6 = %c0 to %9 step %c1 iter_args(%arg7 = %15, %arg8 = %arg2) -> (tensor<?x?xf16>, !util.list<?>) {
      %23 = arith.index_cast %arg6 : index to i32
      %24:2 = util.call @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(%arg7, %arg1, %arg8, %max_seq_len_val, %ctx_len, %23, %11, %12, %10, %13, %7, %6, %cst) : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>, index, index, i32, index, index, index, index, f32, f32, f32) -> (tensor<?x?xf16>, !util.list<?>)
      scf.yield %24#0, %24#1 : tensor<?x?xf16>, !util.list<?>
    }
    %17 = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %18 = util.call @rms_norm_components.rms_norm_linalg(%16#0, %17, %7) : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
    %19 = util.call @model_params.output_weight() : () -> tensor<?x?xf16>
    %20 = tensor.empty(%dim, %8) : tensor<?x?xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %21 = linalg.fill ins(%cst_0 : f16) outs(%20 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %22 = linalg.matmul ins(%18, %19 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%21 : tensor<?x?xf16>) -> tensor<?x?xf16>
    util.return %22, %16#1 : tensor<?x?xf16>, !util.list<?>
  }
  // ---- generate: direct cache ----
  util.func public @generate(%arg0: i64, %arg1: !util.list<?>, %max_seq_len_val: index, %arg3: index, %arg4: i64, %arg5: i64) -> (tensor<?xi64>, index, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f16
    %0 = util.call @hparams.vocab_size() : () -> i64
    %1 = util.call @hparams.block_count() : () -> i64
    %2 = util.call @hparams.embedding_length() : () -> i64
    %3 = util.call @hparams.attention_head_count() : () -> i64
    %4 = util.call @hparams.attention_head_count_kv() : () -> i64
    %5 = util.call @hparams.feed_forward_length() : () -> i64
    %6 = util.call @hparams.rope_freq_base() : () -> f32
    %7 = util.call @hparams.layer_norm_rms_epsilon() : () -> f32
    %8 = arith.index_cast %0 : i64 to index
    %9 = arith.index_cast %1 : i64 to index
    %10 = arith.index_cast %2 : i64 to index
    %11 = arith.index_cast %3 : i64 to index
    %12 = arith.index_cast %4 : i64 to index
    %13 = arith.index_cast %5 : i64 to index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %14 = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %15 = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %16 = util.call @model_params.output_weight() : () -> tensor<?x?xf16>
    %17 = tensor.empty(%arg3) : tensor<?xi64>
    %18 = linalg.fill ins(%c0_i64 : i64) outs(%17 : tensor<?xi64>) -> tensor<?xi64>
    %c1_1 = arith.constant 1 : index
    %19 = arith.index_cast %9 : index to i32
    %20:6 = scf.while (%arg6 = %arg0, %arg7 = %arg5, %arg8 = %arg1, %arg9 = %18, %arg10 = %c0, %arg11 = %true) : (i64, i64, !util.list<?>, tensor<?xi64>, index, i1) -> (i64, i64, !util.list<?>, tensor<?xi64>, index, i1) {
      %21 = arith.cmpi ult, %arg10, %arg3 : index
      %22 = arith.andi %arg11, %21 : i1
      scf.condition(%22) %arg6, %arg7, %arg8, %arg9, %arg10, %arg11 : i64, i64, !util.list<?>, tensor<?xi64>, index, i1
    } do {
    ^bb0(%arg6: i64, %arg7: i64, %arg8: !util.list<?>, %arg9: tensor<?xi64>, %arg10: index, %arg11: i1):
      %from_elements = tensor.from_elements %arg6 : tensor<1xi64>
      %cast = tensor.cast %from_elements : tensor<1xi64> to tensor<?xi64>
      %from_elements_2 = tensor.from_elements %arg7 : tensor<1xi64>
      %cast_3 = tensor.cast %from_elements_2 : tensor<1xi64> to tensor<?xi64>
      // Compute ctx_len from current absolute position (arg7 is the absolute position)
      %ctx_len_val = arith.index_cast %arg7 : i64 to index
      %25 = util.call @embedding_components.embedding_lookup_1d(%14, %cast) : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>
      %26:2 = scf.for %arg12 = %c0 to %9 step %c1 iter_args(%arg13 = %25, %arg14 = %arg8) -> (tensor<?x?xf16>, !util.list<?>) {
        %40 = arith.index_cast %arg12 : index to i32
        %41:2 = util.call @transformer_layer_qwen_decode_components.transformer_layer_qwen_decode(%arg13, %cast_3, %arg14, %max_seq_len_val, %ctx_len_val, %40, %11, %12, %10, %13, %7, %6, %cst_0) : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>, index, index, i32, index, index, index, index, f32, f32, f32) -> (tensor<?x?xf16>, !util.list<?>)
        scf.yield %41#0, %41#1 : tensor<?x?xf16>, !util.list<?>
      }
      %27 = util.call @rms_norm_components.rms_norm_linalg(%26#0, %15, %7) : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>
      %28 = tensor.empty(%c1_1, %8) : tensor<?x?xf16>
      %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %30 = linalg.matmul ins(%27, %16 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%29 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %cst_4 = arith.constant 0xFC00 : f16
      %c-1_i64 = arith.constant -1 : i64
      %31 = tensor.empty() : tensor<f16>
      %32 = tensor.empty() : tensor<i64>
      %33 = linalg.fill ins(%cst_4 : f16) outs(%31 : tensor<f16>) -> tensor<f16>
      %34 = linalg.fill ins(%c-1_i64 : i64) outs(%32 : tensor<i64>) -> tensor<i64>
      %extracted_slice = tensor.extract_slice %30[0, 0] [1, %8] [1, 1] : tensor<?x?xf16> to tensor<?xf16>
      %35:2 = linalg.generic {indexing_maps = [#map2, #map16, #map16], iterator_types = ["reduction"]} ins(%extracted_slice : tensor<?xf16>) outs(%33, %34 : tensor<f16>, tensor<i64>) {
      ^bb0(%in: f16, %out: f16, %out_5: i64):
        %40 = linalg.index 0 : index
        %41 = arith.index_cast %40 : index to i64
        %42 = arith.cmpf ogt, %in, %out : f16
        %43 = arith.select %42, %in, %out : f16
        %44 = arith.select %42, %41, %out_5 : i64
        linalg.yield %43, %44 : f16, i64
      } -> (tensor<f16>, tensor<i64>)
      %extracted = tensor.extract %35#1[] : tensor<i64>
      %inserted = tensor.insert %extracted into %arg9[%arg10] : tensor<?xi64>
      %36 = arith.cmpi eq, %extracted, %arg4 : i64
      %37 = arith.xori %36, %true : i1
      %38 = arith.addi %arg7, %c1_i64 : i64
      %39 = arith.addi %arg10, %c1 : index
      scf.yield %extracted, %38, %26#1, %inserted, %39, %37 : i64, i64, !util.list<?>, tensor<?xi64>, index, i1
    }
    util.return %20#3, %20#4, %20#2 : tensor<?xi64>, index, !util.list<?>
  }
  // Prefill all tokens via scf.for calling @decode for each.
  // This is a SEPARATE function (not inlined into @run) to prevent
  // canonicalization from seeing the list.set in @allocate_kv_cache
  // and the scf.for in the same scope (which triggers dead-store removal).
  // Matches OLMo's @prefill_all structure exactly: carry logits through
  // the loop as an iter_arg, no double-decode of the last token.
  util.func public @prefill_all(%arg0: tensor<?xi64>, %arg1: index, %cache: !util.list<?>, %max_seq_len_val: index) -> (tensor<?x?xf16>, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %bc = util.call @hparams.block_count() : () -> i64
    %n_layers = arith.index_cast %bc : i64 to index
    %emb_i64 = util.call @hparams.embedding_length() : () -> i64
    %emb_dim = arith.index_cast %emb_i64 : i64 to index
    // Initial dummy logits (overwritten on first iteration)
    %vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %vocab = arith.index_cast %vocab_i64 : i64 to index
    %logits_init = tensor.empty(%c1, %vocab) : tensor<?x?xf16>
    %logits_zero = linalg.fill ins(%cst : f16) outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %result:2 = scf.for %pi = %c0 to %arg1 step %c1
        iter_args(%pc = %cache, %prev_logits = %logits_zero) -> (!util.list<?>, tensor<?x?xf16>) {
      %tok_val = tensor.extract %arg0[%pi] : tensor<?xi64>
      %tok_t_e = tensor.empty() : tensor<1xi64>
      %tok_t = tensor.insert %tok_val into %tok_t_e[%c0] : tensor<1xi64>
      %tok = tensor.cast %tok_t : tensor<1xi64> to tensor<?xi64>
      %pos_val = arith.index_cast %pi : index to i64
      %pos_t_e = tensor.empty() : tensor<1xi64>
      %pos_t = tensor.insert %pos_val into %pos_t_e[%c0] : tensor<1xi64>
      %pos = tensor.cast %pos_t : tensor<1xi64> to tensor<?xi64>
      // ctx_len for this position = pi (number of previous tokens already in cache)
      // At position 0, ctx_len=0 (no prior context), at position 1, ctx_len=1, etc.
      %dec:2 = util.call @decode(%tok, %pos, %pc, %max_seq_len_val, %pi) : (tensor<?xi64>, tensor<?xi64>, !util.list<?>, index, index) -> (tensor<?x?xf16>, !util.list<?>)
      scf.yield %dec#1, %dec#0 : !util.list<?>, tensor<?x?xf16>
    }
    util.return %result#1, %result#0 : tensor<?x?xf16>, !util.list<?>
  }
  util.func private @attention_prefill(
      %arg0: tensor<?x?x?x?xf16>,  // Q [batch, n_heads, seq_len, head_dim]
      %arg1: tensor<?x?x?x?xf16>,  // K [batch, n_kv_heads, seq_len, head_dim]
      %arg2: tensor<?x?x?x?xf16>,  // V [batch, n_kv_heads, seq_len, head_dim]
      %arg3: f32                     // scale = 1/sqrt(head_dim)
  ) -> tensor<?x?x?x?xf16> {       // [batch, n_heads, seq_len, head_dim]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Q dims: [batch, n_heads, seq_len, head_dim]
    %batch = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf16>
    %n_heads = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf16>
    %seq_len_q = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf16>
    %head_dim = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf16>

    // K dims
    %n_kv_heads = tensor.dim %arg1, %c1 : tensor<?x?x?x?xf16>
    %seq_len_k = tensor.dim %arg1, %c2 : tensor<?x?x?x?xf16>

    // Transpose Q from [batch, heads, seq, dim] to [batch, seq, heads, dim]
    %q_t_init = tensor.empty(%batch, %seq_len_q, %n_heads, %head_dim) : tensor<?x?x?x?xf16>
    %q_transposed = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<?x?x?x?xf16>) outs(%q_t_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // Transpose K from [batch, kv_heads, seq, dim] to [batch, seq, kv_heads, dim]
    %k_t_init = tensor.empty(%batch, %seq_len_k, %n_kv_heads, %head_dim) : tensor<?x?x?x?xf16>
    %k_transposed = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg1 : tensor<?x?x?x?xf16>) outs(%k_t_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // Transpose V from [batch, kv_heads, seq, dim] to [batch, seq, kv_heads, dim]
    %v_t_init = tensor.empty(%batch, %seq_len_k, %n_kv_heads, %head_dim) : tensor<?x?x?x?xf16>
    %v_transposed = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg2 : tensor<?x?x?x?xf16>) outs(%v_t_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // GQA: expand K from [batch, seq, n_kv_heads, head_dim]
    //   to [batch, seq, n_kv_heads, gqa_ratio, head_dim]
    //   then collapse to [batch, seq, n_heads, head_dim]
    %gqa_ratio = arith.divui %n_heads, %n_kv_heads : index

    %k_exp_init = tensor.empty(%batch, %seq_len_k, %n_kv_heads, %gqa_ratio, %head_dim) : tensor<?x?x?x?x?xf16>
    %k_expanded = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%k_transposed : tensor<?x?x?x?xf16>) outs(%k_exp_init : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %k_full = tensor.collapse_shape %k_expanded [[0], [1], [2, 3], [4]]
        : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>

    %v_exp_init = tensor.empty(%batch, %seq_len_k, %n_kv_heads, %gqa_ratio, %head_dim) : tensor<?x?x?x?x?xf16>
    %v_expanded = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%v_transposed : tensor<?x?x?x?xf16>) outs(%v_exp_init : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %v_full = tensor.collapse_shape %v_expanded [[0], [1], [2, 3], [4]]
        : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>

    // Now Q, K_full, V_full are all [batch, seq, n_heads, head_dim]
    // Collapse batch*n_heads for iree_linalg_ext.attention: [batch*n_heads, seq, head_dim]
    // Rearrange: [batch, seq, n_heads, dim] -> collapse batch,seq -> no
    // iree_linalg_ext.attention expects [batch, seq, dim] where batch includes head dim
    // So: [batch, seq, n_heads, dim] -> [batch*n_heads, seq, dim]
    // Need to permute first: [batch, n_heads, seq, dim] then collapse [batch*n_heads, seq, dim]
    // q_transposed is [batch, seq, n_heads, dim]
    // Permute to [batch, n_heads, seq, dim]:
    %q_perm_init = tensor.empty(%batch, %n_heads, %seq_len_q, %head_dim) : tensor<?x?x?x?xf16>
    %q_perm = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%q_transposed : tensor<?x?x?x?xf16>) outs(%q_perm_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %k_perm_init = tensor.empty(%batch, %n_heads, %seq_len_k, %head_dim) : tensor<?x?x?x?xf16>
    %k_perm = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%k_full : tensor<?x?x?x?xf16>) outs(%k_perm_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %v_perm_init = tensor.empty(%batch, %n_heads, %seq_len_k, %head_dim) : tensor<?x?x?x?xf16>
    %v_perm = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%v_full : tensor<?x?x?x?xf16>) outs(%v_perm_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %bnh = arith.muli %batch, %n_heads : index
    %q_3d = tensor.collapse_shape %q_perm [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %k_3d = tensor.collapse_shape %k_perm [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %v_3d = tensor.collapse_shape %v_perm [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>

    // Causal mask: [batch*n_heads, seq_len_q, seq_len_k]
    // mask[b, i, j] = (j <= i)  — lower triangular
    %mask_init = tensor.empty(%bnh, %seq_len_q, %seq_len_k) : tensor<?x?x?xi1>
    %causal_mask = linalg.generic {
        indexing_maps = [#map8],
        iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%mask_init : tensor<?x?x?xi1>) {
    ^bb0(%out: i1):
      %qi = linalg.index 1 : index
      %ki = linalg.index 2 : index
      %cmp = arith.cmpi ule, %ki, %qi : index
      linalg.yield %cmp : i1
    } -> tensor<?x?x?xi1>

    // Call iree_linalg_ext.attention with causal mask
    %out_init = tensor.empty(%bnh, %seq_len_q, %head_dim) : tensor<?x?x?xf16>
    %attn_out = iree_linalg_ext.attention {
        indexing_maps = [#map9, #map10, #map11, #map12, #map13, #map14]
    } ins(%q_3d, %k_3d, %v_3d, %arg3, %causal_mask
        : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f32, tensor<?x?x?xi1>)
      outs(%out_init : tensor<?x?x?xf16>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
    } -> tensor<?x?x?xf16>

    // Reshape: [batch*n_heads, seq, head_dim] -> [batch, n_heads, seq, head_dim]
    %out_4d = tensor.expand_shape %attn_out [[0, 1], [2], [3]]
        output_shape [%batch, %n_heads, %seq_len_q, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    util.return %out_4d : tensor<?x?x?x?xf16>
  }
  // ---- Transformer layer prefill with direct cache ----
  util.func private @transformer_layer_prefill(
      %hidden: tensor<?x?xf16>,           // [seq_len, hidden_dim]
      %positions: tensor<?xi64>,           // [seq_len]
      %cache: !util.list<?>,               // KV cache
      %max_seq_len_val: index,             // max_seq_len for slot calculation
      %start_pos: index,                   // starting position in cache
      %layer_idx: i32,                      // layer index
      %n_heads_idx: index,                 // 16
      %n_kv_heads_idx: index,              // 8
      %hidden_dim: index,                  // 1024
      %ffn_dim: index,                     // 3072
      %rms_eps: f32,                       // 1e-6
      %rope_base: f32,                     // 1e6
      %rope_scale: f32                     // 1.0
  ) -> (tensor<?x?xf16>, !util.list<?>) { // [seq_len, hidden_dim], updated cache
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst_zero_f16 = arith.constant 0.000000e+00 : f16
    %cst_zero_f32 = arith.constant 0.000000e+00 : f32

    %seq_len = tensor.dim %hidden, %c0 : tensor<?x?xf16>
    %layer_idx_index = arith.index_cast %layer_idx : i32 to index

    // head_dim from hparams (128 for Qwen3 — NOT hidden_dim/n_heads which gives 64)
    %head_dim_i64 = util.call @hparams.head_dim() : () -> i64
    %head_dim = arith.index_cast %head_dim_i64 : i64 to index
    // kv_dim = n_kv_heads * head_dim = 8 * 128 = 1024
    %kv_dim = arith.muli %n_kv_heads_idx, %head_dim : index
    // q_proj_dim = n_heads * head_dim = 16 * 128 = 2048
    %q_proj_dim = arith.muli %n_heads_idx, %head_dim : index

    // --- Load weights ---
    %attn_norm_w = util.call @model_params.attn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %ffn_norm_w = util.call @model_params.ffn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %q_weight = util.call @model_params.attn_q_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %k_weight = util.call @model_params.attn_k_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %v_weight = util.call @model_params.attn_v_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %o_weight = util.call @model_params.attn_output_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %q_norm_w = util.call @model_params.attn_q_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %k_norm_w = util.call @model_params.attn_k_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %gate_up_weight = util.call @model_params.ffn_gate_up_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %down_weight = util.call @model_params.ffn_down_weight(%layer_idx) : (i32) -> tensor<?x?xf16>

    // ---- Attention sub-block ----

    // 1. RMS norm on input
    %normed = util.call @rms_norm_components.rms_norm_linalg(%hidden, %attn_norm_w, %rms_eps) : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // 2. QKV projection: [seq_len, hidden_dim] @ [hidden_dim, proj_dim]
    %q_init = tensor.empty(%seq_len, %q_proj_dim) : tensor<?x?xf16>
    %q_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%q_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %q_proj = linalg.matmul ins(%normed, %q_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%q_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    %k_init = tensor.empty(%seq_len, %kv_dim) : tensor<?x?xf16>
    %k_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%k_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %k_proj = linalg.matmul ins(%normed, %k_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%k_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    %v_init = tensor.empty(%seq_len, %kv_dim) : tensor<?x?xf16>
    %v_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%v_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %v_proj = linalg.matmul ins(%normed, %v_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%v_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 3. Reshape to [seq_len, n_heads/n_kv_heads, head_dim]
    %q_3d = tensor.expand_shape %q_proj [[0], [1, 2]]
        output_shape [%seq_len, %n_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %k_3d = tensor.expand_shape %k_proj [[0], [1, 2]]
        output_shape [%seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %v_3d = tensor.expand_shape %v_proj [[0], [1, 2]]
        output_shape [%seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    // 4. QK norm (RMS norm per-head on Q and K) BEFORE RoPE
    //    Flatten [seq_len, n_heads, head_dim] -> [seq_len*n_heads, head_dim]
    //    Apply RMS norm, then reshape back
    %q_bh = arith.muli %seq_len, %n_heads_idx : index
    %q_flat = tensor.collapse_shape %q_3d [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>

    // Q norm via flow.dispatch.region (fuses reduction + elementwise into 1 dispatch)
    %q_normed_flat = flow.dispatch.region[] -> (tensor<?x?xf16>{%q_bh, %head_dim}) {
      %_si_q = tensor.empty(%q_bh) : tensor<?xf32>
      %_sz_q = linalg.fill ins(%cst_zero_f32 : f32) outs(%_si_q : tensor<?xf32>) -> tensor<?xf32>
      %_hdi_q = arith.index_cast %head_dim : index to i32
      %_hdf_q = arith.sitofp %_hdi_q : i32 to f32
      %_ss_q = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
          ins(%q_flat : tensor<?x?xf16>) outs(%_sz_q : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<?xf32>
      %_oi_q = tensor.empty(%q_bh, %head_dim) : tensor<?x?xf16>
      %_on_q = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]}
          ins(%q_flat, %_ss_q, %q_norm_w : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%_oi_q : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf_q : f32
        %b = arith.addf %a, %rms_eps : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<?x?xf16>
      flow.return %_on_q : tensor<?x?xf16>
    }

    %k_bh = arith.muli %seq_len, %n_kv_heads_idx : index
    %k_flat = tensor.collapse_shape %k_3d [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>

    // K norm via flow.dispatch.region
    %k_normed_flat = flow.dispatch.region[] -> (tensor<?x?xf16>{%k_bh, %head_dim}) {
      %_si_k = tensor.empty(%k_bh) : tensor<?xf32>
      %_sz_k = linalg.fill ins(%cst_zero_f32 : f32) outs(%_si_k : tensor<?xf32>) -> tensor<?xf32>
      %_hdi_k = arith.index_cast %head_dim : index to i32
      %_hdf_k = arith.sitofp %_hdi_k : i32 to f32
      %_ss_k = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
          ins(%k_flat : tensor<?x?xf16>) outs(%_sz_k : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<?xf32>
      %_oi_k = tensor.empty(%k_bh, %head_dim) : tensor<?x?xf16>
      %_on_k = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]}
          ins(%k_flat, %_ss_k, %k_norm_w : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%_oi_k : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf_k : f32
        %b = arith.addf %a, %rms_eps : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<?x?xf16>
      flow.return %_on_k : tensor<?x?xf16>
    }

    // Reshape back to [seq_len, n_heads/n_kv_heads, head_dim]
    %q_normed_3d = tensor.expand_shape %q_normed_flat [[0, 1], [2]]
        output_shape [%seq_len, %n_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %k_normed_3d = tensor.expand_shape %k_normed_flat [[0, 1], [2]]
        output_shape [%seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    // 5. Reshape for RoPE: [batch=1, seq_len, n_heads, head_dim]
    %c1_idx = arith.constant 1 : index
    %q_4d = tensor.expand_shape %q_normed_3d [[0, 1], [2], [3]]
        output_shape [%c1_idx, %seq_len, %n_heads_idx, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %k_4d = tensor.expand_shape %k_normed_3d [[0, 1], [2], [3]]
        output_shape [%c1_idx, %seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    // Build positions tensor for RoPE: [1, seq_len] from the input positions [seq_len]
    %pos_2d = tensor.expand_shape %positions [[0, 1]]
        output_shape [%c1_idx, %seq_len]
        : tensor<?xi64> into tensor<?x?xi64>

    // 6. Apply RoPE
    %q_roped = util.call @position_components.rope(%q_4d, %pos_2d, %rope_base, %rope_scale) : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>
    %k_roped = util.call @position_components.rope(%k_4d, %pos_2d, %rope_base, %rope_scale) : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>

    // 7. Transpose for attention: [1, seq_len, heads, dim] -> [1, heads, seq_len, dim]
    %q_attn_init = tensor.empty(%c1_idx, %n_heads_idx, %seq_len, %head_dim) : tensor<?x?x?x?xf16>
    %q_attn = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%q_roped : tensor<?x?x?x?xf16>) outs(%q_attn_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %k_attn_init = tensor.empty(%c1_idx, %n_kv_heads_idx, %seq_len, %head_dim) : tensor<?x?x?x?xf16>
    %k_attn = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%k_roped : tensor<?x?x?x?xf16>) outs(%k_attn_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // V: reshape [seq_len, n_kv_heads, head_dim] -> [1, seq_len, n_kv_heads, head_dim]
    //    then transpose to [1, n_kv_heads, seq_len, head_dim]
    %v_4d = tensor.expand_shape %v_3d [[0, 1], [2], [3]]
        output_shape [%c1_idx, %seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %v_attn_init = tensor.empty(%c1_idx, %n_kv_heads_idx, %seq_len, %head_dim) : tensor<?x?x?x?xf16>
    %v_attn = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%v_4d : tensor<?x?x?x?xf16>) outs(%v_attn_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // 8. Compute scale = 1/sqrt(head_dim)
    %head_dim_i32 = arith.index_cast %head_dim : index to i32
    %head_dim_f32 = arith.sitofp %head_dim_i32 : i32 to f32
    %scale = math.rsqrt %head_dim_f32 : f32

    // 9. Call attention_prefill
    %attn_out = util.call @attention_prefill(%q_attn, %k_attn, %v_attn, %scale)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f32) -> tensor<?x?x?x?xf16>

    // 10. Reshape attention output: [1, n_heads, seq_len, head_dim]
    //     -> transpose to [1, seq_len, n_heads, head_dim]
    //     -> collapse to [seq_len, n_heads*head_dim]
    %attn_t_init = tensor.empty(%c1_idx, %seq_len, %n_heads_idx, %head_dim) : tensor<?x?x?x?xf16>
    %attn_transposed = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%attn_out : tensor<?x?x?x?xf16>) outs(%attn_t_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    // [1, seq_len, n_heads, head_dim] -> [seq_len, n_heads*head_dim]
    %attn_flat = tensor.collapse_shape %attn_transposed [[0, 1], [2, 3]]
        : tensor<?x?x?x?xf16> into tensor<?x?xf16>

    // 11. Output projection: [seq_len, n_heads*head_dim] @ [n_heads*head_dim, hidden_dim]
    %o_init = tensor.empty(%seq_len, %hidden_dim) : tensor<?x?xf16>
    %o_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%o_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %attn_output = linalg.matmul ins(%attn_flat, %o_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%o_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 12. Scatter K/V into cache for ALL positions using direct scatter
    // K after RoPE: [1, seq_len, n_kv_heads, head_dim] -> [seq_len, n_kv_heads, head_dim]
    %k_for_cache = tensor.collapse_shape %k_roped [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    // v_3d is already [seq_len, n_kv_heads, head_dim]
    %cache_updated = util.call @kvcache_direct_scatter_prefill(
        %cache, %layer_idx_index, %k_for_cache, %v_3d, %max_seq_len_val, %start_pos)
        : (!util.list<?>, index, tensor<?x?x?xf16>, tensor<?x?x?xf16>, index, index) -> !util.list<?>

    // 13. Residual connection: hidden + attn_output
    %res1_init = tensor.empty(%seq_len, %hidden_dim) : tensor<?x?xf16>
    %residual1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%hidden, %attn_output : tensor<?x?xf16>, tensor<?x?xf16>) outs(%res1_init : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %r = arith.addf %in, %in_1 : f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>

    // ---- FFN sub-block ----

    // 14. RMS norm
    %ffn_normed = util.call @rms_norm_components.rms_norm_linalg(%residual1, %ffn_norm_w, %rms_eps) : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // 15. Gate+Up projection: [seq_len, hidden_dim] @ [hidden_dim, 2*ffn_dim]
    %ffn_2x = arith.muli %ffn_dim, %c2 : index
    %gu_init = tensor.empty(%seq_len, %ffn_2x) : tensor<?x?xf16>
    %gu_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%gu_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %gate_up = linalg.matmul ins(%ffn_normed, %gate_up_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%gu_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 16. SwiGLU: split gate and up, apply silu(gate) * up
    %gate_slice = tensor.extract_slice %gate_up[0, 0] [%seq_len, %ffn_dim] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %up_slice = tensor.extract_slice %gate_up[0, %ffn_dim] [%seq_len, %ffn_dim] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>

    %swiglu_init = tensor.empty(%seq_len, %ffn_dim) : tensor<?x?xf16>
    %swiglu = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%gate_slice, %up_slice : tensor<?x?xf16>, tensor<?x?xf16>) outs(%swiglu_init : tensor<?x?xf16>) {
    ^bb0(%gate: f16, %up: f16, %out: f16):
      // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
      %neg_gate = arith.negf %gate : f16
      %exp_neg = math.exp %neg_gate : f16
      %one = arith.constant 1.000000e+00 : f16
      %denom = arith.addf %one, %exp_neg : f16
      %sigmoid = arith.divf %one, %denom : f16
      %silu = arith.mulf %gate, %sigmoid : f16
      %result = arith.mulf %silu, %up : f16
      linalg.yield %result : f16
    } -> tensor<?x?xf16>

    // 17. Down projection: [seq_len, ffn_dim] @ [ffn_dim, hidden_dim]
    %down_init = tensor.empty(%seq_len, %hidden_dim) : tensor<?x?xf16>
    %down_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%down_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %ffn_out = linalg.matmul ins(%swiglu, %down_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%down_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 18. Residual connection: residual1 + ffn_out
    %res2_init = tensor.empty(%seq_len, %hidden_dim) : tensor<?x?xf16>
    %residual2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%residual1, %ffn_out : tensor<?x?xf16>, tensor<?x?xf16>) outs(%res2_init : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %r = arith.addf %in, %in_1 : f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>

    util.return %residual2, %cache_updated : tensor<?x?xf16>, !util.list<?>
  }
  // ---- prefill: direct cache ----
  util.func public @prefill(
      %tokens: tensor<?xi64>,             // [seq_len] input token IDs
      %seq_len_val: index,                 // seq_len (dynamic)
      %cache: !util.list<?>,               // KV cache (pre-allocated)
      %max_seq_len_val: index,             // max_seq_len for slot calculation
      %start_pos: index                    // starting position in cache (0 for first turn)
  ) -> (tensor<?x?xf16>, !util.list<?>) { // [1, vocab_size] logits, updated cache
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Load hyperparams
    %vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layers_i64 = util.call @hparams.block_count() : () -> i64
    %hidden_dim_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_heads_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_kv_heads_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %ffn_dim_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %rope_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    %vocab = arith.index_cast %vocab_i64 : i64 to index
    %n_layers = arith.index_cast %n_layers_i64 : i64 to index
    %hidden_dim = arith.index_cast %hidden_dim_i64 : i64 to index
    %n_heads = arith.index_cast %n_heads_i64 : i64 to index
    %n_kv_heads = arith.index_cast %n_kv_heads_i64 : i64 to index
    %ffn_dim = arith.index_cast %ffn_dim_i64 : i64 to index

    %rope_scale = arith.constant 1.000000e+00 : f32
    %cst_zero_f16 = arith.constant 0.000000e+00 : f16

    // 1. Embedding lookup for ALL tokens: [seq_len] -> [seq_len, hidden_dim]
    %embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %hidden_init = util.call @embedding_components.embedding_lookup_1d(%embd_weight, %tokens)
        : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

    // 2. Build positions tensor: [0, 1, 2, ..., seq_len-1] offset by start_pos
    %pos_init = tensor.empty(%seq_len_val) : tensor<?xi64>
    %positions = linalg.generic {
        indexing_maps = [#map2],
        iterator_types = ["parallel"]
    } outs(%pos_init : tensor<?xi64>) {
    ^bb0(%out: i64):
      %idx = linalg.index 0 : index
      %abs_pos = arith.addi %idx, %start_pos : index
      %val = arith.index_cast %abs_pos : index to i64
      linalg.yield %val : i64
    } -> tensor<?xi64>

    // 3. Loop over 28 layers
    %result:2 = scf.for %layer_iv = %c0 to %n_layers step %c1
        iter_args(%h = %hidden_init, %c = %cache) -> (tensor<?x?xf16>, !util.list<?>) {
      %li32 = arith.index_cast %layer_iv : index to i32
      %layer_out:2 = util.call @transformer_layer_prefill(
          %h, %positions, %c, %max_seq_len_val, %start_pos, %li32,
          %n_heads, %n_kv_heads, %hidden_dim, %ffn_dim,
          %rms_eps, %rope_base, %rope_scale)
          : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>, index, index, i32,
             index, index, index, index, f32, f32, f32) -> (tensor<?x?xf16>, !util.list<?>)
      scf.yield %layer_out#0, %layer_out#1 : tensor<?x?xf16>, !util.list<?>
    }

    // 4. Final RMS norm on full hidden [seq_len, hidden_dim]
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %normed_final = util.call @rms_norm_components.rms_norm_linalg(%result#0, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // 5. Extract LAST token's hidden state: [seq_len, hidden_dim] -> [1, hidden_dim]
    %last_idx = arith.subi %seq_len_val, %c1 : index
    %last_hidden = tensor.extract_slice %normed_final[%last_idx, 0] [1, %hidden_dim] [1, 1]
        : tensor<?x?xf16> to tensor<1x?xf16>
    %last_hidden_dyn = tensor.cast %last_hidden : tensor<1x?xf16> to tensor<?x?xf16>

    // 6. Output projection: [1, hidden_dim] @ [hidden_dim, vocab_size] -> [1, vocab_size]
    %output_w = util.call @model_params.output_weight() : () -> tensor<?x?xf16>
    %logits_init = tensor.empty(%c1, %vocab) : tensor<?x?xf16>
    %logits_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %logits = linalg.matmul ins(%last_hidden_dyn, %output_w : tensor<?x?xf16>, tensor<?x?xf16>) outs(%logits_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    util.return %logits, %result#1 : tensor<?x?xf16>, !util.list<?>
  }
  // ---- run: direct cache ----
  util.func public @run(%arg0: tensor<?xi64>, %arg1: index, %arg2: index, %arg3: i64) -> (tensor<?xi64>, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %total = arith.addi %arg1, %arg2 : index
    %cache_init = util.call @allocate_kv_cache(%total) : (index) -> !util.list<?>
    // Parallel prefill — processes ALL tokens at once (not one-by-one)
    %prefill:2 = util.call @prefill(%arg0, %arg1, %cache_init, %total, %c0) : (tensor<?xi64>, index, !util.list<?>, index, index) -> (tensor<?x?xf16>, !util.list<?>)
    // Argmax on last prefill logits
    %cst_neg_inf = arith.constant 0xFC00 : f16
    %c_neg1_i64 = arith.constant -1 : i64
    %am_val_init = tensor.empty() : tensor<f16>
    %am_idx_init = tensor.empty() : tensor<i64>
    %am_val = linalg.fill ins(%cst_neg_inf : f16) outs(%am_val_init : tensor<f16>) -> tensor<f16>
    %am_idx = linalg.fill ins(%c_neg1_i64 : i64) outs(%am_idx_init : tensor<i64>) -> tensor<i64>
    %vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %vocab = arith.index_cast %vocab_i64 : i64 to index
    %logits_row = tensor.extract_slice %prefill#0[0, 0] [1, %vocab] [1, 1] : tensor<?x?xf16> to tensor<?xf16>
    %argmax:2 = linalg.generic {indexing_maps = [#map2, #map16, #map16], iterator_types = ["reduction"]} ins(%logits_row : tensor<?xf16>) outs(%am_val, %am_idx : tensor<f16>, tensor<i64>) {
    ^bb0(%in: f16, %out: f16, %out_idx: i64):
      %idx_val = linalg.index 0 : index
      %idx_i64 = arith.index_cast %idx_val : index to i64
      %is_gt = arith.cmpf ogt, %in, %out : f16
      %new_val = arith.select %is_gt, %in, %out : f16
      %new_idx = arith.select %is_gt, %idx_i64, %out_idx : i64
      linalg.yield %new_val, %new_idx : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    %first_tok = tensor.extract %argmax#1[] : tensor<i64>
    %last_pos = arith.index_cast %arg1 : index to i64
    %gen:3 = util.call @generate(%first_tok, %prefill#1, %total, %arg2, %arg3, %last_pos) : (i64, !util.list<?>, index, index, i64, i64) -> (tensor<?xi64>, index, !util.list<?>)
    util.return %gen#0, %gen#1 : tensor<?xi64>, index
  }

  // ---- Multi-turn chat with persistent KV cache ----

  // Initialize cache for a conversation (call once)
  util.func public @init_chat(%max_ctx: index) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %cache = util.call @allocate_kv_cache(%max_ctx) : (index) -> !util.list<?>
    util.global.store %cache, @kv_cache : !util.list<?>
    util.global.store %max_ctx, @max_seq_len : index
    util.global.store %c0, @current_pos : index
    util.global.store %true, @is_initialized : i1
    util.return
  }

  // One chat turn: prefill new tokens + generate. All state in globals.
  util.func public @chat_turn(%new_tokens: tensor<?xi64>, %n_tokens: index, %max_gen: index, %eos: i64) -> (tensor<?xi64>, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_zero = arith.constant 0.000000e+00 : f16
    %cst_neg_inf = arith.constant 0xFC00 : f16
    %c_neg1_i64 = arith.constant -1 : i64
    %cache = util.global.load @kv_cache : !util.list<?>
    %mslen = util.global.load @max_seq_len : index
    %pos = util.global.load @current_pos : index
    %vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %vocab = arith.index_cast %vocab_i64 : i64 to index
    // Parallel prefill: process ALL new tokens at once
    %pfill:2 = util.call @prefill(%new_tokens, %n_tokens, %cache, %mslen, %pos) : (tensor<?xi64>, index, !util.list<?>, index, index) -> (tensor<?x?xf16>, !util.list<?>)
    %new_pos = arith.addi %pos, %n_tokens : index
    %logits_row = tensor.extract_slice %pfill#0[0, 0] [1, %vocab] [1, 1] : tensor<?x?xf16> to tensor<?xf16>
    %am_v_i = tensor.empty() : tensor<f16>
    %am_x_i = tensor.empty() : tensor<i64>
    %am_v = linalg.fill ins(%cst_neg_inf : f16) outs(%am_v_i : tensor<f16>) -> tensor<f16>
    %am_x = linalg.fill ins(%c_neg1_i64 : i64) outs(%am_x_i : tensor<i64>) -> tensor<i64>
    %argmax:2 = linalg.generic {indexing_maps = [#map2, #map16, #map16], iterator_types = ["reduction"]} ins(%logits_row : tensor<?xf16>) outs(%am_v, %am_x : tensor<f16>, tensor<i64>) {
    ^bb0(%in: f16, %out: f16, %out_idx: i64):
      %idx = linalg.index 0 : index
      %idx_i64 = arith.index_cast %idx : index to i64
      %gt = arith.cmpf ogt, %in, %out : f16
      %nv = arith.select %gt, %in, %out : f16
      %ni = arith.select %gt, %idx_i64, %out_idx : i64
      linalg.yield %nv, %ni : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    %first_tok = tensor.extract %argmax#1[] : tensor<i64>
    // Generate
    %gen_start = arith.index_cast %new_pos : index to i64
    %gen:3 = util.call @generate(%first_tok, %pfill#1, %mslen, %max_gen, %eos, %gen_start) : (i64, !util.list<?>, index, index, i64, i64) -> (tensor<?xi64>, index, !util.list<?>)
    // Update persistent state
    %gen_plus1 = arith.addi %gen#1, %c1 : index
    %final_pos = arith.addi %new_pos, %gen_plus1 : index
    util.global.store %gen#2, @kv_cache : !util.list<?>
    util.global.store %final_pos, @current_pos : index
    util.return %gen#0, %gen#1 : tensor<?xi64>, index
  }
}
