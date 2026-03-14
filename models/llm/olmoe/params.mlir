// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// OLMoE-1B-7B parameter accessors (stacked parameter variant).
//
// Uses flow.tensor.constant with #flow.parameter.named to load stacked
// parameter tensors from an IREE parameter archive at runtime.
// Layer selection is done via tensor.extract_slice on the stacked tensor
// instead of per-layer scf.if chains.
//
// Stacked parameter archive: olmoe-1b-7b-f16-stacked.irpa
//
// Parameter naming:
//   Global: token_embd.weight, output_norm.weight, output.weight
//   Stacked: stacked.<name> with shape [16, flat_per_layer]
//
// Shapes (OLMoE-1B-7B):
//   vocab_size=50304, n_embd=2048, n_head=16, n_head_kv=16, head_dim=128
//   n_ff=1024, n_expert=64, n_layers=16

module @model_params {

  // ===== Model-level (global) parameters =====

  util.func public @token_embd_weight() -> tensor<?x?xf16> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"token_embd.weight"> : tensor<50304x2048xf16>
    %dyn = tensor.cast %w : tensor<50304x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  util.func public @output_norm_weight() -> tensor<?xf16> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"output_norm.weight"> : tensor<2048xf16>
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @output_weight() -> tensor<?x?xf16> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"output.weight"> : tensor<2048x50304xf16>
    %dyn = tensor.cast %w : tensor<2048x50304xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  // ===== Layer-level parameters (stacked) =====
  // Each function loads a stacked [16, ...] tensor, slices by layer index,
  // and reshapes to the expected per-layer shape.

  // --- 1D normalization weights [16, 2048] -> [2048] ---

  util.func public @attn_norm_weight(%layer: i32) -> tensor<?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_norm.weight"> : tensor<16x2048xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 2048] [1, 1]
        : tensor<16x2048xf16> to tensor<1x2048xf16>
    %result = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x2048xf16> into tensor<2048xf16>
    %dyn = tensor.cast %result : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @ffn_norm_weight(%layer: i32) -> tensor<?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_norm.weight"> : tensor<16x2048xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 2048] [1, 1]
        : tensor<16x2048xf16> to tensor<1x2048xf16>
    %result = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x2048xf16> into tensor<2048xf16>
    %dyn = tensor.cast %result : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_q_norm_weight(%layer: i32) -> tensor<?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q_norm.weight"> : tensor<16x2048xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 2048] [1, 1]
        : tensor<16x2048xf16> to tensor<1x2048xf16>
    %result = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x2048xf16> into tensor<2048xf16>
    %dyn = tensor.cast %result : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_k_norm_weight(%layer: i32) -> tensor<?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k_norm.weight"> : tensor<16x2048xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 2048] [1, 1]
        : tensor<16x2048xf16> to tensor<1x2048xf16>
    %result = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x2048xf16> into tensor<2048xf16>
    %dyn = tensor.cast %result : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  // --- 1D bias vectors [16, 2048] -> [2048] ---

  util.func public @attn_q_bias(%layer: i32) -> tensor<?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q.bias"> : tensor<16x2048xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 2048] [1, 1]
        : tensor<16x2048xf16> to tensor<1x2048xf16>
    %result = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x2048xf16> into tensor<2048xf16>
    %dyn = tensor.cast %result : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_k_bias(%layer: i32) -> tensor<?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k.bias"> : tensor<16x2048xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 2048] [1, 1]
        : tensor<16x2048xf16> to tensor<1x2048xf16>
    %result = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x2048xf16> into tensor<2048xf16>
    %dyn = tensor.cast %result : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_v_bias(%layer: i32) -> tensor<?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_v.bias"> : tensor<16x2048xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 2048] [1, 1]
        : tensor<16x2048xf16> to tensor<1x2048xf16>
    %result = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x2048xf16> into tensor<2048xf16>
    %dyn = tensor.cast %result : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_output_bias(%layer: i32) -> tensor<?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_output.bias"> : tensor<16x2048xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 2048] [1, 1]
        : tensor<16x2048xf16> to tensor<1x2048xf16>
    %result = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x2048xf16> into tensor<2048xf16>
    %dyn = tensor.cast %result : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  // --- 2D attention weights (stored flat) ---

  // attn_qkv_weight: [16, 12582912] -> [2048, 6144]  (12582912 = 2048 * 6144)
  util.func public @attn_qkv_weight(%layer: i32) -> tensor<?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_qkv.weight"> : tensor<16x12582912xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 12582912] [1, 1]
        : tensor<16x12582912xf16> to tensor<1x12582912xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x12582912xf16> into tensor<12582912xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1]]
        output_shape [2048, 6144]
        : tensor<12582912xf16> into tensor<2048x6144xf16>
    %dyn = tensor.cast %reshaped : tensor<2048x6144xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  // attn_q_weight: [16, 4194304] -> [2048, 2048]  (4194304 = 2048 * 2048)
  util.func public @attn_q_weight(%layer: i32) -> tensor<?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q.weight"> : tensor<16x4194304xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 4194304] [1, 1]
        : tensor<16x4194304xf16> to tensor<1x4194304xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x4194304xf16> into tensor<4194304xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1]]
        output_shape [2048, 2048]
        : tensor<4194304xf16> into tensor<2048x2048xf16>
    %dyn = tensor.cast %reshaped : tensor<2048x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  // attn_k_weight: [16, 4194304] -> [2048, 2048]
  util.func public @attn_k_weight(%layer: i32) -> tensor<?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k.weight"> : tensor<16x4194304xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 4194304] [1, 1]
        : tensor<16x4194304xf16> to tensor<1x4194304xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x4194304xf16> into tensor<4194304xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1]]
        output_shape [2048, 2048]
        : tensor<4194304xf16> into tensor<2048x2048xf16>
    %dyn = tensor.cast %reshaped : tensor<2048x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  // attn_v_weight: [16, 4194304] -> [2048, 2048]
  util.func public @attn_v_weight(%layer: i32) -> tensor<?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_v.weight"> : tensor<16x4194304xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 4194304] [1, 1]
        : tensor<16x4194304xf16> to tensor<1x4194304xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x4194304xf16> into tensor<4194304xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1]]
        output_shape [2048, 2048]
        : tensor<4194304xf16> into tensor<2048x2048xf16>
    %dyn = tensor.cast %reshaped : tensor<2048x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  // attn_output_weight: [16, 4194304] -> [2048, 2048]
  util.func public @attn_output_weight(%layer: i32) -> tensor<?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_output.weight"> : tensor<16x4194304xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 4194304] [1, 1]
        : tensor<16x4194304xf16> to tensor<1x4194304xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x4194304xf16> into tensor<4194304xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1]]
        output_shape [2048, 2048]
        : tensor<4194304xf16> into tensor<2048x2048xf16>
    %dyn = tensor.cast %reshaped : tensor<2048x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  // --- 2D router weight ---

  // ffn_gate_inp_weight: [16, 131072] -> [64, 2048]  (131072 = 64 * 2048)
  util.func public @ffn_gate_inp_weight(%layer: i32) -> tensor<?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_gate_inp.weight"> : tensor<16x131072xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 131072] [1, 1]
        : tensor<16x131072xf16> to tensor<1x131072xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x131072xf16> into tensor<131072xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1]]
        output_shape [64, 2048]
        : tensor<131072xf16> into tensor<64x2048xf16>
    %dyn = tensor.cast %reshaped : tensor<64x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  // --- 3D expert weights (stored flat, pre-transposed [n_expert, n_out, n_in]) ---

  // ffn_up_exps_weight: [16, 134217728] -> [64, 1024, 2048]
  //   (134217728 = 64 * 1024 * 2048)
  util.func public @ffn_up_exps_weight(%layer: i32) -> tensor<?x?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_up_exps.weight"> : tensor<16x134217728xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 134217728] [1, 1]
        : tensor<16x134217728xf16> to tensor<1x134217728xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x134217728xf16> into tensor<134217728xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1, 2]]
        output_shape [64, 1024, 2048]
        : tensor<134217728xf16> into tensor<64x1024x2048xf16>
    %dyn = tensor.cast %reshaped : tensor<64x1024x2048xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

  // ffn_gate_exps_weight: [16, 134217728] -> [64, 1024, 2048]
  //   (134217728 = 64 * 1024 * 2048)
  util.func public @ffn_gate_exps_weight(%layer: i32) -> tensor<?x?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_gate_exps.weight"> : tensor<16x134217728xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 134217728] [1, 1]
        : tensor<16x134217728xf16> to tensor<1x134217728xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x134217728xf16> into tensor<134217728xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1, 2]]
        output_shape [64, 1024, 2048]
        : tensor<134217728xf16> into tensor<64x1024x2048xf16>
    %dyn = tensor.cast %reshaped : tensor<64x1024x2048xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

  // ffn_down_exps_weight: [16, 134217728] -> [64, 2048, 1024]
  //   (134217728 = 64 * 2048 * 1024)
  util.func public @ffn_down_exps_weight(%layer: i32) -> tensor<?x?x?xf16> {
    %stacked = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_down_exps.weight"> : tensor<16x134217728xf16>
    %layer_idx = arith.index_cast %layer : i32 to index
    %slice = tensor.extract_slice %stacked[%layer_idx, 0] [1, 134217728] [1, 1]
        : tensor<16x134217728xf16> to tensor<1x134217728xf16>
    %flat = tensor.collapse_shape %slice [[0, 1]]
        : tensor<1x134217728xf16> into tensor<134217728xf16>
    %reshaped = tensor.expand_shape %flat [[0, 1, 2]]
        output_shape [64, 2048, 1024]
        : tensor<134217728xf16> into tensor<64x2048x1024xf16>
    %dyn = tensor.cast %reshaped : tensor<64x2048x1024xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

}
