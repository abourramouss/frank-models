// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MoE FFN block with expert routing and mixture.
// Mixed precision: f16 data path, f32 router logits + softmax (overflow safety).

module @moe_ffn_components {

  // External declarations - resolved by iree-link
  util.func private @moe_components.mul_mat_id(
      tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xi32>
  ) -> tensor<?x?x?xf16>

  util.func private @activation_components.swiglu(
      tensor<?x?x?xf16>, tensor<?x?x?xf16>
  ) -> tensor<?x?x?xf16>

  util.func public @moe_ffn_block(
      %input: tensor<?x?xf16>,          // [n_tokens, n_embd] (flattened batch*seq)
      %gate_inp_w: tensor<?x?xf16>,     // Router weights [n_expert, n_embd]
      %up_exps_w: tensor<?x?x?xf16>,    // Expert up [n_ff, n_embd, n_expert]
      %gate_exps_w: tensor<?x?x?xf16>,  // Expert gate [n_ff, n_embd, n_expert]
      %down_exps_w: tensor<?x?x?xf16>,  // Expert down [n_embd, n_ff, n_expert]
      %n_expert: index,
      %n_expert_used: index,
      %n_embd: index,
      %n_ff: index
  ) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n_tokens = tensor.dim %input, %c0 : tensor<?x?xf16>

    // Step 1: Router logits in f32 (for softmax numerical stability).
    // Transpose input to f32.
    %input_t_init = tensor.empty(%n_embd, %n_tokens) : tensor<?x?xf32>
    %input_t = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<?x?xf16>) outs(%input_t_init : tensor<?x?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %v = arith.extf %in : f16 to f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>

    // Promote gate weights to f32 for router matmul.
    %gate_inp_w_f32_init = tensor.empty(%n_expert, %n_embd) : tensor<?x?xf32>
    %gate_inp_w_f32 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%gate_inp_w : tensor<?x?xf16>) outs(%gate_inp_w_f32_init : tensor<?x?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %v = arith.extf %in : f16 to f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>

    %zero_f32 = arith.constant 0.0 : f32
    %logits_init = tensor.empty(%n_expert, %n_tokens) : tensor<?x?xf32>
    %logits_filled = linalg.fill ins(%zero_f32 : f32) outs(%logits_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %logits = linalg.matmul ins(%gate_inp_w_f32, %input_t : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%logits_filled : tensor<?x?xf32>) -> tensor<?x?xf32>

    // Step 2: Softmax in f32.
    %probs_init = tensor.empty(%n_expert, %n_tokens) : tensor<?x?xf32>
    %probs = linalg.softmax dimension(0) ins(%logits : tensor<?x?xf32>)
        outs(%probs_init : tensor<?x?xf32>) -> tensor<?x?xf32>

    // Step 3: Top-k expert selection (f32 probs).
    %weights_init = tensor.empty(%n_expert_used, %n_tokens) : tensor<?x?xf32>
    %indices_init = tensor.empty(%n_expert_used, %n_tokens) : tensor<?x?xi32>
    %weights, %selected_experts = iree_linalg_ext.topk
        dimension(0)
        ins(%probs : tensor<?x?xf32>)
        outs(%weights_init, %indices_init : tensor<?x?xf32>, tensor<?x?xi32>) {
      ^bb0(%lhs: f32, %rhs: f32):
        %cmp = arith.cmpf ogt, %lhs, %rhs : f32
        iree_linalg_ext.yield %cmp : i1
    } -> tensor<?x?xf32>, tensor<?x?xi32>

    // Step 4: Conditional weight normalization (f32).
    // OLMoE: normalize_weights=false, skip normalization.
    // Truncate weights to f16 for the data path.
    %weights_f16_init = tensor.empty(%n_expert_used, %n_tokens) : tensor<?x?xf16>
    %weights_f16 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%weights : tensor<?x?xf32>) outs(%weights_f16_init : tensor<?x?xf16>) {
    ^bb0(%in: f32, %out: f16):
      %v = arith.truncf %in : f32 to f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // Step 5: Reshape input for mul_mat_id (f16).
    %input_replicated_init = tensor.empty(%n_embd, %n_expert_used, %n_tokens) : tensor<?x?x?xf16>
    %input_replicated = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d0)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<?x?xf16>) outs(%input_replicated_init : tensor<?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?xf16>

    // Steps 6-9: Expert projections and activation (all f16).
    %up = util.call @moe_components.mul_mat_id(%up_exps_w, %input_replicated, %selected_experts)
        : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xi32>) -> tensor<?x?x?xf16>

    %gate = util.call @moe_components.mul_mat_id(%gate_exps_w, %input_replicated, %selected_experts)
        : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xi32>) -> tensor<?x?x?xf16>

    %activated = util.call @activation_components.swiglu(%gate, %up)
        : (tensor<?x?x?xf16>, tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

    %experts_out = util.call @moe_components.mul_mat_id(%down_exps_w, %activated, %selected_experts)
        : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xi32>) -> tensor<?x?x?xf16>

    // Step 10: Apply expert weights element-wise (f16).
    %experts_weighted_init = tensor.empty(%n_embd, %n_expert_used, %n_tokens) : tensor<?x?x?xf16>
    %experts_weighted = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%experts_out, %weights_f16 : tensor<?x?x?xf16>, tensor<?x?xf16>)
      outs(%experts_weighted_init : tensor<?x?x?xf16>) {
    ^bb0(%expert_val: f16, %weight: f16, %out: f16):
      %weighted = arith.mulf %expert_val, %weight : f16
      linalg.yield %weighted : f16
    } -> tensor<?x?x?xf16>

    // Step 11: Sum expert outputs (f16).
    %zero_f16 = arith.constant 0.0 : f16
    %output_init = tensor.empty(%n_embd, %n_tokens) : tensor<?x?xf16>
    %output_filled = linalg.fill ins(%zero_f16 : f16) outs(%output_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    %output_summed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%experts_weighted : tensor<?x?x?xf16>) outs(%output_filled : tensor<?x?xf16>) {
    ^bb0(%expert_val: f16, %acc: f16):
      %sum = arith.addf %expert_val, %acc : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    // Transpose back to [n_tokens, n_embd].
    %final_init = tensor.empty(%n_tokens, %n_embd) : tensor<?x?xf16>
    %final = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%output_summed : tensor<?x?xf16>) outs(%final_init : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?xf16>

    util.return %final : tensor<?x?xf16>
  }

}
