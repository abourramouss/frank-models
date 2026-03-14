// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fused indirect matrix multiply for MoE expert selection.
// Combines gather + matmul into a single linalg.generic that reads
// expert weights inline via tensor.extract — no intermediate gather buffer.
//
// For each expert slot e and token t:
//   expert_id = ids[e, t]
//   result[out_dim, e, t] = sum_k(weights[expert_id, out_dim, k] * input[k, e, t])

module @moe_components {

  util.func public @mul_mat_id(
      %weights: tensor<?x?x?xf16>,      // Expert weights [n_expert, n_out, n_in] (pre-transposed)
      %input: tensor<?x?x?xf16>,        // Input [n_in, n_expert_used, n_tokens]
      %ids: tensor<?x?xi32>             // Expert indices [n_expert_used, n_tokens]
  ) -> tensor<?x?x?xf16> {              // Output [n_out, n_expert_used, n_tokens]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %n_expert = tensor.dim %weights, %c0 : tensor<?x?x?xf16>
    %n_out = tensor.dim %weights, %c1 : tensor<?x?x?xf16>
    %n_in = tensor.dim %weights, %c2 : tensor<?x?x?xf16>
    %n_expert_used = tensor.dim %input, %c1 : tensor<?x?x?xf16>
    %n_tokens = tensor.dim %input, %c2 : tensor<?x?x?xf16>

    // Fused gather + matmul: single linalg.generic with reduction.
    // Output: [n_out, n_expert_used, n_tokens]
    // For each (out_dim, expert_slot, token):
    //   expert_id = ids[expert_slot, token]
    //   result += weights[expert_id, out_dim, k] * input[k, expert_slot, token]
    %zero = arith.constant 0.0 : f16
    %output_init = tensor.empty(%n_out, %n_expert_used, %n_tokens) : tensor<?x?x?xf16>
    %output_filled = linalg.fill ins(%zero : f16) outs(%output_init : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d_out, d_expert, d_token, d_k) -> (d_k, d_expert, d_token)>,   // input [n_in, n_expert_used, n_tokens]
        affine_map<(d_out, d_expert, d_token, d_k) -> (d_expert, d_token)>,          // ids [n_expert_used, n_tokens]
        affine_map<(d_out, d_expert, d_token, d_k) -> (d_out, d_expert, d_token)>   // output [n_out, n_expert_used, n_tokens]
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%input, %ids : tensor<?x?x?xf16>, tensor<?x?xi32>)
      outs(%output_filled : tensor<?x?x?xf16>) {
    ^bb0(%in_val: f16, %expert_id_i32: i32, %acc: f16):
      %out_idx = linalg.index 0 : index
      %k_idx = linalg.index 3 : index

      // Gather weight inline: weights[expert_id, out_idx, k_idx]
      %expert_id = arith.index_cast %expert_id_i32 : i32 to index
      %w_val = tensor.extract %weights[%expert_id, %out_idx, %k_idx] : tensor<?x?x?xf16>

      // Multiply-accumulate
      %prod = arith.mulf %w_val, %in_val : f16
      %sum = arith.addf %acc, %prod : f16
      linalg.yield %sum : f16
    } -> tensor<?x?x?xf16>

    util.return %result : tensor<?x?x?xf16>
  }

}
