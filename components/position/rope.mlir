// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rotary Position Embeddings (RoPE).
// Uses the "rotate_half" convention (HuggingFace / LLaMA style):
//   dimension i pairs with dimension i + head_dim/2.
//
// For position p and frequency freq[i]:
//   out[..., i]             = input[..., i] * cos(p * freq[i])
//                           - input[..., i + half_dim] * sin(p * freq[i])
//   out[..., i + half_dim]  = input[..., i + half_dim] * cos(p * freq[i])
//                           + input[..., i] * sin(p * freq[i])
//
// Usage:
//   %output = call @rope(%input, %positions, %freq_base, %freq_scale)
//       : (tensor<?x?x?x?xf32>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf32>

module @position_components {

  util.func public @rope(
      %input: tensor<?x?x?x?xf32>,   // [batch, seq_len, n_head, head_dim]
      %positions: tensor<?x?xi64>,    // [batch, seq_len]
      %freq_base: f32,                // typically 10000.0
      %freq_scale: f32                // typically 1.0
  ) -> tensor<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %batch = tensor.dim %input, %c0 : tensor<?x?x?x?xf32>
    %seq_len = tensor.dim %input, %c1 : tensor<?x?x?x?xf32>
    %n_head = tensor.dim %input, %c2 : tensor<?x?x?x?xf32>
    %head_dim = tensor.dim %input, %c3 : tensor<?x?x?x?xf32>

    %half_dim = arith.divsi %head_dim, %c2 : index

    // Compute frequencies for each dimension pair.
    // freq[i] = (1.0 / base^(2i/head_dim)) * freq_scale
    %head_dim_f32 = arith.index_cast %head_dim : index to i32
    %head_dim_f32_cast = arith.sitofp %head_dim_f32 : i32 to f32

    %freq_init = tensor.empty(%half_dim) : tensor<?xf32>
    %freqs = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%freq_init : tensor<?xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %idx_f32 = arith.index_cast %idx : index to i32
      %idx_f32_cast = arith.sitofp %idx_f32 : i32 to f32
      %two = arith.constant 2.0 : f32
      %two_i = arith.mulf %two, %idx_f32_cast : f32
      %exp = arith.divf %two_i, %head_dim_f32_cast : f32
      %neg_exp = arith.negf %exp : f32
      %base_freq = math.powf %freq_base, %neg_exp : f32
      %freq = arith.mulf %base_freq, %freq_scale : f32
      linalg.yield %freq : f32
    } -> tensor<?xf32>

    // Apply RoPE with half-style pairing (HuggingFace rotate_half convention).
    // Dimension i pairs with dimension i + half_dim (NOT interleaved).
    //
    // For d3 < half_dim:
    //   out[..., d3] = input[..., d3] * cos - input[..., d3 + half_dim] * sin
    // For d3 >= half_dim (let j = d3 - half_dim):
    //   out[..., d3] = input[..., d3] * cos + input[..., j] * sin
    %output_init = tensor.empty(%batch, %seq_len, %n_head, %head_dim) : tensor<?x?x?x?xf32>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%output_init : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %i3 = linalg.index 3 : index

      // Get position for this (batch, seq) element.
      %pos_i64 = tensor.extract %positions[%i0, %i1] : tensor<?x?xi64>
      %pos_i32 = arith.trunci %pos_i64 : i64 to i32
      %pos_f32 = arith.sitofp %pos_i32 : i32 to f32

      // Determine which half we're in.
      %is_first_half = arith.cmpi slt, %i3, %half_dim : index

      %result = scf.if %is_first_half -> (f32) {
        // First half (d3 < half_dim): freq_idx = d3, partner = d3 + half_dim
        %freq = tensor.extract %freqs[%i3] : tensor<?xf32>
        %angle = arith.mulf %pos_f32, %freq : f32
        %cos_val = math.cos %angle : f32
        %sin_val = math.sin %angle : f32

        %x0 = tensor.extract %input[%i0, %i1, %i2, %i3] : tensor<?x?x?x?xf32>
        %partner = arith.addi %i3, %half_dim : index
        %x1 = tensor.extract %input[%i0, %i1, %i2, %partner] : tensor<?x?x?x?xf32>

        // out = x0 * cos - x1 * sin
        %x0_cos = arith.mulf %x0, %cos_val : f32
        %x1_sin = arith.mulf %x1, %sin_val : f32
        %r = arith.subf %x0_cos, %x1_sin : f32
        scf.yield %r : f32
      } else {
        // Second half (d3 >= half_dim): freq_idx = d3 - half_dim, partner = d3 - half_dim
        %freq_idx = arith.subi %i3, %half_dim : index
        %freq = tensor.extract %freqs[%freq_idx] : tensor<?xf32>
        %angle = arith.mulf %pos_f32, %freq : f32
        %cos_val = math.cos %angle : f32
        %sin_val = math.sin %angle : f32

        %x1 = tensor.extract %input[%i0, %i1, %i2, %i3] : tensor<?x?x?x?xf32>
        %x0 = tensor.extract %input[%i0, %i1, %i2, %freq_idx] : tensor<?x?x?x?xf32>

        // out = x0 * sin + x1 * cos
        %x0_sin = arith.mulf %x0, %sin_val : f32
        %x1_cos = arith.mulf %x1, %cos_val : f32
        %r = arith.addf %x0_sin, %x1_cos : f32
        scf.yield %r : f32
      }

      linalg.yield %result : f32
    } -> tensor<?x?x?x?xf32>

    util.return %output : tensor<?x?x?x?xf32>
  }

}
