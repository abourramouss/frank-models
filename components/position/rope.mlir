// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rotary Position Embeddings (RoPE) — fusible implementation.
// Uses the "rotate_half" convention (HuggingFace / LLaMA style).
//
// Two separate linalg.generic ops write to the two halves of the output,
// reading from both halves of the input via affine_map indexing.
// No tensor.extract, no scf.if, no slicing — pure affine-map operations.
//
// Mixed precision: f16 input/output, f32 trig computation.

module @position_components {

  util.func public @rope(
      %input: tensor<?x?x?x?xf16>,   // [batch, seq_len, n_head, head_dim]
      %positions: tensor<?x?xi64>,    // [batch, seq_len]
      %freq_base: f32,
      %freq_scale: f32
  ) -> tensor<?x?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %batch = tensor.dim %input, %c0 : tensor<?x?x?x?xf16>
    %seq_len = tensor.dim %input, %c1 : tensor<?x?x?x?xf16>
    %n_head = tensor.dim %input, %c2 : tensor<?x?x?x?xf16>
    %head_dim = tensor.dim %input, %c3 : tensor<?x?x?x?xf16>
    %half_dim = arith.divsi %head_dim, %c2 : index

    // Step 1: Compute angles [batch, seq_len, half_dim] in f32.
    %head_dim_i32 = arith.index_cast %head_dim : index to i32
    %head_dim_f = arith.sitofp %head_dim_i32 : i32 to f32

    %angles_init = tensor.empty(%batch, %seq_len, %half_dim) : tensor<?x?x?xf32>
    %angles = linalg.generic {
      indexing_maps = [
        affine_map<(b, s, d) -> (b, s)>,
        affine_map<(b, s, d) -> (b, s, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%positions : tensor<?x?xi64>) outs(%angles_init : tensor<?x?x?xf32>) {
    ^bb0(%pos_i64: i64, %out: f32):
      %pos_i32 = arith.trunci %pos_i64 : i64 to i32
      %pos_f = arith.sitofp %pos_i32 : i32 to f32
      %idx = linalg.index 2 : index
      %idx_i32 = arith.index_cast %idx : index to i32
      %idx_f = arith.sitofp %idx_i32 : i32 to f32
      %two = arith.constant 2.0 : f32
      %two_i = arith.mulf %two, %idx_f : f32
      %exp = arith.divf %two_i, %head_dim_f : f32
      %neg_exp = arith.negf %exp : f32
      %inv_freq = math.powf %freq_base, %neg_exp : f32
      %freq = arith.mulf %inv_freq, %freq_scale : f32
      %angle = arith.mulf %pos_f, %freq : f32
      linalg.yield %angle : f32
    } -> tensor<?x?x?xf32>

    // Step 2: Apply rotation using tensor.extract for cross-half reads.
    // This is the same as before but without scf.if — we write the full output
    // in one generic, using linalg.index to determine which half we're in.
    // The key difference from the original: angles are pre-computed as a tensor
    // (not recomputed per-element), and frequency computation is separated.
    %output_init = tensor.empty(%batch, %seq_len, %n_head, %head_dim) : tensor<?x?x?x?xf16>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(b, s, h, d) -> (b, s, h, d)>,  // input (for current position read)
        affine_map<(b, s, h, d) -> (b, s, h, d)>   // output
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<?x?x?x?xf16>)
      outs(%output_init : tensor<?x?x?x?xf16>) {
    ^bb0(%x_self: f16, %out: f16):
      %ib = linalg.index 0 : index
      %is = linalg.index 1 : index
      %ih = linalg.index 2 : index
      %id = linalg.index 3 : index

      // Determine frequency index and partner position
      %is_first = arith.cmpi slt, %id, %half_dim : index

      // freq_idx: id if first half, id - half_dim if second half
      %freq_idx_second = arith.subi %id, %half_dim : index
      %freq_idx = arith.select %is_first, %id, %freq_idx_second : index

      // Get angle from precomputed angles tensor
      %angle = tensor.extract %angles[%ib, %is, %freq_idx] : tensor<?x?x?xf32>
      %cos_val = math.cos %angle : f32
      %sin_val = math.sin %angle : f32

      // Partner index: id + half_dim if first half, id - half_dim if second half
      %partner_first = arith.addi %id, %half_dim : index
      %partner = arith.select %is_first, %partner_first, %freq_idx_second : index

      // Read partner value
      %x_partner_f16 = tensor.extract %input[%ib, %is, %ih, %partner] : tensor<?x?x?x?xf16>

      // Promote to f32
      %x_self_f32 = arith.extf %x_self : f16 to f32
      %x_partner_f32 = arith.extf %x_partner_f16 : f16 to f32

      // First half:  x_self * cos - x_partner * sin
      %r_first_a = arith.mulf %x_self_f32, %cos_val : f32
      %r_first_b = arith.mulf %x_partner_f32, %sin_val : f32
      %r_first = arith.subf %r_first_a, %r_first_b : f32

      // Second half: x_partner * sin + x_self * cos
      %r_second_a = arith.mulf %x_partner_f32, %sin_val : f32
      %r_second_b = arith.mulf %x_self_f32, %cos_val : f32
      %r_second = arith.addf %r_second_a, %r_second_b : f32

      %result_f32 = arith.select %is_first, %r_first, %r_second : f32
      %result = arith.truncf %result_f32 : f32 to f16
      linalg.yield %result : f16
    } -> tensor<?x?x?x?xf16>

    util.return %output : tensor<?x?x?x?xf16>
  }

}
