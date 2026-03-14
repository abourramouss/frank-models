// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SwiGLU activation: silu(gate) * up
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Usage:
//   %output = call @swiglu(%gate, %up)
//       : (tensor<?x?x?xf16>, tensor<?x?x?xf16>) -> tensor<?x?x?xf16>

module @activation_components {

  util.func public @swiglu(
      %gate: tensor<?x?x?xf16>,  // [batch, seq, n_ff]
      %up: tensor<?x?x?xf16>     // [batch, seq, n_ff]
  ) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %gate, %c0 : tensor<?x?x?xf16>
    %d1 = tensor.dim %gate, %c1 : tensor<?x?x?xf16>
    %d2 = tensor.dim %gate, %c2 : tensor<?x?x?xf16>

    %output_init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%gate, %up : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
      outs(%output_init : tensor<?x?x?xf16>) {
    ^bb0(%g: f16, %u: f16, %out: f16):
      // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
      %neg_g = arith.negf %g : f16
      %exp_neg = math.exp %neg_g : f16
      %one = arith.constant 1.0 : f16
      %denom = arith.addf %one, %exp_neg : f16
      %sigmoid = arith.divf %one, %denom : f16
      %silu = arith.mulf %g, %sigmoid : f16
      // silu(gate) * up
      %result = arith.mulf %silu, %u : f16
      linalg.yield %result : f16
    } -> tensor<?x?x?xf16>

    util.return %output : tensor<?x?x?xf16>
  }

}
