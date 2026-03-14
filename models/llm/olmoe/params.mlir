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

  // --- 3D expert weights (per-layer, pre-transposed [n_expert, n_out, n_in]) ---
  // Loaded per-layer from hybrid irpa to avoid stacking 16 x 256MB tensors.

  // ffn_up_exps_weight: per-layer tensor<64x1024x2048xf16>
  util.func public @ffn_up_exps_weight(%layer: i32) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c9 = arith.constant 9 : i32
    %c10 = arith.constant 10 : i32
    %c11 = arith.constant 11 : i32
    %c12 = arith.constant 12 : i32
    %c13 = arith.constant 13 : i32
    %c14 = arith.constant 14 : i32
    %eq0 = arith.cmpi eq, %layer, %c0 : i32
    %r = scf.if %eq0 -> tensor<64x1024x2048xf16> {
      %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
      scf.yield %w : tensor<64x1024x2048xf16>
    } else {
      %eq1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %eq1 -> tensor<64x1024x2048xf16> {
        %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
        scf.yield %w : tensor<64x1024x2048xf16>
      } else {
        %eq2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %eq2 -> tensor<64x1024x2048xf16> {
          %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
          scf.yield %w : tensor<64x1024x2048xf16>
        } else {
          %eq3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %eq3 -> tensor<64x1024x2048xf16> {
            %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
            scf.yield %w : tensor<64x1024x2048xf16>
          } else {
            %eq4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %eq4 -> tensor<64x1024x2048xf16> {
              %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
              scf.yield %w : tensor<64x1024x2048xf16>
            } else {
              %eq5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %eq5 -> tensor<64x1024x2048xf16> {
                %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                scf.yield %w : tensor<64x1024x2048xf16>
              } else {
                %eq6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %eq6 -> tensor<64x1024x2048xf16> {
                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                  scf.yield %w : tensor<64x1024x2048xf16>
                } else {
                  %eq7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %eq7 -> tensor<64x1024x2048xf16> {
                    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                    scf.yield %w : tensor<64x1024x2048xf16>
                  } else {
                    %eq8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %eq8 -> tensor<64x1024x2048xf16> {
                      %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                      scf.yield %w : tensor<64x1024x2048xf16>
                    } else {
                      %eq9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %eq9 -> tensor<64x1024x2048xf16> {
                        %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                        scf.yield %w : tensor<64x1024x2048xf16>
                      } else {
                        %eq10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %eq10 -> tensor<64x1024x2048xf16> {
                          %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                          scf.yield %w : tensor<64x1024x2048xf16>
                        } else {
                          %eq11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %eq11 -> tensor<64x1024x2048xf16> {
                            %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                            scf.yield %w : tensor<64x1024x2048xf16>
                          } else {
                            %eq12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %eq12 -> tensor<64x1024x2048xf16> {
                              %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                              scf.yield %w : tensor<64x1024x2048xf16>
                            } else {
                              %eq13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %eq13 -> tensor<64x1024x2048xf16> {
                                %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                                scf.yield %w : tensor<64x1024x2048xf16>
                              } else {
                                %eq14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %eq14 -> tensor<64x1024x2048xf16> {
                                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                                  scf.yield %w : tensor<64x1024x2048xf16>
                                } else {
                                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.ffn_up_exps.weight"> : tensor<64x1024x2048xf16>
                                  scf.yield %w : tensor<64x1024x2048xf16>
                                }
                                scf.yield %r14 : tensor<64x1024x2048xf16>
                              }
                              scf.yield %r13 : tensor<64x1024x2048xf16>
                            }
                            scf.yield %r12 : tensor<64x1024x2048xf16>
                          }
                          scf.yield %r11 : tensor<64x1024x2048xf16>
                        }
                        scf.yield %r10 : tensor<64x1024x2048xf16>
                      }
                      scf.yield %r9 : tensor<64x1024x2048xf16>
                    }
                    scf.yield %r8 : tensor<64x1024x2048xf16>
                  }
                  scf.yield %r7 : tensor<64x1024x2048xf16>
                }
                scf.yield %r6 : tensor<64x1024x2048xf16>
              }
              scf.yield %r5 : tensor<64x1024x2048xf16>
            }
            scf.yield %r4 : tensor<64x1024x2048xf16>
          }
          scf.yield %r3 : tensor<64x1024x2048xf16>
        }
        scf.yield %r2 : tensor<64x1024x2048xf16>
      }
      scf.yield %r1 : tensor<64x1024x2048xf16>
    }
    %dyn = tensor.cast %r : tensor<64x1024x2048xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

  // ffn_gate_exps_weight: per-layer tensor<64x1024x2048xf16>
  util.func public @ffn_gate_exps_weight(%layer: i32) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c9 = arith.constant 9 : i32
    %c10 = arith.constant 10 : i32
    %c11 = arith.constant 11 : i32
    %c12 = arith.constant 12 : i32
    %c13 = arith.constant 13 : i32
    %c14 = arith.constant 14 : i32
    %eq0 = arith.cmpi eq, %layer, %c0 : i32
    %r = scf.if %eq0 -> tensor<64x1024x2048xf16> {
      %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
      scf.yield %w : tensor<64x1024x2048xf16>
    } else {
      %eq1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %eq1 -> tensor<64x1024x2048xf16> {
        %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
        scf.yield %w : tensor<64x1024x2048xf16>
      } else {
        %eq2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %eq2 -> tensor<64x1024x2048xf16> {
          %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
          scf.yield %w : tensor<64x1024x2048xf16>
        } else {
          %eq3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %eq3 -> tensor<64x1024x2048xf16> {
            %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
            scf.yield %w : tensor<64x1024x2048xf16>
          } else {
            %eq4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %eq4 -> tensor<64x1024x2048xf16> {
              %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
              scf.yield %w : tensor<64x1024x2048xf16>
            } else {
              %eq5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %eq5 -> tensor<64x1024x2048xf16> {
                %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                scf.yield %w : tensor<64x1024x2048xf16>
              } else {
                %eq6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %eq6 -> tensor<64x1024x2048xf16> {
                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                  scf.yield %w : tensor<64x1024x2048xf16>
                } else {
                  %eq7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %eq7 -> tensor<64x1024x2048xf16> {
                    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                    scf.yield %w : tensor<64x1024x2048xf16>
                  } else {
                    %eq8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %eq8 -> tensor<64x1024x2048xf16> {
                      %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                      scf.yield %w : tensor<64x1024x2048xf16>
                    } else {
                      %eq9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %eq9 -> tensor<64x1024x2048xf16> {
                        %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                        scf.yield %w : tensor<64x1024x2048xf16>
                      } else {
                        %eq10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %eq10 -> tensor<64x1024x2048xf16> {
                          %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                          scf.yield %w : tensor<64x1024x2048xf16>
                        } else {
                          %eq11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %eq11 -> tensor<64x1024x2048xf16> {
                            %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                            scf.yield %w : tensor<64x1024x2048xf16>
                          } else {
                            %eq12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %eq12 -> tensor<64x1024x2048xf16> {
                              %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                              scf.yield %w : tensor<64x1024x2048xf16>
                            } else {
                              %eq13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %eq13 -> tensor<64x1024x2048xf16> {
                                %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                                scf.yield %w : tensor<64x1024x2048xf16>
                              } else {
                                %eq14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %eq14 -> tensor<64x1024x2048xf16> {
                                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                                  scf.yield %w : tensor<64x1024x2048xf16>
                                } else {
                                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.ffn_gate_exps.weight"> : tensor<64x1024x2048xf16>
                                  scf.yield %w : tensor<64x1024x2048xf16>
                                }
                                scf.yield %r14 : tensor<64x1024x2048xf16>
                              }
                              scf.yield %r13 : tensor<64x1024x2048xf16>
                            }
                            scf.yield %r12 : tensor<64x1024x2048xf16>
                          }
                          scf.yield %r11 : tensor<64x1024x2048xf16>
                        }
                        scf.yield %r10 : tensor<64x1024x2048xf16>
                      }
                      scf.yield %r9 : tensor<64x1024x2048xf16>
                    }
                    scf.yield %r8 : tensor<64x1024x2048xf16>
                  }
                  scf.yield %r7 : tensor<64x1024x2048xf16>
                }
                scf.yield %r6 : tensor<64x1024x2048xf16>
              }
              scf.yield %r5 : tensor<64x1024x2048xf16>
            }
            scf.yield %r4 : tensor<64x1024x2048xf16>
          }
          scf.yield %r3 : tensor<64x1024x2048xf16>
        }
        scf.yield %r2 : tensor<64x1024x2048xf16>
      }
      scf.yield %r1 : tensor<64x1024x2048xf16>
    }
    %dyn = tensor.cast %r : tensor<64x1024x2048xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

  // ffn_down_exps_weight: per-layer tensor<64x2048x1024xf16>
  util.func public @ffn_down_exps_weight(%layer: i32) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c9 = arith.constant 9 : i32
    %c10 = arith.constant 10 : i32
    %c11 = arith.constant 11 : i32
    %c12 = arith.constant 12 : i32
    %c13 = arith.constant 13 : i32
    %c14 = arith.constant 14 : i32
    %eq0 = arith.cmpi eq, %layer, %c0 : i32
    %r = scf.if %eq0 -> tensor<64x2048x1024xf16> {
      %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
      scf.yield %w : tensor<64x2048x1024xf16>
    } else {
      %eq1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %eq1 -> tensor<64x2048x1024xf16> {
        %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
        scf.yield %w : tensor<64x2048x1024xf16>
      } else {
        %eq2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %eq2 -> tensor<64x2048x1024xf16> {
          %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
          scf.yield %w : tensor<64x2048x1024xf16>
        } else {
          %eq3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %eq3 -> tensor<64x2048x1024xf16> {
            %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
            scf.yield %w : tensor<64x2048x1024xf16>
          } else {
            %eq4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %eq4 -> tensor<64x2048x1024xf16> {
              %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
              scf.yield %w : tensor<64x2048x1024xf16>
            } else {
              %eq5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %eq5 -> tensor<64x2048x1024xf16> {
                %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                scf.yield %w : tensor<64x2048x1024xf16>
              } else {
                %eq6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %eq6 -> tensor<64x2048x1024xf16> {
                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                  scf.yield %w : tensor<64x2048x1024xf16>
                } else {
                  %eq7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %eq7 -> tensor<64x2048x1024xf16> {
                    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                    scf.yield %w : tensor<64x2048x1024xf16>
                  } else {
                    %eq8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %eq8 -> tensor<64x2048x1024xf16> {
                      %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                      scf.yield %w : tensor<64x2048x1024xf16>
                    } else {
                      %eq9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %eq9 -> tensor<64x2048x1024xf16> {
                        %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                        scf.yield %w : tensor<64x2048x1024xf16>
                      } else {
                        %eq10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %eq10 -> tensor<64x2048x1024xf16> {
                          %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                          scf.yield %w : tensor<64x2048x1024xf16>
                        } else {
                          %eq11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %eq11 -> tensor<64x2048x1024xf16> {
                            %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                            scf.yield %w : tensor<64x2048x1024xf16>
                          } else {
                            %eq12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %eq12 -> tensor<64x2048x1024xf16> {
                              %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                              scf.yield %w : tensor<64x2048x1024xf16>
                            } else {
                              %eq13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %eq13 -> tensor<64x2048x1024xf16> {
                                %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                                scf.yield %w : tensor<64x2048x1024xf16>
                              } else {
                                %eq14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %eq14 -> tensor<64x2048x1024xf16> {
                                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                                  scf.yield %w : tensor<64x2048x1024xf16>
                                } else {
                                  %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.ffn_down_exps.weight"> : tensor<64x2048x1024xf16>
                                  scf.yield %w : tensor<64x2048x1024xf16>
                                }
                                scf.yield %r14 : tensor<64x2048x1024xf16>
                              }
                              scf.yield %r13 : tensor<64x2048x1024xf16>
                            }
                            scf.yield %r12 : tensor<64x2048x1024xf16>
                          }
                          scf.yield %r11 : tensor<64x2048x1024xf16>
                        }
                        scf.yield %r10 : tensor<64x2048x1024xf16>
                      }
                      scf.yield %r9 : tensor<64x2048x1024xf16>
                    }
                    scf.yield %r8 : tensor<64x2048x1024xf16>
                  }
                  scf.yield %r7 : tensor<64x2048x1024xf16>
                }
                scf.yield %r6 : tensor<64x2048x1024xf16>
              }
              scf.yield %r5 : tensor<64x2048x1024xf16>
            }
            scf.yield %r4 : tensor<64x2048x1024xf16>
          }
          scf.yield %r3 : tensor<64x2048x1024xf16>
        }
        scf.yield %r2 : tensor<64x2048x1024xf16>
      }
      scf.yield %r1 : tensor<64x2048x1024xf16>
    }
    %dyn = tensor.cast %r : tensor<64x2048x1024xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

}
