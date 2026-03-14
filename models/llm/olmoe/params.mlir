// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// OLMoE-1B-7B parameter accessors.
//
// Uses flow.tensor.constant with #flow.parameter.named to load parameters
// from an IREE parameter archive at runtime.
//
// Parameter naming follows GGUF convention:
//   Model-level: token_embd.weight, output_norm.weight, output.weight
//   Layer-level: blk.{idx}.{name} (e.g., blk.0.attn_q.weight)
//
// Shapes (OLMoE-1B-7B):
//   vocab_size=50304, n_embd=2048, n_head=16, n_head_kv=16, head_dim=128
//   n_ff=1024, n_expert=64, n_layers=16

module @model_params {

  // ===== Model-level parameters =====

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

  // ===== Layer-level parameters =====
  // Dispatch to correct layer using nested scf.if.

  // --- Normalization weights ---

  util.func public @attn_norm_weight(%layer: i32) -> tensor<?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_norm.weight"> : tensor<2048xf16>
      scf.yield %t : tensor<2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_norm.weight"> : tensor<2048xf16>
          scf.yield %t : tensor<2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_norm.weight"> : tensor<2048xf16>
            scf.yield %t : tensor<2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_norm.weight"> : tensor<2048xf16>
              scf.yield %t : tensor<2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_norm.weight"> : tensor<2048xf16>
                scf.yield %t : tensor<2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_norm.weight"> : tensor<2048xf16>
                  scf.yield %t : tensor<2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_norm.weight"> : tensor<2048xf16>
                    scf.yield %t : tensor<2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_norm.weight"> : tensor<2048xf16>
                      scf.yield %t : tensor<2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_norm.weight"> : tensor<2048xf16>
                        scf.yield %t : tensor<2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_norm.weight"> : tensor<2048xf16>
                          scf.yield %t : tensor<2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_norm.weight"> : tensor<2048xf16>
                            scf.yield %t : tensor<2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_norm.weight"> : tensor<2048xf16>
                              scf.yield %t : tensor<2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_norm.weight"> : tensor<2048xf16>
                                scf.yield %t : tensor<2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_norm.weight"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_norm.weight"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                }
                                scf.yield %r14 : tensor<2048xf16>
                              }
                              scf.yield %r13 : tensor<2048xf16>
                            }
                            scf.yield %r12 : tensor<2048xf16>
                          }
                          scf.yield %r11 : tensor<2048xf16>
                        }
                        scf.yield %r10 : tensor<2048xf16>
                      }
                      scf.yield %r9 : tensor<2048xf16>
                    }
                    scf.yield %r8 : tensor<2048xf16>
                  }
                  scf.yield %r7 : tensor<2048xf16>
                }
                scf.yield %r6 : tensor<2048xf16>
              }
              scf.yield %r5 : tensor<2048xf16>
            }
            scf.yield %r4 : tensor<2048xf16>
          }
          scf.yield %r3 : tensor<2048xf16>
        }
        scf.yield %r2 : tensor<2048xf16>
      }
      scf.yield %r1 : tensor<2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @ffn_norm_weight(%layer: i32) -> tensor<?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_norm.weight"> : tensor<2048xf16>
      scf.yield %t : tensor<2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.ffn_norm.weight"> : tensor<2048xf16>
          scf.yield %t : tensor<2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.ffn_norm.weight"> : tensor<2048xf16>
            scf.yield %t : tensor<2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.ffn_norm.weight"> : tensor<2048xf16>
              scf.yield %t : tensor<2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.ffn_norm.weight"> : tensor<2048xf16>
                scf.yield %t : tensor<2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.ffn_norm.weight"> : tensor<2048xf16>
                  scf.yield %t : tensor<2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.ffn_norm.weight"> : tensor<2048xf16>
                    scf.yield %t : tensor<2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.ffn_norm.weight"> : tensor<2048xf16>
                      scf.yield %t : tensor<2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.ffn_norm.weight"> : tensor<2048xf16>
                        scf.yield %t : tensor<2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.ffn_norm.weight"> : tensor<2048xf16>
                          scf.yield %t : tensor<2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.ffn_norm.weight"> : tensor<2048xf16>
                            scf.yield %t : tensor<2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.ffn_norm.weight"> : tensor<2048xf16>
                              scf.yield %t : tensor<2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.ffn_norm.weight"> : tensor<2048xf16>
                                scf.yield %t : tensor<2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.ffn_norm.weight"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.ffn_norm.weight"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                }
                                scf.yield %r14 : tensor<2048xf16>
                              }
                              scf.yield %r13 : tensor<2048xf16>
                            }
                            scf.yield %r12 : tensor<2048xf16>
                          }
                          scf.yield %r11 : tensor<2048xf16>
                        }
                        scf.yield %r10 : tensor<2048xf16>
                      }
                      scf.yield %r9 : tensor<2048xf16>
                    }
                    scf.yield %r8 : tensor<2048xf16>
                  }
                  scf.yield %r7 : tensor<2048xf16>
                }
                scf.yield %r6 : tensor<2048xf16>
              }
              scf.yield %r5 : tensor<2048xf16>
            }
            scf.yield %r4 : tensor<2048xf16>
          }
          scf.yield %r3 : tensor<2048xf16>
        }
        scf.yield %r2 : tensor<2048xf16>
      }
      scf.yield %r1 : tensor<2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  // --- Attention projection weights ---

  util.func public @attn_q_weight(%layer: i32) -> tensor<?x?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048x2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_q.weight"> : tensor<2048x2048xf16>
      scf.yield %t : tensor<2048x2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048x2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_q.weight"> : tensor<2048x2048xf16>
        scf.yield %t : tensor<2048x2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048x2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_q.weight"> : tensor<2048x2048xf16>
          scf.yield %t : tensor<2048x2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048x2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_q.weight"> : tensor<2048x2048xf16>
            scf.yield %t : tensor<2048x2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048x2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_q.weight"> : tensor<2048x2048xf16>
              scf.yield %t : tensor<2048x2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048x2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_q.weight"> : tensor<2048x2048xf16>
                scf.yield %t : tensor<2048x2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048x2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_q.weight"> : tensor<2048x2048xf16>
                  scf.yield %t : tensor<2048x2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048x2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_q.weight"> : tensor<2048x2048xf16>
                    scf.yield %t : tensor<2048x2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048x2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_q.weight"> : tensor<2048x2048xf16>
                      scf.yield %t : tensor<2048x2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048x2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_q.weight"> : tensor<2048x2048xf16>
                        scf.yield %t : tensor<2048x2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048x2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_q.weight"> : tensor<2048x2048xf16>
                          scf.yield %t : tensor<2048x2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048x2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_q.weight"> : tensor<2048x2048xf16>
                            scf.yield %t : tensor<2048x2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048x2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_q.weight"> : tensor<2048x2048xf16>
                              scf.yield %t : tensor<2048x2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048x2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_q.weight"> : tensor<2048x2048xf16>
                                scf.yield %t : tensor<2048x2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048x2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_q.weight"> : tensor<2048x2048xf16>
                                  scf.yield %t : tensor<2048x2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_q.weight"> : tensor<2048x2048xf16>
                                  scf.yield %t : tensor<2048x2048xf16>
                                }
                                scf.yield %r14 : tensor<2048x2048xf16>
                              }
                              scf.yield %r13 : tensor<2048x2048xf16>
                            }
                            scf.yield %r12 : tensor<2048x2048xf16>
                          }
                          scf.yield %r11 : tensor<2048x2048xf16>
                        }
                        scf.yield %r10 : tensor<2048x2048xf16>
                      }
                      scf.yield %r9 : tensor<2048x2048xf16>
                    }
                    scf.yield %r8 : tensor<2048x2048xf16>
                  }
                  scf.yield %r7 : tensor<2048x2048xf16>
                }
                scf.yield %r6 : tensor<2048x2048xf16>
              }
              scf.yield %r5 : tensor<2048x2048xf16>
            }
            scf.yield %r4 : tensor<2048x2048xf16>
          }
          scf.yield %r3 : tensor<2048x2048xf16>
        }
        scf.yield %r2 : tensor<2048x2048xf16>
      }
      scf.yield %r1 : tensor<2048x2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  util.func public @attn_k_weight(%layer: i32) -> tensor<?x?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048x2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_k.weight"> : tensor<2048x2048xf16>
      scf.yield %t : tensor<2048x2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048x2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_k.weight"> : tensor<2048x2048xf16>
        scf.yield %t : tensor<2048x2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048x2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_k.weight"> : tensor<2048x2048xf16>
          scf.yield %t : tensor<2048x2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048x2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_k.weight"> : tensor<2048x2048xf16>
            scf.yield %t : tensor<2048x2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048x2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_k.weight"> : tensor<2048x2048xf16>
              scf.yield %t : tensor<2048x2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048x2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_k.weight"> : tensor<2048x2048xf16>
                scf.yield %t : tensor<2048x2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048x2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_k.weight"> : tensor<2048x2048xf16>
                  scf.yield %t : tensor<2048x2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048x2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_k.weight"> : tensor<2048x2048xf16>
                    scf.yield %t : tensor<2048x2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048x2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_k.weight"> : tensor<2048x2048xf16>
                      scf.yield %t : tensor<2048x2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048x2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_k.weight"> : tensor<2048x2048xf16>
                        scf.yield %t : tensor<2048x2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048x2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_k.weight"> : tensor<2048x2048xf16>
                          scf.yield %t : tensor<2048x2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048x2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_k.weight"> : tensor<2048x2048xf16>
                            scf.yield %t : tensor<2048x2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048x2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_k.weight"> : tensor<2048x2048xf16>
                              scf.yield %t : tensor<2048x2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048x2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_k.weight"> : tensor<2048x2048xf16>
                                scf.yield %t : tensor<2048x2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048x2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_k.weight"> : tensor<2048x2048xf16>
                                  scf.yield %t : tensor<2048x2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_k.weight"> : tensor<2048x2048xf16>
                                  scf.yield %t : tensor<2048x2048xf16>
                                }
                                scf.yield %r14 : tensor<2048x2048xf16>
                              }
                              scf.yield %r13 : tensor<2048x2048xf16>
                            }
                            scf.yield %r12 : tensor<2048x2048xf16>
                          }
                          scf.yield %r11 : tensor<2048x2048xf16>
                        }
                        scf.yield %r10 : tensor<2048x2048xf16>
                      }
                      scf.yield %r9 : tensor<2048x2048xf16>
                    }
                    scf.yield %r8 : tensor<2048x2048xf16>
                  }
                  scf.yield %r7 : tensor<2048x2048xf16>
                }
                scf.yield %r6 : tensor<2048x2048xf16>
              }
              scf.yield %r5 : tensor<2048x2048xf16>
            }
            scf.yield %r4 : tensor<2048x2048xf16>
          }
          scf.yield %r3 : tensor<2048x2048xf16>
        }
        scf.yield %r2 : tensor<2048x2048xf16>
      }
      scf.yield %r1 : tensor<2048x2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  util.func public @attn_v_weight(%layer: i32) -> tensor<?x?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048x2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_v.weight"> : tensor<2048x2048xf16>
      scf.yield %t : tensor<2048x2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048x2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_v.weight"> : tensor<2048x2048xf16>
        scf.yield %t : tensor<2048x2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048x2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_v.weight"> : tensor<2048x2048xf16>
          scf.yield %t : tensor<2048x2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048x2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_v.weight"> : tensor<2048x2048xf16>
            scf.yield %t : tensor<2048x2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048x2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_v.weight"> : tensor<2048x2048xf16>
              scf.yield %t : tensor<2048x2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048x2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_v.weight"> : tensor<2048x2048xf16>
                scf.yield %t : tensor<2048x2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048x2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_v.weight"> : tensor<2048x2048xf16>
                  scf.yield %t : tensor<2048x2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048x2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_v.weight"> : tensor<2048x2048xf16>
                    scf.yield %t : tensor<2048x2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048x2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_v.weight"> : tensor<2048x2048xf16>
                      scf.yield %t : tensor<2048x2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048x2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_v.weight"> : tensor<2048x2048xf16>
                        scf.yield %t : tensor<2048x2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048x2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_v.weight"> : tensor<2048x2048xf16>
                          scf.yield %t : tensor<2048x2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048x2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_v.weight"> : tensor<2048x2048xf16>
                            scf.yield %t : tensor<2048x2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048x2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_v.weight"> : tensor<2048x2048xf16>
                              scf.yield %t : tensor<2048x2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048x2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_v.weight"> : tensor<2048x2048xf16>
                                scf.yield %t : tensor<2048x2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048x2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_v.weight"> : tensor<2048x2048xf16>
                                  scf.yield %t : tensor<2048x2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_v.weight"> : tensor<2048x2048xf16>
                                  scf.yield %t : tensor<2048x2048xf16>
                                }
                                scf.yield %r14 : tensor<2048x2048xf16>
                              }
                              scf.yield %r13 : tensor<2048x2048xf16>
                            }
                            scf.yield %r12 : tensor<2048x2048xf16>
                          }
                          scf.yield %r11 : tensor<2048x2048xf16>
                        }
                        scf.yield %r10 : tensor<2048x2048xf16>
                      }
                      scf.yield %r9 : tensor<2048x2048xf16>
                    }
                    scf.yield %r8 : tensor<2048x2048xf16>
                  }
                  scf.yield %r7 : tensor<2048x2048xf16>
                }
                scf.yield %r6 : tensor<2048x2048xf16>
              }
              scf.yield %r5 : tensor<2048x2048xf16>
            }
            scf.yield %r4 : tensor<2048x2048xf16>
          }
          scf.yield %r3 : tensor<2048x2048xf16>
        }
        scf.yield %r2 : tensor<2048x2048xf16>
      }
      scf.yield %r1 : tensor<2048x2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  util.func public @attn_output_weight(%layer: i32) -> tensor<?x?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048x2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_output.weight"> : tensor<2048x2048xf16>
      scf.yield %t : tensor<2048x2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048x2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_output.weight"> : tensor<2048x2048xf16>
        scf.yield %t : tensor<2048x2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048x2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_output.weight"> : tensor<2048x2048xf16>
          scf.yield %t : tensor<2048x2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048x2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_output.weight"> : tensor<2048x2048xf16>
            scf.yield %t : tensor<2048x2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048x2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_output.weight"> : tensor<2048x2048xf16>
              scf.yield %t : tensor<2048x2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048x2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_output.weight"> : tensor<2048x2048xf16>
                scf.yield %t : tensor<2048x2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048x2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_output.weight"> : tensor<2048x2048xf16>
                  scf.yield %t : tensor<2048x2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048x2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_output.weight"> : tensor<2048x2048xf16>
                    scf.yield %t : tensor<2048x2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048x2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_output.weight"> : tensor<2048x2048xf16>
                      scf.yield %t : tensor<2048x2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048x2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_output.weight"> : tensor<2048x2048xf16>
                        scf.yield %t : tensor<2048x2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048x2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_output.weight"> : tensor<2048x2048xf16>
                          scf.yield %t : tensor<2048x2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048x2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_output.weight"> : tensor<2048x2048xf16>
                            scf.yield %t : tensor<2048x2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048x2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_output.weight"> : tensor<2048x2048xf16>
                              scf.yield %t : tensor<2048x2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048x2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_output.weight"> : tensor<2048x2048xf16>
                                scf.yield %t : tensor<2048x2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048x2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_output.weight"> : tensor<2048x2048xf16>
                                  scf.yield %t : tensor<2048x2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_output.weight"> : tensor<2048x2048xf16>
                                  scf.yield %t : tensor<2048x2048xf16>
                                }
                                scf.yield %r14 : tensor<2048x2048xf16>
                              }
                              scf.yield %r13 : tensor<2048x2048xf16>
                            }
                            scf.yield %r12 : tensor<2048x2048xf16>
                          }
                          scf.yield %r11 : tensor<2048x2048xf16>
                        }
                        scf.yield %r10 : tensor<2048x2048xf16>
                      }
                      scf.yield %r9 : tensor<2048x2048xf16>
                    }
                    scf.yield %r8 : tensor<2048x2048xf16>
                  }
                  scf.yield %r7 : tensor<2048x2048xf16>
                }
                scf.yield %r6 : tensor<2048x2048xf16>
              }
              scf.yield %r5 : tensor<2048x2048xf16>
            }
            scf.yield %r4 : tensor<2048x2048xf16>
          }
          scf.yield %r3 : tensor<2048x2048xf16>
        }
        scf.yield %r2 : tensor<2048x2048xf16>
      }
      scf.yield %r1 : tensor<2048x2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

  // --- Attention biases (zeros since use_bias=false, but interface requires them) ---

  util.func public @attn_q_bias(%layer: i32) -> tensor<?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_q.bias"> : tensor<2048xf16>
      scf.yield %t : tensor<2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_q.bias"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_q.bias"> : tensor<2048xf16>
          scf.yield %t : tensor<2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_q.bias"> : tensor<2048xf16>
            scf.yield %t : tensor<2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_q.bias"> : tensor<2048xf16>
              scf.yield %t : tensor<2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_q.bias"> : tensor<2048xf16>
                scf.yield %t : tensor<2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_q.bias"> : tensor<2048xf16>
                  scf.yield %t : tensor<2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_q.bias"> : tensor<2048xf16>
                    scf.yield %t : tensor<2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_q.bias"> : tensor<2048xf16>
                      scf.yield %t : tensor<2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_q.bias"> : tensor<2048xf16>
                        scf.yield %t : tensor<2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_q.bias"> : tensor<2048xf16>
                          scf.yield %t : tensor<2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_q.bias"> : tensor<2048xf16>
                            scf.yield %t : tensor<2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_q.bias"> : tensor<2048xf16>
                              scf.yield %t : tensor<2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_q.bias"> : tensor<2048xf16>
                                scf.yield %t : tensor<2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_q.bias"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_q.bias"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                }
                                scf.yield %r14 : tensor<2048xf16>
                              }
                              scf.yield %r13 : tensor<2048xf16>
                            }
                            scf.yield %r12 : tensor<2048xf16>
                          }
                          scf.yield %r11 : tensor<2048xf16>
                        }
                        scf.yield %r10 : tensor<2048xf16>
                      }
                      scf.yield %r9 : tensor<2048xf16>
                    }
                    scf.yield %r8 : tensor<2048xf16>
                  }
                  scf.yield %r7 : tensor<2048xf16>
                }
                scf.yield %r6 : tensor<2048xf16>
              }
              scf.yield %r5 : tensor<2048xf16>
            }
            scf.yield %r4 : tensor<2048xf16>
          }
          scf.yield %r3 : tensor<2048xf16>
        }
        scf.yield %r2 : tensor<2048xf16>
      }
      scf.yield %r1 : tensor<2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_k_bias(%layer: i32) -> tensor<?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_k.bias"> : tensor<2048xf16>
      scf.yield %t : tensor<2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_k.bias"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_k.bias"> : tensor<2048xf16>
          scf.yield %t : tensor<2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_k.bias"> : tensor<2048xf16>
            scf.yield %t : tensor<2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_k.bias"> : tensor<2048xf16>
              scf.yield %t : tensor<2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_k.bias"> : tensor<2048xf16>
                scf.yield %t : tensor<2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_k.bias"> : tensor<2048xf16>
                  scf.yield %t : tensor<2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_k.bias"> : tensor<2048xf16>
                    scf.yield %t : tensor<2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_k.bias"> : tensor<2048xf16>
                      scf.yield %t : tensor<2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_k.bias"> : tensor<2048xf16>
                        scf.yield %t : tensor<2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_k.bias"> : tensor<2048xf16>
                          scf.yield %t : tensor<2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_k.bias"> : tensor<2048xf16>
                            scf.yield %t : tensor<2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_k.bias"> : tensor<2048xf16>
                              scf.yield %t : tensor<2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_k.bias"> : tensor<2048xf16>
                                scf.yield %t : tensor<2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_k.bias"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_k.bias"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                }
                                scf.yield %r14 : tensor<2048xf16>
                              }
                              scf.yield %r13 : tensor<2048xf16>
                            }
                            scf.yield %r12 : tensor<2048xf16>
                          }
                          scf.yield %r11 : tensor<2048xf16>
                        }
                        scf.yield %r10 : tensor<2048xf16>
                      }
                      scf.yield %r9 : tensor<2048xf16>
                    }
                    scf.yield %r8 : tensor<2048xf16>
                  }
                  scf.yield %r7 : tensor<2048xf16>
                }
                scf.yield %r6 : tensor<2048xf16>
              }
              scf.yield %r5 : tensor<2048xf16>
            }
            scf.yield %r4 : tensor<2048xf16>
          }
          scf.yield %r3 : tensor<2048xf16>
        }
        scf.yield %r2 : tensor<2048xf16>
      }
      scf.yield %r1 : tensor<2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_v_bias(%layer: i32) -> tensor<?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_v.bias"> : tensor<2048xf16>
      scf.yield %t : tensor<2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_v.bias"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_v.bias"> : tensor<2048xf16>
          scf.yield %t : tensor<2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_v.bias"> : tensor<2048xf16>
            scf.yield %t : tensor<2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_v.bias"> : tensor<2048xf16>
              scf.yield %t : tensor<2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_v.bias"> : tensor<2048xf16>
                scf.yield %t : tensor<2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_v.bias"> : tensor<2048xf16>
                  scf.yield %t : tensor<2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_v.bias"> : tensor<2048xf16>
                    scf.yield %t : tensor<2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_v.bias"> : tensor<2048xf16>
                      scf.yield %t : tensor<2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_v.bias"> : tensor<2048xf16>
                        scf.yield %t : tensor<2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_v.bias"> : tensor<2048xf16>
                          scf.yield %t : tensor<2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_v.bias"> : tensor<2048xf16>
                            scf.yield %t : tensor<2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_v.bias"> : tensor<2048xf16>
                              scf.yield %t : tensor<2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_v.bias"> : tensor<2048xf16>
                                scf.yield %t : tensor<2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_v.bias"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_v.bias"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                }
                                scf.yield %r14 : tensor<2048xf16>
                              }
                              scf.yield %r13 : tensor<2048xf16>
                            }
                            scf.yield %r12 : tensor<2048xf16>
                          }
                          scf.yield %r11 : tensor<2048xf16>
                        }
                        scf.yield %r10 : tensor<2048xf16>
                      }
                      scf.yield %r9 : tensor<2048xf16>
                    }
                    scf.yield %r8 : tensor<2048xf16>
                  }
                  scf.yield %r7 : tensor<2048xf16>
                }
                scf.yield %r6 : tensor<2048xf16>
              }
              scf.yield %r5 : tensor<2048xf16>
            }
            scf.yield %r4 : tensor<2048xf16>
          }
          scf.yield %r3 : tensor<2048xf16>
        }
        scf.yield %r2 : tensor<2048xf16>
      }
      scf.yield %r1 : tensor<2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_output_bias(%layer: i32) -> tensor<?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_output.bias"> : tensor<2048xf16>
      scf.yield %t : tensor<2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_output.bias"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_output.bias"> : tensor<2048xf16>
          scf.yield %t : tensor<2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_output.bias"> : tensor<2048xf16>
            scf.yield %t : tensor<2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_output.bias"> : tensor<2048xf16>
              scf.yield %t : tensor<2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_output.bias"> : tensor<2048xf16>
                scf.yield %t : tensor<2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_output.bias"> : tensor<2048xf16>
                  scf.yield %t : tensor<2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_output.bias"> : tensor<2048xf16>
                    scf.yield %t : tensor<2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_output.bias"> : tensor<2048xf16>
                      scf.yield %t : tensor<2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_output.bias"> : tensor<2048xf16>
                        scf.yield %t : tensor<2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_output.bias"> : tensor<2048xf16>
                          scf.yield %t : tensor<2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_output.bias"> : tensor<2048xf16>
                            scf.yield %t : tensor<2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_output.bias"> : tensor<2048xf16>
                              scf.yield %t : tensor<2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_output.bias"> : tensor<2048xf16>
                                scf.yield %t : tensor<2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_output.bias"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_output.bias"> : tensor<2048xf16>
                                  scf.yield %t : tensor<2048xf16>
                                }
                                scf.yield %r14 : tensor<2048xf16>
                              }
                              scf.yield %r13 : tensor<2048xf16>
                            }
                            scf.yield %r12 : tensor<2048xf16>
                          }
                          scf.yield %r11 : tensor<2048xf16>
                        }
                        scf.yield %r10 : tensor<2048xf16>
                      }
                      scf.yield %r9 : tensor<2048xf16>
                    }
                    scf.yield %r8 : tensor<2048xf16>
                  }
                  scf.yield %r7 : tensor<2048xf16>
                }
                scf.yield %r6 : tensor<2048xf16>
              }
              scf.yield %r5 : tensor<2048xf16>
            }
            scf.yield %r4 : tensor<2048xf16>
          }
          scf.yield %r3 : tensor<2048xf16>
        }
        scf.yield %r2 : tensor<2048xf16>
      }
      scf.yield %r1 : tensor<2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  // --- MoE weights ---

  util.func public @ffn_gate_inp_weight(%layer: i32) -> tensor<?x?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<64x2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_gate_inp.weight"> : tensor<64x2048xf16>
      scf.yield %t : tensor<64x2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<64x2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_gate_inp.weight"> : tensor<64x2048xf16>
        scf.yield %t : tensor<64x2048xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<64x2048xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.ffn_gate_inp.weight"> : tensor<64x2048xf16>
          scf.yield %t : tensor<64x2048xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<64x2048xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.ffn_gate_inp.weight"> : tensor<64x2048xf16>
            scf.yield %t : tensor<64x2048xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<64x2048xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.ffn_gate_inp.weight"> : tensor<64x2048xf16>
              scf.yield %t : tensor<64x2048xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<64x2048xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                scf.yield %t : tensor<64x2048xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<64x2048xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                  scf.yield %t : tensor<64x2048xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<64x2048xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                    scf.yield %t : tensor<64x2048xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<64x2048xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                      scf.yield %t : tensor<64x2048xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<64x2048xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                        scf.yield %t : tensor<64x2048xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<64x2048xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                          scf.yield %t : tensor<64x2048xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<64x2048xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                            scf.yield %t : tensor<64x2048xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<64x2048xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                              scf.yield %t : tensor<64x2048xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<64x2048xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                                scf.yield %t : tensor<64x2048xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<64x2048xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                                  scf.yield %t : tensor<64x2048xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.ffn_gate_inp.weight"> : tensor<64x2048xf16>
                                  scf.yield %t : tensor<64x2048xf16>
                                }
                                scf.yield %r14 : tensor<64x2048xf16>
                              }
                              scf.yield %r13 : tensor<64x2048xf16>
                            }
                            scf.yield %r12 : tensor<64x2048xf16>
                          }
                          scf.yield %r11 : tensor<64x2048xf16>
                        }
                        scf.yield %r10 : tensor<64x2048xf16>
                      }
                      scf.yield %r9 : tensor<64x2048xf16>
                    }
                    scf.yield %r8 : tensor<64x2048xf16>
                  }
                  scf.yield %r7 : tensor<64x2048xf16>
                }
                scf.yield %r6 : tensor<64x2048xf16>
              }
              scf.yield %r5 : tensor<64x2048xf16>
            }
            scf.yield %r4 : tensor<64x2048xf16>
          }
          scf.yield %r3 : tensor<64x2048xf16>
        }
        scf.yield %r2 : tensor<64x2048xf16>
      }
      scf.yield %r1 : tensor<64x2048xf16>
    }
    %dyn = tensor.cast %w : tensor<64x2048xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<1024x2048x64xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
      scf.yield %t : tensor<1024x2048x64xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<1024x2048x64xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
        scf.yield %t : tensor<1024x2048x64xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<1024x2048x64xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
          scf.yield %t : tensor<1024x2048x64xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<1024x2048x64xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
            scf.yield %t : tensor<1024x2048x64xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<1024x2048x64xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
              scf.yield %t : tensor<1024x2048x64xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<1024x2048x64xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                scf.yield %t : tensor<1024x2048x64xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<1024x2048x64xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                  scf.yield %t : tensor<1024x2048x64xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<1024x2048x64xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                    scf.yield %t : tensor<1024x2048x64xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<1024x2048x64xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                      scf.yield %t : tensor<1024x2048x64xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<1024x2048x64xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                        scf.yield %t : tensor<1024x2048x64xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<1024x2048x64xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                          scf.yield %t : tensor<1024x2048x64xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<1024x2048x64xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                            scf.yield %t : tensor<1024x2048x64xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<1024x2048x64xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                              scf.yield %t : tensor<1024x2048x64xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<1024x2048x64xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                                scf.yield %t : tensor<1024x2048x64xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<1024x2048x64xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                                  scf.yield %t : tensor<1024x2048x64xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.ffn_up_exps.weight"> : tensor<1024x2048x64xf16>
                                  scf.yield %t : tensor<1024x2048x64xf16>
                                }
                                scf.yield %r14 : tensor<1024x2048x64xf16>
                              }
                              scf.yield %r13 : tensor<1024x2048x64xf16>
                            }
                            scf.yield %r12 : tensor<1024x2048x64xf16>
                          }
                          scf.yield %r11 : tensor<1024x2048x64xf16>
                        }
                        scf.yield %r10 : tensor<1024x2048x64xf16>
                      }
                      scf.yield %r9 : tensor<1024x2048x64xf16>
                    }
                    scf.yield %r8 : tensor<1024x2048x64xf16>
                  }
                  scf.yield %r7 : tensor<1024x2048x64xf16>
                }
                scf.yield %r6 : tensor<1024x2048x64xf16>
              }
              scf.yield %r5 : tensor<1024x2048x64xf16>
            }
            scf.yield %r4 : tensor<1024x2048x64xf16>
          }
          scf.yield %r3 : tensor<1024x2048x64xf16>
        }
        scf.yield %r2 : tensor<1024x2048x64xf16>
      }
      scf.yield %r1 : tensor<1024x2048x64xf16>
    }
    %dyn = tensor.cast %w : tensor<1024x2048x64xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<1024x2048x64xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
      scf.yield %t : tensor<1024x2048x64xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<1024x2048x64xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
        scf.yield %t : tensor<1024x2048x64xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<1024x2048x64xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
          scf.yield %t : tensor<1024x2048x64xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<1024x2048x64xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
            scf.yield %t : tensor<1024x2048x64xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<1024x2048x64xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
              scf.yield %t : tensor<1024x2048x64xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<1024x2048x64xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                scf.yield %t : tensor<1024x2048x64xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<1024x2048x64xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                  scf.yield %t : tensor<1024x2048x64xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<1024x2048x64xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                    scf.yield %t : tensor<1024x2048x64xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<1024x2048x64xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                      scf.yield %t : tensor<1024x2048x64xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<1024x2048x64xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                        scf.yield %t : tensor<1024x2048x64xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<1024x2048x64xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                          scf.yield %t : tensor<1024x2048x64xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<1024x2048x64xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                            scf.yield %t : tensor<1024x2048x64xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<1024x2048x64xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                              scf.yield %t : tensor<1024x2048x64xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<1024x2048x64xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                                scf.yield %t : tensor<1024x2048x64xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<1024x2048x64xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                                  scf.yield %t : tensor<1024x2048x64xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.ffn_gate_exps.weight"> : tensor<1024x2048x64xf16>
                                  scf.yield %t : tensor<1024x2048x64xf16>
                                }
                                scf.yield %r14 : tensor<1024x2048x64xf16>
                              }
                              scf.yield %r13 : tensor<1024x2048x64xf16>
                            }
                            scf.yield %r12 : tensor<1024x2048x64xf16>
                          }
                          scf.yield %r11 : tensor<1024x2048x64xf16>
                        }
                        scf.yield %r10 : tensor<1024x2048x64xf16>
                      }
                      scf.yield %r9 : tensor<1024x2048x64xf16>
                    }
                    scf.yield %r8 : tensor<1024x2048x64xf16>
                  }
                  scf.yield %r7 : tensor<1024x2048x64xf16>
                }
                scf.yield %r6 : tensor<1024x2048x64xf16>
              }
              scf.yield %r5 : tensor<1024x2048x64xf16>
            }
            scf.yield %r4 : tensor<1024x2048x64xf16>
          }
          scf.yield %r3 : tensor<1024x2048x64xf16>
        }
        scf.yield %r2 : tensor<1024x2048x64xf16>
      }
      scf.yield %r1 : tensor<1024x2048x64xf16>
    }
    %dyn = tensor.cast %w : tensor<1024x2048x64xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048x1024x64xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
      scf.yield %t : tensor<2048x1024x64xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048x1024x64xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
        scf.yield %t : tensor<2048x1024x64xf16>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<2048x1024x64xf16>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
          scf.yield %t : tensor<2048x1024x64xf16>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<2048x1024x64xf16>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
            scf.yield %t : tensor<2048x1024x64xf16>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<2048x1024x64xf16>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
              scf.yield %t : tensor<2048x1024x64xf16>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<2048x1024x64xf16>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                scf.yield %t : tensor<2048x1024x64xf16>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<2048x1024x64xf16>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                  scf.yield %t : tensor<2048x1024x64xf16>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<2048x1024x64xf16>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                    scf.yield %t : tensor<2048x1024x64xf16>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<2048x1024x64xf16>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                      scf.yield %t : tensor<2048x1024x64xf16>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<2048x1024x64xf16>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                        scf.yield %t : tensor<2048x1024x64xf16>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<2048x1024x64xf16>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                          scf.yield %t : tensor<2048x1024x64xf16>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<2048x1024x64xf16>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                            scf.yield %t : tensor<2048x1024x64xf16>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<2048x1024x64xf16>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                              scf.yield %t : tensor<2048x1024x64xf16>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<2048x1024x64xf16>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                                scf.yield %t : tensor<2048x1024x64xf16>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<2048x1024x64xf16>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                                  scf.yield %t : tensor<2048x1024x64xf16>
                                } else {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.ffn_down_exps.weight"> : tensor<2048x1024x64xf16>
                                  scf.yield %t : tensor<2048x1024x64xf16>
                                }
                                scf.yield %r14 : tensor<2048x1024x64xf16>
                              }
                              scf.yield %r13 : tensor<2048x1024x64xf16>
                            }
                            scf.yield %r12 : tensor<2048x1024x64xf16>
                          }
                          scf.yield %r11 : tensor<2048x1024x64xf16>
                        }
                        scf.yield %r10 : tensor<2048x1024x64xf16>
                      }
                      scf.yield %r9 : tensor<2048x1024x64xf16>
                    }
                    scf.yield %r8 : tensor<2048x1024x64xf16>
                  }
                  scf.yield %r7 : tensor<2048x1024x64xf16>
                }
                scf.yield %r6 : tensor<2048x1024x64xf16>
              }
              scf.yield %r5 : tensor<2048x1024x64xf16>
            }
            scf.yield %r4 : tensor<2048x1024x64xf16>
          }
          scf.yield %r3 : tensor<2048x1024x64xf16>
        }
        scf.yield %r2 : tensor<2048x1024x64xf16>
      }
      scf.yield %r1 : tensor<2048x1024x64xf16>
    }
    %dyn = tensor.cast %w : tensor<2048x1024x64xf16> to tensor<?x?x?xf16>
    util.return %dyn : tensor<?x?x?xf16>
  }

  // --- QK norm weights ---

  util.func public @attn_q_norm_weight(%layer: i32) -> tensor<?xf16> {
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
    %c15 = arith.constant 15 : i32
    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_q_norm.weight"> : tensor<2048xf16>
      scf.yield %t : tensor<2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is2 = arith.cmpi eq, %layer, %c2 : i32
      %r2 = scf.if %is2 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is3 = arith.cmpi eq, %layer, %c3 : i32
      %r3 = scf.if %is3 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is4 = arith.cmpi eq, %layer, %c4 : i32
      %r4 = scf.if %is4 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is5 = arith.cmpi eq, %layer, %c5 : i32
      %r5 = scf.if %is5 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is6 = arith.cmpi eq, %layer, %c6 : i32
      %r6 = scf.if %is6 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is7 = arith.cmpi eq, %layer, %c7 : i32
      %r7 = scf.if %is7 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is8 = arith.cmpi eq, %layer, %c8 : i32
      %r8 = scf.if %is8 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is9 = arith.cmpi eq, %layer, %c9 : i32
      %r9 = scf.if %is9 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is10 = arith.cmpi eq, %layer, %c10 : i32
      %r10 = scf.if %is10 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is11 = arith.cmpi eq, %layer, %c11 : i32
      %r11 = scf.if %is11 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is12 = arith.cmpi eq, %layer, %c12 : i32
      %r12 = scf.if %is12 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is13 = arith.cmpi eq, %layer, %c13 : i32
      %r13 = scf.if %is13 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is14 = arith.cmpi eq, %layer, %c14 : i32
      %r14 = scf.if %is14 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_q_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      }
      scf.yield %r14 : tensor<2048xf16>
      }
      scf.yield %r13 : tensor<2048xf16>
      }
      scf.yield %r12 : tensor<2048xf16>
      }
      scf.yield %r11 : tensor<2048xf16>
      }
      scf.yield %r10 : tensor<2048xf16>
      }
      scf.yield %r9 : tensor<2048xf16>
      }
      scf.yield %r8 : tensor<2048xf16>
      }
      scf.yield %r7 : tensor<2048xf16>
      }
      scf.yield %r6 : tensor<2048xf16>
      }
      scf.yield %r5 : tensor<2048xf16>
      }
      scf.yield %r4 : tensor<2048xf16>
      }
      scf.yield %r3 : tensor<2048xf16>
      }
      scf.yield %r2 : tensor<2048xf16>
      }
      scf.yield %r1 : tensor<2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }

  util.func public @attn_k_norm_weight(%layer: i32) -> tensor<?xf16> {
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
    %c15 = arith.constant 15 : i32
    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_k_norm.weight"> : tensor<2048xf16>
      scf.yield %t : tensor<2048xf16>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is2 = arith.cmpi eq, %layer, %c2 : i32
      %r2 = scf.if %is2 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is3 = arith.cmpi eq, %layer, %c3 : i32
      %r3 = scf.if %is3 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is4 = arith.cmpi eq, %layer, %c4 : i32
      %r4 = scf.if %is4 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is5 = arith.cmpi eq, %layer, %c5 : i32
      %r5 = scf.if %is5 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is6 = arith.cmpi eq, %layer, %c6 : i32
      %r6 = scf.if %is6 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is7 = arith.cmpi eq, %layer, %c7 : i32
      %r7 = scf.if %is7 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is8 = arith.cmpi eq, %layer, %c8 : i32
      %r8 = scf.if %is8 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is9 = arith.cmpi eq, %layer, %c9 : i32
      %r9 = scf.if %is9 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is10 = arith.cmpi eq, %layer, %c10 : i32
      %r10 = scf.if %is10 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is11 = arith.cmpi eq, %layer, %c11 : i32
      %r11 = scf.if %is11 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is12 = arith.cmpi eq, %layer, %c12 : i32
      %r12 = scf.if %is12 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is13 = arith.cmpi eq, %layer, %c13 : i32
      %r13 = scf.if %is13 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
      %is14 = arith.cmpi eq, %layer, %c14 : i32
      %r14 = scf.if %is14 -> (tensor<2048xf16>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      } else {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_k_norm.weight"> : tensor<2048xf16>
        scf.yield %t : tensor<2048xf16>
      }
      scf.yield %r14 : tensor<2048xf16>
      }
      scf.yield %r13 : tensor<2048xf16>
      }
      scf.yield %r12 : tensor<2048xf16>
      }
      scf.yield %r11 : tensor<2048xf16>
      }
      scf.yield %r10 : tensor<2048xf16>
      }
      scf.yield %r9 : tensor<2048xf16>
      }
      scf.yield %r8 : tensor<2048xf16>
      }
      scf.yield %r7 : tensor<2048xf16>
      }
      scf.yield %r6 : tensor<2048xf16>
      }
      scf.yield %r5 : tensor<2048xf16>
      }
      scf.yield %r4 : tensor<2048xf16>
      }
      scf.yield %r3 : tensor<2048xf16>
      }
      scf.yield %r2 : tensor<2048xf16>
      }
      scf.yield %r1 : tensor<2048xf16>
    }
    %dyn = tensor.cast %w : tensor<2048xf16> to tensor<?xf16>
    util.return %dyn : tensor<?xf16>
  }


  util.func public @attn_qkv_weight(%layer: i32) -> tensor<?x?xf16> {
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
    %c15 = arith.constant 15 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is1 = arith.cmpi eq, %layer, %c1 : i32
    %r1 = scf.if %is1 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is2 = arith.cmpi eq, %layer, %c2 : i32
    %r2 = scf.if %is2 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is3 = arith.cmpi eq, %layer, %c3 : i32
    %r3 = scf.if %is3 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is4 = arith.cmpi eq, %layer, %c4 : i32
    %r4 = scf.if %is4 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is5 = arith.cmpi eq, %layer, %c5 : i32
    %r5 = scf.if %is5 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is6 = arith.cmpi eq, %layer, %c6 : i32
    %r6 = scf.if %is6 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is7 = arith.cmpi eq, %layer, %c7 : i32
    %r7 = scf.if %is7 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is8 = arith.cmpi eq, %layer, %c8 : i32
    %r8 = scf.if %is8 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is9 = arith.cmpi eq, %layer, %c9 : i32
    %r9 = scf.if %is9 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is10 = arith.cmpi eq, %layer, %c10 : i32
    %r10 = scf.if %is10 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is11 = arith.cmpi eq, %layer, %c11 : i32
    %r11 = scf.if %is11 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is12 = arith.cmpi eq, %layer, %c12 : i32
    %r12 = scf.if %is12 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is13 = arith.cmpi eq, %layer, %c13 : i32
    %r13 = scf.if %is13 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
    %is14 = arith.cmpi eq, %layer, %c14 : i32
    %r14 = scf.if %is14 -> (tensor<2048x6144xf16>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    } else {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_qkv.weight"> : tensor<2048x6144xf16>
      scf.yield %t : tensor<2048x6144xf16>
    }
    scf.yield %r14 : tensor<2048x6144xf16>
    }
    scf.yield %r13 : tensor<2048x6144xf16>
    }
    scf.yield %r12 : tensor<2048x6144xf16>
    }
    scf.yield %r11 : tensor<2048x6144xf16>
    }
    scf.yield %r10 : tensor<2048x6144xf16>
    }
    scf.yield %r9 : tensor<2048x6144xf16>
    }
    scf.yield %r8 : tensor<2048x6144xf16>
    }
    scf.yield %r7 : tensor<2048x6144xf16>
    }
    scf.yield %r6 : tensor<2048x6144xf16>
    }
    scf.yield %r5 : tensor<2048x6144xf16>
    }
    scf.yield %r4 : tensor<2048x6144xf16>
    }
    scf.yield %r3 : tensor<2048x6144xf16>
    }
    scf.yield %r2 : tensor<2048x6144xf16>
    }
    scf.yield %r1 : tensor<2048x6144xf16>
    }
    %dyn = tensor.cast %w : tensor<2048x6144xf16> to tensor<?x?xf16>
    util.return %dyn : tensor<?x?xf16>
  }

}
