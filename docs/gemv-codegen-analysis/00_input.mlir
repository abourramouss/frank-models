module @bench {
  util.func public @gemv_f16(%x: tensor<1x1024xf16>, %w: tensor<1024x2048xf16>) -> tensor<1x2048xf16> {
    %cst = arith.constant 0.0 : f16
    %init = tensor.empty() : tensor<1x2048xf16>
    %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<1x2048xf16>) -> tensor<1x2048xf16>
    %r = linalg.matmul ins(%x, %w : tensor<1x1024xf16>, tensor<1024x2048xf16>) outs(%fill : tensor<1x2048xf16>) -> tensor<1x2048xf16>
    util.return %r : tensor<1x2048xf16>
  }
}
