#!/bin/bash
# Compile Qwen3-0.6B MLIR to VMFB for CUDA (Jetson AGX Orin, SM_87)
#
# Usage:
#   ./compile_qwen3.sh                    # default (O0)
#   ./compile_qwen3.sh --optimized        # O0 + safe optimization flags
#   ./compile_qwen3.sh --cpu              # compile for CPU
#
# NOTE: --iree-opt-level=O1+ breaks on this model's tensor.concat with
# dynamic shapes. The default is O0. We enable individual safe flags instead.

set -e

IREE_COMPILE=~/iree/build/tools/iree-compile
MLIR=~/frank-models-hetero/architectures/llm/qwen3/qwen3_full.mlir
OUTPUT=/tmp/qwen3_run_SINGLE.vmfb

MODE="${1:---default}"

CUDA_FLAGS="--iree-hal-target-device=cuda \
  --iree-cuda-target=sm_87 \
  --iree-cuda-target-features=+ptx74 \
  --iree-hal-indirect-command-buffers=false"

case "$MODE" in
  --default)
    echo "=== Compiling Qwen3 (default, O0) ==="
    $IREE_COMPILE "$MLIR" $CUDA_FLAGS -o "$OUTPUT"
    ;;

  --optimized|--opt)
    echo "=== Compiling Qwen3 (O0 + safe optimizations) ==="
    # NOTE: O1+ and aggressive-fusion break on tensor.concat with dynamic shapes
    $IREE_COMPILE "$MLIR" $CUDA_FLAGS \
      --iree-opt-const-expr-hoisting \
      --iree-dispatch-creation-element-wise-fuse-multi-reduction \
      -o "$OUTPUT"
    ;;

  --cpu)
    echo "=== Compiling Qwen3 (CPU, llvm-cpu) ==="
    OUTPUT="${OUTPUT%.vmfb}_cpu.vmfb"
    $IREE_COMPILE "$MLIR" \
      --iree-hal-target-backends=llvm-cpu \
      --iree-llvmcpu-target-cpu=cortex-a78ae \
      --iree-llvmcpu-fail-on-large-vector=false \
      -o "$OUTPUT"
    ;;

  *)
    echo "Usage: $0 [--default|--optimized|--cpu]"
    exit 1
    ;;
esac

echo ""
ls -lh "$OUTPUT"
echo ""
echo "Run:"
echo "  cd ~/frank-models-hetero && echo 'Hello' | PYTHONPATH=~/iree-python-runtime:~/iree-python-compiler python3.11 scripts/chat_qwen3_run.py"
