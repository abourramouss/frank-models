#!/bin/bash
# Compile all CUDA kernels to PTX for sm_87 (Jetson AGX Orin)
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

NVCC_FLAGS="--ptx -arch=sm_87 -O2 --use_fast_math"

for cu_file in decode_dispatch_*.cu; do
    ptx_file="${cu_file%.cu}.ptx"
    echo "Compiling $cu_file -> $ptx_file"
    nvcc $NVCC_FLAGS -o "$ptx_file" "$cu_file"
done

echo "All kernels compiled successfully."
