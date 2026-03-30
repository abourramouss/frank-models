#!/usr/bin/env python3
"""Import TinyLlama via ONNX → IREE pipeline and benchmark."""

import subprocess
import sys
import os
import time
import tempfile
from pathlib import Path

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
WORK_DIR = Path(__file__).parent.parent / "models" / "llm" / "tinyllama"
IREE_BIN = Path.home() / "iree-install" / "bin"
SEQ_LEN = 32  # prompt length for benchmarking
BATCH = 1


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def export_onnx():
    """Export TinyLlama to ONNX using optimum."""
    onnx_dir = WORK_DIR / "onnx"
    onnx_path = onnx_dir / "model.onnx"
    if onnx_path.exists():
        print(f"ONNX already exported at {onnx_path}")
        return onnx_path

    step("Step 1: Exporting TinyLlama to ONNX via optimum")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    from optimum.exporters.onnx import main_export

    main_export(
        MODEL_ID,
        output=str(onnx_dir),
        task="text-generation",
        opset=17,
        no_post_process=True,
    )

    # Find the decoder model (optimum may split into encoder/decoder)
    candidates = list(onnx_dir.glob("*.onnx"))
    print(f"Exported ONNX files: {[c.name for c in candidates]}")

    # Prefer decoder_model.onnx if it exists, else model.onnx
    for name in ["decoder_model.onnx", "model.onnx"]:
        p = onnx_dir / name
        if p.exists():
            return p

    return candidates[0]


def import_to_mlir(onnx_path):
    """Convert ONNX to MLIR using iree-import-onnx."""
    mlir_path = WORK_DIR / "tinyllama.mlir"
    if mlir_path.exists():
        print(f"MLIR already exists at {mlir_path}")
        return mlir_path

    step("Step 2: Importing ONNX → MLIR")
    cmd = [
        "iree-import-onnx",
        str(onnx_path),
        "-o", str(mlir_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_mb = mlir_path.stat().st_size / (1024 * 1024)
    print(f"MLIR output: {mlir_path} ({size_mb:.1f} MB)")
    return mlir_path


def compile_iree(mlir_path, backend):
    """Compile MLIR to IREE vmfb."""
    vmfb_path = WORK_DIR / f"tinyllama_{backend}.vmfb"
    if vmfb_path.exists():
        print(f"VMFB already exists at {vmfb_path}")
        return vmfb_path

    step(f"Step 3: Compiling for {backend}")
    iree_compile = str(IREE_BIN / "iree-compile")

    cmd = [iree_compile, str(mlir_path)]
    cmd += [f"--iree-hal-target-backends={backend}"]

    if backend == "cuda":
        cmd += ["--iree-cuda-target=sm_87"]  # Orin GPU
    elif backend == "llvm-cpu":
        cmd += ["--iree-llvmcpu-target-cpu=cortex-a78ae"]  # Orin CPU cores

    cmd += ["-o", str(vmfb_path)]
    cmd += ["--iree-opt-level=3"]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"iree-compile failed for {backend}")

    size_mb = vmfb_path.stat().st_size / (1024 * 1024)
    print(f"Compiled {backend}: {vmfb_path} ({size_mb:.1f} MB) in {elapsed:.1f}s")
    return vmfb_path


def benchmark_vmfb(vmfb_path, backend):
    """Benchmark compiled module using iree-benchmark-module."""
    step(f"Step 4: Benchmarking {backend}")
    iree_bench = str(IREE_BIN / "iree-benchmark-module")

    device = "local-task" if backend == "llvm-cpu" else "cuda"

    cmd = [
        iree_bench,
        f"--module={vmfb_path}",
        f"--device={device}",
        "--benchmark_repetitions=5",
    ]

    # Pass dummy inputs matching TinyLlama shapes
    # input_ids: [1, seq_len] i64, attention_mask: [1, seq_len] i64
    cmd += [
        f"--input={BATCH}x{SEQ_LEN}xi64",
        f"--input={BATCH}x{SEQ_LEN}xi64",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-1000:])
    return result.returncode == 0


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    backends = sys.argv[1:] if len(sys.argv) > 1 else ["llvm-cpu", "cuda"]

    # Step 1: Export
    onnx_path = export_onnx()

    # Step 2: Import
    mlir_path = import_to_mlir(onnx_path)

    # Step 3+4: Compile and benchmark each backend
    for backend in backends:
        try:
            vmfb_path = compile_iree(mlir_path, backend)
            benchmark_vmfb(vmfb_path, backend)
        except Exception as e:
            print(f"ERROR ({backend}): {e}")
            continue


if __name__ == "__main__":
    main()
