#!/usr/bin/env python3.11
"""
TinyLlama benchmark via sharktank direct quantization pipeline.

Pipeline: GGUF (Q4_K_M) → sharktank Dataset → turbine export → iree-compile → benchmark

Requires python3.11 with IREE compiler bindings.
Run with:
  PYTHONPATH="/home/bourram/iree-python-compiler:/home/bourram/iree-python-runtime" python3.11 scripts/bench_tinyllama_sharktank.py
"""

import sys
import types

# ---- Stub AMD-specific wave_lang before any sharktank import ----
for name in [
    "sharktank.kernels.wave",
    "sharktank.kernels.wave.mxfp4_gemm",
    "sharktank.kernels.wave.attention",
    "sharktank.kernels.wave.extend_attention",
    "sharktank.kernels.wave.utils",
]:
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
sys.modules["sharktank.kernels.wave.mxfp4_gemm"].wave_mxfp4_bmm = None
sys.modules["sharktank.kernels.wave.attention"].wave_attention = None
sys.modules["sharktank.kernels.wave.extend_attention"].wave_extend_attention = None
# ---- end stubs ----

import subprocess
import time
from pathlib import Path

WORK_DIR = Path(__file__).parent.parent / "models" / "llm" / "tinyllama"
GGUF_PATH = WORK_DIR / "gguf" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
IREE_BIN = Path.home() / "iree-install" / "bin"


def step(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}\n")


def gguf_to_irpa():
    """Convert GGUF to sharktank Dataset (IRPA file)."""
    irpa_path = WORK_DIR / "tinyllama_q4km.irpa"
    if irpa_path.exists():
        print(f"IRPA already exists: {irpa_path}")
        return irpa_path

    step("Step 1: GGUF → sharktank Dataset → IRPA")

    from sharktank.types.gguf_interop import load_file
    from sharktank.types import Dataset

    print(f"Loading GGUF: {GGUF_PATH}")
    dataset = load_file(GGUF_PATH)
    print(f"Dataset loaded. Properties: {list(dataset.properties.keys())[:10]}...")
    print(f"Root theta keys: {list(dataset.root_theta.keys)[:10]}...")

    dataset.save(irpa_path)
    size_mb = irpa_path.stat().st_size / (1024 * 1024)
    print(f"Saved IRPA: {irpa_path} ({size_mb:.0f} MB)")
    return irpa_path


def export_mlir(irpa_path):
    """Export the model to MLIR using sharktank's PagedLlmModelV1."""
    mlir_path = WORK_DIR / "tinyllama_q4km.mlir"
    if mlir_path.exists():
        print(f"MLIR already exists: {mlir_path}")
        return mlir_path

    step("Step 2: sharktank model → turbine export → MLIR")

    from sharktank.types import Dataset
    from sharktank.layers import LlamaModelConfig
    from sharktank.models.llm import PagedLlmModelV1

    import torch
    from iree.turbine.aot import FxProgramsBuilder, export

    dataset = Dataset.load(irpa_path)
    hp = dataset.properties

    print(f"Model properties: {hp}")

    config = LlamaModelConfig(hp, tensor_parallelism_size=1)
    config.kv_cache_type = "direct"
    config.use_hf = False

    model = PagedLlmModelV1(dataset.root_theta, config)
    print(f"Model created: {model}")

    # Export prefill and decode functions
    fxb = FxProgramsBuilder(model)
    batch_size = 1
    sl = 32  # sequence length for prefill

    # For a simple benchmark, just export a forward pass
    print("Tracing model for export...")

    @fxb.export_program(
        name="prefill",
        args=(
            torch.empty(batch_size, sl, dtype=torch.int64),  # tokens
            torch.empty(batch_size, dtype=torch.int64),  # seq_lens
        ),
    )
    def _(model, tokens, seq_lens):
        return model.prefill(tokens, seq_lens=seq_lens)

    print("Exporting to MLIR...")
    output = export(fxb, import_symbolic_shape_expressions=True)
    output.save_mlir(str(mlir_path))

    size_mb = mlir_path.stat().st_size / (1024 * 1024)
    print(f"MLIR saved: {mlir_path} ({size_mb:.1f} MB)")
    return mlir_path


def compile_iree(mlir_path, irpa_path, backend):
    """Compile MLIR to IREE vmfb."""
    vmfb_path = WORK_DIR / f"tinyllama_q4km_{backend}.vmfb"
    if vmfb_path.exists():
        print(f"VMFB already exists: {vmfb_path}")
        return vmfb_path

    step(f"Step 3: iree-compile for {backend}")
    iree_compile = str(IREE_BIN / "iree-compile")

    cmd = [iree_compile, str(mlir_path)]
    cmd += [f"--iree-hal-target-backends={backend}"]

    if backend == "cuda":
        cmd += ["--iree-cuda-target=sm_87", "--iree-cuda-target-features=+ptx74"]
    elif backend == "llvm-cpu":
        cmd += ["--iree-llvmcpu-target-cpu=cortex-a78ae"]

    cmd += ["--iree-opt-level=O3", "-o", str(vmfb_path)]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"iree-compile failed for {backend}")

    size_mb = vmfb_path.stat().st_size / (1024 * 1024)
    print(f"Compiled: {vmfb_path} ({size_mb:.1f} MB) in {elapsed:.1f}s")
    return vmfb_path


def benchmark(vmfb_path, irpa_path, backend, seq_len=1):
    """Benchmark using iree-benchmark-module."""
    step(f"Step 4: Benchmark {backend} (seq_len={seq_len})")
    iree_bench = str(IREE_BIN / "iree-benchmark-module")
    device = "local-task" if backend == "llvm-cpu" else "cuda"

    cmd = [
        iree_bench,
        f"--module={vmfb_path}",
        f"--parameters=model={irpa_path}",
        f"--device={device}",
        "--function=prefill",
        f"--input=1x{seq_len}xi64",
        "--input=1xi64=32",
        "--benchmark_repetitions=5",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-1000:]}")
    return result.returncode == 0


def main():
    backends = sys.argv[1:] if len(sys.argv) > 1 else ["llvm-cpu", "cuda"]

    # Step 1: GGUF → IRPA
    irpa_path = gguf_to_irpa()

    # Step 2: Export MLIR
    mlir_path = export_mlir(irpa_path)

    # Step 3+4: Compile and benchmark
    for backend in backends:
        try:
            vmfb_path = compile_iree(mlir_path, irpa_path, backend)
            # Benchmark decode (1 token)
            benchmark(vmfb_path, irpa_path, backend, seq_len=1)
            # Benchmark prefill (32 tokens)
            benchmark(vmfb_path, irpa_path, backend, seq_len=32)
        except Exception as e:
            print(f"ERROR ({backend}): {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
