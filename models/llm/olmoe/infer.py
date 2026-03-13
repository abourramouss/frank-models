#!/usr/bin/env python3
"""OLMoE-1B-7B inference with IREE.

Compiles the model on first run (caches .vmfb), then runs greedy generation.

Usage:
    python models/llm/olmoe/infer.py "Tell me about mixture of experts."
    python models/llm/olmoe/infer.py --max-tokens 200 "What is IREE?"
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.parent.parent
ARCH_DIR = REPO_DIR / "architectures" / "llm"
MODELS_DIR = REPO_DIR / "models" / "llm"
COMP_DIR = REPO_DIR / "components"
WEIGHTS_DIR = Path("/home/bourram/models/olmoe-1b-7b")
IREE_BIN = Path("/home/bourram/iree-install/bin")
IREE_PY = [
    "/home/bourram/iree-install/python_packages/iree_compiler",
    "/home/bourram/iree-install/python_packages/iree_runtime",
]
for p in IREE_PY:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import iree.compiler
import iree.runtime.flags
from iree.runtime import (
    BufferUsage,
    DeviceArray,
    HalBufferView,
    MemoryType,
    ParameterIndex,
    VmContext,
    VmInstance,
    VmModule,
    VmVariantList,
    create_hal_module,
    create_io_parameters_module,
    get_device,
)

# Use all available CPU threads for the IREE task system.
import os
_n_cpus = os.cpu_count() or 128
iree.runtime.flags.parse_flags(f"--task_topology_group_count={_n_cpus}")

# Model config
N_LAYERS = 16
VOCAB_SIZE = 50304
BLOCK_SIZE = 16
MAX_BLOCKS_PER_SEQ = 64  # supports up to 1024 tokens per sequence

LIBRARY_PATHS = [
    str(MODELS_DIR / "olmoe" / "hparams.mlir"),
    str(MODELS_DIR / "olmoe" / "params.mlir"),
    str(ARCH_DIR / "transformer_layer_moe_prefill.mlir"),
    str(ARCH_DIR / "transformer_layer_moe_decode.mlir"),
    str(COMP_DIR / "embedding" / "embedding_lookup.mlir"),
    str(COMP_DIR / "normalization" / "rms_norm.mlir"),
    str(COMP_DIR / "kvcache" / "kvcache.mlir"),
    str(COMP_DIR / "attention" / "attention_block_prefill.mlir"),
    str(COMP_DIR / "attention" / "attention_block_decode.mlir"),
    str(COMP_DIR / "attention" / "attention_gqa.mlir"),
    str(COMP_DIR / "position" / "rope.mlir"),
    str(COMP_DIR / "moe" / "moe_ffn_block.mlir"),
    str(COMP_DIR / "moe" / "mul_mat_id.mlir"),
    str(COMP_DIR / "activation" / "swiglu.mlir"),
]

VMFB_CACHE = WEIGHTS_DIR / "olmoe.vmfb"
IRPA_PATH = WEIGHTS_DIR / "olmoe.irpa"
TOKENIZER_PATH = WEIGHTS_DIR / "tokenizer.json"


def compile_model(vmfb_path: Path) -> None:
    """Link all MLIR modules and compile to VMFB."""
    main_mlir = ARCH_DIR / "llm_inference.mlir"
    iree_link = IREE_BIN / "iree-link"

    print("Linking MLIR modules...")
    with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as f:
        linked_path = f.name

    cmd = [str(iree_link), str(main_mlir)]
    for lib in LIBRARY_PATHS:
        cmd += ["--link-module", lib]
    cmd += ["-o", linked_path]
    subprocess.run(cmd, check=True)

    print("Compiling to VMFB (this takes a few minutes)...")
    iree_compile = IREE_BIN / "iree-compile"
    cmd = [
        str(iree_compile),
        linked_path,
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
        "-o", str(vmfb_path),
    ]
    subprocess.run(cmd, check=True)
    # Path(linked_path).unlink(missing_ok=True)
    print(f"Linked MLIR: {linked_path}")
    print(f"Compiled: {vmfb_path}")


def load_runtime(vmfb_path: Path, irpa_path: Path):
    """Load compiled VMFB and parameter archive, return (context, vm_module, device)."""
    instance = VmInstance()
    device = get_device("local-task")
    hal_module = create_hal_module(instance, device)

    # Load weights from .irpa (memory-mapped, no full RAM load)
    param_index = ParameterIndex()
    param_index.load(str(irpa_path))
    provider = param_index.create_provider(scope="model")
    io_module = create_io_parameters_module(instance, provider)

    # Load compiled model
    vm_module = VmModule.mmap(instance, str(vmfb_path))
    context = VmContext(instance, modules=[io_module, hal_module, vm_module])
    return context, vm_module, device


def tokenize(text: str) -> list[int]:
    result = subprocess.run(
        [str(IREE_BIN / "iree-tokenize"), f"--tokenizer={TOKENIZER_PATH}", text],
        capture_output=True, text=True, check=True,
    )
    return [int(t) for t in result.stdout.strip().split(",") if t.strip()]


def detokenize(token_ids: list[int]) -> str:
    ids_str = ",".join(str(t) for t in token_ids)
    result = subprocess.run(
        [str(IREE_BIN / "iree-tokenize"), f"--tokenizer={TOKENIZER_PATH}",
         "--decode", ids_str],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


class OLMoERunner:
    def __init__(self, context: VmContext, vm_module, device):
        self._ctx = context
        self._mod = vm_module
        self._dev = device

    def _to_bv(self, arr: np.ndarray) -> HalBufferView:
        from tests.utils import DTYPE_TO_ELEMENT_TYPE
        arr = np.ascontiguousarray(arr)
        etype = DTYPE_TO_ELEMENT_TYPE[arr.dtype.type]
        return self._dev.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
            device=self._dev, buffer=arr, element_type=etype,
        )

    def _from_bv(self, bv: HalBufferView) -> np.ndarray:
        return DeviceArray(self._dev, bv, implicit_host_transfer=True).to_host()

    def allocate_kv_cache(self, n_blocks: int, block_size: int) -> VmVariantList:
        func = self._mod.lookup_function("allocate_kv_cache")
        args = VmVariantList(2)
        args.push_int(n_blocks)
        args.push_int(block_size)
        results = VmVariantList(1)
        self._ctx.invoke(func, args, results)
        return results.get_as_list(0)

    def prefill(self, tokens, positions, cache, block_tables, start_positions, block_size):
        func = self._mod.lookup_function("prefill")
        args = VmVariantList(6)
        args.push_ref(self._to_bv(tokens))
        args.push_ref(self._to_bv(positions))
        args.push_list(cache)
        args.push_ref(self._to_bv(block_tables))
        args.push_ref(self._to_bv(start_positions))
        args.push_int(block_size)
        results = VmVariantList(2)
        self._ctx.invoke(func, args, results)
        logits = self._from_bv(results.get_as_object(0, HalBufferView))
        return logits, results.get_as_list(1)

    def decode(self, token, position, cache, block_tables, context_lens, max_context_len):
        func = self._mod.lookup_function("decode")
        args = VmVariantList(6)
        args.push_ref(self._to_bv(token))
        args.push_ref(self._to_bv(position))
        args.push_list(cache)
        args.push_ref(self._to_bv(block_tables))
        args.push_ref(self._to_bv(context_lens))
        args.push_int(max_context_len)
        results = VmVariantList(2)
        self._ctx.invoke(func, args, results)
        logits = self._from_bv(results.get_as_object(0, HalBufferView))
        return logits, results.get_as_list(1)


def generate(prompt: str, max_new_tokens: int = 100, eos_token_id: int = 50279) -> str:
    # Compile if needed
    if not VMFB_CACHE.exists():
        compile_model(VMFB_CACHE)

    print("Loading model...")
    context, vm_module, device = load_runtime(VMFB_CACHE, IRPA_PATH)
    runner = OLMoERunner(context, vm_module, device)

    # Tokenize
    input_ids = tokenize(prompt)
    seq_len = len(input_ids)
    print(f"Prompt tokens ({seq_len}): {input_ids}")

    # Allocate KV cache
    blocks_per_seq = MAX_BLOCKS_PER_SEQ
    n_blocks = N_LAYERS * blocks_per_seq
    cache = runner.allocate_kv_cache(n_blocks, BLOCK_SIZE)

    # Build block tables: [n_layers, 1, blocks_per_seq]
    block_tables = np.zeros((N_LAYERS, 1, blocks_per_seq), dtype=np.int32)
    for layer in range(N_LAYERS):
        for blk in range(blocks_per_seq):
            block_tables[layer, 0, blk] = layer * blocks_per_seq + blk

    # Prefill
    tokens = np.array([input_ids], dtype=np.int64)  # [1, seq_len]
    positions = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)  # [1, seq_len]
    start_positions = np.zeros(1, dtype=np.int32)

    print("Prefilling...")
    logits, cache = runner.prefill(tokens, positions, cache, block_tables, start_positions, BLOCK_SIZE)

    # Greedy argmax on last position
    next_token = int(np.argmax(logits[0, -1]))
    generated = [next_token]
    print(f"  prefill -> token {next_token}")

    # Decode loop
    for step in range(max_new_tokens - 1):
        if next_token == eos_token_id:
            break

        pos = seq_len + step
        tok = np.array([next_token], dtype=np.int64)          # [1]
        position = np.array([pos], dtype=np.int64)             # [1]
        context_lens = np.full((N_LAYERS, 1), pos, dtype=np.int32)  # [n_layers, 1]

        logits, cache = runner.decode(tok, position, cache, block_tables, context_lens, pos)
        next_token = int(np.argmax(logits[0]))
        generated.append(next_token)
        print(f"  step {step+1} -> token {next_token}")

    print(f"Generated {len(generated)} tokens: {generated}")
    return detokenize(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--eos", type=int, default=50279)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_DIR))
    output = generate(args.prompt, max_new_tokens=args.max_tokens, eos_token_id=args.eos)
    print("\n--- Generated ---")
    print(output)
