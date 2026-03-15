#!/usr/bin/env python3
"""Interactive chat with OLMoE-1B-7B using IREE.

Compiles (or loads a cached .vmfb) the OLMoE model from MLIR, loads
parameters from an .irpa archive, and runs a greedy-decoding chat loop.

Usage:
    # With pre-compiled .vmfb:
    python scripts/chat_olmoe.py \
        --model-dir /path/to/olmoe-1b-7b \
        --params /path/to/params.irpa \
        --vmfb /path/to/olmoe.vmfb

    # Compile from scratch (requires iree-link and iree-compile on PATH):
    python scripts/chat_olmoe.py \
        --model-dir /path/to/olmoe-1b-7b \
        --params /path/to/params.irpa
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from iree.runtime import (
    BufferUsage,
    DeviceArray,
    HalBufferView,
    HalElementType,
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


# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_LLM_DIR = REPO_ROOT / "architectures" / "llm"
MODELS_DIR = REPO_ROOT / "models" / "llm"
COMPONENTS_DIR = REPO_ROOT / "components"


# ---------------------------------------------------------------------------
# OLMoE-1B-7B configuration
# ---------------------------------------------------------------------------
OLMOE_CONFIG = {
    "vocab_size": 50304,
    "n_embd": 2048,
    "n_head": 16,
    "n_head_kv": 16,
    "head_dim": 128,
    "n_ff": 1024,
    "n_expert": 64,
    "n_expert_used": 8,
    "n_layers": 16,
    "rms_eps": 1e-5,
    "rope_freq_base": 10000.0,
}

EOS_TOKEN = 50279

DTYPE_TO_ELEMENT_TYPE = {
    np.float32: HalElementType.FLOAT_32,
    np.float64: HalElementType.FLOAT_64,
    np.float16: HalElementType.FLOAT_16,
    np.int32: HalElementType.SINT_32,
    np.int64: HalElementType.SINT_64,
    np.int16: HalElementType.SINT_16,
    np.int8: HalElementType.SINT_8,
    np.uint32: HalElementType.UINT_32,
    np.uint64: HalElementType.UINT_64,
    np.uint16: HalElementType.UINT_16,
    np.uint8: HalElementType.UINT_8,
}

DRIVER_MAP = {
    "llvm-cpu": "local-task",
    "rocm": "hip",
    "cuda": "cuda",
    "vulkan": "vulkan",
}


# ---------------------------------------------------------------------------
# Library paths for iree-link (mirrors test_olmoe.py::_get_library_paths)
# ---------------------------------------------------------------------------
def _get_library_paths() -> list[str]:
    """Return the list of MLIR modules needed to link the OLMoE model."""
    return [
        str(MODELS_DIR / "olmoe" / "hparams.mlir"),
        str(MODELS_DIR / "olmoe" / "params.mlir"),
        str(ARCH_LLM_DIR / "transformer_layer_moe_prefill.mlir"),
        str(ARCH_LLM_DIR / "transformer_layer_moe_decode.mlir"),
        str(COMPONENTS_DIR / "embedding" / "embedding_lookup.mlir"),
        str(COMPONENTS_DIR / "normalization" / "rms_norm.mlir"),
        str(COMPONENTS_DIR / "kvcache" / "kvcache.mlir"),
        str(COMPONENTS_DIR / "attention" / "attention_block_prefill.mlir"),
        str(COMPONENTS_DIR / "attention" / "attention_block_decode.mlir"),
        str(COMPONENTS_DIR / "attention" / "attention_gqa.mlir"),
        str(COMPONENTS_DIR / "position" / "rope.mlir"),
        str(COMPONENTS_DIR / "moe" / "moe_ffn_block.mlir"),
        str(COMPONENTS_DIR / "moe" / "mul_mat_id.mlir"),
        str(COMPONENTS_DIR / "activation" / "swiglu.mlir"),
    ]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def _find_tool(name: str, tools_dir: Path | None = None) -> Path:
    """Locate an IREE CLI tool by name."""
    if tools_dir is not None:
        p = tools_dir / name
        if p.exists():
            return p
        raise FileNotFoundError(f"{name} not found in {tools_dir}")
    p = shutil.which(name)
    if p is not None:
        return Path(p)
    raise FileNotFoundError(
        f"{name} not found on PATH. Install IREE tools or pass --iree-tools-dir."
    )


def compile_model(
    *,
    output_vmfb: Path,
    backend: str = "llvm-cpu",
    tools_dir: Path | None = None,
) -> Path:
    """Link and compile the OLMoE model, writing *output_vmfb*.

    Steps:
      1. iree-link  (main + library modules) -> linked.mlir
      2. iree-compile linked.mlir -> output.vmfb
    """
    iree_link = _find_tool("iree-link", tools_dir)
    iree_compile = _find_tool("iree-compile", tools_dir)

    main_path = str(ARCH_LLM_DIR / "llm_inference.mlir")
    library_paths = _get_library_paths()

    with tempfile.TemporaryDirectory() as tmp:
        linked_path = Path(tmp) / "olmoe_linked.mlir"

        # --- iree-link ---
        link_cmd = [str(iree_link), main_path]
        for lib in library_paths:
            link_cmd.extend(["--link-module", lib])
        link_cmd.extend(["-o", str(linked_path)])

        print(f"[compile] Linking MLIR modules ...")
        t0 = time.time()
        subprocess.run(link_cmd, check=True)
        print(f"[compile] Linked in {time.time() - t0:.1f}s")

        # --- iree-compile ---
        compile_cmd = [
            str(iree_compile),
            str(linked_path),
            f"--iree-hal-target-backends={backend}",
            "-o",
            str(output_vmfb),
        ]
        if backend == "llvm-cpu":
            compile_cmd.append("--iree-llvmcpu-target-cpu=host")

        print(f"[compile] Compiling to {output_vmfb} ...")
        t0 = time.time()
        subprocess.run(compile_cmd, check=True)
        print(f"[compile] Compiled in {time.time() - t0:.1f}s")

    return output_vmfb


# ---------------------------------------------------------------------------
# Runtime: model loader + runner
# ---------------------------------------------------------------------------
class OLMoEChat:
    """Manages the IREE runtime for interactive OLMoE chat."""

    def __init__(
        self,
        vmfb_path: Path,
        params_path: Path,
        backend: str = "llvm-cpu",
        block_size: int = 16,
        max_seq_len: int = 2048,
        cpu_vmfb_path: Path | None = None,
    ):
        self.cfg = OLMOE_CONFIG
        self.block_size = block_size
        self.max_seq_len = max_seq_len

        driver = DRIVER_MAP.get(backend, "local-task")
        self.instance = VmInstance()
        self.device = get_device(driver)

        # HAL module for GPU
        hal_module = create_hal_module(self.instance, self.device)

        # Optional: separate CPU device + VMFB for IREE-compiled CPU postprocessing
        self._cpu_context = None
        if cpu_vmfb_path is not None:
            cpu_device = get_device("local-task")
            cpu_hal = create_hal_module(self.instance, cpu_device)
            self._cpu_device = cpu_device
            print(f"[runtime] CPU VMFB: {cpu_vmfb_path}")

        # Parameter module (loads .irpa at runtime)
        print(f"[runtime] Loading parameters from {params_path} ...")
        t0 = time.time()
        param_index = ParameterIndex()
        param_index.load(str(params_path))
        provider = param_index.create_provider(scope="model")
        params_module = create_io_parameters_module(self.instance, provider)
        print(f"[runtime] Parameters loaded in {time.time() - t0:.1f}s")

        # Extract CPU-side weights for hetero post-processing
        print(f"[runtime] Extracting CPU weights for hetero post-processing ...")
        for name, entry in param_index.items():
            if name == "output_norm.weight":
                self.output_norm_weight = np.frombuffer(
                    entry.file_view, dtype=np.float16
                ).copy()  # [2048]
            elif name == "output.weight":
                # MLIR loads as [n_embd, vocab] = [2048, 50304] — match that layout
                self.output_weight = np.frombuffer(
                    entry.file_view, dtype=np.float16
                ).reshape(self.cfg["n_embd"], self.cfg["vocab_size"]).copy()  # [2048, 50304]
        self.rms_eps = self.cfg["rms_eps"]
        # Pre-compute output weight in f32 for CPU LM head matmul
        # Already [2048, 50304] — no transpose needed
        self.output_weight_f32 = self.output_weight.astype(np.float32).copy()
        self.output_norm_weight_f32 = self.output_norm_weight.astype(np.float32).copy()
        print(f"[runtime] CPU weights: output_norm={self.output_norm_weight.shape}, "
              f"output={self.output_weight.shape}")

        # Compiled module
        print(f"[runtime] Loading compiled module from {vmfb_path} ...")
        t0 = time.time()
        with open(vmfb_path, "rb") as f:
            main_module = VmModule.copy_buffer(self.instance, f.read())
        print(f"[runtime] Module loaded in {time.time() - t0:.1f}s")

        self._vm_module = main_module

        # Try multi-device context (GPU + CPU) first for single-VMFB hetero.
        # Fall back to single-device if the VMFB doesn't require a CPU device.
        try:
            cpu_device = get_device("local-task")
            multi_hal = create_hal_module(
                self.instance, devices=[self.device, cpu_device]
            )
            self._context = VmContext(
                self.instance, modules=[params_module, multi_hal, main_module]
            )
            print(f"[runtime] Multi-device context: GPU + CPU")
        except Exception:
            self._context = VmContext(
                self.instance, modules=[params_module, hal_module, main_module]
            )

        # Set up CPU context if cpu_vmfb provided
        if cpu_vmfb_path is not None:
            print(f"[runtime] Loading CPU VMFB from {cpu_vmfb_path} ...")
            with open(cpu_vmfb_path, "rb") as f:
                cpu_module = VmModule.copy_buffer(self.instance, f.read())
            # CPU context needs its own params provider + HAL
            cpu_provider = param_index.create_provider(scope="model")
            cpu_params_module = create_io_parameters_module(self.instance, cpu_provider)
            self._cpu_module = cpu_module
            self._cpu_context = VmContext(
                self.instance, modules=[cpu_params_module, cpu_hal, cpu_module]
            )
            print(f"[runtime] CPU module ready: {cpu_module.function_names}")

    # -- Tensor conversion helpers --

    def _to_bv(self, arr: np.ndarray) -> HalBufferView:
        arr = np.ascontiguousarray(arr)
        etype = DTYPE_TO_ELEMENT_TYPE[arr.dtype.type]
        return self.device.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
            device=self.device,
            buffer=arr,
            element_type=etype,
        )

    def _from_bv(self, bv: HalBufferView) -> np.ndarray:
        return DeviceArray(self.device, bv, implicit_host_transfer=True).to_host()

    # -- Model entry points --

    def allocate_kv_cache(self, n_blocks: int) -> VmVariantList:
        func = self._vm_module.lookup_function("allocate_kv_cache")
        args = VmVariantList(2)
        args.push_int(n_blocks)
        args.push_int(self.block_size)
        results = VmVariantList(1)
        self._context.invoke(func, args, results)
        return results.get_as_list(0)

    def prefill(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        cache: VmVariantList,
        block_tables: np.ndarray,
        start_positions: np.ndarray,
    ) -> tuple[np.ndarray, VmVariantList]:
        func = self._vm_module.lookup_function("prefill")
        args = VmVariantList(6)
        args.push_ref(self._to_bv(tokens))
        args.push_ref(self._to_bv(positions))
        args.push_list(cache)
        args.push_ref(self._to_bv(block_tables))
        args.push_ref(self._to_bv(start_positions))
        args.push_int(self.block_size)
        results = VmVariantList(2)
        self._context.invoke(func, args, results)
        logits = self._from_bv(results.get_as_object(0, HalBufferView))
        cache_out = results.get_as_list(1)
        return logits, cache_out

    def decode(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        cache: VmVariantList,
        block_tables: np.ndarray,
        context_lens: np.ndarray,
        max_context_len: int,
    ) -> tuple[np.ndarray, VmVariantList]:
        func = self._vm_module.lookup_function("decode")
        args = VmVariantList(6)
        args.push_ref(self._to_bv(tokens))
        args.push_ref(self._to_bv(positions))
        args.push_list(cache)
        args.push_ref(self._to_bv(block_tables))
        args.push_ref(self._to_bv(context_lens))
        args.push_int(max_context_len)
        results = VmVariantList(2)
        self._context.invoke(func, args, results)
        logits = self._from_bv(results.get_as_object(0, HalBufferView))
        cache_out = results.get_as_list(1)
        return logits, cache_out

    def decode_body(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        cache: VmVariantList,
        block_tables: np.ndarray,
        context_lens: np.ndarray,
        max_context_len: int,
    ) -> tuple[np.ndarray, VmVariantList]:
        """Run transformer layers only — returns hidden state, not logits."""
        func = self._vm_module.lookup_function("decode_body")
        args = VmVariantList(6)
        args.push_ref(self._to_bv(tokens))
        args.push_ref(self._to_bv(positions))
        args.push_list(cache)
        args.push_ref(self._to_bv(block_tables))
        args.push_ref(self._to_bv(context_lens))
        args.push_int(max_context_len)
        results = VmVariantList(2)
        self._context.invoke(func, args, results)
        hidden = self._from_bv(results.get_as_object(0, HalBufferView))
        cache_out = results.get_as_list(1)
        return hidden, cache_out

    def decode_hetero(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        cache: VmVariantList,
        block_tables: np.ndarray,
        context_lens: np.ndarray,
        max_context_len: int,
    ) -> tuple[np.ndarray, VmVariantList]:
        """Heterogeneous decode: GPU transformer layers + CPU output norm/LM head.
        Device affinities set in MLIR via flow.tensor.transfer."""
        func = self._vm_module.lookup_function("decode_hetero")
        args = VmVariantList(6)
        args.push_ref(self._to_bv(tokens))
        args.push_ref(self._to_bv(positions))
        args.push_list(cache)
        args.push_ref(self._to_bv(block_tables))
        args.push_ref(self._to_bv(context_lens))
        args.push_int(max_context_len)
        results = VmVariantList(2)
        self._context.invoke(func, args, results)
        logits = self._from_bv(results.get_as_object(0, HalBufferView))
        cache_out = results.get_as_list(1)
        return logits, cache_out

    def cpu_postprocess_iree(self, hidden: np.ndarray) -> tuple[np.ndarray]:
        """CPU-side postprocess using IREE llvm-cpu VMFB (compiled MLIR)."""
        func = self._cpu_module.lookup_function("postprocess")
        # Allocate on CPU device
        arr = np.ascontiguousarray(hidden)
        etype = DTYPE_TO_ELEMENT_TYPE[arr.dtype.type]
        bv = self._cpu_device.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
            device=self._cpu_device,
            buffer=arr,
            element_type=etype,
        )
        args = VmVariantList(1)
        args.push_ref(bv)
        results = VmVariantList(1)
        self._cpu_context.invoke(func, args, results)
        logits_bv = results.get_as_object(0, HalBufferView)
        logits = DeviceArray(self._cpu_device, logits_bv, implicit_host_transfer=True).to_host()
        return logits

    def cpu_postprocess(self, hidden: np.ndarray) -> int:
        """CPU-side output norm + LM head + argmax (replaces 2-3 GPU dispatches)."""
        # RMSNorm in f32 for stability (matches MLIR semantics)
        x = hidden.astype(np.float32)
        sum_sq = np.sum(x * x, axis=-1, keepdims=True)
        rms = np.sqrt(sum_sq / x.shape[-1] + self.rms_eps)
        normed = (x / rms) * self.output_norm_weight_f32

        # LM head: [batch, 2048] @ [2048, 50304] -> [batch, 50304]
        logits = normed @ self.output_weight_f32
        return int(np.argmax(logits[0, :]))


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
def generate(
    model: OLMoEChat,
    token_ids: list[int],
    max_new_tokens: int = 256,
    eos_token: int = EOS_TOKEN,
    tokenizer=None,
) -> list[int]:
    """Run greedy autoregressive generation on *token_ids*.

    Returns the list of generated token ids (excluding the prompt).
    If tokenizer is provided, streams tokens to stdout as they are generated.
    """
    cfg = model.cfg
    n_layers = cfg["n_layers"]
    block_size = model.block_size
    batch = 1

    seq_len = len(token_ids)
    # How many blocks we need: enough for prompt + max_new_tokens, per layer
    total_len = seq_len + max_new_tokens
    max_blocks_per_seq = (total_len + block_size - 1) // block_size
    n_blocks = n_layers * batch * max_blocks_per_seq

    # Allocate KV cache
    cache = model.allocate_kv_cache(n_blocks)

    # Block tables: [n_layers, batch, max_blocks_per_seq]
    # Each layer needs its own physical blocks (cache is shared across layers).
    block_tables = np.zeros((n_layers, batch, max_blocks_per_seq), dtype=np.int32)
    for layer in range(n_layers):
        for b in range(batch):
            for blk in range(max_blocks_per_seq):
                block_tables[layer, b, blk] = (
                    layer * batch * max_blocks_per_seq
                    + b * max_blocks_per_seq
                    + blk
                )

    # Process prompt token-by-token using prefill(seq=1) for the first token
    # then decode for subsequent tokens (causal masking workaround).
    # Prefill first token
    tokens_arr = np.array([[token_ids[0]]], dtype=np.int64)  # [1, 1]
    positions_arr = np.array([[0]], dtype=np.int64)
    start_positions = np.zeros(batch, dtype=np.int32)

    prefill_logits, cache = model.prefill(
        tokens_arr, positions_arr, cache, block_tables, start_positions
    )
    last_logits = prefill_logits[0, 0, :]  # [vocab_size]

    # Process remaining prompt tokens via decode (reads from KV cache)
    # Use decode_hetero if available (single-VMFB multi-device mode)
    _prompt_decode_fn = model.decode_hetero if model._vm_module.lookup_function("decode_hetero") else model.decode
    for i in range(1, seq_len):
        decode_token = np.array([token_ids[i]], dtype=np.int64)
        decode_pos = np.array([i], dtype=np.int64)
        context_lens = np.full((n_layers, batch), i, dtype=np.int32)
        max_ctx = i

        decode_logits, cache = _prompt_decode_fn(
            decode_token, decode_pos, cache, block_tables,
            context_lens, max_ctx,
        )
        last_logits = decode_logits[0, :]

    next_token = int(np.argmax(last_logits))

    generated: list[int] = []
    cur_pos = seq_len  # position of the token we just generated

    # Detect available decode modes
    func_dh = model._vm_module.lookup_function("decode_hetero")
    func_db = model._vm_module.lookup_function("decode_body")
    if func_dh is not None:
        decode_mode = "hetero_mlir"
        print(f"[hetero] Using decode_hetero (single VMFB, MLIR device affinities)")
    elif func_db is not None and model._cpu_context is not None:
        decode_mode = "hetero_iree"
        print(f"[hetero] Using decode_body (GPU VMFB) + postprocess (CPU VMFB)")
    elif func_db is not None:
        decode_mode = "hetero_numpy"
        print(f"[hetero] Using decode_body (GPU VMFB) + numpy CPU post-processing")
    else:
        decode_mode = "gpu_only"

    for step in range(max_new_tokens):
        generated.append(next_token)

        # Stream token to stdout
        if tokenizer is not None:
            token_text = tokenizer.decode([next_token])
            sys.stdout.write(token_text)
            sys.stdout.flush()

        if next_token == eos_token:
            break

        # Prepare decode inputs
        decode_token = np.array([next_token], dtype=np.int64)  # [batch]
        decode_pos = np.array([cur_pos], dtype=np.int64)  # [batch]
        context_lens = np.full((n_layers, batch), cur_pos, dtype=np.int32)
        max_ctx = cur_pos

        if decode_mode == "hetero_mlir":
            # Single VMFB: GPU layers + CPU postprocess via flow.tensor.transfer
            decode_logits, cache = model.decode_hetero(
                decode_token, decode_pos, cache,
                block_tables, context_lens, max_ctx,
            )
            next_token = int(np.argmax(decode_logits[0, :]))
        elif decode_mode == "hetero_iree":
            # Two VMFBs: GPU decode_body + CPU postprocess
            hidden, cache = model.decode_body(
                decode_token, decode_pos, cache,
                block_tables, context_lens, max_ctx,
            )
            logits = model.cpu_postprocess_iree(hidden)
            next_token = int(np.argmax(logits[0, :]))
        elif decode_mode == "hetero_numpy":
            # GPU: transformer layers only
            hidden, cache = model.decode_body(
                decode_token, decode_pos, cache,
                block_tables, context_lens, max_ctx,
            )
            # CPU: output norm + LM head + argmax
            next_token = model.cpu_postprocess(hidden)
        else:
            decode_logits, cache = model.decode(
                decode_token, decode_pos, cache,
                block_tables, context_lens, max_ctx,
            )
            next_token = int(np.argmax(decode_logits[0, :]))

        cur_pos += 1

    if tokenizer is not None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return generated


def chat_loop(
    model: OLMoEChat,
    tokenizer,
    max_tokens: int = 256,
):
    """Run an interactive multi-turn chat loop."""
    print()
    print("OLMoE-1B-7B Chat (type 'quit' or Ctrl-D to exit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        # Tokenize the prompt
        encoded = tokenizer.encode(user_input)
        token_ids = encoded.ids

        if not token_ids:
            print("[warning] Tokenizer produced no tokens for that input.")
            continue

        print(f"[{len(token_ids)} prompt tokens]")
        sys.stdout.flush()

        # Generate
        t0 = time.time()
        generated_ids = generate(model, token_ids, max_new_tokens=max_tokens, tokenizer=tokenizer)
        elapsed = time.time() - t0

        n_gen = len(generated_ids)
        tps = n_gen / elapsed if elapsed > 0 else 0
        print(f"[{n_gen} tokens in {elapsed:.1f}s, {tps:.1f} tok/s]")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat with OLMoE-1B-7B via IREE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/bourram/models/olmoe-1b-7b"),
        help="Path to HuggingFace model directory (for tokenizer.json)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        required=True,
        help="Path to .irpa parameter archive",
    )
    parser.add_argument(
        "--vmfb",
        type=Path,
        default=None,
        help="Path to pre-compiled .vmfb. If not provided, compiles from MLIR.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="llvm-cpu",
        choices=list(DRIVER_MAP.keys()),
        help="IREE backend (default: llvm-cpu)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache block size (default: 16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per turn (default: 256)",
    )
    parser.add_argument(
        "--iree-tools-dir",
        type=Path,
        default=None,
        help="Directory containing iree-link and iree-compile",
    )
    parser.add_argument(
        "--cpu-vmfb",
        type=Path,
        default=None,
        help="Path to CPU VMFB for hetero postprocessing (compiled with llvm-cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Tokenizer ----
    tokenizer_path = args.model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Error: tokenizer not found at {tokenizer_path}", file=sys.stderr)
        sys.exit(1)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"[init] Loaded tokenizer from {tokenizer_path}")

    # ---- Compile or load .vmfb ----
    vmfb_path = args.vmfb
    if vmfb_path is None:
        # Compile from scratch; cache next to the params file
        vmfb_path = args.params.parent / "olmoe.vmfb"
        if vmfb_path.exists():
            print(f"[init] Found cached vmfb at {vmfb_path}")
        else:
            print(f"[init] No --vmfb provided; compiling from MLIR ...")
            compile_model(
                output_vmfb=vmfb_path,
                backend=args.backend,
                tools_dir=args.iree_tools_dir,
            )
    else:
        if not vmfb_path.exists():
            print(f"Error: vmfb not found at {vmfb_path}", file=sys.stderr)
            sys.exit(1)

    # ---- Load model ----
    if not args.params.exists():
        print(f"Error: params not found at {args.params}", file=sys.stderr)
        sys.exit(1)

    model = OLMoEChat(
        vmfb_path=vmfb_path,
        params_path=args.params,
        backend=args.backend,
        block_size=args.block_size,
        cpu_vmfb_path=args.cpu_vmfb,
    )
    print("[init] Model ready.")

    # ---- Chat ----
    chat_loop(model, tokenizer, max_tokens=args.max_tokens)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, lambda *_: (print("\nBye!"), os._exit(0)))
    import os
    main()
