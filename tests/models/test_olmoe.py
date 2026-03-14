"""Compilation and smoke tests for OLMoE-1B-7B.

Tests that the OLMoE model (hparams + params + architecture + components)
links, compiles, and runs prefill/decode with random weights.

No numeric oracle — these are compilation and shape correctness tests only.
Numeric validation requires real model weights.

OLMoE-1B-7B dimensions:
  vocab_size=50280, n_embd=2048, n_head=16, n_head_kv=16, head_dim=128
  n_ff=1024, n_expert=64, n_expert_used=8, n_layers=16
"""

import numpy as np
import pytest
from pathlib import Path

from iree.runtime import (
    BufferUsage,
    DeviceArray,
    HalBufferView,
    MemoryType,
    VmVariantList,
)

from tests.utils import (
    link_and_compile_with_params,
    IREEConfig,
    DTYPE_TO_ELEMENT_TYPE,
)


# Paths
ARCH_LLM_DIR = Path(__file__).parent.parent.parent / "architectures" / "llm"
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "llm"
COMPONENTS_DIR = Path(__file__).parent.parent.parent / "components"

# OLMoE-1B-7B configuration
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


def generate_olmoe_params(seed: int = 42) -> dict[str, np.ndarray]:
    """Generate random f32 parameters for OLMoE-1B-7B matching GGUF naming."""
    rng = np.random.default_rng(seed)
    cfg = OLMOE_CONFIG
    params = {}

    # Model-level
    params["token_embd.weight"] = rng.standard_normal(
        (cfg["vocab_size"], cfg["n_embd"])
    ).astype(np.float32)
    params["output_norm.weight"] = rng.standard_normal(cfg["n_embd"]).astype(np.float32)
    params["output.weight"] = rng.standard_normal(
        (cfg["n_embd"], cfg["vocab_size"])
    ).astype(np.float32)

    n_embd_kv = cfg["n_head_kv"] * cfg["head_dim"]

    for layer_idx in range(cfg["n_layers"]):
        prefix = f"blk.{layer_idx}"

        # Normalization
        params[f"{prefix}.attn_norm.weight"] = rng.standard_normal(
            cfg["n_embd"]
        ).astype(np.float32)
        params[f"{prefix}.ffn_norm.weight"] = rng.standard_normal(
            cfg["n_embd"]
        ).astype(np.float32)

        # Attention projections
        params[f"{prefix}.attn_q.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_embd"])
        ).astype(np.float32)
        params[f"{prefix}.attn_k.weight"] = rng.standard_normal(
            (cfg["n_embd"], n_embd_kv)
        ).astype(np.float32)
        params[f"{prefix}.attn_v.weight"] = rng.standard_normal(
            (cfg["n_embd"], n_embd_kv)
        ).astype(np.float32)
        params[f"{prefix}.attn_output.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_embd"])
        ).astype(np.float32)

        # Biases (zeros — OLMoE has no attention bias)
        params[f"{prefix}.attn_q.bias"] = np.zeros(cfg["n_embd"], dtype=np.float32)
        params[f"{prefix}.attn_k.bias"] = np.zeros(n_embd_kv, dtype=np.float32)
        params[f"{prefix}.attn_v.bias"] = np.zeros(n_embd_kv, dtype=np.float32)
        params[f"{prefix}.attn_output.bias"] = np.zeros(cfg["n_embd"], dtype=np.float32)

        # QK norm weights (OLMoE uses QK norm on flat projections)
        params[f"{prefix}.attn_q_norm.weight"] = rng.standard_normal(
            cfg["n_embd"]
        ).astype(np.float32)
        params[f"{prefix}.attn_k_norm.weight"] = rng.standard_normal(
            n_embd_kv
        ).astype(np.float32)

        # MoE weights
        params[f"{prefix}.ffn_gate_inp.weight"] = rng.standard_normal(
            (cfg["n_expert"], cfg["n_embd"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_up_exps.weight"] = rng.standard_normal(
            (cfg["n_ff"], cfg["n_embd"], cfg["n_expert"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_gate_exps.weight"] = rng.standard_normal(
            (cfg["n_ff"], cfg["n_embd"], cfg["n_expert"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_down_exps.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_ff"], cfg["n_expert"])
        ).astype(np.float32)

    return params


def _get_library_paths() -> list[str]:
    return [
        str(MODELS_DIR / "olmoe" / "hparams.mlir"),
        str(MODELS_DIR / "olmoe" / "params.mlir"),
        str(ARCH_LLM_DIR / "transformer_layer_moe_prefill.mlir"),
        str(ARCH_LLM_DIR / "transformer_layer_moe_decode.mlir"),
        "embedding/embedding_lookup.mlir",
        "normalization/rms_norm.mlir",
        "kvcache/kvcache.mlir",
        "attention/attention_block_prefill.mlir",
        "attention/attention_block_decode.mlir",
        "attention/attention_gqa.mlir",
        "position/rope.mlir",
        "moe/moe_ffn_block.mlir",
        "moe/mul_mat_id.mlir",
        "activation/swiglu.mlir",
    ]


class OLMoERunner:
    """Thin runner wrapping a compiled OLMoE module."""

    def __init__(self, model):
        self._model = model
        self._device = model._device

    def _to_bv(self, arr: np.ndarray) -> HalBufferView:
        arr = np.ascontiguousarray(arr)
        etype = DTYPE_TO_ELEMENT_TYPE[arr.dtype.type]
        return self._device.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
            device=self._device,
            buffer=arr,
            element_type=etype,
        )

    def _from_bv(self, bv: HalBufferView) -> np.ndarray:
        return DeviceArray(self._device, bv, implicit_host_transfer=True).to_host()

    def allocate_kv_cache(self, n_blocks: int, block_size: int) -> VmVariantList:
        func = self._model.lookup_function("allocate_kv_cache")
        args = VmVariantList(2)
        args.push_int(n_blocks)
        args.push_int(block_size)
        results = VmVariantList(1)
        self._model._context.invoke(func, args, results)
        return results.get_as_list(0)

    def prefill(self, tokens, positions, cache, block_tables, start_positions, block_size):
        func = self._model.lookup_function("prefill")
        args = VmVariantList(6)
        args.push_ref(self._to_bv(tokens))
        args.push_ref(self._to_bv(positions))
        args.push_list(cache)
        args.push_ref(self._to_bv(block_tables))
        args.push_ref(self._to_bv(start_positions))
        args.push_int(block_size)
        results = VmVariantList(2)
        self._model._context.invoke(func, args, results)
        logits = self._from_bv(results.get_as_object(0, HalBufferView))
        cache_out = results.get_as_list(1)
        return logits, cache_out

    def decode(self, tokens, positions, cache, block_tables, context_lens,
               max_context_len, logical_block, pos_in_block, max_blocks_per_seq):
        func = self._model.lookup_function("decode")
        args = VmVariantList(10)
        args.push_ref(self._to_bv(tokens))
        args.push_ref(self._to_bv(positions))
        args.push_list(cache)
        args.push_ref(self._to_bv(block_tables))
        args.push_ref(self._to_bv(context_lens))
        args.push_int(max_context_len)
        args.push_int(logical_block)
        args.push_int(pos_in_block)
        args.push_int(max_blocks_per_seq)
        results = VmVariantList(2)
        self._model._context.invoke(func, args, results)
        logits = self._from_bv(results.get_as_object(0, HalBufferView))
        cache_out = results.get_as_list(1)
        return logits, cache_out


class TestOLMoECompilation:
    """Test that OLMoE-1B-7B links and compiles."""

    def test_compiles_with_params(self, iree_cfg):
        params = generate_olmoe_params(seed=0)
        model = link_and_compile_with_params(
            main_path=str(ARCH_LLM_DIR / "llm_inference.mlir"),
            library_paths=_get_library_paths(),
            iree_cfg=iree_cfg,
            params=params,
            scope="model",
            debug_name="olmoe",
        )
        assert model.lookup_function("allocate_kv_cache") is not None
        assert model.lookup_function("prefill") is not None
        assert model.lookup_function("decode") is not None


class TestOLMoEPrefill:
    """Smoke tests for OLMoE prefill and decode with random weights."""

    @pytest.fixture
    def runner(self, iree_cfg):
        params = generate_olmoe_params(seed=0)
        model = link_and_compile_with_params(
            main_path=str(ARCH_LLM_DIR / "llm_inference.mlir"),
            library_paths=_get_library_paths(),
            iree_cfg=iree_cfg,
            params=params,
            scope="model",
            debug_name="olmoe_smoke",
        )
        return OLMoERunner(model)

    def test_prefill_shape(self, runner):
        """Prefill with batch=1, seq=4 produces correct output shape."""
        cfg = OLMOE_CONFIG
        batch, seq_len = 1, 4
        block_size = 16
        max_blocks = 2

        n_blocks = cfg["n_layers"] * batch * max_blocks
        cache = runner.allocate_kv_cache(n_blocks, block_size)
        tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
        positions = np.arange(seq_len).reshape(1, seq_len).astype(np.int64)
        block_tables = np.zeros((cfg["n_layers"], batch, max_blocks), dtype=np.int32)
        for layer in range(cfg["n_layers"]):
            for b in range(batch):
                for blk in range(max_blocks):
                    block_tables[layer, b, blk] = (
                        layer * batch * max_blocks + b * max_blocks + blk
                    )
        start_positions = np.zeros(batch, dtype=np.int32)

        logits, _ = runner.prefill(
            tokens, positions, cache, block_tables, start_positions, block_size
        )
        assert logits.shape == (batch, seq_len, cfg["vocab_size"])

    def test_prefill_then_decode(self, runner):
        """Prefill followed by a decode step produces correct shapes."""
        cfg = OLMOE_CONFIG
        batch, prefill_len = 1, 4
        block_size = 16
        max_blocks = 2

        n_blocks = cfg["n_layers"] * batch * max_blocks
        cache = runner.allocate_kv_cache(n_blocks, block_size)
        tokens = np.array([[10, 20, 30, 40]], dtype=np.int64)
        positions = np.arange(prefill_len).reshape(1, prefill_len).astype(np.int64)
        block_tables = np.zeros((cfg["n_layers"], batch, max_blocks), dtype=np.int32)
        for layer in range(cfg["n_layers"]):
            for b in range(batch):
                for blk in range(max_blocks):
                    block_tables[layer, b, blk] = (
                        layer * batch * max_blocks + b * max_blocks + blk
                    )
        start_positions = np.zeros(batch, dtype=np.int32)

        prefill_logits, cache = runner.prefill(
            tokens, positions, cache, block_tables, start_positions, block_size
        )
        assert prefill_logits.shape == (batch, prefill_len, cfg["vocab_size"])

        # Decode one token — inputs are 1D [batch], matching the decode ABI
        decode_token = np.array([99], dtype=np.int64)          # [batch]
        decode_pos = np.array([prefill_len], dtype=np.int64)   # [batch]
        context_lens = np.full(
            (cfg["n_layers"], batch), prefill_len, dtype=np.int32
        )                                                        # [n_layers, batch]
        max_ctx = prefill_len

        # Precompute scatter indices for decode
        logical_blk = prefill_len // block_size
        pos_in_blk = prefill_len % block_size

        decode_logits, _ = runner.decode(
            decode_token, decode_pos, cache, block_tables,
            context_lens, max_ctx,
            logical_blk, pos_in_blk, max_blocks,
        )
        assert decode_logits.shape == (batch, cfg["vocab_size"])
