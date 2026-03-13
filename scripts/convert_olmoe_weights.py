#!/usr/bin/env python3
"""Convert HuggingFace OLMoE safetensors weights to IREE parameter archive.

Loads BF16 safetensors shards from a HuggingFace OLMoE checkpoint,
renames weights to frank-models GGUF naming conventions, stacks
per-expert weights, adds dummy zero biases, casts everything to f32,
and writes an IREE .irpa parameter archive.

Usage:
    python scripts/convert_olmoe_weights.py \
        --model-dir /path/to/olmoe-1b-7b \
        --output /path/to/olmoe-1b-7b.irpa
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

from iree.runtime import ParameterIndex

# OLMoE-1B-7B configuration
N_LAYERS = 16
N_EMBD = 2048
N_FF = 1024
N_EXPERT = 64
VOCAB_SIZE = 50304


def load_safetensors_shards(model_dir: Path) -> dict[str, np.ndarray]:
    """Load all safetensors shards into a single dict of numpy arrays (f32)."""
    shard_paths = sorted(glob.glob(str(model_dir / "model-*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No safetensors shards found in {model_dir}. "
            f"Expected files matching model-*.safetensors"
        )

    print(f"Found {len(shard_paths)} safetensors shard(s)")
    weights = {}
    for path in shard_paths:
        print(f"  Loading {Path(path).name}...")
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                # safetensors returns BF16 as uint16 in numpy;
                # convert via float32 view trick or direct torch-free cast.
                tensor = f.get_tensor(key)
                if tensor.dtype == np.uint16:
                    # BF16 stored as uint16: shift left 16 bits into f32 bit pattern
                    tensor = (tensor.astype(np.uint32) << 16).view(np.float32)
                elif tensor.dtype != np.float32:
                    tensor = tensor.astype(np.float32)
                weights[key] = tensor

    print(f"  Loaded {len(weights)} tensors total")
    return weights


def convert_weights(hf_weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rename and reshape HF weights to frank-models convention."""
    out = {}

    # --- Global weights ---
    print("Converting global weights...")

    # token_embd.weight [50304, 2048]
    out["token_embd.weight"] = hf_weights["model.embed_tokens.weight"]

    # output_norm.weight [2048]
    out["output_norm.weight"] = hf_weights["model.norm.weight"]

    # output.weight [2048, 50304] -- transposed from HF [50304, 2048]
    lm_head = hf_weights["lm_head.weight"]
    out["output.weight"] = lm_head.T.copy()

    # --- Per-layer weights ---
    for i in range(N_LAYERS):
        print(f"Converting layer {i}/{N_LAYERS - 1}...")
        prefix_hf = f"model.layers.{i}"
        prefix_out = f"blk.{i}"

        # Attention norms
        out[f"{prefix_out}.attn_norm.weight"] = hf_weights[
            f"{prefix_hf}.input_layernorm.weight"
        ]
        out[f"{prefix_out}.ffn_norm.weight"] = hf_weights[
            f"{prefix_hf}.post_attention_layernorm.weight"
        ]

        # Self-attention projections: HF stores [out, in], frank-models
        # expects [in, out] (computes y = x @ W), so transpose each.
        for hf_name, out_name in [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
        ]:
            w = hf_weights[f"{prefix_hf}.self_attn.{hf_name}.weight"]
            out[f"{prefix_out}.{out_name}.weight"] = w.T.copy()

        # QK norms [2048]
        out[f"{prefix_out}.attn_q_norm.weight"] = hf_weights[
            f"{prefix_hf}.self_attn.q_norm.weight"
        ]
        out[f"{prefix_out}.attn_k_norm.weight"] = hf_weights[
            f"{prefix_hf}.self_attn.k_norm.weight"
        ]

        # Dummy zero biases [2048]
        for bias_name in ["attn_q", "attn_k", "attn_v", "attn_output"]:
            out[f"{prefix_out}.{bias_name}.bias"] = np.zeros(
                N_EMBD, dtype=np.float32
            )

        # Router gate [64, 2048]
        out[f"{prefix_out}.ffn_gate_inp.weight"] = hf_weights[
            f"{prefix_hf}.mlp.gate.weight"
        ]

        # Expert weights: stack per-expert into [out_dim, in_dim, 64]
        # up_proj: per-expert [1024, 2048] -> stacked [1024, 2048, 64]
        # gate_proj: per-expert [1024, 2048] -> stacked [1024, 2048, 64]
        # down_proj: per-expert [2048, 1024] -> stacked [2048, 1024, 64]
        for proj_name, out_suffix in [
            ("up_proj", "ffn_up_exps"),
            ("gate_proj", "ffn_gate_exps"),
            ("down_proj", "ffn_down_exps"),
        ]:
            expert_tensors = []
            for e in range(N_EXPERT):
                key = f"{prefix_hf}.mlp.experts.{e}.{proj_name}.weight"
                expert_tensors.append(hf_weights[key])
            # Stack along new last axis (dim 2)
            stacked = np.stack(expert_tensors, axis=-1)
            out[f"{prefix_out}.{out_suffix}.weight"] = stacked

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace OLMoE weights to IREE parameter archive"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to HuggingFace OLMoE checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for IREE .irpa parameter archive",
    )
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        print(f"Error: model directory not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    # Load HF weights
    print(f"Loading weights from {args.model_dir}")
    hf_weights = load_safetensors_shards(args.model_dir)

    # Convert to frank-models naming and layout
    print("Converting weights...")
    converted = convert_weights(hf_weights)

    # Free HF weights to reduce peak memory
    del hf_weights

    # Write IREE parameter archive
    print(f"Writing parameter archive to {args.output}")
    print(f"  Total parameters: {len(converted)}")

    index = ParameterIndex()
    for name, array in sorted(converted.items()):
        arr = np.ascontiguousarray(array)
        index.add_buffer(name, arr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.create_archive_file(str(args.output))
    print(f"Done. Archive written to {args.output}")


if __name__ == "__main__":
    main()
