"""Convert OLMoE HuggingFace safetensors weights to IREE parameter archive (.irpa).

Maps HF naming (model.layers.{i}.self_attn.q_proj.weight) to GGUF naming
(blk.{i}.attn_q.weight) as expected by params.mlir.

Usage:
    python models/llm/olmoe/convert_weights.py \
        --input /home/bourram/models/olmoe-1b-7b \
        --output /home/bourram/models/olmoe-1b-7b/olmoe.irpa
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

PYTHONPATH_IREE = [
    "/home/bourram/iree-install/python_packages/iree_compiler",
    "/home/bourram/iree-install/python_packages/iree_runtime",
]
for p in PYTHONPATH_IREE:
    if p not in sys.path:
        sys.path.insert(0, p)

from iree.runtime import ParameterIndex  # noqa: E402


N_LAYERS = 16
N_EXPERTS = 64


def load_safetensors(weights_dir: Path) -> dict[str, np.ndarray]:
    """Load all tensors from sharded safetensors files."""
    index_path = weights_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())

    # Map shard filename → list of tensor names
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        shard_to_keys.setdefault(shard, []).append(key)

    tensors: dict[str, np.ndarray] = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = weights_dir / shard_name
        print(f"  loading {shard_name} ({len(keys)} tensors)...")
        with safetensors.torch.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key).to(torch.float32).numpy()
    return tensors


def convert(weights_dir: Path, output_path: Path) -> None:
    print("Loading safetensors...")
    hf = load_safetensors(weights_dir)

    index = ParameterIndex()

    def add(name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr.astype(np.float32))
        index.add_buffer(name, arr)

    print("Converting model-level tensors...")
    # token_embd.weight: [vocab, n_embd] — no transpose (embedding lookup by row)
    add("token_embd.weight", hf["model.embed_tokens.weight"])
    # output_norm.weight: [n_embd]
    add("output_norm.weight", hf["model.norm.weight"])
    # output.weight: params.mlir expects [n_embd, vocab] — transpose lm_head
    add("output.weight", hf["lm_head.weight"].T)

    print("Converting layer tensors...")
    for i in range(N_LAYERS):
        print(f"  layer {i}...")
        p = f"model.layers.{i}"
        b = f"blk.{i}"

        # Norms
        add(f"{b}.attn_norm.weight", hf[f"{p}.input_layernorm.weight"])
        add(f"{b}.ffn_norm.weight", hf[f"{p}.post_attention_layernorm.weight"])

        # Attention projections: HF stores [out, in], MLIR expects [in, out]
        add(f"{b}.attn_q.weight", hf[f"{p}.self_attn.q_proj.weight"].T)
        add(f"{b}.attn_k.weight", hf[f"{p}.self_attn.k_proj.weight"].T)
        add(f"{b}.attn_v.weight", hf[f"{p}.self_attn.v_proj.weight"].T)
        add(f"{b}.attn_output.weight", hf[f"{p}.self_attn.o_proj.weight"].T)

        # QK norms (OLMoE uses per-head RMSNorm on Q and K)
        add(f"{b}.attn_q_norm.weight", hf[f"{p}.self_attn.q_norm.weight"])
        add(f"{b}.attn_k_norm.weight", hf[f"{p}.self_attn.k_norm.weight"])

        # Dummy zero biases (params.mlir loads these; use_bias=false so they're ignored)
        zero_q = np.zeros(2048, dtype=np.float32)
        zero_kv = np.zeros(2048, dtype=np.float32)
        zero_o = np.zeros(2048, dtype=np.float32)
        add(f"{b}.attn_q.bias", zero_q)
        add(f"{b}.attn_k.bias", zero_kv)
        add(f"{b}.attn_v.bias", zero_kv)
        add(f"{b}.attn_output.bias", zero_o)

        # MoE router: [n_expert, n_embd] — no transpose (logits = gate_w @ input.T)
        add(f"{b}.ffn_gate_inp.weight", hf[f"{p}.mlp.gate.weight"])

        # Stack expert weights along last axis
        # up/gate: each [n_ff, n_embd] → stacked [n_ff, n_embd, n_expert]
        # down:    each [n_embd, n_ff] → stacked [n_embd, n_ff, n_expert]
        up_list = [hf[f"{p}.mlp.experts.{j}.up_proj.weight"] for j in range(N_EXPERTS)]
        gate_list = [hf[f"{p}.mlp.experts.{j}.gate_proj.weight"] for j in range(N_EXPERTS)]
        down_list = [hf[f"{p}.mlp.experts.{j}.down_proj.weight"] for j in range(N_EXPERTS)]

        add(f"{b}.ffn_up_exps.weight", np.stack(up_list, axis=-1))
        add(f"{b}.ffn_gate_exps.weight", np.stack(gate_list, axis=-1))
        add(f"{b}.ffn_down_exps.weight", np.stack(down_list, axis=-1))

    print(f"Writing {output_path}...")
    index.create_archive_file(str(output_path))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/home/bourram/models/olmoe-1b-7b"))
    parser.add_argument("--output", type=Path, default=Path("/home/bourram/models/olmoe-1b-7b/olmoe.irpa"))
    args = parser.parse_args()
    convert(args.input, args.output)
