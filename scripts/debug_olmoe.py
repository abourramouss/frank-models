"""Compare IREE OLMoE outputs vs HuggingFace reference to isolate bugs.

Compares intermediate values step by step:
1. Embedding lookup
2. Layer 0 attention norm
3. Layer 0 Q/K/V projections
4. Layer 0 QK norm
5. Full prefill logits

Usage:
    python scripts/debug_olmoe.py
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

IREE_PY = [
    "/home/bourram/iree-install/python_packages/iree_compiler",
    "/home/bourram/iree-install/python_packages/iree_runtime",
]
for p in IREE_PY:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = Path("/home/bourram/models/olmoe-1b-7b")


def load_hf_model():
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(WEIGHTS_DIR), torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    return model


def load_irpa_params():
    """Load parameters from the IRPA archive."""
    from iree.runtime import ParameterIndex

    print("Loading IRPA parameters...")
    idx = ParameterIndex()
    idx.load(str(WEIGHTS_DIR / "olmoe.irpa"))
    params = {}
    for i in range(len(idx)):
        entry = idx[i]
        name = entry.key
        data = np.frombuffer(entry.file_view, dtype=np.float32).copy()
        params[name] = data
    return params


def compare(name, a, b, rtol=1e-5, atol=1e-6):
    """Compare two arrays and print diagnostics."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH iree={a.shape} hf={b.shape}")
        return False
    diff = np.abs(a - b)
    max_diff = diff.max()
    mean_diff = diff.mean()
    close = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "OK" if close else "MISMATCH"
    print(
        f"  {name}: {status} max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} "
        f"iree_range=[{a.min():.4f},{a.max():.4f}] hf_range=[{b.min():.4f},{b.max():.4f}]"
    )
    if not close:
        # Show first few differing values
        flat_a, flat_b = a.flatten(), b.flatten()
        worst_idx = np.argmax(diff.flatten())
        print(
            f"    worst at flat[{worst_idx}]: iree={flat_a[worst_idx]:.6f} hf={flat_b[worst_idx]:.6f}"
        )
    return close


def main():
    hf_model = load_hf_model()
    irpa = load_irpa_params()

    # Token for "Hello"
    token_id = 12092
    input_ids = torch.tensor([[token_id]])

    # ===== 1. Compare embedding =====
    print("\n=== Embedding ===")
    hf_emb_weight = hf_model.model.embed_tokens.weight.detach().numpy()
    irpa_emb = irpa["token_embd.weight"].reshape(hf_emb_weight.shape)
    compare("embed_weight", irpa_emb, hf_emb_weight)

    hf_embedding = hf_emb_weight[token_id]  # [n_embd]
    irpa_embedding = irpa_emb[token_id]
    compare("token_12092_embedding", irpa_embedding, hf_embedding)

    # ===== 2. Compare layer 0 norms =====
    print("\n=== Layer 0 Norms ===")
    hf_attn_norm_w = (
        hf_model.model.layers[0].input_layernorm.weight.detach().numpy()
    )
    irpa_attn_norm = irpa["blk.0.attn_norm.weight"].reshape(hf_attn_norm_w.shape)
    compare("attn_norm_weight", irpa_attn_norm, hf_attn_norm_w)

    # RMS norm on embedding
    def rms_norm(x, w, eps=1e-5):
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * w

    hf_normed = rms_norm(hf_embedding, hf_attn_norm_w)
    irpa_normed = rms_norm(irpa_embedding, irpa_attn_norm)
    compare("attn_norm_output", irpa_normed, hf_normed)

    # ===== 3. Compare Q/K/V projections =====
    print("\n=== Layer 0 Attention Projections ===")
    hf_q_w = hf_model.model.layers[0].self_attn.q_proj.weight.detach().numpy()
    hf_k_w = hf_model.model.layers[0].self_attn.k_proj.weight.detach().numpy()
    hf_v_w = hf_model.model.layers[0].self_attn.v_proj.weight.detach().numpy()

    irpa_q_w = irpa["blk.0.attn_q.weight"].reshape(hf_q_w.T.shape)
    irpa_k_w = irpa["blk.0.attn_k.weight"].reshape(hf_k_w.T.shape)
    irpa_v_w = irpa["blk.0.attn_v.weight"].reshape(hf_v_w.T.shape)

    compare("q_weight (irpa vs hf.T)", irpa_q_w, hf_q_w.T)
    compare("k_weight (irpa vs hf.T)", irpa_k_w, hf_k_w.T)
    compare("v_weight (irpa vs hf.T)", irpa_v_w, hf_v_w.T)

    # Compute Q/K/V
    # MLIR: input @ weight (weight is [n_embd, n_embd])
    # HF: F.linear(input, weight) = input @ weight.T
    hf_q = hf_normed @ hf_q_w.T  # [n_embd]
    irpa_q = irpa_normed @ irpa_q_w  # [n_embd]
    compare("q_proj_output", irpa_q, hf_q)

    hf_k = hf_normed @ hf_k_w.T
    irpa_k = irpa_normed @ irpa_k_w
    compare("k_proj_output", irpa_k, hf_k)

    # ===== 4. Compare QK norm =====
    print("\n=== Layer 0 QK Norm ===")
    hf_q_norm_w = hf_model.model.layers[0].self_attn.q_norm.weight.detach().numpy()
    hf_k_norm_w = hf_model.model.layers[0].self_attn.k_norm.weight.detach().numpy()

    irpa_q_norm_w = irpa["blk.0.attn_q_norm.weight"].reshape(hf_q_norm_w.shape)
    irpa_k_norm_w = irpa["blk.0.attn_k_norm.weight"].reshape(hf_k_norm_w.shape)

    compare("q_norm_weight", irpa_q_norm_w, hf_q_norm_w)
    compare("k_norm_weight", irpa_k_norm_w, hf_k_norm_w)

    # Apply QK norm
    hf_q_normed = rms_norm(hf_q, hf_q_norm_w)
    irpa_q_normed = rms_norm(irpa_q, irpa_q_norm_w)
    compare("q_normed", irpa_q_normed, hf_q_normed)

    # ===== 5. Compare RoPE =====
    print("\n=== Layer 0 RoPE ===")
    from oracles.position import rope as rope_oracle

    # Reshape for RoPE: [1, 1, n_head, head_dim]
    n_head, head_dim = 16, 128
    q_4d = hf_q_normed.reshape(1, 1, n_head, head_dim)
    positions = np.array([[0]], dtype=np.int64)

    q_rope = rope_oracle(q_4d, positions, freq_base=10000.0, freq_scale=1.0)
    print(f"  q_rope (oracle) first head: {q_rope[0, 0, 0, :4]}")

    # HF RoPE for comparison
    with torch.no_grad():
        hf_out = hf_model(input_ids, output_hidden_states=True)

    hf_logits = hf_out.logits[0, 0].detach().numpy()  # [vocab]
    print(f"\n=== Final Logits ===")
    print(f"  HF argmax: {np.argmax(hf_logits)} value: {hf_logits.max():.4f}")
    print(f"  HF logits stats: min={hf_logits.min():.4f} max={hf_logits.max():.4f} std={hf_logits.std():.4f}")
    print(f"  HF top-5: {np.argsort(hf_logits)[-5:][::-1]}")

    # Compare with IREE logits
    # Load IREE runtime and run
    print("\n=== Running IREE inference ===")
    from models.llm.olmoe.infer import load_runtime, OLMoERunner, VMFB_CACHE, IRPA_PATH
    context, vm_module, device = load_runtime(VMFB_CACHE, IRPA_PATH)
    runner = OLMoERunner(context, vm_module, device)

    N_LAYERS = 16
    block_size = 16
    max_blocks = 2
    n_blocks = N_LAYERS * max_blocks

    cache = runner.allocate_kv_cache(n_blocks, block_size)
    tokens = np.array([[token_id]], dtype=np.int64)
    positions_iree = np.array([[0]], dtype=np.int64)
    block_tables = np.zeros((N_LAYERS, 1, max_blocks), dtype=np.int32)
    for layer in range(N_LAYERS):
        for blk in range(max_blocks):
            block_tables[layer, 0, blk] = layer * max_blocks + blk
    start_positions = np.zeros(1, dtype=np.int32)

    logits, _ = runner.prefill(
        tokens, positions_iree, cache, block_tables, start_positions, block_size
    )
    iree_logits = logits[0, 0]  # [vocab]

    print(f"  IREE argmax: {np.argmax(iree_logits)} value: {iree_logits.max():.4f}")
    print(f"  IREE logits stats: min={iree_logits.min():.4f} max={iree_logits.max():.4f} std={iree_logits.std():.4f}")
    print(f"  IREE top-5: {np.argsort(iree_logits)[-5:][::-1]}")

    compare("final_logits", iree_logits, hf_logits, rtol=1e-3, atol=1e-2)

    # Check hidden states
    print("\n=== Hidden States ===")
    for i, hs in enumerate(hf_out.hidden_states):
        hs_np = hs[0, 0].detach().numpy()
        print(f"  HF layer {i}: min={hs_np.min():.4f} max={hs_np.max():.4f} std={hs_np.std():.4f}")


if __name__ == "__main__":
    main()
