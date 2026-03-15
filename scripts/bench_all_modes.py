#!/usr/bin/env python3
"""Benchmark all decode modes: GPU-only, hetero-numpy, hetero-IREE-CPU."""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts.chat_olmoe import OLMoEChat, OLMOE_CONFIG
from pathlib import Path

cfg = OLMOE_CONFIG
n_layers = cfg["n_layers"]
prompt_tokens = [510, 3361, 273, 6181, 310]
seq_len = len(prompt_tokens)
max_new = 20
block_size = 16
total_len = seq_len + max_new
max_blocks = (total_len + block_size - 1) // block_size
n_blocks = n_layers * 1 * max_blocks

def make_block_tables():
    bt = np.zeros((n_layers, 1, max_blocks), dtype=np.int32)
    for l in range(n_layers):
        for blk in range(max_blocks):
            bt[l, 0, blk] = l * max_blocks + blk
    return bt

def run_decode(model, mode="gpu_only"):
    cache = model.allocate_kv_cache(n_blocks)
    bt = make_block_tables()
    tok = np.array([[prompt_tokens[0]]], dtype=np.int64)
    pos = np.array([[0]], dtype=np.int64)
    sp = np.zeros(1, dtype=np.int32)
    logits, cache = model.prefill(tok, pos, cache, bt, sp)
    last = logits[0, 0, :]
    for i in range(1, seq_len):
        t = np.array([prompt_tokens[i]], dtype=np.int64)
        p = np.array([i], dtype=np.int64)
        ctx = np.full((n_layers, 1), i, dtype=np.int32)
        dl, cache = model.decode(t, p, cache, bt, ctx, i)
        last = dl[0, :]
    next_tok = int(np.argmax(last))
    cur_pos = seq_len
    tokens = []
    t0 = time.time()
    for step in range(max_new):
        t = np.array([next_tok], dtype=np.int64)
        p = np.array([cur_pos], dtype=np.int64)
        ctx = np.full((n_layers, 1), cur_pos, dtype=np.int32)
        if mode == "gpu_only":
            dl, cache = model.decode(t, p, cache, bt, ctx, cur_pos)
            next_tok = int(np.argmax(dl[0, :]))
        elif mode == "hetero_numpy":
            hidden, cache = model.decode_body(t, p, cache, bt, ctx, cur_pos)
            next_tok = model.cpu_postprocess(hidden)
        elif mode == "hetero_iree":
            hidden, cache = model.decode_body(t, p, cache, bt, ctx, cur_pos)
            logits = model.cpu_postprocess_iree(hidden)
            next_tok = int(np.argmax(logits[0, :]))
        tokens.append(next_tok)
        cur_pos += 1
    elapsed = time.time() - t0
    return tokens, elapsed

params = Path("/home/bourram/models/olmoe-1b-7b-f16-stacked.irpa")
vmfb = Path("/home/bourram/models/olmoe_hetero.vmfb")
cpu_vmfb = Path("/home/bourram/models/olmoe_cpu_post.vmfb")

model = OLMoEChat(vmfb_path=vmfb, params_path=params, backend="cuda",
                  cpu_vmfb_path=cpu_vmfb)

modes = [
    ("gpu_only", "GPU-only (all on CUDA)"),
    ("hetero_numpy", "Hetero: GPU decode_body + numpy CPU"),
    ("hetero_iree", "Hetero: GPU decode_body + IREE CPU VMFB"),
]

# Warmup
for mode, _ in modes:
    run_decode(model, mode)

print(f"\n{'Mode':<45} {'ms/tok':>8} {'tok/s':>8} {'tokens match':>12}")
print("-" * 75)

ref_tokens = None
for mode, desc in modes:
    tokens, elapsed = run_decode(model, mode)
    ms = elapsed / max_new * 1000
    tps = max_new / elapsed
    if ref_tokens is None:
        ref_tokens = tokens
    match = "YES" if tokens == ref_tokens else "NO"
    print(f"{desc:<45} {ms:>7.1f} {tps:>8.1f} {match:>12}")
