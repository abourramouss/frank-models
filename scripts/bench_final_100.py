#!/usr/bin/env python3
"""Final benchmark: GPU-only vs Hetero IREE CPU over 100 runs."""
import sys, time, os
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts.chat_olmoe import OLMoEChat, OLMOE_CONFIG
from pathlib import Path

cfg = OLMOE_CONFIG
n_layers = cfg["n_layers"]
prompt_tokens = [510, 3361, 273, 6181, 310]  # "The capital of France is"
seq_len = len(prompt_tokens)
max_new = 20
block_size = 16
total_len = seq_len + max_new
max_blocks = (total_len + block_size - 1) // block_size
n_blocks = n_layers * max_blocks

params = Path("/home/bourram/models/olmoe-1b-7b-f16-stacked.irpa")
vmfb = Path("/home/bourram/models/olmoe_hetero.vmfb")
cpu_vmfb = Path("/home/bourram/models/olmoe_cpu_post.vmfb")
model = OLMoEChat(vmfb_path=vmfb, params_path=params, backend="cuda",
                  cpu_vmfb_path=cpu_vmfb)

def make_bt():
    bt = np.zeros((n_layers, 1, max_blocks), dtype=np.int32)
    for l in range(n_layers):
        for blk in range(max_blocks):
            bt[l, 0, blk] = l * max_blocks + blk
    return bt

def run_once(mode):
    cache = model.allocate_kv_cache(n_blocks)
    bt = make_bt()
    tok = np.array([[prompt_tokens[0]]], dtype=np.int64)
    pos = np.array([[0]], dtype=np.int64)
    sp = np.zeros(1, dtype=np.int32)
    logits, cache = model.prefill(tok, pos, cache, bt, sp)
    last = logits[0, 0, :]
    for i in range(1, seq_len):
        dl, cache = model.decode(
            np.array([prompt_tokens[i]], dtype=np.int64),
            np.array([i], dtype=np.int64),
            cache, bt,
            np.full((n_layers, 1), i, dtype=np.int32), i)
        last = dl[0, :]
    next_tok = int(np.argmax(last))
    cur_pos = seq_len

    t0 = time.time()
    for step in range(max_new):
        t = np.array([next_tok], dtype=np.int64)
        p = np.array([cur_pos], dtype=np.int64)
        ctx = np.full((n_layers, 1), cur_pos, dtype=np.int32)
        if mode == "gpu_only":
            dl, cache = model.decode(t, p, cache, bt, ctx, cur_pos)
            next_tok = int(np.argmax(dl[0, :]))
        elif mode == "hetero_iree":
            hidden, cache = model.decode_body(t, p, cache, bt, ctx, cur_pos)
            logits = model.cpu_postprocess_iree(hidden)
            next_tok = int(np.argmax(logits[0, :]))
        cur_pos += 1
    return (time.time() - t0) / max_new * 1000  # ms/token

N_RUNS = 100

print(f"OLMoE-1B-7B on Jetson AGX Orin")
print(f"Generating {max_new} tokens x {N_RUNS} runs per mode")
print(f"=" * 60)

# Warmup
for _ in range(3):
    run_once("gpu_only")
    run_once("hetero_iree")

results = {}
for mode, label in [("gpu_only", "GPU-only"), ("hetero_iree", "Hetero IREE CPU")]:
    times = []
    for i in range(N_RUNS):
        ms = run_once(mode)
        times.append(ms)
        if (i + 1) % 25 == 0:
            print(f"  {label}: {i+1}/{N_RUNS} done ({np.mean(times):.1f} ms/tok avg)", flush=True)
    results[mode] = times

print(f"\n{'':=<60}")
print(f"{'Metric':<25} {'GPU-only':>15} {'Hetero IREE':>15}")
print(f"{'-'*25} {'-'*15} {'-'*15}")

gpu = np.array(results["gpu_only"])
het = np.array(results["hetero_iree"])

print(f"{'Mean ms/tok':<25} {np.mean(gpu):>14.1f} {np.mean(het):>14.1f}")
print(f"{'Median ms/tok':<25} {np.median(gpu):>14.1f} {np.median(het):>14.1f}")
print(f"{'Std ms/tok':<25} {np.std(gpu):>14.1f} {np.std(het):>14.1f}")
print(f"{'Min ms/tok':<25} {np.min(gpu):>14.1f} {np.min(het):>14.1f}")
print(f"{'Max ms/tok':<25} {np.max(gpu):>14.1f} {np.max(het):>14.1f}")
print(f"{'P5 ms/tok':<25} {np.percentile(gpu,5):>14.1f} {np.percentile(het,5):>14.1f}")
print(f"{'P95 ms/tok':<25} {np.percentile(gpu,95):>14.1f} {np.percentile(het,95):>14.1f}")
print(f"{'Mean tok/s':<25} {1000/np.mean(gpu):>14.1f} {1000/np.mean(het):>14.1f}")
print(f"{'Speedup':<25} {'1.00x':>15} {np.mean(gpu)/np.mean(het):>14.2f}x")
