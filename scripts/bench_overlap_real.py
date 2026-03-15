#!/usr/bin/env python3
"""Benchmark: GPU-only vs hetero-numpy vs pipelined overlap."""
import sys, time, threading, os
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts.chat_olmoe import OLMoEChat, OLMOE_CONFIG
from pathlib import Path

def get_cpu_times():
    with open("/proc/stat") as f:
        parts = f.readline().split()
    busy = int(parts[1]) + int(parts[2]) + int(parts[3])
    total = sum(int(p) for p in parts[1:8])
    return busy, total

cfg = OLMOE_CONFIG
n_layers = cfg["n_layers"]
prompt_tokens = [510, 3361, 273, 6181, 310]
seq_len = len(prompt_tokens)
max_new = 30
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

def prefill_prompt(model):
    cache = model.allocate_kv_cache(n_blocks)
    bt = make_bt()
    tok_arr = np.array([[prompt_tokens[0]]], dtype=np.int64)
    pos_arr = np.array([[0]], dtype=np.int64)
    sp = np.zeros(1, dtype=np.int32)
    logits, cache = model.prefill(tok_arr, pos_arr, cache, bt, sp)
    last = logits[0, 0, :]
    for i in range(1, seq_len):
        dl, cache = model.decode(
            np.array([prompt_tokens[i]], dtype=np.int64),
            np.array([i], dtype=np.int64),
            cache, bt,
            np.full((n_layers, 1), i, dtype=np.int32), i)
        last = dl[0, :]
    return int(np.argmax(last)), cache, bt

# ===== Mode 1: GPU-only =====
def run_gpu_only():
    tok, cache, bt = prefill_prompt(model)
    busy0, total0 = get_cpu_times()
    t0 = time.time()
    tokens = []
    for step in range(max_new):
        dl, cache = model.decode(
            np.array([tok], dtype=np.int64),
            np.array([seq_len + step], dtype=np.int64),
            cache, bt,
            np.full((n_layers, 1), seq_len + step, dtype=np.int32),
            seq_len + step)
        tok = int(np.argmax(dl[0, :]))
        tokens.append(tok)
    elapsed = time.time() - t0
    busy1, total1 = get_cpu_times()
    cpu = (busy1 - busy0) / max(1, total1 - total0) * 100
    return tokens, elapsed, cpu

# ===== Mode 2: Hetero numpy (sequential) =====
def run_hetero_numpy():
    tok, cache, bt = prefill_prompt(model)
    busy0, total0 = get_cpu_times()
    t0 = time.time()
    tokens = []
    for step in range(max_new):
        hidden, cache = model.decode_body(
            np.array([tok], dtype=np.int64),
            np.array([seq_len + step], dtype=np.int64),
            cache, bt,
            np.full((n_layers, 1), seq_len + step, dtype=np.int32),
            seq_len + step)
        tok = model.cpu_postprocess(hidden)
        tokens.append(tok)
    elapsed = time.time() - t0
    busy1, total1 = get_cpu_times()
    cpu = (busy1 - busy0) / max(1, total1 - total0) * 100
    return tokens, elapsed, cpu

# ===== Mode 3: Hetero numpy with GPU+CPU overlap =====
# GPU runs decode_body(N+1) while CPU postprocesses hidden(N)
# Can't truly overlap because GPU needs token from CPU.
# But we CAN overlap the CPU postprocess with _to_bv preparation.
# More importantly: we can time the GPU and CPU parts separately.
def run_hetero_timed():
    tok, cache, bt = prefill_prompt(model)
    busy0, total0 = get_cpu_times()
    t0 = time.time()
    tokens = []
    gpu_times = []
    cpu_times = []
    for step in range(max_new):
        tg0 = time.time()
        hidden, cache = model.decode_body(
            np.array([tok], dtype=np.int64),
            np.array([seq_len + step], dtype=np.int64),
            cache, bt,
            np.full((n_layers, 1), seq_len + step, dtype=np.int32),
            seq_len + step)
        tg1 = time.time()
        gpu_times.append((tg1 - tg0) * 1000)

        tc0 = time.time()
        tok = model.cpu_postprocess(hidden)
        tc1 = time.time()
        cpu_times.append((tc1 - tc0) * 1000)
        tokens.append(tok)
    elapsed = time.time() - t0
    busy1, total1 = get_cpu_times()
    cpu = (busy1 - busy0) / max(1, total1 - total0) * 100
    return tokens, elapsed, cpu, gpu_times, cpu_times

# ===== Mode 4: Hetero IREE CPU =====
def run_hetero_iree():
    tok, cache, bt = prefill_prompt(model)
    busy0, total0 = get_cpu_times()
    t0 = time.time()
    tokens = []
    for step in range(max_new):
        hidden, cache = model.decode_body(
            np.array([tok], dtype=np.int64),
            np.array([seq_len + step], dtype=np.int64),
            cache, bt,
            np.full((n_layers, 1), seq_len + step, dtype=np.int32),
            seq_len + step)
        logits = model.cpu_postprocess_iree(hidden)
        tok = int(np.argmax(logits[0, :]))
        tokens.append(tok)
    elapsed = time.time() - t0
    busy1, total1 = get_cpu_times()
    cpu = (busy1 - busy0) / max(1, total1 - total0) * 100
    return tokens, elapsed, cpu

print(f"OLMoE-1B-7B on Jetson AGX Orin | {os.cpu_count()} CPU cores | {max_new} tokens")
print("=" * 75)

# Warmup
run_gpu_only()
run_hetero_numpy()
run_hetero_iree()

# Benchmark
print(f"\n{'Mode':<40} {'ms/tok':>8} {'tok/s':>8} {'CPU%':>8} {'speedup':>8}")
print("-" * 75)

ref, t, c = run_gpu_only()
gpu_ms = t / max_new * 1000
print(f"{'GPU-only (decode on CUDA)':<40} {gpu_ms:>7.1f} {max_new/t:>8.1f} {c:>7.1f}% {'1.00x':>8}")

tokens, t, c = run_hetero_iree()
ms = t / max_new * 1000
print(f"{'Hetero: decode_body + IREE CPU':<40} {ms:>7.1f} {max_new/t:>8.1f} {c:>7.1f}% {gpu_ms/ms:>7.2f}x")

tokens, t, c, gt, ct = run_hetero_timed()
ms = t / max_new * 1000
print(f"{'Hetero: decode_body + numpy CPU':<40} {ms:>7.1f} {max_new/t:>8.1f} {c:>7.1f}% {gpu_ms/ms:>7.2f}x")

print(f"\n--- Timing breakdown (numpy hetero) ---")
print(f"  GPU decode_body:    {np.mean(gt):>7.1f}ms avg ({np.min(gt):.1f}-{np.max(gt):.1f})")
print(f"  CPU postprocess:    {np.mean(ct):>7.1f}ms avg ({np.min(ct):.1f}-{np.max(ct):.1f})")
print(f"  Total sequential:   {np.mean(gt)+np.mean(ct):>7.1f}ms")
print(f"  Theoretical overlap:{max(np.mean(gt),np.mean(ct)):>7.1f}ms (if pipelined)")
print(f"  Potential speedup:  {(np.mean(gt)+np.mean(ct))/max(np.mean(gt),np.mean(ct)):>7.2f}x from pipelining")
print(f"  Tokens match GPU:   {'YES' if tokens == ref else 'NO'}")
