#!/usr/bin/env python3
"""Benchmark GPU-only vs Hetero IREE CPU with CPU utilization monitoring."""
import sys, time, threading, os
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts.chat_olmoe import OLMoEChat, OLMOE_CONFIG
from pathlib import Path

def get_cpu_times():
    """Read /proc/stat for total CPU times."""
    with open("/proc/stat") as f:
        line = f.readline()  # first line is aggregate
    parts = line.split()
    # user, nice, system, idle, iowait, irq, softirq, steal
    user = int(parts[1]) + int(parts[2])
    system = int(parts[3])
    idle = int(parts[4]) + int(parts[5])
    total = user + system + idle + int(parts[6]) + int(parts[7])
    return user + system, total  # busy, total

def monitor_cpu(stop_event, samples):
    """Sample CPU utilization every 50ms."""
    while not stop_event.is_set():
        busy, total = get_cpu_times()
        samples.append((time.time(), busy, total))
        time.sleep(0.05)

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

    # Start CPU monitor
    stop = threading.Event()
    samples = []
    monitor = threading.Thread(target=monitor_cpu, args=(stop, samples))
    monitor.start()

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

    stop.set()
    monitor.join()

    # Calculate CPU utilization during generation
    if len(samples) >= 2:
        busy_start, total_start = samples[0][1], samples[0][2]
        busy_end, total_end = samples[-1][1], samples[-1][2]
        cpu_util = (busy_end - busy_start) / max(1, (total_end - total_start)) * 100
        n_cpus = os.cpu_count()
        # Per-core utilization (the above is aggregate across all cores)
        cpu_util_per_core = cpu_util
    else:
        cpu_util_per_core = 0

    return tokens, elapsed, cpu_util_per_core, len(samples)

params = Path("/home/bourram/models/olmoe-1b-7b-f16-stacked.irpa")
vmfb = Path("/home/bourram/models/olmoe_hetero.vmfb")
cpu_vmfb = Path("/home/bourram/models/olmoe_cpu_post.vmfb")

model = OLMoEChat(vmfb_path=vmfb, params_path=params, backend="cuda",
                  cpu_vmfb_path=cpu_vmfb)

modes = [
    ("gpu_only", "GPU-only (all on CUDA)"),
    ("hetero_numpy", "Hetero: decode_body + numpy CPU"),
    ("hetero_iree", "Hetero: decode_body + IREE CPU VMFB"),
]

# Warmup
for mode, _ in modes:
    run_decode(model, mode)

n_cpus = os.cpu_count()
print(f"\nCPU cores: {n_cpus}")
print(f"{'Mode':<45} {'ms/tok':>8} {'tok/s':>8} {'CPU%':>6} {'match':>6}")
print("-" * 80)

ref_tokens = None
for mode, desc in modes:
    tokens, elapsed, cpu_pct, n_samples = run_decode(model, mode)
    ms = elapsed / max_new * 1000
    tps = max_new / elapsed
    if ref_tokens is None:
        ref_tokens = tokens
    match = "YES" if tokens == ref_tokens else "NO"
    print(f"{desc:<45} {ms:>7.1f} {tps:>8.1f} {cpu_pct:>5.1f}% {match:>6}")
