#!/usr/bin/env python3
"""Benchmark: GPU decode vs decode_body + CPU postprocess."""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts.chat_olmoe import OLMoEChat, OLMOE_CONFIG
from pathlib import Path

params = Path("/home/bourram/models/olmoe-1b-7b-f16-stacked.irpa")
vmfb = Path("/home/bourram/models/olmoe_hetero.vmfb")
model = OLMoEChat(vmfb_path=vmfb, params_path=params, backend="cuda")

cfg = OLMOE_CONFIG
n_layers = cfg["n_layers"]
block_size = model.block_size
batch = 1
prompt_tokens = [510, 3361, 273, 6181, 310]  # "The capital of France is"
seq_len = len(prompt_tokens)
max_new = 20
total_len = seq_len + max_new
max_blocks = (total_len + block_size - 1) // block_size
n_blocks = n_layers * batch * max_blocks

def run_gpu_only():
    """Benchmark: full decode on GPU."""
    cache = model.allocate_kv_cache(n_blocks)
    block_tables = np.zeros((n_layers, batch, max_blocks), dtype=np.int32)
    for l in range(n_layers):
        for blk in range(max_blocks):
            block_tables[l, 0, blk] = l * max_blocks + blk

    # Prefill
    tok = np.array([[prompt_tokens[0]]], dtype=np.int64)
    pos = np.array([[0]], dtype=np.int64)
    sp = np.zeros(batch, dtype=np.int32)
    logits, cache = model.prefill(tok, pos, cache, block_tables, sp)
    last = logits[0, 0, :]
    for i in range(1, seq_len):
        t = np.array([prompt_tokens[i]], dtype=np.int64)
        p = np.array([i], dtype=np.int64)
        ctx = np.full((n_layers, batch), i, dtype=np.int32)
        dl, cache = model.decode(t, p, cache, block_tables, ctx, i)
        last = dl[0, :]
    next_tok = int(np.argmax(last))
    cur_pos = seq_len

    # Timed generation
    tokens = []
    t0 = time.time()
    for step in range(max_new):
        t = np.array([next_tok], dtype=np.int64)
        p = np.array([cur_pos], dtype=np.int64)
        ctx = np.full((n_layers, batch), cur_pos, dtype=np.int32)
        dl, cache = model.decode(t, p, cache, block_tables, ctx, cur_pos)
        next_tok = int(np.argmax(dl[0, :]))
        tokens.append(next_tok)
        cur_pos += 1
    elapsed = time.time() - t0
    return tokens, elapsed

def run_hetero():
    """Benchmark: decode_body on GPU + CPU postprocess."""
    cache = model.allocate_kv_cache(n_blocks)
    block_tables = np.zeros((n_layers, batch, max_blocks), dtype=np.int32)
    for l in range(n_layers):
        for blk in range(max_blocks):
            block_tables[l, 0, blk] = l * max_blocks + blk

    # Prefill (same as GPU-only)
    tok = np.array([[prompt_tokens[0]]], dtype=np.int64)
    pos = np.array([[0]], dtype=np.int64)
    sp = np.zeros(batch, dtype=np.int32)
    logits, cache = model.prefill(tok, pos, cache, block_tables, sp)
    last = logits[0, 0, :]
    for i in range(1, seq_len):
        t = np.array([prompt_tokens[i]], dtype=np.int64)
        p = np.array([i], dtype=np.int64)
        ctx = np.full((n_layers, batch), i, dtype=np.int32)
        dl, cache = model.decode(t, p, cache, block_tables, ctx, i)
        last = dl[0, :]
    next_tok = int(np.argmax(last))
    cur_pos = seq_len

    # Timed generation with hetero
    tokens = []
    gpu_times = []
    cpu_times = []
    t0 = time.time()
    for step in range(max_new):
        t = np.array([next_tok], dtype=np.int64)
        p = np.array([cur_pos], dtype=np.int64)
        ctx = np.full((n_layers, batch), cur_pos, dtype=np.int32)

        tg0 = time.time()
        hidden, cache = model.decode_body(t, p, cache, block_tables, ctx, cur_pos)
        tg1 = time.time()
        gpu_times.append((tg1 - tg0) * 1000)

        tc0 = time.time()
        next_tok = model.cpu_postprocess(hidden)
        tc1 = time.time()
        cpu_times.append((tc1 - tc0) * 1000)

        tokens.append(next_tok)
        cur_pos += 1
    elapsed = time.time() - t0
    return tokens, elapsed, gpu_times, cpu_times

# Warmup
print("Warmup (GPU-only)...")
run_gpu_only()
print("Warmup (hetero)...")
run_hetero()

# Benchmark
print("\n=== GPU-only decode ===")
gpu_tokens, gpu_time = run_gpu_only()
gpu_ms = gpu_time / max_new * 1000
print(f"Total: {gpu_time:.2f}s, per-token: {gpu_ms:.1f}ms, {max_new/gpu_time:.1f} tok/s")
print(f"Tokens: {gpu_tokens}")

print("\n=== Hetero decode_body + CPU postprocess ===")
het_tokens, het_time, gpu_ts, cpu_ts = run_hetero()
het_ms = het_time / max_new * 1000
print(f"Total: {het_time:.2f}s, per-token: {het_ms:.1f}ms, {max_new/het_time:.1f} tok/s")
print(f"Tokens: {het_tokens}")
print(f"GPU decode_body avg: {np.mean(gpu_ts):.1f}ms")
print(f"CPU postprocess avg: {np.mean(cpu_ts):.1f}ms")
print(f"  CPU breakdown: RMSNorm ~{0.005:.3f}ms + matmul ~{np.mean(cpu_ts)-0.005:.1f}ms + argmax ~{0.001:.3f}ms")

print(f"\n=== Comparison ===")
print(f"GPU-only:  {gpu_ms:.1f}ms/token")
print(f"Hetero:    {het_ms:.1f}ms/token")
speedup = (gpu_ms - het_ms) / gpu_ms * 100
print(f"Speedup:   {speedup:+.1f}%")
print(f"Tokens match: {gpu_tokens == het_tokens}")
