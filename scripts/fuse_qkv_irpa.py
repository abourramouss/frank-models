#!/usr/bin/env python3
"""Convert Qwen3-0.6B IRPA from separate Q/K/V to fused QKV format.

Reads: /home/bourram/models/qwen3-0.6b-f16-stacked.irpa
  - stacked.attn_q.weight:  [28, 2097152] (per layer: [1024, 2048])
  - stacked.attn_k.weight:  [28, 1048576] (per layer: [1024, 1024])
  - stacked.attn_v.weight:  [28, 1048576] (per layer: [1024, 1024])

Writes: /home/bourram/models/qwen3-0.6b-f16-fused-qkv.irpa
  - stacked.attn_qkv.weight: [28, 4194304] (per layer: [1024, 4096])
    where cols 0:2048 = Q, 2048:3072 = K, 3072:4096 = V

All other weights are copied unchanged.
"""
import numpy as np
from iree.runtime import ParameterIndex

INPUT = "/home/bourram/models/qwen3-0.6b-f16-stacked.irpa"
OUTPUT = "/home/bourram/models/qwen3-0.6b-f16-fused-qkv.irpa"

N_LAYERS = 28
N_EMBD = 1024
N_EMBD_Q = 2048   # n_head * head_dim = 16 * 128
N_EMBD_KV = 1024  # n_head_kv * head_dim = 8 * 128

QKV_COLS = N_EMBD_Q + 2 * N_EMBD_KV  # 4096
QKV_FLAT = N_EMBD * QKV_COLS           # 4194304

SKIP_KEYS = {"stacked.attn_q.weight", "stacked.attn_k.weight", "stacked.attn_v.weight"}

# --- Load source index ---
print(f"Loading {INPUT}...")
src = ParameterIndex()
src.load(INPUT)

# --- Build destination index ---
dst = ParameterIndex()

q_data = None
k_data = None
v_data = None

for name, entry in src.items():
    if name in SKIP_KEYS:
        # Read Q/K/V into numpy arrays, fuse later
        raw = np.frombuffer(entry.file_view, dtype=np.float16).copy()
        if name == "stacked.attn_q.weight":
            q_data = raw.reshape(N_LAYERS, -1)
            print(f"  Read {name}: {q_data.shape}")
        elif name == "stacked.attn_k.weight":
            k_data = raw.reshape(N_LAYERS, -1)
            print(f"  Read {name}: {k_data.shape}")
        elif name == "stacked.attn_v.weight":
            v_data = raw.reshape(N_LAYERS, -1)
            print(f"  Read {name}: {v_data.shape}")
    else:
        # Copy unchanged
        data = np.frombuffer(entry.file_view, dtype=np.float16).copy()
        dst.add_buffer(name, data)
        print(f"  Copied {name}: {data.shape}")

# --- Fuse QKV ---
assert q_data is not None and k_data is not None and v_data is not None, "Missing Q/K/V weights"
assert q_data.shape == (N_LAYERS, N_EMBD * N_EMBD_Q), f"Q shape mismatch: {q_data.shape}"
assert k_data.shape == (N_LAYERS, N_EMBD * N_EMBD_KV), f"K shape mismatch: {k_data.shape}"
assert v_data.shape == (N_LAYERS, N_EMBD * N_EMBD_KV), f"V shape mismatch: {v_data.shape}"

print(f"\nFusing QKV for {N_LAYERS} layers...")
# Reshape each to [N_LAYERS, N_EMBD, cols], concatenate along cols, flatten back
q_3d = q_data.reshape(N_LAYERS, N_EMBD, N_EMBD_Q)    # [28, 1024, 2048]
k_3d = k_data.reshape(N_LAYERS, N_EMBD, N_EMBD_KV)   # [28, 1024, 1024]
v_3d = v_data.reshape(N_LAYERS, N_EMBD, N_EMBD_KV)   # [28, 1024, 1024]

qkv_3d = np.concatenate([q_3d, k_3d, v_3d], axis=2)   # [28, 1024, 4096]
assert qkv_3d.shape == (N_LAYERS, N_EMBD, QKV_COLS), f"Fused shape mismatch: {qkv_3d.shape}"

qkv_flat = qkv_3d.reshape(N_LAYERS, QKV_FLAT)          # [28, 4194304]
print(f"  stacked.attn_qkv.weight: {qkv_flat.shape} ({qkv_flat.nbytes / (1024**2):.0f} MB)")

dst.add_buffer("stacked.attn_qkv.weight", qkv_flat)

# Free intermediates
del q_data, k_data, v_data, q_3d, k_3d, v_3d, qkv_3d

# --- Save ---
print(f"\nSaving to {OUTPUT}...")
dst.create_archive_file(OUTPUT)
print("Done!")

# --- Verify ---
print("\nVerifying output...")
check = ParameterIndex()
check.load(OUTPUT)
for name, entry in check.items():
    data = np.frombuffer(entry.file_view, dtype=np.float16)
    print(f"  {name}: {len(data)} elements ({entry.length / (1024**2):.1f} MB)")
