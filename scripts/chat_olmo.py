#!/usr/bin/env python3
"""Interactive chat with OLMo-1B via IREE.

Modes:
  --stream:  tokens appear one by one (prefill_all + decode loop)
  default:   single ctx.invoke(@run), all tokens returned at once
"""
import argparse, sys, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iree.runtime import (
    VmInstance, VmModule, VmContext, VmVariantList,
    HalElementType, MemoryType, BufferUsage, DeviceArray, HalBufferView,
    ParameterIndex, create_hal_module, create_io_parameters_module, get_device,
)

PARAMS = "/home/bourram/models/olmo-1b-f16-stacked.irpa"
TOKENIZER = "/home/bourram/models/olmo-1b-hf/models--allenai--OLMo-1B-hf/snapshots/aee7752d9c08ee4775e9b0091426d8410e8f6a89/tokenizer.json"
N_LAYERS = 16
BLOCK_SIZE = 16
EOS_TOKEN = 50279

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="local-task", choices=["local-task", "local-sync", "cuda"])
parser.add_argument("--vmfb", default="/home/bourram/models/olmo_1b_dense_cpu.vmfb")
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--stream", action="store_true", help="Stream tokens (uses prefill + decode loop)")
args = parser.parse_args()

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER)

inst = VmInstance()
device = get_device(args.backend)
hal = create_hal_module(inst, device)
param_index = ParameterIndex()
param_index.load(PARAMS)
params_module = create_io_parameters_module(inst, param_index.create_provider(scope="model"))
with open(args.vmfb, "rb") as f:
    mod = VmModule.copy_buffer(inst, f.read())
ctx = VmContext(inst, modules=[params_module, hal, mod])
print(f"[init] OLMo-1B loaded ({args.backend})")

def to_bv(arr):
    arr = np.ascontiguousarray(arr)
    etype = {np.float16: HalElementType.FLOAT_16,
             np.int32: HalElementType.SINT_32,
             np.int64: HalElementType.SINT_64}[arr.dtype.type]
    return device.allocator.allocate_buffer_copy(
        memory_type=MemoryType.DEVICE_LOCAL,
        allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
        device=device, buffer=arr, element_type=etype)

def from_bv(bv):
    return DeviceArray(device, bv, implicit_host_transfer=True).to_host()

def generate_batch(prompt_text, max_new_tokens):
    """Single invoke — no streaming."""
    tokens = tokenizer.encode(prompt_text).ids
    if not tokens:
        return [], 0.0
    a = VmVariantList(4)
    a.push_ref(to_bv(np.array(tokens, dtype=np.int64)))
    a.push_int(len(tokens)); a.push_int(max_new_tokens); a.push_int(EOS_TOKEN)
    r = VmVariantList(2)
    t0 = time.time()
    ctx.invoke(mod.lookup_function('run'), a, r)
    dt = time.time() - t0
    out = from_bv(r.get_as_object(0, HalBufferView))
    generated = out[:max_new_tokens].tolist()
    for i, t in enumerate(generated):
        if t == EOS_TOKEN:
            generated = generated[:i+1]; break
    return generated, dt

def generate_stream(prompt_text, max_new_tokens):
    """Streaming — prefill_all + decode per token."""
    tokens = tokenizer.encode(prompt_text).ids
    if not tokens:
        return [], 0.0
    total_len = len(tokens) + max_new_tokens
    max_blocks = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_blocks = N_LAYERS * max_blocks

    # Allocate cache
    a = VmVariantList(2); a.push_int(n_blocks); a.push_int(BLOCK_SIZE)
    r = VmVariantList(1)
    ctx.invoke(mod.lookup_function('allocate_kv_cache'), a, r)
    cache = r.get_as_list(0)

    block_tables = np.zeros((N_LAYERS, 1, max_blocks), dtype=np.int32)
    for l in range(N_LAYERS):
        for b in range(max_blocks):
            block_tables[l, 0, b] = l * max_blocks + b
    bt_bv = to_bv(block_tables)

    # Prefill all prompt tokens in one invoke
    a = VmVariantList(4)
    a.push_ref(to_bv(np.array(tokens, dtype=np.int64)))
    a.push_int(len(tokens)); a.push_list(cache); a.push_ref(bt_bv)
    r = VmVariantList(2)
    t0 = time.time()
    ctx.invoke(mod.lookup_function('prefill_all'), a, r)
    logits = from_bv(r.get_as_object(0, HalBufferView))
    cache = r.get_as_list(1)

    next_token = int(np.argmax(logits[0]))
    generated = []
    func_decode = mod.lookup_function('decode')

    for step in range(max_new_tokens):
        generated.append(next_token)
        text = tokenizer.decode([next_token])
        sys.stdout.write(text); sys.stdout.flush()
        if next_token == EOS_TOKEN:
            break

        pos = len(tokens) + step
        a = VmVariantList(6)
        a.push_ref(to_bv(np.array([next_token], dtype=np.int64)))
        a.push_ref(to_bv(np.array([pos], dtype=np.int64)))
        a.push_list(cache); a.push_ref(bt_bv)
        a.push_ref(to_bv(np.full((N_LAYERS, 1), pos, dtype=np.int32)))
        a.push_int(pos)
        r = VmVariantList(2)
        ctx.invoke(func_decode, a, r)
        logits = from_bv(r.get_as_object(0, HalBufferView))
        cache = r.get_as_list(1)
        next_token = int(np.argmax(logits[0]))

    sys.stdout.write("\n")
    return generated, time.time() - t0

mode = "streaming" if args.stream else "single invoke"
print(f"\nOLMo-1B Chat — {mode}, {args.backend}")
print("=" * 50)

while True:
    try:
        user_input = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break
    if user_input.strip().lower() in ('quit', 'exit', ''):
        break

    if args.stream:
        gen, dt = generate_stream(user_input, args.max_tokens)
    else:
        gen, dt = generate_batch(user_input, args.max_tokens)
        print(tokenizer.decode(gen))

    n = len(gen)
    print(f"[{n} tokens, {dt:.1f}s, {n/dt:.1f} tok/s]")
