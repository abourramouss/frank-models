#!/usr/bin/env python3
"""OLMo-1B single ctx.invoke inference."""
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iree.runtime import *
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("/home/bourram/models/olmo-1b-hf/models--allenai--OLMo-1B-hf/snapshots/aee7752d9c08ee4775e9b0091426d8410e8f6a89/tokenizer.json")
inst = VmInstance()
device = get_device("cuda")
hal = create_hal_module(inst, device)
pi = ParameterIndex()
pi.load("/home/bourram/models/olmo-1b-f16-stacked.irpa")
params = create_io_parameters_module(inst, pi.create_provider(scope="model"))
with open("/tmp/olmo_1b_single.vmfb", "rb") as f:
    mod = VmModule.copy_buffer(inst, f.read())
ctx = VmContext(inst, modules=[params, hal, mod])
run = mod.lookup_function("run")

def to_bv(arr):
    return device.allocator.allocate_buffer_copy(
        memory_type=MemoryType.DEVICE_LOCAL,
        allowed_usage=BufferUsage.DEFAULT | BufferUsage.MAPPING,
        device=device, buffer=np.ascontiguousarray(arr),
        element_type=HalElementType.SINT_64)

while True:
    try:
        text = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        break
    if not text.strip():
        break
    tokens = tokenizer.encode(text).ids
    a = VmVariantList(4)
    a.push_ref(to_bv(np.array(tokens, dtype=np.int64)))
    a.push_int(len(tokens))
    a.push_int(128)
    a.push_int(50279)
    r = VmVariantList(2)
    t0 = time.time()
    ctx.invoke(run, a, r)
    dt = time.time() - t0
    out = DeviceArray(device, r.get_as_object(0, HalBufferView), implicit_host_transfer=True).to_host()
    n = int(r.get_variant(1))
    print(tokenizer.decode(out[:n].tolist()))
    print(f"[{n} tok, {dt:.1f}s, {n/dt:.1f} tok/s]")
