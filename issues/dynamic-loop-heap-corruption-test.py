#!/usr/bin/env python3
"""Test the dynamic loop repro: scf.while with growing tensor."""
import sys
sys.path.insert(0, '/home/bourram/frank-models-hetero')
from iree.runtime import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

inst = VmInstance()
device = get_device('local-sync')
hal = create_hal_module(inst, device)

with open('/tmp/repro.vmfb', 'rb') as f:
    mod = VmModule.copy_buffer(inst, f.read())
ctx = VmContext(inst, modules=[hal, mod])

args = VmVariantList(1)
args.push_int(N)
results = VmVariantList(1)
ctx.invoke(mod.lookup_function('run_loop'), args, results)
print(f"n={N}: OK")
