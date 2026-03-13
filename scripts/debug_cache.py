#!/usr/bin/env python3
"""Debug KV cache: verify scatter_prefill writes and gather reads correctly."""

import sys
import numpy as np
from pathlib import Path

from iree.runtime import (
    BufferUsage,
    DeviceArray,
    HalBufferView,
    HalElementType,
    MemoryType,
    ParameterIndex,
    VmContext,
    VmInstance,
    VmModule,
    VmVariantList,
    create_hal_module,
    create_io_parameters_module,
    get_device,
)

DTYPE_TO_ELEMENT_TYPE = {
    np.float32: HalElementType.FLOAT_32,
    np.int32: HalElementType.SINT_32,
    np.int64: HalElementType.SINT_64,
}


def main():
    vmfb_path = Path("/home/bourram/models/olmoe.vmfb")
    params_path = Path("/home/bourram/models/olmoe-1b-7b.irpa")

    instance = VmInstance()
    device = get_device("local-task")
    hal_module = create_hal_module(instance, device)

    param_index = ParameterIndex()
    param_index.load(str(params_path))
    provider = param_index.create_provider(scope="model")
    params_module = create_io_parameters_module(instance, provider)

    with open(vmfb_path, "rb") as f:
        main_module = VmModule.copy_buffer(instance, f.read())

    context = VmContext(instance, modules=[params_module, hal_module, main_module])

    def to_bv(arr):
        arr = np.ascontiguousarray(arr)
        etype = DTYPE_TO_ELEMENT_TYPE[arr.dtype.type]
        return device.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
            device=device,
            buffer=arr,
            element_type=etype,
        )

    def from_bv(bv):
        return DeviceArray(device, bv, implicit_host_transfer=True).to_host()

    # Config
    n_layers = 16
    batch = 1
    block_size = 16
    max_blocks = 2
    n_blocks = n_layers * batch * max_blocks  # Per-layer blocks

    # Allocate cache
    func = main_module.lookup_function("allocate_kv_cache")
    args = VmVariantList(2)
    args.push_int(n_blocks)
    args.push_int(block_size)
    results = VmVariantList(1)
    context.invoke(func, args, results)
    cache = results.get_as_list(0)

    # Block tables: [n_layers, batch, max_blocks] - each layer gets its own physical blocks
    block_tables = np.zeros((n_layers, batch, max_blocks), dtype=np.int32)
    for layer in range(n_layers):
        for b in range(batch):
            for blk in range(max_blocks):
                block_tables[layer, b, blk] = layer * batch * max_blocks + b * max_blocks + blk

    print(f"n_blocks={n_blocks}, block_tables shape: {block_tables.shape}")
    print(f"Layer 0 phys blocks: {block_tables[0, 0, :]}, Layer 1: {block_tables[1, 0, :]}")

    # Prefill single token "Hello" (id 12092) at position 0
    token_id = 12092
    tokens = np.array([[token_id]], dtype=np.int64)
    positions = np.array([[0]], dtype=np.int64)
    start_positions = np.zeros(batch, dtype=np.int32)

    func = main_module.lookup_function("prefill")
    args = VmVariantList(6)
    args.push_ref(to_bv(tokens))
    args.push_ref(to_bv(positions))
    args.push_list(cache)
    args.push_ref(to_bv(block_tables))
    args.push_ref(to_bv(start_positions))
    args.push_int(block_size)
    results = VmVariantList(2)
    context.invoke(func, args, results)
    logits = from_bv(results.get_as_object(0, HalBufferView))
    cache = results.get_as_list(1)

    prefill_top5 = np.argsort(logits[0, 0, :])[::-1][:5]
    print(f"\nPrefill logits shape: {logits.shape}")
    print(f"Prefill top-5 token IDs: {prefill_top5}")

    # Inspect cache after prefill - check each layer's block
    k_bv_after = cache.get_as_object(0, HalBufferView)
    k_after = from_bv(k_bv_after)
    print(f"\nK cache after prefill shape: {k_after.shape}")
    for layer in range(min(3, n_layers)):
        phys_block = block_tables[layer, 0, 0]
        kdata = k_after[phys_block, 0]
        print(f"  Layer {layer} (phys_block={phys_block}), pos 0: mean={kdata.mean():.4f}, std={kdata.std():.4f}, max_abs={np.abs(kdata).max():.4f}")

    # Verify all layers have data
    zeros_count = sum(
        1 for l in range(n_layers)
        if np.abs(k_after[block_tables[l, 0, 0], 0]).max() < 1e-10
    )
    print(f"\nLayers with ZERO K at pos 0: {zeros_count}/{n_layers}")

    # Now decode next token at position 1
    next_token_id = int(prefill_top5[0])
    print(f"\nDecoding next token: {next_token_id} at position 1")

    decode_token = np.array([next_token_id], dtype=np.int64)
    decode_pos = np.array([1], dtype=np.int64)
    context_lens = np.full((n_layers, batch), 1, dtype=np.int32)
    max_ctx = 1  # Equals context_lens; decode concat adds +1

    func = main_module.lookup_function("decode")
    args = VmVariantList(6)
    args.push_ref(to_bv(decode_token))
    args.push_ref(to_bv(decode_pos))
    args.push_list(cache)
    args.push_ref(to_bv(block_tables))
    args.push_ref(to_bv(context_lens))
    args.push_int(max_ctx)
    results = VmVariantList(2)
    context.invoke(func, args, results)
    decode_logits = from_bv(results.get_as_object(0, HalBufferView))
    cache = results.get_as_list(1)

    decode_top5 = np.argsort(decode_logits[0, :])[::-1][:5]
    print(f"Decode logits shape: {decode_logits.shape}")
    print(f"Decode top-5: {decode_top5}")
    print(f"Decode logits stats: mean={decode_logits[0].mean():.4f}, std={decode_logits[0].std():.4f}")

    # Check cache after decode - layer 0 should have data at pos 1
    k_bv_after2 = cache.get_as_object(0, HalBufferView)
    k_after2 = from_bv(k_bv_after2)
    print(f"\nK cache after decode:")
    for layer in range(min(3, n_layers)):
        phys_block = block_tables[layer, 0, 0]
        k1 = k_after2[phys_block, 1]
        print(f"  Layer {layer} (phys_block={phys_block}), pos 1: mean={k1.mean():.4f}, std={k1.std():.4f}, max_abs={np.abs(k1).max():.4f}")

    zeros_after = sum(
        1 for l in range(n_layers)
        if np.abs(k_after2[block_tables[l, 0, 0], 1]).max() < 1e-10
    )
    print(f"Layers with ZERO K at pos 1 after decode: {zeros_after}/{n_layers}")


if __name__ == "__main__":
    main()
