# RESOLVED: Not an IREE bug — incorrect KV cache block table allocation in test code

## Summary

IREE's generated code produces a heap corruption (malloc(): corrupted top size) when the attention block has `n_head * head_dim ≠ n_embd`. This affects Qwen3-0.6B where `n_head=16, head_dim=128, n_embd=1024` (so `16*128=2048 ≠ 1024`).

The crash occurs on the very first decode call — no loops involved. It happens on both CPU (`local-sync`, `local-task`) and GPU (`cuda` gives ABORTED in queue.dealloca).

## Architecture

Standard (OLMo, Llama): `n_head * head_dim == n_embd` → works
Qwen3: `n_head * head_dim > n_embd` → crashes

The QKV projection output is `[batch, n_embd_q + 2*n_embd_kv]` where `n_embd_q = n_head * head_dim = 2048`. The reshape `[batch, 2048] → [batch, 16, 128]` and subsequent operations likely produce buffer size calculations that assume `n_embd_q == n_embd`.

## Reproduction

Files: `components/attention/attention_block_decode_qwen.mlir`, `models/llm/qwen3/`

```bash
# Link + compile
iree-link architectures/llm/llm_inference_qwen.mlir --link-module models/llm/qwen3/... -o qwen3.mlir
iree-compile qwen3.mlir --iree-hal-target-backends=llvm-cpu -o qwen3.vmfb

# Run single decode → crashes
python3 -c "... ctx.invoke(decode, ...) ..."
# malloc(): corrupted top size
```

## Root cause (hypothesis)

The buffer packing pass (`stream.resource.pack`) or an intermediate allocation computes buffer sizes based on `n_embd` (1024) when it should use `n_embd_q` (2048) for Q-related intermediates. The QKV matmul writes 4096 elements (8192 bytes) but the packed slab may only allocate space based on the smaller `n_embd`-derived sizes.
