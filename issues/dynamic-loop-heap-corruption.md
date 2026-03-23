# Heap corruption in scf.while with dynamically growing tensors

## Status: FIXED (proper compiler fix)

## Summary

IREE's `OptimizeIntArithmeticPass` constant-folds values derived from loop-carried variables to their initial-iteration values. This causes transient buffer allocations sized for iteration 1 to be used for all iterations, leading to heap corruption when the buffer grows past the allocated size.

## Root cause

`IntegerRangeAnalysis` (upstream MLIR) incorrectly narrows the range of `scf.while` block arguments to `[initial_value, initial_value]` instead of widening across back-edges. The `MaterializeKnownConstantValues` pattern then replaces these "constant" values in the IR. Downstream ops like `util.align(pos*4, 64)` get folded to `64` (the value at pos=1), and this constant is used as the transient buffer allocation size.

The pass runs twice: once on structured IR (`scf.while`) and once after SCF-to-CF lowering (`cf.br` back-edges). Both invocations had the same bug.

## Fix (commit 35974a9d62)

`SafeMaterializeKnownConstantValues` in `OptimizeIntArithmetic.cpp` — a guarded replacement for upstream `MaterializeKnownConstantValues` that skips constant folding for values inside loops. Detects both:
- **Structured loops**: `scf.while`, `scf.for` via `RegionBranchOpInterface` back-edge detection
- **CFG loops**: `cf.br`/`cf.cond_br` back-edges via block predecessor ordering

This is a proper fix, not a workaround. It preserves all int-range optimizations outside of loops (unsigned conversion, narrowing, divisibility, trivial remainder elimination) while preventing the specific bug of folding loop-variant values to constants. The guard is conservative — it may skip some valid optimizations inside loops, but never produces incorrect code.

A more precise upstream fix would be to correct `IntegerRangeAnalysis` to properly widen ranges across `scf.while` back-edges (in `llvm-project/mlir/lib/Analysis/DataFlow/IntegerRangeAnalysis.cpp`). That would allow safe constant folding of truly-constant values inside loops while still widening loop-variant ones.

## Reproduction

```bash
iree-compile issues/dynamic-loop-heap-corruption-repro.mlir \
  --iree-hal-target-backends=llvm-cpu -o /tmp/repro.vmfb
python3 issues/dynamic-loop-heap-corruption-test.py 24  # was: crash, now: ok
python3 issues/dynamic-loop-heap-corruption-test.py 500 # was: crash, now: ok
```

## Files

- `dynamic-loop-heap-corruption-repro.mlir` — 100-line minimal MLIR repro
- `dynamic-loop-heap-corruption-test.py` — Python test driver
- Fix: `iree/compiler/src/iree/compiler/Dialect/Util/Transforms/OptimizeIntArithmetic.cpp`
