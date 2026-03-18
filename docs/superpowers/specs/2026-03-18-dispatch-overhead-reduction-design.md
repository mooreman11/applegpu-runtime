# P1: Dispatch Overhead Reduction

**Date**: 2026-03-18
**Status**: Approved
**Success metric**: GPU faster than CPU for LSTM hidden_size=128 training

## Problem

Benchmarks show GPU is slower than CPU for all model sizes. Per-op overhead in eager mode (~0.3ms) dwarfs actual compute time. A training step has hundreds of ops, so overhead >> compute.

Root cause: in eager mode, every op triggers a full cycle of:
1. Metal encode + commit + wait (~10-60us encoder overhead + commit latency)
2. GPU->CPU DMA read via `to_torch()` (~50-200us) -- defensive CPU backing sync
3. CPU `backing.copy_()` (~5-20us) -- unnecessary during training

## Approach: Two-Phase Fix (A + B)

### Phase A: Deferred CPU Sync (Python-only change)

**What**: Stop calling `to_torch()` + `backing.copy_()` after every op in eager mode. Only sync at explicit readback points.

**Why safe**: Subagent analysis confirmed that during normal training:
- All dispatch handlers use `_unwrap()` which returns the GPU tensor directly
- CPU fallback reads directly from GPU tensor via `_gpu_tensor_to_torch_cpu()`
- Autograd, optimizers, and `clip_grad_norm_` all go through `__torch_dispatch__`
- CPU backing is only read in `_extract_cpu_backing()` when a GPU tensor has been freed -- which doesn't happen in eager mode because: (a) `eval()` materializes tensors immediately, (b) the Python-side `_gpu_tensor_registry` holds a reference preventing Rust GC, and (c) the scheduler's container quota (100K tensors, 8GB) is never hit during normal training steps

**Implicit sync points** (safe without changes -- these already do their own GPU readback):
- CPU fallback ops (e.g., `fill`, `zeros`, `bernoulli_`) call `_gpu_tensor_to_torch_cpu()` internally
- `__repr__` only reads metadata (shape), does not access buffer contents

**Changes in `python/applegpu_runtime/torch_backend.py`**:

1. `_wrap()` (line ~178): Remove the `if _eager_mode:` block that calls `gpu_t.to_torch()` + `backing.copy_()` (lines 183-197). Keep `gpu.eval(gpu_t)`.

2. `_update_inplace()` (line ~201): Remove the `if _eager_mode:` block that calls `_sync_cpu_backing_from_gpu()` (lines 213-222). Keep `gpu.eval(result_gpu)`.

**Sync points** (already handle their own readback):
- `to_torch_cpu()` -- calls `_gpu_tensor_to_torch_cpu()` which does `gpu.eval()` + `to_torch()`
- `__reduce_ex__` -- calls `to_torch_cpu()`
- `to_numpy()` -- calls `BACKEND.read_bytes()` which requires materialization
- `.item()` -- goes through dispatch to scalar extraction

**Risk**: If a GPU tensor is unexpectedly freed, `_unwrap()` reconstructs from stale CPU backing. Mitigation: this doesn't happen in eager mode (see safety justification above). Add `warnings.warn()` if reconstruction is triggered in eager mode -- this indicates a bug, not expected behavior.

**Expected speedup**: ~2-3x (eliminates per-op GPU->CPU DMA)

### Phase B: Streaming Command Buffer (Rust + Python change)

**What**: Keep a persistent MTLCommandBuffer open across multiple `eval()` calls. Ops encode into it without committing. Flush only at sync points.

**Why safe**: Verified by Apple Metal documentation:
- Within a single CB, encoders execute sequentially; encoder A's writes are visible to encoder B
- All buffers use `storageModeShared` (Apple Silicon unified memory) -- CPU reads see GPU writes after CB completion
- Buffer pool uses Rust move semantics; a buffer can't be in both the pool and an active tensor simultaneously

**Rust changes in `crates/core/src/compute.rs`**:

1. New functions (wrapping existing Swift batch API):
   - `begin_streaming_batch(queue)` -- calls existing `begin_batch()`, sets a `streaming_active` flag
   - `flush_streaming_batch()` -- calls `end_batch()` + `wait_command_buffer()`, then immediately `begin_batch()` again to reopen
   - `end_streaming_batch()` -- calls `end_batch()` + `wait_command_buffer()`, clears flag

2. State: `static STREAMING_ACTIVE: AtomicBool` + `static STREAMING_OPS_COUNT: AtomicU32` for rolling flush. Using module-level atomics (not LazyRuntime fields) because `read_bytes`/`read_f32`/`read_f16` take `&self` and cannot call `&mut self` methods. The flush function acquires the Swift-side batch lock internally.

**Rust changes in `crates/core/src/lazy.rs`**:

1. `eval_single_cb()`: When `streaming_is_active()`, the streaming CB is already open -- do NOT call `begin_batch()` (Swift guards against double-open and would return null). Encode ops directly into the existing streaming CB. Skip `end_batch()` + `wait_command_buffer()` at the end. Increment `STREAMING_OPS_COUNT`.

2. `eval()` (parallel path): When streaming is active, flush the streaming batch first (commit+wait), then proceed with per-level batch contexts as normal. Streaming and parallel evaluation are mutually exclusive within a single eval call.

3. `read_bytes()`, `read_f16()`, `read_f32()`, `as_bytes()`: Add implicit flush INSIDE the read method, AFTER `get_tensor()` succeeds but BEFORE accessing `buffer.contents()`:
   ```rust
   pub fn read_bytes(&self, id: u64) -> Result<Vec<u8>> {
       let t = self.get_tensor(id)?;
       // Flush streaming CB so buffer contents are valid
       if streaming_is_active() {
           flush_streaming_batch();
       }
       Ok(t.as_bytes()?.to_vec())
   }
   ```

4. Rolling flush: After encoding N ops (configurable via `APPLEGPU_STREAMING_FLUSH_INTERVAL` env var, default 512), auto-flush to prevent unbounded CB growth. The flush commits+waits, then immediately reopens a new batch CB.

**Swift changes**: None. The existing `activeBatchCommandBuffer` / `begin_batch()` / `end_batch()` system already supports this usage pattern. No new FFI declarations needed in `ffi.rs` since streaming functions wrap existing FFI calls.

**Python changes in `python/applegpu_runtime/torch_backend.py`**:

1. `set_eager_mode(True)`: Call `gpu.begin_streaming_batch()` after enabling eager mode
2. `set_eager_mode(False)`: Call `gpu.end_streaming_batch()` before disabling
3. Guard against double-open: if `set_eager_mode(True)` is called when streaming is already active, no-op
4. `_gpu_tensor_to_torch_cpu()`: Call `gpu.flush_streaming_batch()` BEFORE `gpu_t.to_torch()`. This is critical -- `to_torch()` reads buffer contents via `read_bytes`, and the streaming CB must be flushed first for the data to be valid. This makes `to_torch_cpu()`, CPU fallback ops, and `.item()` all safe as sync points.

**PyO3 changes in `crates/python/src/lib.rs`**:

1. Expose `begin_streaming_batch()`, `flush_streaming_batch()`, `end_streaming_batch()` as Python functions

**Error handling**:
- If any op fails during encoding, abort the streaming batch (discard uncommitted work since the last flush)
- Tensors from previously flushed batches remain valid (their GPU buffers were committed)
- Tensors encoded since the last flush are lost -- their entries in `self.tensors` should be removed on abort
- After abort, automatically reopen a new streaming batch so subsequent ops continue working
- The rolling flush at N=512 bounds the maximum work lost on error to 512 ops

**Thread safety**: The Swift batch state uses thread-local storage (`Thread.current.threadDictionary`). Streaming is per-thread. Python's GIL serializes access from the main thread. DataLoader workers would each have independent streaming state (acceptable -- workers don't share GPU tensors).

**Expected speedup**: ~3-5x total (with Phase A), by amortizing commit+wait across hundreds of ops

## Implementation Order

1. Phase A first (Python-only, minimal risk, immediate ~2-3x win)
2. Benchmark to confirm Phase A improvement
3. Phase B (Rust + Python, more complex, additional ~2x on top of A)
4. Benchmark to confirm Phase B improvement against success metric

## Key Files

| File | Changes |
|------|---------|
| `python/applegpu_runtime/torch_backend.py` | A: Remove per-op CPU sync. B: Streaming batch init/teardown |
| `crates/core/src/compute.rs` | B: Streaming batch functions |
| `crates/core/src/lazy.rs` | B: Skip wait in streaming mode, flush before reads |
| `crates/python/src/lib.rs` | B: Expose streaming batch to Python |

## Test Plan

- All existing tests must pass (760 tests across Rust/Swift/Python)
- New test: training loop benchmark (LSTM h=128) comparing CPU vs GPU
- New test: verify `to_torch_cpu()` returns correct data after deferred ops
- New test: verify `loss.item()` returns correct scalar after deferred ops
- New test: CPU fallback op returns correct data during streaming (triggers implicit flush)
- New test: streaming batch error recovery (op fails mid-batch, subsequent ops work)
- New test: rolling flush triggers correctly at N=512
- New test: `set_eager_mode` toggle multiple times without crash (idempotent)
- New test: chain of 50+ ops followed by readback (validates Metal sequential guarantee)
- Benchmark: `benchmarks/bench_training.py` shows GPU faster than CPU for LSTM h=128
