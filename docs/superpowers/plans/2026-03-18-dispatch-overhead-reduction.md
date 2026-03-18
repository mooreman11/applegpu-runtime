# P1: Dispatch Overhead Reduction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GPU faster than CPU for LSTM hidden_size=128 training by eliminating per-op dispatch overhead.

**Architecture:** Two-phase approach. Phase A removes unnecessary GPU→CPU DMA after in-place ops (Python-only). Phase B keeps a persistent Metal command buffer open across `eval()` calls, flushing only at readback points (Rust + Python). Both phases are additive — Phase B builds on A.

**Tech Stack:** Rust (core runtime), Swift (Metal bridge), Python (torch backend), PyO3 (Rust↔Python FFI)

**Spec:** `docs/superpowers/specs/2026-03-18-dispatch-overhead-reduction-design.md`

---

## File Structure

**Phase A (Python-only):**
- Modify: `python/applegpu_runtime/torch_backend.py` — remove CPU sync from `_update_inplace()`, add warning in `_unwrap()` reconstruction path
- Create: `python/tests/test_deferred_sync.py` — Phase A correctness tests

**Phase B (Rust + Python):**
- Modify: `crates/core/src/compute.rs` — add `begin_streaming_batch()`, `flush_streaming_batch()`, `end_streaming_batch()`, `streaming_is_active()`
- Modify: `crates/core/src/lazy.rs` — modify `eval()`/`eval_single_cb()` for streaming mode, add flush guards to read methods
- Modify: `crates/python/src/metal_backend.rs` — add `begin_streaming_batch()`/`flush_streaming_batch()`/`end_streaming_batch()` to `MetalBackend`
- Modify: `crates/python/src/lib.rs` — expose streaming batch functions as Python-callable
- Modify: `python/applegpu_runtime/torch_backend.py` — call streaming batch from `set_eager_mode()` and `_gpu_tensor_to_torch_cpu()`
- Create: `crates/core/tests/streaming_batch_integration.rs` — Rust-level streaming batch tests
- Create: `python/tests/test_streaming_batch.py` — Python-level streaming batch tests

---

## Phase A: Deferred CPU Sync

### Task 1: Remove CPU sync from `_update_inplace()` and add reconstruction warning

**Files:**
- Modify: `python/applegpu_runtime/torch_backend.py:122-161` (`_unwrap`) and `209-223` (`_update_inplace`)
- Create: `python/tests/test_deferred_sync.py`

- [ ] **Step 1: Write failing test — in-place op without CPU sync still produces correct readback**

Create `python/tests/test_deferred_sync.py`:

```python
"""Tests for deferred CPU backing sync in eager mode."""
import pytest
import torch


def _setup():
    import applegpu_runtime as gpu
    gpu.init_backend()
    gpu.enable_torch_backend()
    from applegpu_runtime.torch_backend import set_eager_mode, ApplegpuTensor
    set_eager_mode(True)
    return gpu, ApplegpuTensor


def test_inplace_add_readback():
    """In-place add_ followed by to_torch_cpu returns correct result."""
    gpu, ApplegpuTensor = _setup()
    x = torch.tensor([1.0, 2.0, 3.0])
    gx = gpu.to_applegpu(x)
    gx.add_(torch.tensor([10.0, 20.0, 30.0]))
    result = gx.to_torch_cpu()
    assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0]))


def test_inplace_mul_readback():
    """In-place mul_ followed by to_torch_cpu returns correct result."""
    gpu, ApplegpuTensor = _setup()
    x = torch.tensor([2.0, 3.0, 4.0])
    gx = gpu.to_applegpu(x)
    gx.mul_(torch.tensor([5.0, 6.0, 7.0]))
    result = gx.to_torch_cpu()
    assert torch.allclose(result, torch.tensor([10.0, 18.0, 28.0]))


def test_optimizer_step_correctness():
    """Adam optimizer step produces finite, changed parameters."""
    gpu, ApplegpuTensor = _setup()
    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = gpu.to_applegpu(torch.randn(3, 4))
    y = gpu.to_applegpu(torch.randn(3, 2))

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()

    # Parameters should be updated (in-place ops) and readable
    for p in model.parameters():
        cpu_p = p.to_torch_cpu()
        assert torch.isfinite(cpu_p).all(), f"Non-finite param after step: {cpu_p}"


def test_loss_item_after_inplace():
    """loss.item() returns correct scalar after in-place optimizer updates."""
    gpu, ApplegpuTensor = _setup()
    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = gpu.to_applegpu(torch.randn(3, 4))
    y = gpu.to_applegpu(torch.randn(3, 2))

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    out2 = model(x)
    loss2 = torch.nn.functional.mse_loss(out2, y)
    val = loss2.to_torch_cpu().item()
    assert isinstance(val, float) and val >= 0


def test_clip_grad_norm_after_inplace():
    """clip_grad_norm_ works correctly after in-place ops."""
    gpu, ApplegpuTensor = _setup()
    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)

    x = gpu.to_applegpu(torch.randn(3, 4))
    y = gpu.to_applegpu(torch.randn(3, 2))
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if hasattr(norm, 'to_torch_cpu'):
        norm = norm.to_torch_cpu()
    assert float(norm) >= 0


def test_torch_save_after_inplace():
    """torch.save works after in-place ops (uses __reduce_ex__)."""
    import tempfile, os
    gpu, ApplegpuTensor = _setup()
    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = gpu.to_applegpu(torch.randn(3, 4))
    y = gpu.to_applegpu(torch.randn(3, 2))
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        loaded = torch.load(path, weights_only=True)
        assert len(loaded) > 0
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they pass with current code (baseline)**

Run: `uv run pytest python/tests/test_deferred_sync.py -v`
Expected: All PASS (these test correctness, which should already work)

- [ ] **Step 3: Remove CPU sync from `_update_inplace()`**

In `python/applegpu_runtime/torch_backend.py`, remove lines 221-222:

```python
# REMOVE these two lines from _update_inplace():
        if _eager_mode:
            _sync_cpu_backing_from_gpu(a, result_gpu)
```

The function should become:

```python
def _update_inplace(a, result_gpu):
    """Update an ApplegpuTensor in-place with a new GpuTensor result."""
    if _eager_mode:
        gpu.eval(result_gpu)
    if isinstance(a, ApplegpuTensor):
        _gpu_tensor_registry[a.data_ptr()] = result_gpu
        a._gpu_tensor = result_gpu
    return a
```

- [ ] **Step 4: Add reconstruction warning in `_unwrap()`**

In `python/applegpu_runtime/torch_backend.py`, in `_unwrap()` at the reconstruction path (line 136-144), add a warning when in eager mode:

```python
            except (ValueError, RuntimeError):
                # Tensor was freed by lazy runtime -- recreate from CPU backing.
                if _eager_mode:
                    import warnings
                    warnings.warn(
                        f"GPU tensor {id(t)} was freed in eager mode -- "
                        "reconstructing from CPU backing (may be stale). "
                        "This indicates a bug.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                cpu_data = _extract_cpu_backing(t)
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest python/tests/test_deferred_sync.py -v && uv run pytest python/tests/ -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite to check for regressions**

Run: `make test`
Expected: All ~760 tests pass

- [ ] **Step 7: Commit**

```bash
git add python/applegpu_runtime/torch_backend.py python/tests/test_deferred_sync.py
git commit -m "perf: remove per-op CPU sync from _update_inplace (Phase A)

Remove _sync_cpu_backing_from_gpu() call after every in-place op in eager
mode. CPU backing is only needed for reconstruction when a GPU tensor is
freed, which doesn't happen in eager mode. Adds warnings.warn() if
reconstruction is triggered in eager mode (indicates a bug).

Part of P1 dispatch overhead reduction."
```

### Task 2: Benchmark Phase A

**Files:**
- None (run existing benchmarks)

- [ ] **Step 1: Run training benchmark**

Run: `uv run python benchmarks/bench_training.py --models lstm --epochs 3 --hidden-size 128`
Expected: Note the GPU vs CPU times. Phase A improvement is modest (only 9 in-place ops affected).

- [ ] **Step 2: Record baseline for Phase B comparison**

Save the output. We'll compare after Phase B.

---

## Phase B: Streaming Command Buffer

### Task 3: Add streaming batch functions to Rust compute module

**Files:**
- Modify: `crates/core/src/compute.rs:2510-2540`
- Create: `crates/core/tests/streaming_batch_integration.rs`

- [ ] **Step 1: Write failing Rust integration test**

Create `crates/core/tests/streaming_batch_integration.rs`:

```rust
use applegpu_core::compute;
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::tensor::Tensor;

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn streaming_batch_basic_lifecycle() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    let queue = compute::get_shared_queue(&device);
    assert!(!queue.is_null());

    // Should not be active initially
    assert!(!compute::streaming_is_active());

    // Begin streaming
    compute::begin_streaming_batch(queue);
    assert!(compute::streaming_is_active());

    // Flush (commit+wait, reopen)
    compute::flush_streaming_batch();
    assert!(compute::streaming_is_active()); // still active after flush

    // End streaming
    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_idempotent_begin() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };

    let queue = compute::get_shared_queue(&device);

    // Double begin should not crash
    compute::begin_streaming_batch(queue);
    compute::begin_streaming_batch(queue); // no-op
    assert!(compute::streaming_is_active());

    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_end_when_inactive() {
    // End when not active should be a no-op, not crash
    assert!(!compute::streaming_is_active());
    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_flush_when_inactive() {
    // Flush when not active should be a no-op
    assert!(!compute::streaming_is_active());
    compute::flush_streaming_batch();
    assert!(!compute::streaming_is_active());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core --test streaming_batch_integration`
Expected: FAIL — `streaming_is_active`, `begin_streaming_batch`, etc. don't exist yet

- [ ] **Step 3: Implement streaming batch functions in compute.rs**

Add to `crates/core/src/compute.rs` after the `abort_batch()` function (after line 2539):

First, add imports to the top of `crates/core/src/compute.rs` (after line 3):

```rust
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
```

Then add the following after the `abort_batch()` function (after line 2539):

```rust
// ── Streaming batch mode ──────────────────────────────────────────

/// Wrapper for raw pointer to satisfy Send/Sync requirements.
struct QueuePtr(*mut std::ffi::c_void);
unsafe impl Send for QueuePtr {}
unsafe impl Sync for QueuePtr {}

static STREAMING_ACTIVE: AtomicBool = AtomicBool::new(false);
static STREAMING_OPS_COUNT: AtomicU32 = AtomicU32::new(0);
static STREAMING_QUEUE: Mutex<QueuePtr> = Mutex::new(QueuePtr(std::ptr::null_mut()));

/// Default flush interval (configurable via APPLEGPU_STREAMING_FLUSH_INTERVAL env var).
fn streaming_flush_interval() -> u32 {
    static INTERVAL: once_cell::sync::Lazy<u32> = once_cell::sync::Lazy::new(|| {
        std::env::var("APPLEGPU_STREAMING_FLUSH_INTERVAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    });
    *INTERVAL
}

/// Check if streaming batch mode is active.
pub fn streaming_is_active() -> bool {
    STREAMING_ACTIVE.load(Ordering::Acquire)
}

/// Begin streaming batch mode. Opens a persistent command buffer that stays
/// open across multiple eval() calls. No-op if already active.
pub fn begin_streaming_batch(queue: *mut std::ffi::c_void) {
    if STREAMING_ACTIVE.load(Ordering::Acquire) {
        return; // already active, no-op
    }
    let cb = begin_batch(queue);
    if cb.is_null() {
        return; // failed to create CB
    }
    STREAMING_QUEUE.lock().unwrap().0 = queue;
    STREAMING_OPS_COUNT.store(0, Ordering::Release);
    STREAMING_ACTIVE.store(true, Ordering::Release);
}

/// Flush the streaming batch: commit+wait the current CB, then reopen a new one.
/// No-op if streaming is not active.
pub fn flush_streaming_batch() {
    if !STREAMING_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let cb = end_batch();
    if !cb.is_null() {
        wait_command_buffer(cb);
    }
    // Reopen
    let queue = STREAMING_QUEUE.lock().unwrap().0;
    if !queue.is_null() {
        let _new_cb = begin_batch(queue);
    }
    STREAMING_OPS_COUNT.store(0, Ordering::Release);
}

/// End streaming batch mode. Commits+waits the final CB, clears state.
/// No-op if streaming is not active.
pub fn end_streaming_batch() {
    if !STREAMING_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let cb = end_batch();
    if !cb.is_null() {
        wait_command_buffer(cb);
    }
    STREAMING_QUEUE.lock().unwrap().0 = std::ptr::null_mut();
    STREAMING_OPS_COUNT.store(0, Ordering::Release);
    STREAMING_ACTIVE.store(false, Ordering::Release);
}

/// Abort streaming batch: discard uncommitted work, then reopen a fresh CB.
/// Used for error recovery. No-op if streaming is not active.
pub fn abort_and_reopen_streaming_batch() {
    if !STREAMING_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    abort_batch();
    // Reopen
    let queue = STREAMING_QUEUE.lock().unwrap().0;
    if !queue.is_null() {
        let _new_cb = begin_batch(queue);
    }
    STREAMING_OPS_COUNT.store(0, Ordering::Release);
}

/// Increment streaming ops count and auto-flush if threshold reached.
/// Call this after encoding an op into the streaming CB.
pub fn streaming_tick() {
    if !STREAMING_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let count = STREAMING_OPS_COUNT.fetch_add(1, Ordering::AcqRel) + 1;
    if count >= streaming_flush_interval() {
        flush_streaming_batch();
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-core --test streaming_batch_integration`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/compute.rs crates/core/tests/streaming_batch_integration.rs
git commit -m "feat: add streaming batch API to compute module (Phase B)

New functions: begin_streaming_batch, flush_streaming_batch,
end_streaming_batch, streaming_is_active, streaming_tick.
Keeps a persistent MTLCommandBuffer open across eval() calls.
Auto-flushes at configurable interval (APPLEGPU_STREAMING_FLUSH_INTERVAL,
default 512)."
```

### Task 4: Modify `eval_single_cb()` and `eval()` to use streaming mode

**Files:**
- Modify: `crates/core/src/lazy.rs:134-362`
- Modify: `crates/core/tests/streaming_batch_integration.rs`

- [ ] **Step 1: Write failing test — eval in streaming mode skips wait**

Add to `crates/core/tests/streaming_batch_integration.rs`:

```rust
#[test]
fn streaming_eval_chain_then_read() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    // Create tensors
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    // Build a chain: (a + b) * a
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    let prod_id = ops::mul(&mut rt, sum_id, a_id).unwrap();

    // Enable streaming, eval, then read (should auto-flush)
    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);

    rt.eval(&device, prod_id).unwrap();
    // Read triggers implicit flush
    let result = rt.read_f32(prod_id).unwrap();
    assert_eq!(result, &[11.0, 44.0, 99.0, 176.0]);

    compute::end_streaming_batch();
}

#[test]
fn streaming_multiple_evals_then_read() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    // Multiple independent evals into same streaming CB
    let a = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![3], &[4.0, 5.0, 6.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    let prod_id = ops::mul(&mut rt, a_id, b_id).unwrap();

    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);

    // Two separate evals — both encode into the same streaming CB
    rt.eval(&device, sum_id).unwrap();
    rt.eval(&device, prod_id).unwrap();

    // Read both (flush happens on first read)
    let sum_result = rt.read_f32(sum_id).unwrap();
    let prod_result = rt.read_f32(prod_id).unwrap();

    assert_eq!(sum_result, &[5.0, 7.0, 9.0]);
    assert_eq!(prod_result, &[4.0, 10.0, 18.0]);

    compute::end_streaming_batch();
}

#[test]
fn streaming_50_ops_chain_correctness() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    // Chain 50 add ops: start with 1.0, add 1.0 fifty times -> 51.0
    let one = Tensor::from_f32(&device, vec![1], &[1.0]).unwrap();
    let one_id = one.meta.id;
    rt.insert_tensor(one).unwrap();

    let mut current_id = one_id;
    for _ in 0..50 {
        let inc = Tensor::from_f32(&device, vec![1], &[1.0]).unwrap();
        let inc_id = inc.meta.id;
        rt.insert_tensor(inc).unwrap();
        current_id = ops::add(&mut rt, current_id, inc_id).unwrap();
    }

    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);

    rt.eval(&device, current_id).unwrap();
    let result = rt.read_f32(current_id).unwrap();
    assert_eq!(result, &[51.0]);

    compute::end_streaming_batch();
}
```

Also add tests for error recovery, rolling flush, and parallel eval:

```rust
#[test]
fn streaming_error_recovery() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
    let a_id = a.meta.id;
    rt.insert_tensor(a).unwrap();

    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);

    // Eval a valid op
    let b = Tensor::from_f32(&device, vec![3], &[4.0, 5.0, 6.0]).unwrap();
    let b_id = b.meta.id;
    rt.insert_tensor(b).unwrap();
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, sum_id).unwrap();

    // Eval an invalid op (non-existent tensor) — should error but streaming continues
    let bad_result = rt.eval(&device, 999999);
    assert!(bad_result.is_err());

    // Streaming should still be active after error
    assert!(compute::streaming_is_active());

    // Subsequent valid ops should work
    let c = Tensor::from_f32(&device, vec![3], &[10.0, 20.0, 30.0]).unwrap();
    let c_id = c.meta.id;
    rt.insert_tensor(c).unwrap();
    let sum2_id = ops::add(&mut rt, a_id, c_id).unwrap();
    rt.eval(&device, sum2_id).unwrap();
    let result = rt.read_f32(sum2_id).unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0]);

    compute::end_streaming_batch();
}

#[test]
fn streaming_rolling_flush() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    // Set flush interval to 4 ops for testing
    std::env::set_var("APPLEGPU_STREAMING_FLUSH_INTERVAL", "4");

    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);

    // Encode 5 ops — should auto-flush after the 4th
    let mut current_id = {
        let t = Tensor::from_f32(&device, vec![1], &[1.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        id
    };
    for i in 0..5 {
        let inc = Tensor::from_f32(&device, vec![1], &[1.0]).unwrap();
        let inc_id = inc.meta.id;
        rt.insert_tensor(inc).unwrap();
        current_id = ops::add(&mut rt, current_id, inc_id).unwrap();
        rt.eval(&device, current_id).unwrap();
    }

    // Result should be correct (6.0 = 1.0 + 5 * 1.0)
    let result = rt.read_f32(current_id).unwrap();
    assert_eq!(result, &[6.0]);

    compute::end_streaming_batch();
    std::env::remove_var("APPLEGPU_STREAMING_FLUSH_INTERVAL");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core --test streaming_batch_integration`
Expected: New tests FAIL because eval still calls `begin_batch`/`end_batch` unconditionally

- [ ] **Step 3: Modify `eval_single_cb()` to support streaming mode**

In `crates/core/src/lazy.rs`, modify `eval_single_cb()` (line 288):

Change the batch setup at the top (lines 290-292):
```rust
fn eval_single_cb(&mut self, device: &Device, container_id: ContainerId, levels: Vec<Vec<u64>>) -> Result<()> {
    let order: Vec<u64> = levels.into_iter().flat_map(|l| l.into_iter()).collect();
    let queue = crate::compute::get_shared_queue(device);
    let streaming = crate::compute::streaming_is_active();

    // If streaming, the CB is already open — don't create a new one
    let use_batch = if streaming {
        true // ops encode into the existing streaming CB
    } else {
        let batch_cb = crate::compute::begin_batch(queue);
        !batch_cb.is_null()
    };
    let mut last_cb: Option<*mut std::ffi::c_void> = None;
```

Inside the loop body, after each op is encoded (after `self.tensors.insert(node_id, out);` at line 344), add per-op tick:
```rust
                self.tensors.insert(node_id, out);
                if streaming {
                    crate::compute::streaming_tick();
                }
```

Change the end section (lines 349-361):
```rust
    if streaming {
        if loop_result.is_err() {
            // Error during streaming: abort the CB and reopen for recovery
            crate::compute::abort_and_reopen_streaming_batch();
        }
        // Don't commit on success — CB stays open for next eval() call
    } else if use_batch {
        if loop_result.is_ok() {
            let cb = crate::compute::end_batch();
            if !cb.is_null() {
                crate::compute::wait_command_buffer(cb);
            }
        } else {
            crate::compute::abort_batch();
        }
    } else if let Some(cb) = last_cb {
        crate::compute::wait_command_buffer(cb);
    }
    loop_result
}
```

- [ ] **Step 4: Modify `eval()` parallel path to flush streaming first**

In `crates/core/src/lazy.rs`, modify `eval()` (around line 166-170):

```rust
    // Fast path: linear graph -> single-CB path
    if levels.iter().all(|l| l.len() == 1) {
        return self.eval_single_cb(device, container_id, levels);
    }

    // Parallel path: if streaming is active, flush first
    // (streaming and parallel contexts are mutually exclusive)
    if crate::compute::streaming_is_active() {
        crate::compute::flush_streaming_batch();
    }

    let num_queues = std::cmp::min(
```

- [ ] **Step 5: Add flush guards to read methods**

In `crates/core/src/lazy.rs`, modify `read_bytes()` (line 2442), `read_f16()`, and `read_f32()`:

```rust
pub fn read_bytes(&self, id: u64) -> Result<Vec<u8>> {
    if crate::compute::streaming_is_active() {
        crate::compute::flush_streaming_batch();
    }
    let t = self.get_tensor(id)?;
    Ok(t.as_bytes()?.to_vec())
}

pub fn read_f16(&self, id: u64) -> Result<Vec<u16>> {
    if crate::compute::streaming_is_active() {
        crate::compute::flush_streaming_batch();
    }
    let t = self.get_tensor(id)?;
    Ok(t.as_f16_slice()?.to_vec())
}

pub fn read_f32(&self, id: u64) -> Result<Vec<f32>> {
    if crate::compute::streaming_is_active() {
        crate::compute::flush_streaming_batch();
    }
    let t = self.get_tensor(id)?;
    Ok(t.as_f32_slice()?.to_vec())
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p applegpu-core --test streaming_batch_integration`
Expected: All PASS

- [ ] **Step 7: Run full Rust test suite**

Run: `cargo test -p applegpu-core`
Expected: All existing tests pass (streaming is off by default, so existing tests are unaffected)

- [ ] **Step 8: Commit**

```bash
git add crates/core/src/lazy.rs crates/core/tests/streaming_batch_integration.rs
git commit -m "feat: integrate streaming batch into eval path (Phase B)

eval_single_cb() skips begin_batch/end_batch when streaming is active.
eval() parallel path flushes streaming before using per-level contexts.
read_bytes/read_f16/read_f32 auto-flush before accessing buffer contents."
```

### Task 5: Expose streaming batch to Python via PyO3

**Files:**
- Modify: `crates/python/src/metal_backend.rs`
- Modify: `crates/python/src/lib.rs:1243-1330`

- [ ] **Step 1: Add streaming batch methods to MetalBackend**

In `crates/python/src/metal_backend.rs`, add methods to the `impl Backend for MetalBackend` block. Use the existing `get_device_runtime()` helper (line 43) which returns `&'static Runtime` with a `.device` field:

```rust
fn begin_streaming_batch(&self) -> BackendResult<()> {
    let runtime = get_device_runtime()?;
    let queue = applegpu_core::compute::get_shared_queue(&runtime.device);
    applegpu_core::compute::begin_streaming_batch(queue);
    Ok(())
}

fn flush_streaming_batch(&self) {
    applegpu_core::compute::flush_streaming_batch();
}

fn end_streaming_batch(&self) {
    applegpu_core::compute::end_streaming_batch();
}
```

Also add these method signatures to the `Backend` trait in `crates/python/src/backend.rs`:

```rust
fn begin_streaming_batch(&self) -> BackendResult<()>;
fn flush_streaming_batch(&self);
fn end_streaming_batch(&self);
```

And provide no-op implementations for `SocketBackend` in `crates/python/src/socket_backend.rs` (Linux backend):

```rust
fn begin_streaming_batch(&self) -> BackendResult<()> { Ok(()) }
fn flush_streaming_batch(&self) {}
fn end_streaming_batch(&self) {}
```

- [ ] **Step 2: Add PyO3 wrapper functions in lib.rs**

In `crates/python/src/lib.rs`, add pyfunction wrappers:

```rust
#[pyfunction]
fn begin_streaming_batch() -> PyResult<()> {
    BACKEND.begin_streaming_batch();
    Ok(())
}

#[pyfunction]
fn flush_streaming_batch() -> PyResult<()> {
    BACKEND.flush_streaming_batch();
    Ok(())
}

#[pyfunction]
fn end_streaming_batch() -> PyResult<()> {
    BACKEND.end_streaming_batch();
    Ok(())
}
```

Register them in the `applegpu_runtime` module function:

```rust
m.add_function(wrap_pyfunction!(begin_streaming_batch, m)?)?;
m.add_function(wrap_pyfunction!(flush_streaming_batch, m)?)?;
m.add_function(wrap_pyfunction!(end_streaming_batch, m)?)?;
```

- [ ] **Step 3: Build the Python extension**

Run: `uv run maturin develop`
Expected: Builds successfully

- [ ] **Step 4: Verify functions are callable from Python**

Run: `uv run python -c "import applegpu_runtime as gpu; gpu.init_backend(); gpu.begin_streaming_batch(); gpu.flush_streaming_batch(); gpu.end_streaming_batch(); print('OK')"`
Expected: Prints `OK`

- [ ] **Step 5: Commit**

```bash
git add crates/python/src/metal_backend.rs crates/python/src/lib.rs
git commit -m "feat: expose streaming batch API to Python via PyO3

begin_streaming_batch(), flush_streaming_batch(), end_streaming_batch()
now callable from Python as gpu.begin_streaming_batch() etc."
```

### Task 6: Wire streaming batch into torch_backend.py

**Files:**
- Modify: `python/applegpu_runtime/torch_backend.py:40-49` and `174-175`
- Create: `python/tests/test_streaming_batch.py`

- [ ] **Step 1: Write failing Python test**

Create `python/tests/test_streaming_batch.py`:

```python
"""Tests for streaming command buffer integration with torch backend."""
import pytest
import torch


def _setup():
    import applegpu_runtime as gpu
    gpu.init_backend()
    gpu.enable_torch_backend()
    from applegpu_runtime.torch_backend import set_eager_mode, ApplegpuTensor
    return gpu, ApplegpuTensor, set_eager_mode


def test_eager_mode_enables_streaming():
    """set_eager_mode(True) should activate streaming batch."""
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)
    # Streaming should be active (we can't directly check from Python,
    # but begin/end should not crash)
    set_eager_mode(False)


def test_eager_mode_toggle_idempotent():
    """Toggling eager mode multiple times should not crash."""
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)
    set_eager_mode(True)  # double enable
    set_eager_mode(False)
    set_eager_mode(False)  # double disable
    set_eager_mode(True)
    set_eager_mode(False)


def test_training_loop_with_streaming():
    """Full training loop works with streaming batch."""
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)

    model = torch.nn.Linear(8, 4)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for step in range(5):
        x = gpu.to_applegpu(torch.randn(16, 8))
        y = gpu.to_applegpu(torch.randn(16, 4))

        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Verify loss is readable and finite
    final_loss = loss.to_torch_cpu().item()
    assert isinstance(final_loss, float) and final_loss >= 0

    set_eager_mode(False)


def test_cpu_fallback_during_streaming():
    """CPU fallback ops (fill, zeros) work during streaming."""
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)

    # zeros uses CPU fallback, which calls _gpu_tensor_to_torch_cpu internally
    x = gpu.to_applegpu(torch.randn(4, 4))
    y = gpu.to_applegpu(torch.randn(4, 4))
    out = x + y  # GPU op

    # to_torch_cpu triggers flush
    result = out.to_torch_cpu()
    assert result.shape == (4, 4)
    assert torch.isfinite(result).all()

    set_eager_mode(False)


def test_readback_mid_streaming():
    """Reading data mid-stream produces correct results."""
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)

    a = gpu.to_applegpu(torch.tensor([1.0, 2.0, 3.0]))
    b = gpu.to_applegpu(torch.tensor([10.0, 20.0, 30.0]))
    c = a + b

    # Intermediate readback forces flush
    result = c.to_torch_cpu()
    assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0]))

    # Continue using GPU after readback
    d = c + a
    result2 = d.to_torch_cpu()
    assert torch.allclose(result2, torch.tensor([12.0, 24.0, 36.0]))

    set_eager_mode(False)
```

- [ ] **Step 2: Modify `set_eager_mode()` to manage streaming batch**

In `python/applegpu_runtime/torch_backend.py`, modify `set_eager_mode()` (line 174):

```python
def set_eager_mode(enabled=True):
    """Enable or disable eager evaluation mode.

    When enabled, training ops are synchronized at eager boundaries (CPU
    readback and in-place state updates) instead of every op creation.
    Also activates streaming command buffer to amortize Metal dispatch.
    """
    global _eager_mode
    if enabled and not _eager_mode:
        _eager_mode = True
        gpu.begin_streaming_batch()
    elif not enabled and _eager_mode:
        gpu.end_streaming_batch()
        _eager_mode = False
```

- [ ] **Step 3: Add flush call to `_gpu_tensor_to_torch_cpu()`**

In `python/applegpu_runtime/torch_backend.py`, modify `_gpu_tensor_to_torch_cpu()` (line 40):

```python
def _gpu_tensor_to_torch_cpu(gpu_t):
    """Convert a GpuTensor to a CPU torch.Tensor.

    In eager/training mode we explicitly flush pending lazy work for this tensor
    before host readback. Flushes the streaming command buffer first so all
    encoded ops are committed to the GPU.
    """
    if _eager_mode:
        gpu.flush_streaming_batch()
        gpu.eval(gpu_t)
    return gpu_t.to_torch()
```

- [ ] **Step 4: Build and run tests**

Run: `uv run maturin develop && uv run pytest python/tests/test_streaming_batch.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All ~760+ tests pass

- [ ] **Step 6: Commit**

```bash
git add python/applegpu_runtime/torch_backend.py python/tests/test_streaming_batch.py
git commit -m "feat: wire streaming batch into torch backend (Phase B)

set_eager_mode() now manages streaming batch lifecycle.
_gpu_tensor_to_torch_cpu() flushes streaming batch before readback.
Training loops amortize Metal commit+wait across many ops."
```

### Task 7: Final benchmark and validation

**Files:**
- None (run existing benchmarks and tests)

- [ ] **Step 1: Run full test suite**

Run: `make test`
Expected: All tests pass

- [ ] **Step 2: Run training benchmark**

Run: `uv run python benchmarks/bench_training.py --models lstm --epochs 5 --hidden-size 128`
Expected: GPU faster than CPU for LSTM h=128 (success metric)

- [ ] **Step 3: Run benchmark with all models**

Run: `uv run python benchmarks/bench_training.py --epochs 3`
Expected: Improved GPU/CPU ratios across all models

- [ ] **Step 4: Compare with Phase A baseline**

Compare the numbers from Task 2 with the Phase B results. Document the improvement.

- [ ] **Step 5: Commit benchmark results as README update**

Update the relevant section of `README.md` with new benchmark numbers if the success metric is met.

```bash
git add README.md
git commit -m "docs: update benchmarks after P1 dispatch overhead reduction"
```
