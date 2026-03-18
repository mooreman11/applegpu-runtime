# PrivateUse1 C++ Backend Implementation Plan (Phases 0-2)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Python `__torch_dispatch__` (20µs/op) with C++ PrivateUse1 dispatch (~200ns/op) by building a thin C++ shim that calls into the existing Rust graph engine via `extern "C"`.

**Architecture:** PyTorch C++ Dispatcher → C++ shim (400 lines) → `extern "C"` Rust FFI → LazyRuntime graph recording → fusion + eval at sync points → streaming Metal CB → GPU. This plan covers Phases 0-2 (foundation). Phases 3-5 (op migration, training validation, polish) will be a separate plan.

**Tech Stack:** Rust (applegpu-core), C++ (libtorch/ATen), Swift (Metal bridge), Python (entry point), `torch.utils.cpp_extension` (build)

**Spec:** `docs/superpowers/specs/2026-03-18-privateuse1-backend-design.md`

---

## File Structure

**Phase 0 (Rust-only):**
- Modify: `crates/core/src/lazy.rs` — add `insert_preallocated()`, modify `eval_single_cb()` to use pre-allocated buffers, add deferred-free tracking
- Create: `crates/core/tests/preallocated_integration.rs` — pre-allocation tests

**Phase 1 (Rust FFI):**
- Create: `crates/core/src/backend_ffi.rs` — `extern "C"` FFI bridge (~500 lines)
- Modify: `crates/core/src/lib.rs` — add `pub mod backend_ffi;`
- Modify: `crates/core/Cargo.toml` — add `smallvec` dependency, `crate-type = ["lib", "staticlib"]`
- Create: `crates/core/tests/backend_ffi_integration.rs` — FFI round-trip tests

**Phase 2 (C++ shim):**
- Create: `backend_cpp/applegpu_backend.cpp` — allocator, ops, fallback, device guard (~400 lines)
- Create: `backend_cpp/applegpu_ffi.h` — C header for extern "C" Rust functions
- Create: `backend_cpp/setup.py` — torch.utils.cpp_extension build config
- Create: `python/applegpu_runtime/cpp_backend.py` — `load_cpp_backend()` entry point
- Create: `python/tests/test_cpp_backend.py` — Python-level integration tests
- Modify: `Makefile` — add `build-cpp-backend`, `test-cpp-backend` targets

---

## Phase 0: Pre-allocated Buffers + Deferred-Free

### Task 1: Add `insert_preallocated()` to LazyRuntime

**Files:**
- Modify: `crates/core/src/lazy.rs:46-70` (near existing insert methods)
- Create: `crates/core/tests/preallocated_integration.rs`

- [ ] **Step 1: Write failing test — pre-allocated tensor survives eval**

Create `crates/core/tests/preallocated_integration.rs`:

```rust
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::tensor::{Tensor, DType};

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn preallocated_tensor_is_materialized() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    // Pre-allocate a buffer (simulating what the C++ allocator would do)
    let buf = rt.pool.acquire(&device, 4 * 4).unwrap(); // 4 floats
    let id = applegpu_core::tensor::next_tensor_id();
    let tensor = Tensor::from_raw(id, vec![4], DType::Float32, buf);

    rt.insert_preallocated(tensor).unwrap();

    // Should be materialized immediately (it has a buffer)
    assert!(rt.is_materialized(id));
    assert!(!rt.is_pending(id));
}

#[test]
fn preallocated_tensor_usable_as_op_input() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    // Pre-allocate two tensors with data
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_preallocated(a).unwrap();
    rt.insert_preallocated(b).unwrap();

    // Record an op using pre-allocated inputs
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();

    // Eval should work — inputs are pre-allocated (materialized), output is new
    rt.eval(&device, sum_id).unwrap();
    let result = rt.read_f32(sum_id).unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core --test preallocated_integration`
Expected: FAIL — `insert_preallocated` method doesn't exist

- [ ] **Step 3: Implement `insert_preallocated()`**

In `crates/core/src/lazy.rs`, after `insert_tensor_for()` (line ~70), add:

```rust
    /// Insert a pre-allocated tensor (has a buffer, no graph node).
    /// Used by the PrivateUse1 C++ backend where PyTorch's allocator
    /// creates Metal buffers before ops are recorded.
    /// Skips scheduler quota accounting — PyTorch manages lifetime via refcounting.
    pub fn insert_preallocated(&mut self, tensor: Tensor) -> Result<()> {
        let id = tensor.meta.id;
        self.tensors.insert(id, tensor);
        Ok(())
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core --test preallocated_integration`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/lazy.rs crates/core/tests/preallocated_integration.rs
git commit -m "feat: add insert_preallocated() to LazyRuntime

Pre-allocated tensors have buffers but no graph nodes. They are treated
as materialized leaves by topo_sort. Used by PrivateUse1 backend where
PyTorch's allocator creates Metal buffers before ops are recorded.
Skips scheduler quota accounting (PyTorch manages lifetime)."
```

### Task 2: Modify eval_single_cb() to use pre-allocated output buffers

**Files:**
- Modify: `crates/core/src/lazy.rs:294-366` (`eval_single_cb`)
- Modify: `crates/core/tests/preallocated_integration.rs`

- [ ] **Step 1: Write failing test — eval writes into pre-allocated output buffer**

Add to `crates/core/tests/preallocated_integration.rs`:

```rust
#[test]
fn eval_writes_into_preallocated_output() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    // Create input tensors normally
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    // Record an add op
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();

    // Pre-allocate the OUTPUT buffer before eval
    let out_buf = rt.pool.acquire(&device, 4 * 4).unwrap(); // 4 floats
    let out_ptr = out_buf.contents() as usize;
    let out_tensor = Tensor::from_raw(sum_id, vec![4], DType::Float32, out_buf);
    rt.insert_preallocated(out_tensor).unwrap();

    // Eval should write into the pre-allocated buffer, NOT allocate a new one
    rt.eval(&device, sum_id).unwrap();
    let result = rt.read_f32(sum_id).unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);

    // Verify it used the same buffer (same pointer)
    let final_ptr = rt.get_tensor_ptr(sum_id).unwrap();
    assert_eq!(out_ptr, final_ptr, "eval should write into pre-allocated buffer");
}
```

Note: `get_tensor_ptr` is a new helper we'll need. Add it in Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core --test preallocated_integration -- eval_writes_into_preallocated_output`
Expected: FAIL — eval allocates a new buffer instead of using the pre-allocated one (assertion on pointer equality fails), and `get_tensor_ptr` doesn't exist.

- [ ] **Step 3: Implement pre-allocated buffer support in eval_single_cb**

In `crates/core/src/lazy.rs`, add a helper method:

```rust
    /// Get the raw buffer pointer for a materialized tensor (for testing).
    pub fn get_tensor_ptr(&self, id: u64) -> Result<usize> {
        let t = self.get_tensor(id)?;
        Ok(t.buffer.contents() as usize)
    }
```

Then modify `eval_single_cb()`. In the main op path (around line 353-360), change the buffer acquisition to check for pre-allocated:

```rust
                let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
                // Use pre-allocated buffer if available, otherwise acquire from pool
                let out = if let Some(existing) = self.tensors.remove(&node.id) {
                    existing
                } else {
                    let out_buf = self.pool.acquire(device, out_size)?;
                    Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf)
                };
                let cb = self.execute_node_nb(device, queue, &node, &out)?;
```

Do the same for the `MaxPool2dWithIndices` special case (lines 317-323):

```rust
                    let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
                    let out = if let Some(existing) = self.tensors.remove(&node.id) {
                        existing
                    } else {
                        let out_buf = self.pool.acquire(device, out_size)?;
                        Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf)
                    };
                    let idx_size = node.out_shape.numel() * DType::Int32.size_bytes();
                    let idx_tensor = if let Some(existing) = self.tensors.remove(&indices_id) {
                        existing
                    } else {
                        let idx_buf = self.pool.acquire(device, idx_size)?;
                        Tensor::from_raw(indices_id, node.out_shape.dims().to_vec(), DType::Int32, idx_buf)
                    };
```

Also skip `is_materialized` check at line 310 when the tensor is pre-allocated but has a pending graph node — the node should still be evaluated to write data into the pre-allocated buffer. Change:

```rust
                if self.is_materialized(node_id) { continue; }
```

To:

```rust
                // Skip if materialized AND no pending graph node
                // (pre-allocated tensors are materialized but still need eval)
                if self.is_materialized(node_id) && !self.graph.has_node(node_id) { continue; }
```

Wait — this changes behavior. Actually, the simpler approach: if a tensor is pre-allocated (in `self.tensors`), it IS materialized, so `is_materialized` returns true and the node is skipped. But we WANT the node to be executed (to write data into the buffer).

The fix: `insert_preallocated` should NOT insert into `self.tensors` yet. Instead, maintain a separate `preallocated_buffers: HashMap<u64, Buffer>` that `eval_single_cb` checks before `pool.acquire()`. The tensor only moves to `self.tensors` after eval writes into it.

Revised `insert_preallocated`:

```rust
    /// Pre-allocated buffers for tensors whose output will be written during eval.
    /// Maps tensor_id -> Buffer. Used by PrivateUse1 backend.
    preallocated: HashMap<u64, Buffer>,
```

Add to `LazyRuntime` struct (after `secondary_outputs` at line 30). Initialize in `new()`:

```rust
    preallocated: HashMap::new(),
```

Revised `insert_preallocated`:

```rust
    pub fn insert_preallocated(&mut self, tensor: Tensor) -> Result<()> {
        let id = tensor.meta.id;
        let buffer = tensor.into_buffer();
        self.preallocated.insert(id, buffer);
        Ok(())
    }
```

Wait — `Tensor::into_buffer()` may not exist. Check. Actually, let's keep it simpler: store the full Tensor in preallocated and move it during eval:

```rust
    /// Stash a pre-allocated buffer for a tensor that will be written during eval.
    /// The buffer is held until eval_single_cb needs it as an output buffer.
    /// For tensors that are already materialized (data loaded from host),
    /// insert directly into tensors instead.
    pub fn insert_preallocated_buffer(&mut self, id: u64, buffer: Buffer) {
        self.preallocated.insert(id, buffer);
    }

    /// Insert a fully materialized pre-allocated tensor (has data, no graph node).
    /// Used for tensors created by PyTorch's allocator with host data (e.g., from_numpy).
    pub fn insert_preallocated(&mut self, tensor: Tensor) -> Result<()> {
        let id = tensor.meta.id;
        self.tensors.insert(id, tensor);
        Ok(())
    }
```

Then in `eval_single_cb`, the buffer acquisition becomes:

```rust
                let out = if let Some(pre_buf) = self.preallocated.remove(&node.id) {
                    Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, pre_buf)
                } else {
                    let out_buf = self.pool.acquire(device, out_size)?;
                    Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf)
                };
```

This is cleaner — pre-allocated buffers are consumed during eval, and the tensor ends up in `self.tensors` as usual.

Update the test accordingly — use `insert_preallocated_buffer` for the output buffer case and `insert_preallocated` for the input tensor case.

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core --test preallocated_integration`
Expected: PASS

- [ ] **Step 5: Run full Rust suite for regressions**

Run: `cargo test -p applegpu-core`
Expected: All pass (pre-allocated is opt-in, no existing code uses it)

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/lazy.rs crates/core/tests/preallocated_integration.rs
git commit -m "feat: eval_single_cb uses pre-allocated output buffers

When a tensor_id has a pre-allocated buffer (via insert_preallocated_buffer),
eval writes into it instead of acquiring from the pool. This supports the
PrivateUse1 backend where PyTorch's allocator creates Metal buffers
before ops are recorded."
```

### Task 3: Add deferred-free tracking

**Files:**
- Modify: `crates/core/src/lazy.rs`
- Modify: `crates/core/tests/preallocated_integration.rs`

- [ ] **Step 1: Write failing test — free is deferred when tensor has graph dependents**

Add to `crates/core/tests/preallocated_integration.rs`:

```rust
#[test]
fn deferred_free_keeps_tensor_alive() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    // Create a tensor and register it
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let a_id = a.meta.id;
    rt.insert_preallocated(a).unwrap();

    // Record an op that depends on it
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let b_id = b.meta.id;
    rt.insert_preallocated(b).unwrap();
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();

    // Try to free a — should be deferred because sum_id depends on it
    let freed = rt.try_deferred_free(a_id);
    assert!(!freed, "Should not free: tensor has graph dependents");
    assert!(rt.is_materialized(a_id), "Tensor should still be alive");

    // Eval the dependent op
    rt.eval(&device, sum_id).unwrap();

    // Now free should succeed (no more dependents)
    let freed2 = rt.try_deferred_free(a_id);
    assert!(freed2, "Should free: no more dependents");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core --test preallocated_integration -- deferred_free`
Expected: FAIL — `try_deferred_free` doesn't exist

- [ ] **Step 3: Implement deferred-free**

In `crates/core/src/lazy.rs`, add to the `LazyRuntime` struct:

```rust
    /// Tensor IDs marked for deferred free. Checked after each eval.
    deferred_frees: Vec<u64>,
```

Initialize in `new()`:

```rust
    deferred_frees: Vec::new(),
```

Add methods:

```rust
    /// Try to free a tensor. If it is referenced by pending graph nodes,
    /// defer the free until after those nodes are evaluated.
    /// Returns true if the tensor was freed immediately, false if deferred.
    pub fn try_deferred_free(&mut self, id: u64) -> bool {
        // Check if any pending graph node references this tensor as input
        if self.graph.is_referenced(id) {
            self.deferred_frees.push(id);
            return false;
        }
        // Safe to free now
        if let Some(tensor) = self.tensors.remove(&id) {
            self.pool.release(tensor.buffer);
        }
        self.graph.remove_node(id); // no-op if not in graph
        true
    }

    /// Process deferred frees after eval. Called at the end of eval().
    fn process_deferred_frees(&mut self) {
        let ids: Vec<u64> = self.deferred_frees.drain(..).collect();
        for id in ids {
            if self.graph.is_referenced(id) {
                // Still referenced — re-defer
                self.deferred_frees.push(id);
            } else if let Some(tensor) = self.tensors.remove(&id) {
                self.pool.release(tensor.buffer);
            }
        }
    }
```

Add `is_referenced` to `Graph` in `crates/core/src/graph.rs`:

```rust
    /// Check if a tensor ID is referenced as input by any node in the graph.
    pub fn is_referenced(&self, id: u64) -> bool {
        self.nodes.values().any(|node| node.inputs.contains(&id))
    }
```

Call `process_deferred_frees()` at the end of `eval()` and `eval_single_cb()` (before returning `loop_result`).

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core --test preallocated_integration`
Expected: PASS

- [ ] **Step 5: Run full Rust suite**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/lazy.rs crates/core/src/graph.rs crates/core/tests/preallocated_integration.rs
git commit -m "feat: deferred-free for tensors with graph dependents

try_deferred_free() checks if a tensor is referenced by pending graph
nodes. If yes, defers the free until after eval materializes dependents.
Prevents premature buffer release when PyTorch's refcounting drops a
tensor that the computation graph still needs."
```

---

## Phase 1: Rust FFI Bridge

### Task 4: Create backend_ffi module with init/alloc/free

**Files:**
- Create: `crates/core/src/backend_ffi.rs`
- Modify: `crates/core/src/lib.rs` (add `pub mod backend_ffi;`)
- Create: `crates/core/tests/backend_ffi_integration.rs`

- [ ] **Step 1: Write failing test — FFI alloc/free round-trip**

Create `crates/core/tests/backend_ffi_integration.rs`:

```rust
use applegpu_core::backend_ffi;

#[test]
fn ffi_init_succeeds() {
    let ok = backend_ffi::applegpu_ffi_init();
    assert!(ok, "FFI init should succeed on macOS with Metal");
}

#[test]
fn ffi_alloc_free_roundtrip() {
    backend_ffi::applegpu_ffi_init();
    let mut tensor_id: u64 = 0;
    let ptr = backend_ffi::applegpu_ffi_alloc(
        256, // 256 bytes
        0,   // dtype: Float32 (map from i8)
        &mut tensor_id,
    );
    assert!(!ptr.is_null(), "alloc should return non-null pointer");
    assert!(tensor_id > 0, "should assign a valid tensor_id");

    // Free should not crash
    backend_ffi::applegpu_ffi_free(tensor_id);
}

#[test]
fn ffi_alloc_returns_writable_memory() {
    backend_ffi::applegpu_ffi_init();
    let mut tensor_id: u64 = 0;
    let ptr = backend_ffi::applegpu_ffi_alloc(
        16, // 4 floats
        0,  // Float32
        &mut tensor_id,
    );
    assert!(!ptr.is_null());

    // Write data through the pointer (storageModeShared = CPU writable)
    unsafe {
        let floats = ptr as *mut f32;
        *floats.add(0) = 1.0;
        *floats.add(1) = 2.0;
        *floats.add(2) = 3.0;
        *floats.add(3) = 4.0;
    }

    backend_ffi::applegpu_ffi_free(tensor_id);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core --test backend_ffi_integration`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Create `backend_ffi.rs` with init/alloc/free**

Add `pub mod backend_ffi;` to `crates/core/src/lib.rs` (after line 17).

Create `crates/core/src/backend_ffi.rs`:

```rust
//! C-ABI FFI bridge for the PrivateUse1 C++ backend.
//!
//! All functions are `extern "C"` and use only C-compatible types.
//! The C++ shim calls these functions; all real logic lives in Rust.

use std::sync::Mutex;
use once_cell::sync::OnceCell;

use crate::backend;
use crate::buffer::Buffer;
use crate::device::Device;
use crate::lazy::LazyRuntime;
use crate::tensor::{DType, Tensor, next_tensor_id};

/// Global runtime state for the FFI bridge.
struct FfiState {
    runtime: Mutex<LazyRuntime>,
    device: Device,
}

static FFI_STATE: OnceCell<FfiState> = OnceCell::new();

fn get_state() -> &'static FfiState {
    FFI_STATE.get().expect("applegpu FFI not initialized — call applegpu_ffi_init() first")
}

/// Thread-local error message for the last failed FFI call.
/// Stored as CString so the pointer returned by last_error is null-terminated
/// and valid until the next set_error call on the same thread.
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<std::ffi::CString>> = std::cell::RefCell::new(None);
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = std::ffi::CString::new(msg).ok();
    });
}

// ── Init ──────────────────────────────────────────────────────────

/// Initialize the FFI backend. Must be called once before any other FFI function.
/// Returns true on success, false on failure (check applegpu_ffi_last_error).
#[no_mangle]
pub extern "C" fn applegpu_ffi_init() -> bool {
    if FFI_STATE.get().is_some() {
        return true; // already initialized
    }

    let device = match Device::new() {
        Ok(d) => d,
        Err(e) => {
            set_error(format!("Failed to create Metal device: {}", e));
            return false;
        }
    };

    let state = FfiState {
        runtime: Mutex::new(LazyRuntime::new()),
        device,
    };

    FFI_STATE.set(state).unwrap_or_else(|_| {
        // Race condition — another thread initialized. That's fine.
    });
    true
}

// ── Error ─────────────────────────────────────────────────────────

/// Get the last error message as a null-terminated C string.
/// Valid until the next set_error call on the same thread.
/// Returns null if no error.
#[no_mangle]
pub extern "C" fn applegpu_ffi_last_error() -> *const std::ffi::c_char {
    LAST_ERROR.with(|e| {
        match e.borrow().as_ref() {
            Some(cstr) => cstr.as_ptr(),
            None => std::ptr::null(),
        }
    })
}

// ── Alloc/Free ────────────────────────────────────────────────────

/// Allocate a Metal buffer and register it as a pre-allocated tensor.
/// Returns the buffer's data_ptr (storageModeShared, CPU+GPU accessible).
/// Writes the assigned tensor_id to *out_tensor_id.
/// Returns null on failure (check applegpu_ffi_last_error).
#[no_mangle]
pub extern "C" fn applegpu_ffi_alloc(
    size_bytes: u64,
    _dtype_i8: i8,
    out_tensor_id: *mut u64,
) -> *mut u8 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    let size = size_bytes as usize;
    if size == 0 {
        let id = next_tensor_id();
        unsafe { *out_tensor_id = id; }
        return std::ptr::null_mut(); // zero-size allocation
    }

    match rt.pool.acquire(&state.device, size) {
        Ok(buffer) => {
            let id = next_tensor_id();
            let ptr = buffer.contents();
            rt.insert_preallocated_buffer(id, buffer);
            unsafe { *out_tensor_id = id; }
            ptr
        }
        Err(e) => {
            set_error(format!("alloc failed: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Free a tensor. Defers if the tensor is referenced by pending graph nodes.
#[no_mangle]
pub extern "C" fn applegpu_ffi_free(tensor_id: u64) {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    rt.try_deferred_free(tensor_id);
}

// ── Sync ──────────────────────────────────────────────────────────

/// Evaluate a tensor (flush the graph up to this tensor).
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_eval(tensor_id: u64) -> i32 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match rt.eval(&state.device, tensor_id) {
        Ok(()) => 0,
        Err(e) => {
            set_error(format!("eval failed: {}", e));
            -1
        }
    }
}

/// Flush the streaming batch and wait for GPU completion.
#[no_mangle]
pub extern "C" fn applegpu_ffi_synchronize() {
    crate::compute::flush_streaming_batch();
}

// ── Metadata ──────────────────────────────────────────────────────

/// Get tensor shape. Writes dims to out_dims, ndim to out_ndim.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_shape(
    tensor_id: u64,
    out_dims: *mut u64,
    out_ndim: *mut u32,
) -> i32 {
    let state = get_state();
    let rt = state.runtime.lock().unwrap();
    match rt.shape(tensor_id) {
        Ok(dims) => {
            unsafe {
                *out_ndim = dims.len() as u32;
                for (i, &d) in dims.iter().enumerate() {
                    *out_dims.add(i) = d as u64;
                }
            }
            0
        }
        Err(e) => {
            set_error(format!("shape failed: {}", e));
            -1
        }
    }
}

/// Get tensor dtype as wire protocol discriminant (see DType::to_wire).
/// Returns -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_dtype(tensor_id: u64) -> i8 {
    let state = get_state();
    let rt = state.runtime.lock().unwrap();
    match rt.dtype(tensor_id) {
        Ok(dt) => dt.to_wire() as i8,
        Err(_) => -1,
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core --test backend_ffi_integration`
Expected: PASS

- [ ] **Step 5: Run full Rust suite**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/backend_ffi.rs crates/core/src/lib.rs crates/core/tests/backend_ffi_integration.rs
git commit -m "feat: add Rust FFI bridge for PrivateUse1 backend (Phase 1)

extern \"C\" functions for init, alloc, free, eval, synchronize, shape,
dtype. Global Mutex<LazyRuntime> with OnceCell init guard. Thread-local
error messages for C++ TORCH_CHECK integration."
```

### Task 5: Add core op FFI functions (add, matmul, relu, copy)

**Files:**
- Modify: `crates/core/src/backend_ffi.rs`
- Modify: `crates/core/tests/backend_ffi_integration.rs`

- [ ] **Step 1: Write failing test — FFI add + eval produces correct result**

Add to `crates/core/tests/backend_ffi_integration.rs`:

```rust
#[test]
fn ffi_add_eval_readback() {
    backend_ffi::applegpu_ffi_init();

    // Allocate two tensors
    let mut a_id: u64 = 0;
    let mut b_id: u64 = 0;
    let a_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut a_id); // 4 floats
    let b_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut b_id);

    // Write data
    unsafe {
        let a = a_ptr as *mut f32;
        let b = b_ptr as *mut f32;
        for i in 0..4 {
            *a.add(i) = (i + 1) as f32;        // [1, 2, 3, 4]
            *b.add(i) = ((i + 1) * 10) as f32; // [10, 20, 30, 40]
        }
    }

    // Register tensors with shape metadata
    backend_ffi::applegpu_ffi_register_tensor(a_id, [4].as_ptr(), 1, 0); // Float32
    backend_ffi::applegpu_ffi_register_tensor(b_id, [4].as_ptr(), 1, 0);

    // Record add op
    let result_id = backend_ffi::applegpu_ffi_add(a_id, b_id);
    assert!(result_id > 0);

    // Eval
    let rc = backend_ffi::applegpu_ffi_eval(result_id);
    assert_eq!(rc, 0, "eval should succeed");

    // Readback
    let mut out = [0.0f32; 4];
    let rc = backend_ffi::applegpu_ffi_read_f32(result_id, out.as_mut_ptr(), 4);
    assert_eq!(rc, 0);
    assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);

    backend_ffi::applegpu_ffi_free(a_id);
    backend_ffi::applegpu_ffi_free(b_id);
    backend_ffi::applegpu_ffi_free(result_id);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core --test backend_ffi_integration -- ffi_add_eval`
Expected: FAIL — `applegpu_ffi_add`, `applegpu_ffi_register_tensor`, `applegpu_ffi_read_f32` don't exist

- [ ] **Step 3: Implement op FFI functions**

Add to `crates/core/src/backend_ffi.rs`:

```rust
// ── Tensor Registration ───────────────────────────────────────────

/// Register shape metadata for a pre-allocated tensor.
/// Must be called after alloc, before the tensor is used as an op input.
#[no_mangle]
pub extern "C" fn applegpu_ffi_register_tensor(
    tensor_id: u64,
    dims_ptr: *const u64,
    ndim: u32,
    dtype_i8: i8,
) -> i32 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    let dims: Vec<usize> = unsafe {
        (0..ndim as usize).map(|i| *dims_ptr.add(i) as usize).collect()
    };
    let dtype = match DType::from_wire(dtype_i8 as u32) {
        Some(d) => d,
        None => {
            set_error(format!("Invalid dtype wire value: {}", dtype_i8));
            return -1;
        }
    };

    // Move buffer from preallocated to tensors with proper metadata
    match rt.materialize_preallocated(tensor_id, dims, dtype) {
        Ok(()) => 0,
        Err(e) => {
            set_error(format!("register_tensor failed: {}", e));
            -1
        }
    }
}

// ── Ops ───────────────────────────────────────────────────────────

/// Record an add op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_add(a_id: u64, b_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::add(&mut rt, a_id, b_id) {
        Ok(id) => id,
        Err(e) => {
            set_error(format!("add failed: {}", e));
            0
        }
    }
}

/// Record a matmul op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_matmul(a_id: u64, b_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::matmul(&mut rt, a_id, b_id) {
        Ok(id) => id,
        Err(e) => {
            set_error(format!("matmul failed: {}", e));
            0
        }
    }
}

/// Record a relu op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_relu(input_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::relu(&mut rt, input_id) {
        Ok(id) => id,
        Err(e) => {
            set_error(format!("relu failed: {}", e));
            0
        }
    }
}

/// Copy tensor data. Used for H2D/D2H transfers (CPU fallback).
/// src_id tensor must be materialized. Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_copy(
    src_id: u64,
    dst_id: u64,
) -> i32 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    // Auto-eval source if pending
    if rt.is_pending(src_id) {
        if let Err(e) = rt.eval(&state.device, src_id) {
            set_error(format!("copy: eval src failed: {}", e));
            return -1;
        }
    }

    // Read src bytes, write to dst buffer
    match rt.read_bytes(src_id) {
        Ok(bytes) => {
            // Write into dst's pre-allocated buffer
            if let Ok(dst_tensor) = rt.get_tensor_ptr(dst_id) {
                unsafe {
                    let dst_ptr = dst_tensor as *mut u8;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst_ptr, bytes.len());
                }
                0
            } else {
                set_error(format!("copy: dst tensor {} not found", dst_id));
                -1
            }
        }
        Err(e) => {
            set_error(format!("copy failed: {}", e));
            -1
        }
    }
}

// ── Readback ──────────────────────────────────────────────────────

/// Read tensor data as f32 into the provided buffer.
/// Flushes streaming batch and evaluates if needed.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_read_f32(
    tensor_id: u64,
    out_ptr: *mut f32,
    max_elements: u64,
) -> i32 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    // Auto-eval if pending
    if rt.is_pending(tensor_id) {
        if let Err(e) = rt.eval(&state.device, tensor_id) {
            set_error(format!("eval failed during readback: {}", e));
            return -1;
        }
    }

    match rt.read_f32(tensor_id) {
        Ok(data) => {
            let n = std::cmp::min(data.len(), max_elements as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), out_ptr, n);
            }
            0
        }
        Err(e) => {
            set_error(format!("read_f32 failed: {}", e));
            -1
        }
    }
}
```

Note: Use the existing `DType::from_wire(d as u32)` for dtype conversion (defined in `tensor.rs` lines 76-91). The wire mapping is: 0=Float32, 1=Float16, 2=Float64, 3=Int8, 4=Int16, 5=Int32, 6=Int64, 7=UInt8, 8=UInt32, 9=Bool, 10=BFloat16. Do NOT create a new `from_i8` — reuse `from_wire` to avoid mapping conflicts.

Also, `rt.preallocated` and `rt.tensors` are private fields. The FFI functions that need to access them should either:
- Use existing public methods (`insert_preallocated`, etc.)
- Or add new public methods as needed

The `applegpu_ffi_register_tensor` function directly accesses `rt.preallocated.remove()` and `rt.tensors.insert()`. These are private. Add a public method:

```rust
    /// Move a pre-allocated buffer to a materialized tensor with shape metadata.
    pub fn materialize_preallocated(&mut self, id: u64, dims: Vec<usize>, dtype: DType) -> Result<()> {
        let buffer = self.preallocated.remove(&id).ok_or_else(|| {
            GpuError::GraphError(format!("Tensor {} not in preallocated buffers", id))
        })?;
        let tensor = Tensor::from_raw(id, dims, dtype, buffer);
        self.tensors.insert(id, tensor);
        Ok(())
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core --test backend_ffi_integration`
Expected: PASS

- [ ] **Step 5: Run full Rust suite**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/backend_ffi.rs crates/core/src/lazy.rs crates/core/src/tensor.rs crates/core/tests/backend_ffi_integration.rs
git commit -m "feat: add op FFI functions (add, matmul, relu) + readback

FFI bridge now supports: register_tensor (set shape/dtype on pre-allocated
buffer), add/matmul/relu (record graph ops), read_f32 (eval + readback).
Complete alloc → register → op → eval → readback round-trip working."
```

### Task 6: Add staticlib output to Cargo.toml

**Files:**
- Modify: `crates/core/Cargo.toml`

- [ ] **Step 1: Check current crate-type**

Read `crates/core/Cargo.toml` to see if there's a `[lib]` section.

- [ ] **Step 2: Add staticlib output**

Ensure `crates/core/Cargo.toml` has:

```toml
[lib]
crate-type = ["lib", "staticlib"]
```

This makes `cargo build -p applegpu-core --release` produce both `libapplegpu_core.a` (for C++ linking) and the standard rlib (for Rust consumers).

- [ ] **Step 3: Build and verify .a file is produced**

Run: `cargo build -p applegpu-core --release && ls -la target/release/libapplegpu_core.a`
Expected: File exists

- [ ] **Step 4: Commit**

```bash
git add crates/core/Cargo.toml
git commit -m "build: add staticlib output for C++ backend linking

Produces libapplegpu_core.a alongside the regular rlib."
```

---

## Phase 2: C++ Shim (Minimum Viable)

### Task 7: Create C header for Rust FFI functions

**Files:**
- Create: `backend_cpp/applegpu_ffi.h`

- [ ] **Step 1: Write the C header**

Create `backend_cpp/applegpu_ffi.h`:

```c
#ifndef APPLEGPU_FFI_H
#define APPLEGPU_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Lifecycle */
bool applegpu_ffi_init(void);
const char* applegpu_ffi_last_error(void);

/* Allocation */
uint8_t* applegpu_ffi_alloc(uint64_t size_bytes, int8_t dtype_i8, uint64_t* out_tensor_id);
void applegpu_ffi_free(uint64_t tensor_id);

/* Tensor metadata */
int32_t applegpu_ffi_register_tensor(uint64_t tensor_id, const uint64_t* dims, uint32_t ndim, int8_t dtype_i8);
int32_t applegpu_ffi_shape(uint64_t tensor_id, uint64_t* out_dims, uint32_t* out_ndim);
int8_t applegpu_ffi_dtype(uint64_t tensor_id);

/* Ops (return output tensor_id, 0 on failure) */
uint64_t applegpu_ffi_add(uint64_t a_id, uint64_t b_id);
uint64_t applegpu_ffi_matmul(uint64_t a_id, uint64_t b_id);
uint64_t applegpu_ffi_relu(uint64_t input_id);
int32_t applegpu_ffi_copy(uint64_t src_id, uint64_t dst_id);

/* Eval / sync */
int32_t applegpu_ffi_eval(uint64_t tensor_id);
void applegpu_ffi_synchronize(void);

/* Readback */
int32_t applegpu_ffi_read_f32(uint64_t tensor_id, float* out_ptr, uint64_t max_elements);

#ifdef __cplusplus
}
#endif

#endif /* APPLEGPU_FFI_H */
```

- [ ] **Step 2: Commit**

```bash
git add backend_cpp/applegpu_ffi.h
git commit -m "build: add C header for Rust FFI functions

Declares the extern \"C\" interface that the C++ shim links against."
```

### Task 8: Create C++ shim with custom allocator and minimum ops

**Files:**
- Create: `backend_cpp/applegpu_backend.cpp`
- Create: `backend_cpp/setup.py`

- [ ] **Step 1: Create the C++ shim**

Create `backend_cpp/applegpu_backend.cpp`:

```cpp
#include <torch/torch.h>
#include <torch/library.h>
#include <c10/core/impl/alloc_cpu.h>
#include <ATen/native/CPUFallback.h>
#include "applegpu_ffi.h"

// ── Helpers ──────────────────────────────────────────────────────

namespace {

// Context stored in c10::DataPtr for each applegpu tensor
struct TensorContext {
    uint64_t tensor_id;
};

void applegpu_deleter(void* ptr) {
    // DataPtr context carries the TensorContext
    // Note: the actual free happens via the DataPtr's context deleter below
}

uint64_t get_tensor_id(const at::Tensor& t) {
    auto ctx = static_cast<TensorContext*>(t.storage().data_ptr().get_context());
    TORCH_CHECK(ctx != nullptr, "applegpu tensor has no context (not an applegpu tensor?)");
    return ctx->tensor_id;
}

void check_ffi_error(int32_t rc, const char* op_name) {
    if (rc != 0) {
        const char* err = applegpu_ffi_last_error();
        TORCH_CHECK(false, op_name, " failed: ", err ? err : "unknown error");
    }
}

} // namespace

// ── Allocator ────────────────────────────────────────────────────

struct ApplegpuAllocator final : public c10::Allocator {
    c10::DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return {nullptr, nullptr, &applegpu_deleter, c10::Device(c10::DeviceType::PrivateUse1, 0)};
        }
        uint64_t tensor_id = 0;
        void* ptr = applegpu_ffi_alloc(nbytes, 0 /*dtype*/, &tensor_id);
        TORCH_CHECK(ptr != nullptr, "applegpu alloc failed: ",
                     applegpu_ffi_last_error() ? applegpu_ffi_last_error() : "unknown");

        auto* ctx = new TensorContext{tensor_id};
        auto deleter = [](void* ctx_ptr) {
            auto* tc = static_cast<TensorContext*>(ctx_ptr);
            applegpu_ffi_free(tc->tensor_id);
            delete tc;
        };
        return {ptr, ctx, deleter, c10::Device(c10::DeviceType::PrivateUse1, 0)};
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &applegpu_deleter;
    }
};

static ApplegpuAllocator global_allocator;

// ── Op Implementations ───────────────────────────────────────────

at::Tensor applegpu_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt
) {
    auto dtype = dtype_opt.value_or(at::ScalarType::Float);
    int64_t nbytes = at::detail::computeStorageNbytes(size, stride, at::elementSize(dtype));

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        nbytes,
        global_allocator.allocate(nbytes),
        /*resizable=*/false
    );

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        at::scalarTypeToTypeMeta(dtype)
    );
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

    // Register shape metadata in Rust
    uint64_t tid = get_tensor_id(tensor);
    std::vector<uint64_t> dims(size.begin(), size.end());
    int8_t dtype_i8 = 0; // TODO: map ScalarType to our DType enum
    applegpu_ffi_register_tensor(tid, dims.data(), dims.size(), dtype_i8);

    return tensor;
}

at::Tensor applegpu_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt
) {
    auto strides = c10::get_contiguous_strides(size);
    return applegpu_empty_strided(size, strides, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// CPU fallback for unregistered ops
void applegpu_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack
) {
    // Flush streaming batch before reading any tensor data
    applegpu_ffi_synchronize();
    at::native::cpu_fallback(op, stack);
}

// ── Registration ─────────────────────────────────────────────────

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", applegpu_empty_memory_format);
    m.impl("empty_strided", applegpu_empty_strided);
    // TODO: add remaining minimum ops (copy, resize, view, etc.)
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&applegpu_cpu_fallback>());
}

// ── Module init ──────────────────────────────────────────────────

static auto init = []() {
    applegpu_ffi_init();
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &global_allocator);
    return true;
}();
```

Note: This is the MVP. It has `empty_strided`, `empty.memory_format`, and a CPU fallback. More ops will be added in Phase 3.

- [ ] **Step 2: Create setup.py**

Create `backend_cpp/setup.py`:

```python
import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Build Rust static lib first
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.check_call(["cargo", "build", "-p", "applegpu-core", "--release"], cwd=workspace_root)

# Find paths
rust_lib = os.path.join(workspace_root, "target", "release", "libapplegpu_core.a")
swift_lib_dir = os.path.join(workspace_root, "swift", ".build", "release")
swift_lib = os.path.join(swift_lib_dir, "libAppleGPUBridge.a")

# Swift runtime path
sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
swift_bin = subprocess.check_output(["xcrun", "--toolchain", "default", "--find", "swift"]).decode().strip()
swift_lib_path = os.path.join(os.path.dirname(os.path.dirname(swift_bin)), "lib", "swift", "macosx")

setup(
    name="applegpu_backend",
    ext_modules=[
        CppExtension(
            name="applegpu_backend",
            sources=["applegpu_backend.cpp"],
            include_dirs=["."],
            extra_objects=[rust_lib, swift_lib],
            extra_link_args=[
                f"-L{swift_lib_path}",
                f"-L{sdk_path}/usr/lib/swift",
                "-lswiftCore",
                "-framework", "Metal",
                "-framework", "MetalPerformanceShaders",
                "-framework", "Foundation",
            ],
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

- [ ] **Step 3: Build the C++ extension**

Run: `cd backend_cpp && uv run python setup.py build_ext --inplace`
Expected: Builds successfully, produces `applegpu_backend.*.so`

If the build fails, debug linking issues. Common problems:
- Missing Swift symbols → check `-lswiftCore` and swift_lib_path
- Missing Rust symbols → check that `libapplegpu_core.a` includes `applegpu_ffi_*` symbols: `nm target/release/libapplegpu_core.a | grep applegpu_ffi`
- libtorch headers not found → ensure PyTorch is installed in the venv

- [ ] **Step 4: Commit**

```bash
git add backend_cpp/applegpu_backend.cpp backend_cpp/setup.py
git commit -m "feat: add C++ shim for PrivateUse1 backend (Phase 2 MVP)

Custom c10::Allocator backed by Metal buffers, empty_strided op,
CPU fallback with streaming batch flush. Links against Rust static lib
and Swift static lib."
```

### Task 9: Python entry point and integration tests

**Files:**
- Create: `python/applegpu_runtime/cpp_backend.py`
- Create: `python/tests/test_cpp_backend.py`
- Modify: `Makefile`

- [ ] **Step 1: Create Python entry point**

Create `python/applegpu_runtime/cpp_backend.py`:

```python
"""PrivateUse1 C++ backend loader for applegpu_runtime.

Usage:
    from applegpu_runtime.cpp_backend import load_cpp_backend
    load_cpp_backend()
    x = torch.empty(3, 3, device='applegpu')
"""
import os
import glob
import torch


def _find_backend_dylib():
    """Find the compiled C++ backend shared library."""
    backend_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend_cpp')
    patterns = [
        os.path.join(backend_dir, 'applegpu_backend*.so'),
        os.path.join(backend_dir, 'applegpu_backend*.dylib'),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "applegpu_backend shared library not found. "
        "Run: cd backend_cpp && uv run python setup.py build_ext --inplace"
    )


_loaded = False

def load_cpp_backend():
    """Load the PrivateUse1 C++ backend for applegpu.

    After calling this, torch.empty(..., device='applegpu') will dispatch
    through C++ to the Rust graph engine (no Python per-op overhead).
    """
    global _loaded
    if _loaded:
        return

    dylib_path = _find_backend_dylib()
    torch.ops.load_library(dylib_path)
    torch.utils.rename_privateuse1_backend("applegpu")
    torch.utils.generate_methods_for_privateuse1_backend("applegpu")
    _loaded = True
```

- [ ] **Step 2: Create integration tests**

Create `python/tests/test_cpp_backend.py`:

```python
"""Integration tests for the PrivateUse1 C++ backend."""
import pytest
import torch


def _load():
    """Load C++ backend. Skip if not built."""
    try:
        from applegpu_runtime.cpp_backend import load_cpp_backend
        load_cpp_backend()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"C++ backend not built: {e}")


def test_empty_tensor():
    """torch.empty on applegpu device creates a tensor."""
    _load()
    t = torch.empty(3, 4, device='applegpu')
    assert t.device.type == 'applegpu'
    assert t.shape == (3, 4)
    assert t.dtype == torch.float32


def test_empty_different_dtypes():
    """empty works for various dtypes."""
    _load()
    for dtype in [torch.float32, torch.float16, torch.int32]:
        t = torch.empty(2, 3, device='applegpu', dtype=dtype)
        assert t.dtype == dtype


def test_tensor_to_cpu():
    """Tensor can be copied to CPU."""
    _load()
    t = torch.empty(4, device='applegpu')
    cpu_t = t.cpu()
    assert cpu_t.device.type == 'cpu'
    assert cpu_t.shape == (4,)


def test_cpu_to_applegpu():
    """CPU tensor can be moved to applegpu."""
    _load()
    cpu_t = torch.tensor([1.0, 2.0, 3.0])
    gpu_t = cpu_t.to('applegpu')
    assert gpu_t.device.type == 'applegpu'
    # Copy back and verify data
    back = gpu_t.cpu()
    assert torch.allclose(back, cpu_t)
```

- [ ] **Step 3: Add Makefile targets**

Add to `Makefile`:

```makefile
build-cpp-backend: build-rust
	cd backend_cpp && uv run python setup.py build_ext --inplace

test-cpp-backend: build-cpp-backend
	uv run pytest python/tests/test_cpp_backend.py -v
```

- [ ] **Step 4: Build and test**

Run: `make build-cpp-backend && make test-cpp-backend`
Expected: Tests pass (at minimum `test_empty_tensor`, others may need more ops registered)

- [ ] **Step 5: Commit**

```bash
git add python/applegpu_runtime/cpp_backend.py python/tests/test_cpp_backend.py Makefile
git commit -m "feat: Python entry point and integration tests for C++ backend

load_cpp_backend() loads the .so, registers PrivateUse1 device.
Integration tests verify tensor creation, dtype support, CPU<->GPU copy.
make build-cpp-backend / test-cpp-backend targets added."
```

---

## End of Phases 0-2

After completing these 9 tasks, you will have:

1. **Pre-allocated buffer support** in LazyRuntime (Phase 0)
2. **Deferred-free tracking** for safe tensor lifecycle management (Phase 0)
3. **Rust FFI bridge** with init/alloc/free/ops/eval/readback (Phase 1)
4. **C++ shim** with custom allocator, empty_strided, CPU fallback (Phase 2)
5. **Python entry point** and integration tests (Phase 2)
6. **Build system** with Makefile targets (Phase 2)

The MVP: `torch.empty(3, 3, device='applegpu')` creates a Metal-backed tensor, unregistered ops fall back to CPU, and the foundation is ready for Phase 3 op migration.

## Explicitly Deferred to Phase 3+

These items from the spec are NOT in this plan (Phases 0-2) and will be addressed in the Phase 3+ plan:

- **View/stride handling**: The spec describes passing `(tensor_id, storage_offset, sizes, strides)` per op and a `contiguous()` op. The Phase 2 MVP uses CPU fallback for all view ops. Phase 3 will add native view support as ops are migrated.
- **DeviceGuardImpl**: The spec requires ~50 lines for device/stream management. Phase 2 MVP works without it (single device, no stream switching). Phase 3 will add it when multi-stream support is needed.
- **Tensor quota configuration**: The spec says "remove or make configurable for PrivateUse1." Phase 0's `insert_preallocated` bypasses the scheduler quota. Full quota removal will happen in Phase 3 when training workloads exercise it.
- **Remaining 11 minimum ops** (beyond empty_strided/empty.memory_format): `as_strided`, `view`, `_reshape_alias`, `resize_`, `_copy_from_and_resize`, `set_.source_Tensor`, `set_.source_Storage`, `set_.source_Storage_storage_offset`, `_local_scalar_dense`. These will be added in Phase 3a as needed for MLP training.

**Next plan**: Phases 3a-3c (op migration with MLP/CNN/Transformer checkpoints) + Phase 4 (training validation) + Phase 5 (polish).
