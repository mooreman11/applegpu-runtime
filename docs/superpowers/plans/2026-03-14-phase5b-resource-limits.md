# Phase 5b: Resource Limits Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable resource limits (max tensor size, max total GPU memory, max tensor count, and per-eval rate limiting) to prevent resource exhaustion and enable fair multi-container scheduling.

**Architecture:** A new `ResourceLimits` struct holds configurable limits (with sensible defaults). It's stored on `LazyRuntime` and checked before every buffer allocation and tensor creation. Violations produce clear `GpuError::ResourceLimitExceeded` errors. Limits are configurable via environment variables (`APPLEGPU_MAX_MEMORY_MB`, `APPLEGPU_MAX_TENSOR_SIZE_MB`, `APPLEGPU_MAX_TENSORS`) and can be set programmatically from Python via `gpu.set_limits()`. A `MemoryTracker` tracks current GPU memory usage across all live tensors.

**Tech Stack:** Rust (enforcement in buffer/tensor creation), Python (PyO3 config API)

---

## File Structure

### New Files
- `crates/core/src/limits.rs` — `ResourceLimits` config, `MemoryTracker`, limit checking logic

### Modified Files
- `crates/core/src/lazy.rs` — Store `ResourceLimits` + `MemoryTracker` on `LazyRuntime`, check before allocation
- `crates/core/src/error.rs` — Add `ResourceLimitExceeded` error variant
- `crates/core/src/lib.rs` — Add limits module
- `crates/python/src/lib.rs` — Add `gpu.set_limits()`, `gpu.memory_usage()`, `gpu.tensor_count()`
- `python/applegpu_runtime/__init__.py` — Export new functions

### New Test Files
- `python/tests/test_limits.py` — Resource limit enforcement tests

---

## Chunk 1: Resource Limits Core

### Task 1: Add ResourceLimitExceeded error and limits module

**Files:**
- Modify: `crates/core/src/error.rs`
- Create: `crates/core/src/limits.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Add error variant**

In `crates/core/src/error.rs`, add to `GpuError`:

```rust
    /// Resource limit exceeded
    ResourceLimitExceeded(String),
```

Update `Display`:

```rust
            GpuError::ResourceLimitExceeded(msg) => write!(f, "Resource limit exceeded: {}", msg),
```

- [ ] **Step 2: Create limits.rs**

```rust
use crate::error::{GpuError, Result};

// Note: MemoryTracker uses plain usize (not AtomicUsize) because LazyRuntime
// is behind a Mutex<LazyRuntime> in the Python layer, which serializes all access.
// Plain usize is simpler and doesn't give a false sense of thread safety.

/// Configurable resource limits for the GPU runtime.
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum size of a single tensor in bytes (0 = unlimited).
    pub max_tensor_size_bytes: usize,
    /// Maximum total GPU memory usage across all tensors in bytes (0 = unlimited).
    pub max_total_memory_bytes: usize,
    /// Maximum number of live tensors (0 = unlimited).
    pub max_tensor_count: usize,
}

impl ResourceLimits {
    /// Default limits: 512MB per tensor, 2GB total, 10000 tensors.
    pub fn default_limits() -> Self {
        ResourceLimits {
            max_tensor_size_bytes: 512 * 1024 * 1024,   // 512 MB
            max_total_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            max_tensor_count: 10_000,
        }
    }

    /// No limits (unlimited).
    pub fn unlimited() -> Self {
        ResourceLimits {
            max_tensor_size_bytes: 0,
            max_total_memory_bytes: 0,
            max_tensor_count: 0,
        }
    }

    /// Load limits from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut limits = Self::default_limits();

        if let Ok(val) = std::env::var("APPLEGPU_MAX_TENSOR_SIZE_MB") {
            if let Ok(mb) = val.parse::<usize>() {
                limits.max_tensor_size_bytes = mb * 1024 * 1024;
            }
        }
        if let Ok(val) = std::env::var("APPLEGPU_MAX_MEMORY_MB") {
            if let Ok(mb) = val.parse::<usize>() {
                limits.max_total_memory_bytes = mb * 1024 * 1024;
            }
        }
        if let Ok(val) = std::env::var("APPLEGPU_MAX_TENSORS") {
            if let Ok(n) = val.parse::<usize>() {
                limits.max_tensor_count = n;
            }
        }

        limits
    }
}

/// Tracks current GPU memory usage.
pub struct MemoryTracker {
    /// Current total bytes allocated across all live tensors.
    current_bytes: usize,
    /// Current number of live tensors.
    current_count: usize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        MemoryTracker {
            current_bytes: 0,
            current_count: 0,
        }
    }

    /// Check if allocating `size_bytes` would exceed limits.
    /// Returns Ok if allowed, Err with a clear message if not.
    pub fn check_allocation(&self, size_bytes: usize, limits: &ResourceLimits) -> Result<()> {
        // Check single tensor size
        if limits.max_tensor_size_bytes > 0 && size_bytes > limits.max_tensor_size_bytes {
            return Err(GpuError::ResourceLimitExceeded(format!(
                "Tensor size {} bytes exceeds limit of {} bytes ({} MB)",
                size_bytes, limits.max_tensor_size_bytes,
                limits.max_tensor_size_bytes / (1024 * 1024)
            )));
        }

        // Check total memory
        if limits.max_total_memory_bytes > 0 {
            let current = self.current_bytes.load(Ordering::Relaxed);
            if current + size_bytes > limits.max_total_memory_bytes {
                return Err(GpuError::ResourceLimitExceeded(format!(
                    "Total GPU memory would exceed limit: current {} + new {} > limit {} bytes ({} MB)",
                    current, size_bytes, limits.max_total_memory_bytes,
                    limits.max_total_memory_bytes / (1024 * 1024)
                )));
            }
        }

        // Check tensor count
        if limits.max_tensor_count > 0 {
            let count = self.current_count.load(Ordering::Relaxed);
            if count >= limits.max_tensor_count {
                return Err(GpuError::ResourceLimitExceeded(format!(
                    "Tensor count {} would exceed limit of {}",
                    count + 1, limits.max_tensor_count
                )));
            }
        }

        Ok(())
    }

    /// Record that `size_bytes` was allocated.
    pub fn track_alloc(&mut self, size_bytes: usize) {
        self.current_bytes += size_bytes;
        self.current_count += 1;
    }

    /// Record that `size_bytes` was freed.
    pub fn track_free(&mut self, size_bytes: usize) {
        self.current_bytes -= size_bytes;
        self.current_count -= 1;
    }

    /// Current total GPU memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.current_bytes
    }

    /// Current number of live tensors.
    pub fn tensor_count(&self) -> usize {
        self.current_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_are_reasonable() {
        let limits = ResourceLimits::default_limits();
        assert_eq!(limits.max_tensor_size_bytes, 512 * 1024 * 1024);
        assert_eq!(limits.max_total_memory_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(limits.max_tensor_count, 10_000);
    }

    #[test]
    fn unlimited_allows_anything() {
        let limits = ResourceLimits::unlimited();
        let tracker = MemoryTracker::new();
        assert!(tracker.check_allocation(usize::MAX / 2, &limits).is_ok());
    }

    #[test]
    fn tensor_size_limit_enforced() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 1024,
            max_total_memory_bytes: 0,
            max_tensor_count: 0,
        };
        let tracker = MemoryTracker::new();
        assert!(tracker.check_allocation(512, &limits).is_ok());
        assert!(tracker.check_allocation(2048, &limits).is_err());
    }

    #[test]
    fn total_memory_limit_enforced() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 0,
            max_total_memory_bytes: 1024,
            max_tensor_count: 0,
        };
        let tracker = MemoryTracker::new();
        tracker.track_alloc(512);
        assert!(tracker.check_allocation(256, &limits).is_ok());
        assert!(tracker.check_allocation(1024, &limits).is_err());
    }

    #[test]
    fn tensor_count_limit_enforced() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 0,
            max_total_memory_bytes: 0,
            max_tensor_count: 2,
        };
        let tracker = MemoryTracker::new();
        assert!(tracker.check_allocation(64, &limits).is_ok());
        tracker.track_alloc(64);
        assert!(tracker.check_allocation(64, &limits).is_ok());
        tracker.track_alloc(64);
        assert!(tracker.check_allocation(64, &limits).is_err());
    }

    #[test]
    fn track_free_decreases_usage() {
        let tracker = MemoryTracker::new();
        tracker.track_alloc(1024);
        tracker.track_alloc(2048);
        assert_eq!(tracker.memory_usage(), 3072);
        assert_eq!(tracker.tensor_count(), 2);
        tracker.track_free(1024);
        assert_eq!(tracker.memory_usage(), 2048);
        assert_eq!(tracker.tensor_count(), 1);
    }
}
```

- [ ] **Step 3: Add limits module to lib.rs**

Add `pub mod limits;` to `crates/core/src/lib.rs`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core limits 2>&1`
Expected: 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/error.rs crates/core/src/limits.rs crates/core/src/lib.rs
git commit -m "feat: add ResourceLimits, MemoryTracker, and ResourceLimitExceeded error"
```

---

### Task 2: Integrate limits into LazyRuntime

**Files:**
- Modify: `crates/core/src/lazy.rs`

- [ ] **Step 1: Add ResourceLimits and MemoryTracker to LazyRuntime**

Update `LazyRuntime` struct and constructor:

```rust
use crate::limits::{MemoryTracker, ResourceLimits};
```

Update the struct:

```rust
pub struct LazyRuntime {
    tensors: HashMap<u64, Tensor>,
    graph: Graph,
    pub limits: ResourceLimits,
    pub tracker: MemoryTracker,
}
```

Update `new()`:

```rust
    pub fn new() -> Self {
        LazyRuntime {
            tensors: HashMap::new(),
            graph: Graph::new(),
            limits: ResourceLimits::from_env(),
            tracker: MemoryTracker::new(),
        }
    }
```

- [ ] **Step 2: Check limits in insert_tensor**

Update `insert_tensor`:

```rust
    pub fn insert_tensor(&mut self, tensor: Tensor) -> Result<()> {
        let size = tensor.buffer.len();
        self.tracker.check_allocation(size, &self.limits)?;
        self.tracker.track_alloc(size);
        self.tensors.insert(tensor.meta.id, tensor);
        Ok(())
    }
```

- [ ] **Step 3: Check limits in execute_node (internal tensor creation)**

Since `MemoryTracker` uses plain `usize`, tracking needs `&mut self`. Move the limit check + tracking to `eval()` rather than inside `execute_node()`. In `eval()`, after `execute_node` returns the result tensor, check and track before inserting:

```rust
    // In eval(), replace:
    //   let result = self.execute_node(device, &node)?;
    //   self.tensors.insert(node_id, result);
    // With:
    let result = self.execute_node(device, &node)?;
    let size = result.buffer.len();
    self.tracker.check_allocation(size, &self.limits)?;
    self.tracker.track_alloc(size);
    self.tensors.insert(node_id, result);
```

`execute_node` stays as `&self` (unchanged). The limit check happens in `eval` which has `&mut self`.

- [ ] **Step 4: Track free in destroy**

In the `destroy` method, track deallocation:

```rust
    pub fn destroy(&mut self, id: u64) -> Result<()> {
        // ... existing dependency check ...
        if let Some(tensor) = self.tensors.remove(&id) {
            self.tracker.track_free(tensor.buffer.len());
        }
        self.graph.remove_node(id);
        Ok(())
    }
```

- [ ] **Step 5: Add memory_usage and tensor_count accessors**

```rust
    pub fn memory_usage(&self) -> usize {
        self.tracker.memory_usage()
    }

    pub fn live_tensor_count(&self) -> usize {
        self.tracker.tensor_count()
    }

    pub fn set_limits(&mut self, limits: ResourceLimits) {
        self.limits = limits;
    }
```

- [ ] **Step 6: Update ALL callers of insert_tensor to handle Result**

`insert_tensor` now returns `Result<()>`. Update every call site:

**Rust test files** (add `.unwrap()`):
- `crates/core/src/lazy.rs` (test module)
- `crates/core/src/ops.rs` (test module)
- `crates/core/tests/fusion_integration.rs`
- `crates/core/tests/serial_integration.rs`

**GPU service** (handle error):
- `crates/gpu-service/src/main.rs` — change `rt.insert_tensor(tensor)` to handle the Result:
  ```rust
  if let Err(e) = rt.insert_tensor(tensor) {
      return EvalResponse::Err(format!("Insert tensor failed: {}", e));
  }
  ```

**eval_remote** (track received tensors):
- In `lazy.rs` `eval_remote` method, update the result insertion to track memory:
  ```rust
  let size = tensor.buffer.len();
  self.tracker.track_alloc(size);
  self.tensors.insert(tensor_id, tensor);
  ```

- [ ] **Step 7: Run all Rust tests**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/lazy.rs
git commit -m "feat: integrate resource limits into LazyRuntime with memory tracking"
```

---

## Chunk 2: Python API and Tests

### Task 3: Update Python bindings for insert_tensor Result change

**Files:**
- Modify: `crates/python/src/lib.rs`

- [ ] **Step 1: Update tensor() to handle insert_tensor Result**

The `tensor()` function now needs to handle the `Result` from `insert_tensor`:

```rust
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<GpuTensor> {
    let runtime = get_device_runtime()?;
    let t = Tensor::from_f32(&runtime.device, shape, &data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}
```

- [ ] **Step 2: Add set_limits, memory_usage, tensor_count functions**

```rust
/// Set resource limits. Pass 0 for any field to make it unlimited.
#[pyfunction]
fn set_limits(max_tensor_size_mb: usize, max_memory_mb: usize, max_tensors: usize) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.set_limits(applegpu_core::limits::ResourceLimits {
        max_tensor_size_bytes: if max_tensor_size_mb > 0 { max_tensor_size_mb * 1024 * 1024 } else { 0 },
        max_total_memory_bytes: if max_memory_mb > 0 { max_memory_mb * 1024 * 1024 } else { 0 },
        max_tensor_count: max_tensors,
    });
    Ok(())
}

/// Get current GPU memory usage in bytes.
#[pyfunction]
fn memory_usage() -> PyResult<usize> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    Ok(rt.memory_usage())
}

/// Get current number of live tensors.
#[pyfunction]
fn tensor_count() -> PyResult<usize> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    Ok(rt.live_tensor_count())
}
```

Register in `#[pymodule]`:

```rust
    m.add_function(wrap_pyfunction!(set_limits, m)?)?;
    m.add_function(wrap_pyfunction!(memory_usage, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_count, m)?)?;
```

- [ ] **Step 3: Update __init__.py**

Add `set_limits`, `memory_usage`, `tensor_count` to imports and `__all__`.

- [ ] **Step 4: Build and verify existing tests pass**

Run: `uv run maturin develop && uv run pytest -v 2>&1`
Expected: All existing tests pass (default limits are generous enough)

- [ ] **Step 5: Commit**

```bash
git add crates/python/src/lib.rs python/applegpu_runtime/__init__.py
git commit -m "feat: expose set_limits, memory_usage, and tensor_count to Python"
```

---

### Task 4: Python resource limit tests

**Files:**
- Create: `python/tests/test_limits.py`

- [ ] **Step 1: Write resource limit tests**

```python
import applegpu_runtime as gpu


def test_memory_usage_tracking():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)  # unlimited
    initial = gpu.memory_usage()
    t = gpu.tensor([1.0] * 1000, shape=[1000])
    after = gpu.memory_usage()
    assert after > initial
    assert after - initial == 1000 * 4  # 1000 floats * 4 bytes


def test_tensor_count_tracking():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)
    count_before = gpu.tensor_count()
    a = gpu.tensor([1.0, 2.0], shape=[2])
    b = gpu.tensor([3.0, 4.0], shape=[2])
    assert gpu.tensor_count() >= count_before + 2


def test_tensor_size_limit():
    gpu.init_backend()
    # Set max tensor size to 1KB
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)
    # Override with bytes-level limit via Rust
    # For Python, we use MB granularity — set to 1MB and try to create a 2MB tensor
    gpu.set_limits(max_tensor_size_mb=1, max_memory_mb=0, max_tensors=0)
    # 1MB = 262144 floats
    try:
        t = gpu.tensor([1.0] * 300000, shape=[300000])  # ~1.14 MB > 1 MB limit
        assert False, "Should have raised"
    except ValueError as e:
        assert "Resource limit exceeded" in str(e)
    # Reset to unlimited
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)


def test_tensor_count_limit():
    gpu.init_backend()
    # Set limit relative to current count to handle process-global state
    current = gpu.tensor_count()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=current + 3)
    created = []
    try:
        for i in range(10):
            created.append(gpu.tensor([float(i)], shape=[1]))
        assert False, "Should have raised"
    except ValueError as e:
        assert "Resource limit exceeded" in str(e)
    # Reset
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)


def test_set_limits_unlimited():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)
    # Should be able to create tensors without limit
    t = gpu.tensor([1.0] * 10000, shape=[10000])
    assert t.shape == [10000]
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest python/tests/test_limits.py -v 2>&1`
Expected: All 5 tests pass

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_limits.py
git commit -m "test: add resource limit enforcement tests"
```

---

### Task 5: End-to-end verification and push

- [ ] **Step 1: Run full test suite**

Run: `make clean && make test 2>&1`
Expected: All tests pass

- [ ] **Step 2: Update backlog and README**

Mark Phase 5b as complete. Add resource limits note to README.

- [ ] **Step 3: Push**

```bash
git push origin main
```
