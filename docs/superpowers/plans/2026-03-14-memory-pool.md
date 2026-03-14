# Persistent Memory Pool Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a power-of-two bucketed buffer pool to reduce Metal GPU allocation churn during repeated evaluations.

**Architecture:** `BufferPool` embeds inside `LazyRuntime` alongside the scheduler, behind the existing Mutex. Intermediate buffers from `execute_node` are acquired from the pool; destroyed tensors return buffers to the pool. Watermark-based eviction caps idle memory. All scheduler interactions use logical (not physical) sizes.

**Tech Stack:** Rust (applegpu-core crate), PyO3 (Python bindings), pytest

**Spec:** `docs/superpowers/specs/2026-03-14-memory-pool-design.md`

---

## Chunk 1: Prerequisites

### Task 1: Fix as_f32_slice to use logical size

**Files:**
- Modify: `crates/core/src/tensor.rs:157-160`
- Modify: `crates/core/src/buffer.rs:68-75` (add doc warning)

- [ ] **Step 1: Write failing test**

Add to `crates/core/src/tensor.rs` test module:

```rust
#[test]
fn as_f32_slice_uses_logical_size() {
    let device = match crate::device::Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };
    // Create a buffer larger than the logical tensor data (simulates pooled buffer)
    let buf = Buffer::new(&device, 128).unwrap(); // 128 bytes = 32 f32s
    // Write known data to first 16 bytes (4 f32s)
    let ptr = buf.contents() as *mut f32;
    unsafe {
        *ptr.add(0) = 1.0;
        *ptr.add(1) = 2.0;
        *ptr.add(2) = 3.0;
        *ptr.add(3) = 4.0;
    }
    // Tensor logically has shape [4] = 4 elements = 16 bytes
    let t = Tensor::from_raw(999, vec![4], buf);
    let slice = t.as_f32_slice();
    assert_eq!(slice.len(), 4); // logical count, not 32
    assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core as_f32_slice_uses_logical`
Expected: FAIL — slice.len() returns 32 instead of 4

- [ ] **Step 3: Fix as_f32_slice**

In `crates/core/src/tensor.rs:157-160`, change:

```rust
/// Read tensor data as f32 slice (zero-copy).
/// Uses logical tensor size, not physical buffer size.
pub fn as_f32_slice(&self) -> &[f32] {
    let count = self.meta.shape.numel();
    unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const f32, count) }
}
```

- [ ] **Step 4: Add doc warning to Buffer::as_slice**

In `crates/core/src/buffer.rs:68-75`, update the doc comment:

```rust
/// Read buffer contents as a slice of T (zero-copy view).
/// WARNING: Returns the FULL physical buffer. With pooled buffers, this may
/// be larger than logical tensor data. For reading tensor data, use
/// Tensor::as_f32_slice() which respects logical size.
/// # Safety
/// Caller must ensure the buffer contains valid data of type T
/// and that the buffer length is a multiple of size_of::<T>().
pub unsafe fn as_slice<T: Copy>(&self) -> &[T] {
```

- [ ] **Step 5: Run all tests**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/tensor.rs crates/core/src/buffer.rs
git commit -m "fix: as_f32_slice uses logical tensor size, not physical buffer size

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 2: Switch all scheduler tracking to logical size

**Files:**
- Modify: `crates/core/src/lazy.rs:37,46,118,291,322`

- [ ] **Step 1: Write test**

Add to `crates/core/src/lazy.rs` test module:

```rust
#[test]
fn insert_tensor_tracks_logical_size() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let id = t.meta.id;
    let logical = t.meta.size_bytes(); // 16 bytes
    rt.insert_tensor(t).unwrap();
    let (bytes, _) = rt.scheduler.container_usage(ContainerId::DEFAULT).unwrap();
    assert_eq!(bytes, logical);
}
```

- [ ] **Step 2: Run test — should pass already** (since `buffer.len()` == logical for `from_f32`)

Run: `cargo test -p applegpu-core insert_tensor_tracks_logical`
Expected: PASS (but we still need to change the code for correctness with pooled buffers)

- [ ] **Step 3: Change insert_tensor to use meta.size_bytes()**

In `crates/core/src/lazy.rs:36-42`:

```rust
pub fn insert_tensor(&mut self, tensor: Tensor) -> Result<()> {
    let size = tensor.meta.size_bytes();
    let id = tensor.meta.id;
    self.scheduler.allocate_tensor(ContainerId::DEFAULT, id, size)?;
    self.tensors.insert(id, tensor);
    Ok(())
}
```

- [ ] **Step 4: Change insert_tensor_for similarly**

In `crates/core/src/lazy.rs:44-51`:

```rust
pub fn insert_tensor_for(&mut self, tensor: Tensor, container_id: ContainerId) -> Result<()> {
    let size = tensor.meta.size_bytes();
    let id = tensor.meta.id;
    self.scheduler.allocate_tensor(container_id, id, size)?;
    self.tensors.insert(id, tensor);
    Ok(())
}
```

- [ ] **Step 5: Change eval() intermediate tracking**

In `crates/core/src/lazy.rs:117-120`, change:

```rust
let result = self.execute_node(device, &node)?;
let size = node.out_shape.numel() * node.out_dtype.size_bytes();
self.scheduler.allocate_tensor(container_id, node_id, size)?;
self.tensors.insert(node_id, result);
```

- [ ] **Step 6: Change eval_remote() tracking**

In `crates/core/src/lazy.rs:289-295`, change:

```rust
let buffer = crate::buffer::Buffer::from_bytes(device, &data)?;
let logical_size = shape.iter().product::<usize>() * 4; // f32 = 4 bytes
let tensor = Tensor::from_raw(tensor_id, shape, buffer);
let container_id = self.resolve_container(id);
self.scheduler.allocate_tensor(container_id, tensor_id, logical_size)?;
self.tensors.insert(tensor_id, tensor);
```

- [ ] **Step 7: Change destroy() tracking**

In `crates/core/src/lazy.rs:321-323`:

```rust
if let Some(tensor) = self.tensors.remove(&id) {
    self.scheduler.free_tensor(id, tensor.meta.size_bytes());
}
```

- [ ] **Step 8: Run all tests**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add crates/core/src/lazy.rs
git commit -m "refactor: use logical tensor size for all scheduler tracking

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 3: Add Tensor::into_buffer

**Files:**
- Modify: `crates/core/src/tensor.rs`

- [ ] **Step 1: Write test**

Add to tensor.rs test module:

```rust
#[test]
fn into_buffer_moves_without_dealloc() {
    let device = match crate::device::Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };
    let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let buf = t.into_buffer();
    // Buffer is still valid (not deallocated)
    assert_eq!(buf.len(), 16);
    let data = unsafe { buf.as_slice::<f32>() };
    assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    // buf drops here — Metal deallocation happens once
}
```

- [ ] **Step 2: Implement into_buffer**

Add to `impl Tensor` block in `crates/core/src/tensor.rs`:

```rust
/// Move the buffer out of this tensor without deallocating.
/// The tensor metadata is dropped but the buffer lives on.
pub fn into_buffer(self) -> Buffer {
    let Tensor { buffer, meta: _ } = self;
    buffer
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-core into_buffer`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/tensor.rs
git commit -m "feat: add Tensor::into_buffer for safe buffer extraction

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 2: BufferPool Implementation

### Task 4: Create pool.rs with BufferPool struct and core API

**Files:**
- Create: `crates/core/src/pool.rs`
- Modify: `crates/core/src/lib.rs` (add `pub mod pool;`)

- [ ] **Step 1: Write tests first**

Create `crates/core/src/pool.rs` with test module:

```rust
use std::collections::HashMap;

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::Result;

/// Statistics for the buffer pool.
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub pooled_bytes: usize,
    pub bucket_count: usize,
}

/// A power-of-two bucketed buffer pool for GPU buffer reuse.
pub struct BufferPool {
    buckets: HashMap<usize, Vec<Buffer>>,
    pooled_bytes: usize,
    max_pooled_bytes: usize,
    hits: u64,
    misses: u64,
}

impl BufferPool {
    pub fn new(max_pooled_bytes: usize) -> Self {
        BufferPool {
            buckets: HashMap::new(),
            pooled_bytes: 0,
            max_pooled_bytes,
            hits: 0,
            misses: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    #[test]
    fn test_acquire_returns_sufficient_size() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 100).unwrap();
        assert!(buf.len() >= 100);
    }

    #[test]
    fn test_acquire_rounds_to_power_of_two() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 100).unwrap();
        assert_eq!(buf.len(), 128); // next power of two
    }

    #[test]
    fn test_release_acquire_reuses_buffer() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 64).unwrap();
        let ptr = buf.contents() as usize; // save pointer for comparison
        pool.release(buf);
        assert_eq!(pool.pooled_bytes(), 64);
        let buf2 = pool.acquire(&device, 64).unwrap();
        assert_eq!(buf2.contents() as usize, ptr); // same buffer reused
    }

    #[test]
    fn test_watermark_eviction() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(100); // very small watermark
        let buf = pool.acquire(&device, 128).unwrap();
        pool.release(buf);
        // 128 > 100 watermark, so buffer should be dropped, not pooled
        assert_eq!(pool.pooled_bytes(), 0);
    }

    #[test]
    fn test_drain_clears_pool() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 64).unwrap();
        pool.release(buf);
        assert!(pool.pooled_bytes() > 0);
        pool.drain();
        assert_eq!(pool.pooled_bytes(), 0);
    }

    #[test]
    fn test_non_power_of_two_dropped_on_release() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        // Create a non-power-of-two buffer directly (simulates from_bytes)
        let buf = Buffer::new(&device, 100).unwrap(); // 100 is not power of two
        pool.release(buf);
        assert_eq!(pool.pooled_bytes(), 0); // not pooled
    }

    #[test]
    fn test_stats_track_hits_misses() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 64).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
        pool.release(buf);
        let _ = pool.acquire(&device, 64).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_prewarm_populates_buckets() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        pool.prewarm(&device, &[64, 128, 256]).unwrap();
        assert!(pool.pooled_bytes() > 0);
        // Acquire should hit
        let _ = pool.acquire(&device, 64).unwrap();
        assert_eq!(pool.stats().hits, 1);
    }

    #[test]
    fn test_set_max_pooled_bytes_evicts() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        pool.prewarm(&device, &[64, 128, 256, 512]).unwrap();
        let before = pool.pooled_bytes();
        assert!(before > 0);
        pool.set_max_pooled_bytes(0);
        assert_eq!(pool.pooled_bytes(), 0);
    }

    #[test]
    fn test_acquire_empty_pool_allocates_fresh() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 256).unwrap();
        assert_eq!(buf.len(), 256);
        assert_eq!(pool.stats().misses, 1);
    }

    #[test]
    fn test_prewarm_skips_zero_size() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        pool.prewarm(&device, &[0, 64]).unwrap();
        // Only 64-byte buffer should be pooled (zero skipped)
        assert_eq!(pool.pooled_bytes(), 64);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core pool::tests`
Expected: FAIL — methods not implemented

- [ ] **Step 3: Implement all BufferPool methods**

Add to `impl BufferPool`:

```rust
/// Acquire a buffer of at least `size` bytes. Rounds up to next power of two.
pub fn acquire(&mut self, device: &Device, size: usize) -> Result<Buffer> {
    let bucketed = size.next_power_of_two();

    if let Some(bucket) = self.buckets.get_mut(&bucketed) {
        if let Some(buf) = bucket.pop() {
            self.pooled_bytes -= bucketed;
            self.hits += 1;
            return Ok(buf);
        }
    }

    self.misses += 1;
    Buffer::new(device, bucketed)
}

/// Return a buffer to the pool. Non-power-of-two buffers are dropped.
/// If over the watermark, the buffer is dropped (eviction).
pub fn release(&mut self, buffer: Buffer) {
    let len = buffer.len();
    if !len.is_power_of_two() {
        return; // drop non-poolable buffer
    }
    if self.pooled_bytes + len > self.max_pooled_bytes {
        return; // watermark eviction — buffer drops here
    }
    self.pooled_bytes += len;
    self.buckets.entry(len).or_insert_with(Vec::new).push(buffer);
}

/// Free all pooled buffers.
pub fn drain(&mut self) {
    self.buckets.clear();
    self.pooled_bytes = 0;
}

/// Set the maximum pooled memory. Evicts if over new limit.
pub fn set_max_pooled_bytes(&mut self, max: usize) {
    self.max_pooled_bytes = max;
    // Evict from largest buckets first
    while self.pooled_bytes > self.max_pooled_bytes {
        let largest = self.buckets.keys().copied().max();
        match largest {
            Some(key) => {
                if let Some(bucket) = self.buckets.get_mut(&key) {
                    bucket.pop(); // drop the buffer
                    self.pooled_bytes = self.pooled_bytes.saturating_sub(key);
                    if bucket.is_empty() {
                        self.buckets.remove(&key);
                    }
                }
            }
            None => break,
        }
    }
}

/// Get pool statistics.
pub fn stats(&self) -> PoolStats {
    PoolStats {
        hits: self.hits,
        misses: self.misses,
        pooled_bytes: self.pooled_bytes,
        bucket_count: self.buckets.len(),
    }
}

/// Total bytes currently pooled (idle).
pub fn pooled_bytes(&self) -> usize {
    self.pooled_bytes
}

/// Pre-warm the pool with buffers of given sizes.
pub fn prewarm(&mut self, device: &Device, sizes: &[usize]) -> Result<()> {
    for &size in sizes {
        if size == 0 {
            continue;
        }
        let buf = Buffer::new(device, size.next_power_of_two())?;
        self.release(buf);
    }
    Ok(())
}
```

- [ ] **Step 4: Add `pub mod pool;` to lib.rs**

In `crates/core/src/lib.rs`, add after `pub mod ops;`:

```rust
pub mod pool;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p applegpu-core pool`
Expected: All pass

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/pool.rs crates/core/src/lib.rs
git commit -m "feat: add BufferPool with power-of-two bucketing and watermark eviction

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 3: LazyRuntime Integration

### Task 5: Embed BufferPool in LazyRuntime

**Files:**
- Modify: `crates/core/src/lazy.rs`

- [ ] **Step 1: Add pool field to LazyRuntime**

In `crates/core/src/lazy.rs`, add import:

```rust
use crate::pool::BufferPool;
```

Add field to struct:

```rust
pub struct LazyRuntime {
    tensors: HashMap<u64, Tensor>,
    graph: Graph,
    pub scheduler: Scheduler,
    pub pool: BufferPool,
}
```

Update `new()`:

```rust
pub fn new() -> Self {
    LazyRuntime {
        tensors: HashMap::new(),
        graph: Graph::new(),
        scheduler: Scheduler::new(ResourceLimits::from_env()),
        pool: BufferPool::new(256 * 1024 * 1024), // 256MB default watermark
    }
}
```

- [ ] **Step 2: Change destroy() to return buffer to pool**

Replace `crates/core/src/lazy.rs` destroy method:

```rust
pub fn destroy(&mut self, id: u64) -> Result<()> {
    for node in self.graph.iter_nodes() {
        if node.inputs.contains(&id) {
            return Err(GpuError::GraphError(format!(
                "Cannot destroy tensor {} while pending op {} depends on it",
                id, node.id
            )));
        }
    }
    if let Some(tensor) = self.tensors.remove(&id) {
        self.scheduler.free_tensor(id, tensor.meta.size_bytes());
        self.pool.release(tensor.into_buffer());
    }
    self.graph.remove_node(id);
    Ok(())
}
```

- [ ] **Step 3: Change remove_tensor_raw to return buffer to pool**

```rust
pub fn remove_tensor_raw(&mut self, id: u64) {
    if let Some(tensor) = self.tensors.remove(&id) {
        self.pool.release(tensor.into_buffer());
    }
    self.graph.remove_node(id);
}
```

- [ ] **Step 4: Change set_limits to drain pool**

```rust
pub fn set_limits(&mut self, limits: ResourceLimits) {
    self.scheduler.update_global_limits(limits);
    self.pool.drain();
}
```

- [ ] **Step 5: Change execute_node to use pool**

Replace every `Tensor::empty_f32(device, node.out_shape.dims().to_vec())?` in `execute_node` with pool-based acquisition. The method signature must change to `&mut self` (currently `&self`) because `pool.acquire` needs `&mut self`.

Change method signature:

```rust
fn execute_node(&mut self, device: &Device, node: &OpNode) -> Result<Tensor> {
```

Then replace each `Tensor::empty_f32(device, node.out_shape.dims().to_vec())?` (there are ~8 sites) with:

```rust
let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
let out_buf = self.pool.acquire(device, out_size)?;
let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), out_buf);
```

**Important:** Since `execute_node` changes to `&mut self`, and it calls `self.get_tensor()` which also borrows `self`, you need to restructure the borrows. The simplest approach: extract input tensor refs before calling pool.acquire. For each op type, get the input data (buffer handles, dimensions) before creating the output tensor.

Alternatively, extract pool into a local variable:
- This won't work since pool is part of self.

The cleanest solution: split `execute_node` to first gather inputs, then acquire output buffer. Since the kernel dispatch functions take buffer references (not tensor references), gather the raw buffer handles first:

```rust
fn execute_node(&mut self, device: &Device, node: &OpNode) -> Result<Tensor> {
    let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();

    // Acquire output buffer from pool
    let out_buf = self.pool.acquire(device, out_size)?;
    let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), out_buf);

    // Now dispatch the kernel using input tensors
    // (self.tensors is borrowed immutably for inputs, out is a local)
    // ... kernel dispatch logic unchanged, just uses &out.buffer ...
```

Actually, `self.get_tensor()` borrows `self.tensors` immutably, and `self.pool.acquire()` borrows `self.pool` mutably. These are disjoint fields, but Rust's borrow checker sees them as borrows of `self`. The fix: acquire the output buffer BEFORE reading input tensors, since `pool.acquire` doesn't touch `self.tensors`.

The restructured pattern for each op type:

```rust
// 1. Acquire output buffer (needs &mut self.pool)
let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
let out_buf = self.pool.acquire(device, out_size)?;
let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), out_buf);

// 2. Get input tensors (needs &self.tensors — no conflict with pool)
let input = self.get_tensor(node.inputs[0])?;

// 3. Dispatch kernel
REGISTRY.dispatch_unary(device, node.op.kernel_name(), &input.buffer, &out.buffer, input.numel())?;

Ok(out)
```

This ordering works because after `pool.acquire()` returns, the mutable borrow on `self.pool` is released, and `self.get_tensor()` only borrows `self.tensors`.

Apply this pattern to all 8 output-creation sites in `execute_node`.

- [ ] **Step 6: Run all tests**

Run: `cargo test -p applegpu-core`
Expected: All pass — existing tests work because pool is transparent

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/lazy.rs
git commit -m "feat: integrate BufferPool into LazyRuntime for buffer reuse

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 6: Integration tests

**Files:**
- Create: `crates/core/tests/pool_integration.rs`

- [ ] **Step 1: Write integration tests**

```rust
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::scheduler::ContainerId;
use applegpu_core::tensor::Tensor;

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn test_eval_reuses_pooled_buffers() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    // First eval: creates tensors, pool starts empty (all misses)
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = applegpu_core::ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();
    assert_eq!(rt.read_f32(c_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);

    let stats1 = rt.pool.stats();
    assert!(stats1.misses > 0); // first eval had misses

    // Destroy result, buffer returns to pool
    rt.destroy(c_id).unwrap();
    assert!(rt.pool.pooled_bytes() > 0);

    // Second eval with same shape: should get pool hits
    let c2_id = applegpu_core::ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c2_id).unwrap();
    assert_eq!(rt.read_f32(c2_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);

    let stats2 = rt.pool.stats();
    assert!(stats2.hits > stats1.hits); // reused buffer from pool
}

#[test]
fn test_destroy_returns_buffer_to_pool() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = applegpu_core::ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();

    let before = rt.pool.pooled_bytes();
    rt.destroy(c_id).unwrap();
    assert!(rt.pool.pooled_bytes() > before);
}

#[test]
fn test_pool_respects_scheduler_limits() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    // Scheduler tracks logical size
    let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let id = t.meta.id;
    rt.insert_tensor(t).unwrap();

    let (bytes, count) = rt.scheduler.global_usage();
    assert_eq!(bytes, 16); // logical: 4 * 4 bytes
    assert_eq!(count, 1);

    rt.destroy(id).unwrap();
    assert_eq!(rt.scheduler.global_usage(), (0, 0));
}

#[test]
fn test_backward_compat_existing_eval() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = applegpu_core::ops::matmul(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();
    assert_eq!(rt.read_f32(c_id).unwrap(), &[19.0, 22.0, 43.0, 50.0]);
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test -p applegpu-core --test pool_integration`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add crates/core/tests/pool_integration.rs
git commit -m "test: add buffer pool integration tests

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 4: Python Bindings and Final Tests

### Task 7: Python bindings for pool

**Files:**
- Modify: `crates/python/src/lib.rs`

- [ ] **Step 1: Add pool Python functions**

Add before the `#[pymodule]` block:

```rust
#[pyfunction]
fn pool_stats() -> PyResult<HashMap<String, usize>> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    let stats = rt.pool.stats();
    let mut map = HashMap::new();
    map.insert("hits".to_string(), stats.hits as usize);
    map.insert("misses".to_string(), stats.misses as usize);
    map.insert("pooled_bytes".to_string(), stats.pooled_bytes);
    map.insert("bucket_count".to_string(), stats.bucket_count);
    Ok(map)
}

#[pyfunction]
fn pool_drain() -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.pool.drain();
    Ok(())
}

#[pyfunction]
fn set_pool_watermark(mb: usize) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.pool.set_max_pooled_bytes(mb * 1024 * 1024);
    Ok(())
}
```

- [ ] **Step 2: Register in pymodule**

Add to the `#[pymodule]` function:

```rust
m.add_function(wrap_pyfunction!(pool_stats, m)?)?;
m.add_function(wrap_pyfunction!(pool_drain, m)?)?;
m.add_function(wrap_pyfunction!(set_pool_watermark, m)?)?;
```

- [ ] **Step 3: Update `python/applegpu_runtime/__init__.py`**

Add `pool_stats`, `pool_drain`, and `set_pool_watermark` to the imports and `__all__` list, following the existing pattern.

- [ ] **Step 4: Build extension**

Run: `cd /Users/noahmoore/applegpu_runtime && uv run maturin develop`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add crates/python/src/lib.rs
git commit -m "feat: add pool_stats, pool_drain, set_pool_watermark Python bindings

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 8: Python tests

**Files:**
- Create: `python/tests/test_pool.py`

- [ ] **Step 1: Write Python tests**

```python
import applegpu_runtime as gpu


def test_pool_stats():
    gpu.init_backend()
    stats = gpu.pool_stats()
    assert "hits" in stats
    assert "misses" in stats
    assert "pooled_bytes" in stats
    assert "bucket_count" in stats


def test_pool_drain():
    gpu.init_backend()
    # Create and destroy a tensor to populate pool
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], [4])
    c = a + b
    c.to_list()  # materialize
    gpu.destroy(c)
    gpu.pool_drain()
    stats = gpu.pool_stats()
    assert stats["pooled_bytes"] == 0


def test_set_pool_watermark():
    gpu.init_backend()
    gpu.set_pool_watermark(1)  # 1MB
    stats = gpu.pool_stats()
    # Pool should work with reduced watermark
    a = gpu.tensor([1.0] * 100, [100])
    b = gpu.tensor([2.0] * 100, [100])
    c = a + b
    result = c.to_list()
    assert len(result) == 100
    assert result[0] == 3.0


def test_pool_transparent_to_existing_api():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], [4])
    c = a + b
    assert c.to_list() == [11.0, 22.0, 33.0, 44.0]
    d = (a * b).relu()
    result = d.to_list()
    assert len(result) == 4
    assert all(v >= 0 for v in result)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest python/tests/test_pool.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_pool.py
git commit -m "test: add Python pool tests

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 9: Full test suite and README update

- [ ] **Step 1: Run all Rust tests**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 2: Run all Python tests**

Run: `cd /Users/noahmoore/applegpu_runtime && uv run maturin develop && uv run pytest -v`
Expected: All pass

- [ ] **Step 3: Update README**

Add pool info to the Status section and update test count.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README for persistent memory pool

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Update project status memory**
