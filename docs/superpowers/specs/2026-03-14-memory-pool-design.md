# Persistent Memory Pool (Buffer Reuse)

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Power-of-two bucketed buffer pool with watermark eviction, lazy warmup with optional prewarm API

## Overview

Add a `BufferPool` to applegpu_runtime that reuses GPU buffers across tensor allocations, eliminating Metal allocation churn during repeated evaluations (training loops, iterative inference). The pool uses power-of-two size bucketing, watermark-based eviction, and integrates with the existing scheduler resource tracking.

## Architecture

The pool embeds inside `LazyRuntime` alongside the scheduler, protected by the existing `Mutex<LazyRuntime>`. No new synchronization primitives.

```
LazyRuntime
  ├── tensors: HashMap<u64, Tensor>
  ├── graph: Graph
  ├── scheduler: Scheduler
  └── pool: BufferPool
        ├── buckets: HashMap<usize, Vec<Buffer>>  // key = power-of-two size
        ├── pooled_bytes: usize                    // total idle bytes in pool
        ├── max_pooled_bytes: usize                // watermark (default 256MB)
        ├── hits: u64
        └── misses: u64
```

## BufferPool Struct

```rust
pub struct BufferPool {
    buckets: HashMap<usize, Vec<Buffer>>,
    pooled_bytes: usize,
    max_pooled_bytes: usize,
    hits: u64,
    misses: u64,
}
```

## Public API

```rust
impl BufferPool {
    pub fn new(max_pooled_bytes: usize) -> Self;
    pub fn acquire(&mut self, device: &Device, size: usize) -> Result<Buffer>;
    pub fn release(&mut self, buffer: Buffer);
    pub fn drain(&mut self);
    pub fn set_max_pooled_bytes(&mut self, max: usize);
    pub fn stats(&self) -> PoolStats;
    pub fn pooled_bytes(&self) -> usize;
    pub fn prewarm(&mut self, device: &Device, sizes: &[usize]) -> Result<()>;
}

pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub pooled_bytes: usize,
    pub bucket_count: usize,
}
```

### Acquire

`pool.acquire(device, size)`:
1. Compute `bucketed = size.next_power_of_two()`
2. Check `buckets[bucketed]` — if non-empty, pop (LIFO for cache warmth). Increment `hits`. Decrement `pooled_bytes` by `bucketed`.
3. If empty, allocate fresh via `Buffer::new(device, bucketed)`. Increment `misses`.
4. Return the buffer.

The returned buffer has `buffer.len() == bucketed` (the power-of-two rounded size), which may be larger than the requested `size`.

### Release

`pool.release(buffer)`:
1. If `!buffer.len().is_power_of_two()` — drop the buffer (non-poolable; user-created via `from_bytes` with exact size).
2. If `pooled_bytes + buffer.len() > max_pooled_bytes` — drop the buffer (watermark eviction).
3. Else push to `buckets[buffer.len()]`, increment `pooled_bytes`.

**Watermark eviction limitation (v1):** The incoming buffer is dropped when over the watermark, even if it is smaller than existing pooled buffers. A future v2 could evict the largest pooled buffer first and then try to pool the incoming one. This is acceptable for v1 because the most frequently used sizes cycle through acquire/release fast enough to stay pooled naturally.

### Drain

`drain()`: Clear all buckets, drop all pooled buffers (Metal deallocation), reset `pooled_bytes` to 0.

### set_max_pooled_bytes

`set_max_pooled_bytes(max)`: Update the watermark. If the new max is lower than current `pooled_bytes`, immediately evict buffers until `pooled_bytes <= max`. Eviction order: collect bucket keys, sort descending, pop buffers from largest bucket first. This deterministic ordering ensures large (expensive) buffers are evicted first when shrinking, and the implementation is straightforward since bucket count is small (~15-20 for realistic size ranges).

### Prewarm

`prewarm(device, sizes)`: For each size, allocate a buffer via `Buffer::new(device, size.next_power_of_two())` and immediately `release()` to the pool. If any allocation fails, previously allocated buffers remain in the pool (partial prewarm). Sizes of 0 are skipped.

## Prerequisite: Fix Buffer Read Overread

**Critical prerequisite before enabling pooling.** With pooled buffers, `buffer.len()` is the power-of-two rounded physical size, which is larger than the logical tensor data. Any read path using `buffer.len()` for sizing will read past the logical end into uninitialized memory.

### Fix 1: Tensor::as_f32_slice

Currently delegates to `buffer.as_slice::<f32>()` which uses `buffer.len()`. Change to use the tensor's logical element count:

```rust
pub fn as_f32_slice(&self) -> &[f32] {
    let count = self.meta.shape.numel();
    unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const f32, count) }
}
```

### Fix 2: Buffer::as_slice safety

`Buffer::as_slice<T>()` uses `self.len / size_of::<T>()` which returns the physical element count. With pooled buffers, this is a footgun. Add a doc comment warning that `as_slice` returns the full physical buffer and should not be used for reading logical tensor data:

```rust
/// Returns a slice over the FULL physical buffer (may be larger than logical data).
/// For reading tensor data, use `Tensor::as_f32_slice()` instead which respects logical size.
pub unsafe fn as_slice<T>(&self) -> &[T] { ... }
```

### Fix 3: Buffer::read_bytes

`Buffer::read_bytes()` copies `self.len` bytes (physical size). Add a doc comment noting this returns the full physical buffer. Callers that need logical data should use tensor metadata to determine the correct byte count.

These must be implemented and tested before any pool integration.

## Prerequisite: Consistent Logical Size Tracking

**All scheduler interactions must use `tensor.meta.size_bytes()` (logical size), not `tensor.buffer.len()` (physical size).** This is required because pooled buffers have `buffer.len()` equal to the power-of-two rounded size, which differs from the logical size.

Affected call sites:
- `LazyRuntime::insert_tensor()` — change `tensor.buffer.len()` to `tensor.meta.size_bytes()`
- `LazyRuntime::insert_tensor_for()` — same change (container-attributed tensor path)
- `LazyRuntime::destroy()` — change `tensor.buffer.len()` to `tensor.meta.size_bytes()`
- `LazyRuntime::eval()` — compute logical size from `node.out_shape` and `node.out_dtype` before calling `scheduler.allocate_tensor()`

Today these are equivalent because no rounding occurs, so this change is backward-compatible.

## Tensor Ownership Changes

### into_buffer

New method on `Tensor` using safe destructuring (no `ManuallyDrop`, no unsafe):

```rust
impl Tensor {
    pub fn into_buffer(self) -> Buffer {
        let Tensor { buffer, meta: _ } = self;
        buffer
    }
}
```

Rust's move semantics ensure `Buffer::Drop` does not fire — the buffer is moved out, and `TensorMeta` (plain data, no custom Drop) is dropped normally.

### Pool Scope (v1)

**Pooled paths:**
- `execute_node()` output buffers → acquired from pool via `pool.acquire()`
- `destroy()` → buffer extracted via `into_buffer()` → returned to pool via `pool.release()`
- `remove_tensor_raw()` (deregister cleanup) → buffer extracted → returned to pool

**Non-pooled paths:**
- `Tensor::from_f32()` / `Tensor::from_bytes()` — user-created tensors bypass the pool on creation (exact-size allocation via `Buffer::from_bytes`)
- `eval_remote()` result buffers — created via `Buffer::from_bytes` with exact size, bypass pool on creation
- On destroy, non-power-of-two buffers are dropped by `release()` (step 1 check)
- `GpuTensor::Drop` via `try_lock` failure — buffer dropped without pooling (pre-existing behavior)

## Integration with LazyRuntime

### Modified LazyRuntime

```rust
pub struct LazyRuntime {
    tensors: HashMap<u64, Tensor>,
    graph: Graph,
    pub scheduler: Scheduler,
    pool: BufferPool,
}

impl LazyRuntime {
    pub fn new() -> Self {
        LazyRuntime {
            tensors: HashMap::new(),
            graph: Graph::new(),
            scheduler: Scheduler::new(ResourceLimits::from_env()),
            pool: BufferPool::new(256 * 1024 * 1024), // 256MB default
        }
    }
}
```

### Modified insert_tensor

```rust
pub fn insert_tensor(&mut self, tensor: Tensor) -> Result<()> {
    let size = tensor.meta.size_bytes();  // logical size, not buffer.len()
    let id = tensor.meta.id;
    self.scheduler.allocate_tensor(ContainerId::DEFAULT, id, size)?;
    self.tensors.insert(id, tensor);
    Ok(())
}
```

### Modified eval (execute_node output)

```rust
// In the eval() loop, after execute_node:
let node = self.graph.remove_node(node_id).ok_or_else(|| { ... })?;

let logical_size = node.out_shape.numel() * node.out_dtype.size_bytes();
let result = self.execute_node_pooled(device, &node)?;
self.scheduler.allocate_tensor(container_id, node_id, logical_size)?;
self.tensors.insert(node_id, result);
```

**Refactoring `execute_node` for pool integration:**

`execute_node_pooled` **replaces** `execute_node` entirely — they do not coexist. The refactoring separates buffer acquisition (pool) from kernel dispatch (compute).

Current `execute_node` creates output tensors via `Tensor::empty_f32(device, shape)` at ~8 call sites. Each of these changes to:

```rust
// Before (current):
let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;

// After (pooled):
let logical_size = node.out_shape.numel() * node.out_dtype.size_bytes();
let out_buf = self.pool.acquire(device, logical_size)?;
let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), out_buf);
```

**Tensor ID change:** Currently `Tensor::empty_f32` auto-generates an ID via the global `TENSOR_ID_COUNTER`. The pooled version uses `Tensor::from_raw(node.id, ...)` which sets the tensor ID to the graph node's ID. This is intentional and correct — the node ID was already used as the key in `self.tensors.insert(node_id, result)`, so the tensor ID and node ID should match. This eliminates a redundant ID indirection.

The kernel dispatch logic (which ops to call, how to compute thread groups) remains identical — only the output buffer source changes from `Tensor::empty_f32` to `pool.acquire` + `Tensor::from_raw`.

### Modified destroy

```rust
pub fn destroy(&mut self, id: u64) -> Result<()> {
    // existing dependency check...
    if let Some(tensor) = self.tensors.remove(&id) {
        let size = tensor.meta.size_bytes();  // logical size
        self.scheduler.free_tensor(id, size);
        self.pool.release(tensor.into_buffer());  // return to pool
    }
    self.graph.remove_node(id);
    Ok(())
}
```

### Modified remove_tensor_raw

```rust
pub fn remove_tensor_raw(&mut self, id: u64) {
    if let Some(tensor) = self.tensors.remove(&id) {
        self.pool.release(tensor.into_buffer());  // return to pool
    }
    self.graph.remove_node(id);
}
```

### Modified set_limits

```rust
pub fn set_limits(&mut self, limits: ResourceLimits) {
    self.scheduler.update_global_limits(limits);
    self.pool.drain();  // free cached buffers when limits change
}
```

### Modified eval_remote

The `eval_remote()` result buffer is created via `Buffer::from_bytes()` with exact size. On allocation tracking, use logical size:

```rust
// In eval_remote, after receiving response:
let buffer = crate::buffer::Buffer::from_bytes(device, &data)?;
let logical_size = shape.iter().product::<usize>() * 4; // f32 = 4 bytes
let tensor = Tensor::from_raw(tensor_id, shape, buffer);
let container_id = self.resolve_container(id);
self.scheduler.allocate_tensor(container_id, tensor_id, logical_size)?;
self.tensors.insert(tensor_id, tensor);
```

## Python API

### New Module Functions

```python
gpu.pool_stats() -> dict          # {"hits": int, "misses": int, "pooled_bytes": int, "bucket_count": int}
gpu.pool_drain()                   # free all pooled buffers
gpu.set_pool_watermark(mb: int)    # adjust max pooled memory in MB
```

`prewarm` is Rust-only for v1. It requires a `&Device` which is only available after `init_backend()`. Exposing it to Python can be done in a follow-up if needed.

### Backward Compatibility

All existing functions work unchanged. The pool is transparent — buffers are acquired/released automatically during `eval()` and `destroy()`. Users who never call pool functions see improved performance with no API changes.

### Memory Accounting Note

`gpu.memory_usage()` and `gpu.global_usage()` report **active tensor memory** tracked by the scheduler. Pooled (idle) buffers are GPU memory that has been freed from the scheduler's perspective but not returned to Metal. They are invisible to scheduler accounting. Use `gpu.pool_stats()` to see idle pooled memory. Total actual GPU memory = `memory_usage() + pool_stats()["pooled_bytes"]`.

## Error Handling

- `acquire()` returns `Result<Buffer>` — propagates Metal allocation failures (OOM)
- `release()` never fails — worst case the buffer is dropped (Metal dealloc)
- `prewarm()` returns `Result<()>` — propagates allocation failures, previously allocated buffers remain pooled (partial prewarm)
- Pool errors do NOT affect scheduler accounting — if acquire fails, the error propagates before `scheduler.allocate_tensor()` is called

## Testing Strategy (TDD)

### Rust Unit Tests (in pool.rs, ~10 tests)

1. `test_acquire_returns_sufficient_size` — acquired buffer is >= requested size
2. `test_acquire_rounds_to_power_of_two` — 100-byte request gets 128-byte buffer
3. `test_release_acquire_reuses_buffer` — release then acquire same size = hit (same buffer pointer)
4. `test_watermark_eviction` — release when over watermark drops buffer, pooled_bytes stays at limit
5. `test_drain_clears_pool` — drain frees everything, pooled_bytes = 0
6. `test_non_power_of_two_dropped_on_release` — odd-size buffer is dropped, not pooled
7. `test_prewarm_populates_buckets` — prewarm then acquire = hit
8. `test_stats_track_hits_misses` — verify hit/miss counters
9. `test_set_max_pooled_bytes_evicts` — lowering watermark immediately evicts
10. `test_acquire_empty_pool_allocates_fresh` — miss path works
11. `test_prewarm_skips_zero_size` — zero-size entries ignored

### Prerequisite Tests

12. `test_as_f32_slice_uses_logical_size` — tensor with oversized buffer returns correct slice length (verify trailing padding is not visible in returned slice)
13. `test_destroy_uses_logical_size` — scheduler freed with logical size, not physical
14. `test_insert_tensor_for_uses_logical_size` — container-attributed tensor uses logical size for scheduler

### Rust Integration Tests (scheduler_pool_integration.rs, ~5 tests)

1. `test_eval_reuses_pooled_buffers` — two evals of same-shape graph, second eval has pool hits
2. `test_destroy_returns_buffer_to_pool` — destroy tensor, pool stats show buffer pooled
3. `test_pool_respects_scheduler_limits` — scheduler tracks logical size while pool manages physical
4. `test_backward_compat_existing_tests_pass` — all existing tests pass unchanged
5. `test_deregister_returns_buffers_to_pool` — deregistered container's tensors go to pool

### Python Tests (~4 tests)

1. `test_pool_stats` — returns valid dict with expected keys
2. `test_pool_drain` — after drain, pooled_bytes = 0
3. `test_set_pool_watermark` — adjusts limit
4. `test_pool_transparent_to_existing_api` — existing eval/to_list works with pool active

## Extensibility Notes

- **Size-class swap:** The bucketing function `size.next_power_of_two()` can be replaced with jemalloc-style size classes (powers of two + midpoints) if profiling shows excessive fragmentation. The pool API stays the same.
- **Per-container pools:** The current design uses a single global pool. Phase C could add per-container pools for isolation, using the existing `ContainerId` infrastructure.
- **Multi-GPU:** If multiple devices are supported in the future, each device would need its own pool (Metal buffers are device-specific).
