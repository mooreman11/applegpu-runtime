# Backlog

Tracks what needs to be built, ordered by priority. Derived from the original spec.

## Completed

- [x] **Scaffold** — Three-layer architecture (Rust + Swift + Python), TDD, build system
- [x] **FFI Wiring** — Rust ↔ Swift linked via build.rs, safe wrappers, Device RAII
- [x] **Backend Selection** — Backend enum, init_backend(), APPLEGPU_BACKEND env var
- [x] **Tensor Types** — DType, Shape, TensorMeta, TensorLocation
- [x] **Metal Buffers** — MTLBuffer create/read/write/destroy, zero-copy shared memory
- [x] **Compute Pipeline** — Command queue, MSL kernel compilation, element-wise dispatch
- [x] **First GPU Operation** — `gpu.add(a, b)` end-to-end from Python through Metal
- [x] **Buffer-backed Tensors** — Tensor with from_f32, to_list, shape
- [x] **Element-wise ops** — sub, mul, div, neg, relu, exp, log, sqrt via KernelRegistry
- [x] **Matrix multiply (matmul)** — 2D Metal kernel with shape validation
- [x] **Ops module** — High-level tensor ops API with global kernel caching
- [x] **Computation graph** — OpNode DAG with topological sort and cycle detection
- [x] **Lazy execution** — ops build graph nodes, deferred until `eval()` or `to_list()`
- [x] **LazyRuntime** — unified storage for materialized tensors and pending graph
- [x] **Tensor cleanup** — `gpu.destroy()` with dependency validation
- [x] **Explicit eval** — `gpu.eval()` for manual materialization
- [x] **GpuTensor class** — PyO3 class with operators (+,-,*,/,@), methods, auto-cleanup via Drop

## In Progress

_(nothing currently)_

## Up Next

### Phase 4b: Graph Optimizations
- [ ] **Kernel fusion** — combine compatible ops into single Metal kernels (e.g. matmul+add+relu)
- [ ] **Persistent memory pools** — reduce GPU allocation overhead via buffer reuse

### Phase 3b: Additional Operations
- [ ] **Attention** — fused attention kernel (Q, K, V → output)

### Phase 5b: Resource Limits
- [ ] **Resource limits** — max tensor size, max GPU memory, per-container rate limits

### Phase 6: Multi-Container Scheduler
- [ ] **Priority queues** — per-container/VM scheduling
- [ ] **Dynamic batching** — batch ops across containers for better GPU utilization
- [ ] **Fairness enforcement** — memory limits and fair execution

### Phase 7: AVF VM Backend
- [ ] **IPC layer** — shared-memory communication for VM backend
- [ ] **Graph serialization** — serialize op graphs for IPC transport to host GPU service
- [ ] **VM isolation** — snapshots, DDP, multi-node simulation

### Phase 8: Framework Adapters (Optional)
- [ ] **NumPy adapter** — map NumPy ops to gpu ops
- [ ] **PyTorch adapter** — custom device backend
- [ ] **Transformers adapter** — drop-in acceleration for HuggingFace models
