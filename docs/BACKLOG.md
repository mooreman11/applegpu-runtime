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
- [x] **Kernel fusion** — auto-detect element-wise chains, generate fused MSL kernels at runtime
- [x] **Graph serialization** — binary wire format for EvalRequest/EvalResponse
- [x] **IPC layer** — Unix socket client for remote GPU evaluation
- [x] **GPU service binary** — standalone Metal execution service (`gpu-service`)
- [x] **VM backend routing** — `APPLEGPU_BACKEND=vm` routes eval through IPC
- [x] **Softmax** — numerically stable softmax along last dimension (reduction kernel)
- [x] **Transpose** — 2D matrix transpose kernel
- [x] **ScalarMul** — element-wise scalar multiplication (carries scale in graph node)
- [x] **Attention** — scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`
- [x] **Resource limits** — max tensor size, max GPU memory, max tensor count with MemoryTracker
- [x] **Multi-container scheduler** — priority-based fair queuing, per-container quotas, starvation prevention, pause/resume, job lifecycle
- [x] **Memory pools** — power-of-two bucketed BufferPool with watermark eviction
- [x] **NumPy adapter** — `gpu.from_numpy(arr)` and `tensor.to_numpy()`, copy-based, f32 only

## In Progress

_(nothing currently)_

## Up Next

### Phase 8: Framework Adapters (continued)
- [ ] **PyTorch adapter** — `gpu.from_torch(tensor)`, `tensor.to_torch()`, copy-based
- [ ] **Transformers adapter** — drop-in acceleration for HuggingFace models

### Multi-dtype Support
- [ ] **Multi-dtype adapters** — extend from_numpy/to_numpy to support float16, float64, int8, int32 (adapter layer only, ~1-2 hours)
- [ ] **Multi-dtype compute kernels** — MSL kernel variants for float16/float64/int types (larger lift, touches all kernels + dispatch)

### Post-Ship
- [ ] **Phase 7b: AVF VM integration** — VZVirtualMachine lifecycle, virtio-vsock transport (Metal GPU can't pass through to guest VMs)
- [ ] **Zero-copy from_numpy** — Metal bytesNoCopy FFI, page alignment, GC pinning
- [ ] **Dynamic container lifecycle** — work stealing, multi-node, distributed graph
