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
- [x] **PyTorch adapter** — `gpu.from_torch(tensor)` and `tensor.to_torch()`, routes through NumPy bridge

## v0.1.0 shipped

## Up Next (priority order)

### ~~1. Multi-dtype compute kernels (float16 priority)~~ DONE
- [x] **Multi-dtype dispatch layer** — DType-aware kernel selection in KernelRegistry
- [x] **Float16 MSL kernels** — f16 variants of all 14 ops (preprocessor templates + custom f32-intermediate matmul/softmax/scalar_mul)
- [x] **Multi-dtype adapters** — from_numpy/to_numpy/from_torch/to_torch support float16
- [x] **Dtype inference** — ops infer output dtype from inputs, mixed-dtype errors
- [x] **Dtype-aware fusion** — fused kernels emit `half` types for f16 chains

### 2. Concurrency (phased)
_Eliminate GPU idle time between ops, then enable true parallel execution._

**Phase 2a: Command buffer batching** _(DONE)_
- [x] **Shared device-level command queue** — single MTLCommandQueue per device, not per pipeline
- [x] **Non-blocking dispatch** — remove waitUntilCompleted() from individual ops, wait once at end of eval
- [x] **Spec:** `docs/superpowers/specs/2026-03-14-command-buffer-batching-design.md`

**Phase 2b: Single command buffer** _(future)_
- [ ] **Encode all ops into one MTLCommandBuffer** — reduce CB creation overhead
- [ ] **begin_batch/end_batch FFI** — new Swift/Rust API for batch encoding

**Phase 2c: Concurrent queues** _(future)_
- [ ] **Dependency analysis** — identify independent subgraphs in the topo-sorted order
- [ ] **Concurrent command queue dispatch** — dispatch independent subgraphs to separate Metal queues

**Phase 2d: Async eval + fine-grained locking** _(future)_
- [ ] **Async eval** — `gpu.eval_async(tensor)` returns a future/handle, non-blocking Python
- [ ] **Fine-grained locking** — split `Mutex<LazyRuntime>` into per-component locks (graph, tensor store, scheduler, pool)

### ~~3. New ops for transformer inference~~ PARTIALLY DONE
- [x] **LayerNorm** — f32 + f16 kernels, blocking + non-blocking dispatch
- [x] **Embedding lookup** — f32 + f16 kernels, Int32 index support
- [x] **GELU activation** — f32 + f16 kernels
- [ ] **Gather/scatter** — index-based tensor operations
- [ ] **Conv1d** — used in some transformer variants and audio models

### 4. Transformers adapter → v0.2.0
_Full pipeline: load model, tokenize, run forward pass, generate text. GPT-2 as first target._
- [ ] **Weight loader** — download HuggingFace weights, convert to GpuTensors
- [ ] **GPT-2 forward pass** — Python implementation using our ops (matmul, attention, gelu, layer_norm, embedding)
- [ ] **Text generation** — tokenize → forward → argmax (CPU) → decode loop
- [ ] **`gpu.run_model("gpt2", "Hello world")`** — end-to-end API
- [ ] **Tag v0.2.0** — ship with transformer inference

### Post-v0.2.0 inference optimizations
- [ ] **Argmax GPU kernel** — reduction op, avoids CPU roundtrip during generation
- [ ] **KV cache** — reuse past key/value computations for autoregressive generation (major speedup)
- [ ] **Top-k / top-p sampling** — temperature-based sampling for non-greedy generation
- [ ] **Batch inference** — multiple sequences in parallel

## Further Backlog

### Framework Improvements
- [ ] **Zero-copy from_numpy** — Metal `makeBuffer(bytesNoCopy:)` Swift FFI, page alignment, GC pinning, three-layer work
- [ ] **PyTorch custom device backend** — register `applegpu` via PyTorch PrivateUse1 device API
- [ ] **PyTorch autograd integration** — `torch.autograd.Function` wrappers (requires backward ops)
- [ ] **Direct from_torch via data_ptr()** — bypass NumPy bridge when Metal bytesNoCopy is available

### Multi-dtype Support (partially done)
- [x] **Multi-dtype adapters** — all 10 dtypes (float16/32/64, int8/16/32/64, uint8/32, bool) supported in tensor(), from_numpy, to_numpy, to_list, dtype getter
- [x] **Float16 compute kernels** — all 14 ops have f16 MSL kernels with f32 accumulation
- [x] **Comprehensive multi-dtype tests** — 26 parametrized Python tests covering creation, NumPy roundtrip, to_list type fidelity, compute validation, and backward compat
- [ ] **Float64 compute kernels** — MSL kernel variants for f64 (Apple Silicon emulates f64, slower than f32 — low priority)
- [ ] **Integer compute kernels** — MSL kernel variants for int32/int64 arithmetic (add, mul, etc. — useful for index manipulation)
- [ ] **Bool compute kernels** — logical ops (and, or, not) on bool tensors

### Infrastructure
- [ ] **Phase 7b: AVF VM integration** — VZVirtualMachine lifecycle, virtio-vsock transport (Metal GPU can't pass through to guest VMs)
- [ ] **Phase C: Dynamic container lifecycle** — work stealing, auto-scaling based on queue pressure
- [ ] **Multi-node / distributed graph** — network transport layer, graph partitioning across machines

### Concurrency
- [ ] **Multiple Metal command queues** — dispatch independent graph branches in parallel on the same GPU. Apple Silicon supports concurrent queues natively. Biggest throughput win for workloads with independent subgraphs.
- [ ] **Async eval** — `gpu.eval_async(tensor)` returns a future/handle, doesn't block Python. Submit multiple evals and wait on results. Integrates with the scheduler's job queue.
- [ ] **Fine-grained locking** — split single `Mutex<LazyRuntime>` into per-component locks (graph, tensor store, scheduler, pool). Allows concurrent reads while one eval runs. Requires careful lock ordering to avoid deadlocks.
- [ ] **Multi-GPU support** _(very low priority)_ — device pool, per-device buffer pools, cross-device transfers. Only relevant for Mac Pro with multiple chips or eGPUs. Apple Silicon has one GPU per chip.

### Memory Pool Improvements
- [ ] **Size-aware watermark eviction** — evict largest pooled buffers first (current v1 drops incoming buffer)
- [ ] **Jemalloc-style size classes** — finer-grained bucketing to reduce fragmentation from 50% to ~33%
