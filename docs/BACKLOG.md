# Backlog

Tracks what has been built and what remains. Each item appears exactly once.

---

## Completed (v0.1.0 through v0.7.0)

<details>
<summary>Click to expand completed work</summary>

### v0.1.0 — Foundation
- [x] Three-layer architecture (Rust + Swift + Python), TDD, build system
- [x] Rust ↔ Swift FFI via build.rs, safe wrappers, Device RAII
- [x] Backend enum, init_backend(), APPLEGPU_BACKEND env var
- [x] DType, Shape, TensorMeta, TensorLocation
- [x] MTLBuffer create/read/write/destroy, zero-copy shared memory
- [x] Command queue, MSL kernel compilation, element-wise dispatch
- [x] `gpu.add(a, b)` end-to-end from Python through Metal
- [x] Buffer-backed Tensors with from_f32, to_list, shape
- [x] Element-wise ops: sub, mul, div, neg, relu, exp, log, sqrt
- [x] Matrix multiply (matmul), 2D Metal kernel
- [x] Computation graph (OpNode DAG, topological sort, cycle detection)
- [x] Lazy execution (ops build graph nodes, deferred until eval/to_list)
- [x] LazyRuntime, tensor cleanup with dependency validation
- [x] GpuTensor PyO3 class with operators, auto-cleanup via Drop
- [x] Kernel fusion (auto-detect element-wise chains, generate fused MSL)
- [x] Graph serialization (binary wire format)
- [x] IPC layer (Unix socket client), GPU service binary
- [x] VM backend routing (APPLEGPU_BACKEND=vm)
- [x] Softmax, transpose, scalar multiplication, attention
- [x] Resource limits, multi-container scheduler, memory pools
- [x] NumPy adapter (from_numpy/to_numpy), PyTorch adapter (from_torch/to_torch)

### v0.2.0 — GPT-2 Inference
- [x] HuggingFace GPT-2 weight loader
- [x] GPT-2 forward pass (multi-head causal attention + FFN)
- [x] Text generation (tokenize → forward → argmax → decode)
- [x] `gpu.run_model("gpt2", "Hello world")` end-to-end API
- [x] Foundation ops: reshape, slice, concat, add_bias, softmax_causal, attention_causal, argmax

### v0.3.0 — N-D Tensors + Batched Ops
- [x] N-D Shape/strides (up to 8 dims), NumPy-style broadcasting
- [x] N-D element-wise ops with stride-based MSL kernels
- [x] Batched transformer ops (attention, layer_norm, embedding, softmax, matmul, transpose, softmax_causal)
- [x] KV cache, N-way concat, top-k/top-p sampling
- [x] General transpose_dims for arbitrary dimension swaps
- [x] Batched GPT-2 attention

### v0.4.0 — PyTorch Backend + CNN
- [x] 38 GPU ops total
- [x] PyTorch device backend (ApplegpuTensor, __torch_dispatch__, 40+ aten ops)
- [x] `gpu.to_applegpu(model)` for nn.Module migration
- [x] New ops: sum, mean, where, masked_fill, conv1d, conv2d, batch_norm, max_pool2d, avg_pool2d, pow, abs, sign, clamp, gather, index_select, triu, tril

### v0.5.0 — Model Validation
- [x] GPT-2 small (~11 tok/s), medium (~3 tok/s), large (~0.7 tok/s)
- [x] ResNet-18 inference, BERT encoder inference
- [x] Examples directory (4 demo scripts)

### v0.6.0 — Training
- [x] PyTorch autograd integration (backward ops through __torch_dispatch__)
- [x] Backward ops: grad_add, grad_mul, grad_relu, grad_gelu, grad_tanh, grad_matmul
- [x] SGD optimizer, MLP training verified
- [x] Eager evaluation mode for backward pass

### v0.7.0 — Production Training
- [x] Backward ops on Metal: softmax, layer_norm, conv2d, embedding, batch_norm
- [x] Training verified: MLP, transformer, ResNet-18, GPT-2 fine-tuning
- [x] Adam/AdamW/SGD optimizers, gradient clipping
- [x] Direct data_ptr() transfer (385x faster from_torch, 683x faster from_numpy)
- [x] Zero-copy transfers: from_numpy_shared, from_torch_shared, aligned_numpy

### Concurrency (Phases 2a-2c, completed across v0.3.0-v0.7.0)
- [x] Command buffer batching (non-blocking dispatch, single wait per eval)
- [x] Single command buffer encoding (begin_batch/end_batch/abort_batch FFI)
- [x] Concurrent queues (parallel_levels graph analysis, 4-queue pool, MTLEvent sync, linear fast path)

</details>

---

## v0.8.0 — SHIPPED

_Containerization, multi-dtype completion, wire protocol v3, CI/packaging._

### Containerization
- [x] FusedElementwise rejection (security guard in gpu-service)
- [x] DType handling fix (gpu-service maps all 10 wire dtypes, was hardcoded to Float32)
- [x] ReadTensorRequest/Response in wire protocol
- [x] Legacy ipc.rs deprecated, callers use crates/client
- [x] PID file + signal handling (SIGTERM, stale socket detection)
- [x] Backend trait abstraction (~107 methods): MetalBackend (macOS) / SocketBackend (Linux)
- [x] Conditional compilation (single package, platform-specific wheels)
- [x] `gpu-container run` CLI (Swift, wraps Apple's container tool)
- [x] Auto-start gpu-service (readiness probe, PID file, binary discovery)
- [x] TCP bridge (port 7654 → Unix socket)
- [x] Container env vars (APPLEGPU_HOST, APPLEGPU_PORT)
- [x] Transport trait generalization (Box<dyn Transport>, connect_auto)
- [x] ContainerRunner skeleton (macOS 26 guard, UnixSocketConfiguration)
- [x] TCP bridge hardening (double-close fix, bind to 192.168.64.1)
- [x] Docker bind-mount GPU access (one-liner docker run -v)
- [x] End-to-end Docker container test (F32 add, Int32 add, Cast+embedding, comparison ops)

### Multi-Dtype Completion (Plans 2 + 3)
- [x] Template-based kernel dispatch (all ~25 op categories templated in kernel_templates.rs)
- [x] Cast op (gpu.cast for dtype conversion)
- [x] Byte-copy shape ops (slice/concat parameterized by element size)
- [x] is_compute_supported expansion (all dtypes except Float64)
- [x] Op-level dtype validation (validate_op_dtype coverage matrix)
- [x] Int64 Apple9+ gating (Swift FFI + Device::supports_int64)
- [x] Comparison ops: lt, gt, le, ge, eq, ne (Bool output)
- [x] Bitwise ops: and, or, xor, not, shl, shr (integer types)
- [x] Utility ops: mod, elem_min, elem_max
- [x] Logical ops: logical_not
- [x] Quantize/dequantize (int8/uint8 with scale+zero_point)

### Wire Protocol v3
- [x] 65 op types (up from 46), dtype-aware serialization
- [x] All Plan 2/3 ops wired through SocketBackend + gpu-service
- [x] Int32 compute fix over containers (was hardcoded Float32)

### Packaging + CI
- [x] Version bump to 0.8.0 (dynamic pyproject.toml versioning)
- [x] `pip install applegpu_runtime` from GitHub Releases

---

## Up Next

### PRIORITY 1: Packaging Polish
- [x] CI workflow — enabled on push/PR with `macos-14` runners (Rust + Swift + Python 3.10-3.13 build checks)
- [x] Release workflow — triggered on `v*` tags, builds 8 wheels + binaries uploaded via `gh release upload`
- [x] `--version` flag for gpu-service and gpu-container
- [x] Install script (`install.sh`) — downloads binaries from GitHub Releases with SHA256 checksums
- [x] `make ci` / `make release-local` — local CI via act or direct build
- [ ] TestPyPI validation — upload wheels, verify `pip install` works (needs TESTPYPI_TOKEN)
- [ ] PyPI publishing — real PyPI after TestPyPI validation
- [ ] Homebrew tap — `brew install mooreman11/tap/applegpu-runtime`
- [ ] Binary signing/notarization — Apple Developer ID signing for gpu-container/gpu-service

### PRIORITY 2: Replace TCP Bridge with Unix Socket Relay / vsock
_macOS 26 SDK available — ready to implement._
- [ ] vsock relay — VZVirtioSocketListener relay in Swift process
- [ ] Remove TCP bridge — once Containerization framework path is fully working

### PRIORITY 3: Model Expansion + Polish
- [ ] Whisper — audio model with conv1d
- [ ] Stable Diffusion — requires group_norm (new kernel)
- [ ] Fine-tuned model export — save trained weights
- [ ] Native `model.to("applegpu")` — proper PrivateUse1 storage backend

### PRIORITY 4: Performance Optimization
- [ ] `torch.compile()` support — register as compile backend for graph-level fusion
- [ ] Async eval — `gpu.eval_async(tensor)` returns a GpuFuture, non-blocking Python
- [ ] Fine-grained locking — split `Mutex<LazyRuntime>` into per-component locks

---

## Further Backlog

### Multi-Dtype Remaining
- [ ] Reduction output dtype overrides — sum(Int32)→Int32, mean(Int32)→Float32, sum(Bool)→Int32 count
- [ ] Quantized matmul — Int8 weights x Float16 activations with scale factors
- [ ] `isinf`/`isnan` — float → Bool predicates for numerical debugging
- [ ] `fill`/`zeros`/`ones` — compute kernels for all dtypes
- [ ] Fused comparison chains — `(a > 0) & (a < 10)` as single kernel
- [ ] `where`/`masked_fill` Bool condition enforcement (migration needed)
- [ ] `is_elementwise()` expansion — mark new ops as fusable
- [ ] Backward ops multi-dtype — extend backward kernels to BFloat16
- [ ] Float64 compute kernels — deferred until Apple hardware adds MSL double support

### Infrastructure
- [ ] Unix socket relay / vsock — complete Containerization framework integration (macOS 26 SDK)
- [ ] AVF VM integration — VZVirtualMachine lifecycle, virtio-vsock transport
- [ ] Dynamic container lifecycle — work stealing, auto-scaling based on queue pressure
- [ ] Multi-node / distributed graph — network transport layer, graph partitioning

### Concurrency
- [ ] Read timeout / keepalive — `SO_RCVTIMEO` on GPU service connections
- [ ] Connection limits — max concurrent GPU service connections (e.g., 64 max)
- [ ] Health check endpoint — GPU service status query for monitoring
- [ ] Metrics/observability — per-container GPU utilization, queue depth, latency histograms
- [ ] Multi-GPU support _(very low priority)_ — device pool, per-device buffer pools, cross-device transfers

### Training
- [ ] Int64 compute kernels — batch_norm's num_batches_tracked falls back to CPU
- [ ] Gradient accumulation — for large batch training across multiple micro-batches

### Memory Pool Improvements
- [ ] Size-aware watermark eviction — evict largest pooled buffers first
- [ ] Jemalloc-style size classes — finer-grained bucketing to reduce fragmentation

### Batch Inference
- [ ] Batch inference pipeline — process multiple sequences simultaneously
- [ ] GPT-2 batched forward — rewrite forward pass using N-D ops
