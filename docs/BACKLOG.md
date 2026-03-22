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
- [x] 83 op types (up from 46), dtype-aware serialization
- [x] All Plan 2/3 ops wired through SocketBackend + gpu-service
- [x] Int32 compute fix over containers (was hardcoded Float32)

### Packaging + CI
- [x] Version bump to 0.8.0 (dynamic pyproject.toml versioning)
- [x] `pip install applegpu_runtime` from GitHub Releases
- [x] CI workflow — enabled on push/PR with `macos-14` runners (Rust + Swift + Python 3.10-3.13 build checks)
- [x] Release workflow — triggered on `v*` tags, builds 8 wheels + binaries
- [x] `--version` flag for gpu-service and gpu-container
- [x] Install script (`install.sh`) with SHA256 checksums
- [x] `make ci` / `make release-local` — local CI via act or direct build
- [x] TestPyPI — wheels uploaded, verified (https://test.pypi.org/project/applegpu-runtime/0.9.0/)
- [x] Homebrew tap — `brew install mooreman11/tap/applegpu-runtime`

### N-D Generalization + Missing Ops
- [x] Generalize `add_bias` to N-D — channel-aware kernel with `channel_stride`
- [x] Audit and fix 2D-hardcoded ops — only `argmax` needed fixing, rest already N-D
- [x] `sin`/`cos` — float unary ops
- [x] `log_softmax` — fused with numerical stability
- [x] Fix Conv1d/Conv2d bias CPU fallback in torch_backend — uses `gpu.add_bias()` now
- [x] Verify cross-attention shapes — `q_len != kv_len` works (3D and 4D tested)

### Vsock Relay (partially implemented)
- [x] Package.swift — Containerization dependency, Swift 6.1, macOS 26.0
- [x] Extract shared `UnixSocketHelper`
- [x] ContainerRunner — Tier 1/2 with ContainerManager API
- [x] Run.swift fixes — APPLEGPU_FORCE_TCP, async fix, configurable bridge IP
- [x] VsockRelay.swift deprecated, image normalization, entitlements

### Whisper Speech-to-Text
- [x] Model skeleton + HuggingFace weight loading (168 tensors for tiny)
- [x] Audio encoder — Conv1d + GELU + attention blocks + LayerNorm
- [x] Text decoder — dual KV cache (self-attn grows, cross-attn static)
- [x] Greedy decoding with forced prefix tokens + special token suppression
- [x] Causal mask fix — use `attention_causal` only for prefix step (q_len > 1)
- [x] Encoder positional embedding fix — use HF learned weights, not sinusoidal
- [x] Tokenizer fix — multilingual flag based on model name
- [x] N-D slice/concat fix — generalized 2D-only dispatch to handle 3D+ tensors (lazy.rs)
- [x] End-to-end transcription verified (TTS audio → correct text)
- [x] Regression test suite (10 tests: encoder, decoder, causal mask, slice/concat, e2e)

### GPU Backward Ops + Training Ops (PRs #25, #26)
- [x] 14 backward ops on Metal: threshold, tanh, sigmoid, gelu, gelu_exact, gelu_tanh, softmax, layer_norm, conv2d_input, conv2d_weight, conv1d_input, embedding, batch_norm, max_pool2d
- [x] Forward max_pool2d_with_indices — GPU-side indices output (#16)
- [x] Exact GELU — `approximate="none"` forward + backward (#18)
- [x] Grouped convolution — groups > 1 for conv1d/conv2d (#19)
- [x] Conv2d grad_weight on GPU (#17)
- [x] GPU index/scatter kernels — scatter_write, scatter_add (#21)
- [x] GPU blit_copy — GPU→GPU memory transfer
- [x] Wire protocol expanded to 84 op types (discriminants 0-83)

### PyTorch Aten Ops Expansion (Issues #10-#14)
- [x] sigmoid + sigmoid_backward on Metal (#11)
- [x] var/std with Bessel's correction on Metal
- [x] sin/cos aten dispatch (#15)
- [x] 110+ registered aten ops (up from 40+)
- [x] New aten ops: stack, linspace, normal_, index.Tensor, index_put_, linalg_vector_norm, _unique2, unbind, unsafe_split, var.correction, std.correction
- [x] Docker Compose GPU sidecar pattern (#13)
- [x] Optimizer/scheduler validation — Adam/AdamW/SGD + ReduceLROnPlateau (#14)
- [x] Training lifecycle tests

---

## v0.9.0 — SHIPPED

_C++ PrivateUse1 backend, native ops, view system, in-place ops, CPU fallback diagnostics._

- [x] P0: Dual .so conflict fixed (dynamic Swift library)
- [x] Native mse_loss via graph ops (48x faster)
- [x] Backward ops: threshold_backward, mse_loss_backward
- [x] View ops: view, as_strided, t (zero-copy)
- [x] In-place ops: add_, mul_, fill_, zero_
- [x] ensure_op_ready() view detection
- [x] scalar_mul, div, mean_all FFI functions
- [x] CPU fallback logging (APPLEGPU_LOG_FALLBACK=1)
- [x] C++ backend in make build/test/clean targets
- [x] 15 C++ backend integration tests

---

## Up Next

### PRIORITY 1: MPSGraph Integration for Container GPU Access
_Our key differentiator: PyTorch `device='mps'` doesn't work in Docker containers (no Metal driver inside Linux). Our architecture bridges Metal on the host to containers via IPC. MPSGraph integration brings Apple's fused GPU performance to containers where MPS can't reach._
- [ ] MPSGraph subgraph capture — record op sequences between sync points
- [ ] MPSGraph compilation — convert captured subgraphs to MPSGraph operations
- [ ] MPSGraph execution — run compiled graphs via the Swift bridge (same IPC path containers already use)
- [ ] Benchmark: target parity with `device='mps'` at h≥1024
- [ ] Container verification: Docker + gpu-service → MPSGraph → Metal GPU

### PRIORITY 2: Stable Diffusion / `group_norm`
_One new kernel unlocks a whole model class._
- [ ] `group_norm` kernel — single new Metal kernel
- [ ] Stable Diffusion model wrapper + weight loading
- [ ] End-to-end image generation test

### PRIORITY 3: Models on PrivateUse1
_Prove the C++ backend works beyond MLP._
- [ ] GPT-2 inference on PrivateUse1 — requires view fixes for multi-head attention
- [ ] Whisper on PrivateUse1 — requires conv1d, layer_norm, embedding in eager path
- [ ] Audit remaining CPU fallbacks for each model

### PRIORITY 4: Remaining CPU Fallbacks
_MLP training has zero fallback. These unlock broader model support._
- [ ] `fill`/`zeros`/`ones` GPU kernels — avoid sync for init ops
- [ ] `bernoulli_` / dropout — GPU RNG (Philox counter-based PRNG)
- [ ] `div_.Scalar` — in-place scalar division (easy: scalar_mul with 1/scalar)

### PRIORITY 5: PyPI Publishing
_TestPyPI link exists. Real PyPI when ready for external users._
- [x] TestPyPI — wheels uploaded, verified (https://test.pypi.org/project/applegpu-runtime/0.9.0/)
- [ ] Create PyPI account + API token
- [ ] Publish wheels to real PyPI

### Vsock Socket Relay (blocked)
_Blocked by apple/containerization framework socket staging bug (errno 20 ENOTDIR)._
- [ ] Fix socket staging — try low-level `LinuxContainer(rootfs:vmm:)` API or `dialVsock` manual relay
- [ ] Remove TCP bridge once vsock path is proven

---

## Recently Completed

### Eager Metal Dispatch (D1-D4) — DONE
- [x] D1: EagerRuntime in Rust — stride-aware tensor registry with Arc<Buffer> views, binary/unary/matmul dispatch, make_contiguous, inplace ops, 14 integration tests
- [x] D2: Full C++ backend switchover — all ops use eager FFI, 22 Rust + 16 Python tests pass
- [x] D3: GPU-native threshold_backward + mean_all — no mid-pipeline flush_and_wait, 24 Rust tests
- [x] D4: Strided N-D sum kernel + eliminated hidden flushes. Zero CPU fallback. Forward/loss/step all sub-0.1ms.

### torch.compile Backend — DONE (two implementations)
- [x] Custom FX interpreter: walk graph nodes, map to eager FFI calls via ctypes
- [x] Op mapping: aten.mm → applegpu_eager_matmul, aten.t → applegpu_eager_create_view, etc.
- [x] Rust compiled graph executor: serialized FX graph → single FFI call
- [x] Benchmark result: C++ Dispatcher overhead (7.5µs/op × 30 ops = 225µs) is only ~10% of training time — NOT the bottleneck the original 4.4ms backward was from the Python `__torch_dispatch__` path, not PrivateUse1
- [x] Kernel fusion investigation: no fusible chains in MLP graphs (addmm is not elementwise), fusion engine works but doesn't help this workload

### MPSMatrixMultiplication — DONE
- [x] Replaced custom MSL matmul with Apple's MPSMatrixMultiplication for Float32
- [x] MPS transposed matmul — skip contiguity copies for backward transpose views
- [x] Automatic fallback to custom kernel for tiny buffers (<16 bytes) and non-Float32
- [x] Result: h=4096 training 0.78x → 1.59x CPU (2.2x improvement)

### Bug Fixes
- [x] Scalar tensor dtype bug — `empty_strided` allocated 0-dim scalars as UInt8
- [x] `resize_` dtype — use `applegpu_eager_alloc` with correct dtype
- [x] `torch.applegpu.synchronize()` — was no-op, now calls `flush_and_wait`
- [x] GPU-native `mul_.Scalar` — `scalar_mul` + storage swap instead of flush + CPU loop
- [x] `addmm` null check — prevents tid=0 propagation
- [x] Compiled-graph deferred tensor ID cleanup
- [x] Seed warning — added `manual_seed_all`, `_is_in_bad_fork` to device module

---

## Further Backlog

### Remaining CPU Fallbacks
_MLP training has zero fallback. These are for broader model support._
- [ ] `fill`/`zeros`/`ones` GPU kernels — current impl uses CPU via shared memory, GPU kernel would avoid sync
- [ ] `bernoulli_` / dropout — GPU RNG (Philox counter-based PRNG)
- [ ] `_safe_softmax` — PyTorch 2.10 variant
- [ ] `div_.Scalar` — in-place scalar division (easy: scalar_mul with 1/scalar)
- [ ] Audit: run GPT-2/Whisper on PrivateUse1, list all remaining CPU fallbacks

### GPU Op Gaps
- [ ] GPU linspace kernel — `start + id * step` per thread (cold path, low priority)
- [ ] GPU unique kernel — parallel sort + stream compaction (low priority)
- [ ] `isinf`/`isnan` — float → Bool predicates for numerical debugging
- [ ] `where`/`masked_fill` Bool condition enforcement (migration needed)

### Performance Optimization
_Current state: 1.59x CPU at h=4096, MPS is 3.1x. Gap is ~20 per-op kernel launches vs MPS's ~5 fused dispatches._
- [x] MPSMatrixMultiplication — replaced custom MSL matmul (2.2x improvement at h=4096)
- [x] MPS transposed matmul — skip contiguity copies for backward transpose views
- [x] GPU-native mul_.Scalar — scalar_mul + storage swap instead of flush + CPU loop
- [ ] Encoder caching — keep MTLComputeCommandEncoder open across same-pipeline dispatches
- [ ] Fine-grained locking — split `Mutex<EagerRuntime>` per-component
- [ ] Fused LSTM/GRU kernel — single Metal kernel per timestep for all gates

### Multi-Dtype
- [ ] Reduction output dtype overrides — sum(Int32)→Int32, mean(Int32)→Float32, sum(Bool)→Int32 count
- [ ] Quantized matmul — Int8 weights x Float16 activations with scale factors
- [ ] Backward ops multi-dtype — extend backward kernels to BFloat16
- [ ] Float64 compute kernels — deferred until Apple hardware adds MSL double support

### Models
- [ ] Stable Diffusion — group_norm kernel + model wrapper (PRIORITY 1 above)
- [ ] GPT-2 inference on PrivateUse1 — requires view fixes for multi-head attention (PRIORITY 2 above)
- [ ] Whisper on PrivateUse1 — requires conv1d, layer_norm, embedding in eager path (PRIORITY 2 above)
- [ ] Fine-tuned model export — save trained weights

### Training
- [ ] Int64 compute kernels — batch_norm's num_batches_tracked falls back to CPU
- [ ] Gradient accumulation — for large batch training across multiple micro-batches

### Infrastructure
- [ ] Unix socket relay / vsock — complete Containerization framework integration (macOS 26 SDK)
- [ ] AVF VM integration — VZVirtualMachine lifecycle, virtio-vsock transport
- [ ] Dynamic container lifecycle — work stealing, auto-scaling based on queue pressure
- [ ] Multi-node / distributed graph — network transport layer, graph partitioning
- [ ] Binary signing/notarization — Apple Developer ID

### Concurrency
- [ ] Read timeout / keepalive — `SO_RCVTIMEO` on GPU service connections
- [ ] Connection limits — max concurrent GPU service connections (e.g., 64 max)
- [ ] Health check endpoint — GPU service status query for monitoring
- [ ] Metrics/observability — per-container GPU utilization, queue depth, latency histograms
- [ ] Multi-GPU support _(very low priority)_ — device pool, per-device buffer pools, cross-device transfers

### Memory Pool
- [ ] Size-aware watermark eviction — evict largest pooled buffers first
- [ ] Jemalloc-style size classes — finer-grained bucketing to reduce fragmentation
- [ ] Per-thread pool shards — reduce Mutex contention

### Batch Inference
- [ ] Batch inference pipeline — process multiple sequences simultaneously
- [ ] GPT-2 batched forward — rewrite forward pass using N-D ops
