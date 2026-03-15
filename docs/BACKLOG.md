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

### ~~4. Transformers adapter → v0.2.0~~ DONE
- [x] **Weight loader** — load HuggingFace GPT-2 weights, convert to GpuTensors
- [x] **GPT-2 forward pass** — multi-head causal attention + FFN with all 25 ops
- [x] **Text generation** — tokenize → forward → argmax → decode loop
- [x] **`gpu.run_model("gpt2", "Hello world")`** — end-to-end API
- [x] **Foundation ops** — reshape, slice, concat, add_bias, softmax_causal, attention_causal, argmax
- [x] **Tag v0.2.0** — shipped

### Post-v0.2.0 inference optimizations
- [x] **KV cache** — reuse past key/value computations for autoregressive generation
- [x] **N-way concat** — concat_all convenience function for multi-head concat
- [x] **Top-k / top-p sampling** — temperature, top-k, and nucleus (top-p) sampling
- [x] **General transpose** — transpose_dims(dim0, dim1) for arbitrary dimension swaps
- [x] **Batched GPT-2 attention** — reshape + transpose for [n_head, seq, d_head] batched ops
- [x] **Tag v0.3.0** — shipped with N-D tensors, batched ops, sampling

## Up Next

### PRIORITY 1: N-Dimensional Tensor Support
_Almost everything on the roadmap is blocked by 2D-only tensors. This is the architectural foundation that unlocks batch inference, efficient multi-head attention (1 dispatch instead of 36), proper broadcasting, conv1d, and compatibility with standard ML tensor layouts._

**Phase 1: Core N-D infrastructure** _(DONE)_
- [x] **N-D Shape and strides** — fixed-size Shape (up to 8 dims), TensorLayout with contiguous strides, broadcast_strides_for
- [x] **N-D element-wise ops** — add/sub/mul/div/neg/relu/gelu/exp/log/sqrt on arbitrary-dimensional tensors with stride-based MSL kernels
- [x] **N-D broadcasting** — NumPy-style shape broadcasting for element-wise ops
- [x] **N-D reshape** — reshape to any compatible shape
- [x] **Python bindings verified** — 12 N-D tests (3D/4D creation, broadcasting, relu/gelu on 3D, ndim validation for 2D-only ops, backward compat), GPT-2 unchanged

**Phase 2: Batched transformer ops** _(DONE)_
- [x] **Batched attention** — `[batch, heads, seq, d_head]` in one kernel dispatch (replaces 12× slice+attention+concat)
- [x] **Batched layer_norm** — normalize over last dim for any number of leading dims
- [x] **Batched embedding** — `[batch, seq]` indices
- [x] **Batched softmax** — softmax over last dim for any shape
- [x] **Batched matmul** — `[..., M, K] @ [..., K, N]` with batch broadcasting
- [x] **Batched transpose** — swaps last 2 dims for any ndim
- [x] **Batched softmax_causal** — causal masking with batch dims
- [x] **20 Python tests** in `test_batched_ops.py` covering all batched ops with numeric verification

**Phase 3: Batch inference**
- [ ] **Batch inference pipeline** — process multiple sequences simultaneously
- [ ] **GPT-2 batched forward** — rewrite forward pass using N-D ops
- [ ] **Tag v0.3.0** — ship with N-D tensors and batch inference

### Other improvements (unblocked after N-D tensors)
- [x] **Top-k / top-p sampling** — done
- [x] **PyTorch custom device backend** — done (ApplegpuTensor, __torch_dispatch__, 25+ aten ops)
- [ ] **Conv1d** — needs 3D `[batch, channels, length]`
- [ ] **Gather/scatter** — index-based tensor operations
- [ ] **More models** — GPT-2 medium/large, other architectures

## v0.4.0 — SHIPPED
_38 GPU ops, PyTorch device backend with nn.Module migration, CNN support._

### All v0.4.0 GPU ops: DONE
- [x] sum, mean, where, masked_fill, conv1d, conv2d, batch_norm, max_pool2d, avg_pool2d, pow, abs, sign, clamp, gather, index_select, triu, tril
- [x] PyTorch device backend with `gpu.to_applegpu(model)` for nn.Module migration
- [x] 40+ aten ops dispatched to Metal via __torch_dispatch__

## v0.5.0 — SHIPPED
_Model validation: GPT-2 (small/medium/large), ResNet-18, BERT all run on Metal._

### Validated models:
- [x] **GPT-2 small** — 0.09s/token with KV cache + batched attention
- [x] **GPT-2 medium** — 0.31s/token (345M params, 24 layers)
- [x] **GPT-2 large** — 1.38s/token (774M params, 36 layers)
- [x] **ResNet-18** — CNN inference, output matches CPU within tolerance
- [x] **BERT** — encoder inference, output matches CPU
- [x] **Examples directory** — 4 demo scripts with CLI args

## v0.6.0 — SHIPPED
_Training support: PyTorch autograd works natively on Metal GPU for MLP training._

### What's verified:
- [x] **PyTorch autograd integration** — backward ops flow through __torch_dispatch__ natively
- [x] **Backward ops for element-wise** — grad_add, grad_mul, grad_relu, grad_gelu, grad_tanh verified via MLP training
- [x] **Backward ops for matmul** — grad_matmul verified (dA = dOut @ B^T, dB = A^T @ dOut)
- [x] **SGD optimizer** — torch.optim.SGD parameter updates verified
- [x] **MLP training** — loss decreases over multiple training steps
- [x] **Eager evaluation mode** — enable_training() preserves tensors for backward pass

### Now on Metal (no CPU fallback):
- [x] **Backward ops for softmax** — native Metal kernel (grad_input = output * (grad_output - dot))
- [x] **Backward ops for layer_norm** — native Metal kernel for grad_input, CPU for grad_weight/grad_beta
- [x] **Backward ops for conv2d** — transposed convolution kernel for grad_input on Metal, CPU for grad_weight
- [x] **Backward ops for embedding** — atomic scatter-add for grad_weight on Metal
- [x] **Backward ops for batch_norm** — inference-mode grad_input on Metal, CPU for grad_weight/bias
- [x] **Backward for max_pool2d** — forward now returns real indices (CPU), backward routes gradients correctly. ResNet training unblocked.
- [x] **Adam/AdamW optimizer** — in-place ops (mul_, addcmul_, addcdiv_, lerp_) fixed, loss decreases
- [ ] **Int64 compute kernels** — batch_norm's num_batches_tracked falls back to CPU
- [ ] **Gradient accumulation** — for large batch training across multiple micro-batches

## Up Next

### v0.7.0: Production training + performance
_Scale training to real models. Verify all backward ops. Optimize dispatch overhead._

**Training at scale:**
- [x] **Softmax backward on Metal** — native kernel, zero CPU fallback
- [x] **Layer norm backward on Metal** — native kernel for grad_input, CPU for grad_weight/beta
- [x] **Conv2d backward on Metal** — transposed convolution grad_input, CPU grad_weight
- [x] **Batch norm backward on Metal** — inference-mode grad_input
- [x] **Embedding backward on Metal** — atomic scatter-add
- [x] **Transformer training** — 3-layer GELU+LayerNorm model trains on Metal (loss decreases over 10 steps)
- [x] **GPT-2 fine-tuning** — tiny GPT-2 trains on Metal (loss 4.39 → 3.79 over 5 steps, CE loss via CPU fallback)
- [x] **ResNet training** — ResNet-18 trains on Metal GPU (loss 1.11 → 0.73 over 3 steps)
- [x] **Adam/AdamW optimizer** — in-place ops (mul_, addcmul_, addcdiv_, lerp_) fixed, loss decreases
- [x] **Gradient clipping** — torch.nn.utils.clip_grad_norm_ works (linalg_vector_norm via CPU fallback)

**Performance:**
- [ ] **torch.compile() support** — graph-level fusion, eliminate Python dispatch overhead
- [x] **Direct data_ptr() transfer** — from_torch 385x faster (212ms → 0.55ms), from_numpy 683x faster
- [ ] **Native model.to("applegpu")** — proper PrivateUse1 storage backend

**More models:**
- [ ] **Whisper** — audio model with conv1d
- [ ] **Stable Diffusion** — requires group_norm (new kernel)
- [ ] **Fine-tuned model export** — save trained weights

## Further Backlog

### Framework Improvements
- [ ] **Zero-copy from_numpy** — Metal `makeBuffer(bytesNoCopy:)` Swift FFI, page alignment, GC pinning, three-layer work
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
