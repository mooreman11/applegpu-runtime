# applegpu-runtime

A unified Metal GPU runtime library for Apple Silicon. One API, two backends.

## Architecture

```
Python API (PyO3)  ←  import applegpu_runtime as gpu
        ↓
   Rust Core       ←  lazy graph, ops, tensor management, kernel registry
        ↓
Swift Compat Layer ←  Metal, AVF (Apple Virtualization Framework)
        ↓
   Metal GPU
```

### Backends

- **MLX-native** (default) — direct Metal GPU execution, maximum performance
- **AVF VM** — VM-isolated execution with snapshots, DDP, and multi-node simulation

## Install

```bash
# Prerequisites: Rust, Swift (Xcode), uv
make setup

# Run all tests
make test

# Install as editable Python package
uv run maturin develop
```

## GPT-2 Inference

```python
import applegpu_runtime as gpu

# One line: load GPT-2, tokenize, run forward pass, generate text
output = gpu.run_model("gpt2", "The meaning of life is", max_tokens=50)
print(output)
```

## Quick Start

```python
import applegpu_runtime as gpu

gpu.init_backend()

# Create tensors (data lives on GPU via shared memory)
a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])

# Operators are lazy — they build a computation graph
c = a + b              # no GPU work yet
d = a @ b              # still just graph nodes
e = (c * a).relu()     # chain freely

# Computation happens on materialization
e.to_list()            # evaluates the entire chain on the GPU
e.shape                # [2, 2]
repr(e)                # GpuTensor(id=..., shape=[2, 2], materialized)

# All Python operators work
c = a + b              # element-wise add
c = a - b              # element-wise sub
c = a * b              # element-wise mul
c = a / b              # element-wise div
c = a @ b              # matrix multiply
c = -a                 # negation

# Methods for unary ops
c = a.relu()
c = a.exp()
c = a.log()
c = a.sqrt()

# Lifecycle
c.eval()               # explicit materialization
gpu.destroy(c)         # explicit cleanup (or let GC handle it)
```

## Development

This project uses **TDD** across all three layers:

```bash
make test-rust     # cargo test -p applegpu-core
make test-swift    # cd swift && swift test
make test-python   # uv run pytest -v
```

## Status

v0.2.0. Current capabilities:

- **Two backends** — MLX-native (direct Metal) and VM (IPC to GPU service process)
- **VM backend** — graph serialization over Unix sockets to a standalone `gpu-service` binary
- **Kernel fusion** — auto-detects element-wise chains, generates fused MSL kernels at runtime
- **GpuTensor class** with Python operators (`+`, `-`, `*`, `/`, `@`, unary `-`) and auto-cleanup
- **Scaled dot-product attention** — `gpu.attention(Q, K, V)` with proper 1/sqrt(d_k) scaling
- **Multi-container scheduler** — priority-based fair queuing with per-container resource quotas, deficit-based scheduling, starvation prevention, and pause/resume support
- **Persistent memory pool** — power-of-two bucketed buffer reuse with watermark eviction, reducing Metal allocation churn for iterative workloads
- **NumPy interop** — `gpu.from_numpy(arr)` and `tensor.to_numpy()` for seamless data interchange with the Python ecosystem
- **PyTorch interop** — `gpu.from_torch(tensor)` and `tensor.to_torch()` for bidirectional data exchange with PyTorch
- **Multi-dtype support** — 10 data types (float16, float32, float64, int8, int16, int32, int64, uint8, uint32, bool) with full NumPy/PyTorch roundtrip, native half-precision kernels at 2x throughput on Apple Silicon, f32 accumulation for matmul/softmax, dtype-aware fusion
- **Command buffer batching** — non-blocking GPU dispatch with single wait per eval, eliminating per-op GPU stalls in chain evaluations
- **Resource limits** — configurable max tensor size, total GPU memory, and tensor count via `gpu.set_limits()` or env vars, enforced per-container and globally
- **Lazy execution** — ops build a DAG, computation deferred until materialization
- **Transformer ops** — GELU, LayerNorm (with affine), Embedding (Int32), softmax_causal, attention_causal, argmax
- **GPT-2 inference** — `gpu.run_model("gpt2", "Hello", max_tokens=50)` for end-to-end text generation on Metal GPU
- **N-D tensors (up to 8 dimensions)** — stride-based MSL kernels, NumPy-style broadcasting for all element-wise ops, N-D reshape, 3D/4D tensor creation and roundtrip via NumPy
- **Batched N-D ops** — matmul, softmax, softmax_causal, layer_norm, embedding, transpose, attention, and attention_causal all support arbitrary leading batch dimensions with broadcasting. One kernel dispatch handles all batch elements.
- **Tensor manipulation** — reshape, slice, concat, add_bias (broadcast)
- 25 GPU operations (f32 + f16): add, sub, mul, div, neg, relu, gelu, exp, log, sqrt, matmul, softmax, softmax_causal, layer_norm, transpose, scalar_mul, embedding, attention, attention_causal, reshape, slice, concat, add_bias, argmax + kernel fusion
- **General transpose** — `gpu.transpose_dims(t, dim0, dim1)` for arbitrary dimension swaps, enabling batched multi-head attention via reshape + transpose
- 433 tests passing across all layers (228 Rust + 13 Swift + 192 Python)

### NumPy & PyTorch Interop

```python
import numpy as np
import torch
import applegpu_runtime as gpu

gpu.init_backend()

# NumPy → GPU → NumPy
arr = np.random.randn(128, 64).astype(np.float32)
t = gpu.from_numpy(arr)
result = t.to_numpy()  # shape preserved

# PyTorch → GPU → PyTorch
x = torch.randn(128, 64)
t = gpu.from_torch(x)
result = t.to_torch()  # returns torch.Tensor

# Mix: PyTorch data, GPU compute, NumPy output
q = gpu.from_torch(torch.randn(32, 64))
k = gpu.from_torch(torch.randn(32, 64))
v = gpu.from_torch(torch.randn(32, 64))
out = gpu.attention(q, k, v)
result = out.to_numpy()  # [32, 64] attention output
```

### Multi-Container Scheduler

```python
import applegpu_runtime as gpu
gpu.init_backend()

# Register containers with resource quotas and priorities
high = gpu.register_container(priority="high", max_memory_mb=256, max_tensors=500, max_pending=50)
low = gpu.register_container(priority="low", max_memory_mb=128, max_tensors=200, max_pending=20)

# Create lazy tensors and submit jobs to containers
a = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
b = gpu.tensor([4.0, 5.0, 6.0], shape=[3])
c = a + b  # lazy

job_id = gpu.submit_job(high, c)        # submit to high-priority container
gpu.run_next()                          # executes highest-priority job
print(gpu.job_status(job_id))           # "completed"
print(gpu.container_usage(high))        # (bytes_used, tensor_count)

# Pause/resume containers
gpu.pause_container(low)                # paused container's jobs are skipped
gpu.resume_container(low)

# Cleanup
gpu.deregister_container(high)
gpu.deregister_container(low)
```

### VM Backend Usage

```bash
# Terminal 1: Start GPU service
cargo run -p applegpu-service

# Terminal 2: Use VM backend
APPLEGPU_BACKEND=vm python3 -c "
import applegpu_runtime as gpu
gpu.init_backend()
a = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
b = gpu.tensor([10.0, 20.0, 30.0], shape=[3])
print((a + b).to_list())  # [11.0, 22.0, 33.0] — executed on GPU service
"
```
