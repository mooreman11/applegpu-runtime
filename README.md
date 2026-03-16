# applegpu-runtime

A unified Metal GPU runtime library for Apple Silicon. One API, two backends.

**License:** [Apache-2.0](LICENSE.md)

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

## Install

```bash
# Prerequisites: Rust, Swift (Xcode), uv
make setup
uv run maturin develop
```

## PyTorch Device Backend

Run standard PyTorch models on Metal GPU — no model code changes needed.

```python
import torch
import applegpu_runtime as gpu

gpu.enable_torch_backend()

# Move any nn.Module to Metal GPU
model = torch.nn.Sequential(
    torch.nn.Linear(64, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)
model = gpu.to_applegpu(model)

# Forward pass runs on Metal GPU
from applegpu_runtime.torch_backend import ApplegpuTensor
x = ApplegpuTensor.from_torch(torch.randn(32, 64))
output = model(x)              # matmul + bias + relu on Metal
result = output.to_torch_cpu() # back to CPU when needed
```

**Validated models:** GPT-2 (small/medium/large), ResNet-18, BERT

## GPT-2 Text Generation

```python
import applegpu_runtime as gpu

output = gpu.run_model("gpt2", "The meaning of life is", max_tokens=50,
                       temperature=0.8, top_k=50, top_p=0.9)
print(output)
```

## Low-Level API

```python
import applegpu_runtime as gpu
import numpy as np

gpu.init_backend()

# Tensors — any shape up to 8 dimensions
a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])

# Lazy operators build a computation graph
c = a + b              # no GPU work yet
d = a @ b              # still just graph nodes
e = (c * a).relu()     # chain freely

# Computation happens on materialization
e.to_list()            # evaluates the entire chain on the GPU

# N-D tensors with broadcasting
x = gpu.from_numpy(np.random.randn(2, 3, 4).astype(np.float32))
bias = gpu.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
y = x + bias           # [2, 3, 4] + [4] broadcasts automatically

# Multi-dtype
t = gpu.tensor([1, 2, 3], shape=[3], dtype="int32")
h = gpu.tensor([1.0, 2.0], shape=[2], dtype="float16")

# Transformer ops
q = gpu.from_numpy(np.random.randn(4, 8, 64).astype(np.float32))
k = gpu.from_numpy(np.random.randn(4, 8, 64).astype(np.float32))
v = gpu.from_numpy(np.random.randn(4, 8, 64).astype(np.float32))
out = gpu.attention_causal(q, k, v)  # batched causal attention
```

## Capabilities

### 38 GPU Operations (f32 + f16, all N-D)

| Category | Ops |
|----------|-----|
| Element-wise | add, sub, mul, div, neg, relu, gelu, exp, log, sqrt, abs, sign, pow, clamp, tanh |
| Reduction | softmax, softmax_causal, argmax, sum, mean |
| Matrix | matmul (batched), layer_norm, embedding, attention, attention_causal |
| Shape | reshape, slice, concat, add_bias, transpose, transpose_dims |
| Conditional | where, masked_fill, triu, tril |
| Indexing | gather, index_select |
| CNN | conv1d, conv2d, batch_norm, max_pool2d, avg_pool2d |

All ops support N-D tensors (up to 8 dimensions) with NumPy-style broadcasting and kernel fusion.

### Infrastructure

- **PyTorch device backend** — `ApplegpuTensor` with `__torch_dispatch__`, 40+ aten ops routed to Metal, CPU fallback with warnings
- **Training support** — autograd, SGD/Adam/AdamW, gradient clipping, GPT-2 fine-tuning, ResNet training
- **Fast tensor transfer** — `from_torch` 385x faster via direct `data_ptr()` (0.55ms vs 212ms for 1M elements)
- **Multi-container scheduler** — priority-based fair queuing with per-container resource quotas
- **Persistent memory pool** — power-of-two bucketed buffer reuse with watermark eviction
- **Command buffer batching** — non-blocking GPU dispatch, single wait per eval
- **N-D tensors** — stride-based MSL kernels, NumPy-style broadcasting
- **Multi-dtype** — 10 types (float16/32/64, int8/16/32/64, uint8/32, bool) with NumPy/PyTorch roundtrip
- **Kernel fusion** — auto-detect element-wise chains, generate fused MSL at runtime
- **Two backends** — MetalBackend (direct Metal, macOS) and SocketBackend (wire protocol, Linux containers)
- **Container GPU access** — `gpu-container run` CLI, auto-start gpu-service, TCP bridge, multi-client with fair scheduling
- **Wire protocol** — 46 op types, dtype-aware serialization, ReadTensor support, FusedElementwise rejection (security)

## Container GPU Access

Run GPU workloads inside OCI Linux containers on Apple Silicon — Metal GPU is transparently available.

```bash
# Run any container with GPU access (auto-starts gpu-service)
gpu-container run python:3.11-slim -- python -c "
import applegpu_runtime as gpu
gpu.init_backend()
a = gpu.tensor([1.0, 2.0, 3.0])
b = gpu.tensor([4.0, 5.0, 6.0])
print((a + b).to_list())  # runs on host Metal GPU
"

# Options
gpu-container run pytorch:latest --cpus 8 --memory 8192 -- python train.py
```

```
Container (Linux)              Host (macOS)
┌──────────────────┐           ┌──────────────────┐
│ Python code      │           │  gpu-container   │
│  └─ applegpu     │──TCP───▶  │  ├─ TCP bridge   │
│     (socket      │  :7654    │  ├─ gpu-service  │
│      backend)    │           │  ├─ Scheduler    │
└──────────────────┘           │  ├─ BufferPool   │
                               │  └─ Metal GPU    │
                               └──────────────────┘
```

The `gpu-container` CLI (Swift, `swift/GPUContainer/`) wraps Apple's `container` tool:
1. Auto-starts `gpu-service` if not running (readiness probe, PID file)
2. Starts a TCP bridge forwarding port 7654 → `~/.applegpu/runtime.sock`
3. Launches the container with `APPLEGPU_HOST` and `APPLEGPU_PORT` env vars
4. Container-side Python detects Linux → connects via socket backend

Each connection gets a `ContainerId` with isolated resource quotas. The scheduler ensures fair GPU sharing across containers.

**Next:** Replace TCP bridge with Unix socket relay via Apple Containerization framework (`UnixSocketConfiguration`) or direct vsock for lower latency. See [BACKLOG](docs/BACKLOG.md).

### Performance

| Model | Tokens/sec | Notes |
|-------|-----------|-------|
| GPT-2 small (124M) | ~11 tok/s | KV cache + batched multi-head attention |
| GPT-2 medium (345M) | ~3 tok/s | 24 layers, 1024 dims |
| GPT-2 large (774M) | ~0.7 tok/s | 36 layers, 1280 dims |

### Training

- **PyTorch autograd** — backward ops flow natively through `__torch_dispatch__`
- **Optimizers** — SGD, Adam, AdamW all work with loss decrease
- **Gradient clipping** — `torch.nn.utils.clip_grad_norm_` supported
- **Validated training** — MLP, transformer (GELU+LN), ResNet-18, GPT-2 fine-tuning

### Test Coverage

~600 tests across all layers (283 Rust + 13 Swift + ~300 Python)

## Examples

See [`examples/`](examples/) for standalone demo scripts:
- `gpt2_generate.py` — text generation with sampling
- `resnet_inference.py` — CNN classification with benchmarking
- `bert_inference.py` — transformer encoder inference
- `pytorch_device_backend.py` — MLP, broadcasting, multi-dtype

## Development

TDD across all three layers:

```bash
make test-rust     # cargo test -p applegpu-core
make test-swift    # cd swift && swift test
make test-python   # uv run pytest -v
```
