# applegpu-runtime

A unified Metal GPU runtime library for Apple Silicon. One API, two backends.

## Architecture

```
Python API (PyO3)  ←  import applegpu_runtime as gpu
        ↓
   Rust Core       ←  ops, tensor management, kernel registry
        ↓
Swift Compat Layer ←  Metal, AVF (Apple Virtualization Framework)
        ↓
   Metal GPU
```

### Backends

- **MLX-native** (default) — direct Metal GPU execution, maximum performance
- **AVF VM** — VM-isolated execution with snapshots, DDP, and multi-node simulation

## Quick Start

```bash
# Prerequisites: Rust, Swift (Xcode), uv
make setup

# Run all tests
make test
```

```python
import applegpu_runtime as gpu

gpu.init_backend()

# Create tensors (data lives on GPU via shared memory)
a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])

# Arithmetic — all ops dispatch Metal compute kernels on the GPU
gpu.to_list(gpu.add(a, b))     # [6.0, 8.0, 10.0, 12.0]
gpu.to_list(gpu.sub(a, b))     # [-4.0, -4.0, -4.0, -4.0]
gpu.to_list(gpu.mul(a, b))     # [5.0, 12.0, 21.0, 32.0]
gpu.to_list(gpu.div(a, b))     # [0.2, 0.333..., 0.428..., 0.5]

# Unary ops
gpu.to_list(gpu.neg(a))        # [-1.0, -2.0, -3.0, -4.0]
gpu.to_list(gpu.relu(a))       # [1.0, 2.0, 3.0, 4.0]
gpu.to_list(gpu.exp(a))        # [2.718..., 7.389..., ...]
gpu.to_list(gpu.log(a))        # [0.0, 0.693..., ...]
gpu.to_list(gpu.sqrt(a))       # [1.0, 1.414..., ...]

# Matrix multiply
gpu.to_list(gpu.matmul(a, b))  # [19.0, 22.0, 43.0, 50.0]
```

## Development

This project uses **TDD** across all three layers:

```bash
make test-rust     # cargo test -p applegpu-core
make test-swift    # cd swift && swift test
make test-python   # uv run pytest -v
```

## Status

Active development. Current capabilities:

- Three-layer FFI fully wired (Python → Rust → Swift → Metal GPU)
- Metal buffer management with zero-copy shared memory (`storageModeShared`)
- KernelRegistry with Arc-based pipeline caching (lock-free GPU dispatch)
- 10 GPU operations: add, sub, mul, div, neg, relu, exp, log, sqrt, matmul
- Matrix multiply with 2D Metal compute dispatch and shape validation
- 66 tests passing across all layers (32 Rust + 11 Swift + 23 Python)
