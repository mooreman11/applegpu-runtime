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

# Ops are lazy — they build a computation graph, no GPU work yet
c = gpu.add(a, b)
d = gpu.matmul(c, b)
e = gpu.relu(d)

# Computation happens on materialization
gpu.to_list(e)     # evaluates the entire chain on the GPU
gpu.shape(e)       # [2, 2] — works even before eval

# Explicit evaluation and cleanup
gpu.eval(c)        # materialize a specific tensor
gpu.destroy(a)     # free GPU memory
```

### All Operations

```python
# Binary: add, sub, mul, div, matmul
# Unary: neg, relu, exp, log, sqrt
# Lifecycle: tensor, eval, to_list, shape, destroy
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

- **Lazy execution** — ops build a DAG, computation deferred until materialization
- Topological sort with cycle detection for graph evaluation
- `gpu.eval()` for explicit materialization, `gpu.to_list()` auto-evaluates
- `gpu.destroy()` for memory cleanup with dependency validation
- 10 GPU operations: add, sub, mul, div, neg, relu, exp, log, sqrt, matmul
- KernelRegistry with Arc-based pipeline caching (lock-free GPU dispatch)
- Metal buffer management with zero-copy shared memory (`storageModeShared`)
- 79 tests passing across all layers (39 Rust + 11 Swift + 29 Python)
