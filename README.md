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

Active development. Current capabilities:

- **Two backends** — MLX-native (direct Metal) and VM (IPC to GPU service process)
- **VM backend** — graph serialization over Unix sockets to a standalone `gpu-service` binary
- **Kernel fusion** — auto-detects element-wise chains, generates fused MSL kernels at runtime
- **GpuTensor class** with Python operators (`+`, `-`, `*`, `/`, `@`, unary `-`) and auto-cleanup
- **Lazy execution** — ops build a DAG, computation deferred until materialization
- 10 GPU operations: add, sub, mul, div, neg, relu, exp, log, sqrt, matmul
- 112 tests passing across all layers (55 Rust + 11 Swift + 46 Python)

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
