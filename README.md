# applegpu-runtime

A unified Metal GPU runtime library for Apple Silicon. One API, two backends.

## Architecture

```
Python API (PyO3)  ←  import applegpu_runtime as gpu
        ↓
   Rust Core       ←  scheduler, graph engine, tensor management
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

# Element-wise add — dispatches a Metal compute kernel on the GPU
c = gpu.add(a, b)
gpu.to_list(c)   # [6.0, 8.0, 10.0, 12.0]
gpu.shape(c)     # [2, 2]
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
- Metal buffer management (create, read, write, destroy) with zero-copy shared memory
- GPU compute pipeline with Metal Shading Language kernel compilation
- Element-wise add (`gpu.add`) running on Metal GPU hardware
- 40 tests passing across all layers
