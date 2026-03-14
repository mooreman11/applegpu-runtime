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

gpu.init_backend()       # auto-selects MLX-native or VM
C = gpu.matmul(A, B)
logits = gpu.attention(Q, K, V)
```

## Development

This project uses **TDD** across all three layers:

```bash
make test-rust     # cargo test --workspace
make test-swift    # cd swift && swift test
make test-python   # uv run pytest -v
```

## Status

Early development — scaffold with passing test suites across all layers. FFI wiring between Rust and Swift is next.
