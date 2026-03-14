# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**applegpu_runtime** is a unified Metal GPU runtime library for Apple Silicon. It exposes a single Python API (`import applegpu_runtime as gpu`) that abstracts over two backends:

- **MLX-native backend** (default): direct MLX → Metal GPU execution inside ACF containers
- **AVF VM backend**: VM-mediated Metal execution via Apple Virtualization Framework, providing isolation, snapshots, and DDP/multi-node simulation

## Three-Layer Architecture

```
Python API (PyO3)  ←  user-facing: import applegpu_runtime as gpu
        ↓
   Rust Core       ←  scheduler, graph engine, tensor management, IPC
        ↓
Swift Compat Layer ←  Metal, AVF, Apple frameworks via @_cdecl C ABI
        ↓
   Metal GPU
```

- **Rust ↔ Swift bridge**: Swift exports C ABI functions via `@_cdecl`, Rust calls them via `extern "C"` in `crates/core/src/ffi.rs`. The C header is at `swift/Sources/AppleGPUBridge/include/bridge.h`. `build.rs` compiles the Swift static library and links it automatically.
- **Rust ↔ Python bridge**: PyO3 cdylib in `crates/python/`, maturin builds it. The Python package is in `python/applegpu_runtime/`.
- **Important**: The PyO3 cdylib cannot be built via `cargo build --workspace` (missing Python symbols). Use `cargo build -p applegpu-core` for Rust-only builds, and `uv run maturin develop` for the Python extension.

## Build & Test Commands

```bash
# First-time setup (installs deps, builds native extension)
make setup            # or: uv sync

# Run all tests (Rust + Swift + Python)
make test

# Run individual test suites
make test-rust        # cargo test -p applegpu-core
make test-swift       # cd swift && swift test
make test-python      # uv run pytest -v

# Run a single Rust test
cargo test test_name

# Run a single Python test
uv run pytest python/tests/test_file.py::test_name -v

# Quick compile check (no linking)
make check
```

## Key Files

- `crates/core/src/ops.rs` — High-level tensor operations (add, sub, mul, div, neg, relu, exp, log, sqrt, matmul)
- `crates/core/src/compute.rs` — KernelRegistry, ComputePipeline, MSL kernel sources
- `crates/core/src/buffer.rs` — Safe Rust Buffer wrapper around MTLBuffer (zero-copy shared memory)
- `crates/core/src/tensor.rs` — DType, Shape, TensorMeta, Tensor (buffer-backed)
- `crates/core/src/ffi.rs` — Rust-Swift FFI boundary (extern "C" declarations + safe wrappers)
- `crates/core/src/device.rs` — RAII Device wrapper with Drop-based cleanup
- `crates/core/src/backend.rs` — Backend enum, Runtime, init_backend() with OnceCell
- `crates/core/src/error.rs` — GpuError enum and Result alias
- `crates/core/build.rs` — Compiles Swift static lib and links it into Rust
- `swift/Sources/AppleGPUBridge/bridge.swift` — Swift C ABI bridge (@_cdecl exports) + device handle helper
- `swift/Sources/AppleGPUBridge/buffer.swift` — MTLBuffer C ABI (create, read, write, destroy)
- `swift/Sources/AppleGPUBridge/compute.swift` — Metal compute pipeline C ABI (binary, unary, matmul dispatch)
- `swift/Sources/AppleGPUBridge/kernels.swift` — MSL kernel source strings (used by Swift tests)
- `swift/Sources/AppleGPUBridge/include/bridge.h` — shared C header for the FFI contract
- `crates/python/src/lib.rs` — PyO3 module definition (Python-facing API surface)
- `python/applegpu_runtime/__init__.py` — Python package entry point

## Performance Philosophy

This library must be **hyperoptimized**. Every layer is chosen for maximum performance:
- Rust core for zero-cost abstractions, memory safety without GC overhead
- Swift compatibility layer for native Metal/AVF access with no bridging penalty
- Zero-copy tensor transport via shared memory — avoid unnecessary allocations and copies
- Lazy execution with kernel fusion to minimize GPU dispatch overhead
- Persistent memory pools to reduce GPU allocation churn

Always prefer the fastest path. Profile before and after changes to performance-critical code.

## Development Workflow

This project uses **TDD**. Write tests first, then implement.

**After pushing or committing significant changes**, update `README.md` to reflect new capabilities, API changes, or status updates. Keep the README consistent with the actual state of the library.

- **Rust tests**: unit tests inline (`#[cfg(test)]`), integration tests in `crates/core/tests/`
- **Swift tests**: Swift Testing framework in `swift/Tests/AppleGPUBridgeTests/`
- **Python tests**: pytest in `python/tests/`

## Toolchain

- **Rust**: cargo (workspace with two crates: `applegpu-core` and `applegpu-python`)
- **Swift**: SwiftPM (package in `swift/`, produces static library `libAppleGPUBridge.a`)
- **Python**: uv + maturin (pyproject.toml at root, `uv sync` handles everything)
- Backend selection: `gpu.init_backend()` or `APPLEGPU_BACKEND=mlx|vm` env var

## Architecture Layers (inside Rust core)

1. **API Layer** — backend selection, error propagation, tensor metadata
2. **Tensor Layer** — virtual tensors (ID, shape, dtype, location), zero-copy shared memory
3. **Graph Layer** — lazy op capture, kernel fusion (e.g. matmul+add+gelu → single Metal kernel)
4. **Scheduler/Executor** — multi-container/multi-VM batching, priority queues, fairness
5. **IPC Layer** — shared-memory communication for AVF VM backend only
