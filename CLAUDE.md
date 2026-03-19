# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**applegpu_runtime** is a unified Metal GPU runtime library for Apple Silicon. It exposes a single Python API (`import applegpu_runtime as gpu`) that abstracts over two backends:

- **MLX-native backend** (default): direct MLX → Metal GPU execution inside ACF containers
- **AVF VM backend**: VM-mediated Metal execution via Apple Virtualization Framework, providing isolation, snapshots, and DDP/multi-node simulation

## Two PyTorch Integration Paths

### 1. PyO3 Path (original, `__torch_dispatch__`)
- Entry: `import applegpu_runtime as gpu` → `gpu.init_backend()` → `torch_backend.py`
- ~110 aten ops registered via Python `__torch_dispatch__`
- Works for Whisper, LSTM, GRU, CNN training
- **Bottleneck**: ~20µs per op (Python dispatch overhead), GPU slower than CPU for all model sizes

### 2. PrivateUse1 C++ Path (new, `device='applegpu'`)
- Entry: `from applegpu_runtime.cpp_backend import load_cpp_backend` → `load_cpp_backend()`
- C++ shim (`backend_cpp/applegpu_backend.cpp`) registers ops at PrivateUse1 dispatch key
- Native ops: add, sub, mul, div, mm, relu, neg, addmm, threshold_backward, view, as_strided, t, mse_loss, fill_, zero_, add_, mul_
- CPU fallback for unregistered ops (working — copies via storageModeShared)
- MLP training works end-to-end (15 Python tests passing)
- **Bottleneck**: ~22µs per op (Metal kernel dispatch overhead), GPU still slower than CPU

## Three-Layer Architecture

```
Python API (PyO3 or C++ PrivateUse1)
        ↓
   Rust Core       ←  scheduler, graph engine, tensor management, IPC
        ↓
Swift Compat Layer ←  Metal, AVF, Apple frameworks via @_cdecl C ABI
        ↓
   Metal GPU
```

- **Rust ↔ Swift bridge**: Swift exports C ABI functions via `@_cdecl`, Rust calls them via `extern "C"` in `crates/core/src/ffi.rs`. The C header is at `swift/Sources/AppleGPUBridge/include/bridge.h`. `build.rs` compiles the Swift static library and links it automatically.
- **Rust ↔ Python bridge**: PyO3 cdylib in `crates/python/`, maturin builds it. The Python package is in `python/applegpu_runtime/`.
- **Rust ↔ C++ bridge**: `extern "C"` FFI in `crates/core/src/backend_ffi.rs`, C header in `backend_cpp/applegpu_ffi.h`. C++ shim links against `libapplegpu_core.a` (staticlib).
- **Important**: The PyO3 cdylib cannot be built via `cargo build --workspace` (missing Python symbols). Use `cargo build -p applegpu-core` for Rust-only builds, and `uv run maturin develop` for the Python extension.

## Known Problems (Priority Order)

### P0: Dual `.so` Conflict
Both `applegpu_runtime.cpython-311-darwin.so` (PyO3) and `applegpu_backend.cpython-311-darwin.so` (C++) link `libAppleGPUBridge.a`. Loading both in the same process causes ObjC class duplication (duplicate `GPUDevice`, `GPUBuffer`, `GPUCompute` classes). **Symptoms**: `objc[]: Class _TtC14AppleGPUBridge9GPUDevice is implemented in both` warnings, segfaults, or incorrect behavior. **Workaround**: In test files, stub `sys.modules['applegpu_runtime']` before importing `cpp_backend` to prevent the PyO3 `__init__.py` from loading. **Fix needed**: Either (a) make the C++ backend a shared library that dynamically links the Swift bridge instead of statically linking it, or (b) merge both backends into a single `.so`.

### P1: Per-Op Dispatch Overhead
Each op dispatches a separate Metal kernel (~100µs launch overhead). At batch_size=32, hidden=256 (MLP), Metal kernel compute is ~1µs but launch overhead is ~100µs. GPU is 20x slower than CPU. **The graph engine already has fusion** (`crates/core/src/fusion.rs`) that can merge matmul+add+gelu into single kernels, but the C++ path doesn't accumulate subgraphs — `ensure_op_ready()` and `contiguous()` force eval mid-graph. **Fix needed**: Defer eval until explicit sync points, avoid mid-graph materialization.

### P2: View Tensor Identity
PyTorch view ops (`t()`, `reshape`, `slice`, `as_strided`) create tensors sharing storage with different shapes/strides. Our Rust runtime tracks tensor_ids per storage, not per view. `get_tensor_id()` returns the base storage's tensor_id, which may have a different shape than the view's logical shape. **Symptoms**: `applegpu: failed to query output shape` errors, GPT-2 crashes. **Workaround**: `ensure_op_ready()` detects shape mismatches and forces a contiguous copy. **Fix needed**: Give views their own tensor_ids linked to base storage, or teach the Rust runtime about strides/offsets.

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

# C++ backend (PrivateUse1)
make build-cpp-backend   # cargo build --release + torch.utils.cpp_extension
make test-cpp-backend    # pytest python/tests/test_cpp_backend.py

# Benchmarks
make bench-mlp-cpp       # MLP training: CPU vs applegpu

# Run a single Rust test
cargo test test_name

# Run a single Python test
uv run pytest python/tests/test_file.py::test_name -v

# Quick compile check (no linking)
make check
```

**C++ backend build note**: Requires `ARCHFLAGS="-arch arm64"` on Apple Silicon (universal builds break DeviceGuard registration). The `backend_cpp/setup.py` sets this automatically.

## Key Files

### Rust Core
- `crates/core/src/ops.rs` — High-level tensor operations (84 ops including forward, backward, scatter, grouped conv)
- `crates/core/src/compute.rs` — KernelRegistry, ComputePipeline, MSL kernel sources
- `crates/core/src/buffer.rs` — Safe Rust Buffer wrapper around MTLBuffer (zero-copy shared memory)
- `crates/core/src/tensor.rs` — DType, Shape, TensorMeta, Tensor (buffer-backed)
- `crates/core/src/ffi.rs` — Rust-Swift FFI boundary (extern "C" declarations + safe wrappers)
- `crates/core/src/backend_ffi.rs` — Rust-C++ FFI bridge (17 extern "C" functions for PrivateUse1)
- `crates/core/src/lazy.rs` — LazyRuntime: graph recording, eval, pre-allocated buffers, deferred-free
- `crates/core/src/fusion.rs` — Kernel fusion (matmul+add+gelu → single Metal kernel)
- `crates/core/src/device.rs` — RAII Device wrapper with Drop-based cleanup
- `crates/core/src/backend.rs` — Backend enum, Runtime, init_backend() with OnceCell
- `crates/core/src/error.rs` — GpuError enum and Result alias
- `crates/core/build.rs` — Compiles Swift static lib and links it into Rust

### Swift Bridge
- `swift/Sources/AppleGPUBridge/bridge.swift` — Swift C ABI bridge (@_cdecl exports) + device handle helper
- `swift/Sources/AppleGPUBridge/buffer.swift` — MTLBuffer C ABI (create, read, write, destroy)
- `swift/Sources/AppleGPUBridge/compute.swift` — Metal compute pipeline C ABI (binary, unary, matmul dispatch)
- `swift/Sources/AppleGPUBridge/kernels.swift` — MSL kernel source strings (used by Swift tests)
- `swift/Sources/AppleGPUBridge/include/bridge.h` — shared C header for the FFI contract

### C++ Backend (PrivateUse1)
- `backend_cpp/applegpu_backend.cpp` — C++ shim: allocator, native ops, CPU fallback, DeviceGuard
- `backend_cpp/applegpu_ffi.h` — C header for Rust FFI functions
- `backend_cpp/setup.py` — torch.utils.cpp_extension build config
- `python/applegpu_runtime/cpp_backend.py` — `load_cpp_backend()` entry point

### Python
- `crates/python/src/lib.rs` — PyO3 module definition (Python-facing API surface)
- `python/applegpu_runtime/__init__.py` — Python package entry point (loads PyO3 native extension)
- `python/tests/test_cpp_backend.py` — PrivateUse1 integration tests (15 tests)

## Performance Philosophy

This library must be **hyperoptimized**. Every layer is chosen for maximum performance:
- Rust core for zero-cost abstractions, memory safety without GC overhead
- Swift compatibility layer for native Metal/AVF access with no bridging penalty
- Zero-copy tensor transport via shared memory — avoid unnecessary allocations and copies
- Lazy execution with kernel fusion to minimize GPU dispatch overhead
- Persistent memory pools to reduce GPU allocation churn

Always prefer the fastest path. Profile before and after changes to performance-critical code.

**Current reality**: GPU is slower than CPU for all tested model sizes due to per-op dispatch overhead (P1). Adding more native ops will not fix this — the bottleneck is dispatch, not missing ops. Focus on graph fusion and batched execution before op coverage.

## Development Workflow

This project uses **TDD**. Write tests first, then implement.

**After pushing or committing significant changes**, update `README.md` to reflect new capabilities, API changes, or status updates. Keep the README consistent with the actual state of the library.

- **Rust tests**: unit tests inline (`#[cfg(test)]`), integration tests in `crates/core/tests/`
- **Swift tests**: Swift Testing framework in `swift/Tests/AppleGPUBridgeTests/`
- **Python tests**: pytest in `python/tests/`

## Toolchain

- **Rust**: cargo (workspace with crates: `applegpu-core`, `applegpu-wire`, `applegpu-python`, `applegpu-client`, `applegpu-service`)
- **Swift**: SwiftPM (package in `swift/`, produces static library `libAppleGPUBridge.a`)
- **Python**: uv + maturin (pyproject.toml at root, `uv sync` handles everything)
- **C++ backend**: torch.utils.cpp_extension, links against Rust staticlib + Swift staticlib
- Backend selection: `gpu.init_backend()` or `APPLEGPU_BACKEND=mlx|vm` env var

## Architecture Layers (inside Rust core)

1. **API Layer** — backend selection, error propagation, tensor metadata
2. **Tensor Layer** — virtual tensors (ID, shape, dtype, location), zero-copy shared memory
3. **Graph Layer** — lazy op capture, kernel fusion (e.g. matmul+add+gelu → single Metal kernel)
4. **Scheduler/Executor** — multi-container/multi-VM batching, priority queues, fairness
5. **IPC Layer** — shared-memory communication for AVF VM backend only
