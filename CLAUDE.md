# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**applegpu_runtime** is a unified Metal GPU runtime library for Apple Silicon. It exposes a single Python API (`import applegpu_runtime as gpu`) that abstracts over two backends:

- **MLX-native backend** (default): direct MLX → Metal GPU execution inside ACF containers
- **AVF VM backend**: VM-mediated Metal execution via Apple Virtualization Framework, providing isolation, snapshots, and DDP/multi-node simulation

## Two PyTorch Integration Paths

### 1. PrivateUse1 C++ Path (primary, `device='applegpu'`)
- Entry: `from applegpu_runtime.cpp_backend import load_cpp_backend` → `load_cpp_backend()`
- C++ shim (`backend_cpp/applegpu_backend.cpp`) registers ops at PrivateUse1 dispatch key
- **Eager Metal Dispatch**: ops encode directly into streaming Metal command buffer via `EagerRuntime` (Rust). No graph engine in hot path.
- MLP training end-to-end, **zero CPU fallback**. Forward/loss/step all sub-0.1ms.
- `torch.compile` supported via custom FX interpreter (`python/applegpu_runtime/compile_backend.py`) — walks FX graph nodes, calls Rust eager FFI directly via ctypes, bypasses PyTorch's C++ Dispatcher
- 35 Python + 418 Rust tests passing
- **Bottleneck**: PyTorch C++ Dispatcher overhead (7.5µs/op) dominates backward pass. Custom FX interpreter bypasses dispatcher for forward + backward (P3 in progress).

### 2. PyO3 Path (original, `__torch_dispatch__`)
- Entry: `import applegpu_runtime as gpu` → `gpu.init_backend()` → `torch_backend.py`
- ~110 aten ops registered via Python `__torch_dispatch__`
- Works for Whisper, LSTM, GRU, CNN training
- **Future**: will be repurposed as a `torch.compile` backend for lazy graph optimization with fusion

## Architecture

Three execution paths share the same Swift/Metal bridge:

```
PrivateUse1 C++ Path (eager, per-op dispatch):
  C++ op → Rust FFI → Metal encode (streaming CB) → GPU

FX Interpreter Path (torch.compile, bypasses C++ Dispatcher):
  Python FX walk → ctypes → Rust eager FFI → Metal encode (streaming CB) → GPU

MPSGraph Path (torch.compile, fused execution — WIP):
  Python FX walk → serialize bytecode → Rust FFI → Swift MPSGraph build/run → GPU

PyO3 Path (lazy, future torch.compile graph optimization):
  Python → PyO3 → Rust graph → eval → Metal → GPU
```

All paths share:
```
Swift Compat Layer  ←  Metal, MetalPerformanceShadersGraph, AVF via @_cdecl C ABI (.dylib)
        ↓
   Metal GPU (MPSMatrixMultiplication for matmul, custom MSL for elementwise)
```

- **Rust ↔ Swift bridge**: Swift exports C ABI functions via `@_cdecl`, Rust calls them via `extern "C"` in `crates/core/src/ffi.rs`. The C header is at `swift/Sources/AppleGPUBridge/include/bridge.h`. The Swift bridge is built as a dynamic library (`libAppleGPUBridge.dylib`) so both PyO3 and C++ backends can link against it without ObjC class conflicts. `build.rs` compiles the Swift library and links it automatically.
- **Rust ↔ Python bridge**: PyO3 cdylib in `crates/python/`, maturin builds it. The Python package is in `python/applegpu_runtime/`.
- **Rust ↔ C++ bridge**: `extern "C"` FFI in `crates/core/src/backend_ffi.rs`, C header in `backend_cpp/applegpu_ffi.h`. C++ shim links against `libapplegpu_core.a` (staticlib).
- **Important**: The PyO3 cdylib cannot be built via `cargo build --workspace` (missing Python symbols). Use `cargo build -p applegpu-core` for Rust-only builds, and `uv run maturin develop` for the Python extension.

## Known Problems (Priority Order)

### P0: Dual `.so` Conflict — RESOLVED
Both PyO3 and C++ backends previously statically linked `libAppleGPUBridge.a`, causing ObjC class duplication when both were loaded in the same process. **Fixed** by switching `libAppleGPUBridge` to a dynamic library (`.dylib`). Both backends now link dynamically against the shared `.dylib`, eliminating class conflicts.

### P1: Per-Op Metal Dispatch — RESOLVED
Eager Metal Dispatch (D1-D4) complete. Ops encode directly into a streaming Metal command buffer. Forward/loss/step are all sub-0.1ms. Zero CPU fallback. See spec: `docs/superpowers/specs/2026-03-20-eager-metal-dispatch-design.md`.

### P2: View Tensor Identity — RESOLVED
The eager runtime (`crates/core/src/eager.rs`) uses stride-aware `EagerTensor` with `Arc<Buffer>` sharing. Views carry their own shape/strides/offset referencing a shared Metal buffer. `ensure_op_ready()` eliminated. `binary_op` dispatches with actual tensor strides for correct view handling.

### P3: Per-Op Dispatch Overhead — APPROACH PIVOT NEEDED
**Original problem**: PyTorch C++ Dispatcher adds ~7.5µs per op for PrivateUse1.

**Python FX interpreter** (`compile_backend.py`) — functional but slower than C++ path:
- Bypasses C++ Dispatcher via ctypes, but Python overhead (~50µs/op) exceeds the dispatcher overhead it replaces
- Benchmark: 2-3x slower than C++ dispatcher path (h=256: 2.16ms vs 0.50ms)
- Root cause: Python per-node overhead (ctypes marshaling, dict lookups, isinstance, _query_shape FFI per op)
- Forward + backward + optimizer all working (26 tests, 0 skips)

**Implemented**: Two execution modes for torch.compile:
1. Python FX interpreter (ctypes per-op): functional, 3-5x slower than C++ dispatcher
2. Rust compiled graph executor (single FFI call): functional, 2-4x slower than C++ dispatcher
Both bottlenecked by output wrapping (torch.empty + memcpy) and flush_and_wait overhead, NOT by per-op dispatch.

**The real bottleneck is NOT the C++ Dispatcher** — at 7.5µs/op × 30 ops = 225µs, it's only ~10% of training time. The earlier 4.4ms backward was measured with Python `__torch_dispatch__`, not C++ PrivateUse1. The C++ path is already fast.

**For further speedup**: need kernel fusion (fuse elementwise chains via `lazy.rs` + `fusion.rs`) which reduces op COUNT, not per-op overhead. This is P4 work.

### P4: Competitive Performance with MPS — TILED MATMUL DONE
MPS is 2-3x faster than us at h≥1024 via MPSGraph whole-graph fusion. Our key differentiator: **MPS doesn't work in Docker containers** (no Metal driver inside Linux). Our architecture bridges Metal on the host to containers via IPC.

**Done:**
1. ~~**MPSMatrixMultiplication**~~ — DONE: replaced custom MSL matmul for unbatched (bs==1)
2. ~~**MPS transposed matmul**~~ — DONE: skip contiguity copies for backward transpose views
3. ~~**GPU-native mul_.Scalar**~~ — DONE: scalar_mul + storage swap instead of flush + CPU loop
4. ~~**Tiled matmul kernel**~~ — DONE: 32×32 threadgroup shared memory tiles with cooperative loading, boundary checks, `threadgroup_barrier`. 2.5x GPT-2 speedup at seq=32. Used for batched matmul (bs>1), MPS for unbatched.
5. ~~**div.Scalar / Float64 div fix**~~ — DONE: `x / 2.0` crashed (Float64 not supported in MSL). Fixed: scalar div uses `scalar_mul(1/val)` instead of moving Float64 tensor to GPU.
6. ~~**add/sub scalar Float64 fix**~~ — DONE: `x + 1.0`, `1.0 + x`, `x - 0.5` all crashed (Float64 scalar tensor). Fixed: scalar add/sub via copy + `add_scaled_inplace`.
7. ~~**GPT-2 full forward pass**~~ — DONE: 12-layer GPT-2 (124M params) runs end-to-end on `device='applegpu'`. 29 Python tests passing.
8. ~~**HuggingFace GPT-2 text generation**~~ — DONE: `GPT2LMHeadModel.from_pretrained('gpt2').to('applegpu')` loads and generates text matching CPU output exactly. 6.8 tok/s (vs CPU 16.6 tok/s, no KV cache yet).

**MPSGraph integration** (functional, opt-in via `APPLEGPU_MPSGRAPH=1`):
- Swift `mpsgraph.swift`: deserializes bytecode → MPSGraph ops (all MLP ops)
- C ABI: `gpu_bridge_mpsgraph_build/run/destroy`
- Graph caching: FNV-1a hash of (bytecode + shapes), build once / run many (59x speedup over uncached)
- Square transpose: gatherND index permutation workaround (MPSGraph transposeTensor is no-op for N==M)
- Design spec: `docs/superpowers/specs/2026-03-22-mpsgraph-integration-design.md`
- Status: 40% faster than per-op compiled, but 3x slower than C++ dispatcher (output wrapping overhead)

**Remaining for production:**
- Eliminate `_wrap_output` overhead (needs C++-level tensor creation from Rust buffer)
- Container IPC path: gpu-service on host → Metal GPU
- KV cache generation: works but only 1.1x faster (per-op dispatch overhead dominates, not recomputation)

**Container IPC path** (P1, functional):
- `RemoteEagerRuntime` in `crates/core/src/remote_eager.rs`: proxy buffers with stable pointers, op recording, stride-aware view upload, deferred free
- `eager_ffi.rs` refactored: `EagerBackend::Local` (Metal) vs `EagerBackend::Remote` (socket) dispatched based on `APPLEGPU_SOCKET` env var
- **Zero changes to `applegpu_backend.cpp`** — same C FFI contract
- gpu-service on host processes ops via LazyRuntime → Metal GPU, returns results over socket
- Integration tests: add, matmul, Linear, MLP all pass via socket IPC
- `python/tests/test_remote_ipc.py` — 4 integration tests

**Remaining for container IPC:**
- Docker container image with bind-mounted Unix socket
- Remaining remote ops (layer_norm, sum_dim, embedding, softmax) for GPT-2 via socket
- Performance benchmarking (remote vs local overhead)

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

**C++ backend build note**: Requires `ARCHFLAGS="-arch arm64"` on Apple Silicon (universal builds break DeviceGuard registration). The `backend_cpp/setup.py` sets this automatically. Uses `-Wl,-force_load` to export all Rust FFI symbols (needed for ctypes access from `compile_backend.py`).

## Key Files

### Rust Core
- `crates/core/src/ops.rs` — High-level tensor operations (84 ops including forward, backward, scatter, grouped conv)
- `crates/core/src/compute.rs` — KernelRegistry, ComputePipeline, MSL kernel sources
- `crates/core/src/buffer.rs` — Safe Rust Buffer wrapper around MTLBuffer (zero-copy shared memory)
- `crates/core/src/tensor.rs` — DType, Shape, TensorMeta, Tensor (buffer-backed)
- `crates/core/src/ffi.rs` — Rust-Swift FFI boundary (extern "C" declarations + safe wrappers)
- `crates/core/src/backend_ffi.rs` — Rust-C++ FFI bridge (17 extern "C" functions for PrivateUse1)
- `crates/core/src/eager_ffi.rs` — Eager dispatch FFI bridge (alloc, free, binary/unary/matmul ops, views, find_by_data_ptr reverse lookup)
- `crates/core/src/compiled_graph.rs` — Compiled graph executor (bytecode → per-op or MPSGraph execution)
- `crates/core/src/lazy.rs` — LazyRuntime: graph recording, eval, pre-allocated buffers, deferred-free (future: torch.compile backend)
- `crates/core/src/fusion.rs` — Kernel fusion (matmul+add+gelu → single Metal kernel) (future: torch.compile backend)
- `crates/core/src/device.rs` — RAII Device wrapper with Drop-based cleanup
- `crates/core/src/backend.rs` — Backend enum, Runtime, init_backend() with OnceCell
- `crates/core/src/error.rs` — GpuError enum and Result alias
- `crates/core/build.rs` — Compiles Swift dynamic lib and links it into Rust

### Swift Bridge
- `swift/Sources/AppleGPUBridge/bridge.swift` — Swift C ABI bridge (@_cdecl exports) + device handle helper
- `swift/Sources/AppleGPUBridge/buffer.swift` — MTLBuffer C ABI (create, read, write, destroy)
- `swift/Sources/AppleGPUBridge/compute.swift` — Metal compute pipeline C ABI (binary, unary, matmul dispatch)
- `swift/Sources/AppleGPUBridge/kernels.swift` — MSL kernel source strings (used by Swift tests)
- `swift/Sources/AppleGPUBridge/mpsgraph.swift` — MPSGraph integration (build, cache, execute via C ABI)
- `swift/Sources/AppleGPUBridge/include/bridge.h` — shared C header for the FFI contract

### C++ Backend (PrivateUse1)
- `backend_cpp/applegpu_backend.cpp` — C++ shim: allocator, native ops, CPU fallback, DeviceGuard
- `backend_cpp/applegpu_ffi.h` — C header for Rust FFI functions
- `backend_cpp/setup.py` — torch.utils.cpp_extension build config
- `python/applegpu_runtime/cpp_backend.py` — `load_cpp_backend()` entry point

### Python
- `crates/python/src/lib.rs` — PyO3 module definition (Python-facing API surface)
- `python/applegpu_runtime/__init__.py` — Python package entry point (loads PyO3 native extension)
- `python/applegpu_runtime/compile_backend.py` — Custom FX interpreter for torch.compile (bypasses C++ Dispatcher via ctypes)
- `python/tests/test_cpp_backend.py` — PrivateUse1 integration tests (30 tests)
- `python/tests/test_remote_ipc.py` — Container IPC integration tests (4 tests: add, matmul, linear, MLP)
- `python/tests/test_compile_backend.py` — torch.compile FX interpreter tests (10 tests, backward/training run in subprocesses)

### Design Specs
- `docs/superpowers/specs/2026-03-20-eager-metal-dispatch-design.md` — Eager Metal Dispatch architecture spec

## Performance Philosophy

This library must be **hyperoptimized**. Every layer is chosen for maximum performance:
- Rust core for zero-cost abstractions, memory safety without GC overhead
- Swift compatibility layer for native Metal/AVF access with no bridging penalty
- Zero-copy tensor transport via shared memory — avoid unnecessary allocations and copies
- Eager Metal dispatch for streaming command buffer encoding with minimal sync points
- Lazy graph engine with kernel fusion reserved for `torch.compile` optimization of static graphs
- Persistent memory pools to reduce GPU allocation churn

Always prefer the fastest path. Profile before and after changes to performance-critical code.

**Current reality** (MLP training benchmark, `python/tests/bench_comparison.py`):
```
TRAINING (ms/step)       h=64      h=256     h=1024     h=4096
----------------------------------------------------------------------
                 CPU     0.076     0.129     1.657    34.166
                 MPS     0.271     0.248     0.705     9.435
            applegpu     0.772     0.811     1.772    28.121

SPEEDUP vs CPU           h=64      h=256     h=1024     h=4096
----------------------------------------------------------------------
                 MPS      0.28x     0.52x     2.35x     3.62x
            applegpu     0.10x     0.16x     0.94x     1.21x
```

**GPT-2 forward** (124M params, 12 layers):
```
   Seq   CPU (ms)  applegpu (ms)  Speedup
     8       31.9           29.8    1.07x
    32       20.4           33.8    0.60x
   128       39.8           39.2    1.02x
```
- **applegpu beats CPU at h=4096** (1.21x) — tiled matmul + MPS for unbatched matmul
- **GPT-2 runs end-to-end** on `device='applegpu'` — 12-layer transformer, matches CPU to 1e-3
- **HuggingFace GPT-2 text generation** — `GPT2LMHeadModel.from_pretrained('gpt2').to('applegpu')` generates text matching CPU exactly (6.8 tok/s, no KV cache)
- **MPS still ~3x faster** at large sizes — MPSGraph whole-graph fusion
- **34 Python + 333 Rust tests** passing

## Development Workflow

This project uses **TDD**. Write tests first, then implement.

**After pushing or committing significant changes**, update `README.md` to reflect new capabilities, API changes, or status updates. Keep the README consistent with the actual state of the library.

- **Rust tests**: unit tests inline (`#[cfg(test)]`), integration tests in `crates/core/tests/`
- **Swift tests**: Swift Testing framework in `swift/Tests/AppleGPUBridgeTests/`
- **Python tests**: pytest in `python/tests/`

## Toolchain

- **Rust**: cargo (workspace with crates: `applegpu-core`, `applegpu-wire`, `applegpu-python`, `applegpu-client`, `applegpu-service`)
- **Swift**: SwiftPM (package in `swift/`, produces dynamic library `libAppleGPUBridge.dylib`) — both PyO3 and C++ backends link dynamically to avoid ObjC class conflicts
- **Python**: uv + maturin (pyproject.toml at root, `uv sync` handles everything)
- **C++ backend**: torch.utils.cpp_extension, links against Rust staticlib + Swift dylib
- Backend selection: `gpu.init_backend()` or `APPLEGPU_BACKEND=mlx|vm` env var

## Architecture Layers (inside Rust core)

1. **API Layer** — backend selection, error propagation, tensor metadata
2. **Tensor Layer** — virtual tensors (ID, shape, dtype, strides, storage offset), zero-copy shared memory
3. **Eager Dispatch Layer** — streaming command buffer encoding, sync-point-driven commits (PrivateUse1 path)
4. **Graph Layer** — lazy op capture, kernel fusion (future: `torch.compile` backend)
5. **Scheduler/Executor** — multi-container/multi-VM batching, priority queues, fairness
6. **IPC Layer** — shared-memory communication for AVF VM backend only
