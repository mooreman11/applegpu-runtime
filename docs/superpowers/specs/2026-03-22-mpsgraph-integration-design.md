# MPSGraph Integration Design

**Date**: 2026-03-22
**Status**: Planned
**Priority**: P1

## Why

PyTorch's `device='mps'` doesn't work inside Docker containers — there's no Metal driver inside Linux. Our architecture bridges this: Swift on the macOS host talks to Metal, Rust/Python inside the container communicates via IPC. Integrating MPSGraph brings Apple's fused GPU performance to containers where MPS can't reach.

**Current gap**: We dispatch ~20 individual Metal compute encoders per MLP training step. MPS uses MPSGraph to fuse entire subgraphs into ~5 dispatches. This is a 2x performance gap at h=4096 (1.59x vs 3.15x CPU).

## Architecture

### Key Insight

MPSGraph builds symbolic graphs (like our `lazy.rs`), compiles them, then executes with concrete MTLBuffer bindings. Our `compiled_graph.rs` already serializes FX graphs as bytecode. Instead of replaying that bytecode op-by-op through EagerRuntime, we send it to Swift, build an MPSGraph, and run it.

### Execution Flow

```
torch.compile captures FX graph (Python, once)
  → serialize to bytecode (Python, once)
    → applegpu_eager_execute_graph FFI (Rust, per call)
      → NEW: gpu_bridge_mpsgraph_run (Swift, per call)
        → MPSGraph.run(feeds: buffers, targets: outputs)
          → Metal GPU (fused kernels)
```

### Graph Caching

The graph structure is **static** for a given model — shapes, ops, and topology don't change between calls. Only the buffer contents change. So we:

1. **Build once**: First call hashes the bytecode + input shapes → cache key. Build MPSGraph, optionally compile to MPSGraphExecutable.
2. **Run many**: Subsequent calls with the same cache key skip graph construction, bind fresh buffers, execute.

Cache key: `hash(ops_bytecode) XOR hash(input_shapes_tuple)`

### Buffer Binding (Zero-Copy)

Our `EagerTensor` holds `Arc<Buffer>` which wraps `GPUBufferHandle` which wraps `MTLBuffer`. MPSGraph takes `MPSGraphTensorData(MTLBuffer)` as feed. The chain is zero-copy — same shared memory buffer throughout.

```
Rust EagerTensor.buffer (Arc<Buffer>)
  → Buffer.raw_handle() → GPUBufferHandle* → passed through FFI
    → Swift GPUBuffer.buffer (MTLBuffer)
      → MPSGraphTensorData(device, data: mtlBuffer)
        → MPSGraph reads/writes directly
```

Outputs: Rust pre-allocates output buffers from the pool, passes handles to Swift, MPSGraph writes results directly into them.

## Op Mapping

| Our Op Code | Bytecode | MPSGraph Method |
|-------------|----------|-----------------|
| OP_ADD (0) | binary | `graph.addition(_:_:)` |
| OP_SUB (1) | binary | `graph.subtraction(_:_:)` |
| OP_MUL (2) | binary | `graph.multiplication(_:_:)` |
| OP_DIV (3) | binary | `graph.division(_:_:)` |
| OP_MATMUL (4) | binary | `graph.matrixMultiplication(primary:secondary:)` |
| OP_RELU (5) | unary | `graph.reLU(with:)` |
| OP_NEG (6) | unary | `graph.negative(with:)` |
| OP_THRESHOLD_BACKWARD (7) | ternary | `graph.select(predicate: graph.greaterThan(input, threshold), truePredicate: grad, falsePredicate: zero)` |
| OP_SCALAR_MUL (8) | unary+param | `graph.multiplication(_:graph.constant(scale, shape:[1]))` |
| OP_MEAN_ALL (9) | unary | `graph.mean(of:axes:[all])` |
| OP_SUM_DIM (10) | unary+param | `graph.reductionSum(with:axis:)` |
| OP_TRANSPOSE (11) | unary | `graph.transposeTensor(_:dimension:withDimension:)` |
| OP_VIEW (12) | unary | `graph.reshape(_:shape:)` |
| OP_ADDMM (13) | ternary | `graph.addition(graph.matrixMultiplication(mat1, mat2), bias)` |
| OP_IDENTITY (255) | passthrough | identity (no MPSGraph node) |

This covers the full MLP forward + backward graph.

## Phased Implementation

### Phase 1: MPSGraph Builder + Execute (core path)

**Files to create:**
- `swift/Sources/AppleGPUBridge/mpsgraph.swift` — graph builder, cache, C ABI exports

**Files to modify:**
- `swift/Package.swift` — add `MetalPerformanceShadersGraph` framework
- `crates/core/build.rs` — add `-framework MetalPerformanceShadersGraph` linker flag
- `swift/Sources/AppleGPUBridge/include/bridge.h` — declare new FFI functions
- `crates/core/src/ffi.rs` — `extern "C"` declarations for new Swift functions
- `crates/core/src/compiled_graph.rs` — add MPSGraph execution mode alongside existing per-op mode

**New FFI functions:**
```c
// Build (or retrieve from cache) an MPSGraph from serialized bytecode.
// Returns opaque graph handle, or NULL on error.
void* gpu_bridge_mpsgraph_build(
    void* device_handle,
    const uint8_t* ops_data, uint32_t ops_len,
    uint32_t n_inputs,
    const int64_t* input_shapes,     // flattened [d0,d1,...,d0,d1,...] per input
    const uint32_t* input_ndims,     // ndim per input
    const uint32_t* input_dtypes,    // DType wire value per input
    uint32_t n_outputs,
    const uint16_t* output_indices   // which node indices are outputs
);

// Execute a cached MPSGraph with concrete buffer bindings.
// Returns 0 on success, -1 on error.
int32_t gpu_bridge_mpsgraph_run(
    void* graph_handle,
    void* queue_handle,
    const void** input_buffers, uint32_t n_inputs,
    void** output_buffers, uint32_t n_outputs
);

// Free a cached graph (called when compiled function is GC'd).
void gpu_bridge_mpsgraph_destroy(void* graph_handle);
```

**Swift MPSGraph builder** deserializes the same bytecode wire format that `compiled_graph.rs` uses, building MPSGraph operations instead of dispatching to EagerRuntime. The builder walks the bytecode sequentially, maintaining a `[nodeIndex: MPSGraphTensor]` map, and creates placeholders for inputs and op nodes for each operation.

**Graph caching** uses a Swift dictionary keyed by `(bytecodeHash, inputShapesHash)`. On cache hit, returns the existing handle. On miss, builds the graph, optionally compiles to `MPSGraphExecutable`, caches, and returns.

**Rust integration** in `compiled_graph.rs`: at the top of `execute()`, check if MPSGraph path is available. If so, call `gpu_bridge_mpsgraph_build` (cached) then `gpu_bridge_mpsgraph_run`. If the build fails (unsupported op), fall back to the existing per-op execution path transparently.

### Phase 2: Graph Compilation + Warm Cache

After Phase 1 is working, add `MPSGraphExecutable` compilation for even faster re-execution:

```swift
let executable = graph.compile(
    with: device,
    feeds: placeholders.map { MPSGraphShapedType(shape: $0.shape, dataType: $0.dataType) },
    targetTensors: outputs,
    targetOperations: nil,
    compilationDescriptor: nil
)
```

The compiled executable is stored alongside the graph in the cache. Subsequent runs use `executable.run(...)` instead of `graph.run(...)`.

### Phase 3: Container IPC Path

Extend the graph execution to work over the IPC socket (Unix socket / vsock):

```
Docker container (Rust client)
  → serialize graph bytecode + buffer data over socket
    → gpu-service (macOS host, Rust)
      → gpu_bridge_mpsgraph_run (Swift)
        → MPSGraph → Metal GPU
```

The wire protocol already supports `EvalRequest` with op lists + tensor data. The MPSGraph path is an optimization — the gpu-service builds the MPSGraph on the host and caches it for the container's model.

## Testing Strategy

1. **Swift unit test**: Build `addmm + relu` MPSGraph, execute, verify against CPU reference
2. **Rust integration test**: Serialize MLP forward graph → MPSGraph build → execute → compare to eager path outputs
3. **Python test**: `torch.compile` with MPSGraph backend, full training step, compare params to CPU reference (atol=5e-3)
4. **Benchmark**: MLP training at h=64, 256, 1024, 4096 — compare eager vs MPSGraph vs PyTorch MPS

## Expected Performance

MPSGraph will fuse `addmm(=matmul+add) + relu` into ~2 fused kernels (forward) instead of our current ~6 individual dispatches. For a 2-layer MLP:

- **Forward**: 6 kernels → ~2 fused MPSGraph ops
- **Backward**: ~14 kernels → ~4-6 fused ops
- **Total**: ~20 → ~8 dispatches (2.5x reduction)

At h=4096 (compute-dominated): expect ~1.5-2x speedup over current eager path, approaching MPS parity (3x CPU).

## Risks

1. **MPSGraph compilation cost**: ~1-10ms per graph build. Mitigated by caching — amortized to zero over many training steps.
2. **Output buffer ownership**: MPSGraph may allocate its own output buffers instead of writing to ours. Mitigated by using `MPSGraphTensorData(device:data:)` constructor with pre-allocated MTLBuffer and specifying result tensors in the run call.
3. **Op coverage gaps**: If MPSGraph doesn't support an op, the whole graph can't be built. Mitigated by falling back to per-op execution transparently.
4. **Shape dynamism**: Different batch sizes invalidate the cached graph. Mitigated by caching per shape tuple. MLP training uses fixed shapes.
5. **Framework availability**: `MetalPerformanceShadersGraph` requires macOS 12+. We already target macOS 14+ — no issue.
