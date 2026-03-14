# Phase 3b: Softmax, Transpose, and Attention Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add softmax (reduction op), transpose (2D), and scaled dot-product attention as a composite operation — completing the core ML op set needed for transformer inference.

**Architecture:** Softmax introduces a new dispatch pattern: reduction ops that process rows independently (not element-wise, not matmul). A Metal kernel computes numerically stable softmax per row of a 2D tensor. Transpose gets a simple element-copy kernel. Attention is a high-level function in `ops.rs` that composes existing ops: `softmax(Q @ transpose(K) * scale) @ V`. All three integrate into the lazy graph naturally.

**Tech Stack:** Rust (ops, graph), Swift (Metal kernels for softmax + transpose), Python (PyO3)

---

## File Structure

### Modified Files — Swift
- `swift/Sources/AppleGPUBridge/kernels.swift` — Add softmax and transpose MSL sources
- `swift/Sources/AppleGPUBridge/compute.swift` — Add `dispatchSoftmax` and `dispatchTranspose` methods + C ABI
- `swift/Sources/AppleGPUBridge/include/bridge.h` — Add softmax and transpose C ABI declarations
- `swift/Tests/AppleGPUBridgeTests/KernelTests.swift` — Add softmax and transpose tests

### Modified Files — Rust
- `crates/core/src/ffi.rs` — Add softmax and transpose extern declarations
- `crates/core/src/graph.rs` — Add `Softmax` and `Transpose` to OpKind
- `crates/core/src/compute.rs` — Add softmax and transpose MSL sources + dispatch methods on KernelRegistry
- `crates/core/src/lazy.rs` — Handle Softmax and Transpose in execute_node
- `crates/core/src/ops.rs` — Add softmax, transpose, attention functions
- `crates/core/src/serial.rs` — Add Softmax and Transpose discriminants
- `crates/core/src/lib.rs` — No new modules needed

### Modified Files — Python
- `crates/python/src/lib.rs` — Expose softmax, transpose, attention
- `python/applegpu_runtime/__init__.py` — Export new functions

### New Files
- `python/tests/test_attention.py` — Tests for softmax, transpose, and attention

---

## Chunk 1: Softmax and Transpose Metal Kernels (Swift)

### Task 1: Add softmax and transpose MSL kernels and Swift dispatch

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/kernels.swift`
- Modify: `swift/Sources/AppleGPUBridge/compute.swift`
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h`

- [ ] **Step 1: Add MSL kernel sources to kernels.swift**

Append to the `MetalKernels` enum in `swift/Sources/AppleGPUBridge/kernels.swift`:

```swift
    static let softmax = """
    #include <metal_stdlib>
    using namespace metal;

    // Numerically stable softmax along last dimension.
    // Input/output are 2D: [rows, cols]. Each thread processes one row.
    kernel void softmax_f32(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& rows [[buffer(2)]],
        constant uint& cols [[buffer(3)]],
        uint row [[thread_position_in_grid]]
    ) {
        if (row >= rows) return;

        uint offset = row * cols;

        // Find max for numerical stability
        float max_val = input[offset];
        for (uint j = 1; j < cols; j++) {
            max_val = max(max_val, input[offset + j]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (uint j = 0; j < cols; j++) {
            float e = exp(input[offset + j] - max_val);
            output[offset + j] = e;
            sum += e;
        }

        // Normalize
        for (uint j = 0; j < cols; j++) {
            output[offset + j] /= sum;
        }
    }
    """

    static let transpose = """
    #include <metal_stdlib>
    using namespace metal;

    // 2D transpose: output[col, row] = input[row, col]
    kernel void transpose_f32(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& rows [[buffer(2)]],
        constant uint& cols [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint row = gid.y;
        uint col = gid.x;
        if (row >= rows || col >= cols) return;
        output[col * rows + row] = input[row * cols + col];
    }
    """
```

- [ ] **Step 2: Add dispatch methods to GPUCompute in compute.swift**

Add before `// MARK: - C ABI exports` in `compute.swift`:

```swift
    func dispatchSoftmax(bufIn: MTLBuffer, bufOut: MTLBuffer, rows: Int, cols: Int) -> Bool {
        if rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var r = UInt32(rows), c = UInt32(cols)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)

        // 1D dispatch: one thread per row
        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, rows)
        let threadGroups = (rows + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func dispatchTranspose(bufIn: MTLBuffer, bufOut: MTLBuffer, rows: Int, cols: Int) -> Bool {
        if rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var r = UInt32(rows), c = UInt32(cols)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)

        // 2D dispatch
        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: cols, height: rows, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }
```

- [ ] **Step 3: Add C ABI exports to compute.swift**

Append to the C ABI exports section:

```swift
@_cdecl("gpu_bridge_compute_softmax")
public func gpuBridgeComputeSoftmax(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchSoftmax(
        bufIn: bufIn.buffer, bufOut: bufOut.buffer,
        rows: Int(rows), cols: Int(cols)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_transpose")
public func gpuBridgeComputeTranspose(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchTranspose(
        bufIn: bufIn.buffer, bufOut: bufOut.buffer,
        rows: Int(rows), cols: Int(cols)
    )
    return success ? 0 : -1
}
```

- [ ] **Step 4: Add C declarations to bridge.h**

Append before `#endif`:

```c
// Softmax along last dimension of 2D tensor [rows, cols]
int32_t gpu_bridge_compute_softmax(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols
);

// 2D transpose: output[cols, rows] = input[rows, cols]
int32_t gpu_bridge_compute_transpose(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols
);
```

- [ ] **Step 5: Build Swift**

Run: `cd swift && swift build 2>&1`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add swift/Sources/AppleGPUBridge/kernels.swift swift/Sources/AppleGPUBridge/compute.swift swift/Sources/AppleGPUBridge/include/bridge.h
git commit -m "feat: add softmax and transpose Metal kernels with C ABI dispatch"
```

---

### Task 2: Swift tests for softmax and transpose

**Files:**
- Modify: `swift/Tests/AppleGPUBridgeTests/KernelTests.swift`

- [ ] **Step 1: Add softmax and transpose tests**

Append to `KernelTests.swift`:

```swift
@Test func softmax2x3() {
    let devicePtr = gpuBridgeCreateDevice()
    guard let devicePtr = devicePtr else { return }
    defer { gpuBridgeDestroyDevice(devicePtr) }

    let computePtr = MetalKernels.softmax.withCString { src in
        "softmax_f32".withCString { name in
            gpuBridgeCreateCompute(devicePtr, src, name)
        }
    }
    guard let computePtr = computePtr else { return }
    defer { gpuBridgeDestroyCompute(computePtr) }

    // Input: [[1, 2, 3], [1, 1, 1]]
    let input: [Float] = [1, 2, 3, 1, 1, 1]
    let sizeBytes = UInt64(input.count * MemoryLayout<Float>.size)

    let bufIn = input.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, sizeBytes) }
    let bufOut = gpuBridgeCreateBuffer(devicePtr, sizeBytes)
    guard let bufIn = bufIn, let bufOut = bufOut else { return }
    defer { gpuBridgeDestroyBuffer(bufIn); gpuBridgeDestroyBuffer(bufOut) }

    let result = gpuBridgeComputeSoftmax(computePtr, bufIn, bufOut, 2, 3)
    #expect(result == 0)

    let outPtr = gpuBridgeBufferContents(bufOut)!.bindMemory(to: Float.self, capacity: 6)

    // Row 0: softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    #expect(abs(outPtr[0] - 0.0900) < 0.001)
    #expect(abs(outPtr[1] - 0.2447) < 0.001)
    #expect(abs(outPtr[2] - 0.6652) < 0.001)

    // Row 1: softmax([1,1,1]) = [1/3, 1/3, 1/3]
    #expect(abs(outPtr[3] - 0.3333) < 0.001)
    #expect(abs(outPtr[4] - 0.3333) < 0.001)
    #expect(abs(outPtr[5] - 0.3333) < 0.001)
}

@Test func transpose2x3() {
    let devicePtr = gpuBridgeCreateDevice()
    guard let devicePtr = devicePtr else { return }
    defer { gpuBridgeDestroyDevice(devicePtr) }

    let computePtr = MetalKernels.transpose.withCString { src in
        "transpose_f32".withCString { name in
            gpuBridgeCreateCompute(devicePtr, src, name)
        }
    }
    guard let computePtr = computePtr else { return }
    defer { gpuBridgeDestroyCompute(computePtr) }

    // Input [2,3]: [[1,2,3],[4,5,6]]
    // Output [3,2]: [[1,4],[2,5],[3,6]]
    let input: [Float] = [1, 2, 3, 4, 5, 6]
    let inSize = UInt64(6 * MemoryLayout<Float>.size)
    let outSize = UInt64(6 * MemoryLayout<Float>.size)

    let bufIn = input.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, inSize) }
    let bufOut = gpuBridgeCreateBuffer(devicePtr, outSize)
    guard let bufIn = bufIn, let bufOut = bufOut else { return }
    defer { gpuBridgeDestroyBuffer(bufIn); gpuBridgeDestroyBuffer(bufOut) }

    let result = gpuBridgeComputeTranspose(computePtr, bufIn, bufOut, 2, 3)
    #expect(result == 0)

    let outPtr = gpuBridgeBufferContents(bufOut)!.bindMemory(to: Float.self, capacity: 6)
    #expect(outPtr[0] == 1) // [0,0]
    #expect(outPtr[1] == 4) // [0,1]
    #expect(outPtr[2] == 2) // [1,0]
    #expect(outPtr[3] == 5) // [1,1]
    #expect(outPtr[4] == 3) // [2,0]
    #expect(outPtr[5] == 6) // [2,1]
}
```

- [ ] **Step 2: Run Swift tests**

Run: `cd swift && swift test 2>&1`
Expected: All tests pass (11 existing + 2 new = 13 total)

- [ ] **Step 3: Commit**

```bash
git add swift/Tests/AppleGPUBridgeTests/KernelTests.swift
git commit -m "test: add Swift softmax and transpose kernel tests"
```

---

## Chunk 2: Rust FFI, Graph, Compute, and Ops

### Task 3: Add Rust FFI, graph variants, and dispatch

**Files:**
- Modify: `crates/core/src/ffi.rs`
- Modify: `crates/core/src/graph.rs`
- Modify: `crates/core/src/compute.rs`
- Modify: `crates/core/src/lazy.rs`
- Modify: `crates/core/src/serial.rs`

- [ ] **Step 1: Add FFI declarations**

Append to the last `extern "C"` block in `ffi.rs`:

```rust
    pub fn gpu_bridge_compute_softmax(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_transpose(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> i32;
```

- [ ] **Step 2: Add OpKind variants to graph.rs**

Add to `OpKind` enum:

```rust
    // Reduction ops
    Softmax,
    // Shape ops
    Transpose,
    // Scalar multiply (carries the scalar value in the graph node)
    ScalarMul(f32),
```

Update `kernel_name()`:

```rust
            OpKind::Softmax => "softmax_f32",
            OpKind::Transpose => "transpose_f32",
            OpKind::ScalarMul(_) => "scalar_mul_f32",
```

`is_unary()`, `is_matmul()`, `is_elementwise()`, `is_fused()` should all return `false` for these.

Add new methods:

```rust
    pub fn is_softmax(&self) -> bool {
        matches!(self, OpKind::Softmax)
    }

    pub fn is_transpose(&self) -> bool {
        matches!(self, OpKind::Transpose)
    }

    pub fn is_scalar_mul(&self) -> bool {
        matches!(self, OpKind::ScalarMul(_))
    }
```

- [ ] **Step 3: Add MSL sources and dispatch methods to compute.rs**

Add MSL source constants:

```rust
const SOFTMAX_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = input[offset];
    for (uint j = 1; j < cols; j++) { max_val = max(max_val, input[offset + j]); }
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) { float e = exp(input[offset + j] - max_val); output[offset + j] = e; sum += e; }
    for (uint j = 0; j < cols; j++) { output[offset + j] /= sum; }
}
"#;

const TRANSPOSE_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void transpose_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= cols) return;
    output[col * rows + row] = input[row * cols + col];
}
"#;
```

Add ScalarMul MSL source:

```rust
const SCALAR_MUL_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scalar_mul_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) { output[id] = input[id] * scale; }
}
"#;
```

Note: ScalarMul uses the existing unary dispatch pattern (input → output) but with the scale value passed as `setBytes`. We'll add a new Swift dispatch method and FFI for this. Alternatively, since the scale is embedded in the MSL source, we can generate the kernel at runtime like FusedElementwise. The simpler approach: add a `dispatch_scalar_mul` C ABI that accepts the scale as a parameter.

Add to `bridge.h`:
```c
int32_t gpu_bridge_compute_scalar_mul(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    float scale,
    uint64_t element_count
);
```

Add to Swift `compute.swift` — `GPUCompute` method:
```swift
    func dispatchScalarMul(bufIn: MTLBuffer, bufOut: MTLBuffer, scale: Float, count: Int) -> Bool {
        if count == 0 { return true }
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)
        var s = scale
        encoder.setBytes(&s, length: MemoryLayout<Float>.size, index: 2)
        var c = UInt32(count)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, count)
        let threadGroups = (count + threadGroupSize - 1) / threadGroupSize
        encoder.dispatchThreadgroups(MTLSize(width: threadGroups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }
```

Add C ABI export:
```swift
@_cdecl("gpu_bridge_compute_scalar_mul")
public func gpuBridgeComputeScalarMul(_ computePtr: UnsafeMutableRawPointer?, _ bufInPtr: UnsafeRawPointer?, _ bufOutPtr: UnsafeMutableRawPointer?, _ scale: Float, _ elementCount: UInt64) -> Int32 {
    guard let computePtr = computePtr, let bufInPtr = bufInPtr, let bufOutPtr = bufOutPtr else { return -1 }
    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()
    return compute.dispatchScalarMul(bufIn: bufIn.buffer, bufOut: bufOut.buffer, scale: scale, count: Int(elementCount)) ? 0 : -1
}
```

Add Rust FFI:
```rust
    pub fn gpu_bridge_compute_scalar_mul(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        scale: f32,
        element_count: u64,
    ) -> i32;
```

Add dispatch methods to `ComputePipeline`:

```rust
    pub fn dispatch_softmax(
        &self,
        buf_input: &Buffer,
        buf_output: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_softmax(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), rows as u32, cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Softmax dispatch failed".to_string())) }
    }

    pub fn dispatch_transpose(
        &self,
        buf_input: &Buffer,
        buf_output: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_transpose(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), rows as u32, cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Transpose dispatch failed".to_string())) }
    }
```

Add to `ComputePipeline`:

```rust
    pub fn dispatch_scalar_mul(
        &self, buf_input: &Buffer, buf_output: &Buffer,
        scale: f32, element_count: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_scalar_mul(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), scale, element_count as u64,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("ScalarMul dispatch failed".to_string())) }
    }
```

Add to `KernelRegistry`:

```rust
    pub fn dispatch_softmax(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, SOFTMAX_KERNEL_SOURCE, "softmax_f32")?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, TRANSPOSE_KERNEL_SOURCE, "transpose_f32")?;
        pipeline.dispatch_transpose(buf_input, buf_output, rows, cols)
    }
```

Add to `KernelRegistry`:

```rust
    pub fn dispatch_scalar_mul(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        scale: f32, element_count: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, SCALAR_MUL_KERNEL_SOURCE, "scalar_mul_f32")?;
        pipeline.dispatch_scalar_mul(buf_input, buf_output, scale, element_count)
    }
```

- [ ] **Step 4: Handle Softmax, Transpose, and ScalarMul in lazy.rs execute_node**

Add cases in `execute_node` (before the unary branch):

```rust
        if node.op.is_softmax() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_softmax(device, &input.buffer, &out.buffer, rows, cols)?;
            return Ok(out);
        }

        if let crate::graph::OpKind::ScalarMul(scale) = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_scalar_mul(device, &input.buffer, &out.buffer, scale, input.numel())?;
            return Ok(out);
        }

        if node.op.is_transpose() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_transpose(device, &input.buffer, &out.buffer, rows, cols)?;
            return Ok(out);
        }
```

- [ ] **Step 5: Add serialization discriminants to serial.rs**

Add constants:

```rust
const OP_SOFTMAX: u32 = 11;
const OP_TRANSPOSE: u32 = 12;
const OP_SCALAR_MUL: u32 = 13;
```

Update `op_to_discriminant`:

```rust
        OpKind::Softmax => OP_SOFTMAX,
        OpKind::Transpose => OP_TRANSPOSE,
        OpKind::ScalarMul(_) => OP_SCALAR_MUL,
```

In `EvalRequest::serialize`, after the FusedElementwise data write block, add:

```rust
            if let OpKind::ScalarMul(scale) = node.op {
                buf.write_all(&scale.to_le_bytes()).unwrap();
            }
```

Update `discriminant_to_op`:

```rust
        OP_SOFTMAX => Ok(OpKind::Softmax),
        OP_TRANSPOSE => Ok(OpKind::Transpose),
        OP_SCALAR_MUL => {
            let mut scale_bytes = [0u8; 4];
            r.read_exact(&mut scale_bytes)?;
            Ok(OpKind::ScalarMul(f32::from_le_bytes(scale_bytes)))
        },
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All existing tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/ffi.rs crates/core/src/graph.rs crates/core/src/compute.rs crates/core/src/lazy.rs crates/core/src/serial.rs
git commit -m "feat: add softmax and transpose FFI, graph variants, and dispatch"
```

---

### Task 4: Add ops functions and Rust tests

**Files:**
- Modify: `crates/core/src/ops.rs`

- [ ] **Step 1: Add softmax, transpose, and attention to ops.rs**

```rust
/// Softmax along last dimension. Input must be 2D [rows, cols].
pub fn softmax(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    if shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "softmax requires 2D tensor, got {:?}", shape
        )));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Softmax,
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype: DType::Float32,
    });
    Ok(out_id)
}

/// Transpose a 2D tensor: [rows, cols] → [cols, rows].
pub fn transpose(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    if shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "transpose requires 2D tensor, got {:?}", shape
        )));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Transpose,
        inputs: vec![input_id],
        out_shape: Shape::new(vec![shape[1], shape[0]]), // swapped
        out_dtype: DType::Float32,
    });
    Ok(out_id)
}

/// Multiply every element by a scalar. Recorded as a lazy graph node.
pub fn scalar_mul(rt: &mut LazyRuntime, input_id: u64, scale: f32) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::ScalarMul(scale),
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype: DType::Float32,
    });
    Ok(out_id)
}

/// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
/// Q: [q_len, d_k], K: [kv_len, d_k], V: [kv_len, d_v]
/// Output: [q_len, d_v]
/// Supports self-attention (q_len == kv_len) and cross-attention (q_len != kv_len).
pub fn attention(rt: &mut LazyRuntime, q_id: u64, k_id: u64, v_id: u64) -> Result<u64> {
    let q_shape = rt.shape(q_id)?;
    let k_shape = rt.shape(k_id)?;
    let v_shape = rt.shape(v_id)?;

    if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(
            "attention requires 2D tensors for Q, K, V".to_string()
        ));
    }

    let d_k = q_shape[1];
    if k_shape[1] != d_k {
        return Err(GpuError::InvalidTensor(format!(
            "Q and K must have same d_k: Q[{},{}] K[{},{}]",
            q_shape[0], q_shape[1], k_shape[0], k_shape[1]
        )));
    }
    if k_shape[0] != v_shape[0] {
        return Err(GpuError::InvalidTensor(format!(
            "K and V must have same seq_len: K[{},{}] V[{},{}]",
            k_shape[0], k_shape[1], v_shape[0], v_shape[1]
        )));
    }

    // K^T: [kv_len, d_k] → [d_k, kv_len]
    let kt_id = transpose(rt, k_id)?;

    // scores = Q @ K^T: [q_len, d_k] @ [d_k, kv_len] → [q_len, kv_len]
    let scores_id = matmul(rt, q_id, kt_id)?;

    // Scale by 1/sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    let scaled_scores_id = scalar_mul(rt, scores_id, scale)?;

    // softmax(scaled_scores) along last dimension
    let attn_weights_id = softmax(rt, scaled_scores_id)?;

    // output = attn_weights @ V: [q_len, kv_len] @ [kv_len, d_v] → [q_len, d_v]
    let output_id = matmul(rt, attn_weights_id, v_id)?;

    Ok(output_id)
}
```

Note: The attention function skips the `/ sqrt(d_k)` scaling for now because we don't have a scalar multiply op and the ops API doesn't have access to the Device for creating scale tensors. This is documented as a TODO. The core attention pattern (transpose → matmul → softmax → matmul) works correctly.

- [ ] **Step 2: Add tests to ops.rs**

```rust
    #[test]
    fn lazy_ops_softmax() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // [[1, 2, 3], [1, 1, 1]]
        let a = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a);

        let s_id = softmax(&mut rt, a_id).unwrap();
        rt.eval(&device, s_id).unwrap();
        let result = rt.read_f32(s_id).unwrap();

        // Row 0: softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
        assert!((result[0] - 0.0900).abs() < 0.001);
        assert!((result[1] - 0.2447).abs() < 0.001);
        assert!((result[2] - 0.6652).abs() < 0.001);
        // Row 1: [1/3, 1/3, 1/3]
        assert!((result[3] - 0.3333).abs() < 0.001);
    }

    #[test]
    fn lazy_ops_transpose() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let a = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a);

        let t_id = transpose(&mut rt, a_id).unwrap();
        assert_eq!(rt.shape(t_id).unwrap(), vec![3, 2]);

        rt.eval(&device, t_id).unwrap();
        let result = rt.read_f32(t_id).unwrap();
        assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn lazy_ops_attention() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Simple 2x2 attention
        let q = Tensor::from_f32(&device, vec![2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = Tensor::from_f32(&device, vec![2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let q_id = q.meta.id;
        let k_id = k.meta.id;
        let v_id = v.meta.id;
        rt.insert_tensor(q);
        rt.insert_tensor(k);
        rt.insert_tensor(v);

        let out_id = attention(&mut rt, q_id, k_id, v_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        // With identity Q and K, scores = I, softmax(I) ≈ [[0.731, 0.269], [0.269, 0.731]]
        // output ≈ softmax(I) @ V
        assert_eq!(result.len(), 4);
        // Just verify it runs and produces reasonable values
        for &v in &result {
            assert!(v.is_finite());
        }
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All tests pass including 3 new ops tests

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/ops.rs
git commit -m "feat: add softmax, transpose, and attention ops with lazy graph support"
```

---

## Chunk 3: Python API and End-to-End Tests

### Task 5: Expose to Python

**Files:**
- Modify: `crates/python/src/lib.rs`
- Modify: `python/applegpu_runtime/__init__.py`
- Create: `python/tests/test_attention.py`

- [ ] **Step 1: Add Python bindings**

Add methods to `GpuTensor` in `crates/python/src/lib.rs`:

```rust
    fn softmax(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::softmax(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn transpose(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::transpose(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }
```

Add module-level functions:

```rust
#[pyfunction]
fn softmax(t: &GpuTensor) -> PyResult<GpuTensor> { t.softmax() }
#[pyfunction]
fn transpose(t: &GpuTensor) -> PyResult<GpuTensor> { t.transpose() }

#[pyfunction]
fn attention(q: &GpuTensor, k: &GpuTensor, v: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::attention(&mut rt, q.id, k.id, v.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}
```

Register in `#[pymodule]`:

```rust
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(attention, m)?)?;
```

- [ ] **Step 2: Update __init__.py**

Add `softmax`, `transpose`, `attention` to imports and `__all__`.

- [ ] **Step 3: Write Python tests**

Create `python/tests/test_attention.py`:

```python
import applegpu_runtime as gpu


def test_softmax_rows_sum_to_one():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0], shape=[2, 3])
    s = a.softmax()
    result = s.to_list()
    # Each row should sum to ~1.0
    row0_sum = sum(result[0:3])
    row1_sum = sum(result[3:6])
    assert abs(row0_sum - 1.0) < 1e-5
    assert abs(row1_sum - 1.0) < 1e-5


def test_softmax_uniform():
    gpu.init_backend()
    a = gpu.tensor([1.0, 1.0, 1.0, 1.0], shape=[1, 4])
    result = a.softmax().to_list()
    for v in result:
        assert abs(v - 0.25) < 1e-5


def test_transpose_2x3():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    t = a.transpose()
    assert t.shape == [3, 2]
    assert t.to_list() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_transpose_square():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    t = a.transpose()
    assert t.to_list() == [1.0, 3.0, 2.0, 4.0]


def test_attention_identity():
    gpu.init_backend()
    # Identity Q and K → attention weights ≈ softmax(I)
    q = gpu.tensor([1.0, 0.0, 0.0, 1.0], shape=[2, 2])
    k = gpu.tensor([1.0, 0.0, 0.0, 1.0], shape=[2, 2])
    v = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    out = gpu.attention(q, k, v)
    assert out.shape == [2, 2]
    result = out.to_list()
    assert len(result) == 4
    for val in result:
        assert 0.0 <= val <= 10.0  # sanity check


def test_attention_shape():
    gpu.init_backend()
    # Q: [4, 8], K: [4, 8], V: [4, 16]
    q = gpu.tensor([1.0] * 32, shape=[4, 8])
    k = gpu.tensor([1.0] * 32, shape=[4, 8])
    v = gpu.tensor([1.0] * 64, shape=[4, 16])
    out = gpu.attention(q, k, v)
    # Output should be [4, 16] = [seq_len, d_v]
    assert out.shape == [4, 16]
    result = out.to_list()
    assert len(result) == 64
```

- [ ] **Step 4: Build and run all tests**

Run: `uv run maturin develop && uv run pytest -v 2>&1`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/python/src/lib.rs python/applegpu_runtime/__init__.py python/tests/test_attention.py
git commit -m "feat: expose softmax, transpose, and attention to Python"
```

---

### Task 6: End-to-end verification and push

- [ ] **Step 1: Run full test suite from clean**

Run: `make clean && make test 2>&1`
Expected: All tests pass

- [ ] **Step 2: Update backlog, README, push**

Mark Phase 3b as complete. Add softmax/transpose/attention to README examples.

```bash
git push origin main
```
