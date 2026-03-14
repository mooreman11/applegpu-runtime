# Phase 3: Core GPU Operations Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add element-wise ops (sub, mul, div, neg, relu, exp, log, sqrt), matrix multiply, and a generalized kernel registry so new ops are trivial to add.

**Architecture:** Refactor the single hardcoded add kernel into a `KernelRegistry` that compiles and caches multiple MSL kernels by name. Element-wise ops each get a named kernel function, while matmul gets a simple 2D kernel (one thread per output element, no tiling — sufficient for correctness, optimize later). Python gets `gpu.sub()`, `gpu.mul()`, `gpu.div()`, `gpu.neg()`, `gpu.relu()`, `gpu.exp()`, `gpu.log()`, `gpu.sqrt()`, and `gpu.matmul()`.

**Note:** The existing `compute.rs` tests (`add_pipeline_creates`, `elementwise_add_computes_correctly`) are replaced by the new `KernelRegistry` tests which provide equivalent coverage. MSL kernel sources appear in both Swift (`kernels.swift`, for Swift tests only) and Rust (`compute.rs`, for runtime use).

**Tech Stack:** Rust (FFI, RAII), Swift (Metal compute pipelines), Python (PyO3), Metal Shading Language (MSL)

---

## File Structure

### New Files
- `swift/Sources/AppleGPUBridge/kernels.swift` — All MSL kernel sources as Swift string constants, kernel registry
- `swift/Tests/AppleGPUBridgeTests/KernelTests.swift` — Tests for all kernel operations
- `crates/core/src/ops.rs` — High-level op functions (add, sub, mul, matmul, etc.) that use ComputePipeline

### Modified Files
- `swift/Sources/AppleGPUBridge/compute.swift` — Add unary dispatch and matmul dispatch C ABI
- `swift/Sources/AppleGPUBridge/include/bridge.h` — Add unary and matmul C ABI declarations
- `crates/core/src/ffi.rs` — Add unary and matmul extern "C" declarations
- `crates/core/src/compute.rs` — Add unary dispatch method, matmul dispatch, kernel registry
- `crates/core/src/tensor.rs` — Add matmul shape validation helper
- `crates/core/src/lib.rs` — Add ops module
- `crates/python/src/lib.rs` — Expose all new ops to Python
- `python/applegpu_runtime/__init__.py` — Export new functions
- `python/tests/test_compute.py` — Add tests for all new ops

---

## Chunk 1: Generalized Element-wise Kernels (Swift)

### Task 1: Create MSL kernel source registry and add unary/binary dispatch to Swift

**Files:**
- Create: `swift/Sources/AppleGPUBridge/kernels.swift`
- Modify: `swift/Sources/AppleGPUBridge/compute.swift`
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h`

- [ ] **Step 1: Create kernels.swift with all MSL sources**

Create `swift/Sources/AppleGPUBridge/kernels.swift`:

```swift
import Foundation

/// All Metal Shading Language kernel sources.
enum MetalKernels {

    static let elementwiseBinary = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void elementwise_add(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = a[id] + b[id]; }
    }

    kernel void elementwise_sub(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = a[id] - b[id]; }
    }

    kernel void elementwise_mul(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = a[id] * b[id]; }
    }

    kernel void elementwise_div(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = a[id] / b[id]; }
    }
    """

    static let elementwiseUnary = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void elementwise_neg(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = -input[id]; }
    }

    kernel void elementwise_relu(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = max(input[id], 0.0f); }
    }

    kernel void elementwise_exp(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = exp(input[id]); }
    }

    kernel void elementwise_log(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = log(input[id]); }
    }

    kernel void elementwise_sqrt(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = sqrt(input[id]); }
    }
    """

    static let matmul = """
    #include <metal_stdlib>
    using namespace metal;

    // Simple matmul: C[M,N] = A[M,K] * B[K,N]
    // Each thread computes one element of C.
    kernel void matmul_f32(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& M [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint row = gid.y;
        uint col = gid.x;
        if (row >= M || col >= N) return;

        float sum = 0.0f;
        for (uint i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
    """
}
```

- [ ] **Step 2: Add unary dispatch and matmul dispatch to compute.swift**

Append to `GPUCompute` class in `swift/Sources/AppleGPUBridge/compute.swift`, before the `// MARK: - C ABI exports`:

```swift
    func dispatchUnary(bufIn: MTLBuffer, bufOut: MTLBuffer, count: Int) -> Bool {
        if count == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)

        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, count)
        let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return commandBuffer.status == .completed
    }

    func dispatchMatmul(bufA: MTLBuffer, bufB: MTLBuffer, bufC: MTLBuffer, M: Int, N: Int, K: Int) -> Bool {
        if M == 0 || N == 0 || K == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)

        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)

        // 2D dispatch: each thread computes one element of C[M,N]
        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: N, height: M, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return commandBuffer.status == .completed
    }
```

- [ ] **Step 3: Add unary and matmul C ABI exports to compute.swift**

Append to the C ABI exports section of `compute.swift`:

```swift
@_cdecl("gpu_bridge_compute_unary")
public func gpuBridgeComputeUnary(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchUnary(
        bufIn: bufIn.buffer,
        bufOut: bufOut.buffer,
        count: Int(elementCount)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_matmul")
public func gpuBridgeComputeMatmul(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufCPtr: UnsafeMutableRawPointer?,
    _ M: UInt32,
    _ N: UInt32,
    _ K: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufCPtr = bufCPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufC = Unmanaged<GPUBuffer>.fromOpaque(bufCPtr).takeUnretainedValue()

    let success = compute.dispatchMatmul(
        bufA: bufA.buffer,
        bufB: bufB.buffer,
        bufC: bufC.buffer,
        M: Int(M),
        N: Int(N),
        K: Int(K)
    )
    return success ? 0 : -1
}
```

- [ ] **Step 4: Add C declarations to bridge.h**

Append to `bridge.h` before the `#endif`:

```c
// Execute unary operation: out = op(input)
int32_t gpu_bridge_compute_unary(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    uint64_t element_count
);

// Execute matrix multiply: C[M,N] = A[M,K] * B[K,N]
int32_t gpu_bridge_compute_matmul(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_c,
    uint32_t M,
    uint32_t N,
    uint32_t K
);
```

- [ ] **Step 5: Run Swift build**

Run: `cd swift && swift build 2>&1`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add swift/Sources/AppleGPUBridge/kernels.swift swift/Sources/AppleGPUBridge/compute.swift swift/Sources/AppleGPUBridge/include/bridge.h
git commit -m "feat: add MSL kernels for binary/unary/matmul ops with dispatch"
```

---

### Task 2: Swift kernel tests

**Files:**
- Create: `swift/Tests/AppleGPUBridgeTests/KernelTests.swift`

- [ ] **Step 1: Write kernel tests**

```swift
import Testing
@testable import AppleGPUBridge

/// Helper to create a device, compile a kernel, and run a binary op on 4 floats.
func runBinaryOp(source: String, functionName: String, a: [Float], b: [Float]) -> [Float]? {
    let devicePtr = gpuBridgeCreateDevice()
    guard let devicePtr = devicePtr else { return nil }
    defer { gpuBridgeDestroyDevice(devicePtr) }

    let computePtr = source.withCString { src in
        functionName.withCString { name in
            gpuBridgeCreateCompute(devicePtr, src, name)
        }
    }
    guard let computePtr = computePtr else { return nil }
    defer { gpuBridgeDestroyCompute(computePtr) }

    let count = a.count
    let sizeBytes = UInt64(count * MemoryLayout<Float>.size)

    let bufA = a.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, sizeBytes) }
    let bufB = b.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, sizeBytes) }
    let bufOut = gpuBridgeCreateBuffer(devicePtr, sizeBytes)
    guard let bufA = bufA, let bufB = bufB, let bufOut = bufOut else { return nil }
    defer { gpuBridgeDestroyBuffer(bufA); gpuBridgeDestroyBuffer(bufB); gpuBridgeDestroyBuffer(bufOut) }

    let result = gpuBridgeComputeElementwise(computePtr, bufA, bufB, bufOut, UInt64(count))
    guard result == 0 else { return nil }

    let outPtr = gpuBridgeBufferContents(bufOut)!.bindMemory(to: Float.self, capacity: count)
    return (0..<count).map { outPtr[$0] }
}

/// Helper to run a unary op on 4 floats.
func runUnaryOp(source: String, functionName: String, input: [Float]) -> [Float]? {
    let devicePtr = gpuBridgeCreateDevice()
    guard let devicePtr = devicePtr else { return nil }
    defer { gpuBridgeDestroyDevice(devicePtr) }

    let computePtr = source.withCString { src in
        functionName.withCString { name in
            gpuBridgeCreateCompute(devicePtr, src, name)
        }
    }
    guard let computePtr = computePtr else { return nil }
    defer { gpuBridgeDestroyCompute(computePtr) }

    let count = input.count
    let sizeBytes = UInt64(count * MemoryLayout<Float>.size)

    let bufIn = input.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, sizeBytes) }
    let bufOut = gpuBridgeCreateBuffer(devicePtr, sizeBytes)
    guard let bufIn = bufIn, let bufOut = bufOut else { return nil }
    defer { gpuBridgeDestroyBuffer(bufIn); gpuBridgeDestroyBuffer(bufOut) }

    let result = gpuBridgeComputeUnary(computePtr, bufIn, bufOut, UInt64(count))
    guard result == 0 else { return nil }

    let outPtr = gpuBridgeBufferContents(bufOut)!.bindMemory(to: Float.self, capacity: count)
    return (0..<count).map { outPtr[$0] }
}

@Test func binarySub() {
    let result = runBinaryOp(source: MetalKernels.elementwiseBinary, functionName: "elementwise_sub", a: [10, 20, 30, 40], b: [1, 2, 3, 4])
    #expect(result == [9, 18, 27, 36])
}

@Test func binaryMul() {
    let result = runBinaryOp(source: MetalKernels.elementwiseBinary, functionName: "elementwise_mul", a: [2, 3, 4, 5], b: [10, 10, 10, 10])
    #expect(result == [20, 30, 40, 50])
}

@Test func binaryDiv() {
    let result = runBinaryOp(source: MetalKernels.elementwiseBinary, functionName: "elementwise_div", a: [10, 20, 30, 40], b: [2, 4, 5, 8])
    #expect(result == [5, 5, 6, 5])
}

@Test func unaryNeg() {
    let result = runUnaryOp(source: MetalKernels.elementwiseUnary, functionName: "elementwise_neg", input: [1, -2, 3, -4])
    #expect(result == [-1, 2, -3, 4])
}

@Test func unaryRelu() {
    let result = runUnaryOp(source: MetalKernels.elementwiseUnary, functionName: "elementwise_relu", input: [-1, 0, 3, -4])
    #expect(result == [0, 0, 3, 0])
}

@Test func matmul2x2() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    let devicePtr = gpuBridgeCreateDevice()
    guard let devicePtr = devicePtr else { return }
    defer { gpuBridgeDestroyDevice(devicePtr) }

    let computePtr = MetalKernels.matmul.withCString { src in
        "matmul_f32".withCString { name in
            gpuBridgeCreateCompute(devicePtr, src, name)
        }
    }
    guard let computePtr = computePtr else { return }
    defer { gpuBridgeDestroyCompute(computePtr) }

    let a: [Float] = [1, 2, 3, 4]
    let b: [Float] = [5, 6, 7, 8]
    let sizeBytes = UInt64(4 * MemoryLayout<Float>.size)

    let bufA = a.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, sizeBytes) }
    let bufB = b.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, sizeBytes) }
    let bufC = gpuBridgeCreateBuffer(devicePtr, sizeBytes)
    guard let bufA = bufA, let bufB = bufB, let bufC = bufC else { return }
    defer { gpuBridgeDestroyBuffer(bufA); gpuBridgeDestroyBuffer(bufB); gpuBridgeDestroyBuffer(bufC) }

    let result = gpuBridgeComputeMatmul(computePtr, bufA, bufB, bufC, 2, 2, 2)
    #expect(result == 0)

    let outPtr = gpuBridgeBufferContents(bufC)!.bindMemory(to: Float.self, capacity: 4)
    #expect(outPtr[0] == 19)
    #expect(outPtr[1] == 22)
    #expect(outPtr[2] == 43)
    #expect(outPtr[3] == 50)
}
```

- [ ] **Step 2: Run Swift tests**

Run: `cd swift && swift test 2>&1`
Expected: All tests pass (5 existing + 6 new = 11 total)

- [ ] **Step 3: Commit**

```bash
git add swift/Tests/AppleGPUBridgeTests/KernelTests.swift
git commit -m "test: add Swift tests for sub, mul, div, neg, relu, and matmul kernels"
```

---

## Chunk 2: Rust FFI, Kernel Registry, and Ops Module

### Task 3: Add Rust FFI declarations for unary and matmul

**Files:**
- Modify: `crates/core/src/ffi.rs`

- [ ] **Step 1: Add unary and matmul extern declarations**

Append to the second `extern "C"` block in `ffi.rs` (the one with compute functions):

```rust
    pub fn gpu_bridge_compute_unary(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;

    pub fn gpu_bridge_compute_matmul(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_c: *mut GPUBufferHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> i32;
```

- [ ] **Step 2: Run `cargo check -p applegpu-core`**

Expected: Compiles cleanly

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/ffi.rs
git commit -m "feat: add Rust FFI declarations for unary and matmul dispatch"
```

---

### Task 4: Refactor compute.rs with kernel registry and new dispatch methods

**Files:**
- Modify: `crates/core/src/compute.rs`

- [ ] **Step 1: Replace compute.rs with kernel registry and all ops**

Replace `crates/core/src/compute.rs` entirely:

```rust
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Mutex;

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;

/// MSL source for binary element-wise ops (add, sub, mul, div).
const BINARY_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] + b[id]; } }
kernel void elementwise_sub(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] - b[id]; } }
kernel void elementwise_mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] * b[id]; } }
kernel void elementwise_div(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] / b[id]; } }
"#;

/// MSL source for unary element-wise ops (neg, relu, exp, log, sqrt).
const UNARY_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_neg(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = -input[id]; } }
kernel void elementwise_relu(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = max(input[id], 0.0f); } }
kernel void elementwise_exp(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = exp(input[id]); } }
kernel void elementwise_log(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = log(input[id]); } }
kernel void elementwise_sqrt(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = sqrt(input[id]); } }
"#;

/// MSL source for matrix multiplication.
const MATMUL_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
"#;

/// A Metal compute pipeline. Wraps command queue + pipeline state.
pub struct ComputePipeline {
    handle: *mut ffi::GPUComputeHandle,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Create a compute pipeline from MSL source and function name.
    pub fn new(device: &Device, kernel_source: &str, function_name: &str) -> Result<Self> {
        let source = CString::new(kernel_source).map_err(|_| {
            GpuError::ComputeFailed("Invalid kernel source (null byte)".to_string())
        })?;
        let name = CString::new(function_name).map_err(|_| {
            GpuError::ComputeFailed("Invalid function name (null byte)".to_string())
        })?;

        let handle = unsafe {
            ffi::gpu_bridge_create_compute(device.raw_handle(), source.as_ptr(), name.as_ptr())
        };

        if handle.is_null() {
            Err(GpuError::ComputeFailed(format!(
                "Failed to create compute pipeline for '{}'",
                function_name
            )))
        } else {
            Ok(ComputePipeline { handle })
        }
    }

    /// Dispatch binary element-wise operation: out = op(a, b).
    pub fn dispatch_elementwise(
        &self,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_elementwise(
                self.handle,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Binary kernel dispatch failed".to_string())) }
    }

    /// Dispatch unary element-wise operation: out = op(input).
    pub fn dispatch_unary(
        &self,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_unary(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Unary kernel dispatch failed".to_string())) }
    }

    /// Dispatch matrix multiplication: C[M,N] = A[M,K] * B[K,N].
    pub fn dispatch_matmul(
        &self,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_matmul(
                self.handle,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_c.raw_handle(),
                m as u32,
                n as u32,
                k as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Matmul dispatch failed".to_string())) }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { ffi::gpu_bridge_destroy_compute(self.handle) };
    }
}

use std::sync::Arc;

/// Caches compiled compute pipelines by kernel name.
/// Uses Arc<ComputePipeline> so the lock is released before GPU dispatch.
pub struct KernelRegistry {
    pipelines: Mutex<HashMap<String, Arc<ComputePipeline>>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        KernelRegistry {
            pipelines: Mutex::new(HashMap::new()),
        }
    }

    /// Get or compile a pipeline for the given op. Returns Arc so caller
    /// can dispatch without holding the registry lock.
    fn get_or_create(
        &self,
        device: &Device,
        kernel_source: &str,
        function_name: &str,
    ) -> Result<Arc<ComputePipeline>> {
        let mut map = self.pipelines.lock().unwrap();
        if let Some(pipeline) = map.get(function_name) {
            return Ok(Arc::clone(pipeline));
        }
        let pipeline = Arc::new(ComputePipeline::new(device, kernel_source, function_name)?);
        map.insert(function_name.to_string(), Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// Dispatch a binary op through the registry.
    pub fn dispatch_binary(
        &self,
        device: &Device,
        function_name: &str,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, BINARY_KERNEL_SOURCE, function_name)?;
        // Lock is released here — GPU dispatch runs without holding the registry lock
        pipeline.dispatch_elementwise(buf_a, buf_b, buf_out, element_count)
    }

    /// Dispatch a unary op through the registry.
    pub fn dispatch_unary(
        &self,
        device: &Device,
        function_name: &str,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, UNARY_KERNEL_SOURCE, function_name)?;
        pipeline.dispatch_unary(buf_input, buf_out, element_count)
    }

    /// Dispatch matmul through the registry.
    pub fn dispatch_matmul(
        &self,
        device: &Device,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, MATMUL_KERNEL_SOURCE, "matmul_f32")?;
        pipeline.dispatch_matmul(buf_a, buf_b, buf_c, m, n, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    #[test]
    fn elementwise_add() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[10.0, 20.0, 30.0, 40.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_binary(&device, "elementwise_add", &a, &b, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn elementwise_sub() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[10.0, 20.0, 30.0, 40.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_binary(&device, "elementwise_sub", &a, &b, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn elementwise_mul() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[2.0, 3.0, 4.0, 5.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[10.0, 10.0, 10.0, 10.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_binary(&device, "elementwise_mul", &a, &b, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn unary_neg() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, -2.0, 3.0, -4.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_unary(&device, "elementwise_neg", &input, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn unary_relu() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, f32_as_bytes(&[-1.0, 0.0, 3.0, -4.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_unary(&device, "elementwise_relu", &input, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn matmul_2x2() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        // A=[[1,2],[3,4]] B=[[5,6],[7,8]] => C=[[19,22],[43,50]]
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[5.0, 6.0, 7.0, 8.0])).unwrap();
        let c = Buffer::new(&device, 16).unwrap();
        registry.dispatch_matmul(&device, &a, &b, &c, 2, 2, 2).unwrap();
        assert_eq!(unsafe { c.as_slice::<f32>() }, &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_non_square() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        // A[2,3] * B[3,2] = C[2,2]
        // A=[[1,2,3],[4,5,6]] B=[[7,8],[9,10],[11,12]]
        // C=[[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //  =[[58, 64], [139, 154]]
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])).unwrap();
        let c = Buffer::new(&device, 4 * 4).unwrap();
        registry.dispatch_matmul(&device, &a, &b, &c, 2, 2, 3).unwrap();
        assert_eq!(unsafe { c.as_slice::<f32>() }, &[58.0, 64.0, 139.0, 154.0]);
    }

    /// Helper: cast f32 slice to bytes.
    fn f32_as_bytes(data: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
    }
}
```

- [ ] **Step 2: Run Rust tests**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All tests pass including 7 new kernel registry tests

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/compute.rs
git commit -m "feat: add KernelRegistry with binary/unary/matmul ops"
```

---

### Task 5: Create ops.rs high-level API

**Files:**
- Create: `crates/core/src/ops.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Create ops.rs with tensor-level operations**

```rust
use once_cell::sync::Lazy;

use crate::compute::KernelRegistry;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::tensor::Tensor;

static REGISTRY: Lazy<KernelRegistry> = Lazy::new(KernelRegistry::new);

/// Element-wise binary operation on two tensors. Shapes must match.
fn binary_op(device: &Device, a: &Tensor, b: &Tensor, kernel_name: &str) -> Result<Tensor> {
    if a.meta.shape != b.meta.shape {
        return Err(GpuError::InvalidTensor(format!(
            "Shape mismatch: {:?} vs {:?}",
            a.meta.shape.dims(),
            b.meta.shape.dims()
        )));
    }
    let out = Tensor::empty_f32(device, a.meta.shape.dims().to_vec())?;
    REGISTRY.dispatch_binary(device, kernel_name, &a.buffer, &b.buffer, &out.buffer, a.numel())?;
    Ok(out)
}

/// Element-wise unary operation on a tensor.
fn unary_op(device: &Device, input: &Tensor, kernel_name: &str) -> Result<Tensor> {
    let out = Tensor::empty_f32(device, input.meta.shape.dims().to_vec())?;
    REGISTRY.dispatch_unary(device, kernel_name, &input.buffer, &out.buffer, input.numel())?;
    Ok(out)
}

pub fn add(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "elementwise_add")
}

pub fn sub(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "elementwise_sub")
}

pub fn mul(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "elementwise_mul")
}

pub fn div(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "elementwise_div")
}

pub fn neg(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_neg")
}

pub fn relu(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_relu")
}

pub fn exp(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_exp")
}

pub fn log(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_log")
}

pub fn sqrt(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_sqrt")
}

/// Matrix multiplication: C[M,N] = A[M,K] * B[K,N].
/// A must be 2D with shape [M,K], B must be 2D with shape [K,N].
pub fn matmul(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.meta.shape.dims();
    let b_dims = b.meta.shape.dims();

    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul requires 2D tensors, got {:?} and {:?}",
            a_dims, b_dims
        )));
    }

    let (m, k1) = (a_dims[0], a_dims[1]);
    let (k2, n) = (b_dims[0], b_dims[1]);

    if k1 != k2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul inner dimensions mismatch: A[{},{}] * B[{},{}]",
            m, k1, k2, n
        )));
    }

    let out = Tensor::empty_f32(device, vec![m, n])?;
    REGISTRY.dispatch_matmul(device, &a.buffer, &b.buffer, &out.buffer, m, n, k1)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    #[test]
    fn ops_add_sub_roundtrip() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let c = add(&device, &a, &b).unwrap();
        assert_eq!(c.as_f32_slice(), &[11.0, 22.0, 33.0, 44.0]);
        let d = sub(&device, &c, &b).unwrap();
        assert_eq!(d.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ops_mul_div() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![4], &[2.0, 4.0, 6.0, 8.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![4], &[2.0, 2.0, 2.0, 2.0]).unwrap();
        let c = mul(&device, &a, &b).unwrap();
        assert_eq!(c.as_f32_slice(), &[4.0, 8.0, 12.0, 16.0]);
        let d = div(&device, &c, &b).unwrap();
        assert_eq!(d.as_f32_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn ops_neg_relu() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![4], &[1.0, -2.0, 3.0, -4.0]).unwrap();
        let b = neg(&device, &a).unwrap();
        assert_eq!(b.as_f32_slice(), &[-1.0, 2.0, -3.0, 4.0]);
        let c = relu(&device, &a).unwrap();
        assert_eq!(c.as_f32_slice(), &[1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn ops_matmul_2x2() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = matmul(&device, &a, &b).unwrap();
        assert_eq!(c.meta.shape.dims(), &[2, 2]);
        assert_eq!(c.as_f32_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn ops_matmul_shape_validation() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![2, 3], &[1.0; 6]).unwrap();
        let b = Tensor::from_f32(&device, vec![2, 2], &[1.0; 4]).unwrap();
        let result = matmul(&device, &a, &b);
        assert!(result.is_err()); // inner dimensions 3 != 2
    }
}
```

- [ ] **Step 2: Add ops module to lib.rs**

Add `pub mod ops;` to `crates/core/src/lib.rs`.

- [ ] **Step 3: Run Rust tests**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All tests pass including 5 new ops tests

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/ops.rs crates/core/src/lib.rs
git commit -m "feat: add ops module with add/sub/mul/div/neg/relu/exp/log/sqrt/matmul"
```

---

## Chunk 3: Python API

### Task 6: Expose all ops to Python

**Files:**
- Modify: `crates/python/src/lib.rs`
- Modify: `python/applegpu_runtime/__init__.py`
- Modify: `python/tests/test_compute.py`

- [ ] **Step 1: Write failing Python tests for all new ops**

Append to `python/tests/test_compute.py`:

```python
def test_sub():
    gpu.init_backend()
    a = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    b = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    c = gpu.sub(a, b)
    assert gpu.to_list(c) == [9.0, 18.0, 27.0, 36.0]


def test_mul():
    gpu.init_backend()
    a = gpu.tensor([2.0, 3.0, 4.0, 5.0], shape=[4])
    b = gpu.tensor([10.0, 10.0, 10.0, 10.0], shape=[4])
    assert gpu.to_list(gpu.mul(a, b)) == [20.0, 30.0, 40.0, 50.0]


def test_div():
    gpu.init_backend()
    a = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    b = gpu.tensor([2.0, 4.0, 5.0, 8.0], shape=[4])
    assert gpu.to_list(gpu.div(a, b)) == [5.0, 5.0, 6.0, 5.0]


def test_neg():
    gpu.init_backend()
    a = gpu.tensor([1.0, -2.0, 3.0, -4.0], shape=[4])
    assert gpu.to_list(gpu.neg(a)) == [-1.0, 2.0, -3.0, 4.0]


def test_relu():
    gpu.init_backend()
    a = gpu.tensor([-1.0, 0.0, 3.0, -4.0], shape=[4])
    assert gpu.to_list(gpu.relu(a)) == [0.0, 0.0, 3.0, 0.0]


def test_exp():
    gpu.init_backend()
    import math
    a = gpu.tensor([0.0, 1.0], shape=[2])
    result = gpu.to_list(gpu.exp(a))
    assert abs(result[0] - 1.0) < 1e-6
    assert abs(result[1] - math.e) < 1e-5


def test_log():
    gpu.init_backend()
    import math
    a = gpu.tensor([1.0, math.e], shape=[2])
    result = gpu.to_list(gpu.log(a))
    assert abs(result[0] - 0.0) < 1e-6
    assert abs(result[1] - 1.0) < 1e-5


def test_sqrt():
    gpu.init_backend()
    a = gpu.tensor([4.0, 9.0, 16.0, 25.0], shape=[4])
    assert gpu.to_list(gpu.sqrt(a)) == [2.0, 3.0, 4.0, 5.0]


def test_matmul_2x2():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = gpu.matmul(a, b)
    assert gpu.to_list(c) == [19.0, 22.0, 43.0, 50.0]
    assert gpu.shape(c) == [2, 2]


def test_matmul_non_square():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = gpu.tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape=[3, 2])
    c = gpu.matmul(a, b)
    assert gpu.to_list(c) == [58.0, 64.0, 139.0, 154.0]
    assert gpu.shape(c) == [2, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_compute.py -v 2>&1`
Expected: New tests FAIL (functions don't exist)

- [ ] **Step 3: Add Python bindings for all ops**

In `crates/python/src/lib.rs`, replace the `add` function and add all new ops. Remove the `ADD_PIPELINE` global and `get_or_create_add_pipeline` since the `KernelRegistry` in `ops.rs` handles caching now.

Replace everything from `/// Global add pipeline` through the end of the `add` function with:

```rust
/// Helper: run a binary op on two tensor IDs.
/// Carefully manages lock ordering to avoid holding TENSORS during GPU dispatch.
fn binary_op_py(a_id: u64, b_id: u64, op: fn(&applegpu_core::device::Device, &Tensor, &Tensor) -> applegpu_core::error::Result<Tensor>) -> PyResult<u64> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Brief lock to validate inputs exist and call op.
    // Note: op() internally uses KernelRegistry which acquires its own lock,
    // but KernelRegistry releases its lock before GPU dispatch (returns Arc<ComputePipeline>),
    // so this is safe. The TENSORS lock IS held during dispatch here, but since
    // KernelRegistry no longer holds its lock during dispatch, there is no deadlock.
    // TODO: For true parallelism, restructure to release TENSORS before dispatch.
    let out = {
        let tensors = TENSORS.lock().unwrap();
        let a = tensors.get(&a_id).ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", a_id)))?;
        let b = tensors.get(&b_id).ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", b_id)))?;
        op(&runtime.device, a, b).map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    let id = out.meta.id;
    TENSORS.lock().unwrap().insert(id, out);
    Ok(id)
}

/// Helper: run a unary op on a tensor ID.
fn unary_op_py(input_id: u64, op: fn(&applegpu_core::device::Device, &Tensor) -> applegpu_core::error::Result<Tensor>) -> PyResult<u64> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out = {
        let tensors = TENSORS.lock().unwrap();
        let input = tensors.get(&input_id).ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", input_id)))?;
        op(&runtime.device, input).map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    let id = out.meta.id;
    TENSORS.lock().unwrap().insert(id, out);
    Ok(id)
}

#[pyfunction]
fn add(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::add) }

#[pyfunction]
fn sub(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::sub) }

#[pyfunction]
fn mul(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::mul) }

#[pyfunction]
fn div(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::div) }

#[pyfunction]
fn neg(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::neg) }

#[pyfunction]
fn relu(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::relu) }

#[pyfunction]
fn exp(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::exp) }

#[pyfunction]
fn log(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::log) }

#[pyfunction]
fn sqrt(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::sqrt) }

#[pyfunction]
fn matmul(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::matmul) }
```

Also update the `#[pymodule]` block to register all new functions:

```rust
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
```

And remove the `once_cell` import and the `ADD_PIPELINE` static.

- [ ] **Step 4: Update Python __init__.py**

```python
"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

from applegpu_runtime.applegpu_runtime import (
    version,
    init_backend,
    device_name,
    dtype_size,
    tensor,
    to_list,
    shape,
    add,
    sub,
    mul,
    div,
    neg,
    relu,
    exp,
    log,
    sqrt,
    matmul,
)

__version__ = version()
__all__ = [
    "version",
    "init_backend",
    "device_name",
    "dtype_size",
    "tensor",
    "to_list",
    "shape",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "relu",
    "exp",
    "log",
    "sqrt",
    "matmul",
]
```

- [ ] **Step 5: Rebuild and run all tests**

Run: `uv run maturin develop && uv run pytest -v 2>&1`
Expected: All Python tests pass (13 existing + 10 new = 23 total)

- [ ] **Step 6: Commit**

```bash
git add crates/python/src/lib.rs python/applegpu_runtime/__init__.py python/tests/test_compute.py
git commit -m "feat: expose sub/mul/div/neg/relu/exp/log/sqrt/matmul to Python"
```

---

### Task 7: End-to-end verification and push

- [ ] **Step 1: Run full test suite from clean**

Run: `make clean && make test 2>&1`
Expected: All tests pass across all three layers

- [ ] **Step 2: Update backlog**

Mark Phase 3 items as complete in `docs/BACKLOG.md`.

- [ ] **Step 3: Push**

```bash
git push origin main
```
