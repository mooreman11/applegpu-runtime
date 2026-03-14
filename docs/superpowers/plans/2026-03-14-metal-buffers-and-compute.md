# Metal Buffers and GPU Compute Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Metal buffer allocation, command queue infrastructure, and a first GPU compute operation (`gpu.add(a, b)`) that executes a real Metal compute kernel end-to-end from Python.

**Architecture:** The Swift bridge gains buffer management (create/read/write via `MTLBuffer`), command queue/encoder setup, and Metal compute pipeline compilation from source. Rust wraps these in safe abstractions with a `Tensor` type that owns both metadata and a GPU buffer handle. A hardcoded element-wise add kernel proves the full pipeline works. Python exposes `gpu.add()` and `gpu.tensor()`.

**Tech Stack:** Rust (FFI, RAII), Swift (Metal, MTLBuffer, MTLComputePipelineState, MTLCommandQueue), Python (PyO3), Metal Shading Language (MSL)

---

## File Structure

### New Files — Swift Layer
- `swift/Sources/AppleGPUBridge/buffer.swift` — MTLBuffer creation/read/write via C ABI
- `swift/Sources/AppleGPUBridge/compute.swift` — Command queue, compute pipeline, kernel dispatch via C ABI
- `swift/Tests/AppleGPUBridgeTests/BufferTests.swift` — Buffer lifecycle tests
- `swift/Tests/AppleGPUBridgeTests/ComputeTests.swift` — Compute pipeline tests

Note: The MSL kernel source is inlined as a string constant in Rust (`compute.rs`) and Swift tests, not loaded from a `.metal` file. No changes to `Package.swift` are needed.

### New Files — Rust Layer
- `crates/core/src/buffer.rs` — Safe Rust wrapper around Metal buffer handles (with inline unit tests)
- `crates/core/src/compute.rs` — Command queue and kernel dispatch abstractions (with inline unit tests)

### New Files — Python Layer
- `python/tests/test_tensor.py` — Tensor creation and data round-trip tests
- `python/tests/test_compute.py` — gpu.add() end-to-end tests

### Modified Files
- `swift/Sources/AppleGPUBridge/include/bridge.h` — Add buffer and compute C ABI declarations
- `swift/Sources/AppleGPUBridge/bridge.swift` — Expose device handle for buffer/compute use
- `swift/Package.swift` — Add .metal file resource handling if needed
- `crates/core/src/ffi.rs` — Add buffer and compute extern "C" declarations
- `crates/core/src/tensor.rs` — Upgrade TensorMeta → Tensor with buffer handle
- `crates/core/src/device.rs` — Expose raw handle for buffer/compute creation
- `crates/core/src/error.rs` — Add buffer/compute error variants
- `crates/core/src/lib.rs` — Add buffer and compute modules
- `crates/python/src/lib.rs` — Expose tensor(), add() to Python
- `python/applegpu_runtime/__init__.py` — Export new functions

---

## Chunk 1: Metal Buffer Management (Swift + Rust)

### Task 1: Add buffer C ABI to bridge.h and implement in Swift

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h`
- Create: `swift/Sources/AppleGPUBridge/buffer.swift`
- Modify: `swift/Sources/AppleGPUBridge/bridge.swift`

- [ ] **Step 1: Expose device handle in bridge.swift for internal use**

In `swift/Sources/AppleGPUBridge/bridge.swift`, add a helper to retrieve the `MTLDevice` from an opaque handle. Add after the existing C ABI exports:

```swift
/// Internal helper: extract MTLDevice from opaque handle.
/// Used by buffer.swift and compute.swift.
func getGPUDevice(from ptr: UnsafeRawPointer) -> GPUDevice {
    Unmanaged<GPUDevice>.fromOpaque(ptr).takeUnretainedValue()
}
```

- [ ] **Step 2: Add buffer declarations to bridge.h**

Append to `swift/Sources/AppleGPUBridge/include/bridge.h` before the `#endif`:

```c
// Opaque handle to a GPU buffer
typedef struct GPUBufferHandle GPUBufferHandle;

// Buffer lifecycle
GPUBufferHandle* gpu_bridge_create_buffer(const GPUDeviceHandle* device, uint64_t size_bytes);
GPUBufferHandle* gpu_bridge_create_buffer_with_data(const GPUDeviceHandle* device, const void* data, uint64_t size_bytes);
void gpu_bridge_destroy_buffer(GPUBufferHandle* buffer);

// Buffer data access
void* gpu_bridge_buffer_contents(const GPUBufferHandle* buffer);
uint64_t gpu_bridge_buffer_length(const GPUBufferHandle* buffer);
```

- [ ] **Step 3: Implement buffer.swift**

Create `swift/Sources/AppleGPUBridge/buffer.swift`:

```swift
import Foundation
import Metal

/// Wraps an MTLBuffer with C ABI lifecycle management.
final class GPUBuffer {
    let buffer: MTLBuffer

    init?(device: MTLDevice, length: Int) {
        // Use .storageModeShared for zero-copy CPU/GPU access
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else { return nil }
        self.buffer = buffer
    }

    init?(device: MTLDevice, bytes: UnsafeRawPointer, length: Int) {
        guard let buffer = device.makeBuffer(bytes: bytes, length: length, options: .storageModeShared) else { return nil }
        self.buffer = buffer
    }
}

// MARK: - C ABI exports

@_cdecl("gpu_bridge_create_buffer")
public func gpuBridgeCreateBuffer(
    _ devicePtr: UnsafeRawPointer?,
    _ sizeBytes: UInt64
) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr else { return nil }
    let gpuDevice = getGPUDevice(from: devicePtr)
    guard let buf = GPUBuffer(device: gpuDevice.device, length: Int(sizeBytes)) else { return nil }
    return Unmanaged.passRetained(buf).toOpaque()
}

@_cdecl("gpu_bridge_create_buffer_with_data")
public func gpuBridgeCreateBufferWithData(
    _ devicePtr: UnsafeRawPointer?,
    _ data: UnsafeRawPointer?,
    _ sizeBytes: UInt64
) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr, let data = data else { return nil }
    let gpuDevice = getGPUDevice(from: devicePtr)
    guard let buf = GPUBuffer(device: gpuDevice.device, bytes: data, length: Int(sizeBytes)) else { return nil }
    return Unmanaged.passRetained(buf).toOpaque()
}

@_cdecl("gpu_bridge_destroy_buffer")
public func gpuBridgeDestroyBuffer(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    Unmanaged<GPUBuffer>.fromOpaque(ptr).release()
}

@_cdecl("gpu_bridge_buffer_contents")
public func gpuBridgeBufferContents(_ ptr: UnsafeRawPointer?) -> UnsafeMutableRawPointer? {
    guard let ptr = ptr else { return nil }
    let buf = Unmanaged<GPUBuffer>.fromOpaque(ptr).takeUnretainedValue()
    return buf.buffer.contents()
}

@_cdecl("gpu_bridge_buffer_length")
public func gpuBridgeBufferLength(_ ptr: UnsafeRawPointer?) -> UInt64 {
    guard let ptr = ptr else { return 0 }
    let buf = Unmanaged<GPUBuffer>.fromOpaque(ptr).takeUnretainedValue()
    return UInt64(buf.buffer.length)
}
```

- [ ] **Step 4: Run Swift build to verify compilation**

Run: `cd swift && swift build 2>&1`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add swift/Sources/AppleGPUBridge/bridge.swift swift/Sources/AppleGPUBridge/buffer.swift swift/Sources/AppleGPUBridge/include/bridge.h
git commit -m "feat: add Metal buffer C ABI (create, read, write, destroy)"
```

---

### Task 2: Swift buffer tests

**Files:**
- Create: `swift/Tests/AppleGPUBridgeTests/BufferTests.swift`

- [ ] **Step 1: Write buffer tests**

```swift
import Testing
@testable import AppleGPUBridge

@Test func bufferCreateAndDestroy() {
    let devicePtr = gpuBridgeCreateDevice()
    #expect(devicePtr != nil)
    guard let devicePtr = devicePtr else { return }

    let bufPtr = gpuBridgeCreateBuffer(devicePtr, 1024)
    #expect(bufPtr != nil)

    if let bufPtr = bufPtr {
        let length = gpuBridgeBufferLength(bufPtr)
        #expect(length == 1024)
        gpuBridgeDestroyBuffer(bufPtr)
    }

    gpuBridgeDestroyDevice(devicePtr)
}

@Test func bufferCreateWithData() {
    let devicePtr = gpuBridgeCreateDevice()
    guard let devicePtr = devicePtr else { return }

    var data: [Float] = [1.0, 2.0, 3.0, 4.0]
    let sizeBytes = UInt64(data.count * MemoryLayout<Float>.size)

    let bufPtr = data.withUnsafeBytes { rawBuf in
        gpuBridgeCreateBufferWithData(devicePtr, rawBuf.baseAddress, sizeBytes)
    }
    #expect(bufPtr != nil)

    if let bufPtr = bufPtr {
        let contents = gpuBridgeBufferContents(bufPtr)!
        let floatPtr = contents.bindMemory(to: Float.self, capacity: 4)
        #expect(floatPtr[0] == 1.0)
        #expect(floatPtr[1] == 2.0)
        #expect(floatPtr[2] == 3.0)
        #expect(floatPtr[3] == 4.0)

        gpuBridgeDestroyBuffer(bufPtr)
    }

    gpuBridgeDestroyDevice(devicePtr)
}

@Test func bufferContentsReadWrite() {
    let devicePtr = gpuBridgeCreateDevice()
    guard let devicePtr = devicePtr else { return }

    let bufPtr = gpuBridgeCreateBuffer(devicePtr, UInt64(4 * MemoryLayout<Float>.size))
    guard let bufPtr = bufPtr else { return }

    // Write data via contents pointer
    let contents = gpuBridgeBufferContents(bufPtr)!
    let floatPtr = contents.bindMemory(to: Float.self, capacity: 4)
    floatPtr[0] = 10.0
    floatPtr[1] = 20.0
    floatPtr[2] = 30.0
    floatPtr[3] = 40.0

    // Read back
    #expect(floatPtr[0] == 10.0)
    #expect(floatPtr[3] == 40.0)

    gpuBridgeDestroyBuffer(bufPtr)
    gpuBridgeDestroyDevice(devicePtr)
}
```

- [ ] **Step 2: Run Swift tests**

Run: `cd swift && swift test 2>&1`
Expected: All 4 tests pass (1 existing + 3 new)

- [ ] **Step 3: Commit**

```bash
git add swift/Tests/AppleGPUBridgeTests/BufferTests.swift
git commit -m "test: add Swift buffer lifecycle and data round-trip tests"
```

---

### Task 3: Rust FFI declarations and safe buffer wrapper

**Files:**
- Modify: `crates/core/src/ffi.rs`
- Create: `crates/core/src/buffer.rs`
- Modify: `crates/core/src/device.rs`
- Modify: `crates/core/src/error.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Add buffer error variant**

In `crates/core/src/error.rs`, add to the `GpuError` enum:

```rust
    /// Buffer allocation failed
    BufferAllocationFailed(usize),
    /// Compute operation failed
    ComputeFailed(String),
```

And update the `Display` impl:

```rust
            GpuError::BufferAllocationFailed(size) => write!(f, "Failed to allocate GPU buffer of {} bytes", size),
            GpuError::ComputeFailed(msg) => write!(f, "Compute failed: {}", msg),
```

- [ ] **Step 2: Expose raw device handle**

In `crates/core/src/device.rs`, add a method to `Device`:

```rust
    /// Get the raw FFI handle. Used internally by buffer and compute modules.
    pub(crate) fn raw_handle(&self) -> *const ffi::GPUDeviceHandle {
        self.handle as *const _
    }
```

- [ ] **Step 3: Add buffer FFI declarations to ffi.rs**

Append to `crates/core/src/ffi.rs`:

```rust
/// Opaque handle to a GPU buffer from the Swift side.
#[repr(C)]
pub struct GPUBufferHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_buffer(
        device: *const GPUDeviceHandle,
        size_bytes: u64,
    ) -> *mut GPUBufferHandle;

    pub fn gpu_bridge_create_buffer_with_data(
        device: *const GPUDeviceHandle,
        data: *const std::ffi::c_void,
        size_bytes: u64,
    ) -> *mut GPUBufferHandle;

    pub fn gpu_bridge_destroy_buffer(buffer: *mut GPUBufferHandle);

    pub fn gpu_bridge_buffer_contents(buffer: *const GPUBufferHandle) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_buffer_length(buffer: *const GPUBufferHandle) -> u64;
}
```

- [ ] **Step 4: Create buffer.rs with safe wrapper**

Create `crates/core/src/buffer.rs`:

```rust
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;

/// A Metal GPU buffer. Wraps an MTLBuffer via the Swift bridge.
/// Uses storageModeShared for zero-copy CPU/GPU access.
pub struct Buffer {
    handle: *mut ffi::GPUBufferHandle,
    len: usize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Allocate an empty buffer of `size_bytes` on the GPU.
    pub fn new(device: &Device, size_bytes: usize) -> Result<Self> {
        let handle = unsafe {
            ffi::gpu_bridge_create_buffer(device.raw_handle(), size_bytes as u64)
        };
        if handle.is_null() {
            Err(GpuError::BufferAllocationFailed(size_bytes))
        } else {
            Ok(Buffer {
                handle,
                len: size_bytes,
            })
        }
    }

    /// Create a buffer initialized with data from a byte slice.
    pub fn from_bytes(device: &Device, data: &[u8]) -> Result<Self> {
        let handle = unsafe {
            ffi::gpu_bridge_create_buffer_with_data(
                device.raw_handle(),
                data.as_ptr() as *const _,
                data.len() as u64,
            )
        };
        if handle.is_null() {
            Err(GpuError::BufferAllocationFailed(data.len()))
        } else {
            Ok(Buffer {
                handle,
                len: data.len(),
            })
        }
    }

    /// Get a raw pointer to the buffer contents (shared memory).
    pub fn contents(&self) -> *mut u8 {
        unsafe { ffi::gpu_bridge_buffer_contents(self.handle) as *mut u8 }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Read buffer contents into a Vec<u8>.
    pub fn read_bytes(&self) -> Vec<u8> {
        let ptr = self.contents();
        let mut data = vec![0u8; self.len];
        unsafe { std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), self.len) };
        data
    }

    /// Read buffer contents as a slice of T (zero-copy view).
    /// # Safety
    /// Caller must ensure the buffer contains valid data of type T
    /// and that the buffer length is a multiple of size_of::<T>().
    pub unsafe fn as_slice<T: Copy>(&self) -> &[T] {
        let count = self.len / std::mem::size_of::<T>();
        std::slice::from_raw_parts(self.contents() as *const T, count)
    }

    /// Get the raw FFI handle. Used internally by compute module.
    pub(crate) fn raw_handle(&self) -> *mut ffi::GPUBufferHandle {
        self.handle
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { ffi::gpu_bridge_destroy_buffer(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_alloc_and_length() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return, // No GPU
        };
        let buf = Buffer::new(&device, 1024).unwrap();
        assert_eq!(buf.len(), 1024);
    }

    #[test]
    fn buffer_from_bytes_roundtrip() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        let buf = Buffer::from_bytes(&device, bytes).unwrap();
        assert_eq!(buf.len(), 16);

        let result = unsafe { buf.as_slice::<f32>() };
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn buffer_write_via_contents() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let buf = Buffer::new(&device, 4 * std::mem::size_of::<f32>()).unwrap();
        let ptr = buf.contents() as *mut f32;
        unsafe {
            *ptr.add(0) = 10.0;
            *ptr.add(1) = 20.0;
            *ptr.add(2) = 30.0;
            *ptr.add(3) = 40.0;
        }
        let result = unsafe { buf.as_slice::<f32>() };
        assert_eq!(result, &[10.0, 20.0, 30.0, 40.0]);
    }
}
```

- [ ] **Step 5: Add buffer module to lib.rs**

Add `pub mod buffer;` to `crates/core/src/lib.rs`.

- [ ] **Step 6: Run Rust tests**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All tests pass including 3 new buffer tests

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/buffer.rs crates/core/src/ffi.rs crates/core/src/device.rs crates/core/src/error.rs crates/core/src/lib.rs
git commit -m "feat: add Metal buffer FFI and safe Rust Buffer wrapper"
```

---

## Chunk 2: Metal Compute Pipeline (Swift + Rust)

### Task 4: Add compute C ABI to Swift

**Files:**
- Create: `swift/Sources/AppleGPUBridge/compute.swift`
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h`

- [ ] **Step 1: Add compute declarations to bridge.h**

Append to `swift/Sources/AppleGPUBridge/include/bridge.h` before the `#endif`:

```c
// Opaque handle to a compute context (command queue + pipeline)
typedef struct GPUComputeHandle GPUComputeHandle;

// Compute lifecycle
GPUComputeHandle* gpu_bridge_create_compute(const GPUDeviceHandle* device, const char* kernel_source, const char* function_name);
void gpu_bridge_destroy_compute(GPUComputeHandle* compute);

// Execute element-wise operation: out = op(a, b)
// Returns 0 on success, -1 on failure.
int32_t gpu_bridge_compute_elementwise(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_out,
    uint64_t element_count
);
```

- [ ] **Step 2: Implement compute.swift**

Create `swift/Sources/AppleGPUBridge/compute.swift`:

```swift
import Foundation
import Metal

/// Wraps a Metal compute pipeline with command queue.
final class GPUCompute {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLComputePipelineState

    init?(device: MTLDevice, kernelSource: String, functionName: String) {
        self.device = device

        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue

        do {
            let library = try device.makeLibrary(source: kernelSource, options: nil)
            guard let function = library.makeFunction(name: functionName) else { return nil }
            self.pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            return nil
        }
    }

    func dispatchElementwise(bufA: MTLBuffer, bufB: MTLBuffer, bufOut: MTLBuffer, count: Int) -> Bool {
        if count == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 3)

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
}

// MARK: - C ABI exports

@_cdecl("gpu_bridge_create_compute")
public func gpuBridgeCreateCompute(
    _ devicePtr: UnsafeRawPointer?,
    _ kernelSource: UnsafePointer<CChar>?,
    _ functionName: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr,
          let kernelSource = kernelSource,
          let functionName = functionName else { return nil }

    let gpuDevice = getGPUDevice(from: devicePtr)
    let source = String(cString: kernelSource)
    let name = String(cString: functionName)

    guard let compute = GPUCompute(device: gpuDevice.device, kernelSource: source, functionName: name) else {
        return nil
    }
    return Unmanaged.passRetained(compute).toOpaque()
}

@_cdecl("gpu_bridge_destroy_compute")
public func gpuBridgeDestroyCompute(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    Unmanaged<GPUCompute>.fromOpaque(ptr).release()
}

@_cdecl("gpu_bridge_compute_elementwise")
public func gpuBridgeComputeElementwise(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> Int32 {
    guard let computePtr = computePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchElementwise(
        bufA: bufA.buffer,
        bufB: bufB.buffer,
        bufOut: bufOut.buffer,
        count: Int(elementCount)
    )
    return success ? 0 : -1
}
```

- [ ] **Step 3: Run Swift build**

Run: `cd swift && swift build 2>&1`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add swift/Sources/AppleGPUBridge/compute.swift swift/Sources/AppleGPUBridge/include/bridge.h
git commit -m "feat: add Metal compute pipeline C ABI (create, dispatch, destroy)"
```

---

### Task 5: Swift compute test with inline MSL kernel

**Files:**
- Create: `swift/Tests/AppleGPUBridgeTests/ComputeTests.swift`

- [ ] **Step 1: Write compute test using element-wise add kernel**

```swift
import Testing
@testable import AppleGPUBridge

let addKernelSource = """
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        out[id] = a[id] + b[id];
    }
}
"""

@Test func computeElementwiseAdd() {
    let devicePtr = gpuBridgeCreateDevice()
    guard let devicePtr = devicePtr else { return }

    // Create compute pipeline
    let computePtr = addKernelSource.withCString { src in
        "elementwise_add".withCString { name in
            gpuBridgeCreateCompute(devicePtr, src, name)
        }
    }
    #expect(computePtr != nil)
    guard let computePtr = computePtr else { return }

    // Create buffers
    let count = 4
    let sizeBytes = UInt64(count * MemoryLayout<Float>.size)

    var dataA: [Float] = [1.0, 2.0, 3.0, 4.0]
    var dataB: [Float] = [10.0, 20.0, 30.0, 40.0]

    let bufA = dataA.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, sizeBytes) }
    let bufB = dataB.withUnsafeBytes { gpuBridgeCreateBufferWithData(devicePtr, $0.baseAddress, sizeBytes) }
    let bufOut = gpuBridgeCreateBuffer(devicePtr, sizeBytes)

    guard let bufA = bufA, let bufB = bufB, let bufOut = bufOut else {
        #expect(Bool(false), "Failed to create buffers")
        return
    }

    // Dispatch
    let result = gpuBridgeComputeElementwise(computePtr, bufA, bufB, bufOut, UInt64(count))
    #expect(result == 0)

    // Read result
    let outContents = gpuBridgeBufferContents(bufOut)!
    let outFloats = outContents.bindMemory(to: Float.self, capacity: count)
    #expect(outFloats[0] == 11.0)
    #expect(outFloats[1] == 22.0)
    #expect(outFloats[2] == 33.0)
    #expect(outFloats[3] == 44.0)

    // Cleanup
    gpuBridgeDestroyBuffer(bufA)
    gpuBridgeDestroyBuffer(bufB)
    gpuBridgeDestroyBuffer(bufOut)
    gpuBridgeDestroyCompute(computePtr)
    gpuBridgeDestroyDevice(devicePtr)
}
```

- [ ] **Step 2: Run Swift tests**

Run: `cd swift && swift test 2>&1`
Expected: All tests pass including `computeElementwiseAdd`

- [ ] **Step 3: Commit**

```bash
git add swift/Tests/AppleGPUBridgeTests/ComputeTests.swift
git commit -m "test: add Metal compute element-wise add test with inline MSL"
```

---

### Task 6: Rust compute FFI and safe wrapper

**Files:**
- Modify: `crates/core/src/ffi.rs`
- Create: `crates/core/src/compute.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Add compute FFI declarations**

Append to `crates/core/src/ffi.rs`:

```rust
/// Opaque handle to a compute context from the Swift side.
#[repr(C)]
pub struct GPUComputeHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_compute(
        device: *const GPUDeviceHandle,
        kernel_source: *const std::ffi::c_char,
        function_name: *const std::ffi::c_char,
    ) -> *mut GPUComputeHandle;

    pub fn gpu_bridge_destroy_compute(compute: *mut GPUComputeHandle);

    pub fn gpu_bridge_compute_elementwise(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;
}
```

- [ ] **Step 2: Create compute.rs**

```rust
use std::ffi::CString;

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;

/// Metal Shading Language source for element-wise add.
const ADD_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        out[id] = a[id] + b[id];
    }
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
            ffi::gpu_bridge_create_compute(
                device.raw_handle(),
                source.as_ptr(),
                name.as_ptr(),
            )
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

    /// Create a compute pipeline for element-wise add.
    pub fn add(device: &Device) -> Result<Self> {
        Self::new(device, ADD_KERNEL_SOURCE, "elementwise_add")
    }

    /// Dispatch element-wise operation: out = op(a, b).
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
        if result == 0 {
            Ok(())
        } else {
            Err(GpuError::ComputeFailed(
                "Kernel dispatch failed".to_string(),
            ))
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { ffi::gpu_bridge_destroy_compute(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_pipeline_creates() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let pipeline = ComputePipeline::add(&device);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn elementwise_add_computes_correctly() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let pipeline = ComputePipeline::add(&device).unwrap();

        let a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let b_data: [f32; 4] = [10.0, 20.0, 30.0, 40.0];

        let bytes_a = unsafe {
            std::slice::from_raw_parts(a_data.as_ptr() as *const u8, 16)
        };
        let bytes_b = unsafe {
            std::slice::from_raw_parts(b_data.as_ptr() as *const u8, 16)
        };

        let buf_a = Buffer::from_bytes(&device, bytes_a).unwrap();
        let buf_b = Buffer::from_bytes(&device, bytes_b).unwrap();
        let buf_out = Buffer::new(&device, 16).unwrap();

        pipeline.dispatch_elementwise(&buf_a, &buf_b, &buf_out, 4).unwrap();

        let result = unsafe { buf_out.as_slice::<f32>() };
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }
}
```

- [ ] **Step 3: Add compute module to lib.rs**

Add `pub mod compute;` to `crates/core/src/lib.rs`.

- [ ] **Step 4: Run Rust tests**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All tests pass including `elementwise_add_computes_correctly`

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/compute.rs crates/core/src/ffi.rs crates/core/src/lib.rs
git commit -m "feat: add Rust ComputePipeline with element-wise add kernel"
```

---

## Chunk 3: Tensor with Data + Python API

### Task 7: Upgrade tensor module with buffer-backed Tensor

**Files:**
- Modify: `crates/core/src/tensor.rs`

- [ ] **Step 1: Add Tensor struct that owns a Buffer**

Append to `crates/core/src/tensor.rs` (keep all existing types):

```rust
use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::Result;
use std::sync::atomic::{AtomicU64, Ordering};

static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A tensor backed by a Metal GPU buffer.
pub struct Tensor {
    pub meta: TensorMeta,
    pub(crate) buffer: Buffer,
}

impl Tensor {
    /// Create a tensor from f32 data.
    pub fn from_f32(device: &Device, shape: Vec<usize>, data: &[f32]) -> Result<Self> {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(crate::error::GpuError::InvalidTensor(format!(
                "Shape {:?} expects {} elements but got {}",
                shape, expected, data.len()
            )));
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        let buffer = Buffer::from_bytes(device, bytes)?;
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Tensor {
            meta: TensorMeta {
                id,
                shape: Shape::new(shape),
                dtype: DType::Float32,
                location: TensorLocation::Shared,
            },
            buffer,
        })
    }

    /// Create an uninitialized tensor (for output buffers).
    pub fn empty_f32(device: &Device, shape: Vec<usize>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * std::mem::size_of::<f32>();
        let buffer = Buffer::new(device, size_bytes)?;
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Tensor {
            meta: TensorMeta {
                id,
                shape: Shape::new(shape),
                dtype: DType::Float32,
                location: TensorLocation::Shared,
            },
            buffer,
        })
    }

    /// Read tensor data as f32 slice (zero-copy).
    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe { self.buffer.as_slice::<f32>() }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.meta.shape.numel()
    }
}
```

- [ ] **Step 2: Add tensor with data tests**

Append to the existing `#[cfg(test)] mod tests` in `tensor.rs`:

```rust
    #[test]
    fn tensor_from_f32_roundtrip() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = super::Tensor::from_f32(&device, vec![2, 3], &data).unwrap();
        assert_eq!(t.meta.shape.dims(), &[2, 3]);
        assert_eq!(t.meta.dtype, super::DType::Float32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tensor_shape_mismatch_errors() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = vec![1.0f32, 2.0, 3.0];
        let result = super::Tensor::from_f32(&device, vec![2, 3], &data);
        assert!(result.is_err());
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/tensor.rs
git commit -m "feat: add buffer-backed Tensor with from_f32 and zero-copy read"
```

---

### Task 8: Expose tensor and add() to Python

**Files:**
- Modify: `crates/python/src/lib.rs`
- Modify: `python/applegpu_runtime/__init__.py`
- Create: `python/tests/test_tensor.py`
- Create: `python/tests/test_compute.py`

- [ ] **Step 1: Write failing Python tests**

Create `python/tests/test_tensor.py`:

```python
import applegpu_runtime as gpu


def test_tensor_create():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    assert t is not None


def test_tensor_to_list():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    result = gpu.to_list(t)
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_tensor_shape():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    assert gpu.shape(t) == [2, 3]
```

Create `python/tests/test_compute.py`:

```python
import applegpu_runtime as gpu


def test_add_basic():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    c = gpu.add(a, b)
    result = gpu.to_list(c)
    assert result == [11.0, 22.0, 33.0, 44.0]


def test_add_2d():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = gpu.add(a, b)
    result = gpu.to_list(c)
    assert result == [6.0, 8.0, 10.0, 12.0]
    assert gpu.shape(c) == [2, 2]


def test_add_large():
    gpu.init_backend()
    n = 10000
    a = gpu.tensor([1.0] * n, shape=[n])
    b = gpu.tensor([2.0] * n, shape=[n])
    c = gpu.add(a, b)
    result = gpu.to_list(c)
    assert all(x == 3.0 for x in result)
    assert len(result) == n
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_tensor.py python/tests/test_compute.py -v 2>&1`
Expected: FAIL (functions don't exist)

- [ ] **Step 3: Implement PyO3 bindings for tensor and add**

Replace `crates/python/src/lib.rs` with:

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Mutex;

use applegpu_core::compute::ComputePipeline;
use applegpu_core::tensor::Tensor;

/// Global tensor storage. Tensors are stored by ID and accessed by opaque handle.
static TENSORS: once_cell::sync::Lazy<Mutex<HashMap<u64, Tensor>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Global add pipeline (lazy-initialized).
static ADD_PIPELINE: once_cell::sync::Lazy<Mutex<Option<ComputePipeline>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(None));

fn get_or_create_add_pipeline() -> PyResult<()> {
    let mut pipeline = ADD_PIPELINE.lock().unwrap();
    if pipeline.is_none() {
        let runtime = applegpu_core::backend::get_runtime()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let p = ComputePipeline::add(&runtime.device)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        *pipeline = Some(p);
    }
    Ok(())
}

/// Returns the library version.
#[pyfunction]
fn version() -> &'static str {
    applegpu_core::version()
}

/// Initialize the GPU backend. Returns dict with backend info.
#[pyfunction]
fn init_backend() -> PyResult<HashMap<String, String>> {
    let runtime = applegpu_core::backend::init_backend()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut info = HashMap::new();
    info.insert(
        "backend".to_string(),
        format!("{:?}", runtime.backend).to_lowercase(),
    );
    info.insert("device".to_string(), runtime.device.name());
    Ok(info)
}

/// Get the Metal GPU device name. Requires init_backend() first.
#[pyfunction]
fn device_name() -> PyResult<String> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(runtime.device.name())
}

/// Get the size in bytes of a dtype by name.
#[pyfunction]
fn dtype_size(name: &str) -> PyResult<usize> {
    use applegpu_core::tensor::DType;
    let dt = match name {
        "float16" | "f16" => DType::Float16,
        "float32" | "f32" => DType::Float32,
        "float64" | "f64" => DType::Float64,
        "bfloat16" | "bf16" => DType::BFloat16,
        "int8" | "i8" => DType::Int8,
        "int16" | "i16" => DType::Int16,
        "int32" | "i32" => DType::Int32,
        "int64" | "i64" => DType::Int64,
        "uint8" | "u8" => DType::UInt8,
        "uint32" | "u32" => DType::UInt32,
        "bool" => DType::Bool,
        _ => return Err(PyValueError::new_err(format!("Unknown dtype: {}", name))),
    };
    Ok(dt.size_bytes())
}

/// Create a tensor from a list of f32 values and a shape.
/// Returns an opaque tensor handle (u64 ID).
#[pyfunction]
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<u64> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t = Tensor::from_f32(&runtime.device, shape, &data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    TENSORS.lock().unwrap().insert(id, t);
    Ok(id)
}

/// Read tensor data as a flat list of f32 values.
#[pyfunction]
fn to_list(tensor_id: u64) -> PyResult<Vec<f32>> {
    let tensors = TENSORS.lock().unwrap();
    let t = tensors
        .get(&tensor_id)
        .ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", tensor_id)))?;
    Ok(t.as_f32_slice().to_vec())
}

/// Get the shape of a tensor as a list.
#[pyfunction]
fn shape(tensor_id: u64) -> PyResult<Vec<usize>> {
    let tensors = TENSORS.lock().unwrap();
    let t = tensors
        .get(&tensor_id)
        .ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", tensor_id)))?;
    Ok(t.meta.shape.dims().to_vec())
}

/// Element-wise add: c = a + b. Returns tensor handle for result.
#[pyfunction]
fn add(a_id: u64, b_id: u64) -> PyResult<u64> {
    get_or_create_add_pipeline()?;

    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Step 1: Lock TENSORS briefly to validate inputs and create output buffer.
    // Extract what we need, then drop the lock before compute dispatch.
    let (out_shape, numel) = {
        let tensors = TENSORS.lock().unwrap();
        let a = tensors
            .get(&a_id)
            .ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", a_id)))?;
        let b = tensors
            .get(&b_id)
            .ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", b_id)))?;

        if a.meta.shape != b.meta.shape {
            return Err(PyValueError::new_err(format!(
                "Shape mismatch: {:?} vs {:?}",
                a.meta.shape.dims(),
                b.meta.shape.dims()
            )));
        }
        (a.meta.shape.dims().to_vec(), a.numel())
    }; // TENSORS lock dropped here

    let out = Tensor::empty_f32(&runtime.device, out_shape)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Step 2: Lock TENSORS to get buffer refs, lock ADD_PIPELINE to dispatch.
    // Single lock ordering: TENSORS first, then ADD_PIPELINE.
    {
        let tensors = TENSORS.lock().unwrap();
        let a = tensors.get(&a_id).unwrap();
        let b = tensors.get(&b_id).unwrap();

        let pipeline = ADD_PIPELINE.lock().unwrap();
        let pipeline = pipeline.as_ref().unwrap();
        pipeline
            .dispatch_elementwise(&a.buffer, &b.buffer, &out.buffer, numel)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    } // Both locks dropped here

    // Step 3: Insert result into TENSORS.
    let id = out.meta.id;
    TENSORS.lock().unwrap().insert(id, out);
    Ok(id)
}

/// The Python module definition.
#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_size, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(to_list, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
```

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
]
```

- [ ] **Step 5: Add `once_cell` dependency to python crate**

In `crates/python/Cargo.toml`, add under `[dependencies]`:

```toml
once_cell = "1"
```

- [ ] **Step 6: Rebuild and run all tests**

Run: `uv run maturin develop && uv run pytest -v 2>&1`
Expected: All Python tests pass including tensor and compute tests

- [ ] **Step 7: Commit**

```bash
git add crates/python/src/lib.rs crates/python/Cargo.toml python/applegpu_runtime/__init__.py python/tests/test_tensor.py python/tests/test_compute.py
git commit -m "feat: expose tensor(), to_list(), shape(), and add() to Python"
```

---

### Task 9: End-to-end verification and push

- [ ] **Step 1: Run full test suite from clean**

Run: `make clean && make test 2>&1`
Expected: All tests pass across all three layers

- [ ] **Step 2: Commit any remaining changes and push**

```bash
git push origin main
```
