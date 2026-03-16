import Foundation
import Metal

/// Base class for all GPU buffers — provides the MTLBuffer handle.
class GPUBufferBase {
    let buffer: MTLBuffer
    init(buffer: MTLBuffer) { self.buffer = buffer }
}

/// Owned buffer — Metal allocated and owns the memory.
class GPUBuffer: GPUBufferBase {
    init?(device: MTLDevice, length: Int) {
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else { return nil }
        super.init(buffer: buffer)
    }

    init?(device: MTLDevice, bytes: UnsafeRawPointer, length: Int) {
        guard let buffer = device.makeBuffer(bytes: bytes, length: length, options: .storageModeShared) else { return nil }
        super.init(buffer: buffer)
    }
}

/// Borrowed buffer — Metal references external memory. Deallocator fires when released.
class GPUBufferNoCopy: GPUBufferBase {
    // No additional state — Metal's deallocator block handles cleanup
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
    Unmanaged<GPUBufferBase>.fromOpaque(ptr).release()
}

@_cdecl("gpu_bridge_buffer_contents")
public func gpuBridgeBufferContents(_ ptr: UnsafeRawPointer?) -> UnsafeMutableRawPointer? {
    guard let ptr = ptr else { return nil }
    let buf = Unmanaged<GPUBufferBase>.fromOpaque(ptr).takeUnretainedValue()
    return buf.buffer.contents()
}

@_cdecl("gpu_bridge_buffer_length")
public func gpuBridgeBufferLength(_ ptr: UnsafeRawPointer?) -> UInt64 {
    guard let ptr = ptr else { return 0 }
    let buf = Unmanaged<GPUBufferBase>.fromOpaque(ptr).takeUnretainedValue()
    return UInt64(buf.buffer.length)
}

@_cdecl("gpu_bridge_create_buffer_no_copy")
public func gpuBridgeCreateBufferNoCopy(
    _ devicePtr: UnsafeRawPointer?,
    _ dataPtr: UnsafeMutableRawPointer?,
    _ sizeBytes: UInt64,
    _ deallocator: (@convention(c) (UnsafeMutableRawPointer?, UInt64, UnsafeMutableRawPointer?) -> Void)?,
    _ deallocatorContext: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr, let dataPtr = dataPtr else { return nil }
    let gpuDevice = getGPUDevice(from: devicePtr)
    let length = Int(sizeBytes)

    guard let buffer = gpuDevice.device.makeBuffer(
        bytesNoCopy: dataPtr,
        length: length,
        options: .storageModeShared,
        deallocator: { ptr, len in
            deallocator?(ptr, UInt64(len), deallocatorContext)
        }
    ) else { return nil }

    let buf = GPUBufferNoCopy(buffer: buffer)
    return Unmanaged.passRetained(buf).toOpaque()
}
