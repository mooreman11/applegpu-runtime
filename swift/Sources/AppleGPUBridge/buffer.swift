import Foundation
import Metal

/// Wraps an MTLBuffer with C ABI lifecycle management.
final class GPUBuffer {
    let buffer: MTLBuffer

    init?(device: MTLDevice, length: Int) {
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
