import Foundation
import Metal

/// Internal class wrapping MTLDevice.
final class GPUDevice {
    let device: MTLDevice
    private var nameCString: [CChar]

    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        self.device = device
        self.nameCString = Array(device.name.utf8CString)
    }

    var namePtr: UnsafePointer<CChar> {
        nameCString.withUnsafeBufferPointer { $0.baseAddress! }
    }
}

// MARK: - C ABI exports

@_cdecl("gpu_bridge_create_device")
public func gpuBridgeCreateDevice() -> UnsafeMutableRawPointer? {
    guard let device = GPUDevice() else { return nil }
    return Unmanaged.passRetained(device).toOpaque()
}

@_cdecl("gpu_bridge_destroy_device")
public func gpuBridgeDestroyDevice(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    Unmanaged<GPUDevice>.fromOpaque(ptr).release()
}

@_cdecl("gpu_bridge_device_name")
public func gpuBridgeDeviceName(_ ptr: UnsafeRawPointer?) -> UnsafePointer<CChar>? {
    guard let ptr = ptr else { return nil }
    let device = Unmanaged<GPUDevice>.fromOpaque(ptr).takeUnretainedValue()
    return device.namePtr
}
