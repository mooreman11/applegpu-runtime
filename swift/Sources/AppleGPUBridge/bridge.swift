import Foundation
import Metal

/// Internal class wrapping MTLDevice.
final class GPUDevice {
    let device: MTLDevice
    private let nameCString: UnsafeMutablePointer<CChar>

    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        self.device = device
        let name = device.name
        self.nameCString = strdup(name)!
    }

    deinit {
        free(nameCString)
    }

    var namePtr: UnsafePointer<CChar> {
        UnsafePointer(nameCString)
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
