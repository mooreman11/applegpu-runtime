import Testing
@testable import AppleGPUBridge

@Test func deviceCreationViaCABI() {
    let ptr = gpuBridgeCreateDevice()
    #expect(ptr != nil, "Metal device should be available on macOS")
    if let ptr = ptr {
        let name = gpuBridgeDeviceName(ptr)
        #expect(name != nil)
        gpuBridgeDestroyDevice(ptr)
    }
}
