import Testing
@testable import AppleGPUBridge

@Test func queuePoolReturnsNonNilForIndices0Through3() {
    let devicePtr = gpuBridgeCreateDevice()
    #expect(devicePtr != nil, "Metal device should be available")
    guard let devicePtr = devicePtr else { return }

    for i: UInt32 in 0..<4 {
        let queue = gpuBridgeGetQueue(devicePtr, i)
        #expect(queue != nil, "Queue at index \(i) should be non-nil")
    }

    gpuBridgeDestroyDevice(devicePtr)
}

@Test func createEventReturnsNonNil() {
    let devicePtr = gpuBridgeCreateDevice()
    #expect(devicePtr != nil, "Metal device should be available")
    guard let devicePtr = devicePtr else { return }

    let event = gpuBridgeCreateEvent(devicePtr)
    #expect(event != nil, "MTLEvent should be creatable")

    gpuBridgeDestroyDevice(devicePtr)
}
