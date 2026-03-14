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

    let data: [Float] = [1.0, 2.0, 3.0, 4.0]
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
