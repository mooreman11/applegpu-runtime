import Testing
@testable import AppleGPUBridge

/// Helper to create a device, compile a kernel, and run a binary op on floats.
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

/// Helper to run a unary op on floats.
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
    // C = [[19,22],[43,50]]
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
