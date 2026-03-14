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

    let dataA: [Float] = [1.0, 2.0, 3.0, 4.0]
    let dataB: [Float] = [10.0, 20.0, 30.0, 40.0]

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
