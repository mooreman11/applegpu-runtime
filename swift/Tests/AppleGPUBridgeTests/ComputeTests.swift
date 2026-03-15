import Testing
import Metal
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

@Suite(.serialized) struct BatchTests {
    @Test func testBeginEndBatchBasic() throws {
        // Reset any leftover batch state
        gpuBridgeAbortBatch()

        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let queuePtr = Unmanaged.passUnretained(queue).toOpaque()

        let batchCB = gpuBridgeBeginBatch(queuePtr)
        #expect(batchCB != nil)

        // Double begin should fail
        let batchCB2 = gpuBridgeBeginBatch(queuePtr)
        #expect(batchCB2 == nil)

        let cb = gpuBridgeEndBatch()
        #expect(cb != nil)
        gpuBridgeWaitCommandBuffer(cb!)

        // End with no active batch should return nil
        let cb2 = gpuBridgeEndBatch()
        #expect(cb2 == nil)
    }

    @Test func testBatchMultiOpCorrectness() throws {
        gpuBridgeAbortBatch()

        let gpuDevice = gpuBridgeCreateDevice()!

        let addSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_f32(device const float* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint idx [[thread_position_in_grid]]) {
            if (idx < count) { out[idx] = a[idx] + b[idx]; }
        }
        """
        let mulSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void mul_f32(device const float* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint idx [[thread_position_in_grid]]) {
            if (idx < count) { out[idx] = a[idx] * b[idx]; }
        }
        """

        let addCompute = addSource.withCString { src in "add_f32".withCString { name in gpuBridgeCreateCompute(gpuDevice, src, name) } }!
        let mulCompute = mulSource.withCString { src in "mul_f32".withCString { name in gpuBridgeCreateCompute(gpuDevice, src, name) } }!

        var aData: [Float] = [1, 2, 3, 4]
        var bData: [Float] = [10, 20, 30, 40]
        let bufA = aData.withUnsafeBytes { gpuBridgeCreateBufferWithData(gpuDevice, $0.baseAddress, UInt64(aData.count * 4)) }!
        let bufB = bData.withUnsafeBytes { gpuBridgeCreateBufferWithData(gpuDevice, $0.baseAddress, UInt64(bData.count * 4)) }!
        let bufC = gpuBridgeCreateBuffer(gpuDevice, UInt64(4 * 4))!
        let bufD = gpuBridgeCreateBuffer(gpuDevice, UInt64(4 * 4))!

        let queue = gpuBridgeGetSharedQueue(gpuDevice)!
        let batchCB = gpuBridgeBeginBatch(queue)
        #expect(batchCB != nil)

        // In batch mode, _nb returns non-null (unretained batch CB pointer)
        let cb1 = gpuBridgeComputeElementwiseNB(addCompute, queue, bufA, bufB, bufC, 4)
        #expect(cb1 != nil)

        let cb2 = gpuBridgeComputeElementwiseNB(mulCompute, queue, bufC, bufB, bufD, 4)
        #expect(cb2 != nil)

        // Both should return the same batch command buffer pointer
        #expect(cb1 == batchCB)
        #expect(cb2 == batchCB)

        let finalCB = gpuBridgeEndBatch()
        #expect(finalCB != nil)
        gpuBridgeWaitCommandBuffer(finalCB!)

        let ptr = gpuBridgeBufferContents(bufD)!.assumingMemoryBound(to: Float.self)
        #expect(ptr[0] == 110.0)
        #expect(ptr[1] == 440.0)
        #expect(ptr[2] == 990.0)
        #expect(ptr[3] == 1760.0)

        gpuBridgeDestroyBuffer(bufA)
        gpuBridgeDestroyBuffer(bufB)
        gpuBridgeDestroyBuffer(bufC)
        gpuBridgeDestroyBuffer(bufD)
        gpuBridgeDestroyCompute(addCompute)
        gpuBridgeDestroyCompute(mulCompute)
        gpuBridgeDestroyDevice(gpuDevice)
    }

    @Test func testAbortBatch() throws {
        // Reset any leftover batch state
        gpuBridgeAbortBatch()

        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let queuePtr = Unmanaged.passUnretained(queue).toOpaque()

        let batchCB = gpuBridgeBeginBatch(queuePtr)
        #expect(batchCB != nil)

        gpuBridgeAbortBatch()

        // Should be able to begin a new batch after abort
        let batchCB2 = gpuBridgeBeginBatch(queuePtr)
        #expect(batchCB2 != nil)

        let cb = gpuBridgeEndBatch()
        #expect(cb != nil)
        gpuBridgeWaitCommandBuffer(cb!)
    }
}
