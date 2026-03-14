import Foundation
import Metal

/// Wraps a Metal compute pipeline with command queue.
final class GPUCompute {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLComputePipelineState

    init?(device: MTLDevice, kernelSource: String, functionName: String) {
        self.device = device

        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue

        do {
            let library = try device.makeLibrary(source: kernelSource, options: nil)
            guard let function = library.makeFunction(name: functionName) else { return nil }
            self.pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            return nil
        }
    }

    func dispatchElementwise(bufA: MTLBuffer, bufB: MTLBuffer, bufOut: MTLBuffer, count: Int) -> Bool {
        if count == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 3)

        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, count)
        let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return commandBuffer.status == .completed
    }
    func dispatchUnary(bufIn: MTLBuffer, bufOut: MTLBuffer, count: Int) -> Bool {
        if count == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)

        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, count)
        let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return commandBuffer.status == .completed
    }

    func dispatchMatmul(bufA: MTLBuffer, bufB: MTLBuffer, bufC: MTLBuffer, M: Int, N: Int, K: Int) -> Bool {
        if M == 0 || N == 0 || K == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)

        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)

        // 2D dispatch: each thread computes one element of C[M,N]
        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: N, height: M, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return commandBuffer.status == .completed
    }
}

// MARK: - C ABI exports

@_cdecl("gpu_bridge_create_compute")
public func gpuBridgeCreateCompute(
    _ devicePtr: UnsafeRawPointer?,
    _ kernelSource: UnsafePointer<CChar>?,
    _ functionName: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr,
          let kernelSource = kernelSource,
          let functionName = functionName else { return nil }

    let gpuDevice = getGPUDevice(from: devicePtr)
    let source = String(cString: kernelSource)
    let name = String(cString: functionName)

    guard let compute = GPUCompute(device: gpuDevice.device, kernelSource: source, functionName: name) else {
        return nil
    }
    return Unmanaged.passRetained(compute).toOpaque()
}

@_cdecl("gpu_bridge_destroy_compute")
public func gpuBridgeDestroyCompute(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    Unmanaged<GPUCompute>.fromOpaque(ptr).release()
}

@_cdecl("gpu_bridge_compute_elementwise")
public func gpuBridgeComputeElementwise(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> Int32 {
    guard let computePtr = computePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchElementwise(
        bufA: bufA.buffer,
        bufB: bufB.buffer,
        bufOut: bufOut.buffer,
        count: Int(elementCount)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_unary")
public func gpuBridgeComputeUnary(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchUnary(
        bufIn: bufIn.buffer,
        bufOut: bufOut.buffer,
        count: Int(elementCount)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_matmul")
public func gpuBridgeComputeMatmul(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufCPtr: UnsafeMutableRawPointer?,
    _ M: UInt32,
    _ N: UInt32,
    _ K: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufCPtr = bufCPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufC = Unmanaged<GPUBuffer>.fromOpaque(bufCPtr).takeUnretainedValue()

    let success = compute.dispatchMatmul(
        bufA: bufA.buffer,
        bufB: bufB.buffer,
        bufC: bufC.buffer,
        M: Int(M),
        N: Int(N),
        K: Int(K)
    )
    return success ? 0 : -1
}
