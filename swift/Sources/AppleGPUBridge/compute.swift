import Foundation
import Metal

/// Device-level shared command queue for batched (non-blocking) dispatch.
/// Metal guarantees in-order execution on the same queue.
/// Thread-safe: guarded by a lock since multiple threads may call get_shared_queue.
private var sharedCommandQueue: MTLCommandQueue?
private let sharedQueueLock = NSLock()

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

    func dispatchFused(inputs: [MTLBuffer], output: MTLBuffer, count: Int) -> Bool {
        if count == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)

        // Set input buffers at indices 0..n-1
        for (i, buf) in inputs.enumerated() {
            encoder.setBuffer(buf, offset: 0, index: i)
        }
        // Output at index n
        encoder.setBuffer(output, offset: 0, index: inputs.count)
        // Count at index n+1
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: inputs.count + 1)

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

    func dispatchSoftmax(bufIn: MTLBuffer, bufOut: MTLBuffer, rows: Int, cols: Int) -> Bool {
        if rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var r = UInt32(rows), c = UInt32(cols)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)

        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, rows)
        let threadGroups = (rows + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func dispatchTranspose(bufIn: MTLBuffer, bufOut: MTLBuffer, rows: Int, cols: Int) -> Bool {
        if rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var r = UInt32(rows), c = UInt32(cols)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: cols, height: rows, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func dispatchScalarMul(bufIn: MTLBuffer, bufOut: MTLBuffer, scale: Float, count: Int) -> Bool {
        if count == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var s = scale
        encoder.setBytes(&s, length: MemoryLayout<Float>.size, index: 2)
        var c = UInt32(count)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)

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
    func dispatchLayerNorm(bufIn: MTLBuffer, bufGamma: MTLBuffer, bufBeta: MTLBuffer, bufOut: MTLBuffer, rows: Int, cols: Int, eps: Float) -> Bool {
        if rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufGamma, offset: 0, index: 1)
        encoder.setBuffer(bufBeta, offset: 0, index: 2)
        encoder.setBuffer(bufOut, offset: 0, index: 3)

        var r = UInt32(rows), c = UInt32(cols), e = eps
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&e, length: MemoryLayout<Float>.size, index: 6)

        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, rows)
        let threadGroups = (rows + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func dispatchEmbedding(bufWeights: MTLBuffer, bufIndices: MTLBuffer, bufOut: MTLBuffer, seqLen: Int, embedDim: Int) -> Bool {
        if seqLen == 0 || embedDim == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufWeights, offset: 0, index: 0)
        encoder.setBuffer(bufIndices, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var s = UInt32(seqLen), d = UInt32(embedDim)
        encoder.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 4)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: embedDim, height: seqLen, depth: 1)

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

@_cdecl("gpu_bridge_compute_fused")
public func gpuBridgeComputeFused(
    _ computePtr: UnsafeMutableRawPointer?,
    _ inputBuffers: UnsafePointer<UnsafeRawPointer?>?,
    _ bufferCount: UInt32,
    _ outputPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> Int32 {
    guard let computePtr = computePtr,
          let inputBuffers = inputBuffers,
          let outputPtr = outputPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(outputPtr).takeUnretainedValue()

    var mtlBuffers: [MTLBuffer] = []
    for i in 0..<Int(bufferCount) {
        guard let bufPtr = inputBuffers[i] else { return -1 }
        let buf = Unmanaged<GPUBuffer>.fromOpaque(bufPtr).takeUnretainedValue()
        mtlBuffers.append(buf.buffer)
    }

    let success = compute.dispatchFused(
        inputs: mtlBuffers,
        output: bufOut.buffer,
        count: Int(elementCount)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_softmax")
public func gpuBridgeComputeSoftmax(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchSoftmax(
        bufIn: bufIn.buffer, bufOut: bufOut.buffer,
        rows: Int(rows), cols: Int(cols)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_transpose")
public func gpuBridgeComputeTranspose(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchTranspose(
        bufIn: bufIn.buffer, bufOut: bufOut.buffer,
        rows: Int(rows), cols: Int(cols)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_scalar_mul")
public func gpuBridgeComputeScalarMul(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ scale: Float,
    _ elementCount: UInt64
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    return compute.dispatchScalarMul(
        bufIn: bufIn.buffer, bufOut: bufOut.buffer,
        scale: scale, count: Int(elementCount)
    ) ? 0 : -1
}

@_cdecl("gpu_bridge_compute_layer_norm")
public func gpuBridgeComputeLayerNorm(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufGammaPtr: UnsafeRawPointer?,
    _ bufBetaPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32,
    _ eps: Float
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufGammaPtr = bufGammaPtr,
          let bufBetaPtr = bufBetaPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufGamma = Unmanaged<GPUBuffer>.fromOpaque(bufGammaPtr).takeUnretainedValue()
    let bufBeta = Unmanaged<GPUBuffer>.fromOpaque(bufBetaPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchLayerNorm(
        bufIn: bufIn.buffer, bufGamma: bufGamma.buffer, bufBeta: bufBeta.buffer,
        bufOut: bufOut.buffer, rows: Int(rows), cols: Int(cols), eps: eps
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_embedding")
public func gpuBridgeComputeEmbedding(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufWeightsPtr: UnsafeRawPointer?,
    _ bufIndicesPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ seqLen: UInt32,
    _ embedDim: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufWeightsPtr = bufWeightsPtr,
          let bufIndicesPtr = bufIndicesPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufWeights = Unmanaged<GPUBuffer>.fromOpaque(bufWeightsPtr).takeUnretainedValue()
    let bufIndices = Unmanaged<GPUBuffer>.fromOpaque(bufIndicesPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchEmbedding(
        bufWeights: bufWeights.buffer, bufIndices: bufIndices.buffer,
        bufOut: bufOut.buffer, seqLen: Int(seqLen), embedDim: Int(embedDim)
    )
    return success ? 0 : -1
}

// MARK: - Non-blocking (batched) C ABI exports

@_cdecl("gpu_bridge_get_shared_queue")
public func gpuBridgeGetSharedQueue(_ deviceHandle: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let wrapper = Unmanaged<GPUDevice>.fromOpaque(deviceHandle).takeUnretainedValue()
    sharedQueueLock.lock()
    defer { sharedQueueLock.unlock() }
    if sharedCommandQueue == nil {
        sharedCommandQueue = wrapper.device.makeCommandQueue()
    }
    guard let queue = sharedCommandQueue else { return nil }
    return Unmanaged.passUnretained(queue).toOpaque()
}

@_cdecl("gpu_bridge_wait_command_buffer")
public func gpuBridgeWaitCommandBuffer(_ cbHandle: UnsafeMutableRawPointer) {
    let cb = Unmanaged<AnyObject>.fromOpaque(cbHandle).takeRetainedValue()
    (cb as! MTLCommandBuffer).waitUntilCompleted()
}

@_cdecl("gpu_bridge_compute_elementwise_nb")
public func gpuBridgeComputeElementwiseNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(elementCount)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    var elementCountVal = UInt32(count)
    encoder.setBytes(&elementCountVal, length: MemoryLayout<UInt32>.size, index: 3)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_unary_nb")
public func gpuBridgeComputeUnaryNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(elementCount)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var elementCountVal = UInt32(count)
    encoder.setBytes(&elementCountVal, length: MemoryLayout<UInt32>.size, index: 2)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_matmul_nb")
public func gpuBridgeComputeMatmulNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufCPtr: UnsafeMutableRawPointer?,
    _ M: UInt32,
    _ N: UInt32,
    _ K: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufCPtr = bufCPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufC = Unmanaged<GPUBuffer>.fromOpaque(bufCPtr).takeUnretainedValue()

    if M == 0 || N == 0 || K == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufC.buffer, offset: 0, index: 2)

    var m = M, n = N, k = K
    encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(N), height: Int(M), depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_softmax_nb")
public func gpuBridgeComputeSoftmaxNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if rows == 0 || cols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var r = rows, c = cols
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)

    let rowCount = Int(rows)
    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, rowCount)
    let threadGroups = (rowCount + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_transpose_nb")
public func gpuBridgeComputeTransposeNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if rows == 0 || cols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var r = rows, c = cols
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(cols), height: Int(rows), depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_scalar_mul_nb")
public func gpuBridgeComputeScalarMulNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ scale: Float,
    _ elementCount: UInt64
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(elementCount)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var s = scale
    encoder.setBytes(&s, length: MemoryLayout<Float>.size, index: 2)
    var c = UInt32(count)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_fused_nb")
public func gpuBridgeComputeFusedNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ inputBuffers: UnsafePointer<UnsafeRawPointer?>?,
    _ bufferCount: UInt32,
    _ outputPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let inputBuffers = inputBuffers,
          let outputPtr = outputPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(outputPtr).takeUnretainedValue()

    var mtlBuffers: [MTLBuffer] = []
    for i in 0..<Int(bufferCount) {
        guard let bufPtr = inputBuffers[i] else { return nil }
        let buf = Unmanaged<GPUBuffer>.fromOpaque(bufPtr).takeUnretainedValue()
        mtlBuffers.append(buf.buffer)
    }

    let count = Int(elementCount)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)

    for (i, buf) in mtlBuffers.enumerated() {
        encoder.setBuffer(buf, offset: 0, index: i)
    }
    encoder.setBuffer(bufOut.buffer, offset: 0, index: mtlBuffers.count)
    var elementCountVal = UInt32(count)
    encoder.setBytes(&elementCountVal, length: MemoryLayout<UInt32>.size, index: mtlBuffers.count + 1)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_layer_norm_nb")
public func gpuBridgeComputeLayerNormNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufGammaPtr: UnsafeRawPointer?,
    _ bufBetaPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32,
    _ eps: Float
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufGammaPtr = bufGammaPtr,
          let bufBetaPtr = bufBetaPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufGamma = Unmanaged<GPUBuffer>.fromOpaque(bufGammaPtr).takeUnretainedValue()
    let bufBeta = Unmanaged<GPUBuffer>.fromOpaque(bufBetaPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if rows == 0 || cols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufGamma.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufBeta.buffer, offset: 0, index: 2)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 3)

    var r = rows, c = cols, e = eps
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 5)
    encoder.setBytes(&e, length: MemoryLayout<Float>.size, index: 6)

    let rowCount = Int(rows)
    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, rowCount)
    let threadGroups = (rowCount + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_blit_copy_nb")
public func gpuBridgeBlitCopyNonBlocking(
    _ deviceHandle: UnsafeMutableRawPointer,
    _ queueHandle: UnsafeMutableRawPointer,
    _ srcBuf: UnsafeMutableRawPointer,
    _ dstBuf: UnsafeMutableRawPointer,
    _ sizeBytes: UInt64
) -> UnsafeMutableRawPointer? {
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queueHandle).takeUnretainedValue()
    let src = Unmanaged<GPUBuffer>.fromOpaque(srcBuf).takeUnretainedValue()
    let dst = Unmanaged<GPUBuffer>.fromOpaque(dstBuf).takeUnretainedValue()

    guard let commandBuffer = queue.makeCommandBuffer(),
          let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return nil }

    blitEncoder.copy(from: src.buffer, sourceOffset: 0, to: dst.buffer, destinationOffset: 0, size: Int(sizeBytes))
    blitEncoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_embedding_nb")
public func gpuBridgeComputeEmbeddingNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufWeightsPtr: UnsafeRawPointer?,
    _ bufIndicesPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ seqLen: UInt32,
    _ embedDim: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufWeightsPtr = bufWeightsPtr,
          let bufIndicesPtr = bufIndicesPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufWeights = Unmanaged<GPUBuffer>.fromOpaque(bufWeightsPtr).takeUnretainedValue()
    let bufIndices = Unmanaged<GPUBuffer>.fromOpaque(bufIndicesPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if seqLen == 0 || embedDim == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufWeights.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufIndices.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    var s = seqLen, d = embedDim
    encoder.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 4)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(embedDim), height: Int(seqLen), depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}
