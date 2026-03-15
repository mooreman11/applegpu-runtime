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
        return dispatchMatmulBatched(bufA: bufA, bufB: bufB, bufC: bufC, M: M, N: N, K: K, batchSize: 1, aBatchStride: M * K, bBatchStride: K * N)
    }

    func dispatchMatmulBatched(bufA: MTLBuffer, bufB: MTLBuffer, bufC: MTLBuffer, M: Int, N: Int, K: Int, batchSize: Int, aBatchStride: Int, bBatchStride: Int) -> Bool {
        if M == 0 || N == 0 || K == 0 || batchSize == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)

        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        var bs = UInt32(batchSize), abs_val = UInt32(aBatchStride), bbs = UInt32(bBatchStride)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&abs_val, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&bbs, length: MemoryLayout<UInt32>.size, index: 8)

        // 3D dispatch: (col, row, batch)
        let w = pipelineState.threadExecutionWidth
        let h = max(pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: N, height: M, depth: batchSize)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return commandBuffer.status == .completed
    }

    func dispatchSoftmaxCausal(bufIn: MTLBuffer, bufOut: MTLBuffer, batchSize: Int, rows: Int, cols: Int) -> Bool {
        if batchSize == 0 || rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var bs = UInt32(batchSize), r = UInt32(rows), c = UInt32(cols)
        encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

        // 2D dispatch: (row, batch)
        let w = pipelineState.threadExecutionWidth
        let h = max(pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: rows, height: batchSize, depth: 1)

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

    func dispatchTransposeBatched(bufIn: MTLBuffer, bufOut: MTLBuffer, batchSize: Int, rows: Int, cols: Int) -> Bool {
        if batchSize == 0 || rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var bs = UInt32(batchSize), r = UInt32(rows), c = UInt32(cols)
        encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

        let w = pipelineState.threadExecutionWidth
        let h = max(pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: cols, height: rows, depth: batchSize)

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

    func dispatchSliceDim0(bufIn: MTLBuffer, bufOut: MTLBuffer, cols: Int, startRow: Int, outRows: Int) -> Bool {
        if outRows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var c = UInt32(cols), sr = UInt32(startRow)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&sr, length: MemoryLayout<UInt32>.size, index: 3)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: cols, height: outRows, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func dispatchSliceDim1(bufIn: MTLBuffer, bufOut: MTLBuffer, inCols: Int, outCols: Int, startCol: Int, rows: Int) -> Bool {
        if rows == 0 || outCols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)

        var ic = UInt32(inCols), oc = UInt32(outCols), sc = UInt32(startCol), r = UInt32(rows)
        encoder.setBytes(&ic, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&oc, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&sc, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 5)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: outCols, height: rows, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func dispatchConcatDim0(bufA: MTLBuffer, bufB: MTLBuffer, bufOut: MTLBuffer, rowsA: Int, cols: Int, totalRows: Int) -> Bool {
        if totalRows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var ra = UInt32(rowsA), c = UInt32(cols)
        encoder.setBytes(&ra, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: cols, height: totalRows, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func dispatchConcatDim1(bufA: MTLBuffer, bufB: MTLBuffer, bufOut: MTLBuffer, rows: Int, colsA: Int, colsB: Int) -> Bool {
        let totalCols = colsA + colsB
        if rows == 0 || totalCols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var r = UInt32(rows), ca = UInt32(colsA), cb = UInt32(colsB)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&ca, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&cb, length: MemoryLayout<UInt32>.size, index: 5)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: totalCols, height: rows, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func dispatchAddBias(bufIn: MTLBuffer, bufBias: MTLBuffer, bufOut: MTLBuffer, rows: Int, cols: Int) -> Bool {
        if rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufBias, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var r = UInt32(rows), c = UInt32(cols)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

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

@_cdecl("gpu_bridge_compute_matmul_batched")
public func gpuBridgeComputeMatmulBatched(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufCPtr: UnsafeMutableRawPointer?,
    _ M: UInt32,
    _ N: UInt32,
    _ K: UInt32,
    _ batchSize: UInt32,
    _ aBatchStride: UInt32,
    _ bBatchStride: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufCPtr = bufCPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufC = Unmanaged<GPUBuffer>.fromOpaque(bufCPtr).takeUnretainedValue()

    let success = compute.dispatchMatmulBatched(
        bufA: bufA.buffer, bufB: bufB.buffer, bufC: bufC.buffer,
        M: Int(M), N: Int(N), K: Int(K),
        batchSize: Int(batchSize), aBatchStride: Int(aBatchStride), bBatchStride: Int(bBatchStride)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_softmax_causal")
public func gpuBridgeComputeSoftmaxCausal(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ batchSize: UInt32,
    _ rows: UInt32,
    _ cols: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchSoftmaxCausal(
        bufIn: bufIn.buffer, bufOut: bufOut.buffer,
        batchSize: Int(batchSize), rows: Int(rows), cols: Int(cols)
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

@_cdecl("gpu_bridge_compute_transpose_batched")
public func gpuBridgeComputeTransposeBatched(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ batchSize: UInt32,
    _ rows: UInt32,
    _ cols: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchTransposeBatched(
        bufIn: bufIn.buffer, bufOut: bufOut.buffer,
        batchSize: Int(batchSize), rows: Int(rows), cols: Int(cols)
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

// MARK: - Slice C ABI exports

@_cdecl("gpu_bridge_compute_slice_dim0")
public func gpuBridgeComputeSliceDim0(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ cols: UInt32,
    _ startRow: UInt32,
    _ outRows: UInt32
) -> Int32 {
    guard let computePtr = computePtr, let bufInPtr = bufInPtr, let bufOutPtr = bufOutPtr else { return -1 }
    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()
    return compute.dispatchSliceDim0(bufIn: bufIn.buffer, bufOut: bufOut.buffer, cols: Int(cols), startRow: Int(startRow), outRows: Int(outRows)) ? 0 : -1
}

@_cdecl("gpu_bridge_compute_slice_dim1")
public func gpuBridgeComputeSliceDim1(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inCols: UInt32,
    _ outCols: UInt32,
    _ startCol: UInt32,
    _ rows: UInt32
) -> Int32 {
    guard let computePtr = computePtr, let bufInPtr = bufInPtr, let bufOutPtr = bufOutPtr else { return -1 }
    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()
    return compute.dispatchSliceDim1(bufIn: bufIn.buffer, bufOut: bufOut.buffer, inCols: Int(inCols), outCols: Int(outCols), startCol: Int(startCol), rows: Int(rows)) ? 0 : -1
}

// MARK: - Concat C ABI exports

@_cdecl("gpu_bridge_compute_concat_dim0")
public func gpuBridgeComputeConcatDim0(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rowsA: UInt32,
    _ cols: UInt32,
    _ totalRows: UInt32
) -> Int32 {
    guard let computePtr = computePtr, let bufAPtr = bufAPtr, let bufBPtr = bufBPtr, let bufOutPtr = bufOutPtr else { return -1 }
    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()
    return compute.dispatchConcatDim0(bufA: bufA.buffer, bufB: bufB.buffer, bufOut: bufOut.buffer, rowsA: Int(rowsA), cols: Int(cols), totalRows: Int(totalRows)) ? 0 : -1
}

@_cdecl("gpu_bridge_compute_concat_dim1")
public func gpuBridgeComputeConcatDim1(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ colsA: UInt32,
    _ colsB: UInt32
) -> Int32 {
    guard let computePtr = computePtr, let bufAPtr = bufAPtr, let bufBPtr = bufBPtr, let bufOutPtr = bufOutPtr else { return -1 }
    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()
    return compute.dispatchConcatDim1(bufA: bufA.buffer, bufB: bufB.buffer, bufOut: bufOut.buffer, rows: Int(rows), colsA: Int(colsA), colsB: Int(colsB)) ? 0 : -1
}

// MARK: - AddBias C ABI exports

@_cdecl("gpu_bridge_compute_add_bias")
public func gpuBridgeComputeAddBias(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufBiasPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> Int32 {
    guard let computePtr = computePtr, let bufInPtr = bufInPtr, let bufBiasPtr = bufBiasPtr, let bufOutPtr = bufOutPtr else { return -1 }
    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufBias = Unmanaged<GPUBuffer>.fromOpaque(bufBiasPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()
    return compute.dispatchAddBias(bufIn: bufIn.buffer, bufBias: bufBias.buffer, bufOut: bufOut.buffer, rows: Int(rows), cols: Int(cols)) ? 0 : -1
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
    return gpuBridgeComputeMatmulBatchedNB(computePtr, queuePtr, bufAPtr, bufBPtr, bufCPtr, M, N, K, 1, M * K, K * N)
}

@_cdecl("gpu_bridge_compute_matmul_batched_nb")
public func gpuBridgeComputeMatmulBatchedNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufCPtr: UnsafeMutableRawPointer?,
    _ M: UInt32,
    _ N: UInt32,
    _ K: UInt32,
    _ batchSize: UInt32,
    _ aBatchStride: UInt32,
    _ bBatchStride: UInt32
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

    if M == 0 || N == 0 || K == 0 || batchSize == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufC.buffer, offset: 0, index: 2)

    var m = M, n = N, k = K, bs = batchSize, abs_val = aBatchStride, bbs = bBatchStride
    encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
    encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 6)
    encoder.setBytes(&abs_val, length: MemoryLayout<UInt32>.size, index: 7)
    encoder.setBytes(&bbs, length: MemoryLayout<UInt32>.size, index: 8)

    let w = compute.pipelineState.threadExecutionWidth
    let h = max(compute.pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(N), height: Int(M), depth: Int(batchSize))

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()

    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_softmax_causal_nb")
public func gpuBridgeComputeSoftmaxCausalNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ batchSize: UInt32,
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

    if batchSize == 0 || rows == 0 || cols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var bs = batchSize, r = rows, c = cols
    encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

    let w = compute.pipelineState.threadExecutionWidth
    let h = max(compute.pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(rows), height: Int(batchSize), depth: 1)

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

@_cdecl("gpu_bridge_compute_transpose_batched_nb")
public func gpuBridgeComputeTransposeBatchedNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ batchSize: UInt32,
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

    if batchSize == 0 || rows == 0 || cols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var bs = batchSize, r = rows, c = cols
    encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

    let w = compute.pipelineState.threadExecutionWidth
    let h = max(compute.pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(cols), height: Int(rows), depth: Int(batchSize))

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

// MARK: - Non-blocking slice/concat/add_bias C ABI exports

@_cdecl("gpu_bridge_compute_slice_dim0_nb")
public func gpuBridgeComputeSliceDim0NB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ cols: UInt32,
    _ startRow: UInt32,
    _ outRows: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr, let queuePtr = queuePtr,
          let bufInPtr = bufInPtr, let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if outRows == 0 || cols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var c = cols, sr = startRow
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&sr, length: MemoryLayout<UInt32>.size, index: 3)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(cols), height: Int(outRows), depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()
    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_slice_dim1_nb")
public func gpuBridgeComputeSliceDim1NB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inCols: UInt32,
    _ outCols: UInt32,
    _ startCol: UInt32,
    _ rows: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr, let queuePtr = queuePtr,
          let bufInPtr = bufInPtr, let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if rows == 0 || outCols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var ic = inCols, oc = outCols, sc = startCol, r = rows
    encoder.setBytes(&ic, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&oc, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&sc, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 5)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(outCols), height: Int(rows), depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()
    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_concat_dim0_nb")
public func gpuBridgeComputeConcatDim0NB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rowsA: UInt32,
    _ cols: UInt32,
    _ totalRows: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr, let queuePtr = queuePtr,
          let bufAPtr = bufAPtr, let bufBPtr = bufBPtr, let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if totalRows == 0 || cols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    var ra = rowsA, c = cols
    encoder.setBytes(&ra, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(cols), height: Int(totalRows), depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()
    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_concat_dim1_nb")
public func gpuBridgeComputeConcatDim1NB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ colsA: UInt32,
    _ colsB: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr, let queuePtr = queuePtr,
          let bufAPtr = bufAPtr, let bufBPtr = bufBPtr, let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let totalCols = Int(colsA) + Int(colsB)
    if rows == 0 || totalCols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    var r = rows, ca = colsA, cb = colsB
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&ca, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&cb, length: MemoryLayout<UInt32>.size, index: 5)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: totalCols, height: Int(rows), depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()
    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_compute_add_bias_nb")
public func gpuBridgeComputeAddBiasNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufBiasPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr, let queuePtr = queuePtr,
          let bufInPtr = bufInPtr, let bufBiasPtr = bufBiasPtr, let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufBias = Unmanaged<GPUBuffer>.fromOpaque(bufBiasPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if rows == 0 || cols == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufBias.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    var r = rows, c = cols
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(cols), height: Int(rows), depth: 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()
    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}

// MARK: - N-D stride-based element-wise dispatch

@_cdecl("gpu_bridge_compute_binary_nd")
public func gpuBridgeComputeBinaryND(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ aStrides: UnsafePointer<UInt32>?,
    _ bStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufOutPtr = bufOutPtr,
          let aStrides = aStrides,
          let bStrides = bStrides,
          let outShape = outShape else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return 0 }

    guard let commandBuffer = compute.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(aStrides, length: arrayBytes, index: 3)
    encoder.setBytes(bStrides, length: arrayBytes, index: 4)
    encoder.setBytes(outShape, length: arrayBytes, index: 5)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 6)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 7)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return commandBuffer.status == .completed ? 0 : -1
}

@_cdecl("gpu_bridge_compute_binary_nd_nb")
public func gpuBridgeComputeBinaryNDNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ bufBPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ aStrides: UnsafePointer<UInt32>?,
    _ bStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufAPtr = bufAPtr,
          let bufBPtr = bufBPtr,
          let bufOutPtr = bufOutPtr,
          let aStrides = aStrides,
          let bStrides = bStrides,
          let outShape = outShape else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufA = Unmanaged<GPUBuffer>.fromOpaque(bufAPtr).takeUnretainedValue()
    let bufB = Unmanaged<GPUBuffer>.fromOpaque(bufBPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(aStrides, length: arrayBytes, index: 3)
    encoder.setBytes(bStrides, length: arrayBytes, index: 4)
    encoder.setBytes(outShape, length: arrayBytes, index: 5)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 6)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 7)

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

@_cdecl("gpu_bridge_compute_unary_nd")
public func gpuBridgeComputeUnaryND(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr,
          let inStrides = inStrides,
          let outShape = outShape else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return 0 }

    guard let commandBuffer = compute.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(inStrides, length: arrayBytes, index: 2)
    encoder.setBytes(outShape, length: arrayBytes, index: 3)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 4)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return commandBuffer.status == .completed ? 0 : -1
}

@_cdecl("gpu_bridge_compute_unary_nd_nb")
public func gpuBridgeComputeUnaryNDNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr,
          let inStrides = inStrides,
          let outShape = outShape else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(inStrides, length: arrayBytes, index: 2)
    encoder.setBytes(outShape, length: arrayBytes, index: 3)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 4)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)

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

// MARK: - Pow N-D dispatch (unary + extra float constant at buffer 6)

@_cdecl("gpu_bridge_compute_pow_nd")
public func gpuBridgeComputePowND(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32,
    _ exponent: Float
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr,
          let inStrides = inStrides,
          let outShape = outShape else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return 0 }

    guard let commandBuffer = compute.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(inStrides, length: arrayBytes, index: 2)
    encoder.setBytes(outShape, length: arrayBytes, index: 3)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 4)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
    var exp = exponent
    encoder.setBytes(&exp, length: MemoryLayout<Float>.size, index: 6)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return commandBuffer.status == .completed ? 0 : -1
}

@_cdecl("gpu_bridge_compute_pow_nd_nb")
public func gpuBridgeComputePowNDNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32,
    _ exponent: Float
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr,
          let inStrides = inStrides,
          let outShape = outShape else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(inStrides, length: arrayBytes, index: 2)
    encoder.setBytes(outShape, length: arrayBytes, index: 3)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 4)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
    var exp = exponent
    encoder.setBytes(&exp, length: MemoryLayout<Float>.size, index: 6)

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

// MARK: - Clamp N-D dispatch (unary + two float constants at buffers 6, 7)

@_cdecl("gpu_bridge_compute_clamp_nd")
public func gpuBridgeComputeClampND(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32,
    _ minVal: Float,
    _ maxVal: Float
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr,
          let inStrides = inStrides,
          let outShape = outShape else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return 0 }

    guard let commandBuffer = compute.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(inStrides, length: arrayBytes, index: 2)
    encoder.setBytes(outShape, length: arrayBytes, index: 3)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 4)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
    var mn = minVal
    encoder.setBytes(&mn, length: MemoryLayout<Float>.size, index: 6)
    var mx = maxVal
    encoder.setBytes(&mx, length: MemoryLayout<Float>.size, index: 7)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return commandBuffer.status == .completed ? 0 : -1
}

@_cdecl("gpu_bridge_compute_clamp_nd_nb")
public func gpuBridgeComputeClampNDNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32,
    _ minVal: Float,
    _ maxVal: Float
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr,
          let inStrides = inStrides,
          let outShape = outShape else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(inStrides, length: arrayBytes, index: 2)
    encoder.setBytes(outShape, length: arrayBytes, index: 3)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 4)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
    var mn = minVal
    encoder.setBytes(&mn, length: MemoryLayout<Float>.size, index: 6)
    var mx = maxVal
    encoder.setBytes(&mx, length: MemoryLayout<Float>.size, index: 7)

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

// MARK: - N-D Fused dispatch

@_cdecl("gpu_bridge_compute_fused_nd")
public func gpuBridgeComputeFusedND(
    _ computePtr: UnsafeMutableRawPointer?,
    _ inputBuffers: UnsafePointer<UnsafeRawPointer?>?,
    _ bufferCount: UInt32,
    _ outputPtr: UnsafeMutableRawPointer?,
    _ inputStrides: UnsafePointer<UnsafePointer<UInt32>?>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let inputBuffers = inputBuffers,
          let outputPtr = outputPtr,
          let inputStrides = inputStrides,
          let outShape = outShape else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(outputPtr).takeUnretainedValue()

    let n = Int(bufferCount)
    let count = Int(numel)
    if count == 0 { return 0 }

    guard let commandBuffer = compute.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(compute.pipelineState)

    // Input data buffers: buffer(0)..buffer(n-1)
    for i in 0..<n {
        guard let bufPtr = inputBuffers[i] else { return -1 }
        let buf = Unmanaged<GPUBuffer>.fromOpaque(bufPtr).takeUnretainedValue()
        encoder.setBuffer(buf.buffer, offset: 0, index: i)
    }
    // Output buffer: buffer(n)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: n)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    // Per-input stride arrays: buffer(n+1)..buffer(2n)
    for i in 0..<n {
        guard let strides = inputStrides[i] else { return -1 }
        encoder.setBytes(strides, length: arrayBytes, index: n + 1 + i)
    }
    // Output shape: buffer(2n+1)
    encoder.setBytes(outShape, length: arrayBytes, index: 2 * n + 1)
    // ndim: buffer(2n+2)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 2 * n + 2)
    // numel: buffer(2n+3)
    var ne = numel
    encoder.setBytes(&ne, length: MemoryLayout<UInt32>.size, index: 2 * n + 3)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return commandBuffer.status == .completed ? 0 : -1
}

@_cdecl("gpu_bridge_compute_fused_nd_nb")
public func gpuBridgeComputeFusedNDNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ inputBuffers: UnsafePointer<UnsafeRawPointer?>?,
    _ bufferCount: UInt32,
    _ outputPtr: UnsafeMutableRawPointer?,
    _ inputStrides: UnsafePointer<UnsafePointer<UInt32>?>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let inputBuffers = inputBuffers,
          let outputPtr = outputPtr,
          let inputStrides = inputStrides,
          let outShape = outShape else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(outputPtr).takeUnretainedValue()

    let n = Int(bufferCount)
    let count = Int(numel)
    if count == 0 { return nil }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)

    // Input data buffers: buffer(0)..buffer(n-1)
    for i in 0..<n {
        guard let bufPtr = inputBuffers[i] else { return nil }
        let buf = Unmanaged<GPUBuffer>.fromOpaque(bufPtr).takeUnretainedValue()
        encoder.setBuffer(buf.buffer, offset: 0, index: i)
    }
    // Output buffer: buffer(n)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: n)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    // Per-input stride arrays: buffer(n+1)..buffer(2n)
    for i in 0..<n {
        guard let strides = inputStrides[i] else { return nil }
        encoder.setBytes(strides, length: arrayBytes, index: n + 1 + i)
    }
    // Output shape: buffer(2n+1)
    encoder.setBytes(outShape, length: arrayBytes, index: 2 * n + 1)
    // ndim: buffer(2n+2)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 2 * n + 2)
    // numel: buffer(2n+3)
    var ne = numel
    encoder.setBytes(&ne, length: MemoryLayout<UInt32>.size, index: 2 * n + 3)

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
