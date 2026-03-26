import Foundation
import Metal
import MetalPerformanceShaders

/// Device-level shared command queue for batched (non-blocking) dispatch.
/// Metal guarantees in-order execution on the same queue.
/// Thread-safe: guarded by a lock since multiple threads may call get_shared_queue.
private var sharedCommandQueue: MTLCommandQueue?
private let sharedQueueLock = NSLock()

/// Thread-local keys for per-thread batch state.
/// This prevents SIGSEGV when Rust tests run in parallel — each thread gets its own
/// batch command buffer and context ID. In production, the outer Mutex<LazyRuntime>
/// serializes access anyway, but thread-local storage is correct for both cases.
private let kActiveBatchCB = "com.applegpu.activeBatchCommandBuffer"
private let kCurrentContextId = "com.applegpu.currentContextId"

/// Thread-local active batch command buffer for single-CB encoding mode.
/// When non-nil, _nb dispatch functions encode into this CB instead of creating their own.
private var activeBatchCommandBuffer: MTLCommandBuffer? {
    get { Thread.current.threadDictionary[kActiveBatchCB] as? MTLCommandBuffer }
    set { Thread.current.threadDictionary[kActiveBatchCB] = newValue }
}

/// Thread-local current context ID for batch context system.
private var currentContextId: UInt32 {
    get { Thread.current.threadDictionary[kCurrentContextId] as? UInt32 ?? 0 }
    set { Thread.current.threadDictionary[kCurrentContextId] = newValue }
}

/// Lock for legacy batch state access patterns that check activeBatchCommandBuffer.
/// Still needed because begin_batch/end_batch/abort_batch are called from the same thread
/// but we need to prevent other threads from seeing partial state.
private let batchLock = NSLock()

// --- Queue pool for concurrent dispatch ---
private var queuePool: [MTLCommandQueue] = []
private let queuePoolLock = NSLock()
private let maxQueueCount: Int = 4

// --- Batch context system (global dict, guarded by contextLock) ---
private var batchContexts: [UInt32: MTLCommandBuffer] = [:]
private let contextLock = NSLock()

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

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return false
        }

        let elementSize = MemoryLayout<Float>.size
        let aBytes = M * K * elementSize
        let bBytes = K * N * elementSize
        let cBytes = M * N * elementSize
        let useMPS = bufA.length >= aBytes && bufB.length >= bBytes && bufC.length >= cBytes
            && aBytes >= 16 && bBytes >= 16 && cBytes >= 16

        if !useMPS {
            // Tiny matrix — use custom MSL kernel
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
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
            // Tiled kernel: 32×32 threadgroups
            let ts = 32
            encoder.dispatchThreadgroups(
                MTLSize(width: (N + ts - 1) / ts, height: (M + ts - 1) / ts, depth: batchSize),
                threadsPerThreadgroup: MTLSize(width: ts, height: ts, depth: 1))
            encoder.endEncoding()
        } else if batchSize == 1 {
            let matA = MPSMatrix(buffer: bufA, descriptor: MPSMatrixDescriptor(
                rows: M, columns: K, rowBytes: K * elementSize, dataType: .float32))
            let matB = MPSMatrix(buffer: bufB, descriptor: MPSMatrixDescriptor(
                rows: K, columns: N, rowBytes: N * elementSize, dataType: .float32))
            let matC = MPSMatrix(buffer: bufC, descriptor: MPSMatrixDescriptor(
                rows: M, columns: N, rowBytes: N * elementSize, dataType: .float32))

            let mm = MPSMatrixMultiplication(
                device: pipelineState.device,
                transposeLeft: false, transposeRight: false,
                resultRows: M, resultColumns: N, interiorColumns: K,
                alpha: 1.0, beta: 0.0)
            mm.encode(commandBuffer: commandBuffer, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
        } else {
            let aStride = aBatchStride * elementSize
            let bStride = bBatchStride * elementSize
            let cStride = M * N * elementSize

            let mm = MPSMatrixMultiplication(
                device: pipelineState.device,
                transposeLeft: false, transposeRight: false,
                resultRows: M, resultColumns: N, interiorColumns: K,
                alpha: 1.0, beta: 0.0)

            for batch in 0..<batchSize {
                let matA = MPSMatrix(buffer: bufA, offset: batch * aStride, descriptor: MPSMatrixDescriptor(
                    rows: M, columns: K, rowBytes: K * elementSize, dataType: .float32))
                let matB = MPSMatrix(buffer: bufB, offset: batch * bStride, descriptor: MPSMatrixDescriptor(
                    rows: K, columns: N, rowBytes: N * elementSize, dataType: .float32))
                let matC = MPSMatrix(buffer: bufC, offset: batch * cStride, descriptor: MPSMatrixDescriptor(
                    rows: M, columns: N, rowBytes: N * elementSize, dataType: .float32))
                mm.encode(commandBuffer: commandBuffer, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
            }
        }

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

    func dispatchSoftmaxBackward(bufGradOut: MTLBuffer, bufOut: MTLBuffer, bufGradIn: MTLBuffer, rows: Int, cols: Int) -> Bool {
        if rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufGradOut, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)
        encoder.setBuffer(bufGradIn, offset: 0, index: 2)

        var r = UInt32(rows), c = UInt32(cols)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

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

    func dispatchLayerNormBackward(bufGradOut: MTLBuffer, bufIn: MTLBuffer, bufGamma: MTLBuffer, bufGradIn: MTLBuffer, rows: Int, cols: Int, eps: Float) -> Bool {
        if rows == 0 || cols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufGradOut, offset: 0, index: 0)
        encoder.setBuffer(bufIn, offset: 0, index: 1)
        encoder.setBuffer(bufGamma, offset: 0, index: 2)
        encoder.setBuffer(bufGradIn, offset: 0, index: 3)

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

    /// Dispatch gather/index_select: 3 buffers (input, indices, output) + 3 uint params.
    func dispatchGather(bufIn: MTLBuffer, bufIndices: MTLBuffer, bufOut: MTLBuffer,
                        rows: Int, inCols: Int, outCols: Int) -> Bool {
        if rows == 0 || outCols == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufIndices, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var r = UInt32(rows), ic = UInt32(inCols), oc = UInt32(outCols)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&ic, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&oc, length: MemoryLayout<UInt32>.size, index: 5)

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

    /// Non-blocking gather dispatch.
    func dispatchGatherNB(queue: MTLCommandQueue, bufIn: MTLBuffer, bufIndices: MTLBuffer,
                          bufOut: MTLBuffer, rows: Int, inCols: Int, outCols: Int) -> (AnyObject, Bool)? {
        if rows == 0 || outCols == 0 {
            let ctxId = currentContextId
            contextLock.lock()
            if ctxId > 0, let batchCB = batchContexts[ctxId] {
                contextLock.unlock()
                return (batchCB as AnyObject, true)
            }
            contextLock.unlock()
            batchLock.lock()
            let result = activeBatchCommandBuffer.map { ($0 as AnyObject, true) }
            batchLock.unlock()
            return result
        }

        let commandBuffer: MTLCommandBuffer
        let isBatch: Bool
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            commandBuffer = batchCB
            isBatch = true
            contextLock.unlock()
        } else {
            contextLock.unlock()
            batchLock.lock()
            if let batchCB = activeBatchCommandBuffer {
                commandBuffer = batchCB
                isBatch = true
                batchLock.unlock()
            } else {
                batchLock.unlock()
                guard let cb = queue.makeCommandBuffer() else { return nil }
                commandBuffer = cb
                isBatch = false
            }
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufIn, offset: 0, index: 0)
        encoder.setBuffer(bufIndices, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        var r = UInt32(rows), ic = UInt32(inCols), oc = UInt32(outCols)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&ic, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&oc, length: MemoryLayout<UInt32>.size, index: 5)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: outCols, height: rows, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        if isBatch {
            return (commandBuffer as AnyObject, true)
        } else {
            commandBuffer.commit()
            return (commandBuffer as AnyObject, false)
        }
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

@_cdecl("gpu_bridge_compute_softmax_backward")
public func gpuBridgeComputeSoftmaxBackward(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufGradOutPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeRawPointer?,
    _ bufGradInPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufGradOutPtr = bufGradOutPtr,
          let bufOutPtr = bufOutPtr,
          let bufGradInPtr = bufGradInPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufGradOut = Unmanaged<GPUBuffer>.fromOpaque(bufGradOutPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()
    let bufGradIn = Unmanaged<GPUBuffer>.fromOpaque(bufGradInPtr).takeUnretainedValue()

    let success = compute.dispatchSoftmaxBackward(
        bufGradOut: bufGradOut.buffer, bufOut: bufOut.buffer, bufGradIn: bufGradIn.buffer,
        rows: Int(rows), cols: Int(cols)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_layer_norm_backward")
public func gpuBridgeComputeLayerNormBackward(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufGradOutPtr: UnsafeRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufGammaPtr: UnsafeRawPointer?,
    _ bufGradInPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32,
    _ eps: Float
) -> Int32 {
    guard let computePtr = computePtr,
          let bufGradOutPtr = bufGradOutPtr,
          let bufInPtr = bufInPtr,
          let bufGammaPtr = bufGammaPtr,
          let bufGradInPtr = bufGradInPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufGradOut = Unmanaged<GPUBuffer>.fromOpaque(bufGradOutPtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufGamma = Unmanaged<GPUBuffer>.fromOpaque(bufGammaPtr).takeUnretainedValue()
    let bufGradIn = Unmanaged<GPUBuffer>.fromOpaque(bufGradInPtr).takeUnretainedValue()

    let success = compute.dispatchLayerNormBackward(
        bufGradOut: bufGradOut.buffer, bufIn: bufIn.buffer, bufGamma: bufGamma.buffer,
        bufGradIn: bufGradIn.buffer, rows: Int(rows), cols: Int(cols), eps: eps
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

// MARK: - Gather C ABI exports

@_cdecl("gpu_bridge_compute_gather")
public func gpuBridgeComputeGather(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufIndicesPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ inCols: UInt32,
    _ outCols: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufIndicesPtr = bufIndicesPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufIndices = Unmanaged<GPUBuffer>.fromOpaque(bufIndicesPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let success = compute.dispatchGather(
        bufIn: bufIn.buffer, bufIndices: bufIndices.buffer,
        bufOut: bufOut.buffer, rows: Int(rows), inCols: Int(inCols), outCols: Int(outCols)
    )
    return success ? 0 : -1
}

@_cdecl("gpu_bridge_compute_gather_nb")
public func gpuBridgeComputeGatherNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufIndicesPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ inCols: UInt32,
    _ outCols: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufIndicesPtr = bufIndicesPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufIndices = Unmanaged<GPUBuffer>.fromOpaque(bufIndicesPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    guard let (cb, isBatch) = compute.dispatchGatherNB(
        queue: queue, bufIn: bufIn.buffer, bufIndices: bufIndices.buffer,
        bufOut: bufOut.buffer, rows: Int(rows), inCols: Int(inCols), outCols: Int(outCols)
    ) else { return nil }

    if isBatch {
        return Unmanaged.passUnretained(cb).toOpaque()
    } else {
        return Unmanaged.passRetained(cb).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if M == 0 || N == 0 || K == 0 || batchSize == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    // Use MPSMatrixMultiplication for optimized matmul (Apple's hand-tuned BLAS).
    // C = alpha * A × B + beta * C, with alpha=1, beta=0.
    // Fallback to custom MSL kernel when:
    //   - Buffer < 16 bytes (MPS requirement)
    //   - Non-Float32 dtype (MPS only supports float32 in this path)
    let m = Int(M), n = Int(N), k = Int(K)
    let elementSize = MemoryLayout<Float>.size
    let bs = Int(batchSize)
    // MPS requires buffer.length ≥ max(rowBytes*rows, 16).
    // Also: MPS MPSMatrixMultiplication only supports .float32 and .float16.
    // We detect float32 by checking if the required bytes match float32 layout.
    // If the pipeline was compiled for a different dtype, the buffer sizes won't match.
    let aNeeded = m * k * elementSize  // elementSize = 4 (Float32)
    let bNeeded = k * n * elementSize
    let cNeeded = m * n * elementSize
    let useMPS = bufA.buffer.length >= aNeeded
        && bufB.buffer.length >= bNeeded
        && bufC.buffer.length >= cNeeded
        && aNeeded >= 16 && bNeeded >= 16 && cNeeded >= 16

    if !useMPS {
        // Tiny or non-Float32 matrix — use tiled custom MSL kernel
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
        encoder.setComputePipelineState(compute.pipelineState)
        encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
        encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
        encoder.setBuffer(bufC.buffer, offset: 0, index: 2)
        var mV = M, nV = N, kV = K, bsV = batchSize, absV = aBatchStride, bbsV = bBatchStride
        encoder.setBytes(&mV, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nV, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&kV, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&bsV, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&absV, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&bbsV, length: MemoryLayout<UInt32>.size, index: 8)
        // Tiled kernel requires 32×32 threadgroups (uses threadgroup shared memory)
        let ts = 32
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + ts - 1) / ts, height: (m + ts - 1) / ts, depth: bs),
            threadsPerThreadgroup: MTLSize(width: ts, height: ts, depth: 1))
        encoder.endEncoding()
    } else if bs == 1 {
        // Single matrix multiply via MPS
        let matA = MPSMatrix(buffer: bufA.buffer, descriptor: MPSMatrixDescriptor(
            rows: m, columns: k, rowBytes: k * elementSize, dataType: .float32))
        let matB = MPSMatrix(buffer: bufB.buffer, descriptor: MPSMatrixDescriptor(
            rows: k, columns: n, rowBytes: n * elementSize, dataType: .float32))
        let matC = MPSMatrix(buffer: bufC.buffer, descriptor: MPSMatrixDescriptor(
            rows: m, columns: n, rowBytes: n * elementSize, dataType: .float32))

        let mm = MPSMatrixMultiplication(
            device: compute.pipelineState.device,
            transposeLeft: false, transposeRight: false,
            resultRows: m, resultColumns: n, interiorColumns: k,
            alpha: 1.0, beta: 0.0)
        mm.encode(commandBuffer: commandBuffer, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
    } else {
        // Batched matmul — tiled custom MSL kernel with 32×32 threadgroups.
        // Single encoder for all batches (gid.z = batch), vs N separate MPS encodes.
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
        encoder.setComputePipelineState(compute.pipelineState)
        encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
        encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
        encoder.setBuffer(bufC.buffer, offset: 0, index: 2)
        var mV = M, nV = N, kV = K, bsV = batchSize, absV = aBatchStride, bbsV = bBatchStride
        encoder.setBytes(&mV, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nV, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&kV, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&bsV, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&absV, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&bbsV, length: MemoryLayout<UInt32>.size, index: 8)
        // Tiled kernel uses 32×32 threadgroups — dispatch in threadgroups (rounded up)
        let ts = 32
        let tgX = (n + ts - 1) / ts
        let tgY = (m + ts - 1) / ts
        encoder.dispatchThreadgroups(
            MTLSize(width: tgX, height: tgY, depth: bs),
            threadsPerThreadgroup: MTLSize(width: ts, height: ts, depth: 1))
        encoder.endEncoding()
    }

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

@_cdecl("gpu_bridge_compute_matmul_ex_nb")
public func gpuBridgeComputeMatmulExNB(
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
    _ bBatchStride: UInt32,
    _ transposeA: Bool,
    _ transposeB: Bool
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

    let m = Int(M), n = Int(N), k = Int(K), bs = Int(batchSize)
    if m == 0 || n == 0 || k == 0 || bs == 0 {
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    let elementSize = MemoryLayout<Float>.size

    // For transposed matmul, the physical buffer has the original (pre-transpose) layout.
    // MPS reads with the transpose flag — no contiguity copy needed.
    // Physical A dims: if transposeA, buffer is [K,M] row-major; MPS reads as [M,K] transposed.
    let aRows = transposeA ? k : m
    let aCols = transposeA ? m : k
    let bRows = transposeB ? n : k
    let bCols = transposeB ? k : n
    let aNeeded = aRows * aCols * elementSize
    let bNeeded = bRows * bCols * elementSize
    let cNeeded = m * n * elementSize
    let useMPS = bufA.buffer.length >= aNeeded && bufB.buffer.length >= bNeeded
        && bufC.buffer.length >= cNeeded
        && aNeeded >= 16 && bNeeded >= 16 && cNeeded >= 16

    if useMPS && bs == 1 {
        let matA = MPSMatrix(buffer: bufA.buffer, descriptor: MPSMatrixDescriptor(
            rows: aRows, columns: aCols, rowBytes: aCols * elementSize, dataType: .float32))
        let matB = MPSMatrix(buffer: bufB.buffer, descriptor: MPSMatrixDescriptor(
            rows: bRows, columns: bCols, rowBytes: bCols * elementSize, dataType: .float32))
        let matC = MPSMatrix(buffer: bufC.buffer, descriptor: MPSMatrixDescriptor(
            rows: m, columns: n, rowBytes: n * elementSize, dataType: .float32))

        let mm = MPSMatrixMultiplication(
            device: compute.pipelineState.device,
            transposeLeft: transposeA, transposeRight: transposeB,
            resultRows: m, resultColumns: n, interiorColumns: k,
            alpha: 1.0, beta: 0.0)
        mm.encode(commandBuffer: commandBuffer, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
    } else {
        // Fallback: custom MSL kernel (no transpose support in custom kernel,
        // caller must ensure contiguous inputs for this path)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
        encoder.setComputePipelineState(compute.pipelineState)
        encoder.setBuffer(bufA.buffer, offset: 0, index: 0)
        encoder.setBuffer(bufB.buffer, offset: 0, index: 1)
        encoder.setBuffer(bufC.buffer, offset: 0, index: 2)
        var mV = M, nV = N, kV = K, bsV = batchSize, absV = aBatchStride, bbsV = bBatchStride
        encoder.setBytes(&mV, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nV, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&kV, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&bsV, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&absV, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&bbsV, length: MemoryLayout<UInt32>.size, index: 8)
        let w = compute.pipelineState.threadExecutionWidth
        let h = max(compute.pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
        encoder.dispatchThreads(
            MTLSize(width: n, height: m, depth: bs),
            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        encoder.endEncoding()
    }

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if batchSize == 0 || rows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if rows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if rows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if batchSize == 0 || rows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if rows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

@_cdecl("gpu_bridge_compute_softmax_backward_nb")
public func gpuBridgeComputeSoftmaxBackwardNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufGradOutPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeRawPointer?,
    _ bufGradInPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufGradOutPtr = bufGradOutPtr,
          let bufOutPtr = bufOutPtr,
          let bufGradInPtr = bufGradInPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufGradOut = Unmanaged<GPUBuffer>.fromOpaque(bufGradOutPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()
    let bufGradIn = Unmanaged<GPUBuffer>.fromOpaque(bufGradInPtr).takeUnretainedValue()

    if rows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufGradOut.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufGradIn.buffer, offset: 0, index: 2)

    var r = rows, c = cols
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)

    let rowCount = Int(rows)
    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, rowCount)
    let threadGroups = (rowCount + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

@_cdecl("gpu_bridge_compute_layer_norm_backward_nb")
public func gpuBridgeComputeLayerNormBackwardNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufGradOutPtr: UnsafeRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufGammaPtr: UnsafeRawPointer?,
    _ bufGradInPtr: UnsafeMutableRawPointer?,
    _ rows: UInt32,
    _ cols: UInt32,
    _ eps: Float
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufGradOutPtr = bufGradOutPtr,
          let bufInPtr = bufInPtr,
          let bufGammaPtr = bufGammaPtr,
          let bufGradInPtr = bufGradInPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufGradOut = Unmanaged<GPUBuffer>.fromOpaque(bufGradOutPtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufGamma = Unmanaged<GPUBuffer>.fromOpaque(bufGammaPtr).takeUnretainedValue()
    let bufGradIn = Unmanaged<GPUBuffer>.fromOpaque(bufGradInPtr).takeUnretainedValue()

    if rows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufGradOut.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufGamma.buffer, offset: 0, index: 2)
    encoder.setBuffer(bufGradIn.buffer, offset: 0, index: 3)

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return nil }

    blitEncoder.copy(from: src.buffer, sourceOffset: 0, to: dst.buffer, destinationOffset: 0, size: Int(sizeBytes))
    blitEncoder.endEncoding()

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if seqLen == 0 || embedDim == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if outRows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if rows == 0 || outCols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if totalRows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if rows == 0 || totalCols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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

    if rows == 0 || cols == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

@_cdecl("gpu_bridge_compute_binary_nd_offset_nb")
public func gpuBridgeComputeBinaryNDOffsetNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufAPtr: UnsafeRawPointer?,
    _ aByteOffset: UInt32,
    _ bufBPtr: UnsafeRawPointer?,
    _ bByteOffset: UInt32,
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
    if count == 0 {
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufA.buffer, offset: Int(aByteOffset), index: 0)
    encoder.setBuffer(bufB.buffer, offset: Int(bByteOffset), index: 1)
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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

// MARK: - Where (ternary) N-D dispatch

@_cdecl("gpu_bridge_compute_where_nd")
public func gpuBridgeComputeWhereND(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufCondPtr: UnsafeRawPointer?,
    _ bufXPtr: UnsafeRawPointer?,
    _ bufYPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ condStrides: UnsafePointer<UInt32>?,
    _ xStrides: UnsafePointer<UInt32>?,
    _ yStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufCondPtr = bufCondPtr,
          let bufXPtr = bufXPtr,
          let bufYPtr = bufYPtr,
          let bufOutPtr = bufOutPtr,
          let condStrides = condStrides,
          let xStrides = xStrides,
          let yStrides = yStrides,
          let outShape = outShape else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufCond = Unmanaged<GPUBuffer>.fromOpaque(bufCondPtr).takeUnretainedValue()
    let bufX = Unmanaged<GPUBuffer>.fromOpaque(bufXPtr).takeUnretainedValue()
    let bufY = Unmanaged<GPUBuffer>.fromOpaque(bufYPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return 0 }

    guard let commandBuffer = compute.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufCond.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufX.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufY.buffer, offset: 0, index: 2)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 3)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(condStrides, length: arrayBytes, index: 4)
    encoder.setBytes(xStrides, length: arrayBytes, index: 5)
    encoder.setBytes(yStrides, length: arrayBytes, index: 6)
    encoder.setBytes(outShape, length: arrayBytes, index: 7)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 8)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 9)

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

@_cdecl("gpu_bridge_compute_where_nd_nb")
public func gpuBridgeComputeWhereNDNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufCondPtr: UnsafeRawPointer?,
    _ bufXPtr: UnsafeRawPointer?,
    _ bufYPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ condStrides: UnsafePointer<UInt32>?,
    _ xStrides: UnsafePointer<UInt32>?,
    _ yStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufCondPtr = bufCondPtr,
          let bufXPtr = bufXPtr,
          let bufYPtr = bufYPtr,
          let bufOutPtr = bufOutPtr,
          let condStrides = condStrides,
          let xStrides = xStrides,
          let yStrides = yStrides,
          let outShape = outShape else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufCond = Unmanaged<GPUBuffer>.fromOpaque(bufCondPtr).takeUnretainedValue()
    let bufX = Unmanaged<GPUBuffer>.fromOpaque(bufXPtr).takeUnretainedValue()
    let bufY = Unmanaged<GPUBuffer>.fromOpaque(bufYPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufCond.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufX.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufY.buffer, offset: 0, index: 2)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 3)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(condStrides, length: arrayBytes, index: 4)
    encoder.setBytes(xStrides, length: arrayBytes, index: 5)
    encoder.setBytes(yStrides, length: arrayBytes, index: 6)
    encoder.setBytes(outShape, length: arrayBytes, index: 7)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 8)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 9)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

// MARK: - MaskedFill N-D dispatch

@_cdecl("gpu_bridge_compute_masked_fill_nd")
public func gpuBridgeComputeMaskedFillND(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufMaskPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inStrides: UnsafePointer<UInt32>?,
    _ maskStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32,
    _ fillValue: Float
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufMaskPtr = bufMaskPtr,
          let bufOutPtr = bufOutPtr,
          let inStrides = inStrides,
          let maskStrides = maskStrides,
          let outShape = outShape else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufMask = Unmanaged<GPUBuffer>.fromOpaque(bufMaskPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 { return 0 }

    guard let commandBuffer = compute.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufMask.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(inStrides, length: arrayBytes, index: 3)
    encoder.setBytes(maskStrides, length: arrayBytes, index: 4)
    encoder.setBytes(outShape, length: arrayBytes, index: 5)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 6)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 7)
    var fv = fillValue
    encoder.setBytes(&fv, length: MemoryLayout<Float>.size, index: 8)

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

@_cdecl("gpu_bridge_compute_masked_fill_nd_nb")
public func gpuBridgeComputeMaskedFillNDNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufMaskPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ inStrides: UnsafePointer<UInt32>?,
    _ maskStrides: UnsafePointer<UInt32>?,
    _ outShape: UnsafePointer<UInt32>?,
    _ ndim: UInt32,
    _ numel: UInt32,
    _ fillValue: Float
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufMaskPtr = bufMaskPtr,
          let bufOutPtr = bufOutPtr,
          let inStrides = inStrides,
          let maskStrides = maskStrides,
          let outShape = outShape else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufMask = Unmanaged<GPUBuffer>.fromOpaque(bufMaskPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    let count = Int(numel)
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufMask.buffer, offset: 0, index: 1)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 2)

    let arrayBytes = MemoryLayout<UInt32>.size * 8
    encoder.setBytes(inStrides, length: arrayBytes, index: 3)
    encoder.setBytes(maskStrides, length: arrayBytes, index: 4)
    encoder.setBytes(outShape, length: arrayBytes, index: 5)
    var nd = ndim
    encoder.setBytes(&nd, length: MemoryLayout<UInt32>.size, index: 6)
    var n = numel
    encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 7)
    var fv = fillValue
    encoder.setBytes(&fv, length: MemoryLayout<Float>.size, index: 8)

    let threadGroupSize = min(compute.pipelineState.maxTotalThreadsPerThreadgroup, count)
    let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

    encoder.dispatchThreadgroups(
        MTLSize(width: threadGroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
    )
    encoder.endEncoding()

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

// MARK: - Triangular (triu/tril) dispatch

@_cdecl("gpu_bridge_compute_triangular")
public func gpuBridgeComputeTriangular(
    _ computePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ batchSize: UInt32,
    _ rows: UInt32,
    _ cols: UInt32,
    _ diagonal: Int32
) -> Int32 {
    guard let computePtr = computePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if rows == 0 || cols == 0 || batchSize == 0 { return 0 }

    guard let commandBuffer = compute.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var bs = batchSize, r = rows, c = cols, d = diagonal
    encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&d, length: MemoryLayout<Int32>.size, index: 5)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(cols), height: Int(rows), depth: Int(batchSize))

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return commandBuffer.status == .completed ? 0 : -1
}

@_cdecl("gpu_bridge_compute_triangular_nb")
public func gpuBridgeComputeTriangularNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ bufInPtr: UnsafeRawPointer?,
    _ bufOutPtr: UnsafeMutableRawPointer?,
    _ batchSize: UInt32,
    _ rows: UInt32,
    _ cols: UInt32,
    _ diagonal: Int32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let bufInPtr = bufInPtr,
          let bufOutPtr = bufOutPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufIn = Unmanaged<GPUBuffer>.fromOpaque(bufInPtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(bufOutPtr).takeUnretainedValue()

    if rows == 0 || cols == 0 || batchSize == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)
    encoder.setBuffer(bufIn.buffer, offset: 0, index: 0)
    encoder.setBuffer(bufOut.buffer, offset: 0, index: 1)

    var bs = batchSize, r = rows, c = cols, d = diagonal
    encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&d, length: MemoryLayout<Int32>.size, index: 5)

    let w = compute.pipelineState.threadExecutionWidth
    let h = compute.pipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let gridSize = MTLSize(width: Int(cols), height: Int(rows), depth: Int(batchSize))

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
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
    if count == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

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

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

// MARK: - Generic 3D dispatch for CNN ops (conv, pool, batch_norm)

/// Generic 3D dispatch: N input buffers + 1 output buffer + uint32 params + float params + 3D grid.
/// Buffer layout: input[0]..input[N-1] at index 0..N-1, output at index N,
/// uint params as individual setBytes at index N+1..N+upc, float params at N+1+upc..
@_cdecl("gpu_bridge_compute_3d")
public func gpuBridgeCompute3D(
    _ computePtr: UnsafeMutableRawPointer?,
    _ inputBuffers: UnsafePointer<UnsafeRawPointer?>?,
    _ bufferCount: UInt32,
    _ outputPtr: UnsafeMutableRawPointer?,
    _ uintParams: UnsafePointer<UInt32>?,
    _ uintParamCount: UInt32,
    _ floatParams: UnsafePointer<Float>?,
    _ floatParamCount: UInt32,
    _ gridX: UInt32,
    _ gridY: UInt32,
    _ gridZ: UInt32
) -> Int32 {
    guard let computePtr = computePtr,
          let inputBuffers = inputBuffers,
          let outputPtr = outputPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(outputPtr).takeUnretainedValue()

    let n = Int(bufferCount)
    let gx = Int(gridX), gy = Int(gridY), gz = Int(gridZ)
    if gx == 0 || gy == 0 || gz == 0 { return 0 }

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

    // uint32 params: each as individual setBytes
    let upc = Int(uintParamCount)
    if let params = uintParams, upc > 0 {
        for i in 0..<upc {
            var val = params[i]
            encoder.setBytes(&val, length: MemoryLayout<UInt32>.size, index: n + 1 + i)
        }
    }

    // float params: each as individual setBytes continuing after uint params
    let fpc = Int(floatParamCount)
    if let fparams = floatParams, fpc > 0 {
        for i in 0..<fpc {
            var val = fparams[i]
            encoder.setBytes(&val, length: MemoryLayout<Float>.size, index: n + 1 + upc + i)
        }
    }

    let w = compute.pipelineState.threadExecutionWidth
    let h = max(compute.pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
    let threadsPerGroup = MTLSize(width: min(w, gx), height: min(h, gy), depth: 1)
    let gridSize = MTLSize(width: gx, height: gy, depth: gz)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return commandBuffer.status == .completed ? 0 : -1
}

@_cdecl("gpu_bridge_compute_3d_nb")
public func gpuBridgeCompute3DNB(
    _ computePtr: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ inputBuffers: UnsafePointer<UnsafeRawPointer?>?,
    _ bufferCount: UInt32,
    _ outputPtr: UnsafeMutableRawPointer?,
    _ uintParams: UnsafePointer<UInt32>?,
    _ uintParamCount: UInt32,
    _ floatParams: UnsafePointer<Float>?,
    _ floatParamCount: UInt32,
    _ gridX: UInt32,
    _ gridY: UInt32,
    _ gridZ: UInt32
) -> UnsafeMutableRawPointer? {
    guard let computePtr = computePtr,
          let queuePtr = queuePtr,
          let inputBuffers = inputBuffers,
          let outputPtr = outputPtr else { return nil }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(outputPtr).takeUnretainedValue()

    let n = Int(bufferCount)
    let gx = Int(gridX), gy = Int(gridY), gz = Int(gridZ)
    if gx == 0 || gy == 0 || gz == 0 {
        let ctxId = currentContextId
        contextLock.lock()
        if ctxId > 0, let batchCB = batchContexts[ctxId] {
            contextLock.unlock()
            return Unmanaged.passUnretained(batchCB as AnyObject).toOpaque()
        }
        contextLock.unlock()
        batchLock.lock()
        let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
        batchLock.unlock()
        return ptr
    }

    let commandBuffer: MTLCommandBuffer
    let isBatch: Bool
    let ctxId = currentContextId
    contextLock.lock()
    if ctxId > 0, let batchCB = batchContexts[ctxId] {
        commandBuffer = batchCB
        isBatch = true
        contextLock.unlock()
    } else {
        contextLock.unlock()
        batchLock.lock()
        if let batchCB = activeBatchCommandBuffer {
            commandBuffer = batchCB
            isBatch = true
            batchLock.unlock()
        } else {
            batchLock.unlock()
            guard let cb = queue.makeCommandBuffer() else { return nil }
            commandBuffer = cb
            isBatch = false
        }
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }

    encoder.setComputePipelineState(compute.pipelineState)

    for i in 0..<n {
        guard let bufPtr = inputBuffers[i] else { return nil }
        let buf = Unmanaged<GPUBuffer>.fromOpaque(bufPtr).takeUnretainedValue()
        encoder.setBuffer(buf.buffer, offset: 0, index: i)
    }
    encoder.setBuffer(bufOut.buffer, offset: 0, index: n)

    let upc = Int(uintParamCount)
    if let params = uintParams, upc > 0 {
        for i in 0..<upc {
            var val = params[i]
            encoder.setBytes(&val, length: MemoryLayout<UInt32>.size, index: n + 1 + i)
        }
    }

    let fpc = Int(floatParamCount)
    if let fparams = floatParams, fpc > 0 {
        for i in 0..<fpc {
            var val = fparams[i]
            encoder.setBytes(&val, length: MemoryLayout<Float>.size, index: n + 1 + upc + i)
        }
    }

    let w = compute.pipelineState.threadExecutionWidth
    let h = max(compute.pipelineState.maxTotalThreadsPerThreadgroup / w, 1)
    let threadsPerGroup = MTLSize(width: min(w, gx), height: min(h, gy), depth: 1)
    let gridSize = MTLSize(width: gx, height: gy, depth: gz)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()

    if isBatch {
        return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    } else {
        commandBuffer.commit()
        return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
    }
}

// MARK: - Batch encoding (single command buffer per eval)

@_cdecl("gpu_bridge_begin_batch")
public func gpuBridgeBeginBatch(_ queuePtr: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let queuePtr = queuePtr else { return nil }
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    batchLock.lock()
    defer { batchLock.unlock() }
    guard activeBatchCommandBuffer == nil else { return nil }
    guard let cb = queue.makeCommandBuffer() else { return nil }
    activeBatchCommandBuffer = cb
    return Unmanaged.passUnretained(cb as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_end_batch")
public func gpuBridgeEndBatch() -> UnsafeMutableRawPointer? {
    batchLock.lock()
    defer { batchLock.unlock() }
    guard let cb = activeBatchCommandBuffer else { return nil }
    cb.commit()
    activeBatchCommandBuffer = nil
    return Unmanaged.passRetained(cb as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_abort_batch")
public func gpuBridgeAbortBatch() {
    batchLock.lock()
    defer { batchLock.unlock() }
    activeBatchCommandBuffer = nil
}

// MARK: - Concurrent queue pool

@_cdecl("gpu_bridge_get_queue")
public func gpuBridgeGetQueue(_ devicePtr: UnsafeRawPointer?, _ index: UInt32) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr else { return nil }
    let gpuDevice = getGPUDevice(from: devicePtr)

    queuePoolLock.lock()
    defer { queuePoolLock.unlock() }

    let idx = Int(index) % maxQueueCount
    // Lazily create queues as needed
    while queuePool.count <= idx {
        guard let q = gpuDevice.device.makeCommandQueue() else { return nil }
        queuePool.append(q)
    }
    return Unmanaged.passUnretained(queuePool[idx]).toOpaque()
}

// MARK: - Batch context system

@_cdecl("gpu_bridge_set_active_context")
public func gpuBridgeSetActiveContext(_ contextId: UInt32) {
    currentContextId = contextId
}

@_cdecl("gpu_bridge_set_batch_context")
public func gpuBridgeSetBatchContext(_ contextId: UInt32, _ queuePtr: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let queuePtr = queuePtr else { return nil }
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    guard let cb = queue.makeCommandBuffer() else { return nil }

    contextLock.lock()
    batchContexts[contextId] = cb
    contextLock.unlock()

    return Unmanaged.passUnretained(cb).toOpaque()
}

@_cdecl("gpu_bridge_commit_batch_context")
public func gpuBridgeCommitBatchContext(_ contextId: UInt32) -> UnsafeMutableRawPointer? {
    contextLock.lock()
    guard let cb = batchContexts.removeValue(forKey: contextId) else {
        contextLock.unlock()
        return nil
    }
    contextLock.unlock()

    cb.commit()
    return Unmanaged.passRetained(cb as AnyObject).toOpaque()
}

// MARK: - MTLEvent synchronization

@_cdecl("gpu_bridge_create_event")
public func gpuBridgeCreateEvent(_ devicePtr: UnsafeRawPointer?) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr else { return nil }
    let gpuDevice = getGPUDevice(from: devicePtr)
    guard let event = gpuDevice.device.makeEvent() else { return nil }
    return Unmanaged.passRetained(event as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_encode_signal_event")
public func gpuBridgeEncodeSignalEvent(_ cbPtr: UnsafeMutableRawPointer?, _ eventPtr: UnsafeMutableRawPointer?, _ value: UInt64) {
    guard let cbPtr = cbPtr, let eventPtr = eventPtr else { return }
    let cb = Unmanaged<AnyObject>.fromOpaque(cbPtr).takeUnretainedValue() as! MTLCommandBuffer
    let event = Unmanaged<AnyObject>.fromOpaque(eventPtr).takeUnretainedValue() as! MTLEvent
    cb.encodeSignalEvent(event, value: value)
}

@_cdecl("gpu_bridge_encode_wait_event")
public func gpuBridgeEncodeWaitEvent(_ cbPtr: UnsafeMutableRawPointer?, _ eventPtr: UnsafeMutableRawPointer?, _ value: UInt64) {
    guard let cbPtr = cbPtr, let eventPtr = eventPtr else { return }
    let cb = Unmanaged<AnyObject>.fromOpaque(cbPtr).takeUnretainedValue() as! MTLCommandBuffer
    let event = Unmanaged<AnyObject>.fromOpaque(eventPtr).takeUnretainedValue() as! MTLEvent
    cb.encodeWaitForEvent(event, value: value)
}

@_cdecl("gpu_bridge_destroy_event")
public func gpuBridgeDestroyEvent(_ eventPtr: UnsafeMutableRawPointer?) {
    guard let eventPtr = eventPtr else { return }
    Unmanaged<AnyObject>.fromOpaque(eventPtr).release()
}
