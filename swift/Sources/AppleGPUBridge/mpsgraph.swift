/// MPSGraph integration for fused GPU execution.
///
/// Deserializes the compiled graph bytecode (same wire format as Rust's
/// compiled_graph.rs), builds an MPSGraph, caches it by shape signature,
/// and executes with zero-copy MTLBuffer bindings.
///
/// C ABI exports:
///   gpu_bridge_mpsgraph_build  — build/cache a graph from bytecode
///   gpu_bridge_mpsgraph_run    — execute a cached graph with buffer bindings
///   gpu_bridge_mpsgraph_destroy — free a cached graph

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

// MARK: - Op codes (must match compiled_graph.rs)

private let OP_ADD: UInt8 = 0
private let OP_SUB: UInt8 = 1
private let OP_MUL: UInt8 = 2
private let OP_DIV: UInt8 = 3
private let OP_MATMUL: UInt8 = 4
private let OP_RELU: UInt8 = 5
private let OP_NEG: UInt8 = 6
private let OP_THRESHOLD_BACKWARD: UInt8 = 7
private let OP_SCALAR_MUL: UInt8 = 8
private let OP_MEAN_ALL: UInt8 = 9
private let OP_SUM_DIM: UInt8 = 10
private let OP_TRANSPOSE: UInt8 = 11
private let OP_VIEW: UInt8 = 12
private let OP_ADDMM: UInt8 = 13
private let OP_IDENTITY: UInt8 = 255

// MARK: - Cached graph structure

private final class CachedMPSGraph {
    let graph: MPSGraph
    let placeholders: [MPSGraphTensor]  // ordered input placeholders
    let outputs: [MPSGraphTensor]       // ordered output tensors
    let inputShapes: [[NSNumber]]       // shapes for feed construction
    let inputDtypes: [MPSDataType]

    init(graph: MPSGraph, placeholders: [MPSGraphTensor],
         outputs: [MPSGraphTensor], inputShapes: [[NSNumber]],
         inputDtypes: [MPSDataType]) {
        self.graph = graph
        self.placeholders = placeholders
        self.outputs = outputs
        self.inputShapes = inputShapes
        self.inputDtypes = inputDtypes
    }
}

// MARK: - Wire format parsing

private func readU8(_ data: UnsafePointer<UInt8>, _ cursor: inout Int) -> UInt8 {
    let v = data[cursor]; cursor += 1; return v
}

private func readU16LE(_ data: UnsafePointer<UInt8>, _ cursor: inout Int) -> UInt16 {
    let v = UInt16(data[cursor]) | (UInt16(data[cursor+1]) << 8)
    cursor += 2; return v
}

private func readU64LE(_ data: UnsafePointer<UInt8>, _ cursor: inout Int) -> UInt64 {
    var v: UInt64 = 0
    for i in 0..<8 { v |= UInt64(data[cursor+i]) << (i * 8) }
    cursor += 8; return v
}

private func readF32LE(_ data: UnsafePointer<UInt8>, _ cursor: inout Int) -> Float {
    var bits: UInt32 = 0
    for i in 0..<4 { bits |= UInt32(data[cursor+i]) << (i * 8) }
    cursor += 4; return Float(bitPattern: bits)
}

// MARK: - Graph builder

/// Build an MPSGraph from serialized bytecode.
/// Returns nil if any op is unsupported.
private func buildGraph(
    device: MTLDevice,
    opsData: UnsafePointer<UInt8>, opsLen: Int,
    nInputs: Int,
    inputShapes: [[NSNumber]],
    inputDtypes: [MPSDataType],
    outputIndices: [Int]
) -> CachedMPSGraph? {
    let graph = MPSGraph()

    // Create input placeholders
    var nodes: [MPSGraphTensor] = []
    var placeholders: [MPSGraphTensor] = []
    for i in 0..<nInputs {
        let ph = graph.placeholder(
            shape: inputShapes[i],
            dataType: inputDtypes[i],
            name: "input_\(i)")
        placeholders.append(ph)
        nodes.append(ph)
    }

    // Parse and build ops
    var cursor = 0
    while cursor < opsLen {
        let opCode = readU8(opsData, &cursor)
        let nInp = Int(readU8(opsData, &cursor))

        var inputIndices: [Int] = []
        for _ in 0..<nInp {
            inputIndices.append(Int(readU16LE(opsData, &cursor)))
        }

        let outNdim = Int(readU8(opsData, &cursor))
        var outShape: [NSNumber] = []
        for _ in 0..<outNdim {
            outShape.append(NSNumber(value: Int(readU64LE(opsData, &cursor))))
        }

        let _ = readU8(opsData, &cursor) // dtype wire (we use input dtypes)

        let paramsLen = Int(readU8(opsData, &cursor))
        var params: [Float] = []
        for _ in 0..<paramsLen {
            params.append(readF32LE(opsData, &cursor))
        }

        // Build MPSGraph operation
        if opCode == OP_IDENTITY {
            nodes.append(nodes[inputIndices[0]])
            continue
        }

        guard let result = buildOp(
            graph: graph, opCode: opCode, nodes: nodes,
            inputIndices: inputIndices, outShape: outShape, params: params
        ) else {
            return nil // Unsupported op — caller falls back to per-op execution
        }

        nodes.append(result)
    }

    // Collect output tensors
    var outputs: [MPSGraphTensor] = []
    for idx in outputIndices {
        guard idx < nodes.count else { return nil }
        outputs.append(nodes[idx])
    }

    return CachedMPSGraph(
        graph: graph, placeholders: placeholders,
        outputs: outputs, inputShapes: inputShapes,
        inputDtypes: inputDtypes)
}

/// Map a single op code to an MPSGraph operation.
private func buildOp(
    graph: MPSGraph, opCode: UInt8, nodes: [MPSGraphTensor],
    inputIndices: [Int], outShape: [NSNumber], params: [Float]
) -> MPSGraphTensor? {
    let inp = { (i: Int) -> MPSGraphTensor in nodes[inputIndices[i]] }

    switch opCode {
    case OP_ADD:
        return graph.addition(inp(0), inp(1), name: nil)
    case OP_SUB:
        return graph.subtraction(inp(0), inp(1), name: nil)
    case OP_MUL:
        return graph.multiplication(inp(0), inp(1), name: nil)
    case OP_DIV:
        return graph.division(inp(0), inp(1), name: nil)
    case OP_MATMUL:
        return graph.matrixMultiplication(
            primary: inp(0), secondary: inp(1), name: nil)
    case OP_RELU:
        return graph.reLU(with: inp(0), name: nil)
    case OP_NEG:
        return graph.negative(with: inp(0), name: nil)
    case OP_THRESHOLD_BACKWARD:
        // grad * (input > threshold)
        let threshold = params.first ?? 0.0
        let threshConst = graph.constant(Double(threshold), dataType: inp(1).dataType)
        let mask = graph.greaterThan(inp(1), threshConst, name: nil)
        let maskFloat = graph.cast(mask, to: inp(0).dataType, name: "mask_cast")
        return graph.multiplication(inp(0), maskFloat, name: nil)
    case OP_SCALAR_MUL:
        let scale = params.first ?? 1.0
        let scaleConst = graph.constant(Double(scale), dataType: inp(0).dataType)
        return graph.multiplication(inp(0), scaleConst, name: nil)
    case OP_MEAN_ALL:
        // Reduce all dims to scalar [1]
        let rank = inp(0).shape?.count ?? 1
        let axes = (0..<rank).map { NSNumber(value: $0) }
        return graph.mean(of: inp(0), axes: axes, name: nil)
    case OP_SUM_DIM:
        let dim = Int(params.first ?? 0)
        let keepdim = (params.count > 1 && params[1] != 0.0)
        let result = graph.reductionSum(with: inp(0), axis: dim, name: nil)
        if !keepdim {
            // squeeze the reduced dim
            return graph.squeeze(result, axis: dim, name: nil)
        }
        return result
    case OP_TRANSPOSE:
        // Transpose last two dims
        let rank = inp(0).shape?.count ?? 2
        return graph.transposeTensor(inp(0),
            dimension: rank - 2, withDimension: rank - 1, name: nil)
    case OP_VIEW:
        return graph.reshape(inp(0), shape: outShape, name: nil)
    case OP_ADDMM:
        // bias + mm(mat1, mat2)
        let mm = graph.matrixMultiplication(
            primary: inp(1), secondary: inp(2), name: nil)
        return graph.addition(mm, inp(0), name: nil)
    default:
        return nil // Unsupported
    }
}

// MARK: - C ABI exports

@_cdecl("gpu_bridge_mpsgraph_build")
public func gpuBridgeMPSGraphBuild(
    _ devicePtr: UnsafeMutableRawPointer?,
    _ opsData: UnsafePointer<UInt8>?,
    _ opsLen: UInt32,
    _ nInputs: UInt32,
    _ inputShapesFlat: UnsafePointer<Int64>?,  // flattened shapes
    _ inputNdims: UnsafePointer<UInt32>?,       // ndim per input
    _ inputDtypesRaw: UnsafePointer<UInt32>?,   // DType wire per input
    _ nOutputs: UInt32,
    _ outputIndicesPtr: UnsafePointer<UInt16>?
) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr,
          let opsData = opsData,
          let inputShapesFlat = inputShapesFlat,
          let inputNdims = inputNdims,
          let inputDtypesRaw = inputDtypesRaw,
          let outputIndicesPtr = outputIndicesPtr else { return nil }

    let gpuDevice = Unmanaged<GPUDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let device = gpuDevice.device

    // Parse input shapes
    var inputShapes: [[NSNumber]] = []
    var shapeOffset = 0
    for i in 0..<Int(nInputs) {
        let ndim = Int(inputNdims[i])
        var shape: [NSNumber] = []
        for j in 0..<ndim {
            shape.append(NSNumber(value: Int(inputShapesFlat[shapeOffset + j])))
        }
        shapeOffset += ndim
        inputShapes.append(shape)
    }

    // Parse input dtypes (wire: 0=f32, 1=f16, 10=bf16)
    var inputDtypes: [MPSDataType] = []
    for i in 0..<Int(nInputs) {
        let wire = inputDtypesRaw[i]
        switch wire {
        case 0:  inputDtypes.append(.float32)
        case 1:  inputDtypes.append(.float16)
        case 10: inputDtypes.append(.bFloat16)
        default: inputDtypes.append(.float32)
        }
    }

    // Parse output indices
    var outputIndices: [Int] = []
    for i in 0..<Int(nOutputs) {
        outputIndices.append(Int(outputIndicesPtr[i]))
    }

    guard let cached = buildGraph(
        device: device,
        opsData: opsData, opsLen: Int(opsLen),
        nInputs: Int(nInputs),
        inputShapes: inputShapes,
        inputDtypes: inputDtypes,
        outputIndices: outputIndices
    ) else {
        NSLog("[mpsgraph] build failed — unsupported op or invalid graph")
        return nil
    }
    NSLog("[mpsgraph] built graph: %d inputs, %d outputs", nInputs, nOutputs)

    return Unmanaged.passRetained(cached as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_mpsgraph_run")
public func gpuBridgeMPSGraphRun(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ queuePtr: UnsafeMutableRawPointer?,
    _ inputBuffers: UnsafePointer<UnsafeRawPointer?>?,
    _ nInputs: UInt32,
    _ outputBuffers: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ nOutputs: UInt32
) -> Int32 {
    guard let graphHandle = graphHandle,
          let queuePtr = queuePtr,
          let inputBuffers = inputBuffers,
          let outputBuffers = outputBuffers else { return -1 }

    let cached = Unmanaged<CachedMPSGraph>.fromOpaque(graphHandle).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()

    // Build feeds dictionary: placeholder → MPSGraphTensorData backed by input MTLBuffer
    var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
    for i in 0..<Int(nInputs) {
        guard let bufPtr = inputBuffers[i] else { return -2 }
        let gpuBuf = Unmanaged<GPUBufferBase>.fromOpaque(bufPtr).takeUnretainedValue()
        let tensorData = MPSGraphTensorData(
            gpuBuf.buffer,
            shape: cached.inputShapes[i],
            dataType: cached.inputDtypes[i])
        feeds[cached.placeholders[i]] = tensorData
    }

    // Execute graph using the command queue (synchronous — waits for completion)
    let results = cached.graph.run(
        with: queue,
        feeds: feeds,
        targetTensors: cached.outputs,
        targetOperations: nil)

    // Copy results to pre-allocated output buffers using a blit encoder
    guard let cb = queue.makeCommandBuffer() else { return -3 }
    for i in 0..<Int(nOutputs) {
        guard let outBufPtr = outputBuffers[i] else { return -4 }
        let outGpuBuf = Unmanaged<GPUBufferBase>.fromOpaque(outBufPtr).takeUnretainedValue()
        guard let resultData = results[cached.outputs[i]] else { return -5 }

        // Export result to our pre-allocated buffer
        resultData.mpsndarray().exportData(
            with: cb,
            to: outGpuBuf.buffer,
            destinationDataType: cached.inputDtypes.first ?? .float32,
            offset: 0,
            rowStrides: nil)
    }
    cb.commit()
    cb.waitUntilCompleted()

    return 0
}

@_cdecl("gpu_bridge_mpsgraph_destroy")
public func gpuBridgeMPSGraphDestroy(_ graphHandle: UnsafeMutableRawPointer?) {
    guard let graphHandle = graphHandle else { return }
    Unmanaged<CachedMPSGraph>.fromOpaque(graphHandle).release()
}
