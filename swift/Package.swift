// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AppleGPUBridge",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "AppleGPUBridge",
            type: .dynamic,
            targets: ["AppleGPUBridge"]
        ),
    ],
    targets: [
        .target(
            name: "AppleGPUBridge",
            path: "Sources/AppleGPUBridge",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
            ]
        ),
        .testTarget(
            name: "AppleGPUBridgeTests",
            dependencies: ["AppleGPUBridge"]
        ),
    ]
)
