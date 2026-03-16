// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "GPUContainer",
    platforms: [.macOS("26.0")],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        // .package(url: "https://github.com/apple/containerization.git", from: "0.1.0"),
    ],
    targets: [
        .executableTarget(
            name: "gpu-container",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources"
        ),
    ]
)
