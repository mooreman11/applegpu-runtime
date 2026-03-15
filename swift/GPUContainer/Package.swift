// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GPUContainer",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(url: "https://github.com/apple/containerization.git", from: "0.26.5"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "gpu-container",
            dependencies: [
                .product(name: "Containerization", package: "containerization"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources"
        ),
    ]
)
