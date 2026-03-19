use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let swift_dir = workspace_root.join("swift");

    // Build Swift static library in release mode
    let status = Command::new("swift")
        .args(["build", "-c", "release"])
        .current_dir(&swift_dir)
        .status()
        .expect("Failed to run swift build. Is Swift installed?");

    assert!(status.success(), "swift build failed");

    let swift_build_dir = swift_dir.join(".build/release");

    // Link the static library
    println!(
        "cargo:rustc-link-search=native={}",
        swift_build_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=AppleGPUBridge");

    // Set rpath so the dylib is found at runtime (Cargo test binaries, Python extensions)
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        swift_build_dir.display()
    );

    // Also copy dylib next to Python extensions for @loader_path resolution
    let dylib_src = swift_build_dir.join("libAppleGPUBridge.dylib");
    if dylib_src.exists() {
        let python_dir = workspace_root.join("python/applegpu_runtime");
        let _ = std::fs::copy(&dylib_src, python_dir.join("libAppleGPUBridge.dylib"));
        let backend_dir = workspace_root.join("backend_cpp");
        if backend_dir.exists() {
            let _ = std::fs::copy(&dylib_src, backend_dir.join("libAppleGPUBridge.dylib"));
        }
    }

    // Link Swift runtime and Apple frameworks
    println!("cargo:rustc-link-lib=dylib=swiftCore");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Find Swift library path for swiftCore
    let swift_path_output = Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .expect("Failed to run xcrun");
    let sdk_path = String::from_utf8(swift_path_output.stdout)
        .unwrap()
        .trim()
        .to_string();

    // Swift toolchain lib path
    let swift_lib_output = Command::new("xcrun")
        .args(["--toolchain", "default", "--find", "swift"])
        .output()
        .expect("Failed to find swift");
    let swift_bin = String::from_utf8(swift_lib_output.stdout)
        .unwrap()
        .trim()
        .to_string();
    let swift_lib_dir = std::path::Path::new(&swift_bin)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("lib/swift/macosx");

    println!("cargo:rustc-link-search=native={}", swift_lib_dir.display());
    println!(
        "cargo:rustc-link-search=native={}/usr/lib/swift",
        sdk_path
    );

    // Rerun if Swift sources change (absolute paths)
    println!(
        "cargo:rerun-if-changed={}",
        swift_dir.join("Sources").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        swift_dir.join("Package.swift").display()
    );
}
