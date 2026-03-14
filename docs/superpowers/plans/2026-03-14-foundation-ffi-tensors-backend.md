# Foundation: FFI Wiring, Tensor Types, and Backend Selection

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the three layers together (Rust ↔ Swift FFI), define core tensor types, and implement backend selection so `gpu.init_backend()` and `gpu.device_name()` work end-to-end from Python through Rust to Metal.

**Architecture:** `build.rs` compiles and links the Swift static library into the Rust core. Rust FFI module wraps the unsafe C calls in safe abstractions. A `Backend` enum and `init_backend()` function handle backend selection (MLX-native only for now, VM backend stubbed). Tensor type definitions establish the foundation for all future GPU operations.

**Tech Stack:** Rust (cargo, FFI), Swift (SwiftPM, Metal, `@_cdecl`), Python (PyO3, maturin/uv)

---

## File Structure

### New Files
- `crates/core/src/device.rs` — Safe Rust wrapper around FFI device handle (lifecycle, queries)
- `crates/core/src/backend.rs` — `Backend` enum, `init_backend()`, backend selection logic
- `crates/core/src/tensor.rs` — `DType`, `Shape`, `TensorMeta` types
- `crates/core/src/error.rs` — `GpuError` enum, `Result<T>` alias
- `crates/core/tests/device_integration.rs` — Integration test: Rust → Swift → Metal device
- `python/tests/test_backend.py` — Python test: backend init, device name, and selection
- `python/tests/test_tensor_types.py` — Python test: tensor metadata types

### Modified Files
- `crates/core/build.rs` — Invoke `swift build`, link `libAppleGPUBridge.a` + Swift runtime
- `crates/core/src/ffi.rs` — Uncomment `extern "C"` block, add safe wrappers
- `crates/core/src/lib.rs` — Add new module declarations
- `crates/core/Cargo.toml` — Add `once_cell` dependency (for stable `get_or_try_init`)
- `swift/Sources/AppleGPUBridge/bridge.swift` — Fix dangling pointer in `namePtr`
- `crates/python/src/lib.rs` — Expose `device_name()`, `init_backend()`, tensor types to Python
- `python/applegpu_runtime/__init__.py` — Export new functions
- `Makefile` — Update `build-python` to depend on Swift build

---

## Chunk 1: Wire Rust ↔ Swift FFI

### Task 1: Fix Swift dangling pointer and add once_cell dependency

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/bridge.swift`
- Modify: `crates/core/Cargo.toml`

- [ ] **Step 1: Fix Swift `namePtr` to use stable allocated memory**

The current `namePtr` computed property uses `withUnsafeBufferPointer` which returns a pointer
only valid within the closure scope. Replace with a stable `UnsafeMutablePointer`:

In `swift/Sources/AppleGPUBridge/bridge.swift`, replace the `GPUDevice` class:

```swift
import Foundation
import Metal

/// Internal class wrapping MTLDevice.
final class GPUDevice {
    let device: MTLDevice
    private let nameCString: UnsafeMutablePointer<CChar>

    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        self.device = device
        let name = device.name
        self.nameCString = strdup(name)!
    }

    deinit {
        free(nameCString)
    }

    var namePtr: UnsafePointer<CChar> {
        UnsafePointer(nameCString)
    }
}
```

- [ ] **Step 2: Add `once_cell` to Cargo.toml**

In `crates/core/Cargo.toml`, add under `[dependencies]`:

```toml
[dependencies]
once_cell = "1"
```

- [ ] **Step 3: Run Swift tests to verify fix**

Run: `cd swift && swift test 2>&1`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add swift/Sources/AppleGPUBridge/bridge.swift crates/core/Cargo.toml
git commit -m "fix: use stable allocated memory for device name, add once_cell dep"
```

---

### Task 2: Update build.rs to compile and link the Swift static library

**Files:**
- Modify: `crates/core/build.rs`

- [ ] **Step 1: Write the build.rs that invokes swift build and links the library**

```rust
// crates/core/build.rs
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
    println!("cargo:rustc-link-lib=static=AppleGPUBridge");

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
```

- [ ] **Step 2: Run `cargo build -p applegpu-core` to verify linkage**

Run: `cargo build -p applegpu-core 2>&1`
Expected: Build succeeds (swift build runs, library links)

- [ ] **Step 3: Commit**

```bash
git add crates/core/build.rs
git commit -m "feat: wire build.rs to compile and link Swift static library"
```

---

### Task 3: Uncomment FFI declarations and add error type

**Files:**
- Create: `crates/core/src/error.rs`
- Modify: `crates/core/src/ffi.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Create error module with types and tests**

Create `crates/core/src/error.rs`:

```rust
/// GPU runtime error type.
#[derive(Debug)]
pub enum GpuError {
    /// Metal device not available (e.g. headless CI, no GPU)
    DeviceNotAvailable,
    /// Backend not initialized
    BackendNotInitialized,
    /// Invalid tensor specification
    InvalidTensor(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DeviceNotAvailable => write!(f, "Metal GPU device not available"),
            GpuError::BackendNotInitialized => write!(f, "Backend not initialized, call init_backend() first"),
            GpuError::InvalidTensor(msg) => write!(f, "Invalid tensor: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

pub type Result<T> = std::result::Result<T, GpuError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = GpuError::DeviceNotAvailable;
        assert_eq!(e.to_string(), "Metal GPU device not available");
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuError>();
    }
}
```

- [ ] **Step 2: Uncomment and update ffi.rs**

Replace `crates/core/src/ffi.rs` with:

```rust
use std::ffi::CStr;

/// Opaque handle to a GPU device from the Swift side.
#[repr(C)]
pub struct GPUDeviceHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_device() -> *mut GPUDeviceHandle;
    pub fn gpu_bridge_destroy_device(device: *mut GPUDeviceHandle);
    pub fn gpu_bridge_device_name(device: *const GPUDeviceHandle) -> *const std::ffi::c_char;
}

/// Safe wrapper: create a Metal device. Returns None if no GPU available.
pub fn create_device() -> Option<*mut GPUDeviceHandle> {
    let ptr = unsafe { gpu_bridge_create_device() };
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}

/// Safe wrapper: destroy a Metal device.
///
/// # Safety
/// The pointer must have been returned by `create_device()` and not yet destroyed.
pub fn destroy_device(device: *mut GPUDeviceHandle) {
    unsafe { gpu_bridge_destroy_device(device) }
}

/// Safe wrapper: get the device name as a Rust string.
///
/// # Safety
/// The pointer must be a valid device handle.
pub fn device_name(device: *const GPUDeviceHandle) -> Option<String> {
    let name_ptr = unsafe { gpu_bridge_device_name(device) };
    if name_ptr.is_null() {
        None
    } else {
        let c_str = unsafe { CStr::from_ptr(name_ptr) };
        Some(c_str.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_query_device() {
        if let Some(device) = create_device() {
            let name = device_name(device);
            assert!(name.is_some());
            assert!(!name.unwrap().is_empty());
            destroy_device(device);
        }
        // If no Metal GPU (CI), test passes by skipping
    }
}
```

- [ ] **Step 3: Update lib.rs to add new modules**

```rust
// crates/core/src/lib.rs
pub mod error;
pub mod ffi;
pub mod scheduler;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        assert_eq!(super::version(), "0.1.0");
    }
}
```

- [ ] **Step 4: Run tests to verify FFI works**

Run: `cargo test --workspace 2>&1`
Expected: All tests pass including the new `create_and_query_device` test

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/error.rs crates/core/src/ffi.rs crates/core/src/lib.rs
git commit -m "feat: uncomment FFI declarations, add safe wrappers and error types"
```

---

### Task 4: Create Device abstraction with RAII

**Files:**
- Create: `crates/core/src/device.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Create Device module with RAII wrapper and tests**

Create `crates/core/src/device.rs`:

```rust
use crate::error::{GpuError, Result};
use crate::ffi;

/// A Metal GPU device. Wraps the Swift-side device handle with RAII.
pub struct Device {
    handle: *mut ffi::GPUDeviceHandle,
}

// Safety: the Swift GPUDevice is thread-safe (MTLDevice is thread-safe)
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Create a new Metal GPU device.
    pub fn new() -> Result<Self> {
        ffi::create_device()
            .map(|handle| Device { handle })
            .ok_or(GpuError::DeviceNotAvailable)
    }

    /// Get the device name (e.g. "Apple M1 Pro").
    pub fn name(&self) -> String {
        ffi::device_name(self.handle)
            .unwrap_or_else(|| "Unknown".to_string())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        ffi::destroy_device(self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_lifecycle() {
        match Device::new() {
            Ok(device) => {
                let name = device.name();
                assert!(!name.is_empty());
                assert_ne!(name, "Unknown");
                // Drop cleans up automatically
            }
            Err(GpuError::DeviceNotAvailable) => {
                // No Metal GPU (CI) — pass
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn device_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Device>();
    }
}
```

- [ ] **Step 2: Add `device` module to lib.rs**

Add `pub mod device;` to `crates/core/src/lib.rs` after `pub mod error;`.

- [ ] **Step 3: Run tests**

Run: `cargo test --workspace 2>&1`
Expected: All tests pass including `device_lifecycle`

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/device.rs crates/core/src/lib.rs
git commit -m "feat: add Device struct with RAII and safe Metal device access"
```

---

### Task 5: Integration test for Rust → Swift → Metal

**Files:**
- Create: `crates/core/tests/device_integration.rs`

- [ ] **Step 1: Write integration test**

```rust
// crates/core/tests/device_integration.rs
use applegpu_core::device::Device;

#[test]
fn metal_device_name_contains_apple() {
    match Device::new() {
        Ok(device) => {
            let name = device.name();
            // Apple Silicon device names contain "Apple"
            assert!(
                name.contains("Apple"),
                "Expected device name to contain 'Apple', got: {}",
                name
            );
        }
        Err(_) => {
            // No Metal GPU available (CI) — skip
        }
    }
}

#[test]
fn multiple_devices_independent() {
    let d1 = Device::new();
    let d2 = Device::new();
    match (d1, d2) {
        (Ok(dev1), Ok(dev2)) => {
            assert_eq!(dev1.name(), dev2.name());
        }
        _ => {
            // No GPU — skip
        }
    }
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test --test device_integration 2>&1`
Expected: PASS (on Apple Silicon; skipped on CI)

- [ ] **Step 3: Commit**

```bash
git add crates/core/tests/device_integration.rs
git commit -m "test: add Rust-to-Swift-to-Metal integration tests"
```

---

## Chunk 2: Backend Selection

### Task 6: Implement Backend enum and init_backend()

**Files:**
- Create: `crates/core/src/backend.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Create backend module with Backend enum, Runtime, and tests**

Create `crates/core/src/backend.rs`:

```rust
use crate::device::Device;
use crate::error::{GpuError, Result};
use once_cell::sync::OnceCell;

/// Available GPU backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Direct MLX → Metal execution (default, high performance)
    Mlx,
    /// VM-mediated Metal execution via Apple Virtualization Framework
    Vm,
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mlx" => Ok(Backend::Mlx),
            "vm" => Ok(Backend::Vm),
            other => Err(format!("Unknown backend: {}", other)),
        }
    }
}

/// Global runtime state.
static RUNTIME: OnceCell<Runtime> = OnceCell::new();

/// Runtime holds the initialized backend and device.
pub struct Runtime {
    pub backend: Backend,
    pub device: Device,
}

/// Initialize the GPU backend. Reads `APPLEGPU_BACKEND` env var,
/// defaults to MLX. Can only be called once.
pub fn init_backend() -> Result<&'static Runtime> {
    RUNTIME.get_or_try_init(|| {
        let backend = std::env::var("APPLEGPU_BACKEND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Backend::Mlx);

        let device = Device::new()?;

        Ok(Runtime { backend, device })
    })
}

/// Get the runtime if already initialized.
pub fn get_runtime() -> Result<&'static Runtime> {
    RUNTIME.get().ok_or(GpuError::BackendNotInitialized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_from_str() {
        assert_eq!("mlx".parse::<Backend>().unwrap(), Backend::Mlx);
        assert_eq!("MLX".parse::<Backend>().unwrap(), Backend::Mlx);
        assert_eq!("vm".parse::<Backend>().unwrap(), Backend::Vm);
        assert_eq!("VM".parse::<Backend>().unwrap(), Backend::Vm);
        assert!("invalid".parse::<Backend>().is_err());
    }

    // Note: init_backend() uses OnceCell so it can only be tested once per process.
    // The integration test in tests/ handles the full init flow.
}
```

- [ ] **Step 2: Add `backend` module to lib.rs**

Add `pub mod backend;` to `crates/core/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test --workspace 2>&1`
Expected: All tests pass including `backend_from_str`

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/backend.rs crates/core/src/lib.rs
git commit -m "feat: add Backend enum and init_backend() with env var support"
```

---

## Chunk 3: Tensor Type Definitions

### Task 7: Define core tensor metadata types

**Files:**
- Create: `crates/core/src/tensor.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Create tensor module with DType, Shape, TensorMeta, and tests**

Create `crates/core/src/tensor.rs`:

```rust
/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt32,
    Bool,
    BFloat16,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 1,
            DType::Float16 | DType::BFloat16 | DType::Int16 => 2,
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float64 | DType::Int64 => 8,
        }
    }
}

/// Shape of a tensor (dimensions).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Get dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

/// Where a tensor's data lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLocation {
    /// In host (CPU) memory
    Host,
    /// On Metal GPU
    Device,
    /// In shared memory (zero-copy between host and GPU)
    Shared,
}

/// Metadata for a virtual tensor (no data, just description).
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub id: u64,
    pub shape: Shape,
    pub dtype: DType,
    pub location: TensorLocation,
}

impl TensorMeta {
    /// Total size in bytes for the tensor data.
    pub fn size_bytes(&self) -> usize {
        self.shape.numel() * self.dtype.size_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::Float64.size_bytes(), 8);
        assert_eq!(DType::Int8.size_bytes(), 1);
        assert_eq!(DType::BFloat16.size_bytes(), 2);
    }

    #[test]
    fn shape_numel() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn shape_scalar() {
        let s = Shape::new(vec![]);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1); // product of empty = 1
    }

    #[test]
    fn tensor_meta_size() {
        let meta = TensorMeta {
            id: 1,
            shape: Shape::new(vec![32, 768]),
            dtype: DType::Float32,
            location: TensorLocation::Device,
        };
        assert_eq!(meta.size_bytes(), 32 * 768 * 4);
    }
}
```

- [ ] **Step 2: Add `tensor` module to lib.rs**

Add `pub mod tensor;` to `crates/core/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test --workspace 2>&1`
Expected: All tests pass including tensor type tests

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/tensor.rs crates/core/src/lib.rs
git commit -m "feat: add DType, Shape, TensorMeta, and TensorLocation types"
```

---

## Chunk 4: Expose to Python via PyO3

### Task 8: Expose device_name() and init_backend() to Python

**Files:**
- Modify: `crates/python/src/lib.rs`
- Modify: `python/applegpu_runtime/__init__.py`

- [ ] **Step 1: Write failing Python tests**

Create `python/tests/test_backend.py`:

```python
import applegpu_runtime as gpu


def test_init_backend():
    runtime = gpu.init_backend()
    assert runtime is not None


def test_init_backend_returns_backend_name():
    runtime = gpu.init_backend()
    assert runtime["backend"] in ("mlx", "vm")


def test_device_name():
    gpu.init_backend()
    name = gpu.device_name()
    assert isinstance(name, str)
    assert "Apple" in name
```

Create `python/tests/test_tensor_types.py`:

```python
import applegpu_runtime as gpu


def test_dtype_size():
    assert gpu.dtype_size("float32") == 4
    assert gpu.dtype_size("float16") == 2
    assert gpu.dtype_size("int8") == 1


def test_dtype_size_invalid():
    try:
        gpu.dtype_size("invalid")
        assert False, "Should have raised"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_backend.py python/tests/test_tensor_types.py -v 2>&1`
Expected: FAIL (functions don't exist yet)

- [ ] **Step 3: Implement PyO3 bindings**

Replace `crates/python/src/lib.rs` with:

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

/// Returns the library version.
#[pyfunction]
fn version() -> &'static str {
    applegpu_core::version()
}

/// Initialize the GPU backend. Returns dict with backend info.
#[pyfunction]
fn init_backend() -> PyResult<HashMap<String, String>> {
    let runtime = applegpu_core::backend::init_backend()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut info = HashMap::new();
    info.insert(
        "backend".to_string(),
        format!("{:?}", runtime.backend).to_lowercase(),
    );
    info.insert("device".to_string(), runtime.device.name());
    Ok(info)
}

/// Get the Metal GPU device name. Requires init_backend() first.
#[pyfunction]
fn device_name() -> PyResult<String> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(runtime.device.name())
}

/// Get the size in bytes of a dtype by name.
#[pyfunction]
fn dtype_size(name: &str) -> PyResult<usize> {
    use applegpu_core::tensor::DType;
    let dt = match name {
        "float16" | "f16" => DType::Float16,
        "float32" | "f32" => DType::Float32,
        "float64" | "f64" => DType::Float64,
        "bfloat16" | "bf16" => DType::BFloat16,
        "int8" | "i8" => DType::Int8,
        "int16" | "i16" => DType::Int16,
        "int32" | "i32" => DType::Int32,
        "int64" | "i64" => DType::Int64,
        "uint8" | "u8" => DType::UInt8,
        "uint32" | "u32" => DType::UInt32,
        "bool" => DType::Bool,
        _ => return Err(PyValueError::new_err(format!("Unknown dtype: {}", name))),
    };
    Ok(dt.size_bytes())
}

/// The Python module definition.
#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_size, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Update Python __init__.py**

```python
"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

from applegpu_runtime.applegpu_runtime import (
    version,
    init_backend,
    device_name,
    dtype_size,
)

__version__ = version()
__all__ = ["version", "init_backend", "device_name", "dtype_size"]
```

- [ ] **Step 5: Rebuild and run tests**

Run: `uv sync && uv run pytest -v 2>&1`
Expected: All Python tests pass (including new backend and tensor type tests)

- [ ] **Step 6: Commit**

```bash
git add crates/python/src/lib.rs python/applegpu_runtime/__init__.py python/tests/test_backend.py python/tests/test_tensor_types.py
git commit -m "feat: expose init_backend, device_name, and dtype_size to Python"
```

---

### Task 9: Update Makefile for cross-layer build

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Update Makefile so build-python depends on Swift**

```makefile
.PHONY: all build test clean setup check

all: build test

setup:
	uv tool install maturin
	uv sync

build-rust: build-swift
	cargo build --workspace

build-swift:
	cd swift && swift build -c release

build-python: build-rust
	uv sync

build: build-swift build-rust build-python

test-rust: build-swift
	cargo test --workspace

test-swift:
	cd swift && swift test

test-python: build-python
	uv run pytest -v

test: test-rust test-swift test-python

check:
	cargo check --workspace
	cd swift && swift build

clean:
	cargo clean
	cd swift && swift package clean
	rm -rf target
```

- [ ] **Step 2: Run `make test` to verify full pipeline**

Run: `make test 2>&1`
Expected: All tests pass across all three layers

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: update Makefile with cross-layer build dependencies"
```

---

### Task 10: End-to-end verification and push

- [ ] **Step 1: Run full test suite**

Run: `make clean && make test 2>&1`
Expected: Clean build, all tests pass

- [ ] **Step 2: Push to remote**

```bash
git push origin main
```
