"""Build the applegpu PrivateUse1 C++ backend extension.

Usage:
    cd backend_cpp && uv run python setup.py build_ext --inplace
"""
import os
import subprocess
from setuptools import setup

# Force arm64-only build on Apple Silicon.
# The Rust/Swift static libs are arm64-only; universal builds break
# C10_REGISTER_GUARD_IMPL (DeviceGuard not registered for arm64 slice).
os.environ.setdefault("ARCHFLAGS", "-arch arm64")
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Workspace root (parent of backend_cpp/)
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build Rust static lib (release mode for the .a)
subprocess.check_call(
    ["cargo", "build", "-p", "applegpu-core", "--release"],
    cwd=workspace_root,
)

# Locate static libraries
rust_lib = os.path.join(workspace_root, "target", "release", "libapplegpu_core.a")
swift_build_dir = os.path.join(workspace_root, "swift", ".build", "release")
swift_lib = os.path.join(swift_build_dir, "libAppleGPUBridge.a")

# Verify they exist
for lib in [rust_lib, swift_lib]:
    if not os.path.exists(lib):
        raise FileNotFoundError(f"Required static library not found: {lib}")

# Find Swift runtime paths (mirrors crates/core/build.rs)
sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
swift_bin = subprocess.check_output(
    ["xcrun", "--toolchain", "default", "--find", "swift"]
).decode().strip()
swift_lib_path = os.path.join(
    os.path.dirname(os.path.dirname(swift_bin)), "lib", "swift", "macosx"
)

setup(
    name="applegpu_backend",
    ext_modules=[
        CppExtension(
            name="applegpu_backend",
            sources=["applegpu_backend.cpp"],
            include_dirs=["."],
            extra_objects=[rust_lib, swift_lib],
            extra_link_args=[
                f"-L{swift_lib_path}",
                f"-L{sdk_path}/usr/lib/swift",
                "-lswiftCore",
                "-framework", "Metal",
                "-framework", "MetalPerformanceShaders",
                "-framework", "Foundation",
            ],
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
