"""PrivateUse1 C++ backend loader for applegpu_runtime.

Usage:
    from applegpu_runtime.cpp_backend import load_cpp_backend
    load_cpp_backend()
    x = torch.empty(3, 3, device='applegpu')
"""
import os
import sys
import glob
import types


def _find_backend_dylib():
    """Find the compiled C++ backend shared library."""
    backend_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend_cpp')
    backend_dir = os.path.normpath(backend_dir)
    patterns = [
        os.path.join(backend_dir, 'applegpu_backend*.so'),
        os.path.join(backend_dir, 'applegpu_backend*.dylib'),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"applegpu_backend shared library not found in {backend_dir}. "
        "Run: cd backend_cpp && uv run python setup.py build_ext --inplace"
    )


_loaded = False


def load_cpp_backend():
    """Load the PrivateUse1 C++ backend for applegpu.

    After calling this, torch.empty(..., device='applegpu') will dispatch
    through C++ to the Rust graph engine (no Python per-op overhead).
    """
    global _loaded
    if _loaded:
        return

    import torch

    # Register a stub module so PyTorch can find 'torch.applegpu'.
    # Required for generate_methods_for_privateuse1_backend().
    mod = types.ModuleType('torch.applegpu')
    mod.device_count = lambda: 1
    mod.is_available = lambda: True
    mod.current_device = lambda: 0
    mod.set_device = lambda d: None
    mod.synchronize = lambda d=None: None
    mod._exchange_device = lambda d: 0
    mod._maybe_exchange_device = lambda d: 0
    sys.modules['torch.applegpu'] = mod
    torch.applegpu = mod

    dylib_path = _find_backend_dylib()
    torch.ops.load_library(dylib_path)
    torch.utils.rename_privateuse1_backend("applegpu")
    torch.utils.generate_methods_for_privateuse1_backend()
    _loaded = True
