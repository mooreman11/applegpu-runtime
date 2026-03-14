"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

from applegpu_runtime.applegpu_runtime import (
    version,
    init_backend,
    device_name,
    dtype_size,
)

__version__ = version()
__all__ = ["version", "init_backend", "device_name", "dtype_size"]
