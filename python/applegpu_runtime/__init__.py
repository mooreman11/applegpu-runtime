"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

from applegpu_runtime.applegpu_runtime import (
    version,
    init_backend,
    device_name,
    dtype_size,
    tensor,
    to_list,
    shape,
    add,
    sub,
    mul,
    div,
    neg,
    relu,
    exp,
    log,
    sqrt,
    matmul,
)

__version__ = version()
__all__ = [
    "version",
    "init_backend",
    "device_name",
    "dtype_size",
    "tensor",
    "to_list",
    "shape",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "relu",
    "exp",
    "log",
    "sqrt",
    "matmul",
]
