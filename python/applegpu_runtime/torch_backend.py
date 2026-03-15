"""PyTorch custom device backend for applegpu_runtime."""

import warnings
import torch
import applegpu_runtime as gpu

_backend_enabled = False
_warned_ops = set()


def enable():
    """Register applegpu as a PyTorch device backend. Requires torch >= 2.1."""
    global _backend_enabled
    if _backend_enabled:
        return

    gpu.init_backend()
    torch.utils.rename_privateuse1_backend("applegpu")
    torch.utils.generate_methods_for_privateuse1_backend()
    _backend_enabled = True


def _warn_fallback(op_name):
    """Warn once per unsupported op."""
    if op_name not in _warned_ops:
        _warned_ops.add(op_name)
        warnings.warn(
            f"{op_name} not supported on applegpu, falling back to CPU",
            stacklevel=3,
        )


def _gpu_tensor_to_torch_cpu(gpu_t):
    """Convert a GpuTensor to a CPU torch.Tensor."""
    return gpu_t.to_torch()


def _torch_cpu_to_gpu_tensor(t):
    """Convert a CPU torch.Tensor to a GpuTensor."""
    return gpu.from_torch(t)


# Map PyTorch dtype to our dtype string
_TORCH_TO_GPU_DTYPE = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_GPU_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_TO_GPU_DTYPE.items()}


class ApplegpuTensor(torch.Tensor):
    """A torch.Tensor subclass backed by applegpu Metal buffers."""

    @staticmethod
    def __new__(cls, gpu_tensor, torch_dtype=None):
        # Create a dummy tensor with the right shape and dtype
        shape = tuple(gpu_tensor.shape)
        if torch_dtype is None:
            torch_dtype = _GPU_TO_TORCH_DTYPE.get(gpu_tensor.dtype, torch.float32)

        # Create empty shell tensor (no real CPU storage)
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=torch_dtype,
            device=torch.device("cpu"),  # placeholder, overridden by dispatch
        )
        r._gpu_tensor = gpu_tensor
        return r

    def __repr__(self):
        return f"ApplegpuTensor(shape={list(self._gpu_tensor.shape)}, dtype={self._gpu_tensor.dtype}, device='applegpu')"

    def __del__(self):
        # GpuTensor handles cleanup via its own Drop
        pass

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Tensor lifecycle ops
        if func == torch.ops.aten._to_copy.default:
            return _handle_to_copy(args, kwargs)
        if func == torch.ops.aten.detach.default:
            return _handle_detach(args, kwargs)
        if func == torch.ops.aten.clone.default:
            return _handle_clone(args, kwargs)

        # Fallback: move to CPU, run, move back
        return _cpu_fallback(func, args, kwargs)

    @classmethod
    def from_torch(cls, tensor):
        """Convert a torch.Tensor to an ApplegpuTensor."""
        gpu_t = gpu.from_torch(tensor)
        return cls(gpu_t, torch_dtype=tensor.dtype)

    def to_torch_cpu(self):
        """Convert back to a CPU torch.Tensor."""
        return _gpu_tensor_to_torch_cpu(self._gpu_tensor)


def _unwrap(t):
    """Get the GpuTensor from an ApplegpuTensor, or convert CPU tensor."""
    if isinstance(t, ApplegpuTensor):
        return t._gpu_tensor
    if isinstance(t, torch.Tensor):
        return gpu.from_torch(t)
    return t


def _wrap(gpu_t, torch_dtype=None):
    """Wrap a GpuTensor in an ApplegpuTensor."""
    return ApplegpuTensor(gpu_t, torch_dtype=torch_dtype)


def _handle_to_copy(args, kwargs):
    """Handle aten._to_copy — device transfer and dtype casting."""
    src = args[0]
    target_device = kwargs.get("device", None)
    target_dtype = kwargs.get("dtype", None)

    if isinstance(src, ApplegpuTensor):
        if target_device is not None and str(target_device) == "cpu":
            # Move to CPU
            return src.to_torch_cpu()
        # Stay on device (maybe cast dtype)
        if target_dtype is not None and target_dtype != src.dtype:
            # Dtype cast: move to CPU, cast, move back
            cpu_t = src.to_torch_cpu().to(dtype=target_dtype)
            return ApplegpuTensor.from_torch(cpu_t)
        return _handle_clone(args, kwargs)
    else:
        # CPU tensor moving to applegpu
        if target_device is not None and "applegpu" in str(target_device):
            return ApplegpuTensor.from_torch(src)
        # CPU to CPU copy — just return normal
        return src.clone()


def _handle_detach(args, kwargs):
    """Handle aten.detach — return same tensor (no autograd graph)."""
    src = args[0]
    if isinstance(src, ApplegpuTensor):
        return _wrap(src._gpu_tensor, torch_dtype=src.dtype)
    return src.detach()


def _handle_clone(args, kwargs):
    """Handle aten.clone — create a copy."""
    src = args[0]
    if isinstance(src, ApplegpuTensor):
        # Convert to CPU, then back to create a fresh copy
        cpu_t = src.to_torch_cpu()
        return ApplegpuTensor.from_torch(cpu_t)
    return src.clone()


def _cpu_fallback(func, args, kwargs):
    """Fallback: move all ApplegpuTensors to CPU, run op, move results back."""
    op_name = str(func.name())
    _warn_fallback(op_name)

    # Convert args
    def to_cpu(x):
        if isinstance(x, ApplegpuTensor):
            return x.to_torch_cpu()
        if isinstance(x, (list, tuple)):
            return type(x)(to_cpu(v) for v in x)
        return x

    cpu_args = to_cpu(args)
    cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}

    # Run on CPU
    result = func(*cpu_args, **cpu_kwargs)

    # Wrap results back
    def to_applegpu(x):
        if isinstance(x, torch.Tensor) and not isinstance(x, ApplegpuTensor):
            try:
                return ApplegpuTensor.from_torch(x)
            except Exception:
                return x
        if isinstance(x, (list, tuple)):
            return type(x)(to_applegpu(v) for v in x)
        return x

    return to_applegpu(result)
