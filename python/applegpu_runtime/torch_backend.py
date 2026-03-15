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


# ============================================================
# Op dispatch registry
# ============================================================

SUPPORTED_OPS = {}


def register_op(aten_op):
    """Decorator to register an aten op handler."""
    def decorator(fn):
        SUPPORTED_OPS[aten_op] = fn
        return fn
    return decorator


# ============================================================
# Helper functions
# ============================================================

def _unsqueeze_shape(t, dim):
    """Compute shape after unsqueeze."""
    shape = list(t._gpu_tensor.shape)
    if dim < 0:
        dim = len(shape) + 1 + dim
    shape.insert(dim, 1)
    return shape


def _squeeze_shape(t, dim):
    """Compute shape after squeeze."""
    shape = list(t._gpu_tensor.shape)
    if dim < 0:
        dim = len(shape) + dim
    if 0 <= dim < len(shape) and shape[dim] == 1:
        shape.pop(dim)
    return shape


def _unwrap(t):
    """Get the GpuTensor from an ApplegpuTensor, or convert CPU tensor/scalar."""
    if isinstance(t, ApplegpuTensor):
        return t._gpu_tensor
    if isinstance(t, torch.Tensor):
        return gpu.from_torch(t)
    return t


def _unwrap_scalar(t):
    """Unwrap, handling Python scalars by creating a 1-element GpuTensor."""
    if isinstance(t, (int, float)):
        return gpu.tensor([float(t)], shape=[1])
    return _unwrap(t)


def _wrap(gpu_t, torch_dtype=None):
    """Wrap a GpuTensor in an ApplegpuTensor."""
    return ApplegpuTensor(gpu_t, torch_dtype=torch_dtype)


# ============================================================
# Element-wise binary ops
# ============================================================

@register_op(torch.ops.aten.add.Tensor)
def _op_add(a, b, alpha=1):
    b_gpu = _unwrap_scalar(b)
    if alpha != 1:
        b_gpu = gpu.scalar_mul(b_gpu, float(alpha))
    return _wrap(gpu.add(_unwrap(a), b_gpu))


@register_op(torch.ops.aten.sub.Tensor)
def _op_sub(a, b, alpha=1):
    b_gpu = _unwrap_scalar(b)
    if alpha != 1:
        b_gpu = gpu.scalar_mul(b_gpu, float(alpha))
    return _wrap(gpu.sub(_unwrap(a), b_gpu))


@register_op(torch.ops.aten.mul.Tensor)
def _op_mul(a, b):
    return _wrap(gpu.mul(_unwrap(a), _unwrap_scalar(b)))


@register_op(torch.ops.aten.div.Tensor)
def _op_div(a, b, rounding_mode=None):
    # rounding_mode: None (true division), "trunc", "floor" — we only support true division natively
    if rounding_mode is not None:
        return NotImplemented
    return _wrap(gpu.div(_unwrap(a), _unwrap_scalar(b)))


# ============================================================
# Element-wise unary ops
# ============================================================

@register_op(torch.ops.aten.neg.default)
def _op_neg(a):
    return _wrap(gpu.neg(_unwrap(a)))


@register_op(torch.ops.aten.relu.default)
def _op_relu(a):
    return _wrap(gpu.relu(_unwrap(a)))


@register_op(torch.ops.aten.gelu.default)
def _op_gelu(a, approximate="none"):
    return _wrap(gpu.gelu(_unwrap(a)))


@register_op(torch.ops.aten.exp.default)
def _op_exp(a):
    return _wrap(gpu.exp(_unwrap(a)))


@register_op(torch.ops.aten.log.default)
def _op_log(a):
    return _wrap(gpu.log(_unwrap(a)))


@register_op(torch.ops.aten.sqrt.default)
def _op_sqrt(a):
    return _wrap(gpu.sqrt(_unwrap(a)))


# ============================================================
# Matrix ops
# ============================================================

@register_op(torch.ops.aten.mm.default)
def _op_mm(a, b):
    return _wrap(gpu.matmul(_unwrap(a), _unwrap(b)))


@register_op(torch.ops.aten.bmm.default)
def _op_bmm(a, b):
    return _wrap(gpu.matmul(_unwrap(a), _unwrap(b)))


@register_op(torch.ops.aten.matmul.default)
def _op_matmul(a, b):
    return _wrap(gpu.matmul(_unwrap(a), _unwrap(b)))


# ============================================================
# Reductions
# ============================================================

@register_op(torch.ops.aten._softmax.default)
def _op_softmax(a, dim, half_to_float):
    ndim = len(a._gpu_tensor.shape) if isinstance(a, ApplegpuTensor) else len(_unwrap(a).shape)
    # Our softmax always operates on last dim
    if dim != -1 and dim != ndim - 1:
        return NotImplemented
    return _wrap(gpu.softmax(_unwrap(a)))


@register_op(torch.ops.aten.argmax.default)
def _op_argmax(a, dim=None, keepdim=False):
    # Our argmax operates on last dim, flattened-style
    return _wrap(gpu.argmax(_unwrap(a)))


@register_op(torch.ops.aten.sum.dim_IntList)
def _op_sum(a, dim, keepdim=False, dtype=None):
    ndim = len(_unwrap(a).shape)
    # Normalize dim list
    dims = [d if d >= 0 else d + ndim for d in dim]
    if dims != [ndim - 1]:
        return NotImplemented
    result = _wrap(gpu.sum(_unwrap(a)))
    if keepdim:
        new_shape = list(_unwrap(a).shape)
        new_shape[-1] = 1
        result = _wrap(gpu.reshape(_unwrap(result), new_shape))
    return result


@register_op(torch.ops.aten.mean.dim)
def _op_mean(a, dim, keepdim=False, dtype=None):
    ndim = len(_unwrap(a).shape)
    dims = [d if d >= 0 else d + ndim for d in dim]
    if dims != [ndim - 1]:
        return NotImplemented
    result = _wrap(gpu.mean(_unwrap(a)))
    if keepdim:
        new_shape = list(_unwrap(a).shape)
        new_shape[-1] = 1
        result = _wrap(gpu.reshape(_unwrap(result), new_shape))
    return result


# ============================================================
# Shape ops
# ============================================================

@register_op(torch.ops.aten.reshape.default)
def _op_reshape(a, shape):
    return _wrap(gpu.reshape(_unwrap(a), list(shape)))


@register_op(torch.ops.aten.view.default)
def _op_view(a, size):
    return _wrap(gpu.reshape(_unwrap(a), list(size)))


@register_op(torch.ops.aten.t.default)
def _op_t(a):
    return _wrap(gpu.transpose(_unwrap(a)))


@register_op(torch.ops.aten.transpose.int)
def _op_transpose_int(a, dim0, dim1):
    return _wrap(gpu.transpose_dims(_unwrap(a), dim0, dim1))


@register_op(torch.ops.aten.permute.default)
def _op_permute(a, dims):
    # Check if exactly 2 dims are swapped (i.e., it's a transpose)
    identity = list(range(len(dims)))
    diff = [i for i in range(len(dims)) if dims[i] != identity[i]]
    if len(diff) == 2:
        return _wrap(gpu.transpose_dims(_unwrap(a), diff[0], diff[1]))
    # General permutation — fall back to CPU
    return NotImplemented


@register_op(torch.ops.aten.expand.default)
def _op_expand(a, size, implicit=False):
    # Broadcasting via add with zeros
    # Create a zeros tensor with the target shape and add
    gpu_a = _unwrap(a)
    n_elems = 1
    for s in size:
        if s == -1:
            return NotImplemented
        n_elems *= s
    zeros = gpu.tensor([0.0] * n_elems, shape=list(size))
    return _wrap(gpu.add(gpu_a, zeros))


@register_op(torch.ops.aten.unsqueeze.default)
def _op_unsqueeze(a, dim):
    return _wrap(gpu.reshape(_unwrap(a), _unsqueeze_shape(a, dim)))


@register_op(torch.ops.aten.squeeze.dim)
def _op_squeeze_dim(a, dim):
    return _wrap(gpu.reshape(_unwrap(a), _squeeze_shape(a, dim)))


@register_op(torch.ops.aten.contiguous_format if hasattr(torch.ops.aten, "contiguous_format") else None)
def _op_contiguous_dummy(a, memory_format=None):
    return a

# Remove None key if the attr didn't exist
SUPPORTED_OPS.pop(None, None)


# Try to register aten.contiguous.default (may not exist on all torch versions)
try:
    @register_op(torch.ops.aten.contiguous.default)
    def _op_contiguous(a, memory_format=None):
        return a
except AttributeError:
    pass


# ============================================================
# Tensor creation ops
# ============================================================

@register_op(torch.ops.aten.empty.memory_format)
def _op_empty(*args, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    # args[0] is the size tuple
    size = list(args[0])
    n_elems = 1
    for s in size:
        n_elems *= s
    # Create a tensor of zeros (empty is undefined, but zeros is safe)
    gpu_t = gpu.tensor([0.0] * max(n_elems, 1), shape=size if n_elems > 0 else [1])
    torch_dtype = dtype if dtype is not None else torch.float32
    return _wrap(gpu_t, torch_dtype=torch_dtype)


@register_op(torch.ops.aten.zeros.default)
def _op_zeros(size, dtype=None, layout=None, device=None, pin_memory=None):
    n_elems = 1
    for s in size:
        n_elems *= s
    gpu_t = gpu.tensor([0.0] * max(n_elems, 1), shape=list(size) if n_elems > 0 else [1])
    torch_dtype = dtype if dtype is not None else torch.float32
    return _wrap(gpu_t, torch_dtype=torch_dtype)


@register_op(torch.ops.aten.ones.default)
def _op_ones(size, dtype=None, layout=None, device=None, pin_memory=None):
    n_elems = 1
    for s in size:
        n_elems *= s
    gpu_t = gpu.tensor([1.0] * max(n_elems, 1), shape=list(size) if n_elems > 0 else [1])
    torch_dtype = dtype if dtype is not None else torch.float32
    return _wrap(gpu_t, torch_dtype=torch_dtype)


@register_op(torch.ops.aten.full.default)
def _op_full(size, fill_value, dtype=None, layout=None, device=None, pin_memory=None):
    n_elems = 1
    for s in size:
        n_elems *= s
    gpu_t = gpu.tensor([float(fill_value)] * max(n_elems, 1), shape=list(size) if n_elems > 0 else [1])
    torch_dtype = dtype if dtype is not None else torch.float32
    return _wrap(gpu_t, torch_dtype=torch_dtype)


@register_op(torch.ops.aten.scalar_tensor.default)
def _op_scalar_tensor(val, dtype=None, layout=None, device=None, pin_memory=None):
    torch_dtype = dtype if dtype is not None else torch.float32
    gpu_dtype = _TORCH_TO_GPU_DTYPE.get(torch_dtype, "float32")
    gpu_t = gpu.tensor([float(val)], shape=[1], dtype=gpu_dtype)
    return _wrap(gpu_t, torch_dtype=torch_dtype)


@register_op(torch.ops.aten.copy_.default)
def _op_copy_(dst, src, non_blocking=False):
    """In-place copy of src into dst."""
    if isinstance(src, ApplegpuTensor):
        cpu_data = src.to_torch_cpu()
        dst._gpu_tensor = gpu.from_torch(cpu_data)
        return dst
    elif isinstance(src, torch.Tensor):
        dst._gpu_tensor = gpu.from_torch(src)
        return dst
    return dst


# ============================================================
# Tensor manipulation ops
# ============================================================

@register_op(torch.ops.aten.cat.default)
def _op_cat(tensors, dim=0):
    gpu_tensors = [_unwrap(t) for t in tensors]
    result = gpu_tensors[0]
    for t in gpu_tensors[1:]:
        result = gpu.concat(result, t, dim)
    return _wrap(result)


@register_op(torch.ops.aten.select.int)
def _op_select(a, dim, index):
    gpu_a = _unwrap(a)
    shape = list(gpu_a.shape)
    if dim < 0:
        dim = len(shape) + dim
    if index < 0:
        index = shape[dim] + index
    # Slice one element along dim, then squeeze that dim
    sliced = gpu.slice(gpu_a, dim, index, index + 1)
    new_shape = list(sliced.shape)
    new_shape.pop(dim)
    if len(new_shape) == 0:
        new_shape = [1]
    return _wrap(gpu.reshape(sliced, new_shape))


@register_op(torch.ops.aten.slice.Tensor)
def _op_slice(a, dim=0, start=None, end=None, step=1):
    if step != 1:
        return NotImplemented
    gpu_a = _unwrap(a)
    shape = list(gpu_a.shape)
    if dim < 0:
        dim = len(shape) + dim
    actual_start = start if start is not None else 0
    actual_end = end if end is not None else shape[dim]
    # Clamp
    if actual_start < 0:
        actual_start = max(0, shape[dim] + actual_start)
    if actual_end < 0:
        actual_end = max(0, shape[dim] + actual_end)
    actual_end = min(actual_end, shape[dim])
    # If the slice is the full dimension, just return the tensor as-is
    if actual_start == 0 and actual_end == shape[dim]:
        return a if isinstance(a, ApplegpuTensor) else _wrap(gpu_a)
    return _wrap(gpu.slice(gpu_a, dim, actual_start, actual_end))


# ============================================================
# Layer ops
# ============================================================

@register_op(torch.ops.aten.layer_norm.default)
def _op_layer_norm(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    gpu_a = _unwrap(a)
    if weight is not None:
        gpu_weight = _unwrap(weight)
    else:
        # Default weight = ones
        n = 1
        for s in normalized_shape:
            n *= s
        gpu_weight = gpu.tensor([1.0] * n, shape=list(normalized_shape))
    if bias is not None:
        gpu_bias = _unwrap(bias)
    else:
        n = 1
        for s in normalized_shape:
            n *= s
        gpu_bias = gpu.tensor([0.0] * n, shape=list(normalized_shape))
    result = gpu.layer_norm(gpu_a, gpu_weight, gpu_bias, eps)
    return _wrap(result)


@register_op(torch.ops.aten.native_layer_norm.default)
def _op_native_layer_norm(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    # native_layer_norm returns (output, mean, rstd)
    # We compute output natively, return dummy mean/rstd
    gpu_a = _unwrap(a)
    if weight is not None:
        gpu_weight = _unwrap(weight)
    else:
        n = 1
        for s in normalized_shape:
            n *= s
        gpu_weight = gpu.tensor([1.0] * n, shape=list(normalized_shape))
    if bias is not None:
        gpu_bias = _unwrap(bias)
    else:
        n = 1
        for s in normalized_shape:
            n *= s
        gpu_bias = gpu.tensor([0.0] * n, shape=list(normalized_shape))
    result = gpu.layer_norm(gpu_a, gpu_weight, gpu_bias, eps)
    output = _wrap(result)
    # Return dummy mean/rstd as CPU tensors (these are rarely used in forward pass)
    shape = list(gpu_a.shape)
    outer_shape = shape[: len(shape) - len(normalized_shape)]
    if not outer_shape:
        outer_shape = [1]
    dummy_mean = torch.zeros(outer_shape)
    dummy_rstd = torch.ones(outer_shape)
    return (output, dummy_mean, dummy_rstd)


@register_op(torch.ops.aten.embedding.default)
def _op_embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return _wrap(gpu.embedding(_unwrap(weight), _unwrap(indices)))


# ============================================================
# ApplegpuTensor class
# ============================================================

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

        # Check dispatch table
        if func in SUPPORTED_OPS:
            result = SUPPORTED_OPS[func](*args, **kwargs)
            if result is not NotImplemented:
                return result

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


# ============================================================
# Lifecycle handlers (Phase A)
# ============================================================

def _handle_to_copy(args, kwargs):
    """Handle aten._to_copy -- device transfer and dtype casting."""
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
        # CPU to CPU copy -- just return normal
        return src.clone()


def _handle_detach(args, kwargs):
    """Handle aten.detach -- return same tensor (no autograd graph)."""
    src = args[0]
    if isinstance(src, ApplegpuTensor):
        return _wrap(src._gpu_tensor, torch_dtype=src.dtype)
    return src.detach()


def _handle_clone(args, kwargs):
    """Handle aten.clone -- create a copy."""
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
