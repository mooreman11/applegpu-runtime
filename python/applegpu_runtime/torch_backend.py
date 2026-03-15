"""PyTorch custom device backend for applegpu_runtime."""

import warnings
import torch
import applegpu_runtime as gpu

_backend_enabled = False
_warned_ops = set()

# Global registry mapping CPU data_ptr -> GpuTensor.
# When ApplegpuTensor is stored in nn.Parameter, custom attributes are lost.
# We use the backing CPU tensor's data_ptr as a stable key to retrieve
# the associated GpuTensor.
_gpu_tensor_registry = {}


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
    shape = list(_unwrap(t).shape)
    if dim < 0:
        dim = len(shape) + 1 + dim
    shape.insert(dim, 1)
    return shape


def _squeeze_shape(t, dim):
    """Compute shape after squeeze."""
    shape = list(_unwrap(t).shape)
    if dim < 0:
        dim = len(shape) + dim
    if 0 <= dim < len(shape) and shape[dim] == 1:
        shape.pop(dim)
    return shape


def _unwrap(t):
    """Get the GpuTensor from an ApplegpuTensor, or convert CPU tensor/scalar."""
    if isinstance(t, ApplegpuTensor):
        return t._gpu
    if isinstance(t, torch.Tensor):
        # Check registry in case this is an ApplegpuTensor that lost its type
        gt = _gpu_tensor_registry.get(t.data_ptr())
        if gt is not None:
            return gt
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


@register_op(torch.ops.aten.abs.default)
def _op_abs(a):
    return _wrap(gpu.abs(_unwrap(a)))


@register_op(torch.ops.aten.sign.default)
def _op_sign(a):
    return _wrap(gpu.sign(_unwrap(a)))


@register_op(torch.ops.aten.pow.Tensor_Scalar)
def _op_pow(a, exponent):
    return _wrap(gpu.pow(_unwrap(a), float(exponent)))


@register_op(torch.ops.aten.clamp.default)
def _op_clamp(a, min=None, max=None):
    min_val = float(min) if min is not None else -1e30
    max_val = float(max) if max is not None else 1e30
    return _wrap(gpu.clamp(_unwrap(a), min_val, max_val))


@register_op(torch.ops.aten.where.self)
def _op_where(condition, x, y):
    return _wrap(gpu.where_cond(_unwrap(condition), _unwrap(x), _unwrap(y)))


@register_op(torch.ops.aten.masked_fill.Scalar)
def _op_masked_fill(input, mask, value):
    return _wrap(gpu.masked_fill(_unwrap(input), _unwrap(mask), float(value)))


@register_op(torch.ops.aten.triu.default)
def _op_triu(input, diagonal=0):
    return _wrap(gpu.triu(_unwrap(input), diagonal))


@register_op(torch.ops.aten.tril.default)
def _op_tril(input, diagonal=0):
    return _wrap(gpu.tril(_unwrap(input), diagonal))


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


@register_op(torch.ops.aten.addmm.default)
def _op_addmm(bias, input, weight, beta=1, alpha=1):
    """addmm: beta * bias + alpha * (input @ weight). Used by nn.Linear."""
    result = _wrap(gpu.matmul(_unwrap(input), _unwrap(weight)))
    if alpha != 1:
        result = _wrap(gpu.scalar_mul(_unwrap(result), float(alpha)))
    if bias is not None:
        bias_gpu = _unwrap(bias)
        if beta != 1:
            bias_gpu = gpu.scalar_mul(bias_gpu, float(beta))
        result = _wrap(gpu.add(_unwrap(result), bias_gpu))
    return result


# ============================================================
# Reductions
# ============================================================

@register_op(torch.ops.aten._softmax.default)
def _op_softmax(a, dim, half_to_float):
    ndim = len(_unwrap(a).shape)
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
        new_gpu = gpu.from_torch(cpu_data)
    elif isinstance(src, torch.Tensor):
        new_gpu = gpu.from_torch(src)
    else:
        return dst
    # Update both the direct attribute and the registry
    dst._gpu_tensor = new_gpu
    _gpu_tensor_registry[dst.data_ptr()] = new_gpu
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
# CNN ops
# ============================================================

@register_op(torch.ops.aten.convolution.default)
def _convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    # Only support non-transposed, dilation=1, groups=1
    if transposed or any(d != 1 for d in dilation) or groups != 1:
        return NotImplemented

    ndim = len(stride)
    if ndim == 1:
        result = _wrap(gpu.conv1d(_unwrap(input), _unwrap(weight), stride[0], padding[0]))
    elif ndim == 2:
        result = _wrap(gpu.conv2d(_unwrap(input), _unwrap(weight), stride[0], stride[1], padding[0], padding[1]))
    else:
        return NotImplemented

    if bias is not None:
        # Add bias: result is [B, OC, ...], bias is [OC]
        # Need to reshape bias for broadcasting
        result_cpu = result.to_torch_cpu()
        result_cpu = result_cpu + bias.reshape(1, -1, *([1] * ndim))
        result = ApplegpuTensor.from_torch(result_cpu)
    return result


@register_op(torch.ops.aten._native_batch_norm_legit_no_training.default)
def _batch_norm(input, weight, bias, running_mean, running_var, momentum, eps):
    result = _wrap(gpu.batch_norm(_unwrap(input), _unwrap(running_mean), _unwrap(running_var), _unwrap(weight), _unwrap(bias), eps))
    # Returns (output, mean, rstd) -- return dummies for mean/rstd
    return result, torch.tensor([]), torch.tensor([])


@register_op(torch.ops.aten.max_pool2d_with_indices.default)
def _max_pool2d(input, kernel_size, stride=None, padding=(0, 0), dilation=(1, 1), ceil_mode=False):
    if stride is None or len(stride) == 0:
        stride = kernel_size
    kh, kw = kernel_size[0], kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]
    sh, sw = stride[0], stride[1] if len(stride) > 1 else stride[0]
    ph, pw = padding[0], padding[1] if len(padding) > 1 else padding[0]
    result = _wrap(gpu.max_pool2d(_unwrap(input), kh, kw, sh, sw, ph, pw))
    # Returns (output, indices) -- dummy indices
    return result, torch.tensor([0])


@register_op(torch.ops.aten.avg_pool2d.default)
def _avg_pool2d(input, kernel_size, stride=None, padding=(0, 0), ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None or len(stride) == 0:
        stride = kernel_size
    kh, kw = kernel_size[0], kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]
    sh, sw = stride[0], stride[1] if len(stride) > 1 else stride[0]
    ph, pw = padding[0], padding[1] if len(padding) > 1 else padding[0]
    return _wrap(gpu.avg_pool2d(_unwrap(input), kh, kw, sh, sw, ph, pw))


# ============================================================
# ApplegpuTensor class
# ============================================================

class ApplegpuTensor(torch.Tensor):
    """A torch.Tensor subclass backed by applegpu Metal buffers.

    Uses _make_subclass with real CPU backing storage so that repr/str never
    segfault. The actual GPU data is tracked in _gpu_tensor_registry keyed
    by data_ptr, since nn.Parameter strips custom attributes from subclasses.
    """

    @staticmethod
    def __new__(cls, gpu_tensor, torch_dtype=None):
        shape = tuple(gpu_tensor.shape)
        if torch_dtype is None:
            torch_dtype = _GPU_TO_TORCH_DTYPE.get(gpu_tensor.dtype, torch.float32)

        # Create a real CPU tensor as backing storage (prevents segfaults in repr)
        backing = torch.zeros(shape, dtype=torch_dtype)
        r = torch.Tensor._make_subclass(cls, backing)
        # Store directly on the instance (works for direct access)
        r._gpu_tensor = gpu_tensor
        # Also store in the global registry (survives Parameter.data round-trip)
        _gpu_tensor_registry[r.data_ptr()] = gpu_tensor
        return r

    @property
    def _gpu(self):
        """Retrieve the GpuTensor, even after nn.Parameter strips attributes."""
        if hasattr(self, "_gpu_tensor"):
            return self._gpu_tensor
        return _gpu_tensor_registry.get(self.data_ptr())

    def __repr__(self):
        gt = self._gpu
        if gt is not None:
            return f"ApplegpuTensor(shape={list(gt.shape)}, dtype={gt.dtype}, device='applegpu')"
        return f"ApplegpuTensor(shape={list(self.shape)}, device='applegpu')"

    def __str__(self):
        return self.__repr__()

    def __del__(self):
        # Clean up the registry entry
        try:
            ptr = self.data_ptr()
            _gpu_tensor_registry.pop(ptr, None)
        except Exception:
            pass

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Intercept torch functions to prevent unsafe operations on our tensors."""
        kwargs = kwargs or {}

        # For most operations, fall through to __torch_dispatch__
        with torch.no_grad():
            return super().__torch_function__(func, types, args, kwargs)

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
        gt = self._gpu
        if gt is not None:
            return _gpu_tensor_to_torch_cpu(gt)
        raise RuntimeError("ApplegpuTensor has no associated GPU data")


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
        return _wrap(src._gpu, torch_dtype=src.dtype)
    return src.detach()


def _handle_clone(args, kwargs):
    """Handle aten.clone -- create a copy."""
    src = args[0]
    if isinstance(src, ApplegpuTensor):
        # Convert to CPU, then back to create a fresh copy
        cpu_t = src.to_torch_cpu()
        return ApplegpuTensor.from_torch(cpu_t)
    return src.clone()


def to_applegpu(model_or_tensor):
    """Move a PyTorch model or tensor to applegpu Metal GPU.

    For nn.Module: converts all parameters and buffers in-place.
    For torch.Tensor: wraps as ApplegpuTensor.

    Args:
        model_or_tensor: an nn.Module or torch.Tensor

    Returns:
        The same object with data on applegpu.
    """
    import torch.nn as nn

    if isinstance(model_or_tensor, ApplegpuTensor):
        return model_or_tensor
    elif isinstance(model_or_tensor, nn.Module):
        _move_module_to_applegpu(model_or_tensor)
        return model_or_tensor
    elif isinstance(model_or_tensor, torch.Tensor):
        return ApplegpuTensor.from_torch(model_or_tensor)
    else:
        raise TypeError(f"Expected nn.Module or torch.Tensor, got {type(model_or_tensor)}")


def _move_module_to_applegpu(module):
    """Recursively replace parameters and buffers in a module with ApplegpuTensors.

    We must replace parameters at the module level (not via param.data = ...)
    because PyTorch's Parameter.data setter strips tensor subclass types.
    """
    import torch.nn as nn

    # Replace parameters on this module (non-recursive to avoid double-processing)
    for name, param in list(module._parameters.items()):
        if param is not None and not isinstance(param.data, ApplegpuTensor):
            new_data = ApplegpuTensor.from_torch(param.data)
            new_param = nn.Parameter(new_data, requires_grad=param.requires_grad)
            module._parameters[name] = new_param

    # Replace buffers on this module
    for name, buf in list(module._buffers.items()):
        if buf is not None and not isinstance(buf, ApplegpuTensor):
            module._buffers[name] = ApplegpuTensor.from_torch(buf)

    # Recurse into child modules
    for child in module.children():
        _move_module_to_applegpu(child)


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
