"""PyTorch custom device backend for applegpu_runtime."""

import warnings
import torch
import applegpu_runtime as gpu

_backend_enabled = False
_warned_ops = set()
_in_fallback = False

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


def _extract_cpu_backing(t):
    """Extract the CPU backing tensor from an ApplegpuTensor without going through
    __torch_dispatch__. This prevents infinite recursion when the GPU tensor has been
    freed and we need to reconstruct from the backing data."""
    # Create a new plain tensor from the same storage, completely bypassing dispatch.
    # untyped_storage() gives the raw memory, then we wrap it as a plain torch.Tensor.
    storage = t.untyped_storage()
    cpu_tensor = torch.tensor([], dtype=t.dtype).set_(storage).reshape(t.shape).clone()
    return cpu_tensor


def _unwrap(t):
    """Get the GpuTensor from an ApplegpuTensor, or convert CPU tensor/scalar.

    If a GpuTensor's Rust-side tensor was freed by the lazy runtime,
    recreate it from the CPU backing data.
    """
    global _in_fallback
    if isinstance(t, ApplegpuTensor):
        gt = t._gpu
        if gt is not None:
            # Check if the tensor is still alive on the Rust side
            try:
                _ = gt.shape  # will raise if destroyed
                return gt
            except (ValueError, RuntimeError):
                # Tensor was freed by lazy runtime -- recreate from CPU backing.
                # Access the backing CPU data WITHOUT going through __torch_dispatch__
                # to avoid infinite recursion.
                cpu_data = _extract_cpu_backing(t)
                new_gt = gpu.from_torch(cpu_data)
                t._gpu_tensor = new_gt
                _gpu_tensor_registry[t.data_ptr()] = new_gt
                return new_gt
        # No GPU tensor at all -- create from backing.
        cpu_data = _extract_cpu_backing(t)
        new_gt = gpu.from_torch(cpu_data)
        t._gpu_tensor = new_gt
        _gpu_tensor_registry[t.data_ptr()] = new_gt
        return new_gt
    if isinstance(t, torch.Tensor):
        # Check registry in case this is an ApplegpuTensor that lost its type
        gt = _gpu_tensor_registry.get(t.data_ptr())
        if gt is not None:
            try:
                _ = gt.shape
                return gt
            except (ValueError, RuntimeError):
                pass
        return gpu.from_torch(t)
    return t


def _unwrap_scalar(t):
    """Unwrap, handling Python scalars by creating a 1-element GpuTensor."""
    if isinstance(t, (int, float)):
        return gpu.tensor([float(t)], shape=[1])
    return _unwrap(t)


_eager_mode = False


def set_eager_mode(enabled=True):
    """Enable or disable eager evaluation mode.

    When enabled, all GPU operations are immediately materialized, which is
    needed for autograd training (backward pass requires live tensor data).
    When disabled (default), lazy evaluation with kernel fusion is used.
    """
    global _eager_mode
    _eager_mode = enabled


def _wrap(gpu_t, torch_dtype=None, requires_grad=False):
    """Wrap a GpuTensor in an ApplegpuTensor."""
    if _eager_mode:
        gpu.eval(gpu_t)
    result = ApplegpuTensor(gpu_t, torch_dtype=torch_dtype, requires_grad=requires_grad)
    if _eager_mode:
        # Sync CPU backing so _unwrap can reconstruct if needed
        try:
            cpu_data = gpu_t.to_torch()
            global _in_fallback
            was = _in_fallback
            _in_fallback = True
            try:
                backing = result.data
                if backing.shape == cpu_data.shape:
                    backing.copy_(cpu_data)
            finally:
                _in_fallback = was
        except Exception:
            pass
    return result


def _update_inplace(a, result_gpu):
    """Update an ApplegpuTensor in-place with a new GpuTensor result.

    Evaluates eagerly if needed, updates the registry and attribute,
    and syncs the CPU backing storage so _unwrap can reconstruct if
    the Rust tensor ID is later freed.
    """
    if _eager_mode:
        gpu.eval(result_gpu)
    if isinstance(a, ApplegpuTensor):
        _gpu_tensor_registry[a.data_ptr()] = result_gpu
        a._gpu_tensor = result_gpu
        if _eager_mode:
            try:
                cpu_data = result_gpu.to_torch()
                global _in_fallback
                was = _in_fallback
                _in_fallback = True
                try:
                    backing = a.data
                    if backing.shape == cpu_data.shape:
                        backing.copy_(cpu_data)
                finally:
                    _in_fallback = was
            except Exception:
                pass
    return a


# ============================================================
# Element-wise binary ops
# ============================================================

@register_op(torch.ops.aten.add.Tensor)
def _op_add(a, b, alpha=1):
    b_gpu = _unwrap_scalar(b)
    if alpha != 1:
        b_gpu = gpu.scalar_mul(b_gpu, float(alpha))
    return _wrap(gpu.add(_unwrap(a), b_gpu))


@register_op(torch.ops.aten.add_.Tensor)
def _op_add_inplace(a, b, alpha=1):
    """In-place add: a += alpha * b."""
    # TODO: Int64 compute kernels needed — currently falls back to CPU for non-float types
    # (e.g., batch_norm's num_batches_tracked is Int64). See backlog.
    a_gpu = _unwrap(a) if isinstance(a, ApplegpuTensor) else None
    if a_gpu is not None and a_gpu.dtype not in ("float32", "float16"):
        return NotImplemented
    b_gpu = _unwrap_scalar(b)
    if alpha != 1:
        b_gpu = gpu.scalar_mul(b_gpu, float(alpha))
    result_gpu = gpu.add(_unwrap(a), b_gpu)
    _update_inplace(a, result_gpu)
    return a


@register_op(torch.ops.aten.add_.Scalar)
def _op_add_inplace_scalar(a, scalar):
    """In-place scalar add: a += scalar. Used by Adam (eps addition)."""
    a_gpu = _unwrap(a)
    scalar_t = gpu.tensor([float(scalar)], shape=[1])
    result_gpu = gpu.add(a_gpu, scalar_t)
    return _update_inplace(a, result_gpu)


@register_op(torch.ops.aten.mul_.Tensor)
def _op_mul_inplace_tensor(a, b):
    """In-place tensor mul: a *= b. Used by Adam/AdamW weight decay."""
    if isinstance(b, (int, float)):
        result_gpu = gpu.scalar_mul(_unwrap(a), float(b))
    else:
        result_gpu = gpu.mul(_unwrap(a), _unwrap(b))
    return _update_inplace(a, result_gpu)


@register_op(torch.ops.aten.mul_.Scalar)
def _op_mul_inplace_scalar(a, scalar):
    """In-place scalar mul: a *= scalar. Used by Adam beta scaling."""
    result_gpu = gpu.scalar_mul(_unwrap(a), float(scalar))
    return _update_inplace(a, result_gpu)


@register_op(torch.ops.aten.addcmul_.default)
def _op_addcmul_inplace(a, b, c, value=1):
    """In-place addcmul: a += value * b * c. Used by Adam for variance update."""
    bc = gpu.mul(_unwrap(b), _unwrap(c))
    if value != 1:
        bc = gpu.scalar_mul(bc, float(value))
    result_gpu = gpu.add(_unwrap(a), bc)
    return _update_inplace(a, result_gpu)


@register_op(torch.ops.aten.addcdiv_.default)
def _op_addcdiv_inplace(a, b, c, value=1):
    """In-place addcdiv: a += value * b / c. Used by Adam for parameter update."""
    bc = gpu.div(_unwrap(b), _unwrap(c))
    if value != 1:
        bc = gpu.scalar_mul(bc, float(value))
    result_gpu = gpu.add(_unwrap(a), bc)
    return _update_inplace(a, result_gpu)


@register_op(torch.ops.aten.lerp_.Scalar)
def _op_lerp_inplace(a, b, weight):
    """In-place lerp: a = a + weight * (b - a). Used by Adam for EMA updates."""
    diff = gpu.sub(_unwrap(b), _unwrap(a))
    scaled = gpu.scalar_mul(diff, float(weight))
    result_gpu = gpu.add(_unwrap(a), scaled)
    return _update_inplace(a, result_gpu)


@register_op(torch.ops.aten.reciprocal.default)
def _op_reciprocal(a):
    """Reciprocal: 1/x via pow(x, -1). Used by Adam for inverse sqrt."""
    return _wrap(gpu.pow(_unwrap(a), -1.0))


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


@register_op(torch.ops.aten.relu_.default)
def _op_relu_inplace(a):
    """In-place relu: return new tensor (we don't mutate GPU buffers)."""
    result = _wrap(gpu.relu(_unwrap(a)))
    # Update the source tensor's registry entry so it reflects the new data
    if isinstance(a, ApplegpuTensor):
        _gpu_tensor_registry[a.data_ptr()] = result._gpu
        a._gpu_tensor = result._gpu
    return result


@register_op(torch.ops.aten.gelu.default)
def _op_gelu(a, approximate="none"):
    if approximate == "tanh":
        return _wrap(gpu.gelu(_unwrap(a)))
    else:
        return _wrap(gpu.gelu_exact(_unwrap(a)))


@register_op(torch.ops.aten.exp.default)
def _op_exp(a):
    return _wrap(gpu.exp(_unwrap(a)))


@register_op(torch.ops.aten.log.default)
def _op_log(a):
    return _wrap(gpu.log(_unwrap(a)))


@register_op(torch.ops.aten.sqrt.default)
def _op_sqrt(a):
    return _wrap(gpu.sqrt(_unwrap(a)))


@register_op(torch.ops.aten.tanh.default)
def _op_tanh(a):
    return _wrap(gpu.tanh(_unwrap(a)))


@register_op(torch.ops.aten.sigmoid.default)
def _op_sigmoid(a):
    return _wrap(gpu.sigmoid(_unwrap(a)))


@register_op(torch.ops.aten.var.correction)
def _op_var(a, dim=None, *, correction=1, keepdim=False):
    """Variance reduction along last dim."""
    corr = int(correction) if correction is not None else 1
    return _wrap(gpu.var(_unwrap(a), corr))


@register_op(torch.ops.aten.std.correction)
def _op_std(a, dim=None, *, correction=1, keepdim=False):
    """Standard deviation reduction along last dim."""
    corr = int(correction) if correction is not None else 1
    return _wrap(gpu.std_dev(_unwrap(a), corr))


@register_op(torch.ops.aten.sigmoid_.default)
def _op_sigmoid_inplace(a):
    """In-place sigmoid."""
    result = gpu.sigmoid(_unwrap(a))
    if isinstance(a, ApplegpuTensor):
        a._gpu_tensor = result
        _gpu_tensor_registry[a.data_ptr()] = result
    return a


@register_op(torch.ops.aten.tanh_.default)
def _op_tanh_inplace(a):
    """In-place tanh."""
    result = gpu.tanh(_unwrap(a))
    if isinstance(a, ApplegpuTensor):
        a._gpu_tensor = result
        _gpu_tensor_registry[a.data_ptr()] = result
    return a


@register_op(torch.ops.aten.abs.default)
def _op_abs(a):
    return _wrap(gpu.abs(_unwrap(a)))


@register_op(torch.ops.aten.sign.default)
def _op_sign(a):
    return _wrap(gpu.sign(_unwrap(a)))


@register_op(torch.ops.aten.sin.default)
def _op_sin(a):
    return _wrap(gpu.sin(_unwrap(a)))


@register_op(torch.ops.aten.cos.default)
def _op_cos(a):
    return _wrap(gpu.cos(_unwrap(a)))


@register_op(torch.ops.aten.pow.Tensor_Scalar)
def _op_pow(a, exponent):
    return _wrap(gpu.pow(_unwrap(a), float(exponent)))


@register_op(torch.ops.aten.clamp.default)
def _op_clamp(a, min=None, max=None):
    min_val = float(min) if min is not None else -1e30
    max_val = float(max) if max is not None else 1e30
    return _wrap(gpu.clamp(_unwrap(a), min_val, max_val))


# ============================================================
# Backward / autograd ops
# ============================================================

@register_op(torch.ops.aten.threshold_backward.default)
def _op_threshold_backward(grad_output, self_tensor, threshold):
    """Backward for relu: grad * (self > threshold) — Metal GPU."""
    return _wrap(gpu.threshold_backward(_unwrap(grad_output), _unwrap(self_tensor), float(threshold)),
                 torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)


@register_op(torch.ops.aten.gelu_backward.default)
def _op_gelu_backward(grad_output, self_tensor, approximate="none"):
    """Backward for gelu — Metal GPU for both tanh and exact modes."""
    if approximate == "tanh":
        return _wrap(gpu.gelu_tanh_backward(_unwrap(grad_output), _unwrap(self_tensor)),
                     torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)
    else:
        return _wrap(gpu.gelu_exact_backward(_unwrap(grad_output), _unwrap(self_tensor)),
                     torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)


@register_op(torch.ops.aten.tanh_backward.default)
def _op_tanh_backward(grad_output, output):
    """Backward for tanh: grad * (1 - output^2) — Metal GPU."""
    return _wrap(gpu.tanh_backward(_unwrap(grad_output), _unwrap(output)),
                 torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)


@register_op(torch.ops.aten.sigmoid_backward.default)
def _op_sigmoid_backward(grad_output, output):
    """Backward for sigmoid: grad * output * (1 - output) — Metal GPU."""
    return _wrap(gpu.sigmoid_backward(_unwrap(grad_output), _unwrap(output)),
                 torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)


@register_op(torch.ops.aten.mul.Scalar)
def _op_mul_scalar(a, scalar):
    """Multiply tensor by a scalar."""
    return _wrap(gpu.scalar_mul(_unwrap(a), float(scalar)))


@register_op(torch.ops.aten.div.Scalar)
def _op_div_scalar(a, scalar):
    """Divide tensor by a scalar."""
    return _wrap(gpu.scalar_mul(_unwrap(a), 1.0 / float(scalar)))


@register_op(torch.ops.aten.sum.default)
def _op_sum_default(a, dtype=None):
    """Sum all elements (no dim specified)."""
    gpu_a = _unwrap(a)
    # Reduce all dimensions by repeated sum on last dim + reshape
    current = gpu_a
    shape = list(current.shape)
    while len(shape) > 1:
        current = gpu.sum(current)
        shape = shape[:-1]
        current = gpu.reshape(current, shape)
    # Final reduction
    current = gpu.sum(current)
    return _wrap(gpu.reshape(current, [1]))


@register_op(torch.ops.aten.mean.default)
def _op_mean_default(a, dtype=None):
    """Mean of all elements."""
    gpu_a = _unwrap(a)
    total_elems = 1
    for s in gpu_a.shape:
        total_elems *= s
    sum_result = _op_sum_default(a, dtype)
    return _wrap(gpu.scalar_mul(_unwrap(sum_result), 1.0 / total_elems))


@register_op(torch.ops.aten.fill_.Scalar)
def _op_fill_scalar(a, value):
    """Fill tensor in-place with a scalar value."""
    gpu_a = _unwrap(a)
    n_elems = 1
    for s in gpu_a.shape:
        n_elems *= s
    new_gpu = gpu.tensor([float(value)] * n_elems, shape=list(gpu_a.shape))
    if isinstance(a, ApplegpuTensor):
        _gpu_tensor_registry[a.data_ptr()] = new_gpu
        a._gpu_tensor = new_gpu
    return _wrap(new_gpu)


@register_op(torch.ops.aten.zero_.default)
def _op_zero_(a):
    """Zero a tensor in-place."""
    return _op_fill_scalar(a, 0.0)


@register_op(torch.ops.aten.ones_like.default)
def _op_ones_like(a, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    gpu_a = _unwrap(a)
    n_elems = 1
    for s in gpu_a.shape:
        n_elems *= s
    gpu_t = gpu.tensor([1.0] * max(n_elems, 1), shape=list(gpu_a.shape))
    torch_dtype = dtype if dtype is not None else (a.dtype if isinstance(a, torch.Tensor) else torch.float32)
    return _wrap(gpu_t, torch_dtype=torch_dtype)


@register_op(torch.ops.aten.zeros_like.default)
def _op_zeros_like(a, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    gpu_a = _unwrap(a)
    n_elems = 1
    for s in gpu_a.shape:
        n_elems *= s
    gpu_t = gpu.tensor([0.0] * max(n_elems, 1), shape=list(gpu_a.shape))
    torch_dtype = dtype if dtype is not None else (a.dtype if isinstance(a, torch.Tensor) else torch.float32)
    return _wrap(gpu_t, torch_dtype=torch_dtype)


@register_op(torch.ops.aten.linspace.default)
def _op_linspace(start, end, steps, dtype=None, layout=None, device=None, pin_memory=None):
    """Create linearly spaced tensor on GPU. Uses numpy for fast generation, single memcpy to Metal.

    TODO: No dedicated Metal kernel — data is generated on CPU via numpy and copied.
    A GPU kernel computing `start + id * step` per thread would avoid the copy,
    but linspace is typically called once during model init (positional encodings),
    not on the hot path. The numpy→Metal memcpy is negligible for typical sizes.
    """
    import numpy as np
    steps = int(steps)
    data = np.linspace(float(start), float(end), steps, dtype=np.float32)
    return _wrap(gpu.from_numpy(data))


@register_op(torch.ops.aten.normal_.default)
def _op_normal_(a, mean=0.0, std=1.0, generator=None):
    """Fill with random normal values. Generated on CPU, transferred to GPU.

    TODO: No Metal kernel — CPU RNG + memcpy. A GPU Philox counter-based PRNG
    kernel would eliminate the transfer, but weight initialization is a cold path
    (called once per model init, not per forward pass).
    """
    import numpy as np
    if isinstance(a, ApplegpuTensor):
        shape = tuple(a.shape)
    else:
        shape = tuple(a.shape)
    n = 1
    for s in shape:
        n *= s
    data = np.random.normal(float(mean), float(std), n).astype(np.float32)
    new_gpu = gpu.from_numpy(data.reshape(shape))
    # Replace the GPU tensor on the ApplegpuTensor in-place
    if isinstance(a, ApplegpuTensor):
        a._gpu_tensor = new_gpu
        _gpu_tensor_registry[a.data_ptr()] = new_gpu
    return a


@register_op(torch.ops.aten.new_empty_strided.default)
def _op_new_empty_strided(a, size, stride, dtype=None, layout=None, device=None, pin_memory=None):
    """Create empty tensor with given size (ignore stride — always contiguous)."""
    shape = list(size)
    n_elems = 1
    for s in shape:
        n_elems *= s
    torch_dtype = dtype if dtype is not None else (a.dtype if isinstance(a, torch.Tensor) else torch.float32)
    gpu_dtype = _TORCH_TO_GPU_DTYPE.get(torch_dtype, "float32")
    gpu_t = gpu.tensor([0.0] * max(n_elems, 1), shape=shape, dtype=gpu_dtype)
    return _wrap(gpu_t, torch_dtype=torch_dtype)


@register_op(torch.ops.aten.where.self)
def _op_where(condition, x, y):
    cond_gpu = _unwrap(condition)
    # where requires float condition on GPU; fall back if bool/int
    if cond_gpu.dtype not in ("float32", "float16"):
        return NotImplemented
    return _wrap(gpu.where_cond(cond_gpu, _unwrap(x), _unwrap(y)))


@register_op(torch.ops.aten.masked_fill.Scalar)
def _op_masked_fill(input, mask, value):
    return _wrap(gpu.masked_fill(_unwrap(input), _unwrap(mask), float(value)))


@register_op(torch.ops.aten.triu.default)
def _op_triu(input, diagonal=0):
    return _wrap(gpu.triu(_unwrap(input), diagonal))


@register_op(torch.ops.aten.tril.default)
def _op_tril(input, diagonal=0):
    return _wrap(gpu.tril(_unwrap(input), diagonal))


@register_op(torch.ops.aten.bitwise_and.Tensor)
def _op_bitwise_and(a, b):
    """Handle bitwise_and by converting to CPU — used in attention masking."""
    return NotImplemented


@register_op(torch.ops.aten.bitwise_or.Tensor)
def _op_bitwise_or(a, b):
    return NotImplemented


@register_op(torch.ops.aten.bitwise_not.default)
def _op_bitwise_not(a):
    return NotImplemented


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
    gpu_a = _unwrap(a)
    shape = list(gpu_a.shape)
    ndim = len(shape)
    # Normalize dim list
    dims = sorted([d if d >= 0 else d + ndim for d in dim])

    if dims == [ndim - 1]:
        # Simple last-dim reduction
        result = _wrap(gpu.sum(gpu_a))
        if keepdim:
            new_shape = shape[:]
            new_shape[-1] = 1
            result = _wrap(gpu.reshape(_unwrap(result), new_shape))
        return result

    # Multi-dim or non-last-dim reduction: reduce dims from highest to lowest
    # to avoid shape index invalidation
    current = gpu_a
    current_shape = shape[:]

    # Reduce all requested dims; must handle non-last dims via transpose trick
    for d in reversed(dims):
        cs = list(current.shape)
        cur_ndim = len(cs)
        if d == cur_ndim - 1:
            # Last dim: reduce directly
            current = gpu.sum(current)
            cs.pop()
            if len(cs) == 0:
                cs = [1]
            current = gpu.reshape(current, cs)
        elif cur_ndim == 2:
            # 2D, reducing dim 0: transpose, sum last, reshape
            current = gpu.transpose(current)
            current = gpu.sum(current)
            new_len = cs[1]
            current = gpu.reshape(current, [new_len])
        else:
            # General case: transpose target dim to last, reduce, transpose back
            current = gpu.transpose_dims(current, d, cur_ndim - 1)
            current = gpu.sum(current)
            new_cs = list(current.shape)
            if len(new_cs) == 0:
                new_cs = [1]
            current = gpu.reshape(current, new_cs)
            # The shape after removing the last dim (which was originally dim d)
            # No need to transpose back since the dim is gone

    if keepdim:
        result_shape = shape[:]
        for d in dims:
            result_shape[d] = 1
        current = gpu.reshape(current, result_shape)

    return _wrap(current)


@register_op(torch.ops.aten.mean.dim)
def _op_mean(a, dim, keepdim=False, dtype=None):
    gpu_a = _unwrap(a)
    shape = list(gpu_a.shape)
    ndim = len(shape)
    dims = sorted([d if d >= 0 else d + ndim for d in dim])

    if dims == [ndim - 1]:
        # Single last-dim reduction -- use native mean
        result = _wrap(gpu.mean(gpu_a))
        if keepdim:
            new_shape = shape[:]
            new_shape[-1] = 1
            result = _wrap(gpu.reshape(_unwrap(result), new_shape))
        return result

    # Multi-dim reduction: reduce one dim at a time from highest to lowest
    # to avoid shape index invalidation
    current = gpu_a
    current_shape = shape[:]
    for d in reversed(dims):
        # Reshape to merge the target dim into the last dim, reduce, then reshape back
        # Simpler: use avg_pool2d for spatial dims, or fall back to CPU for complex cases
        pass

    # For now, handle the common case: reducing last two spatial dims [H, W]
    # which is what adaptive_avg_pool2d(1) produces via mean
    if len(dims) == 2 and dims[-1] == ndim - 1 and dims[-2] == ndim - 2:
        # Reduce last dim first, then second-to-last
        result = gpu.mean(current)  # reduces last dim
        result_shape = current_shape[:-1]
        result = gpu.reshape(result, result_shape)
        result = gpu.mean(result)   # reduces what was second-to-last (now last)
        result_shape = result_shape[:-1]
        result = gpu.reshape(result, result_shape)
        if keepdim:
            for d in dims:
                result_shape.insert(d, 1)
            result = gpu.reshape(result, result_shape)
        return _wrap(result)

    return NotImplemented


# ============================================================
# Shape ops
# ============================================================

@register_op(torch.ops.aten.reshape.default)
def _op_reshape(a, shape):
    shape = list(shape)
    gpu_a = _unwrap(a)
    # Handle -1 (infer dimension)
    if -1 in shape:
        total = 1
        for s in gpu_a.shape:
            total *= s
        known = 1
        neg_idx = -1
        for i, s in enumerate(shape):
            if s == -1:
                neg_idx = i
            else:
                known *= s
        if neg_idx >= 0 and known > 0:
            shape[neg_idx] = total // known
    return _wrap(gpu.reshape(gpu_a, shape))


@register_op(torch.ops.aten.view.default)
def _op_view(a, size):
    size = list(size)
    gpu_a = _unwrap(a)
    # Handle -1 (infer dimension)
    if -1 in size:
        total = 1
        for s in gpu_a.shape:
            total *= s
        known = 1
        neg_idx = -1
        for i, s in enumerate(size):
            if s == -1:
                neg_idx = i
            else:
                known *= s
        if neg_idx >= 0 and known > 0:
            size[neg_idx] = total // known
    return _wrap(gpu.reshape(gpu_a, size))


@register_op(torch.ops.aten.t.default)
def _op_t(a):
    return _wrap(gpu.transpose(_unwrap(a)))


@register_op(torch.ops.aten.transpose.int)
def _op_transpose_int(a, dim0, dim1):
    return _wrap(gpu.transpose_dims(_unwrap(a), dim0, dim1))


@register_op(torch.ops.aten.transpose_.default)
def _op_transpose_inplace(a, dim0, dim1):
    # transpose_ is a view op that swaps dims. Return a new wrapped tensor
    # with the correct shape since we can't modify PyTorch's internal metadata.
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
    # Non-float dtypes: fall back to CPU
    if gpu_a.dtype not in ("float32", "float16"):
        return NotImplemented
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


@register_op(torch.ops.aten.alias.default)
def _op_alias(a):
    """Alias: create a new ApplegpuTensor sharing the same GPU data.

    Must return a distinct tensor object to satisfy PyTorch autograd
    invariant that views are not identical to their base.
    """
    if isinstance(a, ApplegpuTensor):
        gt = a._gpu
        if gt is not None:
            return _wrap(gt, torch_dtype=a.dtype)
    return _handle_clone((a,), {})


@register_op(torch.ops.aten._unsafe_view.default)
def _op_unsafe_view(a, size):
    size = list(size)
    gpu_a = _unwrap(a)
    if -1 in size:
        total = 1
        for s in gpu_a.shape:
            total *= s
        known = 1
        neg_idx = -1
        for i, s in enumerate(size):
            if s == -1:
                neg_idx = i
            else:
                known *= s
        if neg_idx >= 0 and known > 0:
            size[neg_idx] = total // known
    return _wrap(gpu.reshape(gpu_a, size))


@register_op(torch.ops.aten.native_dropout.default)
def _native_dropout(input, p, train):
    """Dropout: no-op in eval mode."""
    if not train:
        # Return (input, all-true mask)
        gpu_input = _unwrap(input)
        n_elems = 1
        for s in gpu_input.shape:
            n_elems *= s
        ones = gpu.tensor([1.0] * n_elems, shape=list(gpu_input.shape))
        return input, _wrap(ones, torch_dtype=torch.bool)
    return NotImplemented


@register_op(torch.ops.aten.rsub.Scalar)
def _op_rsub_scalar(a, other, alpha=1):
    """rsub: other - alpha * a."""
    gpu_a = _unwrap(a)
    n_elems = 1
    for s in gpu_a.shape:
        n_elems *= s
    scalar_t = gpu.tensor([float(other)] * n_elems, shape=list(gpu_a.shape))
    if alpha != 1:
        gpu_a = gpu.scalar_mul(gpu_a, float(alpha))
    return _wrap(gpu.sub(scalar_t, gpu_a))


@register_op(torch.ops.aten.baddbmm.default)
def _op_baddbmm(self_tensor, batch1, batch2, beta=1, alpha=1):
    """baddbmm: beta * self + alpha * (batch1 @ batch2)."""
    result = _wrap(gpu.matmul(_unwrap(batch1), _unwrap(batch2)))
    if alpha != 1:
        result = _wrap(gpu.scalar_mul(_unwrap(result), float(alpha)))
    if beta != 0:
        self_gpu = _unwrap(self_tensor)
        if beta != 1:
            self_gpu = gpu.scalar_mul(self_gpu, float(beta))
        result = _wrap(gpu.add(_unwrap(result), self_gpu))
    return result


@register_op(torch.ops.aten.arange.default)
def _op_arange(end, dtype=None, layout=None, device=None, pin_memory=None):
    n = int(end)
    gpu_t = gpu.tensor([float(i) for i in range(n)], shape=[n])
    torch_dtype = dtype if dtype is not None else torch.float32
    return _wrap(gpu_t, torch_dtype=torch_dtype)


try:
    @register_op(torch.ops.aten.arange.start)
    def _op_arange_start(start, end, dtype=None, layout=None, device=None, pin_memory=None):
        s, e = int(start), int(end)
        vals = [float(i) for i in range(s, e)]
        n = len(vals)
        if n == 0:
            vals = [0.0]
            n = 1
        gpu_t = gpu.tensor(vals, shape=[n])
        torch_dtype = dtype if dtype is not None else torch.float32
        return _wrap(gpu_t, torch_dtype=torch_dtype)
except Exception:
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
    """In-place copy — GPU blit when both are on GPU, CPU fallback otherwise."""
    if isinstance(dst, ApplegpuTensor) and isinstance(src, ApplegpuTensor):
        try:
            gpu.blit_copy(_unwrap(src), _unwrap(dst))
            return dst
        except (RuntimeError, ValueError) as e:
            import warnings
            warnings.warn(f"GPU blit copy failed ({e}), falling back to CPU", UserWarning, stacklevel=2)
    # CPU fallback
    src_cpu = src.to_torch_cpu() if isinstance(src, ApplegpuTensor) else src
    new_gpu = gpu.from_torch(src_cpu)
    if isinstance(dst, ApplegpuTensor):
        dst._gpu_tensor = new_gpu
        _gpu_tensor_registry[dst.data_ptr()] = new_gpu
    return dst


# ============================================================
# Tensor manipulation ops
# ============================================================

@register_op(torch.ops.aten.cat.default)
def _op_cat(tensors, dim=0):
    # Filter out empty tensors (0 elements) — these come from KV cache init
    non_empty = []
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.numel() == 0:
            continue
        non_empty.append(t)
    if len(non_empty) == 0:
        return tensors[0] if isinstance(tensors[0], ApplegpuTensor) else _wrap(_unwrap(tensors[0]))
    if len(non_empty) == 1:
        return non_empty[0] if isinstance(non_empty[0], ApplegpuTensor) else _wrap(_unwrap(non_empty[0]))
    gpu_tensors = [_unwrap(t) for t in non_empty]
    result = gpu_tensors[0]
    for t in gpu_tensors[1:]:
        result = gpu.concat(result, t, dim)
    return _wrap(result)


@register_op(torch.ops.aten.stack.default)
def _op_stack(tensors, dim=0):
    """Stack tensors along a new dimension. Equivalent to unsqueeze + cat."""
    # Insert new dimension via reshape, then concatenate
    gpu_tensors = [_unwrap(t) for t in tensors]
    reshaped = []
    for gt in gpu_tensors:
        shape = list(gt.shape)
        # Insert size-1 dimension at `dim`
        new_shape = shape[:dim] + [1] + shape[dim:]
        reshaped.append(gpu.reshape(gt, new_shape))
    result = reshaped[0]
    for t in reshaped[1:]:
        result = gpu.concat(result, t, dim)
    return _wrap(result)


@register_op(torch.ops.aten.index.Tensor)
def _op_index_tensor(a, indices):
    """Advanced/boolean indexing. Routes simple integer index on dim 0 of 2D
    tensors to GPU index_select; falls back to CPU for other patterns.
    """
    # GPU fast path: single integer index tensor on dim 0, 2D input
    if (len(indices) == 1 and indices[0] is not None
            and isinstance(indices[0], ApplegpuTensor)):
        idx = indices[0]
        idx_cpu = idx.to_torch_cpu()
        if idx_cpu.dtype in (torch.int32, torch.int64):
            a_unwrapped = _unwrap(a)
            a_shape = gpu.shape(a_unwrapped)
            if len(a_shape) == 2:
                # Cast to int32 if needed (GPU kernels require Int32 indices)
                if idx_cpu.dtype == torch.int64:
                    idx_i32 = ApplegpuTensor.from_torch(idx_cpu.to(torch.int32))
                    idx_gpu = _unwrap(idx_i32)
                else:
                    idx_gpu = _unwrap(idx)
                return _wrap(gpu.index_select(a_unwrapped, 0, idx_gpu))
    # CPU fallback for all other cases (boolean masks, multi-index, etc.)
    a_cpu = a.to_torch_cpu() if isinstance(a, ApplegpuTensor) else a
    idx_cpu = []
    for idx in indices:
        if idx is None:
            idx_cpu.append(None)
        elif isinstance(idx, ApplegpuTensor):
            idx_cpu.append(idx.to_torch_cpu())
        else:
            idx_cpu.append(idx)
    result = torch.ops.aten.index.Tensor(a_cpu, idx_cpu)
    return ApplegpuTensor.from_torch(result)


@register_op(torch.ops.aten.index_put_.default)
def _op_index_put_(a, indices, values, accumulate=False):
    """Advanced index assignment. Routes simple integer index on dim 0 of 2D
    tensors to GPU scatter_write/scatter_add; falls back to CPU for other patterns.
    """
    # GPU fast path: single integer index on dim 0, 2D tensor
    if (len(indices) == 1 and indices[0] is not None
            and isinstance(indices[0], ApplegpuTensor)):
        idx = indices[0]
        idx_cpu = idx.to_torch_cpu()
        if idx_cpu.dtype in (torch.int32, torch.int64):
            a_unwrapped = _unwrap(a)
            a_shape = gpu.shape(a_unwrapped)
            if len(a_shape) == 2:
                # Cast to int32 if needed
                if idx_cpu.dtype == torch.int64:
                    idx_i32 = ApplegpuTensor.from_torch(idx_cpu.to(torch.int32))
                    idx_gpu = _unwrap(idx_i32)
                else:
                    idx_gpu = _unwrap(idx)
                v_unwrapped = _unwrap(values)
                if accumulate:
                    result = gpu.scatter_add(a_unwrapped, idx_gpu, v_unwrapped)
                else:
                    result = gpu.scatter_write(a_unwrapped, idx_gpu, v_unwrapped)
                return _update_inplace(a, result)
    # CPU fallback
    a_cpu = a.to_torch_cpu() if isinstance(a, ApplegpuTensor) else a
    idx_cpu = []
    for idx in indices:
        if idx is None:
            idx_cpu.append(None)
        elif isinstance(idx, ApplegpuTensor):
            idx_cpu.append(idx.to_torch_cpu())
        else:
            idx_cpu.append(idx)
    v_cpu = values.to_torch_cpu() if isinstance(values, ApplegpuTensor) else values
    result = torch.ops.aten.index_put_(a_cpu, idx_cpu, v_cpu, accumulate)
    return ApplegpuTensor.from_torch(result)


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
    # If the slice is the full dimension, return a new wrapper (not the same object,
    # to satisfy PyTorch autograd invariant that view ops return distinct tensors)
    if actual_start == 0 and actual_end == shape[dim]:
        return _wrap(gpu_a)
    # Non-float dtypes: fall back to CPU for slice
    if gpu_a.dtype not in ("float32", "float16"):
        return NotImplemented
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
    idx_gpu = _unwrap(indices)
    # Our embedding kernel requires Int32 indices; cast Int64 on GPU (no CPU roundtrip)
    if idx_gpu.dtype == "int64":
        idx_gpu = gpu.cast(idx_gpu, "int32")
    return _wrap(gpu.embedding(_unwrap(weight), idx_gpu))


# ============================================================
# CNN ops
# ============================================================

@register_op(torch.ops.aten.convolution.default)
def _convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    # Only support non-transposed, dilation=1
    if transposed or any(d != 1 for d in dilation):
        return NotImplemented

    ndim = len(stride)
    if ndim == 1:
        result = _wrap(gpu.conv1d(_unwrap(input), _unwrap(weight), stride[0], padding[0], groups))
    elif ndim == 2:
        result = _wrap(gpu.conv2d(_unwrap(input), _unwrap(weight), stride[0], stride[1], padding[0], padding[1], groups))
    else:
        return NotImplemented

    if bias is not None:
        # add_bias supports N-D: bias[channel] added along dim 1
        result = _wrap(gpu.add_bias(_unwrap(result), _unwrap(bias)))
    return result


@register_op(torch.ops.aten._native_batch_norm_legit_no_training.default)
def _batch_norm(input, weight, bias, running_mean, running_var, momentum, eps):
    result = _wrap(gpu.batch_norm(_unwrap(input), _unwrap(running_mean), _unwrap(running_var), _unwrap(weight), _unwrap(bias), eps))
    # Returns (output, mean, rstd) -- return dummies for mean/rstd
    return result, torch.tensor([]), torch.tensor([])


@register_op(torch.ops.aten.native_batch_norm.default)
def _native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps):
    """native_batch_norm for eval mode (training=False)."""
    if training:
        return NotImplemented
    result = _wrap(gpu.batch_norm(_unwrap(input), _unwrap(running_mean), _unwrap(running_var), _unwrap(weight), _unwrap(bias), eps))
    # Returns (output, save_mean, save_invstd)
    gpu_input = _unwrap(input)
    n_channels = gpu_input.shape[1]
    return result, torch.zeros(n_channels), torch.zeros(n_channels)


@register_op(torch.ops.aten.max_pool2d_with_indices.default)
def _max_pool2d(input, kernel_size, stride=None, padding=(0, 0), dilation=(1, 1), ceil_mode=False):
    if stride is None or len(stride) == 0:
        stride = kernel_size
    kh, kw = kernel_size[0], kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]
    sh, sw = stride[0], stride[1] if len(stride) > 1 else stride[0]
    ph, pw = padding[0], padding[1] if len(padding) > 1 else padding[0]
    values, indices = gpu.max_pool2d_with_indices(_unwrap(input), kh, kw, sh, sw, ph, pw)
    return _wrap(values), _wrap(indices)


@register_op(torch.ops.aten.avg_pool2d.default)
def _avg_pool2d(input, kernel_size, stride=None, padding=(0, 0), ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None or len(stride) == 0:
        stride = kernel_size
    kh, kw = kernel_size[0], kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]
    sh, sw = stride[0], stride[1] if len(stride) > 1 else stride[0]
    ph, pw = padding[0], padding[1] if len(padding) > 1 else padding[0]
    return _wrap(gpu.avg_pool2d(_unwrap(input), kh, kw, sh, sw, ph, pw))


@register_op(torch.ops.aten.adaptive_avg_pool2d.default)
def _adaptive_avg_pool2d(input, output_size):
    """Adaptive average pooling: compute kernel_size from input/output sizes."""
    gpu_input = _unwrap(input)
    in_h, in_w = gpu_input.shape[-2], gpu_input.shape[-1]
    out_h = output_size[0] if isinstance(output_size, (list, tuple)) else output_size
    out_w = output_size[1] if isinstance(output_size, (list, tuple)) and len(output_size) > 1 else out_h
    kh = in_h // out_h
    kw = in_w // out_w
    return _wrap(gpu.avg_pool2d(gpu_input, kh, kw, kh, kw, 0, 0))


@register_op(torch.ops.aten.unbind.int)
def _op_unbind(a, dim=0):
    """Unbind tensor along a dimension into a tuple of slices."""
    gpu_a = _unwrap(a)
    shape = list(gpu_a.shape)
    n = shape[dim]
    results = []
    for i in range(n):
        sliced = gpu.slice(gpu_a, dim, i, i + 1)
        # Remove the sliced dimension via reshape
        new_shape = shape[:dim] + shape[dim+1:]
        if not new_shape:
            new_shape = [1]
        results.append(_wrap(gpu.reshape(sliced, new_shape)))
    return tuple(results)


@register_op(torch.ops.aten.unsafe_split.Tensor)
def _op_unsafe_split(a, split_size, dim=0):
    """Split tensor into chunks along a dimension."""
    gpu_a = _unwrap(a)
    shape = list(gpu_a.shape)
    total = shape[dim]
    split_size = int(split_size)
    results = []
    start = 0
    while start < total:
        end = min(start + split_size, total)
        results.append(_wrap(gpu.slice(gpu_a, dim, start, end)))
        start = end
    return tuple(results)


@register_op(torch.ops.aten.linalg_vector_norm.default)
def _op_linalg_vector_norm(a, ord=2.0, dim=None, keepdim=False, dtype=None):
    """Vector norm — GPU-composed for L1/L2, CPU fallback for others."""
    if ord == 2.0:
        # L2: sqrt(sum(x^2))
        gpu_a = _unwrap(a)
        squared = _wrap(gpu.mul(gpu_a, gpu_a))
        if dim is not None:
            if isinstance(dim, int):
                dim = [dim]
            summed = _op_sum(squared, dim, keepdim=keepdim)
        else:
            summed = _op_sum_default(squared)
        return _wrap(gpu.sqrt(_unwrap(summed)))
    elif ord == 1.0:
        # L1: sum(|x|)
        gpu_a = _unwrap(a)
        abs_wrapped = _wrap(gpu.abs(gpu_a))
        if dim is not None:
            if isinstance(dim, int):
                dim = [dim]
            return _op_sum(abs_wrapped, dim, keepdim=keepdim)
        else:
            return _op_sum_default(abs_wrapped)
    elif ord == float('inf'):
        # L-inf: max(|x|) — amax kernel applies abs() internally
        if dim is None:
            # Global L-inf: flatten, then amax over last (only) dim
            gpu_a = _unwrap(a)
            flat = _wrap(gpu.reshape(gpu_a, [-1]))
            return _wrap(gpu.amax(_unwrap(flat)))
        elif isinstance(dim, int) and (dim == -1 or dim == len(a.shape) - 1):
            # Last-dim reduction — direct GPU path
            result = _wrap(gpu.amax(_unwrap(a)))
            if keepdim:
                result_shape = list(a.shape)
                result_shape[dim] = 1
                result = result.reshape(result_shape)
            return result
        else:
            # Non-last-dim L-inf — CPU fallback
            # TODO: GPU kernel supporting arbitrary reduction dim
            kwargs = {}
            if dtype is not None:
                kwargs['dtype'] = dtype
            return _cpu_fallback(torch.ops.aten.linalg_vector_norm.default,
                                 (a, ord, dim, keepdim), kwargs)
    else:
        # Other norms — CPU fallback
        kwargs = {}
        if dtype is not None:
            kwargs['dtype'] = dtype
        return _cpu_fallback(torch.ops.aten.linalg_vector_norm.default,
                             (a, ord, dim, keepdim), kwargs)


@register_op(torch.ops.aten._unique2.default)
def _op_unique2(a, sorted=True, return_inverse=False, return_counts=False):
    """Unique elements. Falls back to CPU.

    TODO: CPU fallback. GPU unique requires parallel sort + adjacent-difference
    + stream compaction — complex but doable. Used in hybrid model vectorized
    per-symbol dispatch.
    """
    a_cpu = a.to_torch_cpu() if isinstance(a, ApplegpuTensor) else a
    results = torch.ops.aten._unique2.default(a_cpu, sorted, return_inverse, return_counts)
    out = []
    for r in results:
        if isinstance(r, torch.Tensor) and r.numel() > 0:
            out.append(ApplegpuTensor.from_torch(r))
        elif isinstance(r, torch.Tensor):
            out.append(r)  # Keep empty tensors as CPU (can't allocate 0-byte GPU buffer)
        else:
            out.append(r)
    return tuple(out)


@register_op(torch.ops.aten.flatten.using_ints)
def _flatten(input, start_dim=0, end_dim=-1):
    """Flatten dimensions from start_dim to end_dim into a single dimension."""
    gpu_input = _unwrap(input)
    shape = list(gpu_input.shape)
    ndim = len(shape)
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim

    new_shape = shape[:start_dim]
    flat_size = 1
    for d in range(start_dim, end_dim + 1):
        flat_size *= shape[d]
    new_shape.append(flat_size)
    new_shape += shape[end_dim + 1:]

    return _wrap(gpu.reshape(gpu_input, new_shape))


# ============================================================
# Backward ops (native Metal kernels)
# ============================================================

@register_op(torch.ops.aten._softmax_backward_data.default)
def _op_softmax_backward(grad_output, output, dim, input_dtype):
    """Softmax backward — native Metal kernel."""
    go_gpu = _unwrap(grad_output)
    out_gpu = _unwrap(output)
    result = gpu.softmax_backward(go_gpu, out_gpu)
    return _wrap(result)


@register_op(torch.ops.aten.native_layer_norm_backward.default)
def _op_layer_norm_backward(grad_output, input, normalized_shape, mean, rstd, weight, bias, output_mask):
    """Layer norm backward — native Metal kernel for grad_input, CPU for grad_weight/grad_beta.

    TODO: grad_weight and grad_bias use CPU fallback (sum over batch dims).
    These are small tensors (size = hidden_dim) so the CPU path is not a
    bottleneck, but a Metal reduction kernel could eliminate the transfer.
    """
    go_gpu = _unwrap(grad_output)
    input_gpu = _unwrap(input)
    weight_gpu = _unwrap(weight) if weight is not None else None

    # grad_input via native Metal kernel
    if weight_gpu is not None:
        grad_input = _wrap(gpu.layer_norm_backward(go_gpu, input_gpu, weight_gpu, 1e-5))
    else:
        # Without weight, layer_norm_backward may not be available — fallback
        return NotImplemented

    # grad_weight and grad_beta: compute on CPU (sum over batch dims)
    # These are small tensors — not a performance bottleneck
    go_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
    input_cpu = input.to_torch_cpu() if isinstance(input, ApplegpuTensor) else input

    # Recompute normalized input
    mean_val = input_cpu.mean(dim=-1, keepdim=True)
    var_val = input_cpu.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (input_cpu - mean_val) / torch.sqrt(var_val + 1e-5)

    # Flatten batch dims for summation
    flat_go = go_cpu.reshape(-1, normalized_shape[0])
    flat_xnorm = x_norm.reshape(-1, normalized_shape[0])

    grad_weight = (flat_go * flat_xnorm).sum(dim=0) if output_mask[1] else None
    grad_bias = flat_go.sum(dim=0) if output_mask[2] else None

    if grad_weight is not None:
        grad_weight = ApplegpuTensor.from_torch(grad_weight)
    if grad_bias is not None:
        grad_bias = ApplegpuTensor.from_torch(grad_bias)

    return grad_input, grad_weight, grad_bias


@register_op(torch.ops.aten.convolution_backward.default)
def _op_conv_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask):
    """Conv backward — supports both conv1d (3D) and conv2d (4D).

    conv1d and conv2d grad_input use native Metal GPU kernels.
    TODO: conv2d grad_weight uses CPU fallback (weight gradient is a correlation,
    harder to parallelize than the input gradient which is a transposed conv).
    grad_bias uses CPU sum (could use existing gpu.sum reduction).
    """
    grad_input = None
    grad_weight = None
    grad_bias = None

    in_shape = input.shape if isinstance(input, ApplegpuTensor) else input.shape
    is_conv1d = len(in_shape) == 3

    if output_mask[0]:
        if is_conv1d:
            # Conv1d grad_input on Metal GPU
            go_gpu = _unwrap(grad_output)
            w_gpu = _unwrap(weight)
            in_shape_actual = input.shape if isinstance(input, ApplegpuTensor) else input.shape
            grad_input = _wrap(gpu.conv1d_backward_input(
                go_gpu, w_gpu,
                in_channels=in_shape_actual[1], in_len=in_shape_actual[2],
                stride=stride[0], padding=padding[0], groups=groups),
                torch_dtype=grad_output.dtype)
        else:
            # Conv2d grad_input on Metal
            in_h, in_w = in_shape[-2], in_shape[-1]
            sh, sw = stride[0], stride[1] if len(stride) > 1 else stride[0]
            ph, pw = padding[0], padding[1] if len(padding) > 1 else padding[0]
            grad_input = _wrap(gpu.conv2d_backward_input(
                _unwrap(grad_output), _unwrap(weight),
                int(in_h), int(in_w), int(sh), int(sw), int(ph), int(pw), groups
            ))

    if output_mask[1]:
        if is_conv1d:
            # Conv1d grad_weight: CPU fallback (no Metal kernel for conv1d backward weight)
            go_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
            in_cpu = input.to_torch_cpu() if isinstance(input, ApplegpuTensor) else input
            w_cpu = weight.to_torch_cpu() if isinstance(weight, ApplegpuTensor) else weight
            gw_cpu = torch.ops.aten.convolution_backward(
                go_cpu, in_cpu, w_cpu, bias_sizes, stride, padding, dilation,
                transposed, output_padding, groups, [False, True, False]
            )[1]
            grad_weight = ApplegpuTensor.from_torch(gw_cpu)
        else:
            # Conv2d grad_weight on Metal GPU
            w_shape = weight.shape if isinstance(weight, ApplegpuTensor) else weight.shape
            go_gpu = _unwrap(grad_output)
            in_gpu = _unwrap(input)
            sh, sw = stride[0], stride[1] if len(stride) > 1 else stride[0]
            ph, pw = padding[0], padding[1] if len(padding) > 1 else padding[0]
            in_channels_total = in_shape[1]
            grad_weight = _wrap(gpu.conv2d_backward_weight(
                go_gpu, in_gpu,
                int(w_shape[2]), int(w_shape[3]),
                int(w_shape[0]), int(in_channels_total),
                int(sh), int(sw), int(ph), int(pw), groups))

    if output_mask[2] and bias_sizes is not None:
        go_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
        if is_conv1d:
            grad_bias = ApplegpuTensor.from_torch(go_cpu.sum(dim=[0, 2]))
        else:
            grad_bias = ApplegpuTensor.from_torch(go_cpu.sum(dim=[0, 2, 3]))

    return grad_input, grad_weight, grad_bias


@register_op(torch.ops.aten.embedding_dense_backward.default)
def _op_embedding_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
    """Embedding backward — atomic scatter-add on Metal."""
    idx_gpu = _unwrap(indices)
    # Our embedding kernel requires Int32 indices; cast Int64 on GPU (no CPU roundtrip)
    if idx_gpu.dtype == "int64":
        idx_gpu = gpu.cast(idx_gpu, "int32")
    return _wrap(gpu.embedding_backward(_unwrap(grad_output), idx_gpu, int(num_weights)))


@register_op(torch.ops.aten.native_batch_norm_backward.default)
def _op_batch_norm_backward(grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask):
    """Batch norm backward (inference mode) — grad_input on Metal."""
    grad_input = None
    grad_weight = None
    grad_bias = None

    if output_mask[0]:
        grad_input = _wrap(gpu.batch_norm_backward(_unwrap(grad_output), _unwrap(weight), _unwrap(running_var), eps))

    if output_mask[1]:
        # grad_weight on CPU
        go_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
        in_cpu = input.to_torch_cpu() if isinstance(input, ApplegpuTensor) else input
        rm_cpu = running_mean.to_torch_cpu() if isinstance(running_mean, ApplegpuTensor) else running_mean
        rv_cpu = running_var.to_torch_cpu() if isinstance(running_var, ApplegpuTensor) else running_var
        x_norm = (in_cpu - rm_cpu.reshape(1, -1, 1, 1)) / torch.sqrt(rv_cpu.reshape(1, -1, 1, 1) + eps)
        grad_weight = ApplegpuTensor.from_torch((go_cpu * x_norm).sum(dim=[0, 2, 3]))

    if output_mask[2]:
        go_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
        grad_bias = ApplegpuTensor.from_torch(go_cpu.sum(dim=[0, 2, 3]))

    return grad_input, grad_weight, grad_bias


@register_op(torch.ops.aten.max_pool2d_with_indices_backward.default)
def _op_max_pool2d_backward(grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices):
    """max_pool2d backward — scatter gradients to max positions via Metal GPU."""
    in_shape = input.shape if isinstance(input, ApplegpuTensor) else input.shape
    idx_gpu = _unwrap(indices)
    # PyTorch MaxPool2d returns Int64 indices; our Metal kernel requires Int32
    if idx_gpu.dtype == "int64":
        idx_gpu = gpu.cast(idx_gpu, "int32")
    go_gpu = _unwrap(grad_output)
    result = gpu.max_pool2d_backward(go_gpu, idx_gpu,
                                      batch=in_shape[0], channels=in_shape[1],
                                      in_h=in_shape[2], in_w=in_shape[3])
    return _wrap(result, torch_dtype=grad_output.dtype)


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
    def __new__(cls, gpu_tensor, torch_dtype=None, requires_grad=False):
        shape = tuple(gpu_tensor.shape)
        if torch_dtype is None:
            torch_dtype = _GPU_TO_TORCH_DTYPE.get(gpu_tensor.dtype, torch.float32)

        # Create a real CPU tensor as backing storage (prevents segfaults in repr)
        backing = torch.zeros(shape, dtype=torch_dtype)
        r = torch.Tensor._make_subclass(cls, backing, require_grad=requires_grad)
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

    def __reduce_ex__(self, protocol):
        """Pickle support for torch.save — single GPU→CPU DMA, then serialize.

        Serialization requires host memory (pickle writes bytes, not GPU buffers).
        to_torch_cpu() uses direct data_ptr() copy — one DMA transfer, no
        intermediate Python objects. Same approach as CUDA and MPS backends.

        On load, tensors are plain CPU torch.Tensors. Call to_applegpu() to
        move back to GPU.
        """
        cpu_tensor = self.to_torch_cpu()
        return cpu_tensor.__reduce_ex__(protocol)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Intercept torch functions — pass through to __torch_dispatch__ with autograd."""
        kwargs = kwargs or {}

        # Let autograd see all operations so gradients can be computed
        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        global _in_fallback
        kwargs = kwargs or {}

        # If we're inside a CPU fallback, don't dispatch — let torch handle it natively
        if _in_fallback:
            return func(*args, **kwargs)

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
        return cls(gpu_t, torch_dtype=tensor.dtype, requires_grad=tensor.requires_grad)

    def to_torch_cpu(self):
        """Convert back to a CPU torch.Tensor.

        Also syncs the CPU backing storage so that if the lazy runtime later
        frees this tensor's ID, we can still reconstruct it.
        """
        gt = self._gpu
        if gt is not None:
            cpu_t = _gpu_tensor_to_torch_cpu(gt)
            # Sync backing storage (so _unwrap can recreate if Rust tensor freed)
            try:
                backing = torch.Tensor._make_subclass(torch.Tensor, self)
                backing.copy_(cpu_t)
            except Exception:
                pass
            return cpu_t
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

    # Patch GRU/RNN batch_first — their C++ implementation applies transpose_
    # internally, bypassing __torch_dispatch__. We intercept at the module level.
    _patch_rnn_batch_first(module)


def _patch_rnn_batch_first(module):
    """Wrap GRU/RNN forward to handle batch_first in Python instead of C++.

    PyTorch's C++ GRU/RNN implementation applies transpose_(0,1) internally
    when batch_first=True. This bypasses __torch_dispatch__, producing wrong
    output shapes. We disable batch_first on the module and do the transposes
    ourselves in Python where they flow through our dispatch.

    LSTM is NOT patched — its C++ batch_first implementation works correctly
    with __torch_dispatch__.
    """
    import torch.nn as nn
    # Only patch GRU and RNN (not LSTM — LSTM works fine with batch_first=True)
    if isinstance(module, (nn.GRU, nn.RNN)) and module.batch_first:
        original_forward = module.forward

        def patched_forward(input, hx=None):
            # Transpose (batch, seq, feat) → (seq, batch, feat)
            input = input.transpose(0, 1)
            # Run with batch_first=False
            module.batch_first = False
            try:
                output, hidden = original_forward(input, hx)
            finally:
                module.batch_first = True
            # Transpose output back: (seq, batch, feat) → (batch, seq, feat)
            output = output.transpose(0, 1)
            return output, hidden

        module.forward = patched_forward


def _cpu_fallback(func, args, kwargs):
    """Fallback: move all ApplegpuTensors to CPU, run op, move results back."""
    global _in_fallback
    op_name = str(func.name())
    _warn_fallback(op_name)

    # Convert args
    def to_cpu(x):
        if isinstance(x, ApplegpuTensor):
            gt = x._gpu
            if gt is not None:
                cpu_t = _gpu_tensor_to_torch_cpu(gt)
                # Restore original torch dtype if different from GPU dtype
                if cpu_t.dtype != x.dtype:
                    cpu_t = cpu_t.to(x.dtype)
                return cpu_t
            # _gpu lost: return backing zeros with correct shape/dtype
            return torch.zeros(x.shape, dtype=x.dtype)
        if isinstance(x, (list, tuple)):
            return type(x)(to_cpu(v) for v in x)
        return x

    # Set flag to prevent __torch_dispatch__ recursion during to_cpu
    was_in_fallback = _in_fallback
    _in_fallback = True
    try:
        cpu_args = to_cpu(args)
        cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}

        # Run on CPU
        try:
            result = func(*cpu_args, **cpu_kwargs)
        except (NotImplementedError, RuntimeError):
            # Retry with dtype coercion for bitwise/comparison ops
            # that fail when float is given instead of bool/int
            def coerce_bool(x):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    return x.bool()
                if isinstance(x, (list, tuple)):
                    return type(x)(coerce_bool(v) for v in x)
                return x
            cpu_args = coerce_bool(cpu_args)
            cpu_kwargs = {k: coerce_bool(v) for k, v in cpu_kwargs.items()}
            result = func(*cpu_args, **cpu_kwargs)
    finally:
        _in_fallback = was_in_fallback

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
