"""Custom torch.compile backend for applegpu.

Implements a FX graph interpreter that calls the Rust eager FFI directly
via ctypes, bypassing PyTorch's C++ Dispatcher. This eliminates the
~7.5us per-op dispatch overhead that dominates backward pass timing.

Usage:
    from applegpu_runtime.compile_backend import applegpu_compile_backend
    compiled_model = torch.compile(model, backend=applegpu_compile_backend)
"""

import ctypes
import glob
import os
import types
import sys

import torch
from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import make_boxed_func


# ── Ctypes setup ──────────────────────────────────────────────────

def _setup_accelerator():
    """Register applegpu as a torch accelerator (required for torch.compile)."""
    if hasattr(torch, 'applegpu'):
        return
    torch._C._rename_privateuse1_backend('applegpu')
    mod = types.ModuleType('torch.applegpu')
    mod.is_available = lambda: True
    mod.current_device = lambda: 0
    mod.device_count = lambda: 1
    mod.synchronize = lambda: _get_lib().applegpu_eager_flush_and_wait()
    mod.Stream = type('Stream', (), {'__init__': lambda self, *a, **kw: None})
    sys.modules['torch.applegpu'] = mod
    torch.applegpu = mod


def _get_lib():
    """Get ctypes handle to the C++ backend .so (contains Rust eager FFI)."""
    if hasattr(_get_lib, '_lib'):
        return _get_lib._lib
    backend_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend_cpp')
    so_files = glob.glob(os.path.join(backend_dir, 'applegpu_backend*.so'))
    if not so_files:
        raise FileNotFoundError("applegpu_backend .so not found")
    lib = ctypes.CDLL(so_files[0])
    _setup_ffi_signatures(lib)
    _get_lib._lib = lib
    return lib


def _setup_ffi_signatures(lib):
    """Set up ctypes signatures for all eager FFI functions."""
    u64 = ctypes.c_uint64
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i64 = ctypes.c_int64
    i32 = ctypes.c_int32
    i8 = ctypes.c_int8
    u32 = ctypes.c_uint32
    f32 = ctypes.c_float
    vp = ctypes.c_void_p

    sigs = [
        # Binary ops
        ('applegpu_eager_add', [u64, u64, u64p], vp),
        ('applegpu_eager_sub', [u64, u64, u64p], vp),
        ('applegpu_eager_mul', [u64, u64, u64p], vp),
        ('applegpu_eager_div', [u64, u64, u64p], vp),
        ('applegpu_eager_matmul', [u64, u64, u64p], vp),
        # Unary ops
        ('applegpu_eager_relu', [u64, u64p], vp),
        ('applegpu_eager_neg', [u64, u64p], vp),
        # Compound ops
        ('applegpu_eager_threshold_backward', [u64, u64, f32, u64p], vp),
        ('applegpu_eager_scalar_mul', [u64, f32, u64p], vp),
        ('applegpu_eager_mean_all', [u64, u64p], vp),
        ('applegpu_eager_sum_dim', [u64, i64, ctypes.c_bool, u64p], vp),
        # Views
        ('applegpu_eager_create_view', [u64, u64p, u64p, u32, u64, u64p], vp),
        # In-place
        ('applegpu_eager_add_inplace', [u64, u64], i32),
        ('applegpu_eager_add_scaled_inplace', [u64, u64, f32], i32),
        # Lifecycle
        ('applegpu_eager_free', [u64], None),
        ('applegpu_eager_flush_and_wait', [], None),
        ('applegpu_eager_synchronize', [], None),
        # Metadata
        ('applegpu_eager_shape', [u64, u64p, ctypes.POINTER(u32)], i32),
        ('applegpu_eager_dtype', [u64], i8),
        ('applegpu_eager_alloc', [u64p, u32, i8, u64p], vp),
        ('applegpu_eager_register_shape', [u64, u64p, u32], i32),
        # Reverse lookup
        ('applegpu_eager_find_by_data_ptr', [ctypes.c_void_p], u64),
    ]
    for fn_name, argtypes, restype in sigs:
        fn = getattr(lib, fn_name)
        fn.argtypes = argtypes
        fn.restype = restype


# ── DType mapping ─────────────────────────────────────────────────

_DTYPE_TO_WIRE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.float64: 2,
    torch.int8: 3,
    torch.int16: 4,
    torch.int32: 5,
    torch.int64: 6,
    torch.uint8: 7,
    torch.bool: 9,
    torch.bfloat16: 10,
}

_WIRE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_WIRE.items()}

_DTYPE_ITEMSIZE = {
    torch.float32: 4, torch.float16: 2, torch.float64: 8,
    torch.int8: 1, torch.int16: 2, torch.int32: 4, torch.int64: 8,
    torch.uint8: 1, torch.bool: 1, torch.bfloat16: 2,
}


# ── Tensor reference (lightweight, no PyTorch overhead) ───────────

class TensorRef:
    """Lightweight reference to an eager runtime tensor.

    Tracks tensor_id, shape, dtype, data_ptr, and contiguity without
    creating a PyTorch tensor. Used as intermediates in the FX interpreter.
    """
    __slots__ = ('tid', 'shape', 'dtype', 'data_ptr', 'is_contiguous')

    def __init__(self, tid, shape, dtype, data_ptr, is_contiguous=True):
        self.tid = tid
        self.shape = tuple(shape)
        self.dtype = dtype
        self.data_ptr = data_ptr
        self.is_contiguous = is_contiguous

    @property
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self):
        return self.numel * _DTYPE_ITEMSIZE[self.dtype]

    def dim(self):
        return len(self.shape)


# ── Tensor ID resolution ─────────────────────────────────────────

def _resolve_tensor_id(lib, t):
    """Resolve a PyTorch applegpu tensor to its eager tensor_id.

    For base tensors, does a reverse lookup by data_ptr.
    For views (different strides/offset from base), creates an eager view.
    """
    if not isinstance(t, torch.Tensor):
        return 0
    if t.device.type not in ('privateuseone', 'applegpu'):
        return 0

    storage_ptr = t.untyped_storage().data_ptr()
    if storage_ptr == 0:
        return 0

    base_id = lib.applegpu_eager_find_by_data_ptr(ctypes.c_void_p(storage_ptr))
    if base_id == 0:
        return 0

    # Check if we need to create a view (shape/strides/offset mismatch)
    dims = (ctypes.c_uint64 * 8)()
    ndim = ctypes.c_uint32(0)
    rc = lib.applegpu_eager_shape(base_id, dims, ctypes.byref(ndim))
    if rc != 0:
        return base_id

    # Compare shapes
    matches = (ndim.value == t.dim())
    if matches:
        for i in range(ndim.value):
            if dims[i] != t.size(i):
                matches = False
                break

    if matches and t.storage_offset() == 0:
        return base_id

    # Create an eager view with the tensor's actual shape/strides/offset
    ndim_val = t.dim()
    shape_arr = (ctypes.c_uint64 * ndim_val)()
    strides_arr = (ctypes.c_uint64 * ndim_val)()

    if t.is_contiguous():
        # Compute true contiguous strides (PyTorch allows arbitrary strides for size-1 dims)
        stride = 1
        for i in range(ndim_val - 1, -1, -1):
            shape_arr[i] = t.size(i)
            strides_arr[i] = stride
            stride *= t.size(i)
    else:
        for i in range(ndim_val):
            shape_arr[i] = t.size(i)
            strides_arr[i] = t.stride(i)

    view_id = ctypes.c_uint64(0)
    ptr = lib.applegpu_eager_create_view(
        base_id, shape_arr, strides_arr,
        ctypes.c_uint32(ndim_val),
        ctypes.c_uint64(t.storage_offset()),
        ctypes.byref(view_id))
    if ptr is None or ptr == 0:
        return base_id
    return view_id.value


def _query_shape(lib, tid):
    """Query tensor shape from eager runtime."""
    dims = (ctypes.c_uint64 * 8)()
    ndim = ctypes.c_uint32(0)
    lib.applegpu_eager_shape(tid, dims, ctypes.byref(ndim))
    return tuple(int(dims[i]) for i in range(ndim.value))


# ── Output wrapping ───────────────────────────────────────────────

def _wrap_output(lib, ref):
    """Convert a TensorRef to a proper PyTorch PrivateUse1 tensor.

    Allocates a new PyTorch tensor, memcpys the eager result into it.
    The eager result must already be flushed (GPU work complete).
    """
    if isinstance(ref, torch.Tensor):
        return ref

    shape = ref.shape
    dtype = ref.dtype
    nbytes = ref.nbytes

    if nbytes == 0:
        return torch.empty(shape, device='applegpu', dtype=dtype)

    # Allocate a fresh PyTorch tensor (goes through C++ Dispatcher once)
    out = torch.empty(shape, device='applegpu', dtype=dtype)
    # memcpy from eager result buffer to the new tensor's buffer
    # Both are in Metal shared memory (CPU-accessible)
    ctypes.memmove(out.data_ptr(), ref.data_ptr, nbytes)
    return out


# ── FX interpreter ────────────────────────────────────────────────

class EagerFXInterpreter:
    """Interprets an FX graph by calling eager FFI directly via ctypes.

    Instead of dispatching through PyTorch's C++ Dispatcher (7.5us/op),
    calls the Rust eager runtime directly (~0.5us/op via ctypes).

    Intermediates are tracked as TensorRef objects (no PyTorch overhead).
    Only output tensors are wrapped as proper PyTorch tensors.
    """

    def __init__(self, gm, lib, flush_on_entry=False):
        self.gm = gm
        self.lib = lib
        self._flush_on_entry = flush_on_entry
        self._out_id = ctypes.c_uint64(0)
        # Pre-allocated scratch for shape/stride arrays
        self._dims8 = (ctypes.c_uint64 * 8)()
        self._ndim = ctypes.c_uint32(0)
        # Deferred cleanup: tensor IDs from the PREVIOUS run, freed at the
        # start of the next run. This ensures buffers survive long enough for
        # the autograd engine and optimizer to consume the output tensors.
        self._deferred_free = []

    def run(self, *args):
        lib = self.lib

        # Backward graphs need a flush to ensure C++ dispatcher ops
        # (mse_loss_backward, sum_backward, etc.) have committed their
        # streaming CB before we encode our ops.
        if self._flush_on_entry:
            lib.applegpu_eager_flush_and_wait()

        # Free tensor IDs deferred from the previous run.
        if self._deferred_free:
            for tid in self._deferred_free:
                lib.applegpu_eager_free(tid)
            self._deferred_free = []

        env = {}
        created_ids = []  # tensor_ids created during this run
        ephemeral_view_ids = []  # view tensor_ids for input resolution
        ph_idx = 0

        try:
            for node in self.gm.graph.nodes:
                if node.op == 'placeholder':
                    env[node.name] = self._register_input(args[ph_idx], ephemeral_view_ids)
                    ph_idx += 1
                elif node.op == 'call_function':
                    env[node.name] = self._dispatch(node, env, created_ids)
                elif node.op == 'output':
                    return self._handle_output(node, env, created_ids, ephemeral_view_ids)

            raise RuntimeError("FX graph has no output node")
        except Exception:
            # On error, free immediately (no outputs were returned)
            self._cleanup(created_ids, ephemeral_view_ids)
            raise

    def _register_input(self, arg, ephemeral_view_ids):
        """Register an input tensor, creating TensorRef for it."""
        if not isinstance(arg, torch.Tensor):
            return arg
        if arg.device.type not in ('privateuseone', 'applegpu'):
            return arg

        storage_ptr = arg.untyped_storage().data_ptr()
        if storage_ptr == 0:
            return arg

        lib = self.lib
        base_id = lib.applegpu_eager_find_by_data_ptr(ctypes.c_void_p(storage_ptr))
        if base_id == 0:
            return arg  # Not an eager tensor — return as PyTorch tensor

        # Check if we need a view
        rc = lib.applegpu_eager_shape(base_id, self._dims8, ctypes.byref(self._ndim))
        if rc != 0:
            return TensorRef(base_id, arg.shape, arg.dtype, arg.data_ptr())

        matches = (self._ndim.value == arg.dim())
        if matches:
            for i in range(self._ndim.value):
                if self._dims8[i] != arg.size(i):
                    matches = False
                    break

        if matches and arg.storage_offset() == 0:
            return TensorRef(base_id, arg.shape, arg.dtype, arg.data_ptr(),
                             is_contiguous=arg.is_contiguous())

        # Create an eager view
        ndim_val = arg.dim()
        shape_arr = (ctypes.c_uint64 * ndim_val)()
        strides_arr = (ctypes.c_uint64 * ndim_val)()
        if arg.is_contiguous():
            stride = 1
            for i in range(ndim_val - 1, -1, -1):
                shape_arr[i] = arg.size(i)
                strides_arr[i] = stride
                stride *= arg.size(i)
        else:
            for i in range(ndim_val):
                shape_arr[i] = arg.size(i)
                strides_arr[i] = arg.stride(i)

        view_id = ctypes.c_uint64(0)
        ptr = lib.applegpu_eager_create_view(
            base_id, shape_arr, strides_arr,
            ctypes.c_uint32(ndim_val),
            ctypes.c_uint64(arg.storage_offset()),
            ctypes.byref(view_id))
        if ptr is None or ptr == 0:
            return TensorRef(base_id, arg.shape, arg.dtype, arg.data_ptr())

        ephemeral_view_ids.append(view_id.value)
        return TensorRef(view_id.value, arg.shape, arg.dtype, ptr,
                         is_contiguous=arg.is_contiguous())

    def _get_tid(self, val):
        """Get tensor_id from a TensorRef or PyTorch tensor."""
        if isinstance(val, TensorRef):
            return val.tid
        if isinstance(val, torch.Tensor) and val.device.type in ('privateuseone', 'applegpu'):
            return _resolve_tensor_id(self.lib, val)
        return 0

    def _get_shape(self, val):
        """Get shape from a TensorRef or tensor."""
        if isinstance(val, TensorRef):
            return val.shape
        if isinstance(val, torch.Tensor):
            return tuple(val.shape)
        return ()

    def _get_dtype(self, val):
        """Get dtype from a TensorRef or tensor."""
        if isinstance(val, TensorRef):
            return val.dtype
        if isinstance(val, torch.Tensor):
            return val.dtype
        return torch.float32

    def _ensure_contiguous(self, val, created_ids):
        """Ensure a TensorRef is contiguous for matmul.

        If already contiguous, returns (tid, False).
        If non-contiguous, copies via scalar_mul(1.0) and returns (new_tid, True).
        The bool indicates whether a GPU copy was encoded (caller must flush).
        """
        if not isinstance(val, TensorRef):
            return (self._get_tid(val), False)
        if val.is_contiguous:
            return (val.tid, False)
        # Non-contiguous: encode a strided→contiguous copy
        lib = self.lib
        ptr = lib.applegpu_eager_scalar_mul(
            val.tid, ctypes.c_float(1.0), ctypes.byref(self._out_id))
        if ptr:
            new_tid = self._out_id.value
            created_ids.append(new_tid)
            return (new_tid, True)
        return (val.tid, False)

    def _dispatch(self, node, env, created_ids):
        """Dispatch a single FX node to eager FFI or fallback."""
        target = node.target
        args = [env[a.name] if isinstance(a, torch.fx.Node) else a for a in node.args]
        kwargs = {k: env[v.name] if isinstance(v, torch.fx.Node) else v
                  for k, v in node.kwargs.items()}

        # Try eager dispatch first
        result = self._try_eager_dispatch(target, args, kwargs, created_ids)
        if result is not None:
            return result

        # Fallback: convert TensorRefs to PyTorch tensors, dispatch via PyTorch
        return self._fallback_dispatch(target, args, kwargs, created_ids)

    def _try_eager_dispatch(self, target, args, kwargs, created_ids):
        """Try to dispatch an aten op directly via eager FFI.

        Returns TensorRef on success, None if this op isn't handled.
        """
        lib = self.lib

        # ── Binary element-wise ops ──

        if target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.default):
            a_tid = self._get_tid(args[0])
            b_tid = self._get_tid(args[1])
            if a_tid and b_tid:
                # Check alpha parameter
                alpha = args[2] if len(args) > 2 else kwargs.get('alpha', 1)
                if alpha != 1:
                    return None  # Fall back for non-unit alpha
                ptr = lib.applegpu_eager_add(a_tid, b_tid, ctypes.byref(self._out_id))
                if ptr:
                    tid = self._out_id.value
                    created_ids.append(tid)
                    shape = _query_shape(lib, tid)
                    return TensorRef(tid, shape, self._get_dtype(args[0]), ptr)

        if target in (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.default):
            a_tid = self._get_tid(args[0])
            b_tid = self._get_tid(args[1])
            if a_tid and b_tid:
                alpha = args[2] if len(args) > 2 else kwargs.get('alpha', 1)
                if alpha != 1:
                    return None
                ptr = lib.applegpu_eager_sub(a_tid, b_tid, ctypes.byref(self._out_id))
                if ptr:
                    tid = self._out_id.value
                    created_ids.append(tid)
                    shape = _query_shape(lib, tid)
                    return TensorRef(tid, shape, self._get_dtype(args[0]), ptr)

        if target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.default):
            a_tid = self._get_tid(args[0])
            b_tid = self._get_tid(args[1])
            if a_tid and b_tid:
                ptr = lib.applegpu_eager_mul(a_tid, b_tid, ctypes.byref(self._out_id))
                if ptr:
                    tid = self._out_id.value
                    created_ids.append(tid)
                    shape = _query_shape(lib, tid)
                    return TensorRef(tid, shape, self._get_dtype(args[0]), ptr)

        if target == torch.ops.aten.div.Tensor:
            a_tid = self._get_tid(args[0])
            b_tid = self._get_tid(args[1])
            if a_tid and b_tid:
                ptr = lib.applegpu_eager_div(a_tid, b_tid, ctypes.byref(self._out_id))
                if ptr:
                    tid = self._out_id.value
                    created_ids.append(tid)
                    shape = _query_shape(lib, tid)
                    return TensorRef(tid, shape, self._get_dtype(args[0]), ptr)

        # ── Matmul ──

        if target == torch.ops.aten.mm.default:
            a_tid = self._get_tid(args[0])
            b_tid = self._get_tid(args[1])
            if a_tid and b_tid:
                # Matmul requires contiguous inputs — only copy if non-contiguous
                a_tid, _ = self._ensure_contiguous(args[0], created_ids)
                b_tid, _ = self._ensure_contiguous(args[1], created_ids)
                ptr = lib.applegpu_eager_matmul(a_tid, b_tid, ctypes.byref(self._out_id))
                if ptr:
                    tid = self._out_id.value
                    created_ids.append(tid)
                    shape = _query_shape(lib, tid)
                    return TensorRef(tid, shape, self._get_dtype(args[0]), ptr)

        # ── addmm: bias + mm(mat1, mat2) ──

        if target == torch.ops.aten.addmm.default:
            bias_tid = self._get_tid(args[0])
            mat1_tid = self._get_tid(args[1])
            mat2_tid = self._get_tid(args[2])
            beta = kwargs.get('beta', 1)
            alpha = kwargs.get('alpha', 1)
            if bias_tid and mat1_tid and mat2_tid and beta == 1 and alpha == 1:
                # Matmul requires contiguous inputs — only copy if non-contiguous
                mat1_tid, _ = self._ensure_contiguous(args[1], created_ids)
                mat2_tid, _ = self._ensure_contiguous(args[2], created_ids)
                # mm(mat1, mat2)
                mm_ptr = lib.applegpu_eager_matmul(
                    mat1_tid, mat2_tid, ctypes.byref(self._out_id))
                if mm_ptr:
                    mm_tid = self._out_id.value
                    created_ids.append(mm_tid)
                    # add(mm_result, bias) — bias broadcasts
                    add_ptr = lib.applegpu_eager_add(
                        mm_tid, bias_tid, ctypes.byref(self._out_id))
                    if add_ptr:
                        add_tid = self._out_id.value
                        created_ids.append(add_tid)
                        shape = _query_shape(lib, add_tid)
                        return TensorRef(add_tid, shape, self._get_dtype(args[1]), add_ptr)

        # ── Unary ops ──

        if target == torch.ops.aten.relu.default:
            tid = self._get_tid(args[0])
            if tid:
                ptr = lib.applegpu_eager_relu(tid, ctypes.byref(self._out_id))
                if ptr:
                    out_tid = self._out_id.value
                    created_ids.append(out_tid)
                    shape = _query_shape(lib, out_tid)
                    return TensorRef(out_tid, shape, self._get_dtype(args[0]), ptr)

        if target == torch.ops.aten.neg.default:
            tid = self._get_tid(args[0])
            if tid:
                ptr = lib.applegpu_eager_neg(tid, ctypes.byref(self._out_id))
                if ptr:
                    out_tid = self._out_id.value
                    created_ids.append(out_tid)
                    shape = _query_shape(lib, out_tid)
                    return TensorRef(out_tid, shape, self._get_dtype(args[0]), ptr)

        # ── threshold_backward (ReLU backward) ──

        if target == torch.ops.aten.threshold_backward.default:
            grad_tid = self._get_tid(args[0])
            input_tid = self._get_tid(args[1])
            threshold = float(args[2])
            if grad_tid and input_tid:
                ptr = lib.applegpu_eager_threshold_backward(
                    grad_tid, input_tid, ctypes.c_float(threshold),
                    ctypes.byref(self._out_id))
                if ptr:
                    out_tid = self._out_id.value
                    created_ids.append(out_tid)
                    shape = _query_shape(lib, out_tid)
                    return TensorRef(out_tid, shape, self._get_dtype(args[0]), ptr)

        # ── Scalar multiply ──

        if target == torch.ops.aten.mul.Scalar:
            tid = self._get_tid(args[0])
            scalar = float(args[1])
            if tid:
                ptr = lib.applegpu_eager_scalar_mul(
                    tid, ctypes.c_float(scalar), ctypes.byref(self._out_id))
                if ptr:
                    out_tid = self._out_id.value
                    created_ids.append(out_tid)
                    shape = _query_shape(lib, out_tid)
                    return TensorRef(out_tid, shape, self._get_dtype(args[0]), ptr)

        # ── Reductions ──

        if target == torch.ops.aten.mean.default:
            tid = self._get_tid(args[0])
            if tid:
                ptr = lib.applegpu_eager_mean_all(tid, ctypes.byref(self._out_id))
                if ptr:
                    out_tid = self._out_id.value
                    created_ids.append(out_tid)
                    return TensorRef(out_tid, (1,), self._get_dtype(args[0]), ptr)

        if target == torch.ops.aten.sum.dim_IntList:
            tid = self._get_tid(args[0])
            dim_list = args[1]
            keepdim = args[2] if len(args) > 2 else kwargs.get('keepdim', False)
            if tid and len(dim_list) == 1:
                dim = dim_list[0]
                ptr = lib.applegpu_eager_sum_dim(
                    tid, ctypes.c_int64(dim), ctypes.c_bool(keepdim),
                    ctypes.byref(self._out_id))
                if ptr:
                    out_tid = self._out_id.value
                    created_ids.append(out_tid)
                    shape = _query_shape(lib, out_tid)
                    return TensorRef(out_tid, shape, self._get_dtype(args[0]), ptr)

        # ── View/reshape ops (no compute, just metadata) ──

        if target == torch.ops.aten.t.default:
            val = args[0]
            tid = self._get_tid(val)
            src_shape = self._get_shape(val)
            if tid and len(src_shape) == 2:
                # Transpose: swap shape and strides
                new_shape = (ctypes.c_uint64 * 2)(src_shape[1], src_shape[0])
                # For a contiguous 2D tensor, strides are (cols, 1)
                # For transpose, strides become (1, cols) → new_strides = (original_stride[1], original_stride[0])
                # Query the actual strides from the eager runtime
                # For simplicity, use: transposed stride of [M,K] contiguous = (1, M)
                new_strides = (ctypes.c_uint64 * 2)(1, src_shape[1])
                view_id = ctypes.c_uint64(0)
                ptr = lib.applegpu_eager_create_view(
                    tid, new_shape, new_strides,
                    ctypes.c_uint32(2), ctypes.c_uint64(0),
                    ctypes.byref(view_id))
                if ptr:
                    vid = view_id.value
                    created_ids.append(vid)
                    return TensorRef(vid, (src_shape[1], src_shape[0]),
                                     self._get_dtype(val), ptr,
                                     is_contiguous=False)

        if target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default,
                      torch.ops.aten._unsafe_view.default):
            val = args[0]
            tid = self._get_tid(val)
            new_shape_list = list(args[1])
            if tid:
                # Infer -1 dimension
                src_numel = 1
                for s in self._get_shape(val):
                    src_numel *= s
                neg_idx = -1
                known_product = 1
                for i, s in enumerate(new_shape_list):
                    if s == -1:
                        neg_idx = i
                    else:
                        known_product *= s
                if neg_idx >= 0:
                    new_shape_list[neg_idx] = src_numel // known_product

                # Compute contiguous strides for the new shape
                ndim = len(new_shape_list)
                shape_arr = (ctypes.c_uint64 * ndim)()
                strides_arr = (ctypes.c_uint64 * ndim)()
                stride = 1
                for i in range(ndim - 1, -1, -1):
                    shape_arr[i] = new_shape_list[i]
                    strides_arr[i] = stride
                    stride *= new_shape_list[i]

                view_id = ctypes.c_uint64(0)
                ptr = lib.applegpu_eager_create_view(
                    tid, shape_arr, strides_arr,
                    ctypes.c_uint32(ndim), ctypes.c_uint64(0),
                    ctypes.byref(view_id))
                if ptr:
                    vid = view_id.value
                    created_ids.append(vid)
                    return TensorRef(vid, tuple(new_shape_list),
                                     self._get_dtype(val), ptr)

        # ── No-op passthrough ops ──

        if target in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return args[0]

        # Not handled — return None to trigger fallback
        return None

    def _fallback_dispatch(self, target, args, kwargs, created_ids):
        """Fallback: convert TensorRefs to PyTorch tensors, dispatch via PyTorch."""
        lib = self.lib
        # Flush GPU before reading any eager buffers
        lib.applegpu_eager_flush_and_wait()

        real_args = [self._to_pytorch(a, created_ids) for a in args]
        real_kwargs = {k: self._to_pytorch(v, created_ids) for k, v in kwargs.items()}
        result = target(*real_args, **real_kwargs)

        # Convert result back to TensorRef if it's on applegpu
        if isinstance(result, torch.Tensor) and result.device.type in ('privateuseone', 'applegpu'):
            tid = _resolve_tensor_id(lib, result)
            if tid:
                return TensorRef(tid, tuple(result.shape), result.dtype, result.data_ptr())
        return result

    def _to_pytorch(self, val, created_ids=None):
        """Convert a TensorRef to a proper PyTorch tensor."""
        if not isinstance(val, TensorRef):
            return val
        # Non-contiguous views must be made contiguous before flat memcpy
        if not val.is_contiguous:
            lib = self.lib
            ptr = lib.applegpu_eager_scalar_mul(
                val.tid, ctypes.c_float(1.0), ctypes.byref(self._out_id))
            if ptr:
                new_tid = self._out_id.value
                if created_ids is not None:
                    created_ids.append(new_tid)
                lib.applegpu_eager_flush_and_wait()
                val = TensorRef(new_tid, val.shape, val.dtype, ptr, is_contiguous=True)
        return _wrap_output(self.lib, val)

    def _handle_output(self, node, env, created_ids, ephemeral_view_ids):
        """Handle the output node: flush GPU, wrap results as PyTorch tensors."""
        lib = self.lib
        # Flush all pending GPU work
        lib.applegpu_eager_flush_and_wait()

        out_args = node.args[0]

        def resolve_output(a):
            if isinstance(a, torch.fx.Node):
                val = env[a.name]
                if isinstance(val, TensorRef):
                    # Non-contiguous views need strided→contiguous copy first
                    if not val.is_contiguous:
                        ptr = lib.applegpu_eager_scalar_mul(
                            val.tid, ctypes.c_float(1.0),
                            ctypes.byref(self._out_id))
                        if ptr:
                            new_tid = self._out_id.value
                            created_ids.append(new_tid)
                            lib.applegpu_eager_flush_and_wait()
                            val = TensorRef(new_tid, val.shape, val.dtype,
                                            ptr, is_contiguous=True)
                    return _wrap_output(lib, val)
                return val
            return a

        if isinstance(out_args, (tuple, list)):
            result = tuple(resolve_output(a) for a in out_args)
        elif isinstance(out_args, torch.fx.Node):
            result = resolve_output(out_args)
        else:
            result = out_args

        # Defer cleanup: the autograd engine and optimizer may still reference
        # the output tensors' underlying data. Free on the NEXT run() call.
        self._deferred_free.extend(created_ids)
        self._deferred_free.extend(ephemeral_view_ids)

        return result

    def _cleanup(self, created_ids, ephemeral_view_ids):
        """Free all tensor_ids immediately (used on error paths only)."""
        lib = self.lib
        for tid in created_ids:
            lib.applegpu_eager_free(tid)
        for tid in ephemeral_view_ids:
            lib.applegpu_eager_free(tid)


# ── Compiled graph runner (single FFI call) ───────────────────────

# Op codes — must match Rust compiled_graph.rs
_OP_CODES = {
    torch.ops.aten.add.Tensor: 0, torch.ops.aten.add.default: 0,
    torch.ops.aten.sub.Tensor: 1, torch.ops.aten.sub.default: 1,
    torch.ops.aten.mul.Tensor: 2, torch.ops.aten.mul.default: 2,
    torch.ops.aten.div.Tensor: 3,
    torch.ops.aten.mm.default: 4,
    torch.ops.aten.relu.default: 5,
    torch.ops.aten.neg.default: 6,
    torch.ops.aten.threshold_backward.default: 7,
    torch.ops.aten.t.default: 11,
    torch.ops.aten.addmm.default: 13,
}
_IDENTITY_OPS = {torch.ops.aten.detach.default, torch.ops.aten.alias.default}

import struct

def _serialize_graph(gm):
    """Serialize an FX graph into the wire format for Rust execution.

    Returns (ops_bytes, n_placeholders, output_indices).
    """
    node_index = {}  # node.name → index in the node array
    idx = 0
    ops = bytearray()
    n_placeholders = 0
    output_indices = []

    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            node_index[node.name] = idx
            idx += 1
            n_placeholders += 1

        elif node.op == 'call_function':
            target = node.target
            args = node.args
            kwargs = node.kwargs

            # Resolve input indices
            input_indices = []
            for a in args:
                if isinstance(a, torch.fx.Node):
                    input_indices.append(node_index[a.name])

            # Determine op code and params
            if target in _IDENTITY_OPS:
                op_code = 255  # OP_IDENTITY
                params = []
            elif target in _OP_CODES:
                op_code = _OP_CODES[target]
                params = []
                if op_code == 7:  # threshold_backward
                    threshold = float(args[2]) if len(args) > 2 else 0.0
                    params = [threshold]
            elif target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default,
                            torch.ops.aten._unsafe_view.default):
                op_code = 12  # OP_VIEW
                params = []
                # Infer shape with -1
                src_node = args[0]
                new_shape = list(args[1])
                if -1 in new_shape:
                    src_numel = 1
                    if isinstance(src_node, torch.fx.Node) and 'val' in src_node.meta:
                        for s in src_node.meta['val'].shape:
                            src_numel *= s
                    neg_idx = new_shape.index(-1)
                    known = 1
                    for s in new_shape:
                        if s != -1: known *= s
                    new_shape[neg_idx] = src_numel // known
            elif target == torch.ops.aten.mul.Scalar:
                op_code = 8  # OP_SCALAR_MUL
                params = [float(args[1])]
                input_indices = [node_index[args[0].name]]
            elif target == torch.ops.aten.mean.default:
                op_code = 9  # OP_MEAN_ALL
                params = []
            elif target == torch.ops.aten.sum.dim_IntList:
                op_code = 10  # OP_SUM_DIM
                dim_list = args[1]
                keepdim = args[2] if len(args) > 2 else kwargs.get('keepdim', False)
                if len(dim_list) == 1:
                    params = [float(dim_list[0]), 1.0 if keepdim else 0.0]
                else:
                    # Multi-dim sum: fall back (not serialized)
                    node_index[node.name] = idx
                    idx += 1
                    continue
            else:
                # Unsupported op — can't serialize
                return None

            # Get output shape from FX metadata
            if 'val' in node.meta:
                out_shape = list(node.meta['val'].shape)
            else:
                out_shape = [0]

            if target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default,
                          torch.ops.aten._unsafe_view.default):
                out_shape = new_shape

            # Encode op
            n_inp = len(input_indices)
            ops += struct.pack('<BB', op_code, n_inp)
            for i in input_indices:
                ops += struct.pack('<H', i)
            ops += struct.pack('<B', len(out_shape))
            for d in out_shape:
                ops += struct.pack('<Q', d)
            ops += struct.pack('<BB', 0, len(params))  # dtype=f32, n_params
            for p in params:
                ops += struct.pack('<f', p)

            node_index[node.name] = idx
            idx += 1

        elif node.op == 'output':
            out_args = node.args[0]
            # Track None positions for gradients that don't exist
            none_positions = []
            if isinstance(out_args, (tuple, list)):
                for i, a in enumerate(out_args):
                    if isinstance(a, torch.fx.Node):
                        output_indices.append(node_index[a.name])
                    elif a is None:
                        output_indices.append(0xFFFF)  # sentinel for None
                        none_positions.append(i)
            elif isinstance(out_args, torch.fx.Node):
                output_indices.append(node_index[out_args.name])

    return bytes(ops), n_placeholders, output_indices


class CompiledGraphRunner:
    """Executes a pre-serialized FX graph via a single Rust FFI call."""

    def __init__(self, gm, lib):
        self.gm = gm
        self.lib = lib
        self._deferred_free = []  # tensor IDs to free from previous run
        result = _serialize_graph(gm)
        if result is None:
            # Can't serialize — fall back to Python FX interpreter
            self._serialized = None
            self._interp = EagerFXInterpreter(gm, lib, flush_on_entry=True)
        else:
            self._serialized, self._n_placeholders, self._output_indices = result
            self._interp = None

        # Setup FFI signature
        if not hasattr(lib, '_graph_ffi_setup'):
            lib.applegpu_eager_execute_graph.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32,  # ops_data, ops_len
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,  # input_tids, n_inputs
                ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32,  # output_indices, n_outputs
                ctypes.POINTER(ctypes.c_uint64),  # out_tids
                ctypes.POINTER(ctypes.c_void_p),  # out_ptrs
            ]
            lib.applegpu_eager_execute_graph.restype = ctypes.c_int32
            lib._graph_ffi_setup = True

    def run(self, *args):
        if self._interp is not None:
            return self._interp.run(*args)
        return self._run_compiled(*args)

    def _run_compiled(self, *args):
        lib = self.lib

        # Free tensor IDs deferred from the previous run
        if self._deferred_free:
            for tid in self._deferred_free:
                lib.applegpu_eager_free(tid)
            self._deferred_free = []

        # Separate real outputs from None sentinels (0xFFFF)
        real_indices = [i for i in self._output_indices if i != 0xFFFF]
        n_real = len(real_indices)

        # Resolve input tensor IDs
        input_tids = (ctypes.c_uint64 * self._n_placeholders)()
        for i, arg in enumerate(args[:self._n_placeholders]):
            if isinstance(arg, torch.Tensor) and arg.device.type in ('privateuseone', 'applegpu'):
                tid = _resolve_tensor_id(lib, arg)
                input_tids[i] = tid
            else:
                input_tids[i] = 0

        # Prepare output arrays (only for real outputs, not None sentinels)
        out_tids = (ctypes.c_uint64 * n_real)()
        out_ptrs = (ctypes.c_void_p * n_real)()
        out_indices_arr = (ctypes.c_uint16 * n_real)(*real_indices)

        # Single FFI call — all ops execute in Rust
        ops_buf = ctypes.create_string_buffer(self._serialized)
        rc = lib.applegpu_eager_execute_graph(
            ctypes.cast(ops_buf, ctypes.c_void_p),
            ctypes.c_uint32(len(self._serialized)),
            input_tids, ctypes.c_uint32(self._n_placeholders),
            out_indices_arr, ctypes.c_uint32(n_real),
            out_tids, out_ptrs,
        )

        if rc < 0:
            err_fn = lib.applegpu_eager_last_error
            err_fn.restype = ctypes.c_char_p
            err = err_fn()
            raise RuntimeError(f"compiled graph execution failed: {err}")

        # Flush and wrap outputs as PyTorch tensors
        lib.applegpu_eager_flush_and_wait()
        # Collect input tensor IDs to avoid freeing pass-throughs
        input_tid_set = set(input_tids[i] for i in range(self._n_placeholders))

        real_results = []
        for i in range(n_real):
            tid = out_tids[i]
            ptr = out_ptrs[i]
            shape = _query_shape(lib, tid)
            dtype = torch.float32
            ref = TensorRef(tid, shape, dtype, ptr)
            real_results.append(_wrap_output(lib, ref))
            # Only defer-free tensor IDs CREATED by graph execution,
            # not input pass-throughs (which are still in use by the model)
            if tid not in input_tid_set:
                self._deferred_free.append(tid)

        # Reconstruct full output tuple with None at sentinel positions
        n_total = len(self._output_indices)
        if n_total == 1 and n_real == 1:
            return real_results[0]

        results = []
        real_idx = 0
        for oi in self._output_indices:
            if oi == 0xFFFF:
                results.append(None)
            else:
                results.append(real_results[real_idx])
                real_idx += 1
        return tuple(results)


# ── Compiler entry points ─────────────────────────────────────────

def _compiled_graph_compiler(gm, example_inputs):
    """Compile an FX graph into a serialized op list for Rust execution."""
    lib = _get_lib()
    runner = CompiledGraphRunner(gm, lib)

    def compiled(*args):
        return runner.run(*args)

    return make_boxed_func(compiled)


def applegpu_compile_backend(gm, example_inputs):
    """torch.compile backend for applegpu.

    Usage:
        compiled = torch.compile(model, backend=applegpu_compile_backend)
    """
    _setup_accelerator()
    return aot_module_simplified(gm, example_inputs,
        fw_compiler=_compiled_graph_compiler,
        bw_compiler=_compiled_graph_compiler)
