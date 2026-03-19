// PrivateUse1 C++ backend for applegpu_runtime.
//
// Registers a custom allocator and minimum ops at the PrivateUse1 dispatch key.
// All GPU work is delegated to the Rust graph engine via extern "C" FFI.

#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/InferSize.h>
#include "applegpu_ffi.h"

// ── Helpers ──────────────────────────────────────────────────────

namespace {

// Context stored in c10::DataPtr for each applegpu tensor.
struct TensorContext {
    uint64_t tensor_id;
};

uint64_t get_tensor_id(const at::Tensor& t) {
    auto* ctx = static_cast<TensorContext*>(t.storage().data_ptr().get_context());
    TORCH_CHECK(ctx != nullptr, "applegpu: tensor has no context (not an applegpu tensor?)");
    return ctx->tensor_id;
}

// Ensure a tensor is "op-ready": contiguous AND its logical shape matches
// the Rust runtime's registered shape. View tensors (e.g., weight.t()) share
// storage with a different tensor_id, so their logical shape may differ from
// what Rust knows. Force a contiguous copy in that case.
at::Tensor ensure_op_ready(const at::Tensor& t) {
    if (!t.device().is_privateuseone()) return t;
    if (!t.is_contiguous()) return t.contiguous();

    auto* ctx = static_cast<TensorContext*>(t.storage().data_ptr().get_context());
    if (ctx == nullptr) return t;

    // Query the registered shape from Rust and compare with PyTorch's shape
    uint64_t rust_dims[8];
    uint32_t rust_ndim = 0;
    int32_t rc = applegpu_ffi_shape(ctx->tensor_id, rust_dims, &rust_ndim);
    if (rc != 0) return t; // tensor not in Rust runtime — let op handle it

    // Compare shapes
    bool shape_matches = (rust_ndim == (uint32_t)t.dim());
    if (shape_matches) {
        for (uint32_t i = 0; i < rust_ndim; i++) {
            if (rust_dims[i] != (uint64_t)t.size(i)) {
                shape_matches = false;
                break;
            }
        }
    }

    if (!shape_matches) {
        // This is a view with different shape — copy to fresh tensor
        auto fresh = torch::empty(t.sizes(),
            at::TensorOptions().dtype(t.dtype()).device(t.device()));
        fresh.copy_(t);
        return fresh;
    }

    // Also check for storage_offset (subview)
    if (t.storage_offset() != 0) {
        auto fresh = torch::empty(t.sizes(),
            at::TensorOptions().dtype(t.dtype()).device(t.device()));
        fresh.copy_(t);
        return fresh;
    }

    return t;
}

// Map PyTorch ScalarType to our wire DType discriminant.
// Wire: 0=f32, 1=f16, 2=f64, 3=i8, 4=i16, 5=i32, 6=i64, 7=u8, 8=u32, 9=bool, 10=bf16
int8_t scalar_type_to_wire(at::ScalarType st) {
    switch (st) {
        case at::ScalarType::Float:   return 0;
        case at::ScalarType::Half:    return 1;
        case at::ScalarType::Double:  return 2;
        case at::ScalarType::Char:    return 3;  // Int8
        case at::ScalarType::Short:   return 4;
        case at::ScalarType::Int:     return 5;
        case at::ScalarType::Long:    return 6;
        case at::ScalarType::Byte:    return 7;  // UInt8
        case at::ScalarType::Bool:    return 9;
        case at::ScalarType::BFloat16: return 10;
        default:
            TORCH_CHECK(false, "applegpu: unsupported dtype ", at::toString(st));
    }
}

// Query output shape from Rust graph node (handles broadcasting).
std::vector<int64_t> query_output_shape(uint64_t tid) {
    uint64_t dims[8];
    uint32_t ndim = 0;
    int32_t rc = applegpu_ffi_shape(tid, dims, &ndim);
    TORCH_CHECK(rc == 0, "applegpu: failed to query output shape");
    std::vector<int64_t> sizes(ndim);
    for (uint32_t i = 0; i < ndim; i++) sizes[i] = static_cast<int64_t>(dims[i]);
    return sizes;
}

} // namespace

// ── Allocator ────────────────────────────────────────────────────

struct ApplegpuAllocator final : public c10::Allocator {
    c10::DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return {nullptr, nullptr, [](void*){}, c10::Device(c10::DeviceType::PrivateUse1, 0)};
        }
        uint64_t tensor_id = 0;
        void* ptr = applegpu_ffi_alloc(nbytes, 0, &tensor_id);
        TORCH_CHECK(ptr != nullptr, "applegpu alloc failed: ",
                     applegpu_ffi_last_error() ? applegpu_ffi_last_error() : "unknown");

        auto* ctx = new TensorContext{tensor_id};
        auto deleter = [](void* ctx_ptr) {
            auto* tc = static_cast<TensorContext*>(ctx_ptr);
            applegpu_ffi_free(tc->tensor_id);
            delete tc;
        };
        return {ptr, ctx, deleter, c10::Device(c10::DeviceType::PrivateUse1, 0)};
    }

    c10::DeleterFnPtr raw_deleter() const override {
        // Not used for DataPtr-based allocation, but required by interface.
        return [](void*){};
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        std::memcpy(dest, src, count);
    }
};

static ApplegpuAllocator global_allocator;

// ── Op Implementations ───────────────────────────────────────────

at::Tensor applegpu_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt
) {
    auto dtype = dtype_opt.value_or(at::ScalarType::Float);
    int64_t nbytes = at::detail::computeStorageNbytes(size, stride, at::elementSize(dtype));

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        nbytes,
        global_allocator.allocate(nbytes),
        &global_allocator
    );

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        at::scalarTypeToTypeMeta(dtype)
    );
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

    // Register shape metadata in Rust runtime
    if (nbytes > 0) {
        uint64_t tid = get_tensor_id(tensor);
        std::vector<uint64_t> dims(size.begin(), size.end());
        int8_t dtype_i8 = scalar_type_to_wire(dtype);
        applegpu_ffi_register_tensor(tid, dims.data(), dims.size(), dtype_i8);
    }

    return tensor;
}

at::Tensor applegpu_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt
) {
    auto strides = c10::contiguous_strides(size);
    return applegpu_empty_strided(size, strides, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// Eval a pending applegpu tensor before reading its data.
// Needed when an op result (recorded lazily) is about to be copied out.
void eval_applegpu_tensor_if_needed(const at::Tensor& t) {
    if (!t.device().is_privateuseone()) return;
    auto* ctx = static_cast<TensorContext*>(t.storage().data_ptr().get_context());
    if (ctx == nullptr) return;
    // applegpu_ffi_eval is idempotent on already-materialized tensors
    int32_t rc = applegpu_ffi_eval(ctx->tensor_id);
    TORCH_CHECK(rc == 0, "applegpu: eval failed before copy: ",
                applegpu_ffi_last_error() ? applegpu_ffi_last_error() : "unknown");
}

// _copy_from: handles both GPU→CPU and CPU→GPU copies.
// Our Metal buffers are storageModeShared, so they're CPU-accessible —
// we just memcpy between the raw data pointers.
at::Tensor& applegpu_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    // Flush any pending GPU work before reading source data
    applegpu_ffi_synchronize();

    // If src is an applegpu tensor with a pending lazy graph op, evaluate it now
    eval_applegpu_tensor_if_needed(src);

    TORCH_CHECK(self.numel() == src.numel(),
        "applegpu copy_: size mismatch (", self.numel(), " vs ", src.numel(), ")");

    auto nbytes = self.nbytes();
    if (nbytes == 0) return self;

    // Both contiguous: direct memcpy (handles GPU→CPU, CPU→GPU, GPU→GPU)
    if (self.is_contiguous() && src.is_contiguous() && self.dtype() == src.dtype()) {
        std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
        return self;
    }

    // Non-contiguous or dtype mismatch: use CPU view of shared-memory buffer.
    // Our Metal buffers are storageModeShared, so the data pointer is CPU-accessible.
    // We create a CPU tensor that aliases the same memory (via from_blob) to avoid
    // recursive copy_ calls that would crash on non-contiguous PrivateUse1 tensors.
    at::Tensor src_cpu_view;
    if (src.device().is_cpu()) {
        src_cpu_view = src;
    } else {
        // Create CPU tensor aliasing the applegpu buffer (storageModeShared = CPU-accessible)
        src_cpu_view = at::from_blob(
            src.data_ptr(), src.sizes(), src.strides(),
            at::TensorOptions().dtype(src.dtype()).device(at::kCPU));
    }
    auto src_contig = src_cpu_view.contiguous().to(self.dtype());

    if (self.is_contiguous()) {
        std::memcpy(self.data_ptr(), src_contig.data_ptr(), self.nbytes());
    } else if (self.device().is_cpu()) {
        self.copy_(src_contig);
    } else {
        // dst is applegpu + non-contiguous: create CPU alias and copy through it
        auto dst_cpu_view = at::from_blob(
            self.data_ptr(), self.sizes(), self.strides(),
            at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
        dst_cpu_view.copy_(src_contig);
    }
    return self;
}

// _copy_from: copy src (any device) into dst (PrivateUse1).
at::Tensor applegpu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
    applegpu_ffi_synchronize();
    at::Tensor dst_mut = dst;
    dst_mut.copy_(self);
    return dst_mut;
}

// _copy_from_and_resize: resize dst to match self, then copy.
at::Tensor applegpu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    applegpu_ffi_synchronize();
    at::Tensor dst_mut = dst;
    dst_mut.resize_(self.sizes());
    dst_mut.copy_(self);
    return dst_mut;
}

// resize_: explicit resize for PrivateUse1 tensors.
// Registered so _copy_from_and_resize->resize_ doesn't recurse through the fallback.
const at::Tensor& applegpu_resize_(const at::Tensor& self, c10::IntArrayRef size,
                                    std::optional<at::MemoryFormat> fmt) {
    auto* impl = self.unsafeGetTensorImpl();
    auto strides = c10::contiguous_strides(size);
    auto nbytes = at::detail::computeStorageNbytes(size, strides, self.dtype().itemsize());
    if (nbytes > (int64_t)self.storage().nbytes()) {
        auto new_storage = c10::Storage(
            c10::Storage::use_byte_size_t(), nbytes,
            global_allocator.allocate(nbytes), &global_allocator);
        impl->set_storage_and_dtype(std::move(new_storage), self.dtype());
    }
    impl->set_sizes_and_strides(size, strides);
    if (nbytes > 0 && self.storage().data_ptr().get_context()) {
        uint64_t tid = get_tensor_id(self);
        std::vector<uint64_t> dims(size.begin(), size.end());
        applegpu_ffi_register_tensor(tid, dims.data(), dims.size(), scalar_type_to_wire(self.scalar_type()));
    }
    return self;
}

// ── Native Op Wrappers ────────────────────────────────────────────

// Create a PyTorch tensor wrapping an FFI op result.
at::Tensor wrap_ffi_output(void* ptr, uint64_t tid, const std::vector<int64_t>& sizes, at::ScalarType dtype) {
    if (ptr == nullptr) {
        const char* err = applegpu_ffi_last_error();
        TORCH_CHECK(false, "applegpu op failed: ", err ? err : "unknown");
    }

    auto strides = c10::contiguous_strides(c10::IntArrayRef(sizes));
    int64_t nbytes = at::detail::computeStorageNbytes(
        c10::IntArrayRef(sizes), strides, at::elementSize(dtype));

    auto* ctx = new TensorContext{tid};
    auto deleter = [](void* ctx_ptr) {
        auto* tc = static_cast<TensorContext*>(ctx_ptr);
        applegpu_ffi_free(tc->tensor_id);
        delete tc;
    };
    c10::DataPtr dptr{ptr, ctx, deleter, c10::Device(c10::DeviceType::PrivateUse1, 0)};

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(), nbytes,
        std::move(dptr), &global_allocator);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        at::scalarTypeToTypeMeta(dtype));
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        c10::IntArrayRef(sizes), strides);
    return tensor;
}

at::Tensor applegpu_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (alpha.toDouble() != 1.0) {
        // Fall back to CPU for scaled add (SGD optimizer uses alpha=-lr)
        applegpu_ffi_synchronize();
        return at::add(self.cpu(), other.cpu(), alpha).to(self.device());
    }
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_add_out(get_tensor_id(self), get_tensor_id(other), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_mul_tensor(const at::Tensor& self, const at::Tensor& other) {
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_mul_out(get_tensor_id(self), get_tensor_id(other), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_sub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (alpha.toDouble() != 1.0) {
        applegpu_ffi_synchronize();
        return at::sub(self.cpu(), other.cpu(), alpha).to(self.device());
    }
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_sub_out(get_tensor_id(self), get_tensor_id(other), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_mm(const at::Tensor& self, const at::Tensor& mat2) {
    auto self_c = ensure_op_ready(self);
    auto mat2_c = ensure_op_ready(mat2);
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_matmul_out(
        get_tensor_id(self_c), get_tensor_id(mat2_c), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_relu(const at::Tensor& self) {
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_relu_out(get_tensor_id(self), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_neg(const at::Tensor& self) {
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_neg_out(get_tensor_id(self), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

// addmm: bias + mm(mat1, mat2).
// For alpha=1, beta=1 (the common case in nn.Linear): fully GPU-native.
// mat2 is often weight.t() — a non-contiguous transposed view — so we
// make it contiguous first. Non-unit alpha/beta falls back to CPU (rare).
at::Tensor applegpu_addmm(const at::Tensor& self, const at::Tensor& mat1,
                           const at::Tensor& mat2, const at::Scalar& beta,
                           const at::Scalar& alpha) {
    double alpha_val = alpha.toDouble();
    double beta_val = beta.toDouble();

    // Non-unit alpha/beta: fall back to CPU (rare)
    if (alpha_val != 1.0 || beta_val != 1.0) {
        applegpu_ffi_synchronize();
        eval_applegpu_tensor_if_needed(mat1);
        eval_applegpu_tensor_if_needed(mat2);
        eval_applegpu_tensor_if_needed(self);
        return at::addmm(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha)
            .to(c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    // Ensure inputs are contiguous and not views (mat2 is often weight.t())
    auto mat1_c = ensure_op_ready(mat1);
    auto mat2_c = ensure_op_ready(mat2);
    auto self_c = ensure_op_ready(self);

    // mm(mat1, mat2) → [M, N]
    uint64_t mm_out_id = 0;
    void* mm_ptr = applegpu_ffi_matmul_out(
        get_tensor_id(mat1_c), get_tensor_id(mat2_c), &mm_out_id);
    auto mm_result = wrap_ffi_output(
        mm_ptr, mm_out_id, query_output_shape(mm_out_id), mat1.scalar_type());

    // add(mm_result, bias) → [M, N] (bias broadcasts from [N])
    uint64_t add_out_id = 0;
    void* add_ptr = applegpu_ffi_add_out(
        get_tensor_id(mm_result), get_tensor_id(self_c), &add_out_id);
    return wrap_ffi_output(
        add_ptr, add_out_id, query_output_shape(add_out_id), mat1.scalar_type());
}

// threshold_backward: ReLU backward (grad * (input > threshold))
at::Tensor applegpu_threshold_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Scalar& threshold) {
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_threshold_backward_out(
        get_tensor_id(grad_output), get_tensor_id(self),
        threshold.toFloat(), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), grad_output.scalar_type());
}

// t: transpose last two dimensions. Returns a view with swapped strides.
at::Tensor applegpu_t(const at::Tensor& self) {
    TORCH_CHECK(self.dim() == 2, "t expects a 2D tensor, got ", self.dim(), "D");
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::swap(sizes[0], sizes[1]);
    std::swap(strides[0], strides[1]);
    auto result = self.as_strided(sizes, strides);
    return result;
}

// div.Tensor
at::Tensor applegpu_div(const at::Tensor& self, const at::Tensor& other) {
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto other_c = other.is_contiguous() ? other : other.contiguous();
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_div_out(
        get_tensor_id(self_c), get_tensor_id(other_c), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

// add_.Tensor: in-place add. Decompose to out-of-place add + copy back.
at::Tensor& applegpu_add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto self_r = ensure_op_ready(self);
    auto other_r = ensure_op_ready(other);

    // Eval inputs so we can read their data
    eval_applegpu_tensor_if_needed(self_r);
    eval_applegpu_tensor_if_needed(other_r);
    applegpu_ffi_synchronize();

    // Direct CPU manipulation on storageModeShared buffers (no H2D/D2H copy).
    auto self_cpu = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    auto other_cpu = at::from_blob(other_r.data_ptr(), other_r.sizes(), other_r.strides(),
        at::TensorOptions().dtype(other_r.dtype()).device(at::kCPU));
    self_cpu.add_(other_cpu, alpha);
    return self;
}

// mul_.Tensor: in-place multiply
at::Tensor& applegpu_mul_(at::Tensor& self, const at::Tensor& other) {
    eval_applegpu_tensor_if_needed(self);
    eval_applegpu_tensor_if_needed(other);
    applegpu_ffi_synchronize();
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_mul_out(get_tensor_id(self), get_tensor_id(other), &out_id);
    if (ptr == nullptr) {
        TORCH_CHECK(false, "applegpu mul_ failed: ", applegpu_ffi_last_error());
    }
    applegpu_ffi_eval(out_id);
    applegpu_ffi_synchronize();
    std::memcpy(self.data_ptr(), ptr, self.nbytes());
    applegpu_ffi_free(out_id);
    return self;
}

// mul_.Scalar: in-place scalar multiply
at::Tensor& applegpu_mul_scalar_(at::Tensor& self, const at::Scalar& other) {
    eval_applegpu_tensor_if_needed(self);
    applegpu_ffi_synchronize();
    float scale = other.toFloat();
    // Direct CPU manipulation on shared memory buffer
    if (self.scalar_type() == at::ScalarType::Float && self.is_contiguous()) {
        float* data = static_cast<float*>(self.data_ptr());
        int64_t n = self.numel();
        for (int64_t i = 0; i < n; i++) data[i] *= scale;
        return self;
    }
    // Fallback for other dtypes
    auto cpu_view = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    cpu_view.mul_(other);
    return self;
}

// fill_.Scalar: fill tensor with a constant value
at::Tensor& applegpu_fill_(at::Tensor& self, const at::Scalar& value) {
    eval_applegpu_tensor_if_needed(self);
    applegpu_ffi_synchronize();
    // Direct CPU manipulation on shared memory buffer
    auto cpu_view = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    cpu_view.fill_(value);
    return self;
}

// zero_: fill with zeros
at::Tensor& applegpu_zero_(at::Tensor& self) {
    eval_applegpu_tensor_if_needed(self);
    applegpu_ffi_synchronize();
    std::memset(self.data_ptr(), 0, self.nbytes());
    return self;
}

// CPU fallback for unregistered ops
void applegpu_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack
) {
    // Flush streaming batch before reading any tensor data
    applegpu_ffi_synchronize();
    at::native::cpu_fallback(op, stack);
}

// view: reshape with compatible strides. Shares storage.
at::Tensor applegpu_view(const at::Tensor& self, c10::SymIntArrayRef size) {
    // Convert SymIntArrayRef to concrete int64 sizes (our tensors have no symbolic dims)
    std::vector<int64_t> size_int64;
    size_int64.reserve(size.size());
    for (const auto& s : size) {
        size_int64.push_back(s.expect_int());
    }
    auto inferred = at::infer_size_dv(c10::IntArrayRef(size_int64), self.numel());
    auto strides = at::detail::computeStride(self.sizes(), self.strides(), inferred);
    TORCH_CHECK(strides.has_value(), "view size is not compatible with input tensor's size and stride");
    auto result = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(self.storage()),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        self.dtype());
    result.unsafeGetTensorImpl()->set_sizes_and_strides(
        c10::IntArrayRef(inferred), c10::IntArrayRef(*strides));
    result.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
    return result;
}

// as_strided: create a view with explicit sizes, strides, and offset.
at::Tensor applegpu_as_strided(const at::Tensor& self, c10::IntArrayRef size,
                                c10::IntArrayRef stride,
                                std::optional<int64_t> storage_offset) {
    auto offset = storage_offset.value_or(self.storage_offset());
    auto result = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(self.storage()),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        self.dtype());
    result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    result.unsafeGetTensorImpl()->set_storage_offset(offset);
    return result;
}

// mse_loss: native graph decomposition — no eval, fully lazy.
// mse_loss(input, target, reduction) = mean((input - target)²)
at::Tensor applegpu_mse_loss(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
    auto self_r = ensure_op_ready(self);
    auto target_r = ensure_op_ready(target);

    // diff = self - target
    uint64_t diff_id = 0;
    void* diff_ptr = applegpu_ffi_sub_out(get_tensor_id(self_r), get_tensor_id(target_r), &diff_id);
    TORCH_CHECK(diff_ptr, "applegpu mse_loss: sub failed");

    // sq = diff * diff
    uint64_t sq_id = 0;
    void* sq_ptr = applegpu_ffi_mul_out(diff_id, diff_id, &sq_id);
    TORCH_CHECK(sq_ptr, "applegpu mse_loss: mul failed");

    // reduction: 1=mean, 2=sum, 0=none
    if (reduction == 1) {  // Mean
        uint64_t mean_id = 0;
        void* mean_ptr = applegpu_ffi_mean_all_out(sq_id, &mean_id);
        TORCH_CHECK(mean_ptr, "applegpu mse_loss: mean_all failed");
        // Scalar output shape [1]
        return wrap_ffi_output(mean_ptr, mean_id, {1}, self.scalar_type());
    } else if (reduction == 2) {  // Sum — not yet native, fall back
        applegpu_ffi_synchronize();
        eval_applegpu_tensor_if_needed(self_r);
        eval_applegpu_tensor_if_needed(target_r);
        auto s = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
            at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
        auto t = at::from_blob(target.data_ptr(), target.sizes(), target.strides(),
            at::TensorOptions().dtype(target.dtype()).device(at::kCPU));
        return at::mse_loss(s, t, reduction).to(self.device());
    }
    // None: return sq as-is
    return wrap_ffi_output(sq_ptr, sq_id, query_output_shape(sq_id), self.scalar_type());
}

// mse_loss_backward: native graph decomposition.
// grad_input = 2/n * (input - target) * grad_output  (for reduction=mean)
at::Tensor applegpu_mse_loss_backward(const at::Tensor& grad_output,
                                       const at::Tensor& self,
                                       const at::Tensor& target,
                                       int64_t reduction) {
    auto grad_r = ensure_op_ready(grad_output);
    auto self_r = ensure_op_ready(self);
    auto target_r = ensure_op_ready(target);

    // diff = self - target
    uint64_t diff_id = 0;
    applegpu_ffi_sub_out(get_tensor_id(self_r), get_tensor_id(target_r), &diff_id);

    if (reduction == 1) {  // Mean: scale by 2/n
        int64_t n = self.numel();
        float scale = 2.0f / static_cast<float>(n);
        uint64_t scaled_id = 0;
        applegpu_ffi_scalar_mul_out(diff_id, scale, &scaled_id);

        // Multiply by grad_output (broadcasts scalar grad to diff shape)
        uint64_t out_id = 0;
        void* out_ptr = applegpu_ffi_mul_out(scaled_id, get_tensor_id(grad_r), &out_id);
        TORCH_CHECK(out_ptr, "applegpu mse_loss_backward: mul failed");
        return wrap_ffi_output(out_ptr, out_id, query_output_shape(out_id), self.scalar_type());
    }

    // Sum or None: fall back to CPU
    eval_applegpu_tensor_if_needed(grad_r);
    eval_applegpu_tensor_if_needed(self_r);
    eval_applegpu_tensor_if_needed(target_r);
    applegpu_ffi_synchronize();
    auto g = at::from_blob(grad_output.data_ptr(), grad_output.sizes(), grad_output.strides(),
        at::TensorOptions().dtype(grad_output.dtype()).device(at::kCPU));
    auto s = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    auto t = at::from_blob(target.data_ptr(), target.sizes(), target.strides(),
        at::TensorOptions().dtype(target.dtype()).device(at::kCPU));
    return at::mse_loss_backward(g, s, t, reduction).to(self.device());
}

// ── Registration ─────────────────────────────────────────────────

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", applegpu_empty_memory_format);
    m.impl("empty_strided", applegpu_empty_strided);
    m.impl("copy_", applegpu_copy_);
    m.impl("resize_", applegpu_resize_);
    m.impl("_copy_from", applegpu_copy_from);
    m.impl("_copy_from_and_resize", applegpu_copy_from_and_resize);
    m.impl("add.Tensor", applegpu_add);
    m.impl("mul.Tensor", applegpu_mul_tensor);
    m.impl("sub.Tensor", applegpu_sub);
    m.impl("mm", applegpu_mm);
    m.impl("relu", applegpu_relu);
    m.impl("neg", applegpu_neg);
    m.impl("addmm", applegpu_addmm);
    m.impl("threshold_backward", applegpu_threshold_backward);
    m.impl("t", applegpu_t);
    m.impl("div.Tensor", applegpu_div);
    m.impl("add_.Tensor", applegpu_add_);
    m.impl("mul_.Tensor", applegpu_mul_);
    m.impl("mul_.Scalar", applegpu_mul_scalar_);
    m.impl("fill_.Scalar", applegpu_fill_);
    m.impl("zero_", applegpu_zero_);
    m.impl("view", applegpu_view);
    m.impl("as_strided", applegpu_as_strided);
    m.impl("mse_loss", applegpu_mse_loss);
    m.impl("mse_loss_backward", applegpu_mse_loss_backward);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&applegpu_cpu_fallback>());
}

// ── DeviceGuard ──────────────────────────────────────────────────

// Minimal DeviceGuard implementation for single-device, single-stream backend.
struct ApplegpuGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }

    c10::Device exchangeDevice(c10::Device d) const override {
        // Single device — always device 0
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }

    c10::Device getDevice() const override {
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }

    void setDevice(c10::Device d) const override {
        // No-op: single device
    }

    void uncheckedSetDevice(c10::Device d) const noexcept override {
        // No-op: single device
    }

    c10::Stream getStream(c10::Device d) const noexcept override {
        return c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    c10::Stream getDefaultStream(c10::Device d) const override {
        return c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    c10::Stream exchangeStream(c10::Stream s) const noexcept override {
        return c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        return 1;
    }
};

static ApplegpuGuardImpl guard_impl;
C10_REGISTER_GUARD_IMPL(PrivateUse1, ApplegpuGuardImpl);

// ── PrivateUse1 Hooks ────────────────────────────────────────────

// Required for autograd backward pass to create tensors on the correct device.
struct ApplegpuHooksInterface : public at::PrivateUse1HooksInterface {
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
        return true;
    }
};

static auto hooks_registered = []() {
    at::RegisterPrivateUse1HooksInterface(new ApplegpuHooksInterface());
    return true;
}();

// ── Module init ──────────────────────────────────────────────────

static auto init = []() {
    bool ok = applegpu_ffi_init();
    TORCH_CHECK(ok, "applegpu FFI init failed: ",
                applegpu_ffi_last_error() ? applegpu_ffi_last_error() : "unknown");
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &global_allocator);
    return true;
}();
