// PrivateUse1 C++ backend for applegpu_runtime.
//
// Registers a custom allocator and minimum ops at the PrivateUse1 dispatch key.
// All GPU work is delegated to the Rust eager runtime via extern "C" FFI.
// The eager runtime encodes Metal commands into a streaming command buffer,
// committing only at sync points (flush_and_wait / synchronize).

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

// Ephemeral eager view IDs created by resolve_tensor_id.
// These are freed after each op via ScopeGuard.
thread_local std::vector<uint64_t> ephemeral_views;

// Resolve the correct eager tensor_id for a PyTorch tensor.
// If the tensor is a view (shape/strides/offset differ from the eager runtime's
// registration), create an eager view and track it for cleanup.
uint64_t resolve_tensor_id(const at::Tensor& t) {
    uint64_t base_id = get_tensor_id(t);

    // Query the eager runtime's shape for this ID
    uint64_t eager_dims[8];
    uint32_t eager_ndim = 0;
    int32_t rc = applegpu_eager_shape(base_id, eager_dims, &eager_ndim);
    if (rc != 0) return base_id;  // Not found — let the op handle the error

    // Compare shapes
    bool matches = (eager_ndim == (uint32_t)t.dim());
    if (matches) {
        for (uint32_t i = 0; i < eager_ndim; i++) {
            if (eager_dims[i] != (uint64_t)t.size(i)) {
                matches = false;
                break;
            }
        }
    }

    // Also check storage offset
    if (matches && t.storage_offset() == 0) {
        return base_id;  // Shape matches, no offset — use the base ID
    }

    // Create an eager view with the correct shape/strides/offset.
    // If PyTorch considers the tensor contiguous, use true contiguous strides
    // (PyTorch allows any stride for size-1 dims, but our runtime doesn't).
    std::vector<uint64_t> shape(t.dim());
    std::vector<uint64_t> strides(t.dim());
    if (t.is_contiguous()) {
        // Compute true contiguous strides (row-major)
        uint64_t stride = 1;
        for (int64_t i = t.dim() - 1; i >= 0; i--) {
            shape[i] = static_cast<uint64_t>(t.size(i));
            strides[i] = stride;
            stride *= shape[i];
        }
    } else {
        for (int64_t i = 0; i < t.dim(); i++) {
            shape[i] = static_cast<uint64_t>(t.size(i));
            strides[i] = static_cast<uint64_t>(t.stride(i));
        }
    }
    uint64_t view_id = 0;
    void* view_ptr = applegpu_eager_create_view(
        base_id, shape.data(), strides.data(),
        static_cast<uint32_t>(t.dim()),
        static_cast<uint64_t>(t.storage_offset()),
        &view_id);
    TORCH_CHECK(view_ptr != nullptr, "applegpu: failed to create eager view: ",
                applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");
    ephemeral_views.push_back(view_id);
    return view_id;
}

// RAII guard to free ephemeral views after an op completes.
struct EphemeralViewGuard {
    ~EphemeralViewGuard() {
        for (uint64_t id : ephemeral_views) {
            applegpu_eager_free(id);
        }
        ephemeral_views.clear();
    }
};

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

// Query output shape from the eager runtime.
std::vector<int64_t> query_output_shape(uint64_t tid) {
    uint64_t dims[8];
    uint32_t ndim = 0;
    int32_t rc = applegpu_eager_shape(tid, dims, &ndim);
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
        // Eager alloc requires dims+ndim+dtype. For raw byte allocation (e.g. resize_),
        // we create a 1-D tensor of the right byte count using uint8 dtype.
        uint64_t dim0 = nbytes;
        uint64_t tensor_id = 0;
        void* ptr = applegpu_eager_alloc(&dim0, 1, 7 /* u8 */, &tensor_id);
        TORCH_CHECK(ptr != nullptr, "applegpu alloc failed: ",
                     applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");

        auto* ctx = new TensorContext{tensor_id};
        auto deleter = [](void* ctx_ptr) {
            auto* tc = static_cast<TensorContext*>(ctx_ptr);
            applegpu_eager_free(tc->tensor_id);
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

    // Allocate via eager runtime with proper dims and dtype
    std::vector<uint64_t> dims(size.begin(), size.end());
    int8_t dtype_wire = scalar_type_to_wire(dtype);
    uint64_t tensor_id = 0;
    void* ptr = nullptr;

    if (size.empty() || std::any_of(size.begin(), size.end(), [](int64_t s) { return s == 0; })) {
        // Zero-element tensor: use raw allocator path
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
        return tensor;
    }

    ptr = applegpu_eager_alloc(dims.data(), dims.size(), dtype_wire, &tensor_id);
    TORCH_CHECK(ptr != nullptr, "applegpu empty_strided alloc failed: ",
                applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");

    int64_t nbytes = at::detail::computeStorageNbytes(size, stride, at::elementSize(dtype));

    auto* ctx = new TensorContext{tensor_id};
    auto deleter = [](void* ctx_ptr) {
        auto* tc = static_cast<TensorContext*>(ctx_ptr);
        applegpu_eager_free(tc->tensor_id);
        delete tc;
    };
    c10::DataPtr dptr{ptr, ctx, deleter, c10::Device(c10::DeviceType::PrivateUse1, 0)};

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        nbytes,
        std::move(dptr),
        &global_allocator
    );

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        at::scalarTypeToTypeMeta(dtype)
    );
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

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

// Flush pending GPU work before reading tensor data.
void eval_applegpu_tensor_if_needed(const at::Tensor& t) {
    if (!t.device().is_privateuseone()) return;
    // Eager runtime: just flush and wait for all pending work
    applegpu_eager_flush_and_wait();
}

// _copy_from: handles both GPU→CPU and CPU→GPU copies.
// Our Metal buffers are storageModeShared, so they're CPU-accessible —
// we just memcpy between the raw data pointers.
at::Tensor& applegpu_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    TORCH_CHECK(self.numel() == src.numel(),
        "applegpu copy_: size mismatch (", self.numel(), " vs ", src.numel(), ")");

    auto nbytes = self.nbytes();
    if (nbytes == 0) return self;

    // GPU→GPU copy: use eager dispatch (no flush for strided copies)
    if (self.device().is_privateuseone() && src.device().is_privateuseone()
        && self.dtype() == src.dtype() && self.is_contiguous()) {
        // Encode a strided copy via scalar_mul(src, 1.0).
        // scalar_mul reads input with strides (via broadcast binary mul),
        // writes output contiguously. No flush needed — stays in streaming CB.
        EphemeralViewGuard evg;
        uint64_t src_id = resolve_tensor_id(src);
        uint64_t copy_id = 0;
        void* copy_ptr = applegpu_eager_scalar_mul(src_id, 1.0f, &copy_id);
        if (copy_ptr) {
            // We have a contiguous copy in copy_id's buffer.
            // Now we need to get the data into self's buffer.
            // Encode an in-place "copy" by using add(self*0, copy) — but that's complex.
            // Simpler: just swap self's storage to point to copy_id's buffer.
            auto* impl = self.unsafeGetTensorImpl();
            auto new_dptr = c10::DataPtr{copy_ptr, new TensorContext{copy_id},
                [](void* ctx) { auto* tc = static_cast<TensorContext*>(ctx);
                    applegpu_eager_free(tc->tensor_id); delete tc; },
                c10::Device(c10::DeviceType::PrivateUse1, 0)};
            auto new_storage = c10::Storage(c10::Storage::use_byte_size_t(),
                nbytes, std::move(new_dptr), &global_allocator);
            impl->set_storage_and_dtype(std::move(new_storage), self.dtype());
            return self;
        }
        // Fall through to flush-based path on error
    }

    // Cross-device copy or fallback: flush + memcpy
    applegpu_eager_flush_and_wait();

    if (self.is_contiguous() && src.is_contiguous() && self.dtype() == src.dtype()) {
        std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
        if (self.device().is_privateuseone() && self.storage().data_ptr().get_context()) {
            uint64_t tid = get_tensor_id(self);
            std::vector<uint64_t> dims(self.sizes().begin(), self.sizes().end());
            applegpu_eager_register_shape(tid, dims.data(), dims.size());
        }
        return self;
    }

    // Non-contiguous cross-device: CPU view of shared-memory buffer
    at::Tensor src_cpu_view;
    if (src.device().is_cpu()) {
        src_cpu_view = src;
    } else {
        src_cpu_view = at::from_blob(
            src.data_ptr(), src.sizes(), src.strides(),
            at::TensorOptions().dtype(src.dtype()).device(at::kCPU));
    }
    auto src_contig = src_cpu_view.contiguous().to(self.dtype());

    if (self.is_contiguous()) {
        std::memcpy(self.data_ptr(), src_contig.data_ptr(), self.nbytes());
        if (self.device().is_privateuseone() && self.storage().data_ptr().get_context()) {
            uint64_t tid = get_tensor_id(self);
            std::vector<uint64_t> dims(self.sizes().begin(), self.sizes().end());
            applegpu_eager_register_shape(tid, dims.data(), dims.size());
        }
    } else if (self.device().is_cpu()) {
        self.copy_(src_contig);
    } else {
        auto dst_cpu_view = at::from_blob(
            self.data_ptr(), self.sizes(), self.strides(),
            at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
        dst_cpu_view.copy_(src_contig);
    }
    return self;
}

// _copy_from: copy src (any device) into dst (PrivateUse1).
at::Tensor applegpu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
    applegpu_eager_synchronize();
    at::Tensor dst_mut = dst;
    dst_mut.copy_(self);
    return dst_mut;
}

// _copy_from_and_resize: resize dst to match self, then copy.
at::Tensor applegpu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    applegpu_eager_synchronize();
    at::Tensor dst_mut = dst;
    dst_mut.resize_(self.sizes());
    dst_mut.copy_(self);
    return dst_mut;
}

// resize_: explicit resize for PrivateUse1 tensors.
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
        applegpu_eager_register_shape(tid, dims.data(), dims.size());
    }
    return self;
}

// ── Native Op Wrappers ────────────────────────────────────────────

// Create a PyTorch tensor wrapping an eager op result.
at::Tensor wrap_eager_output(void* ptr, uint64_t tid, const std::vector<int64_t>& sizes, at::ScalarType dtype) {
    if (ptr == nullptr) {
        const char* err = applegpu_eager_last_error();
        TORCH_CHECK(false, "applegpu op failed: ", err ? err : "unknown");
    }

    auto strides = c10::contiguous_strides(c10::IntArrayRef(sizes));
    int64_t nbytes = at::detail::computeStorageNbytes(
        c10::IntArrayRef(sizes), strides, at::elementSize(dtype));

    auto* ctx = new TensorContext{tid};
    auto deleter = [](void* ctx_ptr) {
        auto* tc = static_cast<TensorContext*>(ctx_ptr);
        applegpu_eager_free(tc->tensor_id);
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
        applegpu_eager_synchronize();
        return at::add(self.cpu(), other.cpu(), alpha).to(self.device());
    }
    EphemeralViewGuard evg;
    uint64_t out_id = 0;
    void* ptr = applegpu_eager_add(resolve_tensor_id(self), resolve_tensor_id(other), &out_id);
    return wrap_eager_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_mul_tensor(const at::Tensor& self, const at::Tensor& other) {
    EphemeralViewGuard evg;
    uint64_t out_id = 0;
    void* ptr = applegpu_eager_mul(resolve_tensor_id(self), resolve_tensor_id(other), &out_id);
    return wrap_eager_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_sub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (alpha.toDouble() != 1.0) {
        applegpu_eager_synchronize();
        return at::sub(self.cpu(), other.cpu(), alpha).to(self.device());
    }
    EphemeralViewGuard evg;
    uint64_t out_id = 0;
    void* ptr = applegpu_eager_sub(resolve_tensor_id(self), resolve_tensor_id(other), &out_id);
    return wrap_eager_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_mm(const at::Tensor& self, const at::Tensor& mat2) {
    // Eager matmul requires contiguous inputs. Use PyTorch's .contiguous()
    // which goes through our allocator + copy_ to create a fresh contiguous tensor.
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto mat2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    EphemeralViewGuard evg;
    uint64_t out_id = 0;
    void* ptr = applegpu_eager_matmul(
        resolve_tensor_id(self_c), resolve_tensor_id(mat2_c), &out_id);
    return wrap_eager_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_relu(const at::Tensor& self) {
    EphemeralViewGuard evg;
    uint64_t out_id = 0;
    void* ptr = applegpu_eager_relu(resolve_tensor_id(self), &out_id);
    return wrap_eager_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_neg(const at::Tensor& self) {
    EphemeralViewGuard evg;
    uint64_t out_id = 0;
    void* ptr = applegpu_eager_neg(resolve_tensor_id(self), &out_id);
    return wrap_eager_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

// addmm: bias + mm(mat1, mat2).
// Decomposed into eager mm + eager add.
at::Tensor applegpu_addmm(const at::Tensor& self, const at::Tensor& mat1,
                           const at::Tensor& mat2, const at::Scalar& beta,
                           const at::Scalar& alpha) {
    double alpha_val = alpha.toDouble();
    double beta_val = beta.toDouble();

    // Non-unit alpha/beta: fall back to CPU (rare)
    if (alpha_val != 1.0 || beta_val != 1.0) {
        applegpu_eager_synchronize();
        return at::addmm(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha)
            .to(c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    // Ensure inputs are contiguous (mat2 is often weight.t())
    auto mat1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    auto mat2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();

    EphemeralViewGuard evg;

    // mm(mat1, mat2) → [M, N]
    uint64_t mm_out_id = 0;
    void* mm_ptr = applegpu_eager_matmul(
        resolve_tensor_id(mat1_c), resolve_tensor_id(mat2_c), &mm_out_id);
    auto mm_result = wrap_eager_output(
        mm_ptr, mm_out_id, query_output_shape(mm_out_id), mat1.scalar_type());

    // add(mm_result, bias) → [M, N] (bias broadcasts from [N])
    uint64_t add_out_id = 0;
    void* add_ptr = applegpu_eager_add(
        get_tensor_id(mm_result), resolve_tensor_id(self), &add_out_id);
    return wrap_eager_output(
        add_ptr, add_out_id, query_output_shape(add_out_id), mat1.scalar_type());
}

// threshold_backward: ReLU backward (grad * (input > threshold))
at::Tensor applegpu_threshold_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Scalar& threshold) {
    EphemeralViewGuard evg;
    uint64_t out_id = 0;
    void* ptr = applegpu_eager_threshold_backward(
        resolve_tensor_id(grad_output), resolve_tensor_id(self),
        threshold.toFloat(), &out_id);
    return wrap_eager_output(ptr, out_id, query_output_shape(out_id), grad_output.scalar_type());
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
    EphemeralViewGuard evg;
    uint64_t out_id = 0;
    void* ptr = applegpu_eager_div(
        resolve_tensor_id(self), resolve_tensor_id(other), &out_id);
    return wrap_eager_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

// add_.Tensor: in-place add using eager add_scaled_inplace.
at::Tensor& applegpu_add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    EphemeralViewGuard evg;
    float alpha_val = alpha.toFloat();
    int32_t rc = applegpu_eager_add_scaled_inplace(
        resolve_tensor_id(self), resolve_tensor_id(other), alpha_val);
    TORCH_CHECK(rc == 0, "applegpu add_ failed: ",
                applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");
    return self;
}

// mul_.Tensor: in-place multiply via flush + CPU view
at::Tensor& applegpu_mul_(at::Tensor& self, const at::Tensor& other) {
    applegpu_eager_flush_and_wait();
    auto self_cpu = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    auto other_cpu = at::from_blob(other.data_ptr(), other.sizes(), other.strides(),
        at::TensorOptions().dtype(other.dtype()).device(at::kCPU));
    self_cpu.mul_(other_cpu);
    return self;
}

// mul_.Scalar: in-place scalar multiply via flush + CPU view
at::Tensor& applegpu_mul_scalar_(at::Tensor& self, const at::Scalar& other) {
    applegpu_eager_flush_and_wait();
    float scale = other.toFloat();
    if (self.scalar_type() == at::ScalarType::Float && self.is_contiguous()) {
        float* data = static_cast<float*>(self.data_ptr());
        int64_t n = self.numel();
        for (int64_t i = 0; i < n; i++) data[i] *= scale;
        return self;
    }
    auto cpu_view = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    cpu_view.mul_(other);
    return self;
}

// fill_.Scalar: fill tensor with a constant value
at::Tensor& applegpu_fill_(at::Tensor& self, const at::Scalar& value) {
    applegpu_eager_flush_and_wait();
    auto cpu_view = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    cpu_view.fill_(value);
    return self;
}

// zero_: fill with zeros
at::Tensor& applegpu_zero_(at::Tensor& self) {
    applegpu_eager_flush_and_wait();
    std::memset(self.data_ptr(), 0, self.nbytes());
    return self;
}

// CPU fallback for unregistered ops
void applegpu_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack
) {
    static bool log_fallback = std::getenv("APPLEGPU_LOG_FALLBACK") != nullptr;
    if (log_fallback) {
        std::cerr << "[fallback] " << op.schema().name() << std::endl;
    }
    applegpu_eager_synchronize();
    at::native::cpu_fallback(op, stack);
}

// view: reshape with compatible strides. Shares storage.
at::Tensor applegpu_view(const at::Tensor& self, c10::SymIntArrayRef size) {
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

// mse_loss: decomposed via eager sub/mul/mean_all.
at::Tensor applegpu_mse_loss(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
    EphemeralViewGuard evg;
    // diff = self - target
    uint64_t diff_id = 0;
    void* diff_ptr = applegpu_eager_sub(resolve_tensor_id(self), resolve_tensor_id(target), &diff_id);
    TORCH_CHECK(diff_ptr, "applegpu mse_loss: sub failed: ",
                applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");

    // sq = diff * diff
    uint64_t sq_id = 0;
    void* sq_ptr = applegpu_eager_mul(diff_id, diff_id, &sq_id);
    TORCH_CHECK(sq_ptr, "applegpu mse_loss: mul failed: ",
                applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");

    // reduction: 1=mean, 2=sum, 0=none
    if (reduction == 1) {  // Mean
        uint64_t mean_id = 0;
        void* mean_ptr = applegpu_eager_mean_all(sq_id, &mean_id);
        TORCH_CHECK(mean_ptr, "applegpu mse_loss: mean_all failed: ",
                    applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");
        return wrap_eager_output(mean_ptr, mean_id, {1}, self.scalar_type());
    } else if (reduction == 2) {  // Sum — not yet native, fall back
        applegpu_eager_synchronize();
        auto s = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
            at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
        auto t = at::from_blob(target.data_ptr(), target.sizes(), target.strides(),
            at::TensorOptions().dtype(target.dtype()).device(at::kCPU));
        return at::mse_loss(s, t, reduction).to(self.device());
    }
    // None: return sq as-is
    return wrap_eager_output(sq_ptr, sq_id, query_output_shape(sq_id), self.scalar_type());
}

// mse_loss_backward: decomposed via eager sub/scalar_mul/mul.
at::Tensor applegpu_mse_loss_backward(const at::Tensor& grad_output,
                                       const at::Tensor& self,
                                       const at::Tensor& target,
                                       int64_t reduction) {
    EphemeralViewGuard evg;
    // diff = self - target
    uint64_t diff_id = 0;
    applegpu_eager_sub(resolve_tensor_id(self), resolve_tensor_id(target), &diff_id);

    if (reduction == 1) {  // Mean: scale by 2/n
        int64_t n = self.numel();
        float scale = 2.0f / static_cast<float>(n);
        uint64_t scaled_id = 0;
        applegpu_eager_scalar_mul(diff_id, scale, &scaled_id);

        // Multiply by grad_output (broadcasts scalar grad to diff shape)
        uint64_t out_id = 0;
        void* out_ptr = applegpu_eager_mul(scaled_id, resolve_tensor_id(grad_output), &out_id);
        TORCH_CHECK(out_ptr, "applegpu mse_loss_backward: mul failed: ",
                    applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");
        return wrap_eager_output(out_ptr, out_id, query_output_shape(out_id), self.scalar_type());
    }

    // Sum or None: fall back to CPU
    applegpu_eager_synchronize();
    auto g = at::from_blob(grad_output.data_ptr(), grad_output.sizes(), grad_output.strides(),
        at::TensorOptions().dtype(grad_output.dtype()).device(at::kCPU));
    auto s = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    auto t = at::from_blob(target.data_ptr(), target.sizes(), target.strides(),
        at::TensorOptions().dtype(target.dtype()).device(at::kCPU));
    return at::mse_loss_backward(g, s, t, reduction).to(self.device());
}

// sum.dim_IntList: reduce along specified dimensions via eager GPU dispatch.
at::Tensor applegpu_sum_dim(const at::Tensor& self,
                             at::OptionalIntArrayRef dim,
                             bool keepdim,
                             std::optional<at::ScalarType> dtype) {
    auto self_id = resolve_tensor_id(self);

    // Single-dim reduction via eager FFI
    if (dim.has_value() && dim.value().size() == 1) {
        int64_t d = dim.value()[0];
        uint64_t out_id = 0;
        void* ptr = applegpu_eager_sum_dim(self_id, d, keepdim, &out_id);
        if (ptr) {
            return wrap_eager_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
        }
        // Fall through to CPU fallback on error
    }

    // Multi-dim or no-dim: CPU fallback via shared memory
    applegpu_eager_flush_and_wait();
    auto cpu_view = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
        at::TensorOptions().dtype(self.dtype()).device(at::kCPU));
    auto result = dim.has_value()
        ? cpu_view.sum(dim.value(), keepdim, dtype)
        : cpu_view.sum(dtype);
    return result.to(self.device());
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
    m.impl("sum.dim_IntList", applegpu_sum_dim);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&applegpu_cpu_fallback>());
}

// ── DeviceGuard ──────────────────────────────────────────────────

struct ApplegpuGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }

    c10::Device exchangeDevice(c10::Device d) const override {
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }

    c10::Device getDevice() const override {
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }

    void setDevice(c10::Device d) const override {}
    void uncheckedSetDevice(c10::Device d) const noexcept override {}

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
    // Initialize graph-based FFI (needed for PyO3 path / future torch.compile)
    bool ok = applegpu_ffi_init();
    TORCH_CHECK(ok, "applegpu FFI init failed: ",
                applegpu_ffi_last_error() ? applegpu_ffi_last_error() : "unknown");
    // Initialize eager dispatch runtime (streaming command buffer path)
    bool eager_ok = applegpu_eager_init();
    TORCH_CHECK(eager_ok, "applegpu eager init failed: ",
                applegpu_eager_last_error() ? applegpu_eager_last_error() : "unknown");
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &global_allocator);
    return true;
}();
