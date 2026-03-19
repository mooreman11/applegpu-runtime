// PrivateUse1 C++ backend for applegpu_runtime.
//
// Registers a custom allocator and minimum ops at the PrivateUse1 dispatch key.
// All GPU work is delegated to the Rust graph engine via extern "C" FFI.

#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
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
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_matmul_out(get_tensor_id(self), get_tensor_id(mat2), &out_id);
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

// addmm: bias + mm(mat1, mat2). Decomposed to CPU because mat2 is typically
// transposed (non-contiguous view) which our Rust graph engine doesn't handle yet.
at::Tensor applegpu_addmm(const at::Tensor& self, const at::Tensor& mat1,
                           const at::Tensor& mat2, const at::Scalar& beta,
                           const at::Scalar& alpha) {
    applegpu_ffi_synchronize();
    return at::addmm(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha)
        .to(c10::Device(c10::DeviceType::PrivateUse1, 0));
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
