// PrivateUse1 C++ backend for applegpu_runtime.
//
// Registers a custom allocator and minimum ops at the PrivateUse1 dispatch key.
// All GPU work is delegated to the Rust graph engine via extern "C" FFI.

#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
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

// _copy_from: handles both GPU→CPU and CPU→GPU copies.
// Our Metal buffers are storageModeShared, so they're CPU-accessible —
// we just memcpy between the raw data pointers.
at::Tensor& applegpu_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    // Flush any pending GPU work before reading source data
    applegpu_ffi_synchronize();

    TORCH_CHECK(self.numel() == src.numel(),
        "applegpu copy_: size mismatch (", self.numel(), " vs ", src.numel(), ")");

    auto nbytes = self.nbytes();
    if (nbytes == 0) return self;

    // Both contiguous: direct memcpy (handles GPU→CPU, CPU→GPU, GPU→GPU)
    if (self.is_contiguous() && src.is_contiguous() && self.dtype() == src.dtype()) {
        std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
        return self;
    }

    // Non-contiguous or dtype cast: fall back to element-wise copy via CPU
    auto src_cpu = src.device().is_cpu() ? src : src.to(at::kCPU);
    auto self_cpu = self.device().is_cpu() ? self : at::empty_like(self, at::kCPU);
    self_cpu.copy_(src_cpu);
    if (!self.device().is_cpu()) {
        std::memcpy(self.data_ptr(), self_cpu.data_ptr(), self.nbytes());
    }
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

// ── Registration ─────────────────────────────────────────────────

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", applegpu_empty_memory_format);
    m.impl("empty_strided", applegpu_empty_strided);
    m.impl("copy_", applegpu_copy_);
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

// ── Module init ──────────────────────────────────────────────────

static auto init = []() {
    bool ok = applegpu_ffi_init();
    TORCH_CHECK(ok, "applegpu FFI init failed: ",
                applegpu_ffi_last_error() ? applegpu_ffi_last_error() : "unknown");
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &global_allocator);
    return true;
}();
