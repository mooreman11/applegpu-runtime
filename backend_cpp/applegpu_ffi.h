#ifndef APPLEGPU_FFI_H
#define APPLEGPU_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Lifecycle */
bool applegpu_ffi_init(void);
const char* applegpu_ffi_last_error(void);

/* Allocation */
uint8_t* applegpu_ffi_alloc(uint64_t size_bytes, int8_t dtype_i8, uint64_t* out_tensor_id);
void applegpu_ffi_free(uint64_t tensor_id);

/* Tensor metadata */
int32_t applegpu_ffi_register_tensor(uint64_t tensor_id, const uint64_t* dims, uint32_t ndim, int8_t dtype_i8);
int32_t applegpu_ffi_shape(uint64_t tensor_id, uint64_t* out_dims, uint32_t* out_ndim);
int8_t applegpu_ffi_dtype(uint64_t tensor_id);

/* Ops (return output tensor_id, 0 on failure) */
uint64_t applegpu_ffi_add(uint64_t a_id, uint64_t b_id);
uint64_t applegpu_ffi_matmul(uint64_t a_id, uint64_t b_id);
uint64_t applegpu_ffi_relu(uint64_t input_id);
uint64_t applegpu_ffi_mul(uint64_t a_id, uint64_t b_id);
uint64_t applegpu_ffi_sub(uint64_t a_id, uint64_t b_id);
uint64_t applegpu_ffi_neg(uint64_t input_id);
int32_t applegpu_ffi_copy(uint64_t src_id, uint64_t dst_id);

/* Ops with output allocation (return data_ptr, write tensor_id to *out_id) */
uint8_t* applegpu_ffi_add_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_mul_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_sub_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_matmul_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_relu_out(uint64_t input_id, uint64_t* out_id);
uint8_t* applegpu_ffi_neg_out(uint64_t input_id, uint64_t* out_id);
uint8_t* applegpu_ffi_div_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_threshold_backward_out(uint64_t grad_id, uint64_t input_id, float threshold, uint64_t* out_id);
uint8_t* applegpu_ffi_scalar_mul_out(uint64_t input_id, float scale, uint64_t* out_id);
uint8_t* applegpu_ffi_mean_all_out(uint64_t input_id, uint64_t* out_id);

/* Eval / sync */
int32_t applegpu_ffi_eval(uint64_t tensor_id);
void applegpu_ffi_synchronize(void);

/* Readback */
int32_t applegpu_ffi_read_f32(uint64_t tensor_id, float* out_ptr, uint64_t max_elements);

/* ── Eager dispatch (bypasses graph engine) ── */
bool applegpu_eager_init(void);
const char* applegpu_eager_last_error(void);

uint8_t* applegpu_eager_alloc(const uint64_t* dims, uint32_t ndim, int8_t dtype, uint64_t* out_id);
void applegpu_eager_free(uint64_t id);
int32_t applegpu_eager_register_shape(uint64_t id, const uint64_t* dims, uint32_t ndim);
int32_t applegpu_eager_shape(uint64_t id, uint64_t* out_dims, uint32_t* out_ndim);
int8_t applegpu_eager_dtype(uint64_t id);

uint8_t* applegpu_eager_add(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_eager_sub(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_eager_mul(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_eager_div(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_eager_relu(uint64_t input_id, uint64_t* out_id);
uint8_t* applegpu_eager_neg(uint64_t input_id, uint64_t* out_id);
uint8_t* applegpu_eager_matmul(uint64_t a_id, uint64_t b_id, uint64_t* out_id);

uint8_t* applegpu_eager_threshold_backward(uint64_t grad_id, uint64_t input_id, float threshold, uint64_t* out_id);
uint8_t* applegpu_eager_scalar_mul(uint64_t input_id, float scale, uint64_t* out_id);
uint8_t* applegpu_eager_mean_all(uint64_t input_id, uint64_t* out_id);

uint8_t* applegpu_eager_sum_dim(uint64_t input_id, int64_t dim, bool keepdim, uint64_t* out_id);

uint8_t* applegpu_eager_create_view(uint64_t base_id, const uint64_t* shape, const uint64_t* strides, uint32_t ndim, uint64_t offset, uint64_t* out_id);

int32_t applegpu_eager_add_inplace(uint64_t self_id, uint64_t other_id);
int32_t applegpu_eager_add_scaled_inplace(uint64_t self_id, uint64_t other_id, float alpha);

void applegpu_eager_flush_and_wait(void);
void applegpu_eager_synchronize(void);

/* Reverse lookup: find tensor_id by buffer data pointer. Returns 0 if not found. */
uint64_t applegpu_eager_find_by_data_ptr(const uint8_t* ptr);

#ifdef __cplusplus
}
#endif

#endif /* APPLEGPU_FFI_H */
