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

/* Eval / sync */
int32_t applegpu_ffi_eval(uint64_t tensor_id);
void applegpu_ffi_synchronize(void);

/* Readback */
int32_t applegpu_ffi_read_f32(uint64_t tensor_id, float* out_ptr, uint64_t max_elements);

#ifdef __cplusplus
}
#endif

#endif /* APPLEGPU_FFI_H */
