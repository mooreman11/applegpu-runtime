#ifndef APPLE_GPU_BRIDGE_H
#define APPLE_GPU_BRIDGE_H

#include <stdint.h>

// Opaque handle to a GPU device
typedef struct GPUDeviceHandle GPUDeviceHandle;

// Lifecycle
GPUDeviceHandle* gpu_bridge_create_device(void);
void gpu_bridge_destroy_device(GPUDeviceHandle* device);

// Query
const char* gpu_bridge_device_name(const GPUDeviceHandle* device);

#endif
