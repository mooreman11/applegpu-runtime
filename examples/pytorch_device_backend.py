"""PyTorch device backend demo — run standard nn.Module on Metal GPU.

Shows how to use gpu.enable_torch_backend() + gpu.to_applegpu() to
run any PyTorch model on Apple Silicon Metal without modifying model code.

Usage:
    python examples/pytorch_device_backend.py
"""

import torch
import torch.nn as nn
import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import ApplegpuTensor


def main():
    gpu.init_backend()
    gpu.enable_torch_backend()

    print("=== PyTorch Device Backend Demo ===\n")

    # 1. Simple MLP
    print("1. MLP on Metal GPU:")
    mlp = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    mlp = gpu.to_applegpu(mlp)

    x = ApplegpuTensor.from_torch(torch.randn(32, 64))
    with torch.no_grad():
        output = mlp(x)
    print(f"   Input: {x.shape} → Output: {output.shape}")
    print(f"   All finite: {torch.all(torch.isfinite(output.to_torch_cpu())).item()}")

    # 2. Operations with broadcasting
    print("\n2. Broadcasting on Metal GPU:")
    a = ApplegpuTensor.from_torch(torch.ones(4, 3))
    b = ApplegpuTensor.from_torch(torch.tensor([10.0, 20.0, 30.0]))
    c = a + b  # [4,3] + [3] broadcasts
    print(f"   [4,3] + [3] = {c.to_torch_cpu()[0].tolist()}")

    # 3. Matmul chain
    print("\n3. Matmul → ReLU → Softmax chain:")
    x = ApplegpuTensor.from_torch(torch.randn(8, 16))
    w = ApplegpuTensor.from_torch(torch.randn(16, 32))
    y = torch.softmax(torch.relu(x @ w), dim=-1)
    print(f"   Shape: {y.shape}, row sums to 1: {y.to_torch_cpu()[0].sum().item():.4f}")

    # 4. Multi-dtype
    print("\n4. Multi-dtype support:")
    for dtype in [torch.float32, torch.float16, torch.int32, torch.bool]:
        if dtype == torch.bool:
            t = ApplegpuTensor.from_torch(torch.tensor([True, False, True]))
        elif dtype in (torch.int32,):
            t = ApplegpuTensor.from_torch(torch.tensor([1, 2, 3], dtype=dtype))
        else:
            t = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
        print(f"   {str(dtype):15s} → roundtrip OK: {t.to_torch_cpu().tolist()}")

    print("\n=== All demos passed! ===")


if __name__ == "__main__":
    main()
