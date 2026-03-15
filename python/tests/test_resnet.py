"""Tests for ResNet-18 on applegpu device backend."""
import pytest
import numpy as np

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()
    gpu.enable_torch_backend()


@pytest.fixture(scope="module")
def resnet_model():
    """Load ResNet-18 (random weights, no download needed)."""
    model = torchvision.models.resnet18(weights=None)
    model.eval()
    return model


def test_resnet18_forward_shape(resnet_model):
    """ResNet-18 forward pass produces correct output shape."""
    from applegpu_runtime.torch_backend import ApplegpuTensor

    model = gpu.to_applegpu(resnet_model)
    x = ApplegpuTensor.from_torch(torch.randn(1, 3, 224, 224))

    with torch.no_grad():
        output = model(x)

    result = output.to_torch_cpu()
    assert result.shape == (1, 1000)  # 1000 ImageNet classes
    assert torch.all(torch.isfinite(result))


def test_resnet18_matches_cpu(resnet_model):
    """GPU output matches CPU output within tolerance."""
    from applegpu_runtime.torch_backend import ApplegpuTensor

    x = torch.randn(1, 3, 224, 224)

    # CPU reference
    with torch.no_grad():
        cpu_output = resnet_model(x)

    # GPU
    model_gpu = gpu.to_applegpu(resnet_model)
    x_gpu = ApplegpuTensor.from_torch(x)
    with torch.no_grad():
        gpu_output = model_gpu(x_gpu)

    result = gpu_output.to_torch_cpu()
    # Allow reasonable tolerance for f32 GPU compute
    assert torch.allclose(result, cpu_output, atol=1e-2, rtol=1e-2)


def test_resnet18_batch(resnet_model):
    """ResNet-18 with batch size > 1."""
    from applegpu_runtime.torch_backend import ApplegpuTensor

    model = gpu.to_applegpu(resnet_model)
    x = ApplegpuTensor.from_torch(torch.randn(4, 3, 224, 224))

    with torch.no_grad():
        output = model(x)

    result = output.to_torch_cpu()
    assert result.shape == (4, 1000)
    assert torch.all(torch.isfinite(result))
