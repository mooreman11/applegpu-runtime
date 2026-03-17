"""Tests for linalg_vector_norm, unique, and gradient clipping."""
import math
import applegpu_runtime as gpu


def test_vector_norm_l2():
    """L2 norm (default)."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([3.0, 4.0]))
    norm = torch.linalg.vector_norm(x)
    assert abs(norm.to_torch_cpu().item() - 5.0) < 1e-5


def test_vector_norm_l1():
    """L1 norm."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([3.0, -4.0]))
    norm = torch.linalg.vector_norm(x, ord=1)
    assert abs(norm.to_torch_cpu().item() - 7.0) < 1e-5


def test_vector_norm_linf():
    """L-infinity norm."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([3.0, -4.0, 2.0]))
    norm = torch.linalg.vector_norm(x, ord=float('inf'))
    assert abs(norm.to_torch_cpu().item() - 4.0) < 1e-5


def test_unique_sorted():
    """torch.unique returns sorted unique values."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([3, 1, 2, 1, 3, 2]))
    unique_vals = torch.unique(x)
    assert unique_vals.to_torch_cpu().tolist() == [1, 2, 3]


def test_unique_with_inverse():
    """torch.unique with return_inverse."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([3, 1, 2, 1, 3, 2]))
    unique_vals, inverse = torch.unique(x, return_inverse=True)
    assert unique_vals.to_torch_cpu().tolist() == [1, 2, 3]
    # inverse maps each element to its position in unique_vals
    inv = inverse.to_torch_cpu().tolist()
    assert inv == [2, 0, 1, 0, 2, 1]


def test_clip_grad_norm():
    """torch.nn.utils.clip_grad_norm_ works with applegpu tensors."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    w = gpu.to_applegpu(torch.randn(4, 4, requires_grad=True))
    y = (w * 3).sum()
    y.backward()
    # Gradient should be all 3s, norm = 3 * sqrt(16) = 12
    total_norm = torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
    assert total_norm.item() > 0, "Norm should be positive"
    # After clipping, grad norm should be <= 1.0 (approximately)
    clipped_norm = torch.linalg.vector_norm(
        torch.stack([w.grad.to_torch_cpu().flatten()])
    ).item()
    assert clipped_norm <= 1.1, f"Clipped norm should be ~1.0, got {clipped_norm}"
