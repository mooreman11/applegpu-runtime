"""Tests for stack op and conv1d backward."""
import applegpu_runtime as gpu


def test_stack_dim0():
    """Stack along dim 0 creates new first dimension."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    a = gpu.to_applegpu(torch.tensor([1.0, 2.0, 3.0]))
    b = gpu.to_applegpu(torch.tensor([4.0, 5.0, 6.0]))
    c = torch.stack([a, b], dim=0)
    result = c.to_torch_cpu()
    assert result.shape == (2, 3)
    assert result.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_stack_dim1():
    """Stack along dim 1 interleaves."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    a = gpu.to_applegpu(torch.tensor([1.0, 2.0, 3.0]))
    b = gpu.to_applegpu(torch.tensor([4.0, 5.0, 6.0]))
    c = torch.stack([a, b], dim=1)
    result = c.to_torch_cpu()
    assert result.shape == (3, 2)
    assert result.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_stack_multiple():
    """Stack 3+ tensors."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    tensors = [gpu.to_applegpu(torch.tensor([float(i)])) for i in range(5)]
    c = torch.stack(tensors, dim=0)
    result = c.to_torch_cpu()
    assert result.shape == (5, 1)
    assert result.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_stack_2d():
    """Stack 2D tensors along dim 0."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    a = gpu.to_applegpu(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    b = gpu.to_applegpu(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    c = torch.stack([a, b], dim=0)
    result = c.to_torch_cpu()
    assert result.shape == (2, 2, 2)


def test_conv1d_backward_gradients():
    """Conv1d backward produces correct gradient shapes and finite values."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch

    x = gpu.to_applegpu(torch.randn(1, 2, 8, requires_grad=True))
    w = gpu.to_applegpu(torch.randn(4, 2, 3, requires_grad=True))
    b = gpu.to_applegpu(torch.randn(4, requires_grad=True))

    y = torch.nn.functional.conv1d(x, w, b, padding=1)
    y.sum().backward()

    assert x.grad is not None, "x.grad should not be None"
    assert w.grad is not None, "w.grad should not be None"
    assert b.grad is not None, "b.grad should not be None"
    assert x.grad.shape == (1, 2, 8)
    assert w.grad.shape == (4, 2, 3)
    assert b.grad.shape == (4,)
    assert x.grad.to_torch_cpu().isfinite().all()
    assert w.grad.to_torch_cpu().isfinite().all()
    assert b.grad.to_torch_cpu().isfinite().all()


def test_conv1d_backward_matches_cpu():
    """Conv1d backward produces same gradients as CPU reference."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch

    torch.manual_seed(42)
    x_data = torch.randn(1, 2, 8)
    w_data = torch.randn(4, 2, 3)
    b_data = torch.randn(4)

    # CPU reference
    x_cpu = x_data.clone().requires_grad_(True)
    w_cpu = w_data.clone().requires_grad_(True)
    b_cpu = b_data.clone().requires_grad_(True)
    y_cpu = torch.nn.functional.conv1d(x_cpu, w_cpu, b_cpu, padding=1)
    y_cpu.sum().backward()

    # GPU
    x_gpu = gpu.to_applegpu(x_data.clone().requires_grad_(True))
    w_gpu = gpu.to_applegpu(w_data.clone().requires_grad_(True))
    b_gpu = gpu.to_applegpu(b_data.clone().requires_grad_(True))
    y_gpu = torch.nn.functional.conv1d(x_gpu, w_gpu, b_gpu, padding=1)
    y_gpu.sum().backward()

    # Compare
    assert torch.allclose(x_gpu.grad.to_torch_cpu(), x_cpu.grad, atol=1e-4), \
        f"x grad mismatch: {(x_gpu.grad.to_torch_cpu() - x_cpu.grad).abs().max()}"
    assert torch.allclose(w_gpu.grad.to_torch_cpu(), w_cpu.grad, atol=1e-4), \
        f"w grad mismatch: {(w_gpu.grad.to_torch_cpu() - w_cpu.grad).abs().max()}"
    assert torch.allclose(b_gpu.grad.to_torch_cpu(), b_cpu.grad, atol=1e-4), \
        f"b grad mismatch: {(b_gpu.grad.to_torch_cpu() - b_cpu.grad).abs().max()}"
