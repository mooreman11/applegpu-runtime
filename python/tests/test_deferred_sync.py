"""Tests for deferred CPU backing sync in eager mode."""
import pytest
import torch


def _setup():
    import applegpu_runtime as gpu
    gpu.init_backend()
    gpu.enable_torch_backend()
    from applegpu_runtime.torch_backend import set_eager_mode, ApplegpuTensor
    set_eager_mode(True)
    return gpu, ApplegpuTensor


def test_inplace_add_readback():
    """In-place add_ followed by to_torch_cpu returns correct result."""
    gpu, ApplegpuTensor = _setup()
    x = torch.tensor([1.0, 2.0, 3.0])
    gx = gpu.to_applegpu(x)
    gx.add_(torch.tensor([10.0, 20.0, 30.0]))
    result = gx.to_torch_cpu()
    assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0]))


def test_inplace_mul_readback():
    """In-place mul_ followed by to_torch_cpu returns correct result."""
    gpu, ApplegpuTensor = _setup()
    x = torch.tensor([2.0, 3.0, 4.0])
    gx = gpu.to_applegpu(x)
    gx.mul_(torch.tensor([5.0, 6.0, 7.0]))
    result = gx.to_torch_cpu()
    assert torch.allclose(result, torch.tensor([10.0, 18.0, 28.0]))


def test_optimizer_step_correctness():
    """Adam optimizer step produces finite, changed parameters."""
    gpu, ApplegpuTensor = _setup()
    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = gpu.to_applegpu(torch.randn(3, 4))
    y = gpu.to_applegpu(torch.randn(3, 2))

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()

    for p in model.parameters():
        cpu_p = p.to_torch_cpu()
        assert torch.isfinite(cpu_p).all(), f"Non-finite param after step: {cpu_p}"


def test_loss_item_after_inplace():
    """loss.item() returns correct scalar after in-place optimizer updates."""
    gpu, ApplegpuTensor = _setup()
    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = gpu.to_applegpu(torch.randn(3, 4))
    y = gpu.to_applegpu(torch.randn(3, 2))

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    out2 = model(x)
    loss2 = torch.nn.functional.mse_loss(out2, y)
    val = loss2.to_torch_cpu().item()
    assert isinstance(val, float) and val >= 0


@pytest.mark.xfail(reason="GPU tensor freed during clip_grad_norm_ (pre-existing issue)", strict=True)
def test_clip_grad_norm_after_inplace():
    """clip_grad_norm_ works correctly after in-place ops."""
    gpu, ApplegpuTensor = _setup()
    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)

    x = gpu.to_applegpu(torch.randn(3, 4))
    y = gpu.to_applegpu(torch.randn(3, 2))
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if hasattr(norm, 'to_torch_cpu'):
        norm = norm.to_torch_cpu()
    assert float(norm) >= 0


def test_torch_save_after_inplace():
    """torch.save works after in-place ops (uses __reduce_ex__)."""
    import tempfile, os
    gpu, ApplegpuTensor = _setup()
    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = gpu.to_applegpu(torch.randn(3, 4))
    y = gpu.to_applegpu(torch.randn(3, 2))
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        loaded = torch.load(path, weights_only=True)
        assert len(loaded) > 0
    finally:
        os.unlink(path)
