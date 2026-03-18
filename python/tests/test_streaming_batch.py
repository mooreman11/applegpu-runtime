"""Tests for streaming command buffer integration with torch backend."""
import torch


def _setup():
    import applegpu_runtime as gpu
    gpu.init_backend()
    gpu.enable_torch_backend()
    from applegpu_runtime.torch_backend import set_eager_mode, ApplegpuTensor
    return gpu, ApplegpuTensor, set_eager_mode


def test_eager_mode_enables_streaming():
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)
    set_eager_mode(False)


def test_eager_mode_toggle_idempotent():
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)
    set_eager_mode(True)
    set_eager_mode(False)
    set_eager_mode(False)
    set_eager_mode(True)
    set_eager_mode(False)


def test_training_loop_with_streaming():
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)
    model = torch.nn.Linear(8, 4)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(5):
        x = gpu.to_applegpu(torch.randn(16, 8))
        y = gpu.to_applegpu(torch.randn(16, 4))
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    final_loss = loss.to_torch_cpu().item()
    assert isinstance(final_loss, float) and final_loss >= 0
    set_eager_mode(False)


def test_cpu_fallback_during_streaming():
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)
    x = gpu.to_applegpu(torch.randn(4, 4))
    y = gpu.to_applegpu(torch.randn(4, 4))
    out = x + y
    result = out.to_torch_cpu()
    assert result.shape == (4, 4)
    assert torch.isfinite(result).all()
    set_eager_mode(False)


def test_readback_mid_streaming():
    gpu, _, set_eager_mode = _setup()
    set_eager_mode(True)
    a = gpu.to_applegpu(torch.tensor([1.0, 2.0, 3.0]))
    b = gpu.to_applegpu(torch.tensor([10.0, 20.0, 30.0]))
    c = a + b
    result = c.to_torch_cpu()
    assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0]))
    d = c + a
    result2 = d.to_torch_cpu()
    assert torch.allclose(result2, torch.tensor([12.0, 24.0, 36.0]))
    set_eager_mode(False)
