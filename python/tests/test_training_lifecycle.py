"""Tests for issue #14: optimizer state, schedulers, checkpointing, grad clipping."""
import applegpu_runtime as gpu


def test_adam_training_loop():
    """Adam optimizer with multi-layer model produces decreasing loss."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 8),
    )
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = gpu.to_applegpu(torch.randn(16, 32))
    target = gpu.to_applegpu(torch.randn(16, 8))

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.to_torch_cpu().item())

    assert losses[-1] < losses[0], f"Loss should decrease: {losses}"


def test_adam_with_grad_clipping():
    """Adam + clip_grad_norm_ works without errors."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch

    model = torch.nn.Linear(16, 4)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = gpu.to_applegpu(torch.randn(8, 16))
    y = model(x)
    y.sum().backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert norm.item() > 0
    optimizer.step()


def test_reduce_lr_on_plateau():
    """ReduceLROnPlateau adjusts learning rate."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch

    model = torch.nn.Linear(4, 2)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=0, factor=0.5
    )

    initial_lr = optimizer.param_groups[0]['lr']
    # Simulate no improvement
    scheduler.step(1.0)
    scheduler.step(1.0)
    new_lr = optimizer.param_groups[0]['lr']
    assert new_lr < initial_lr, f"LR should decrease: {initial_lr} → {new_lr}"


def test_checkpoint_save_load():
    """Model state dict can be saved and loaded."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    import tempfile
    import os

    model = torch.nn.Linear(8, 4)
    model = gpu.to_applegpu(model)

    # Save
    state = {k: v.to_torch_cpu() if hasattr(v, 'to_torch_cpu') else v
             for k, v in model.state_dict().items()}
    path = tempfile.mktemp(suffix='.pt')
    torch.save(state, path)

    # Load
    loaded = torch.load(path, weights_only=True)
    assert 'weight' in loaded
    assert 'bias' in loaded
    assert loaded['weight'].shape == (4, 8)
    os.unlink(path)
