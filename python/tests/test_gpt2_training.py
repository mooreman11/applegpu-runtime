"""Tests for GPT-2 fine-tuning on applegpu."""
import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")

import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import ApplegpuTensor


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()
    gpu.enable_torch_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)
    gpu.enable_training()
    yield
    gpu.disable_training()


@pytest.fixture(scope="module")
def tiny_gpt2():
    """Create a tiny GPT-2 for fast testing."""
    from transformers import GPT2LMHeadModel, GPT2Config
    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=100, n_positions=32,
    )
    model = GPT2LMHeadModel(config)
    model.eval()  # batch_norm etc in eval mode
    return model


def test_gpt2_forward_on_applegpu(tiny_gpt2):
    """GPT-2 forward pass through device backend."""
    model = gpu.to_applegpu(tiny_gpt2)
    input_ids = ApplegpuTensor.from_torch(torch.tensor([[1, 5, 3, 7, 2]]))

    with torch.no_grad():
        output = model(input_ids)

    logits = output.logits
    if isinstance(logits, ApplegpuTensor):
        logits = logits.to_torch_cpu()
    assert logits.shape == (1, 5, 100)  # batch=1, seq=5, vocab=100
    assert torch.all(torch.isfinite(logits))


def test_gpt2_training_step(tiny_gpt2):
    """One training step: forward + loss + backward + optimizer step."""
    model = gpu.to_applegpu(tiny_gpt2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    input_ids = ApplegpuTensor.from_torch(torch.tensor([[1, 5, 3, 7, 2]]))
    labels = ApplegpuTensor.from_torch(torch.tensor([[5, 3, 7, 2, 1]]))

    optimizer.zero_grad()
    output = model(input_ids, labels=labels)
    loss = output.loss

    if isinstance(loss, ApplegpuTensor):
        loss_val = loss.to_torch_cpu().item()
    else:
        loss_val = loss.item()

    assert loss_val > 0  # cross-entropy loss should be positive

    loss.backward()
    optimizer.step()

    print(f"Training step loss: {loss_val:.4f}")


def test_gpt2_loss_decreases(tiny_gpt2):
    """Training for multiple steps -- loss should decrease."""
    model = gpu.to_applegpu(tiny_gpt2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    input_ids = ApplegpuTensor.from_torch(torch.tensor([[1, 5, 3, 7, 2]]))
    labels = ApplegpuTensor.from_torch(torch.tensor([[5, 3, 7, 2, 1]]))

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        output = model(input_ids, labels=labels)
        loss = output.loss

        if isinstance(loss, ApplegpuTensor):
            loss_val = loss.to_torch_cpu().item()
        else:
            loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()

    print(f"Losses: {[f'{l:.4f}' for l in losses]}")
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"
