"""Tests for BERT on applegpu device backend."""
import pytest
import numpy as np

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()
    gpu.enable_torch_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)


@pytest.fixture(scope="module")
def bert_model():
    """Load tiny BERT (random weights, no download)."""
    from transformers import BertModel, BertConfig
    config = BertConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=128, vocab_size=100, max_position_embeddings=32,
        attn_implementation="eager",
    )
    model = BertModel(config)
    model.eval()
    return model


def test_bert_forward_shape(bert_model):
    """BERT forward pass produces correct output shape on CPU first."""
    input_ids = torch.tensor([[1, 5, 3, 7, 2]])
    attention_mask = torch.ones(1, 5)

    with torch.no_grad():
        cpu_output = bert_model(input_ids, attention_mask=attention_mask)

    assert cpu_output.last_hidden_state.shape == (1, 5, 64)


def test_bert_gpu_forward(bert_model):
    """BERT inference on applegpu."""
    from applegpu_runtime.torch_backend import ApplegpuTensor

    model_gpu = gpu.to_applegpu(bert_model)

    input_ids = torch.tensor([[1, 5, 3, 7, 2]])
    attention_mask = torch.ones(1, 5)

    # Move inputs to applegpu
    input_ids_gpu = ApplegpuTensor.from_torch(input_ids)
    attention_mask_gpu = ApplegpuTensor.from_torch(attention_mask)

    with torch.no_grad():
        output = model_gpu(input_ids_gpu, attention_mask=attention_mask_gpu)
        result = output.last_hidden_state
        if isinstance(result, ApplegpuTensor):
            result = result.to_torch_cpu()
        assert result.shape == (1, 5, 64)
        assert torch.all(torch.isfinite(result))


def test_bert_matches_cpu():
    """GPU output matches CPU within tolerance."""
    from transformers import BertModel, BertConfig
    from applegpu_runtime.torch_backend import ApplegpuTensor

    config = BertConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=128, vocab_size=100, max_position_embeddings=32,
        attn_implementation="eager",
    )

    input_ids = torch.tensor([[1, 5, 3, 7, 2]])
    attention_mask = torch.ones(1, 5)

    # CPU run with a fresh model
    cpu_model = BertModel(config)
    cpu_model.eval()
    # Save state dict before GPU conversion
    state = cpu_model.state_dict()

    with torch.no_grad():
        cpu_output = cpu_model(input_ids, attention_mask=attention_mask)
        cpu_hidden = cpu_output.last_hidden_state.clone()

    # GPU run: load same weights
    gpu_model = BertModel(config)
    gpu_model.load_state_dict(state)
    gpu_model.eval()
    model_gpu = gpu.to_applegpu(gpu_model)

    input_ids_gpu = ApplegpuTensor.from_torch(input_ids)
    attention_mask_gpu = ApplegpuTensor.from_torch(attention_mask)

    with torch.no_grad():
        gpu_output = model_gpu(input_ids_gpu, attention_mask=attention_mask_gpu)
        gpu_hidden = gpu_output.last_hidden_state
        if isinstance(gpu_hidden, ApplegpuTensor):
            gpu_hidden = gpu_hidden.to_torch_cpu()

    assert torch.allclose(gpu_hidden, cpu_hidden, atol=1e-2, rtol=1e-2), \
        f"Max diff: {(gpu_hidden - cpu_hidden).abs().max().item()}"
