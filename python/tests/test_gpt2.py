"""Tests for GPT-2 model. Requires transformers + torch."""

import pytest
import numpy as np

# Skip all tests if transformers not installed
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 once for all tests in this module."""
    return gpu.load_model("gpt2")


def test_load_model(gpt2_model):
    """Model loads with expected structure."""
    assert "weights" in gpt2_model
    assert "config" in gpt2_model
    cfg = gpt2_model["config"]
    assert cfg["n_layer"] == 12
    assert cfg["n_head"] == 12
    assert cfg["n_embd"] == 768
    assert cfg["vocab_size"] == 50257


def test_weight_shapes(gpt2_model):
    """Key weights have correct shapes."""
    w = gpt2_model["weights"]
    assert w["transformer.wte.weight"].shape == [50257, 768]
    assert w["transformer.wpe.weight"].shape == [1024, 768]
    assert w["transformer.h.0.attn.c_attn.weight"].shape == [768, 2304]
    assert w["transformer.h.0.mlp.c_fc.weight"].shape == [768, 3072]


def test_forward_pass_shape(gpt2_model):
    """Forward pass produces correct output shape."""
    from applegpu_runtime.models.gpt2 import gpt2_forward
    input_ids = [15496, 11, 995]  # "Hello, world"
    logits = gpt2_forward(gpt2_model, input_ids)
    result = logits.to_numpy()
    assert result.shape == (3, 50257)  # [seq_len, vocab_size]
    assert np.all(np.isfinite(result))


def test_generate_produces_tokens(gpt2_model):
    """Generation loop produces additional tokens."""
    from applegpu_runtime.models.generate import tokenize, generate
    input_ids = tokenize("gpt2", "Hello")
    output_ids = generate(gpt2_model, input_ids, max_tokens=5)
    assert len(output_ids) == len(input_ids) + 5
    assert all(isinstance(t, int) for t in output_ids)


def test_run_model(gpt2_model):
    """End-to-end: text in, text out."""
    result = gpu.generate_text(gpt2_model, "The capital of France is", max_tokens=10)
    assert isinstance(result, str)
    assert len(result) > len("The capital of France is")
    print(f"Generated: {result}")  # for manual inspection
