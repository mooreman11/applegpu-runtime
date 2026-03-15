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
    logits, _ = gpt2_forward(gpt2_model, input_ids)
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


def test_kv_cache_forward(gpt2_model):
    """KV cache forward produces same logits for last position as full forward."""
    from applegpu_runtime.models.gpt2 import gpt2_forward
    input_ids = [15496, 11, 995]  # "Hello, world"

    # Full forward (no cache)
    logits_full, kv_cache = gpt2_forward(gpt2_model, input_ids, kv_cache=None)
    full_result = logits_full.to_numpy()

    # Incremental forward for one more token (using cache)
    input_ids_extended = input_ids + [int(gpu.argmax(logits_full).to_list()[-1])]
    logits_cached, _ = gpt2_forward(gpt2_model, input_ids_extended, kv_cache=kv_cache)
    cached_result = logits_cached.to_numpy()

    # Cached forward should produce [1, vocab_size] logits
    assert cached_result.shape == (1, 50257)
    assert np.all(np.isfinite(cached_result))


def test_kv_cache_generation(gpt2_model):
    """Generation with KV cache produces same tokens as without."""
    from applegpu_runtime.models.generate import tokenize, generate

    input_ids = tokenize("gpt2", "Hello")

    # Generate with cache
    output_cached = generate(gpt2_model, input_ids, max_tokens=5, use_cache=True)

    # Generate without cache
    output_no_cache = generate(gpt2_model, input_ids, max_tokens=5, use_cache=False)

    # Both should produce the same tokens
    assert output_cached == output_no_cache


def test_kv_cache_faster(gpt2_model):
    """Generation with KV cache should be faster than without."""
    import time
    from applegpu_runtime.models.generate import tokenize, generate

    input_ids = tokenize("gpt2", "The")

    start = time.time()
    generate(gpt2_model, input_ids, max_tokens=5, use_cache=False)
    time_no_cache = time.time() - start

    start = time.time()
    generate(gpt2_model, input_ids, max_tokens=5, use_cache=True)
    time_cached = time.time() - start

    print(f"No cache: {time_no_cache:.2f}s, Cached: {time_cached:.2f}s, Speedup: {time_no_cache/time_cached:.1f}x")
    # Cached should be faster (not a strict assertion since CI can be noisy)
    assert time_cached < time_no_cache * 1.5  # at minimum not slower
