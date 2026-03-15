"""Text generation utilities for applegpu_runtime models."""

import numpy as np


def tokenize(model_name, text):
    """Tokenize text using the HuggingFace tokenizer.

    Returns list of int token IDs.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers package required")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.encode(text)


def decode(model_name, token_ids):
    """Decode token IDs back to text.

    Returns string.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers package required")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.decode(token_ids)


def generate(model, input_ids, max_tokens=50, use_cache=True):
    """Autoregressive text generation with optional KV cache.

    Args:
        model: dict from load_gpt2_weights()
        input_ids: list of int token IDs (prompt)
        max_tokens: number of tokens to generate
        use_cache: if True, use KV cache for faster generation

    Returns:
        list of int token IDs (prompt + generated)
    """
    import applegpu_runtime as gpu
    from .gpt2 import gpt2_forward

    ids = list(input_ids)
    kv_cache = None

    for step in range(max_tokens):
        if use_cache:
            logits, kv_cache = gpt2_forward(model, ids, kv_cache=kv_cache)
        else:
            logits, _ = gpt2_forward(model, ids, kv_cache=None)

        # Argmax of last position
        last_logits = gpu.argmax(logits)
        last_idx = last_logits.to_list()[-1]

        # Clean up forward pass outputs to free memory
        gpu.destroy(logits)
        gpu.destroy(last_logits)

        ids.append(int(last_idx))

    return ids
