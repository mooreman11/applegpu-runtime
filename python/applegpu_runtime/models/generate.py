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


def generate(model, input_ids, max_tokens=50):
    """Autoregressive text generation.

    Args:
        model: dict from load_gpt2_weights()
        input_ids: list of int token IDs (prompt)
        max_tokens: number of tokens to generate

    Returns:
        list of int token IDs (prompt + generated)
    """
    import applegpu_runtime as gpu
    from .gpt2 import gpt2_forward

    ids = list(input_ids)

    for _ in range(max_tokens):
        # Forward pass
        logits = gpt2_forward(model, ids)

        # Get logits for last position
        last_logits = gpu.argmax(logits)  # [seq_len] Int32
        last_idx = last_logits.to_list()[-1]  # argmax of last row

        # Wait -- argmax gives the index of max per ROW, but we want
        # the argmax of the LAST ROW only. The above is correct:
        # logits is [seq_len, vocab_size], argmax gives [seq_len] indices,
        # we take the last one.

        ids.append(int(last_idx))

    return ids
