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


def _sample_token(logits_np, temperature=1.0, top_k=0, top_p=0.0):
    """Sample a token from logits using temperature, top-k, and top-p.

    Args:
        logits_np: numpy array of shape [vocab_size]
        temperature: scaling factor (lower = more deterministic, higher = more random)
        top_k: if > 0, only sample from the top k tokens
        top_p: if > 0, only sample from the smallest set of tokens whose cumulative probability >= top_p

    Returns:
        int token ID
    """
    if temperature <= 0:
        # Greedy
        return int(np.argmax(logits_np))

    # Apply temperature
    logits_np = logits_np / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, len(logits_np))
        indices = np.argpartition(logits_np, -top_k)[-top_k:]
        mask = np.full_like(logits_np, -np.inf)
        mask[indices] = logits_np[indices]
        logits_np = mask

    # Softmax
    logits_np = logits_np - np.max(logits_np)  # numerical stability
    probs = np.exp(logits_np)
    probs = probs / probs.sum()

    # Top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        # Find cutoff index where cumulative >= top_p
        cutoff = np.searchsorted(cumulative, top_p) + 1
        # Zero out everything below cutoff
        allowed = sorted_indices[:cutoff]
        mask = np.zeros_like(probs)
        mask[allowed] = probs[allowed]
        probs = mask / mask.sum()

    # Sample
    return int(np.random.choice(len(probs), p=probs))


def generate(model, input_ids, max_tokens=50, use_cache=True,
             temperature=1.0, top_k=50, top_p=0.9):
    """Autoregressive text generation with sampling and optional KV cache.

    Args:
        model: dict from load_gpt2_weights()
        input_ids: list of int token IDs (prompt)
        max_tokens: number of tokens to generate
        use_cache: if True, use KV cache for faster generation
        temperature: sampling temperature (0 = greedy, 1.0 = standard)
        top_k: if > 0, only sample from top k tokens (default 50)
        top_p: if > 0, nucleus sampling threshold (default 0.9)

    Returns:
        list of int token IDs (prompt + generated)
    """
    import applegpu_runtime as gpu
    from .gpt2 import gpt2_forward

    ids = list(input_ids)
    kv_cache = None

    # Free any stale pooled buffers from previous runs
    gpu.pool_drain()

    for step in range(max_tokens):
        if use_cache:
            logits, kv_cache = gpt2_forward(model, ids, kv_cache=kv_cache)
        else:
            logits, _ = gpt2_forward(model, ids, kv_cache=None)

        # Get logits for last position as numpy
        logits_np = logits.to_numpy()
        last_logits = logits_np[-1]  # [vocab_size]

        # Sample next token
        next_id = _sample_token(last_logits, temperature=temperature,
                                top_k=top_k, top_p=top_p)

        # Clean up
        gpu.destroy(logits)

        ids.append(next_id)

    return ids
