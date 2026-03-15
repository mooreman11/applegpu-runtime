"""GPT-2 model implementation using applegpu_runtime ops."""

import numpy as np


def load_gpt2_weights(model_name="gpt2"):
    """Load GPT-2 weights from HuggingFace and convert to GPU tensors.

    Returns a dict with model config and GPU tensors for all parameters.
    """
    import applegpu_runtime as gpu

    try:
        from transformers import GPT2LMHeadModel, GPT2Config
    except ImportError:
        raise ImportError(
            "transformers package required. Install with: pip install transformers torch"
        )

    gpu.init_backend()

    # GPT-2 has ~148 weight tensors + many intermediates during generation
    # Unlimited resource limits — GPT-2 creates many intermediates during generation
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)

    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    config = hf_model.config
    state_dict = hf_model.state_dict()

    weights = {}
    for name, param in state_dict.items():
        np_param = param.detach().cpu().float().numpy()
        weights[name] = gpu.from_numpy(np_param)

    return {
        "weights": weights,
        "config": {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "vocab_size": config.vocab_size,
            "n_positions": config.n_positions,
        },
    }


def gpt2_forward(model, input_ids, kv_cache=None):
    """Run GPT-2 forward pass with optional KV cache.

    Uses batched multi-head attention: Q/K/V are reshaped to
    [n_head, seq, d_head] so all heads run in a single batched
    matmul/attention dispatch per layer.

    Args:
        model: dict from load_gpt2_weights()
        input_ids: list of int token IDs (full sequence if no cache, single token if cached)
        kv_cache: optional list of (k, v) tuples per layer, shape [n_head, past, d_head]

    Returns:
        (logits, new_kv_cache) where logits is [seq_len, vocab_size] (or [1, vocab_size] with cache)
        and new_kv_cache is the updated cache for the next call.
    """
    import applegpu_runtime as gpu

    w = model["weights"]
    cfg = model["config"]
    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    n_embd = cfg["n_embd"]
    d_head = n_embd // n_head

    use_cache = kv_cache is not None

    if use_cache:
        # Incremental: only process the last token
        token_ids = [input_ids[-1]]
        pos_offset = len(input_ids) - 1
    else:
        token_ids = input_ids
        pos_offset = 0

    seq_len = len(token_ids)

    # Token + position embeddings
    indices = gpu.from_numpy(np.array(token_ids, dtype=np.int32))
    pos_indices = gpu.from_numpy(np.array(list(range(pos_offset, pos_offset + seq_len)), dtype=np.int32))

    tok_emb = gpu.embedding(w["transformer.wte.weight"], indices)
    pos_emb = gpu.embedding(w["transformer.wpe.weight"], pos_indices)
    x = tok_emb + pos_emb

    new_kv_cache = []

    # Transformer blocks
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"

        # Layer norm 1
        ln1_g = w[f"{prefix}.ln_1.weight"]
        ln1_b = w[f"{prefix}.ln_1.bias"]
        x_norm = gpu.layer_norm(x, ln1_g, ln1_b)

        # QKV projection
        attn_w = w[f"{prefix}.attn.c_attn.weight"]
        attn_b = w[f"{prefix}.attn.c_attn.bias"]
        qkv = x_norm @ attn_w
        qkv = gpu.add_bias(qkv, attn_b)

        # Split into Q, K, V: each [seq, n_embd]
        q_all = gpu.slice(qkv, dim=1, start=0, end=n_embd)
        k_new = gpu.slice(qkv, dim=1, start=n_embd, end=2*n_embd)
        v_new = gpu.slice(qkv, dim=1, start=2*n_embd, end=3*n_embd)

        # Reshape to [seq, n_head, d_head] then transpose to [n_head, seq, d_head]
        q = gpu.reshape(q_all, [seq_len, n_head, d_head])
        q = gpu.transpose_dims(q, 0, 1)  # [n_head, seq, d_head]
        k = gpu.reshape(k_new, [seq_len, n_head, d_head])
        k = gpu.transpose_dims(k, 0, 1)  # [n_head, seq, d_head]
        v = gpu.reshape(v_new, [seq_len, n_head, d_head])
        v = gpu.transpose_dims(v, 0, 1)  # [n_head, seq, d_head]

        if use_cache:
            # Concat new K/V with cached: [n_head, past, d_head] + [n_head, 1, d_head]
            cached_k, cached_v = kv_cache[i]
            k = gpu.concat(cached_k, k, dim=1)  # [n_head, past+1, d_head]
            v = gpu.concat(cached_v, v, dim=1)  # [n_head, past+1, d_head]

        new_kv_cache.append((k, v))

        # Batched attention: one dispatch for all heads
        # Q: [n_head, seq, d_head], K/V: [n_head, total_seq, d_head]
        if use_cache:
            attn_out = gpu.attention(q, k, v)
        else:
            attn_out = gpu.attention_causal(q, k, v)
        # attn_out: [n_head, seq, d_head]

        # Reshape back: [n_head, seq, d_head] -> [seq, n_head, d_head] -> [seq, n_embd]
        attn_out = gpu.transpose_dims(attn_out, 0, 1)  # [seq, n_head, d_head]
        attn_out = gpu.reshape(attn_out, [seq_len, n_embd])

        # Output projection
        proj_w = w[f"{prefix}.attn.c_proj.weight"]
        proj_b = w[f"{prefix}.attn.c_proj.bias"]
        attn_out = attn_out @ proj_w
        attn_out = gpu.add_bias(attn_out, proj_b)

        # Residual connection
        x = x + attn_out

        # Layer norm 2
        ln2_g = w[f"{prefix}.ln_2.weight"]
        ln2_b = w[f"{prefix}.ln_2.bias"]
        x_norm2 = gpu.layer_norm(x, ln2_g, ln2_b)

        # FFN
        fc_w = w[f"{prefix}.mlp.c_fc.weight"]
        fc_b = w[f"{prefix}.mlp.c_fc.bias"]
        ffn = x_norm2 @ fc_w
        ffn = gpu.add_bias(ffn, fc_b)
        ffn = gpu.gelu(ffn)

        proj2_w = w[f"{prefix}.mlp.c_proj.weight"]
        proj2_b = w[f"{prefix}.mlp.c_proj.bias"]
        ffn = ffn @ proj2_w
        ffn = gpu.add_bias(ffn, proj2_b)

        # Residual connection
        x = x + ffn

    # Final layer norm
    ln_f_g = w["transformer.ln_f.weight"]
    ln_f_b = w["transformer.ln_f.bias"]
    x = gpu.layer_norm(x, ln_f_g, ln_f_b)

    # LM head
    lm_head_w = gpu.transpose(w["transformer.wte.weight"])
    logits = x @ lm_head_w

    return logits, new_kv_cache
