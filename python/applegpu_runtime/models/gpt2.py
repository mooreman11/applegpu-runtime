"""GPT-2 model implementation using applegpu_runtime ops."""

import numpy as np


def _safe_gelu(tensor):
    """GELU activation with numpy fallback for numerical stability.

    The Metal GELU kernel uses a tanh approximation that produces NaN for
    |x| > ~10. We fall back to a numpy-based GELU using the tanh
    approximation with clamped input to avoid overflow.
    """
    import applegpu_runtime as gpu

    x = tensor.to_numpy()
    # Tanh-approximation GELU with clamped inner argument to prevent overflow
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    inner = np.clip(inner, -20.0, 20.0)  # tanh saturates well before this
    result = x * 0.5 * (1.0 + np.tanh(inner))
    return gpu.from_numpy(result.astype(np.float32))


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

    # GPT-2 has ~148 weight tensors totaling ~500MB; raise resource limits
    gpu.set_limits(max_tensor_size_mb=512, max_memory_mb=4096, max_tensors=50000)

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


def gpt2_forward(model, input_ids):
    """Run GPT-2 forward pass.

    Args:
        model: dict from load_gpt2_weights()
        input_ids: list of int token IDs

    Returns:
        logits GpuTensor of shape [seq_len, vocab_size]
    """
    import applegpu_runtime as gpu

    w = model["weights"]
    cfg = model["config"]
    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    n_embd = cfg["n_embd"]
    d_head = n_embd // n_head
    seq_len = len(input_ids)

    # Token + position embeddings
    indices = gpu.from_numpy(np.array(input_ids, dtype=np.int32))
    pos_indices = gpu.from_numpy(np.array(list(range(seq_len)), dtype=np.int32))

    tok_emb = gpu.embedding(w["transformer.wte.weight"], indices)      # [seq, n_embd]
    pos_emb = gpu.embedding(w["transformer.wpe.weight"], pos_indices)  # [seq, n_embd]
    x = tok_emb + pos_emb  # [seq, n_embd]

    # Transformer blocks
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"

        # Layer norm 1
        ln1_g = w[f"{prefix}.ln_1.weight"]
        ln1_b = w[f"{prefix}.ln_1.bias"]
        x_norm = gpu.layer_norm(x, ln1_g, ln1_b)

        # Self-attention
        # QKV projection: [seq, n_embd] @ [n_embd, 3*n_embd] + bias
        attn_w = w[f"{prefix}.attn.c_attn.weight"]
        attn_b = w[f"{prefix}.attn.c_attn.bias"]
        qkv = x_norm @ attn_w  # [seq, 3*n_embd]
        qkv = gpu.add_bias(qkv, attn_b)

        # Split into Q, K, V
        q_all = gpu.slice(qkv, dim=1, start=0, end=n_embd)
        k_all = gpu.slice(qkv, dim=1, start=n_embd, end=2*n_embd)
        v_all = gpu.slice(qkv, dim=1, start=2*n_embd, end=3*n_embd)

        # Multi-head attention
        head_outputs = []
        for h in range(n_head):
            q_h = gpu.slice(q_all, dim=1, start=h*d_head, end=(h+1)*d_head)
            k_h = gpu.slice(k_all, dim=1, start=h*d_head, end=(h+1)*d_head)
            v_h = gpu.slice(v_all, dim=1, start=h*d_head, end=(h+1)*d_head)
            out_h = gpu.attention_causal(q_h, k_h, v_h)
            head_outputs.append(out_h)

        # Concat heads
        attn_out = head_outputs[0]
        for h in head_outputs[1:]:
            attn_out = gpu.concat(attn_out, h, dim=1)

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
        ffn = _safe_gelu(ffn)

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

    # LM head: project to vocabulary
    # GPT-2 ties weights: lm_head weight = wte weight transposed
    lm_head_w = gpu.transpose(w["transformer.wte.weight"])  # [n_embd, vocab_size]
    logits = x @ lm_head_w  # [seq_len, vocab_size]

    return logits
