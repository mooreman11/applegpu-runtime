"""Layer-by-layer Whisper comparison: applegpu_runtime vs HuggingFace PyTorch reference.

Compares encoder output, decoder hidden states, and final logits to pinpoint
where our implementation diverges from the reference.
"""
import numpy as np
import torch
import torch.nn.functional as F

MODEL_NAME = "tiny"


def load_reference():
    """Load HuggingFace WhisperModel and return model + state dict."""
    from transformers import WhisperModel as HFWhisper
    hf = HFWhisper.from_pretrained(f"openai/whisper-{MODEL_NAME}")
    hf.eval()
    return hf


def make_test_mel():
    """Create a deterministic test mel spectrogram [1, 80, 3000]."""
    np.random.seed(42)
    mel = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
    return mel


def sinusoids(length, channels):
    """Match WhisperModel._sinusoids."""
    pos = np.arange(length, dtype=np.float32)[:, np.newaxis]
    dim = np.arange(0, channels, 2, dtype=np.float32)[np.newaxis, :]
    freqs = pos / (10000 ** (dim / channels))
    emb = np.zeros((length, channels), dtype=np.float32)
    emb[:, 0::2] = np.sin(freqs)
    emb[:, 1::2] = np.cos(freqs)
    return emb


def compare(name, ours_np, ref_np, atol=1e-3, rtol=1e-3):
    """Compare two numpy arrays, print max error and pass/fail."""
    if ours_np.shape != ref_np.shape:
        print(f"  SHAPE MISMATCH  {name}: ours={ours_np.shape} ref={ref_np.shape}")
        return False
    abs_err = np.abs(ours_np - ref_np)
    max_err = abs_err.max()
    mean_err = abs_err.mean()
    rel_err = abs_err / (np.abs(ref_np) + 1e-8)
    max_rel = rel_err.max()
    ok = np.allclose(ours_np, ref_np, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_abs={max_err:.6f} mean_abs={mean_err:.6f} max_rel={max_rel:.6f}")
    if not ok:
        # Show first few divergent values
        flat_ours = ours_np.flatten()
        flat_ref = ref_np.flatten()
        flat_err = abs_err.flatten()
        worst_idx = np.argsort(flat_err)[-5:][::-1]
        for idx in worst_idx:
            print(f"    idx={idx}: ours={flat_ours[idx]:.6f} ref={flat_ref[idx]:.6f} err={flat_err[idx]:.6f}")
    return ok


# ─── Reference encoder (step by step) ─────────────────────────────────────

def ref_encoder_stepwise(hf, mel_torch):
    """Run HF encoder step by step, returning intermediates."""
    enc = hf.encoder
    sd = hf.state_dict()
    intermediates = {}

    with torch.no_grad():
        # Conv1d layers
        x = F.gelu(enc.conv1(mel_torch))
        intermediates["enc_conv1"] = x.numpy()
        x = F.gelu(enc.conv2(x))
        intermediates["enc_conv2"] = x.numpy()

        # Transpose to [batch, seq, channels]
        x = x.permute(0, 2, 1)
        intermediates["enc_transposed"] = x.numpy()

        # Positional embedding
        pos_emb = sinusoids(x.shape[1], x.shape[2])
        x = x + torch.from_numpy(pos_emb)
        intermediates["enc_after_pos"] = x.numpy()

        # Encoder blocks
        for i, layer in enumerate(enc.layers):
            x = layer(x, attention_mask=None)[0]
            intermediates[f"enc_block_{i}"] = x.numpy()

        # Final layer norm
        x = enc.layer_norm(x)
        intermediates["enc_output"] = x.numpy()

    return intermediates


# ─── Reference decoder (step by step) ─────────────────────────────────────

def ref_decoder_stepwise(hf, encoder_output_torch, tokens):
    """Run HF decoder step by step with full prefix, returning intermediates."""
    dec = hf.decoder
    intermediates = {}

    with torch.no_grad():
        tokens_t = torch.tensor([tokens], dtype=torch.long)

        # Embedding + positional
        x = dec.embed_tokens(tokens_t)
        intermediates["dec_embed"] = x.numpy()

        pos = dec.embed_positions(tokens_t)
        intermediates["dec_pos_emb"] = pos.numpy()

        x = x + pos
        intermediates["dec_after_pos"] = x.numpy()

        # Decoder blocks
        # Build causal mask
        seq_len = x.shape[1]
        causal_mask = torch.full((seq_len, seq_len), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

        for i, layer in enumerate(dec.layers):
            x = layer(
                x,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_output_torch,
            )[0]
            intermediates[f"dec_block_{i}"] = x.numpy()

        # Final layer norm
        x = dec.layer_norm(x)
        intermediates["dec_output"] = x.numpy()

        # Logits via weight tying
        logits = x @ dec.embed_tokens.weight.T
        intermediates["logits"] = logits.numpy()

    return intermediates


# ─── Our encoder (step by step) ───────────────────────────────────────────

def our_encoder_stepwise(model, mel_gpu):
    """Run our encoder step by step, returning intermediates as numpy."""
    import applegpu_runtime as gpu
    w = model.weights
    n_head = model.config["n_head"]
    n_layer = model.config["n_layer"]
    intermediates = {}

    # Conv1d layers
    x = gpu.conv1d(mel_gpu, w["encoder.conv1.weight"], stride=1, padding=1)
    x = gpu.add_bias(x, w["encoder.conv1.bias"])
    x = gpu.gelu(x)
    x.eval()
    intermediates["enc_conv1"] = np.array(x.to_list()).reshape(x.shape)

    x = gpu.conv1d(x, w["encoder.conv2.weight"], stride=2, padding=1)
    x = gpu.add_bias(x, w["encoder.conv2.bias"])
    x = gpu.gelu(x)
    x.eval()
    intermediates["enc_conv2"] = np.array(x.to_list()).reshape(x.shape)

    # Transpose
    x = gpu.transpose_dims(x, 1, 2)
    x.eval()
    intermediates["enc_transposed"] = np.array(x.to_list()).reshape(x.shape)

    # Positional embedding
    x = x + w["encoder.positional_embedding"]
    x.eval()
    intermediates["enc_after_pos"] = np.array(x.to_list()).reshape(x.shape)

    # Encoder blocks
    for i in range(n_layer):
        prefix = f"encoder.layers.{i}"
        x = model._encoder_block(x, prefix, n_head)
        x.eval()
        intermediates[f"enc_block_{i}"] = np.array(x.to_list()).reshape(x.shape)

    # Final layer norm
    x = model._layer_norm(x, "encoder.layer_norm")
    x.eval()
    intermediates["enc_output"] = np.array(x.to_list()).reshape(x.shape)

    return intermediates, x


def our_decoder_stepwise(model, encoder_output, tokens):
    """Run our decoder step by step, returning intermediates as numpy."""
    import applegpu_runtime as gpu
    from applegpu_runtime.models.whisper import KVCache
    w = model.weights
    n_head = model.config["n_head"]
    n_layer = model.config["n_layer"]
    intermediates = {}

    tokens_arr = np.array([tokens], dtype=np.int32)
    x_tokens = gpu.from_numpy(tokens_arr)

    # Embedding
    x = gpu.embedding(w["decoder.embed_tokens.weight"], x_tokens)
    x.eval()
    intermediates["dec_embed"] = np.array(x.to_list()).reshape(x.shape)

    # Positional embedding
    seq_len = x.shape[1]
    pos_emb = gpu.slice(w["decoder.embed_positions.weight"], 0, 0, seq_len)
    pos_emb.eval()
    intermediates["dec_pos_emb"] = np.array(pos_emb.to_list()).reshape(pos_emb.shape)

    x = x + pos_emb
    x.eval()
    intermediates["dec_after_pos"] = x.numpy() if hasattr(x, 'numpy') else np.array(x.to_list()).reshape(x.shape)

    # Decoder blocks
    kv_cache = KVCache(n_layer)
    for i in range(n_layer):
        prefix = f"decoder.layers.{i}"
        x = model._decoder_block(x, encoder_output, prefix, n_head, kv_cache, i)
        x.eval()
        intermediates[f"dec_block_{i}"] = np.array(x.to_list()).reshape(x.shape)

    # Final layer norm
    x = model._layer_norm(x, "decoder.layer_norm")
    x.eval()
    intermediates["dec_output"] = np.array(x.to_list()).reshape(x.shape)

    # Logits via weight tying
    logits = gpu.matmul(x, gpu.transpose(w["decoder.embed_tokens.weight"]))
    logits.eval()
    intermediates["logits"] = np.array(logits.to_list()).reshape(logits.shape)

    return intermediates


# ─── Focused attention comparison ─────────────────────────────────────────

def compare_self_attention_causal_vs_noncausal(hf, model, encoder_output_torch, encoder_output_gpu, tokens):
    """Specifically test whether causal masking matters for this prefix."""
    import applegpu_runtime as gpu
    from applegpu_runtime.models.whisper import KVCache

    print("\n" + "="*70)
    print("CAUSAL vs NON-CAUSAL SELF-ATTENTION ANALYSIS")
    print("="*70)

    n_head = model.config["n_head"]
    n_layer = model.config["n_layer"]
    w = model.weights

    tokens_arr = np.array([tokens], dtype=np.int32)
    x_tokens = gpu.from_numpy(tokens_arr)
    x = gpu.embedding(w["decoder.embed_tokens.weight"], x_tokens)
    seq_len = x.shape[1]
    pos_emb = gpu.slice(w["decoder.embed_positions.weight"], 0, 0, seq_len)
    x = x + pos_emb

    print(f"\n  Prefix length: {len(tokens)} tokens (seq_len={seq_len})")
    print(f"  When seq_len > 1, decoder self-attention MUST be causal")
    print(f"  Code should use: gpu.attention_causal for decoder self-attention")

    # Compare layer 0 self-attention with and without causal mask
    prefix = "decoder.layers.0"
    x_normed = model._layer_norm(x, f"{prefix}.self_attn_layer_norm")
    q = model._linear(x_normed, f"{prefix}.self_attn.q_proj")
    k = model._linear(x_normed, f"{prefix}.self_attn.k_proj")
    v = model._linear(x_normed, f"{prefix}.self_attn.v_proj")
    q = model._split_heads(q, n_head)
    k = model._split_heads(k, n_head)
    v = model._split_heads(v, n_head)

    # Non-causal (current code)
    out_noncausal = gpu.attention(q, k, v)
    out_noncausal = model._merge_heads(out_noncausal, n_head)
    out_noncausal = model._linear(out_noncausal, f"{prefix}.self_attn.out_proj")
    out_noncausal.eval()
    noncausal_np = np.array(out_noncausal.to_list()).reshape(out_noncausal.shape)

    # Causal (should be used)
    out_causal = gpu.attention_causal(q, k, v)
    out_causal = model._merge_heads(out_causal, n_head)
    out_causal = model._linear(out_causal, f"{prefix}.self_attn.out_proj")
    out_causal.eval()
    causal_np = np.array(out_causal.to_list()).reshape(out_causal.shape)

    diff = np.abs(noncausal_np - causal_np)
    print(f"\n  Layer 0 self-attn output difference (causal vs non-causal):")
    print(f"    max_abs_diff: {diff.max():.6f}")
    print(f"    mean_abs_diff: {diff.mean():.6f}")

    # Check per-position — position 0 should differ most, last position should match
    for pos in range(seq_len):
        pos_diff = np.abs(noncausal_np[0, pos] - causal_np[0, pos]).max()
        print(f"    position {pos} max_diff: {pos_diff:.6f}")

    if diff.max() > 1e-5:
        print(f"\n  >>> CONFIRMED: Causal masking significantly affects decoder self-attention output")
        print(f"  >>> This is the likely root cause of wrong logits")
    else:
        print(f"\n  Causal masking does NOT affect output (unexpected for seq_len > 1)")


# ─── Positional embedding comparison ──────────────────────────────────────

def compare_positional_embeddings(hf, model, tokens):
    """Check if our positional embedding matches HF's learned embeddings."""
    import applegpu_runtime as gpu

    print("\n" + "="*70)
    print("POSITIONAL EMBEDDING COMPARISON")
    print("="*70)

    dec = hf.decoder
    tokens_t = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        # HF uses a learned positional embedding lookup
        ref_pos = dec.embed_positions(tokens_t).numpy()

    # Ours: slice from weight
    w = model.weights
    seq_len = len(tokens)
    pos_emb = gpu.slice(w["decoder.embed_positions.weight"], 0, 0, seq_len)
    pos_emb.eval()
    ours_pos = np.array(pos_emb.to_list()).reshape(pos_emb.shape)

    # HF embed_positions may have an offset -- check shapes
    print(f"  HF pos_emb shape: {ref_pos.shape}")
    print(f"  Our pos_emb shape: {ours_pos.shape}")

    # HF WhisperDecoder.embed_positions is WhisperPositionalEmbedding which inherits from nn.Embedding
    # It has an internal offset of 0 for Whisper, so positions 0..seq_len map to indices 0..seq_len
    # But our weights key is "decoder.embed_positions.weight" — check if HF stores it the same way
    hf_pos_weight = hf.state_dict()["decoder.embed_positions.weight"].numpy()
    print(f"  HF embed_positions.weight shape: {hf_pos_weight.shape}")

    # Compare our stored weight slice vs HF's forward output
    compare("dec_pos_emb (slice vs HF forward)", ours_pos, ref_pos[0:1, :, :] if ref_pos.ndim == 3 else ref_pos)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    import applegpu_runtime as gpu
    from applegpu_runtime.models.whisper import WhisperModel

    print("Loading models...")
    hf = load_reference()
    model = WhisperModel(MODEL_NAME)

    mel = make_test_mel()
    mel_torch = torch.from_numpy(mel)
    mel_gpu = gpu.from_numpy(mel)

    # Prefix tokens for English transcription
    tokens = [50258, 50259, 50359, 50363]  # <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ENCODER COMPARISON")
    print("="*70)
    ref_enc = ref_encoder_stepwise(hf, mel_torch)
    our_enc, encoder_output_gpu = our_encoder_stepwise(model, mel_gpu)

    for key in ref_enc:
        if key in our_enc:
            compare(key, our_enc[key], ref_enc[key])

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DECODER COMPARISON")
    print("="*70)

    # Use HF encoder output for decoder comparison (isolates decoder bugs)
    encoder_output_torch = torch.from_numpy(ref_enc["enc_output"])
    encoder_output_for_dec = gpu.from_numpy(ref_enc["enc_output"])

    ref_dec = ref_decoder_stepwise(hf, encoder_output_torch, tokens)
    our_dec = our_decoder_stepwise(model, encoder_output_for_dec, tokens)

    for key in ref_dec:
        if key in our_dec:
            compare(key, our_dec[key], ref_dec[key])

    # ═══════════════════════════════════════════════════════════════════════
    # Focused diagnostics
    compare_positional_embeddings(hf, model, tokens)
    compare_self_attention_causal_vs_noncausal(
        hf, model, encoder_output_torch, encoder_output_for_dec, tokens
    )

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("LOGIT ANALYSIS")
    print("="*70)

    if "logits" in ref_dec and "logits" in our_dec:
        ref_logits = ref_dec["logits"]
        our_logits = our_dec["logits"]
        # Check top-5 tokens at last position
        for label, logits in [("REF", ref_logits), ("OURS", our_logits)]:
            last = logits[0, -1, :]
            top5 = np.argsort(last)[-5:][::-1]
            print(f"  {label} top-5 tokens at last position: {list(top5)}")
            print(f"    values: {[f'{last[i]:.4f}' for i in top5]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
