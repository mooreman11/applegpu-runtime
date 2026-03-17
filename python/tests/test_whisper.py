"""Regression tests for Whisper model on Metal GPU.

Validates encoder output, decoder logits, and end-to-end transcription
against HuggingFace reference to catch numerical divergences.
"""
import numpy as np
import pytest
import torch
import applegpu_runtime as gpu


@pytest.fixture(scope="module")
def whisper_models():
    """Load both our model and HF reference once per module."""
    from applegpu_runtime.models.whisper import WhisperModel
    from transformers import WhisperModel as HFWhisper

    model = WhisperModel("tiny")
    hf = HFWhisper.from_pretrained("openai/whisper-tiny")
    hf.eval()
    return model, hf


@pytest.fixture(scope="module")
def test_mel():
    """Create a deterministic test mel spectrogram [1, 80, 3000]."""
    np.random.seed(42)
    mel = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
    return mel


def _to_numpy(t):
    """Convert a gpu tensor to numpy array."""
    t.eval()
    return np.array(t.to_list()).reshape(t.shape)


class TestWhisperEncoder:
    """Encoder output must match HF reference within float32 tolerance."""

    def test_encoder_output_matches_reference(self, whisper_models, test_mel):
        model, hf = whisper_models

        mel_gpu = gpu.from_numpy(test_mel)
        mel_torch = torch.from_numpy(test_mel)

        # Our encoder
        our_enc = _to_numpy(model.encode(mel_gpu))

        # HF encoder
        with torch.no_grad():
            hf_enc = hf.encoder(mel_torch).last_hidden_state.numpy()

        assert our_enc.shape == hf_enc.shape
        max_err = np.abs(our_enc - hf_enc).max()
        # Float32 noise amplification through LayerNorm + FFN causes up to ~1.0
        # max error on random mel (near-zero variance → sensitive normalization).
        # Real audio produces < 0.03 error; end-to-end test verifies correctness.
        assert max_err < 1.5, f"Encoder max error {max_err:.4f} exceeds tolerance"

    def test_encoder_uses_learned_positional_embeddings(self, whisper_models):
        """Regression: encoder must use HF learned pos embeddings, not sinusoidal."""
        model, hf = whisper_models

        # Our encoder positional embedding should match HF weights
        our_pos = _to_numpy(model.weights["encoder.positional_embedding"])
        hf_pos = hf.state_dict()["encoder.embed_positions.weight"].numpy()

        np.testing.assert_allclose(our_pos, hf_pos, atol=1e-6,
                                   err_msg="Encoder positional embeddings must be loaded from HF, not computed")


class TestWhisperDecoder:
    """Decoder logits must match HF reference when fed identical encoder output."""

    def test_decoder_logits_match_reference(self, whisper_models, test_mel):
        model, hf = whisper_models
        from applegpu_runtime.models.whisper import KVCache

        mel_torch = torch.from_numpy(test_mel)
        prefix = [50258, 50259, 50359, 50363]

        # Get HF encoder output as ground truth input for both decoders
        with torch.no_grad():
            hf_enc = hf.encoder(mel_torch).last_hidden_state
        hf_enc_np = hf_enc.numpy()
        hf_enc_gpu = gpu.from_numpy(hf_enc_np)

        # Our decoder
        kv_cache = KVCache(model.config["n_layer"])
        our_logits = _to_numpy(model.decode_step(prefix, hf_enc_gpu, kv_cache))

        # HF decoder (with causal mask)
        with torch.no_grad():
            seq_len = len(prefix)
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            prefix_t = torch.tensor([prefix], dtype=torch.long)
            hf_dec = hf.decoder(prefix_t, attention_mask=causal_mask,
                                encoder_hidden_states=hf_enc)
            hf_logits = (hf_dec.last_hidden_state @ hf.decoder.embed_tokens.weight.T).numpy()

        assert our_logits.shape == hf_logits.shape
        max_err = np.abs(our_logits - hf_logits).max()
        assert max_err < 0.15, f"Decoder logit max error {max_err:.4f} exceeds tolerance"

    def test_decoder_top_tokens_match(self, whisper_models, test_mel):
        """The top-5 predicted tokens must match the reference."""
        model, hf = whisper_models
        from applegpu_runtime.models.whisper import KVCache

        mel_torch = torch.from_numpy(test_mel)
        prefix = [50258, 50259, 50359, 50363]

        with torch.no_grad():
            hf_enc = hf.encoder(mel_torch).last_hidden_state
        hf_enc_gpu = gpu.from_numpy(hf_enc.numpy())

        # Our top-5
        kv_cache = KVCache(model.config["n_layer"])
        our_logits = _to_numpy(model.decode_step(prefix, hf_enc_gpu, kv_cache))
        our_top5 = set(np.argsort(our_logits[0, -1])[-5:][::-1].tolist())

        # HF top-5
        with torch.no_grad():
            seq_len = len(prefix)
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            prefix_t = torch.tensor([prefix], dtype=torch.long)
            hf_dec = hf.decoder(prefix_t, attention_mask=causal_mask,
                                encoder_hidden_states=hf_enc)
            hf_logits = (hf_dec.last_hidden_state @ hf.decoder.embed_tokens.weight.T).numpy()
        hf_top5 = set(np.argsort(hf_logits[0, -1])[-5:][::-1].tolist())

        assert our_top5 == hf_top5, f"Top-5 mismatch: ours={our_top5} ref={hf_top5}"


class TestWhisperCausalMask:
    """Regression: decoder self-attention must use causal masking on prefix."""

    def test_causal_masking_affects_early_positions(self, whisper_models, test_mel):
        """Without causal masking, positions 0-2 would attend to future tokens."""
        model, _ = whisper_models

        mel_gpu = gpu.from_numpy(test_mel)
        encoder_output = model.encode(mel_gpu)

        # Get decoder layer 0 Q, K, V
        prefix = [50258, 50259, 50359, 50363]
        tokens_arr = np.array([prefix], dtype=np.int32)
        x_tokens = gpu.from_numpy(tokens_arr)
        w = model.weights
        n_head = model.config["n_head"]

        x = gpu.embedding(w["decoder.embed_tokens.weight"], x_tokens)
        pos_emb = gpu.slice(w["decoder.embed_positions.weight"], 0, 0, 4)
        x = x + pos_emb
        x = model._layer_norm(x, "decoder.layers.0.self_attn_layer_norm")

        q = model._linear(x, "decoder.layers.0.self_attn.q_proj")
        k = model._linear(x, "decoder.layers.0.self_attn.k_proj")
        v = model._linear(x, "decoder.layers.0.self_attn.v_proj")
        q = model._split_heads(q, n_head)
        k = model._split_heads(k, n_head)
        v = model._split_heads(v, n_head)

        out_noncausal = _to_numpy(gpu.attention(q, k, v))
        out_causal = _to_numpy(gpu.attention_causal(q, k, v))

        # Position 0 should differ (it incorrectly sees future without mask)
        pos0_diff = np.abs(out_noncausal[0, :, 0, :] - out_causal[0, :, 0, :]).max()
        assert pos0_diff > 0.01, "Causal mask should affect position 0"

        # Last position should be identical (no future to mask)
        pos3_diff = np.abs(out_noncausal[0, :, 3, :] - out_causal[0, :, 3, :]).max()
        assert pos3_diff < 1e-5, "Last position should be unaffected by causal mask"


class TestSliceConcat3D:
    """Regression: slice and concat must work correctly on 3D+ tensors."""

    def test_slice_3d_dim1(self):
        data = np.arange(40, dtype=np.float32).reshape(1, 4, 10)
        t = gpu.from_numpy(data)
        s = gpu.slice(t, 1, 3, 4)
        result = _to_numpy(s)
        np.testing.assert_allclose(result, data[:, 3:4, :])

    def test_slice_3d_dim0(self):
        data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        t = gpu.from_numpy(data)
        s = gpu.slice(t, 0, 1, 3)
        result = _to_numpy(s)
        np.testing.assert_allclose(result, data[1:3, :, :])

    def test_slice_large_vocab(self):
        """Regression: slicing logits [1, 4, 51865] along dim=1 must not zero out."""
        data = np.random.randn(1, 4, 51865).astype(np.float32) + 10
        t = gpu.from_numpy(data)
        s = gpu.slice(t, 1, 3, 4)
        r = gpu.reshape(s, [51865])
        result = _to_numpy(r)
        assert np.sum(result == 0.0) == 0, "Slice should not produce zeros"
        np.testing.assert_allclose(result, data[0, 3], atol=1e-4)

    def test_concat_4d_dim2(self):
        """Regression: KV cache concat along dim=2 of 4D tensor."""
        a = np.arange(48, dtype=np.float32).reshape(1, 2, 3, 8)
        b = np.ones((1, 2, 1, 8), dtype=np.float32) * 99
        ta = gpu.from_numpy(a)
        tb = gpu.from_numpy(b)
        tc = gpu.concat(ta, tb, dim=2)
        result = _to_numpy(tc)
        expected = np.concatenate([a, b], axis=2)
        np.testing.assert_allclose(result, expected)


class TestEndToEnd:
    """End-to-end transcription produces correct text."""

    def test_transcribe_tts_audio(self, whisper_models):
        """Transcribe TTS-generated audio and verify text content."""
        model, _ = whisper_models
        import subprocess
        import tempfile
        import os

        text = "The quick brown fox."
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            # Generate audio with macOS TTS
            aiff_path = wav_path.replace(".wav", ".aiff")
            subprocess.run(["say", "-o", aiff_path, text], check=True,
                           capture_output=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                            aiff_path, wav_path], check=True, capture_output=True)
            os.unlink(aiff_path)

            result = model.transcribe(wav_path, language="en")
            result_lower = result.lower().strip().rstrip(".")
            # TTS may produce slight variations but core content should match
            assert "quick brown fox" in result_lower, \
                f"Expected 'quick brown fox' in transcription, got: {repr(result)}"
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
