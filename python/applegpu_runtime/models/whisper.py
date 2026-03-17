"""Whisper speech-to-text model running on Metal GPU.

Supports whisper-tiny and whisper-base models with greedy decoding,
language detection, and translation.
"""
import numpy as np
import applegpu_runtime as gpu

# Whisper model dimensions
WHISPER_CONFIGS = {
    "tiny":  {"n_mels": 80, "n_ctx": 1500, "n_state": 384, "n_head": 6, "n_layer": 4, "n_vocab": 51865},
    "base":  {"n_mels": 80, "n_ctx": 1500, "n_state": 512, "n_head": 8, "n_layer": 6, "n_vocab": 51865},
}


class KVCache:
    """Dual KV cache for Whisper decoder.

    Self-attention cache grows each step. Cross-attention cache is computed
    once from encoder output and reused for all subsequent steps.
    """

    def __init__(self, n_layers):
        self.self_attn = [{"k": None, "v": None} for _ in range(n_layers)]
        self.cross_attn = [{"k": None, "v": None} for _ in range(n_layers)]
        self.offset = 0

    def advance(self, n_tokens):
        self.offset += n_tokens


class WhisperModel:
    def __init__(self, model_name="tiny"):
        gpu.init_backend()
        if model_name not in WHISPER_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(WHISPER_CONFIGS.keys())}")
        self.config = WHISPER_CONFIGS[model_name]
        self.model_name = model_name

        # Whisper creates many weight tensors + intermediates during generation
        gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)

        self._load_weights(model_name)

    def _load_weights(self, model_name):
        """Load weights from HuggingFace openai/whisper-{model_name}."""
        from transformers import WhisperModel as HFWhisper

        hf = HFWhisper.from_pretrained(f"openai/whisper-{model_name}")
        sd = hf.state_dict()

        self.weights = {}
        for key, tensor in sd.items():
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            self.weights[key] = gpu.from_numpy(arr)

        # Precompute sinusoidal positional embeddings for encoder
        n_ctx = self.config["n_ctx"]
        n_state = self.config["n_state"]
        pos_emb = self._sinusoids(n_ctx, n_state)
        self.weights["encoder.positional_embedding"] = gpu.from_numpy(pos_emb)

    @staticmethod
    def _sinusoids(length, channels):
        """Generate sinusoidal positional embeddings."""
        pos = np.arange(length, dtype=np.float32)[:, np.newaxis]
        dim = np.arange(0, channels, 2, dtype=np.float32)[np.newaxis, :]
        freqs = pos / (10000 ** (dim / channels))
        emb = np.zeros((length, channels), dtype=np.float32)
        emb[:, 0::2] = np.sin(freqs)
        emb[:, 1::2] = np.cos(freqs)
        return emb

    # ------------------------------------------------------------------ #
    # Audio encoder
    # ------------------------------------------------------------------ #

    def encode(self, mel):
        """Encode mel spectrogram to encoder output.

        Args:
            mel: gpu tensor [1, 80, n_frames] (padded to 3000 frames)

        Returns:
            encoder_output: gpu tensor [1, n_ctx, n_state]
        """
        n_head = self.config["n_head"]
        n_layer = self.config["n_layer"]
        w = self.weights

        # Conv1d layers
        x = gpu.conv1d(mel, w["encoder.conv1.weight"], stride=1, padding=1)
        x = gpu.add_bias(x, w["encoder.conv1.bias"])
        x = gpu.gelu(x)

        x = gpu.conv1d(x, w["encoder.conv2.weight"], stride=2, padding=1)
        x = gpu.add_bias(x, w["encoder.conv2.bias"])
        x = gpu.gelu(x)

        # x is [1, n_state, n_ctx] -- transpose to [1, n_ctx, n_state]
        x = gpu.transpose_dims(x, 1, 2)

        # Add positional embedding
        x = x + w["encoder.positional_embedding"]

        # Encoder blocks
        for i in range(n_layer):
            prefix = f"encoder.layers.{i}"
            x = self._encoder_block(x, prefix, n_head)

        # Final layer norm
        x = self._layer_norm(x, "encoder.layer_norm")
        return x

    def _encoder_block(self, x, prefix, n_head):
        """Residual attention block for encoder (self-attention only)."""
        # Self-attention
        residual = x
        x = self._layer_norm(x, f"{prefix}.self_attn_layer_norm")
        x = self._self_attention(x, f"{prefix}.self_attn", n_head)
        x = residual + x

        # FFN
        residual = x
        x = self._layer_norm(x, f"{prefix}.final_layer_norm")
        x = self._ffn(x, prefix)
        x = residual + x
        return x

    # ------------------------------------------------------------------ #
    # Text decoder
    # ------------------------------------------------------------------ #

    def decode_step(self, tokens, encoder_output, kv_cache):
        """Decode one step.

        Args:
            tokens: list of token IDs (or gpu tensor [1, seq_len])
            encoder_output: gpu tensor [1, n_ctx, n_state]
            kv_cache: KVCache object

        Returns:
            logits: gpu tensor [1, seq_len, n_vocab]
        """
        n_head = self.config["n_head"]
        n_layer = self.config["n_layer"]
        w = self.weights

        if isinstance(tokens, list):
            tokens_arr = np.array([tokens], dtype=np.int32)
            x_tokens = gpu.from_numpy(tokens_arr)
        else:
            x_tokens = tokens

        # Token embedding + positional embedding
        x = gpu.embedding(w["decoder.embed_tokens.weight"], x_tokens)
        seq_len = x.shape[1]

        # Slice positional embedding to correct range
        offset = kv_cache.offset if kv_cache else 0
        pos_emb = gpu.slice(w["decoder.embed_positions.weight"], 0, offset, offset + seq_len)
        x = x + pos_emb

        # Decoder blocks
        for i in range(n_layer):
            prefix = f"decoder.layers.{i}"
            x = self._decoder_block(x, encoder_output, prefix, n_head, kv_cache, i)

        # Final layer norm
        x = self._layer_norm(x, "decoder.layer_norm")

        # Logits via weight tying
        logits = gpu.matmul(x, gpu.transpose(w["decoder.embed_tokens.weight"]))
        return logits

    def _decoder_block(self, x, encoder_output, prefix, n_head, kv_cache, layer_idx):
        """Residual attention block for decoder (self-attn + cross-attn + FFN)."""
        # Self-attention (causal)
        residual = x
        x = self._layer_norm(x, f"{prefix}.self_attn_layer_norm")
        self_cache = kv_cache.self_attn[layer_idx] if kv_cache else None
        x = self._self_attention(x, f"{prefix}.self_attn", n_head, kv_cache=self_cache)
        x = residual + x

        # Cross-attention
        residual = x
        x = self._layer_norm(x, f"{prefix}.encoder_attn_layer_norm")
        cross_cache = kv_cache.cross_attn[layer_idx] if kv_cache else None
        x = self._cross_attention(x, encoder_output, f"{prefix}.encoder_attn", n_head, cross_cache)
        x = residual + x

        # FFN
        residual = x
        x = self._layer_norm(x, f"{prefix}.final_layer_norm")
        x = self._ffn(x, prefix)
        x = residual + x
        return x

    # ------------------------------------------------------------------ #
    # Greedy decoding + transcription
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_audio(audio_path):
        """Load audio from file. Falls back to wave module if ffmpeg unavailable."""
        import torch
        try:
            import whisper
            audio = whisper.load_audio(audio_path)
            return torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
        except FileNotFoundError:
            # ffmpeg not found — try wave module for WAV files
            import wave
            with wave.open(audio_path, 'r') as f:
                frames = f.readframes(f.getnframes())
                sr = f.getframerate()
                sw = f.getsampwidth()
            if sw == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sw}")
            # Resample to 16kHz if needed
            if sr != 16000:
                ratio = 16000 / sr
                n = int(len(audio) * ratio)
                audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio).astype(np.float32)
            return torch.from_numpy(audio)

    def transcribe(self, audio_path, language=None, task="transcribe", max_tokens=224):
        """Transcribe an audio file.

        Args:
            audio_path: path to audio file (any format ffmpeg supports, or .wav without ffmpeg)
            language: language code (e.g., "en") or None for auto-detect
            task: "transcribe" or "translate" (translate to English)
            max_tokens: maximum tokens to generate

        Returns:
            text: transcribed text string
        """
        import whisper

        # 1. Audio preprocessing (CPU)
        audio = self._load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).numpy()
        mel_gpu = gpu.from_numpy(mel)

        # 2. Encode
        encoder_output = self.encode(mel_gpu)

        # 3. Detect language if needed
        if language is None:
            language = self.detect_language(encoder_output)

        # 4. Build prefix tokens
        tokenizer = whisper.tokenizer.get_tokenizer(False, language=language, task=task)
        prefix = tokenizer.sot_sequence_including_notimestamps

        # 5. Greedy decode
        tokens = list(prefix)
        kv_cache = KVCache(self.config["n_layer"])

        # First pass with full prefix
        logits = self.decode_step(tokens, encoder_output, kv_cache)
        kv_cache.advance(len(tokens))

        # Special tokens to suppress during generation (all tokens >= 50257)
        eot_token = 50257
        suppress_above = eot_token  # suppress all special tokens

        for _ in range(max_tokens):
            # Get logits for last position, suppress special tokens on CPU
            # Slice to last token's logits: [1, 1, vocab] → flatten
            last_logit = gpu.slice(logits, 1, logits.shape[1] - 1, logits.shape[1])
            last_logit = gpu.reshape(last_logit, [self.config["n_vocab"]])
            last_logit.eval()
            last_logits = last_logit.to_list()  # flat [vocab_size]

            # Suppress special tokens (except EOT which we need to stop)
            eot_val = last_logits[eot_token]
            for i in range(suppress_above, len(last_logits)):
                last_logits[i] = float("-inf")
            last_logits[eot_token] = eot_val

            token_id = max(range(len(last_logits)), key=lambda i: last_logits[i])

            if token_id == eot_token:
                break

            tokens.append(token_id)

            # Next step: only feed the new token
            logits = self.decode_step([token_id], encoder_output, kv_cache)
            kv_cache.advance(1)

        # 6. Decode tokens to text
        text = tokenizer.decode(tokens[len(prefix):])
        return text.strip()

    def detect_language(self, encoder_output):
        """Detect language from encoder output.

        Args:
            encoder_output: gpu tensor [1, n_ctx, n_state]

        Returns:
            language code string (e.g., "en")
        """
        kv_cache = KVCache(self.config["n_layer"])
        logits = self.decode_step([50258], encoder_output, kv_cache)
        # logits shape: [1, 1, n_vocab] — slice to get last token logits
        n_vocab = self.config["n_vocab"]
        last_logits = gpu.slice(logits, 1, logits.shape[1] - 1, logits.shape[1])
        last_logits = gpu.reshape(last_logits, [1, n_vocab])
        last_logits.eval()

        # to_list() returns flat list; shape is [1, n_vocab]
        flat = last_logits.to_list()
        # Language tokens are 50259-50357
        lang_logits = flat[50259:50358]
        lang_idx = max(range(len(lang_logits)), key=lambda i: lang_logits[i])

        import whisper.tokenizer
        languages = list(whisper.tokenizer.LANGUAGES.keys())
        return languages[lang_idx]

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def _layer_norm(self, x, prefix):
        gamma = self.weights[f"{prefix}.weight"]
        beta = self.weights[f"{prefix}.bias"]
        return gpu.layer_norm(x, gamma, beta, 1e-5)

    def _linear(self, x, prefix):
        """x @ weight.T + bias

        For 3D input [batch, seq, in_features], output is [batch, seq, out_features].
        Bias is reshaped to [1, 1, out_features] for broadcasting.
        """
        weight = self.weights[f"{prefix}.weight"]
        bias = self.weights.get(f"{prefix}.bias")
        # weight is [out_features, in_features], need x @ weight.T
        out = gpu.matmul(x, gpu.transpose(weight))
        if bias is not None:
            # Reshape bias [out_features] -> [1, 1, out_features] for 3D broadcast
            ndim = len(out.shape)
            if ndim == 3:
                bias_shape = [1, 1, out.shape[2]]
            elif ndim == 2:
                bias_shape = [1, out.shape[1]]
            else:
                bias_shape = list(bias.shape)
            out = out + gpu.reshape(bias, bias_shape)
        return out

    def _self_attention(self, x, prefix, n_head, kv_cache=None):
        """Multi-head self-attention."""
        # QKV projection
        q = self._linear(x, f"{prefix}.q_proj")
        k = self._linear(x, f"{prefix}.k_proj")
        v = self._linear(x, f"{prefix}.v_proj")

        # Reshape to [batch, n_head, seq_len, d_head]
        q = self._split_heads(q, n_head)
        k = self._split_heads(k, n_head)
        v = self._split_heads(v, n_head)

        # KV cache for decoder
        if kv_cache is not None:
            if kv_cache["k"] is not None:
                k = gpu.concat(kv_cache["k"], k, dim=2)
                v = gpu.concat(kv_cache["v"], v, dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Attention
        out = gpu.attention(q, k, v)

        # Merge heads: [batch, n_head, seq_len, d_head] -> [batch, seq, n_state]
        out = self._merge_heads(out, n_head)

        # Output projection
        out = self._linear(out, f"{prefix}.out_proj")
        return out

    def _cross_attention(self, x, encoder_output, prefix, n_head, kv_cache):
        """Multi-head cross-attention (Q from decoder, KV from encoder)."""
        q = self._linear(x, f"{prefix}.q_proj")
        q = self._split_heads(q, n_head)

        # KV from encoder -- compute once, cache forever
        if kv_cache["k"] is None:
            k = self._linear(encoder_output, f"{prefix}.k_proj")
            v = self._linear(encoder_output, f"{prefix}.v_proj")
            k = self._split_heads(k, n_head)
            v = self._split_heads(v, n_head)
            kv_cache["k"] = k
            kv_cache["v"] = v
        else:
            k = kv_cache["k"]
            v = kv_cache["v"]

        out = gpu.attention(q, k, v)  # cross-attention: q_len != kv_len
        out = self._merge_heads(out, n_head)
        out = self._linear(out, f"{prefix}.out_proj")
        return out

    def _ffn(self, x, prefix):
        x = self._linear(x, f"{prefix}.fc1")
        x = gpu.gelu(x)
        x = self._linear(x, f"{prefix}.fc2")
        return x

    def _split_heads(self, x, n_head):
        """[batch, seq, n_state] -> [batch, n_head, seq, d_head]"""
        shape = x.shape
        batch, seq_len = shape[0], shape[1]
        d_head = shape[2] // n_head
        x = gpu.reshape(x, [batch, seq_len, n_head, d_head])
        x = gpu.transpose_dims(x, 1, 2)  # [batch, n_head, seq, d_head]
        return x

    def _merge_heads(self, x, n_head):
        """[batch, n_head, seq, d_head] -> [batch, seq, n_state]"""
        shape = x.shape
        batch, seq_len, d_head = shape[0], shape[2], shape[3]
        x = gpu.transpose_dims(x, 1, 2)  # [batch, seq, n_head, d_head]
        x = gpu.reshape(x, [batch, seq_len, n_head * d_head])
        return x
