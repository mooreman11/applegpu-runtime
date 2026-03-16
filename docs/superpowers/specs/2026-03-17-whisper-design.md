# Whisper Inference Design

**Date:** 2026-03-17
**Status:** Approved
**Scope:** Native Whisper speech-to-text inference on Metal GPU. Transcription, translation, language detection. Tiny + base models. Greedy decoding.

## Overview

Add Whisper (OpenAI's speech-to-text model) as a native `applegpu_runtime` model, following the GPT-2 pattern. Audio in → text out, running entirely on Metal GPU except for mel spectrogram preprocessing (CPU, one-time cost).

## Architecture

```
Audio file → ffmpeg → 16kHz mono → mel spectrogram (CPU/numpy)
                                          ↓
                     ┌─────────── AudioEncoder ──────────────┐
                     │ Conv1d(80→n_state, k=3, pad=1) + GELU │
                     │ Conv1d(n_state→n_state, k=3, s=2, p=1)│
                     │   + GELU + positional embedding        │
                     │ N × ResidualAttentionBlock              │
                     │ LayerNorm                               │
                     └────────────── encoder_output ──────────┘
                                          ↓
                     ┌─────────── TextDecoder ───────────────┐
                     │ token_embedding + positional_embedding │
                     │ N × ResidualAttentionBlock              │
                     │   (self-attention + cross-attention)    │
                     │ LayerNorm → logits via weight tying     │
                     └────────────── next token ──────────────┘
                                          ↓
                     Greedy decode with forced prefix → text
```

## Model Sizes

| Model | Params | Enc Layers | Dec Layers | Width | Heads |
|-------|--------|-----------|-----------|-------|-------|
| tiny | 39M | 4 | 4 | 384 | 6 |
| base | 74M | 6 | 6 | 512 | 8 |

## Ops Used (all already implemented)

| Op | Usage |
|----|-------|
| conv1d | Audio encoder: 2 conv layers with padding/stride |
| gelu | Activation after conv and in FFN |
| add_bias | Conv bias (N-D), linear bias |
| layer_norm | Pre-attention and pre-FFN normalization |
| matmul | Linear layers, attention QKV projections |
| softmax | Attention weights |
| log_softmax | Token probability for greedy decoding |
| sin/cos | Sinusoidal positional embeddings (precomputed) |
| embedding | Token embedding lookup |
| attention | Self-attention (causal in decoder) and cross-attention (encoder→decoder) |
| attention_causal | Decoder self-attention with causal mask |
| transpose | Attention head reshaping |
| reshape | Tensor shape manipulation |
| add | Residual connections |
| scalar_mul | Attention scaling |
| argmax | Greedy token selection |

## Components

### 1. `python/applegpu_runtime/models/whisper.py`

Native Whisper model implementation:

```python
class WhisperModel:
    def __init__(self, model_name="tiny"):
        # Load weights from HuggingFace or OpenAI checkpoint
        # Create GPU tensors for all parameters

    def encode(self, mel):
        # mel: [1, 80, 3000] → encoder_output: [1, 1500, n_state]
        # Conv1d → GELU → Conv1d → GELU → positional_embedding
        # N × ResidualAttentionBlock (self-attention only)
        # LayerNorm

    def decode_step(self, tokens, encoder_output, kv_cache):
        # tokens: [1, seq_len] → logits: [1, seq_len, vocab_size]
        # token_embedding + positional_embedding
        # N × ResidualAttentionBlock (self-attn + cross-attn)
        #   - Self-attention KV cache grows each step
        #   - Cross-attention KV cache is STATIC (computed once from encoder_output)
        # LayerNorm → logits (weight-tied with token_embedding)

    def transcribe(self, audio_path, language=None, task="transcribe"):
        # 1. Load audio via ffmpeg → 16kHz mono
        # 2. Compute mel spectrogram (reuse whisper.audio)
        # 3. Encode audio
        # 4. Greedy decode with forced prefix tokens
        # 5. Return text
```

### 2. KV Cache Design

Two types of KV cache per decoder layer:

- **Self-attention cache**: grows each decode step (append new K/V)
- **Cross-attention cache**: computed ONCE from encoder output, reused for all steps

```python
class KVCache:
    def __init__(self, n_layers):
        self.self_attn = [{"k": None, "v": None} for _ in range(n_layers)]
        self.cross_attn = [{"k": None, "v": None} for _ in range(n_layers)]

    def get_cross_attn(self, layer):
        return self.cross_attn[layer]  # Static after first step

    def update_self_attn(self, layer, new_k, new_v):
        # Concat with existing cache
```

### 3. Decoding Logic

Whisper uses forced prefix tokens + greedy decode:

```python
# Forced prefix (always injected, not predicted):
# <|startoftranscript|> <|lang|> <|task|> <|notimestamps|>

SPECIAL_TOKENS = {
    "startoftranscript": 50258,
    "translate": 50358,
    "transcribe": 50359,
    "notimestamps": 50363,
    "endoftext": 50257,
    # Language tokens: 50259-50357 (99 languages)
}

def greedy_decode(model, encoder_output, language="en", task="transcribe"):
    tokens = [50258]  # <|startoftranscript|>
    tokens.append(50259 + LANGUAGE_CODES.index(language))  # <|lang|>
    tokens.append(50359 if task == "transcribe" else 50358)  # <|task|>
    tokens.append(50363)  # <|notimestamps|>

    kv_cache = KVCache(n_layers)

    for _ in range(max_tokens):
        logits = model.decode_step(tokens, encoder_output, kv_cache)
        next_token = argmax(logits[:, -1, :])
        if next_token == 50257:  # <|endoftext|>
            break
        tokens.append(next_token)

    return tokenizer.decode(tokens[4:])  # Skip prefix
```

### 4. Audio Preprocessing

Reuse OpenAI Whisper's `audio.py`:
- Load audio via ffmpeg → 16kHz mono numpy array
- Pad/trim to 30 seconds
- Compute log-mel spectrogram (80 mel bins, 128 STFT bins)
- Result: `[1, 80, 3000]` tensor

This runs on CPU (numpy FFT) — one-time cost per audio file, milliseconds.

### 5. Weight Loading

Load from HuggingFace `openai/whisper-tiny` or `openai/whisper-base`:

```python
from transformers import WhisperModel as HFWhisper

hf_model = HFWhisper.from_pretrained("openai/whisper-tiny")
# Map HF state dict keys to our parameter names
# Convert to GPU tensors
```

### 6. `examples/whisper_transcribe.py`

```python
import applegpu_runtime as gpu

model = gpu.load_model("whisper-tiny")
text = model.transcribe("audio.wav")
print(text)

# With options:
text = model.transcribe("audio.wav", language="es", task="translate")
```

### 7. Language Detection

```python
def detect_language(model, mel):
    encoder_output = model.encode(mel)
    # Decode just the language token position
    tokens = [50258]  # <|startoftranscript|>
    logits = model.decode_step(tokens, encoder_output, KVCache(n_layers))
    # Language tokens are 50259-50357
    lang_logits = logits[0, -1, 50259:50358]
    lang_idx = argmax(lang_logits)
    return LANGUAGE_CODES[lang_idx]
```

## Testing Strategy

- **Unit tests**: encode produces correct shape, decode_step produces logits, KV cache grows correctly
- **Integration test**: transcribe a short WAV file, verify output is non-empty English text
- **Cross-attention test**: already verified (`q_len != kv_len` works)
- **Performance**: measure tokens/sec for tiny and base models

## Not Included (v1)

- Beam search (greedy only)
- Word-level alignment / timestamps
- Streaming/real-time transcription
- PyTorch device backend path (separate PR)
- `gpu.run_model("whisper", ...)` high-level API (build model first)
- Fine-tuning
