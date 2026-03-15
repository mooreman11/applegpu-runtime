"""Tests for top-k/top-p sampling."""

import numpy as np
import pytest
from applegpu_runtime.models.generate import _sample_token


def test_greedy_sampling():
    """temperature=0 returns argmax."""
    logits = np.array([1.0, 5.0, 2.0, 3.0])
    assert _sample_token(logits, temperature=0) == 1


def test_temperature_affects_distribution():
    """Lower temperature makes distribution peakier."""
    logits = np.array([1.0, 2.0, 3.0])
    np.random.seed(42)

    # Low temp: almost always picks the max
    counts = [0, 0, 0]
    for _ in range(100):
        idx = _sample_token(logits, temperature=0.1, top_k=0, top_p=0.0)
        counts[idx] += 1
    assert counts[2] > 90  # token 2 (highest logit) dominates


def test_top_k_limits_candidates():
    """top_k=1 should always return the top token."""
    logits = np.array([1.0, 5.0, 2.0, 3.0])
    for _ in range(20):
        assert _sample_token(logits, temperature=1.0, top_k=1, top_p=0.0) == 1


def test_top_p_nucleus():
    """top_p filters to smallest set with cumulative prob >= p."""
    # With very peaked logits and low top_p, should mostly pick the top
    logits = np.array([0.0, 0.0, 0.0, 10.0])
    for _ in range(20):
        idx = _sample_token(logits, temperature=1.0, top_k=0, top_p=0.5)
        assert idx == 3  # token 3 has >99% prob, so top_p=0.5 always selects it


def test_top_k_and_top_p_combined():
    """Both filters applied together."""
    logits = np.array([1.0, 2.0, 10.0, 3.0, 0.5])
    # top_k=3 keeps indices 1,2,3. top_p=0.5 further filters.
    for _ in range(20):
        idx = _sample_token(logits, temperature=1.0, top_k=3, top_p=0.5)
        assert idx in [1, 2, 3]


def test_sampling_reproducible_with_seed():
    """Same seed produces same sequence."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.random.seed(123)
    seq1 = [_sample_token(logits, temperature=0.8, top_k=3) for _ in range(10)]
    np.random.seed(123)
    seq2 = [_sample_token(logits, temperature=0.8, top_k=3) for _ in range(10)]
    assert seq1 == seq2


def test_uniform_logits_samples_uniformly():
    """Equal logits should sample roughly uniformly."""
    logits = np.zeros(4)
    np.random.seed(42)
    counts = [0] * 4
    for _ in range(1000):
        idx = _sample_token(logits, temperature=1.0, top_k=0, top_p=0.0)
        counts[idx] += 1
    # Each should be ~250 ± 50
    for c in counts:
        assert 150 < c < 350
