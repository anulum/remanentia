# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real numerical retrieval-spiking tests

"""Exercise production encoding and deterministic LIF math on concrete arrays."""

from __future__ import annotations

import numpy as np
import pytest

from retrieval_spiking import (
    cosine_similarity,
    cosine_similarity_python,
    encode_text,
    hash_encode_python,
    snn_affinity,
    snn_affinity_python,
    spike_feature,
    spike_feature_python,
)


def test_portable_hash_encoding_is_deterministic_bounded_and_content_sensitive() -> None:
    """Real unigram and bigram inputs produce stable bounded activation patterns."""
    first = hash_encode_python("tokamak disruption control", 256)
    repeated = hash_encode_python("tokamak disruption control", 256)
    different = hash_encode_python("bread fermentation recipe", 256)
    empty = hash_encode_python("the and or", 256)

    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (256,)
    assert first.dtype == np.float64
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0
    assert np.count_nonzero(first) > 0
    assert not np.array_equal(first, different)
    assert np.count_nonzero(empty) == 0


def test_hash_encoding_saturates_repeated_collisions() -> None:
    """Repeated terms cannot push a neuron beyond the declared activation ceiling."""
    encoded = hash_encode_python(" ".join(["memory"] * 20), 8)

    assert float(encoded.max()) == 1.0
    assert float(encoded.min()) >= 0.0


def test_active_encoding_backend_matches_production_encoding_module() -> None:
    """The retrieval facade traverses the actual active encoding.py backend."""
    import encoding

    original_backend = encoding.get_backend()
    encoding.set_backend("hash")
    try:
        expected = encoding.encode_text("persistent memory", 128)
        actual = encode_text("persistent memory", 128)
    finally:
        encoding.set_backend(original_backend)

    np.testing.assert_array_equal(actual, expected)


def test_portable_cosine_similarity_handles_geometry_and_both_zero_positions() -> None:
    """Concrete vectors cover identical, orthogonal, opposite and zero-norm geometry."""
    x = np.array([1.0, 0.0], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    zero = np.zeros(2, dtype=np.float64)

    assert cosine_similarity_python(x, x) == pytest.approx(1.0)
    assert cosine_similarity_python(x, y) == pytest.approx(0.0)
    assert cosine_similarity_python(x, -x) == pytest.approx(-1.0)
    assert cosine_similarity_python(zero, x) == 0.0
    assert cosine_similarity_python(x, zero) == 0.0
    assert cosine_similarity(x, y) == pytest.approx(cosine_similarity_python(x, y))


def test_portable_spike_feature_is_deterministic_and_respects_step_count() -> None:
    """The real LIF recurrence is deterministic and zero steps emit no spikes."""
    weights = np.eye(24, dtype=np.float64) * 0.05
    stimulus = np.linspace(0.4, 1.0, 24, dtype=np.float64)

    zero_steps = spike_feature_python(weights, stimulus, steps=0)
    short = spike_feature_python(weights, stimulus, steps=10)
    repeated = spike_feature_python(weights, stimulus, steps=10)
    long = spike_feature_python(weights, stimulus, steps=100)

    assert zero_steps.shape == (24,)
    assert zero_steps.dtype == np.float32
    assert float(zero_steps.sum()) == 0.0
    np.testing.assert_array_equal(short, repeated)
    assert float(long.sum()) >= float(short.sum())
    assert float(long.sum()) > 0.0


def test_spike_dispatch_matches_portable_surface_without_native_wheel() -> None:
    """Runtime dispatch preserves the concrete portable recurrence on this interpreter."""
    weights = np.zeros((12, 12), dtype=np.float64)
    stimulus = np.linspace(0.0, 1.0, 12, dtype=np.float64)

    np.testing.assert_array_equal(
        spike_feature(weights, stimulus, steps=25),
        spike_feature_python(weights, stimulus, steps=25),
    )


def test_snn_affinity_uses_real_spike_features_and_is_bounded() -> None:
    """Identical and distinct stimuli are compared through the complete LIF feature path."""
    weights = np.random.default_rng(42).normal(0.0, 0.03, (32, 32))
    query = np.linspace(0.2, 1.0, 32, dtype=np.float64)
    other = np.linspace(1.0, 0.1, 32, dtype=np.float64)

    identical = snn_affinity_python(weights, query, query)
    distinct = snn_affinity_python(weights, query, other)

    assert identical == pytest.approx(1.0)
    assert -1.0 <= distinct <= 1.0
    assert snn_affinity(weights, query, other) == pytest.approx(distinct)
