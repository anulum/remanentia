# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Deterministic retrieval encoding and spiking features

"""Encode retrieval text and compute deterministic LIF similarity features."""

from __future__ import annotations

import hashlib
from importlib import import_module
from typing import Any, Callable, TypeAlias, cast

import numpy as np
import numpy.typing as npt

try:
    from .retrieval_text import (  # type: ignore[import-not-found]
        STOPWORDS,
        python_bigrams,
        python_tokenize,
    )
except ImportError:
    from retrieval_text import STOPWORDS, python_bigrams, python_tokenize

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
HASH_PRIMES: tuple[int, ...] = (
    7919,
    104729,
    15485863,
    32452843,
    49979687,
    67867967,
    86028121,
)
_UINT64_MASK = (1 << 64) - 1
_XORSHIFT_SEED = 0x123456789ABCDEF0


def hash_encode_python(text: str, n_neurons: int) -> npt.NDArray[np.float64]:
    """Encode unigrams and bigrams into a deterministic bounded hash pattern."""
    pattern = np.zeros(n_neurons, dtype=np.float64)
    tokens = python_tokenize(text)
    for word in tokens:
        digest = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for prime in HASH_PRIMES:
            index = (digest + prime) % n_neurons
            pattern[index] = min(pattern[index] + 0.15, 1.0)

    for bigram in python_bigrams(tokens):
        digest = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
        for prime in HASH_PRIMES[:5]:
            index = (digest + prime) % n_neurons
            pattern[index] = min(pattern[index] + 0.25, 1.0)
    return pattern


def encode_text(text: str, n_neurons: int) -> FloatArray:
    """Encode with the active backend, falling back to deterministic hashing."""
    try:
        encoding_module = import_module("encoding")
    except ImportError:  # pragma: no cover - standalone install fallback
        try:  # pragma: no cover - optional native wheel
            from remanentia_retrieve import (  # type: ignore[import-not-found]
                hash_encode as native_encode,
            )
        except ImportError:  # pragma: no cover - portable standalone fallback
            return hash_encode_python(text, n_neurons)
        return cast(  # pragma: no cover - optional native wheel
            FloatArray,
            native_encode(text, n_neurons, list(HASH_PRIMES), STOPWORDS),
        )
    active_encode = cast(Callable[[str, int], FloatArray], encoding_module.encode_text)
    return active_encode(text, n_neurons)


def cosine_similarity_python(a: FloatArray, b: FloatArray) -> float:
    """Return cosine similarity, treating either zero vector as unrelated."""
    vector_a = a.astype(np.float64, copy=False)
    vector_b = b.astype(np.float64, copy=False)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def cosine_similarity(a: FloatArray, b: FloatArray) -> float:
    """Dispatch cosine similarity to the native wheel when installed."""
    try:  # pragma: no cover - optional native wheel
        from remanentia_retrieve import (
            cosine_sim as native_cosine,
        )
    except ImportError:
        return cosine_similarity_python(a, b)
    return float(  # pragma: no cover - optional native wheel
        native_cosine(a.astype(np.float64), b.astype(np.float64))
    )


def spike_feature_python(
    weights: FloatArray,
    stimulus: FloatArray,
    steps: int = 50,
) -> npt.NDArray[np.float32]:
    """Run a deterministic LIF burst and return per-neuron spike counts."""
    neuron_count = len(stimulus)
    voltage = np.empty(neuron_count, dtype=np.float64)
    rng_state = _XORSHIFT_SEED
    for index in range(neuron_count):
        rng_state ^= (rng_state << 13) & _UINT64_MASK
        rng_state ^= rng_state >> 7
        rng_state ^= (rng_state << 17) & _UINT64_MASK
        rng_state &= _UINT64_MASK
        voltage[index] = -70.0 + (rng_state / _UINT64_MASK) * 15.0
    external_current = 0.3 + stimulus * 2.0
    spike_count = np.zeros(neuron_count, dtype=np.float32)

    for _ in range(steps):
        fired = (voltage >= -55.0).astype(np.float32)
        synaptic_current = weights.dot(fired)
        voltage += (-(voltage + 65.0) / 10.0 + external_current + synaptic_current * 0.5)
        spiked = voltage >= -55.0
        spike_count += spiked.astype(np.float32)
        voltage[spiked] = -70.0
    return spike_count


def spike_feature(
    weights: FloatArray,
    stimulus: FloatArray,
    steps: int = 50,
) -> npt.NDArray[np.float32]:
    """Dispatch the deterministic spike feature to the native wheel when installed."""
    try:  # pragma: no cover - optional native wheel
        from remanentia_retrieve import (
            spike_feature as native_spike_feature,
        )
    except ImportError:
        return spike_feature_python(weights, stimulus, steps)
    return cast(  # pragma: no cover - optional native wheel
        npt.NDArray[np.float32],
        native_spike_feature(
            weights.astype(np.float64),
            stimulus.astype(np.float64),
            steps,
        ),
    )


def snn_affinity_python(
    weights: FloatArray,
    query_stimulus: FloatArray,
    trace_stimulus: FloatArray,
) -> float:
    """Compare portable deterministic query and trace spike-count features."""
    return cosine_similarity_python(
        spike_feature_python(weights, query_stimulus),
        spike_feature_python(weights, trace_stimulus),
    )


def snn_affinity(
    weights: FloatArray,
    query_stimulus: FloatArray,
    trace_stimulus: FloatArray,
) -> float:
    """Dispatch SNN affinity to the native wheel when installed."""
    try:  # pragma: no cover - optional native wheel
        from remanentia_retrieve import (
            snn_affinity as native_affinity,
        )
    except ImportError:
        return snn_affinity_python(weights, query_stimulus, trace_stimulus)
    return float(  # pragma: no cover - optional native wheel
        native_affinity(
            weights.astype(np.float64),
            query_stimulus.astype(np.float64),
            trace_stimulus.astype(np.float64),
        )
    )
