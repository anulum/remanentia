# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN trainer tests

"""Real training, calibration and deterministic-resume tests."""

from __future__ import annotations

import numpy as np
import pytest

from snn_memory.contracts import ModelConfig, TrainConfig
from snn_memory.state import initialise_weights
from snn_memory.trainer import calibrate_signatures, train_memories


def _sequence(model: ModelConfig, steps: int, active: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    currents = np.zeros((steps, model.n_neurons), dtype=np.float64)
    for step in range(steps):
        index = rng.choice(model.n_neurons, active, replace=False)
        currents[step, index] = 18.0
    return currents


def _corpus(model: ModelConfig) -> tuple[list[np.ndarray], list[str]]:
    return [_sequence(model, 8, 3, seed) for seed in (1, 2)], ["alpha", "beta"]


def test_training_freezes_signatures_and_records_every_replay() -> None:
    model = ModelConfig(n_neurons=20, connectivity=0.3)
    sequences, labels = _corpus(model)
    result = train_memories(sequences, labels, model, TrainConfig(seed=11, epochs=3))
    assert result.signatures.shape[0] == len(labels)
    assert len(result.events) == 3 * len(labels)
    assert {event["epoch"] for event in result.events} == {0, 1, 2}


def test_calibration_is_deterministic_under_frozen_connectivity() -> None:
    model = ModelConfig(n_neurons=20, connectivity=0.3)
    sequences, labels = _corpus(model)
    result = train_memories(sequences, labels, model, TrainConfig(seed=11, epochs=2))
    first = calibrate_signatures(sequences, result.weights, result.topology, model)
    second = calibrate_signatures(sequences, result.weights, result.topology, model)
    np.testing.assert_array_equal(first, second)


def test_resumed_training_reproduces_uninterrupted_weights() -> None:
    model = ModelConfig(n_neurons=20, connectivity=0.3)
    sequences, labels = _corpus(model)
    first = train_memories(sequences, labels, model, TrainConfig(seed=11, epochs=1))
    resumed = train_memories(
        sequences,
        labels,
        model,
        TrainConfig(seed=11, epochs=3),
        initial_weights=first.weights,
        initial_topology=first.topology,
        start_epoch=1,
    )
    direct = train_memories(sequences, labels, model, TrainConfig(seed=11, epochs=3))
    np.testing.assert_array_equal(resumed.weights, direct.weights)


def test_training_rejects_mismatched_or_duplicate_labels() -> None:
    model = ModelConfig(n_neurons=12, connectivity=0.3)
    sequences, _ = _corpus(model)
    with pytest.raises(ValueError, match="unique labels"):
        train_memories(sequences, ["only-one"], model, TrainConfig(seed=11, epochs=1))
    with pytest.raises(ValueError, match="unique labels"):
        train_memories(sequences, ["same", "same"], model, TrainConfig(seed=11, epochs=1))


def test_training_rejects_invalid_start_epoch() -> None:
    model = ModelConfig(n_neurons=12, connectivity=0.3)
    sequences, labels = _corpus(model)
    with pytest.raises(ValueError, match="start_epoch"):
        train_memories(sequences, labels, model, TrainConfig(seed=11, epochs=1), start_epoch=1)


def test_calibration_rejects_a_sequence_without_external_cue() -> None:
    model = ModelConfig(n_neurons=12, connectivity=0.3)
    weights, topology = initialise_weights(model, 11)
    with pytest.raises(ValueError, match="no external cue"):
        calibrate_signatures([np.zeros((6, model.n_neurons))], weights, topology, model)


def test_resume_requires_both_weights_and_topology() -> None:
    model = ModelConfig(n_neurons=12, connectivity=0.3)
    sequences, labels = _corpus(model)
    weights, _ = np.zeros((12, 12)), None
    with pytest.raises(ValueError, match="both weights and topology"):
        train_memories(
            sequences, labels, model, TrainConfig(seed=11, epochs=2), initial_weights=weights
        )


def _stream_backend():
    """Load the installed Rust streamed backend, skipping when the extension is absent."""
    import hashlib
    from pathlib import Path

    from snn_memory.stream_backend import BackendIdentity, load_stream_backend

    module = pytest.importorskip("rust_snn_memory.rust_snn_memory")
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return load_stream_backend(
        BackendIdentity(module.STREAMED_API_VERSION, module.CRATE_VERSION, digest)
    )


def test_rust_backend_trains_bit_identically_to_the_numpy_oracle() -> None:
    backend = _stream_backend()
    model = ModelConfig(n_neurons=20, excitatory_fraction=0.8, connectivity=0.3, weight_max=2.0)
    sequences, labels = _corpus(model)
    train = TrainConfig(seed=11, epochs=3)
    oracle = train_memories(sequences, labels, model, train)
    accelerated = train_memories(sequences, labels, model, train, backend=backend)
    # Chained multi-episode training + calibration are bit-identical on the Rust streamed path.
    np.testing.assert_array_equal(oracle.weights, accelerated.weights)
    np.testing.assert_array_equal(oracle.signatures, accelerated.signatures)


def test_rust_backend_calibration_matches_the_numpy_oracle() -> None:
    backend = _stream_backend()
    model = ModelConfig(n_neurons=20, excitatory_fraction=0.8, connectivity=0.3, weight_max=2.0)
    sequences, labels = _corpus(model)
    trained = train_memories(sequences, labels, model, TrainConfig(seed=11, epochs=2))
    oracle = calibrate_signatures(sequences, trained.weights, trained.topology, model)
    accelerated = calibrate_signatures(
        sequences, trained.weights, trained.topology, model, backend=backend
    )
    np.testing.assert_array_equal(oracle, accelerated)
