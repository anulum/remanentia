# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN training and calibration

"""Seeded online training and frozen temporal-signature calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snn_memory.contracts import ModelConfig, TrainConfig
from snn_memory.metrics import temporal_signature
from snn_memory.reference import run_episode
from snn_memory.state import BoolArray, FloatArray, initialise_state, initialise_weights
from snn_memory.stream_backend import StreamBackend, StreamInputs


@dataclass(frozen=True)
class TrainingResult:
    """Frozen weights, topology, signatures and ordered audit events."""

    weights: FloatArray
    topology: BoolArray
    signatures: FloatArray
    events: list[dict[str, int | str]]


def _densify_spikes(
    offsets: FloatArray, indices: FloatArray, timesteps: int, n_neurons: int
) -> BoolArray:
    """Reconstruct the dense (timesteps, n) spike matrix from the streamed CSR rows."""
    matrix = np.zeros((timesteps, n_neurons), dtype=np.bool_)
    for step in range(timesteps):
        start = int(offsets[step])
        stop = int(offsets[step + 1])
        matrix[step, indices[start:stop].astype(np.int64)] = True
    return matrix


def _episode(
    model: ModelConfig,
    weights: FloatArray,
    topology: BoolArray,
    currents: FloatArray,
    *,
    plasticity_enabled: bool,
    backend: StreamBackend | None,
) -> tuple[FloatArray, BoolArray]:
    """Run one episode and return ``(final_weights, dense_spikes)``.

    With ``backend`` unset the numpy reference is the oracle. With a ``StreamBackend`` the
    bit-identical Rust streamed path runs it (~100x faster at 2048 neurons); the whole episode is
    external-cue driven (``cue_steps`` = every timestep) so there is no autonomous completion phase.
    Single-episode plasticity parity is proven by the stream_stage1 gate; chained-training parity
    is covered by ``test_snn_memory_trainer``.
    """
    if backend is None:
        result = run_episode(
            initialise_state(model),
            weights,
            topology,
            currents,
            model,
            plasticity_enabled=plasticity_enabled,
        )
        return result.final_weights, result.spikes
    state = initialise_state(model)
    inputs = StreamInputs(
        voltage_mv=np.ascontiguousarray(state.voltage, dtype=np.float64),
        refractory_steps=np.ascontiguousarray(state.refractory, dtype=np.uint32),
        spikes=np.ascontiguousarray(state.spikes, dtype=np.bool_),
        pre_trace=np.ascontiguousarray(state.pre_trace, dtype=np.float64),
        post_trace=np.ascontiguousarray(state.post_trace, dtype=np.float64),
        weights=np.ascontiguousarray(weights, dtype=np.float64),
        topology=np.ascontiguousarray(topology, dtype=np.bool_),
        packets=np.ascontiguousarray(currents, dtype=np.float64),
    )
    streamed = backend.run(inputs, len(currents), plasticity_enabled, model)
    spikes = _densify_spikes(
        streamed.spike_offsets, streamed.spike_indices, len(currents), model.n_neurons
    )
    return np.array(streamed.final_weights, dtype=np.float64), spikes


def train_memories(
    sequences: list[FloatArray],
    labels: list[str],
    model: ModelConfig,
    train: TrainConfig,
    *,
    initial_weights: FloatArray | None = None,
    initial_topology: BoolArray | None = None,
    start_epoch: int = 0,
    backend: StreamBackend | None = None,
) -> TrainingResult:
    """Train complete ordered sequences then calibrate frozen signatures.

    ``backend`` unset trains on the numpy reference (the oracle). Passing a verified
    :class:`StreamBackend` runs the bit-identical Rust streamed path (~100x faster at 2048 neurons).
    """
    if not sequences or len(sequences) != len(labels) or len(set(labels)) != len(labels):
        raise ValueError("training requires unique labels and one sequence per label")
    if start_epoch < 0 or start_epoch >= train.epochs:
        raise ValueError("start_epoch must precede the configured total epochs")
    if (initial_weights is None) != (initial_topology is None):
        raise ValueError("resume requires both weights and topology")
    if initial_weights is None or initial_topology is None:
        weights, topology = initialise_weights(model, train.seed)
    else:
        weights, topology = initial_weights.copy(), initial_topology.copy()
    events: list[dict[str, int | str]] = []
    for epoch in range(start_epoch, train.epochs):
        rng = np.random.default_rng(np.random.SeedSequence([train.seed, epoch]))
        for memory_index in rng.permutation(len(sequences)):
            weights, spikes = _episode(
                model,
                weights,
                topology,
                sequences[int(memory_index)],
                plasticity_enabled=True,
                backend=backend,
            )
            events.append(
                {"epoch": epoch, "label": labels[int(memory_index)], "timesteps": len(spikes)}
            )
    signatures = calibrate_signatures(sequences, weights, topology, model, backend=backend)
    return TrainingResult(weights, topology, signatures, events)


def calibrate_signatures(
    sequences: list[FloatArray],
    weights: FloatArray,
    topology: BoolArray,
    model: ModelConfig,
    *,
    completion_steps: int = 40,
    cue_fraction: float = 0.5,
    backend: StreamBackend | None = None,
) -> FloatArray:
    """Recompute signatures under one frozen connectivity condition.

    ``backend`` unset uses the numpy reference oracle; a :class:`StreamBackend` runs the
    bit-identical Rust path. Calibration is plasticity-disabled, so the weights never change.
    """
    rows: list[FloatArray] = []
    for sequence in sequences:
        cue_steps = max(1, int(len(sequence) * cue_fraction))
        sequence = sequence[:cue_steps]
        active_rows = np.flatnonzero(np.any(sequence != 0.0, axis=1))
        if active_rows.size == 0:
            raise ValueError("calibration sequence contains no external cue")
        sequence = sequence[: int(active_rows[-1]) + 1]
        completion: FloatArray = np.zeros((completion_steps, model.n_neurons), dtype=np.float64)
        currents = np.concatenate((sequence, completion), axis=0)
        _weights, spikes = _episode(
            model, weights, topology, currents, plasticity_enabled=False, backend=backend
        )
        rows.append(temporal_signature(spikes[-completion_steps:]))
    return np.asarray(rows, dtype=np.float64)
