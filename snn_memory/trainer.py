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


@dataclass(frozen=True)
class TrainingResult:
    """Frozen weights, topology, signatures and ordered audit events."""

    weights: FloatArray
    topology: BoolArray
    signatures: FloatArray
    events: list[dict[str, int | str]]


def train_memories(
    sequences: list[FloatArray],
    labels: list[str],
    model: ModelConfig,
    train: TrainConfig,
    *,
    initial_weights: FloatArray | None = None,
    initial_topology: BoolArray | None = None,
    start_epoch: int = 0,
) -> TrainingResult:
    """Train complete ordered sequences then calibrate frozen signatures."""
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
            result = run_episode(
                initialise_state(model),
                weights,
                topology,
                sequences[int(memory_index)],
                model,
                plasticity_enabled=True,
            )
            weights = result.final_weights
            events.append(
                {"epoch": epoch, "label": labels[int(memory_index)], "timesteps": len(result.spikes)}
            )
    signatures = calibrate_signatures(sequences, weights, topology, model)
    return TrainingResult(weights, topology, signatures, events)


def calibrate_signatures(
    sequences: list[FloatArray],
    weights: FloatArray,
    topology: BoolArray,
    model: ModelConfig,
    *,
    completion_steps: int = 40,
    cue_fraction: float = 0.5,
) -> FloatArray:
    """Recompute signatures under one frozen connectivity condition."""
    rows: list[FloatArray] = []
    for sequence in sequences:
        cue_steps = max(1, int(len(sequence) * cue_fraction))
        sequence = sequence[:cue_steps]
        active_rows = np.flatnonzero(np.any(sequence != 0.0, axis=1))
        if active_rows.size == 0:
            raise ValueError("calibration sequence contains no external cue")
        sequence = sequence[: int(active_rows[-1]) + 1]
        completion: FloatArray = np.zeros(
            (completion_steps, model.n_neurons), dtype=np.float64
        )
        currents = np.concatenate((sequence, completion), axis=0)
        result = run_episode(
            initialise_state(model),
            weights,
            topology,
            currents,
            model,
            plasticity_enabled=False,
        )
        rows.append(temporal_signature(result.spikes[-completion_steps:]))
    return np.asarray(rows, dtype=np.float64)
