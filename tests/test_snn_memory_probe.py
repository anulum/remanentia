# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Frozen cue-only probe tests

"""Real fresh-state probe tests: sustained recall, abstention and cue guard."""

from __future__ import annotations

import numpy as np
import pytest

from snn_memory.checkpoint import Checkpoint
from snn_memory.contracts import ModelConfig, ProbeConfig
from snn_memory.probe import probe_checkpoint


def _sustaining_checkpoint() -> tuple[Checkpoint, np.ndarray]:
    """A valid checkpoint whose strong E→E block self-sustains during completion."""
    model = ModelConfig(n_neurons=30, excitatory_fraction=0.9, connectivity=1.0, refractory_ms=0.0)
    excitatory = model.n_excitatory
    weights = np.zeros((30, 30), dtype=np.float64)
    weights[:excitatory, :excitatory] = 1.0
    weights[:excitatory, excitatory:] = 0.1
    weights[excitatory:, :] = -0.01
    np.fill_diagonal(weights, 0.0)
    topology = weights != 0.0
    signatures = np.ones((2, 8 * 30), dtype=np.float64)
    checkpoint = Checkpoint(weights, topology, signatures, ("alpha", "beta"), model, {"metadata": {}})
    cue = np.zeros((5, 30), dtype=np.float64)
    cue[:, :excitatory] = 30.0
    return checkpoint, cue


def test_probe_recovers_a_label_when_the_attractor_sustains() -> None:
    checkpoint, cue = _sustaining_checkpoint()
    result = probe_checkpoint(checkpoint, cue, ProbeConfig(cue_fraction=1.0, completion_steps=8))
    assert result.label in {"alpha", "beta"}
    assert result.scores.shape == (2,)
    assert result.completion_spikes > 0
    assert result.recurrence_input_ratio > 0.0
    assert result.weight_digest_unchanged is True


def test_probe_abstains_when_no_recurrent_activity_survives() -> None:
    model = ModelConfig(n_neurons=20, excitatory_fraction=0.8, connectivity=1.0)
    zero = Checkpoint(
        np.zeros((20, 20)),
        np.zeros((20, 20), dtype=np.bool_),
        np.ones((2, 8 * 20)),
        ("alpha", "beta"),
        model,
        {"metadata": {}},
    )
    cue = np.zeros((5, 20), dtype=np.float64)
    cue[:, : model.n_excitatory] = 30.0
    result = probe_checkpoint(zero, cue, ProbeConfig(cue_fraction=1.0, completion_steps=8))
    assert result.label is None
    assert not np.any(result.scores > 0.0)
    assert result.weight_digest_unchanged is True


def test_probe_rejects_a_cue_without_external_input() -> None:
    checkpoint, _ = _sustaining_checkpoint()
    with pytest.raises(ValueError, match="no external input"):
        probe_checkpoint(checkpoint, np.zeros((4, 30)), ProbeConfig(completion_steps=4))
