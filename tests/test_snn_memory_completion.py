# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Recurrence/activity-control smoke test

"""Recurrent-activity control for temporal completion after cue removal.

This is NOT a recall or G1 test. ``evaluate_condition`` calibrates each stored
signature from the same cue prefix it scores, so its P@1 is circular self-matching;
this test therefore makes no learned-recall claim. It asserts only the recurrent
mechanism: trained excitatory weights sustain spiking after the external cue is
removed, and the E->E-local shuffled control silences that completion.
"""

from __future__ import annotations

import numpy as np

from snn_memory.checkpoint import Checkpoint
from snn_memory.contracts import ModelConfig, ProbeConfig, TrainConfig
from snn_memory.experiment import evaluate_condition
from snn_memory.state import FloatArray
from snn_memory.trainer import train_memories


def _controlled_sequences(neurons: int) -> tuple[list[FloatArray], list[str]]:
    sequences: list[FloatArray] = []
    labels = [f"memory-{index}" for index in range(4)]
    for index in range(4):
        rows: list[FloatArray] = []
        groups = (range(index * 12, (index + 1) * 12), range(48 + index * 12, 60 + index * 12))
        for group in groups:
            pulse: FloatArray = np.zeros(neurons, dtype=np.float64)
            pulse[list(group)] = 18.0
            rows.extend((pulse, np.zeros(neurons), np.zeros(neurons)))
        sequences.append(np.asarray(rows, dtype=np.float64))
    return sequences, labels


def test_trained_recurrence_completes_while_local_control_silences_it() -> None:
    model = ModelConfig(n_neurons=128, excitatory_fraction=0.8, connectivity=1.0)
    sequences, labels = _controlled_sequences(model.n_neurons)
    seeds = [11, 29, 47, 71]
    for seed in seeds:
        trained = train_memories(
            sequences, labels, model, TrainConfig(seed=seed, epochs=220)
        )
        checkpoint = Checkpoint(
            trained.weights,
            trained.topology,
            trained.signatures,
            tuple(labels),
            model,
            {},
        )
        probe = ProbeConfig(seed=seed, cue_fraction=0.17, completion_steps=10)
        _, details = evaluate_condition(checkpoint, sequences, labels, "trained", seed, probe)
        _, shuffled_details = evaluate_condition(
            checkpoint, sequences, labels, "shuffled", seed, probe
        )
        # Mechanism signal only (no P@1/recall claim, which would be self-matching):
        # trained recurrence sustains completion spikes; the E->E-local shuffled
        # control silences them.
        assert all(row["completion_spikes"] > 0 for row in details)
        assert all(row["completion_spikes"] == 0 for row in shuffled_details)
