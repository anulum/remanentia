# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Frozen cue-only SNN probing

"""Fresh-state cue-only recall with plasticity disabled by construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snn_memory.checkpoint import Checkpoint
from snn_memory.contracts import ProbeConfig
from snn_memory.metrics import cosine_scores, recurrence_to_input_ratio, temporal_signature
from snn_memory.reference import run_episode
from snn_memory.state import FloatArray, initialise_state


@dataclass(frozen=True)
class ProbeResult:
    """Cue-only prediction plus temporal and mechanism observables."""

    label: str | None
    scores: FloatArray
    spikes: np.ndarray
    completion_spikes: int
    recurrence_input_ratio: float
    weight_digest_unchanged: bool


def probe_checkpoint(
    checkpoint: Checkpoint,
    cue_currents: FloatArray,
    config: ProbeConfig,
) -> ProbeResult:
    """Probe a frozen checkpoint without any corpus or candidate-document input."""
    cue_steps = max(1, int(len(cue_currents) * config.cue_fraction))
    cue = cue_currents[:cue_steps]
    active_rows = np.flatnonzero(np.any(cue != 0.0, axis=1))
    if active_rows.size == 0:
        raise ValueError("probe cue contains no external input")
    cue = cue[: int(active_rows[-1]) + 1]
    cue_steps = len(cue)
    completion = np.zeros((config.completion_steps, checkpoint.model.n_neurons))
    currents = np.concatenate((cue, completion), axis=0)
    before = checkpoint.weights.tobytes()
    episode = run_episode(
        initialise_state(checkpoint.model),
        checkpoint.weights,
        checkpoint.topology,
        currents,
        checkpoint.model,
        plasticity_enabled=False,
    )
    signature = temporal_signature(episode.spikes[-config.completion_steps :])
    scores = cosine_scores(signature, checkpoint.signatures)
    prediction = None if not np.any(scores > 0.0) else checkpoint.labels[int(np.argmax(scores))]
    return ProbeResult(
        prediction,
        scores,
        episode.spikes,
        int(episode.spikes[cue_steps:].sum()),
        recurrence_to_input_ratio(episode.recurrent_current, currents),
        before == episode.final_weights.tobytes(),
    )
