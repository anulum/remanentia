# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN benchmark orchestration

"""Matched-control orchestration and preregistered gate evaluation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from snn_memory.checkpoint import Checkpoint
from snn_memory.contracts import ProbeConfig
from snn_memory.controls import ControlName, make_control
from snn_memory.metrics import cosine_scores, temporal_signature
from snn_memory.reference import run_episode
from snn_memory.state import FloatArray, initialise_state
from snn_memory.statistics import paired_interval
from snn_memory.trainer import calibrate_signatures

LOCKED_SEEDS = [11, 29, 47, 71, 101, 131, 167, 211, 257, 307]


def evaluate_condition(
    checkpoint: Checkpoint,
    full_sequences: list[FloatArray],
    labels: list[str],
    condition: ControlName | str,
    seed: int,
    probe: ProbeConfig,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Score all cues under one matched condition with recalibrated signatures."""
    if condition == "encoder-only":
        weights = np.zeros_like(checkpoint.weights)
        signatures = np.asarray(
            [temporal_signature(sequence > 0.0) for sequence in full_sequences],
            dtype=np.float64,
        )
    else:
        if condition not in {"trained", "shuffled", "random", "zero"}:
            raise ValueError(f"unsupported control condition: {condition}")
        weights = make_control(
            checkpoint.weights,
            checkpoint.topology,
            checkpoint.model,
            condition,
            seed,
        )
        signatures = calibrate_signatures(
            full_sequences,
            weights,
            checkpoint.topology,
            checkpoint.model,
            completion_steps=probe.completion_steps,
            cue_fraction=probe.cue_fraction,
        )
    correctness: list[float] = []
    details: list[dict[str, Any]] = []
    for label, sequence in zip(labels, full_sequences, strict=True):
        cue_steps = max(1, int(len(sequence) * probe.cue_fraction))
        cue = sequence[:cue_steps]
        active_rows = np.flatnonzero(np.any(cue != 0.0, axis=1))
        if active_rows.size == 0:
            raise ValueError("benchmark cue contains no external input")
        cue = cue[: int(active_rows[-1]) + 1]
        cue_steps = len(cue)
        completion = np.zeros((probe.completion_steps, checkpoint.model.n_neurons))
        currents = np.concatenate((cue, completion), axis=0)
        result = run_episode(
            initialise_state(checkpoint.model),
            weights,
            checkpoint.topology,
            currents,
            checkpoint.model,
            plasticity_enabled=False,
        )
        query_signature = (
            temporal_signature(currents > 0.0)
            if condition == "encoder-only"
            else temporal_signature(result.spikes[-probe.completion_steps :])
        )
        scores = cosine_scores(query_signature, signatures)
        predicted = None if not np.any(scores > 0.0) else labels[int(np.argmax(scores))]
        correct = float(predicted == label)
        correctness.append(correct)
        details.append(
            {
                "label": label,
                "prediction": predicted,
                "correct": bool(correct),
                "completion_spikes": int(result.spikes[cue_steps:].sum()),
            }
        )
    return correctness, details


def benchmark(
    checkpoint: Checkpoint,
    sequences: list[FloatArray],
    labels: list[str],
    seeds: list[int],
    probe: ProbeConfig,
) -> dict[str, Any]:
    """Run all recurrent controls and return an honest G1/G2 decision report."""
    conditions = ("trained", "shuffled", "random", "zero", "encoder-only")
    scores: dict[str, dict[int, list[float]]] = {name: {} for name in conditions}
    details: dict[str, dict[int, list[dict[str, Any]]]] = {name: {} for name in conditions}
    for run_seed in seeds:
        for condition in conditions:
            score, rows = evaluate_condition(
                checkpoint, sequences, labels, condition, run_seed, probe
            )
            scores[condition][run_seed] = score
            details[condition][run_seed] = rows
    return decision_report(scores, details, seeds)


def decision_report(
    scores: dict[str, dict[int, list[float]]],
    details: dict[str, dict[int, list[dict[str, Any]]]],
    seeds: list[int],
) -> dict[str, Any]:
    """Build one seed-complete decision report from isolated condition results."""
    trained_shuffled = paired_interval(scores["trained"], scores["shuffled"], seeds)
    p_at_1 = {
        name: float(np.mean([value for rows in runs.values() for value in rows]))
        for name, runs in scores.items()
    }
    trained_completion = [
        int(row["completion_spikes"])
        for seed_rows in details["trained"].values()
        for row in seed_rows
    ]
    shuffled_completion = [
        int(row["completion_spikes"])
        for seed_rows in details["shuffled"].values()
        for row in seed_rows
    ]
    zero_completion = [
        int(row["completion_spikes"])
        for seed_rows in details["zero"].values()
        for row in seed_rows
    ]
    g1_pass = (
        trained_shuffled.mean >= 0.25
        and trained_shuffled.lower > 0.0
        and bool(trained_completion)
        and all(value > 0 for value in trained_completion)
        and float(np.mean(trained_completion)) > float(np.mean(shuffled_completion))
        and float(np.mean(trained_completion)) > float(np.mean(zero_completion))
    )
    effect_threshold = (
        seeds == LOCKED_SEEDS
        and trained_shuffled.mean >= 0.15
        and trained_shuffled.lower > 0.05
        and p_at_1["trained"] > p_at_1["random"]
        and p_at_1["trained"] > p_at_1["zero"]
    )
    return {
        "schema_version": 1,
        "seeds": seeds,
        "conditions": details,
        "p_at_1": p_at_1,
        "trained_minus_shuffled": asdict(trained_shuffled),
        "gates": {
            "g1_pass": g1_pass,
            "g2_effect_threshold_pass": effect_threshold,
            "g2_pass": False,
        },
        "limits": [
            "G2 also requires locked-corpus corruption and false-recall audits.",
            "This report does not evaluate G4 or product retrieval.",
        ],
    }
