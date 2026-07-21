# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Held-out temporal-recall probe tests

"""Disjoint held-out recall harness: split, corruption, false-recall, and the real G1/G2 gate logic."""

from __future__ import annotations

import numpy as np
import pytest

from snn_memory import held_out_probe as h
from snn_memory.checkpoint import Checkpoint
from snn_memory.contracts import ModelConfig, ProbeConfig, TrainConfig
from snn_memory.trainer import train_memories


def _sequence(model: ModelConfig, steps: int, active: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    currents = np.zeros((steps, model.n_neurons), dtype=np.float64)
    for step in range(steps):
        currents[step, rng.choice(model.n_neurons, active, replace=False)] = 18.0
    return currents


def _checkpoint() -> tuple[Checkpoint, list[np.ndarray], list[str]]:
    model = ModelConfig(n_neurons=20, connectivity=0.3)
    labels = ["alpha", "beta"]
    sequences = [_sequence(model, 8, 3, seed) for seed in (1, 2)]
    result = train_memories(sequences, labels, model, TrainConfig(seed=11, epochs=2))
    checkpoint = Checkpoint(
        result.weights, result.topology, result.signatures, tuple(labels), model, {"metadata": {}}
    )
    return checkpoint, sequences, labels


# ---- disjoint_split -------------------------------------------------------------------------------


def test_disjoint_split_is_strictly_non_overlapping() -> None:
    sequence = np.arange(80.0).reshape(8, 10)
    cue, suffix = h.disjoint_split(sequence, 0.5)
    assert len(cue) == 4 and len(suffix) == 4
    assert np.array_equal(np.concatenate((cue, suffix)), sequence)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_disjoint_split_rejects_a_degenerate_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="cue_fraction must be in"):
        h.disjoint_split(np.zeros((4, 3)), fraction)


def test_disjoint_split_rejects_a_sequence_too_short_to_split() -> None:
    with pytest.raises(ValueError, match="too short to split"):
        h.disjoint_split(np.ones((1, 3)), 0.9)


# ---- corrupt_cue ----------------------------------------------------------------------------------


def test_corrupt_cue_drops_a_deterministic_fraction_of_driven_steps() -> None:
    cue = np.zeros((8, 4))
    cue[[1, 3, 5, 7]] = 5.0
    corrupted = h.corrupt_cue(cue, 0.25, seed=7)
    driven_before = int(np.any(cue != 0.0, axis=1).sum())
    driven_after = int(np.any(corrupted != 0.0, axis=1).sum())
    assert driven_after == driven_before - 1
    assert np.all(
        np.isin(
            np.flatnonzero(np.any(corrupted != 0.0, axis=1)),
            np.flatnonzero(np.any(cue != 0.0, axis=1)),
        )
    )


def test_corrupt_cue_with_too_small_a_fraction_drops_nothing() -> None:
    cue = np.zeros((8, 4))
    cue[[1, 3]] = 5.0
    corrupted = h.corrupt_cue(cue, 0.1, seed=7)
    assert np.array_equal(corrupted, cue)


def test_corrupt_cue_rejects_an_out_of_range_fraction() -> None:
    with pytest.raises(ValueError, match="corruption fraction"):
        h.corrupt_cue(np.ones((2, 2)), 1.0, seed=1)


def test_corrupt_cue_rejects_a_cue_without_input() -> None:
    with pytest.raises(ValueError, match="no external input"):
        h.corrupt_cue(np.zeros((4, 3)), 0.25, seed=1)


# ---- _trim_to_input via drive ---------------------------------------------------------------------


def test_drive_rejects_a_driver_without_input() -> None:
    checkpoint, _sequences, _labels = _checkpoint()
    with pytest.raises(ValueError, match="no external input"):
        h._drive_completion(
            checkpoint.weights,
            checkpoint.topology,
            checkpoint.model,
            np.zeros((4, checkpoint.model.n_neurons)),
            4,
            "empty",
        )


# ---- _predict -------------------------------------------------------------------------------------


def test_predict_returns_the_best_positive_match() -> None:
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert h._predict(np.array([0.9, 0.1]), targets, ["a", "b"]) == "a"
    assert h._predict(np.array([0.1, 0.9]), targets, ["a", "b"]) == "b"


def test_predict_abstains_when_no_target_matches() -> None:
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert h._predict(np.zeros(2), targets, ["a", "b"]) is None


# ---- _condition_weights ---------------------------------------------------------------------------


def test_condition_weights_encoder_only_is_zeroed() -> None:
    checkpoint, _sequences, _labels = _checkpoint()
    assert np.array_equal(
        h._condition_weights(checkpoint, "encoder-only", 11), np.zeros_like(checkpoint.weights)
    )


@pytest.mark.parametrize("condition", ["trained", "shuffled", "random", "zero"])
def test_condition_weights_builds_each_matched_control(condition: str) -> None:
    checkpoint, _sequences, _labels = _checkpoint()
    weights = h._condition_weights(checkpoint, condition, 11)
    assert weights.shape == checkpoint.weights.shape


def test_condition_weights_rejects_an_unsupported_condition() -> None:
    checkpoint, _sequences, _labels = _checkpoint()
    with pytest.raises(ValueError, match="unsupported control condition"):
        h._condition_weights(checkpoint, "made-up", 11)


# ---- held_out_target_signatures + probe_condition -------------------------------------------------


def test_target_signatures_have_one_row_per_label() -> None:
    checkpoint, sequences, labels = _checkpoint()
    targets = h.held_out_target_signatures(
        sequences,
        labels,
        checkpoint.weights,
        checkpoint.topology,
        checkpoint.model,
        ProbeConfig(completion_steps=8),
    )
    assert targets.shape[0] == len(labels)


@pytest.mark.parametrize("condition", list(h.CONDITIONS))
def test_probe_condition_scores_every_cue_per_condition(condition: str) -> None:
    checkpoint, sequences, labels = _checkpoint()
    scores, details = h.probe_condition(
        checkpoint, sequences, labels, condition, 11, ProbeConfig(completion_steps=8)
    )
    assert len(scores) == len(labels)
    assert all(value in (0.0, 1.0) for value in scores)
    assert {row["label"] for row in details} == set(labels)


def test_probe_condition_corruption_and_blank_paths_run() -> None:
    checkpoint, sequences, labels = _checkpoint()
    probe = ProbeConfig(completion_steps=8)
    corrupted, _ = h.probe_condition(
        checkpoint, sequences, labels, "trained", 11, probe, corruption=0.25
    )
    blank, _ = h.probe_condition(
        checkpoint, sequences, labels, "trained", 11, probe, blank_cue=True
    )
    assert len(corrupted) == len(labels) and len(blank) == len(labels)


# ---- benchmark + decision report ------------------------------------------------------------------


def test_held_out_benchmark_runs_and_reports_gates() -> None:
    checkpoint, sequences, labels = _checkpoint()
    report = h.held_out_benchmark(
        checkpoint, sequences, labels, [11, 29], ProbeConfig(completion_steps=6)
    )
    assert report["seeds"] == [11, 29]
    assert set(report["p_at_1"]) == set(h.CONDITIONS)
    assert set(report["gates"]) >= {"g1_pass", "g2_pass", "g2_false_recall_ok"}


def _const(value: float, seeds: list[int], rows: int = 2) -> dict[int, list[float]]:
    return {seed: [value] * rows for seed in seeds}


def test_decision_report_passes_both_gates_on_a_strong_effect() -> None:
    seeds = [11, 29, 47]
    scores = {
        "trained": _const(1.0, seeds),
        "shuffled": _const(0.0, seeds),
        "random": _const(0.0, seeds),
        "zero": _const(0.0, seeds),
        "encoder-only": _const(0.0, seeds),
    }
    corrupted = {"trained": _const(1.0, seeds), "shuffled": _const(0.0, seeds)}
    false_recall = _const(0.0, seeds)
    report = h.held_out_decision_report(scores, corrupted, false_recall, seeds)
    gates = report["gates"]
    assert gates["g1_pass"] and gates["g2_pass"]
    assert gates["g2_beats_random"] and gates["g2_beats_zero"]
    assert gates["g2_corruption_margin_positive"] and gates["g2_false_recall_ok"]


def test_decision_report_fails_gates_on_a_null_effect() -> None:
    seeds = [11, 29, 47]
    flat = {name: _const(0.2, seeds) for name in h.CONDITIONS}
    corrupted = {"trained": _const(0.2, seeds), "shuffled": _const(0.2, seeds)}
    report = h.held_out_decision_report(flat, corrupted, _const(0.0, seeds), seeds)
    gates = report["gates"]
    assert not gates["g1_pass"] and not gates["g2_pass"]
    assert not gates["g2_beats_random"] and not gates["g2_beats_zero"]
    assert not gates["g2_corruption_margin_positive"]


def test_decision_report_fails_g2_when_false_recall_is_high() -> None:
    seeds = [11, 29, 47]
    scores = {
        "trained": _const(1.0, seeds),
        "shuffled": _const(0.0, seeds),
        "random": _const(0.0, seeds),
        "zero": _const(0.0, seeds),
        "encoder-only": _const(0.0, seeds),
    }
    corrupted = {"trained": _const(1.0, seeds), "shuffled": _const(0.0, seeds)}
    report = h.held_out_decision_report(scores, corrupted, _const(0.5, seeds), seeds)
    gates = report["gates"]
    assert gates["g1_pass"]
    assert gates["g2_effect_threshold_pass"]
    assert not gates["g2_false_recall_ok"]
    assert not gates["g2_pass"]
