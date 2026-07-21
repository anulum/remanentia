# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN benchmark-orchestration tests

"""Real matched-control orchestration, abstention and preregistered gate logic."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest

from snn_memory.checkpoint import Checkpoint
from snn_memory.contracts import ModelConfig, ProbeConfig, TrainConfig
from snn_memory.experiment import (
    LOCKED_SEEDS,
    benchmark,
    decision_report,
    evaluate_condition,
)
from snn_memory.trainer import train_memories

CONDITIONS = ("trained", "shuffled", "random", "zero", "encoder-only")


def _sequence(model: ModelConfig, steps: int, active: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    currents = np.zeros((steps, model.n_neurons), dtype=np.float64)
    for step in range(steps):
        index = rng.choice(model.n_neurons, active, replace=False)
        currents[step, index] = 18.0
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


def test_evaluate_trained_condition_scores_every_cue() -> None:
    checkpoint, sequences, labels = _checkpoint()
    scores, details = evaluate_condition(
        checkpoint, sequences, labels, "trained", 11, ProbeConfig(completion_steps=8)
    )
    assert len(scores) == len(labels)
    assert all(value in (0.0, 1.0) for value in scores)
    assert {row["label"] for row in details} == set(labels)


def test_evaluate_encoder_only_condition_uses_input_signatures() -> None:
    checkpoint, sequences, labels = _checkpoint()
    scores, details = evaluate_condition(
        checkpoint, sequences, labels, "encoder-only", 11, ProbeConfig(completion_steps=8)
    )
    assert len(scores) == len(labels)
    assert all("prediction" in row for row in details)


def test_evaluate_condition_rejects_an_unsupported_condition() -> None:
    checkpoint, sequences, labels = _checkpoint()
    with pytest.raises(ValueError, match="unsupported control condition"):
        evaluate_condition(checkpoint, sequences, labels, "made-up", 11, ProbeConfig())


def test_encoder_only_condition_rejects_a_cue_without_external_input() -> None:
    checkpoint, sequences, labels = _checkpoint()
    empty = [np.zeros_like(sequences[0]), sequences[1]]
    with pytest.raises(ValueError, match="no external input"):
        evaluate_condition(
            checkpoint, empty, labels, "encoder-only", 11, ProbeConfig(completion_steps=8)
        )


def test_benchmark_reports_all_five_conditions_and_holds_gates_closed() -> None:
    checkpoint, sequences, labels = _checkpoint()
    report = benchmark(checkpoint, sequences, labels, [11, 29], ProbeConfig(completion_steps=6))
    assert report["seeds"] == [11, 29]
    assert set(report["conditions"]) == set(CONDITIONS)
    assert report["gates"]["g1_pass"] is False
    assert report["gates"]["g2_effect_threshold_pass"] is False
    assert report["gates"]["g2_pass"] is False
    # Always-on model-free contract guard: the real benchmark(...) output, round-tripped
    # through JSON exactly as the CLI writes it (integer seed keys coerced to strings),
    # must satisfy the tracked public schema. The validator is loaded from the real
    # jsonschema module at runtime and only its callable is typed, so strict mypy holds
    # without a stub, ignore, or duplicated schema logic.
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "docs/schema/snn_memory_result.schema.json").read_text(encoding="utf-8")
    )
    validate = cast(Callable[[object, object], None], import_module("jsonschema").validate)
    validate(json.loads(json.dumps(report)), schema)
    # Schema v1 pins all three reported G1/G2 gates to false structurally, so the guard
    # enforces semantics and not only shape: a report claiming ANY passed gate must fail
    # validation. Exercise every const-false key so a future accidental boolean
    # regression in either G2 field is caught too.
    validation_error = cast(type[Exception], import_module("jsonschema.exceptions").ValidationError)
    for gate in ("g1_pass", "g2_effect_threshold_pass", "g2_pass"):
        tampered = json.loads(json.dumps(report))
        tampered["gates"][gate] = True
        with pytest.raises(validation_error):
            validate(tampered, schema)
    for section in ("conditions", "p_at_1"):
        missing_encoder_only = json.loads(json.dumps(report))
        del missing_encoder_only[section]["encoder-only"]
        with pytest.raises(validation_error):
            validate(missing_encoder_only, schema)


def _scores(seeds: list[int]) -> dict[str, dict[int, list[float]]]:
    return {
        "trained": {seed: [1.0, 1.0] for seed in seeds},
        "shuffled": {seed: [0.0, 0.0] for seed in seeds},
        "random": {seed: [0.0, 0.0] for seed in seeds},
        "zero": {seed: [0.0, 0.0] for seed in seeds},
        "encoder-only": {seed: [0.0, 0.0] for seed in seeds},
    }


def _details(seeds: list[int]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    def rows(spikes: int) -> list[dict[str, Any]]:
        return [
            {"label": "alpha", "prediction": "alpha", "correct": True, "completion_spikes": spikes}
        ]

    completion = {"trained": 9, "shuffled": 1, "random": 1, "zero": 0, "encoder-only": 0}
    return {name: {seed: rows(completion[name]) for seed in seeds} for name in CONDITIONS}


def test_decision_report_holds_every_gate_fail_closed() -> None:
    seeds = [11, 29]
    report = decision_report(_scores(seeds), _details(seeds), seeds)
    assert report["gates"]["g1_pass"] is False
    assert report["gates"]["g2_effect_threshold_pass"] is False
    assert report["gates"]["g2_pass"] is False
    assert any("circular self-matching" in line for line in report["limits"])


def test_decision_report_keeps_gates_closed_even_with_perfect_circular_scores() -> None:
    # Even flawless self-matching scores on the locked seed set must not open a gate:
    # the harness calibrates each signature from the cue it scores, so p_at_1 is
    # circular. The raw figure is retained only as an explicitly non-gating diagnostic.
    report = decision_report(_scores(LOCKED_SEEDS), _details(LOCKED_SEEDS), LOCKED_SEEDS)
    assert report["gates"]["g1_pass"] is False
    assert report["gates"]["g2_effect_threshold_pass"] is False
    assert report["gates"]["g2_pass"] is False
    assert report["p_at_1"]["trained"] > report["p_at_1"]["random"]
