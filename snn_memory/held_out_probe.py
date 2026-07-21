# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Valid held-out temporal-recall probe harness

"""Held-out temporal-recall probe that breaks the circular self-matching of the in-place benchmark.

The in-place ``experiment.evaluate_condition`` calibrates each stored label signature from the SAME cue
prefix it later scores, so its ``p_at_1`` is circular self-matching, not recall. This harness fixes that per
the preregistration (2026-07-17): each label's stored target signature is calibrated from the completion the
frozen network produces when driven by the calibration SUFFIX — the strict complement of the cue prefix, so
no target is ever derived from bytes the scored cue exposes. Recall is the cue-prefix-driven completion
matched against those disjoint targets. The harness also runs the preregistered 25% cue-corruption margin and
the no-input (blank-cue) false-recall audit. All conditions (trained / shuffled / random / zero /
encoder-only) use the identical disjoint protocol; only the recurrent weights differ, so the comparison
isolates learned E→E structure.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

import numpy as np

from snn_memory.checkpoint import Checkpoint
from snn_memory.contracts import ProbeConfig
from snn_memory.controls import ControlName, make_control
from snn_memory.metrics import cosine_scores, temporal_signature
from snn_memory.reference import run_episode
from snn_memory.state import FloatArray, initialise_state
from snn_memory.statistics import paired_interval

CONDITIONS: tuple[str, ...] = ("trained", "shuffled", "random", "zero", "encoder-only")
_CORRUPTION_FRACTION = 0.25
_G1_MIN_EFFECT = 0.25
_G2_MIN_EFFECT = 0.15
_G2_LOWER_BOUND = 0.05
_FALSE_RECALL_MAX = 0.05


def _trim_to_input(currents: FloatArray, label: str) -> FloatArray:
    """Trim trailing all-zero rows so a driver ends on its last externally-driven step."""
    active = np.flatnonzero(np.any(currents != 0.0, axis=1))
    if active.size == 0:
        raise ValueError(f"held-out driver for {label!r} contains no external input")
    return currents[: int(active[-1]) + 1]


def _drive_completion(
    weights: FloatArray,
    topology: Any,
    model: Any,
    driver: FloatArray,
    completion_steps: int,
    label: str,
) -> FloatArray:
    """Drive the frozen network with ``driver`` then let it complete autonomously (input removed).

    Returns the temporal signature of the completion window only, so the score reflects the attractor the
    driver pulls the network into, not the driver itself.
    """
    driver = _trim_to_input(driver, label)
    completion = np.zeros((completion_steps, model.n_neurons), dtype=np.float64)
    currents = np.concatenate((driver, completion), axis=0)
    result = run_episode(
        initialise_state(model), weights, topology, currents, model, plasticity_enabled=False
    )
    return temporal_signature(result.spikes[-completion_steps:])


def disjoint_split(sequence: FloatArray, cue_fraction: float) -> tuple[FloatArray, FloatArray]:
    """Split an ordered latency sequence into a cue prefix and its strict complement suffix.

    The cue is the first ``cue_fraction`` of the timesteps; the calibration suffix is everything after it.
    The two never overlap, which is what makes recall held-out rather than self-matching.
    """
    if not 0.0 < cue_fraction < 1.0:
        raise ValueError("cue_fraction must be in (0, 1) for a disjoint held-out split")
    cue_steps = max(1, int(len(sequence) * cue_fraction))
    if cue_steps >= len(sequence):
        raise ValueError("sequence too short to split into a disjoint cue prefix and suffix")
    return sequence[:cue_steps], sequence[cue_steps:]


def corrupt_cue(cue: FloatArray, fraction: float, seed: int) -> FloatArray:
    """Zero a deterministic ``fraction`` of the cue's externally-driven timesteps.

    Corruption removes whole input steps (a partial/degraded cue), never fabricates spikes, so a corrupted
    cue is a strict subset of the clean cue's drive.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("corruption fraction must be in [0, 1)")
    driven = np.flatnonzero(np.any(cue != 0.0, axis=1))
    if driven.size == 0:
        raise ValueError("cannot corrupt a cue with no external input")
    n_drop = int(driven.size * fraction)
    if n_drop == 0:
        return cue.copy()
    rng = np.random.default_rng(seed)
    drop = rng.choice(driven, size=n_drop, replace=False)
    corrupted = cue.copy()
    corrupted[drop] = 0.0
    return corrupted


def held_out_target_signatures(
    sequences: list[FloatArray],
    labels: list[str],
    weights: FloatArray,
    topology: Any,
    model: Any,
    probe: ProbeConfig,
) -> FloatArray:
    """Calibrate each label's stored target signature from its DISJOINT calibration suffix.

    This is the anti-circularity core: the target is the completion the frozen network settles into when
    driven by the suffix, which the scored cue prefix never contains.
    """
    rows: list[FloatArray] = []
    for label, sequence in zip(labels, sequences, strict=True):
        _cue, suffix = disjoint_split(sequence, probe.cue_fraction)
        rows.append(
            _drive_completion(weights, topology, model, suffix, probe.completion_steps, label)
        )
    return np.asarray(rows, dtype=np.float64)


def _condition_weights(
    checkpoint: Checkpoint, condition: ControlName | str, seed: int
) -> FloatArray:
    if condition == "encoder-only":
        return np.zeros_like(checkpoint.weights)
    if condition not in {"trained", "shuffled", "random", "zero"}:
        raise ValueError(f"unsupported control condition: {condition}")
    return make_control(
        checkpoint.weights,
        checkpoint.topology,
        checkpoint.model,
        cast(ControlName, condition),
        seed,
    )


def _blank_completion(
    weights: FloatArray, topology: Any, model: Any, completion_steps: int
) -> FloatArray:
    """Autonomous completion from rest with NO external input, for the false-recall audit."""
    currents = np.zeros((completion_steps, model.n_neurons), dtype=np.float64)
    result = run_episode(
        initialise_state(model), weights, topology, currents, model, plasticity_enabled=False
    )
    return temporal_signature(result.spikes)


def _predict(query: FloatArray, targets: FloatArray, labels: list[str]) -> str | None:
    """Return the best positively-matching label for a completion signature, or None (abstain).

    A network whose completion is silent or orthogonal to every stored memory yields non-positive cosine
    scores and abstains, so a wrong guess is never manufactured from an empty recall.
    """
    scores = cosine_scores(query, targets)
    if not np.any(scores > 0.0):
        return None
    return labels[int(np.argmax(scores))]


def probe_condition(
    checkpoint: Checkpoint,
    sequences: list[FloatArray],
    labels: list[str],
    condition: ControlName | str,
    seed: int,
    probe: ProbeConfig,
    *,
    corruption: float = 0.0,
    blank_cue: bool = False,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Score held-out recall for one matched condition.

    Targets are calibrated from the disjoint suffix; the query is the cue-prefix-driven completion. With
    ``corruption`` a deterministic fraction of the cue's input steps is dropped (partial/degraded cue). With
    ``blank_cue`` the query receives no external input at all — the false-recall audit: a network that
    "recalls" the correct label from nothing is hallucinating, not completing.
    """
    weights = _condition_weights(checkpoint, condition, seed)
    targets = held_out_target_signatures(
        sequences, labels, weights, checkpoint.topology, checkpoint.model, probe
    )
    blank_query = (
        _blank_completion(weights, checkpoint.topology, checkpoint.model, probe.completion_steps)
        if blank_cue
        else None
    )
    correctness: list[float] = []
    details: list[dict[str, Any]] = []
    for index, (label, sequence) in enumerate(zip(labels, sequences, strict=True)):
        if blank_query is not None:
            query = blank_query
        else:
            cue, _suffix = disjoint_split(sequence, probe.cue_fraction)
            if corruption > 0.0:
                cue = corrupt_cue(cue, corruption, seed * 1_000_003 + index)
            query = _drive_completion(
                weights, checkpoint.topology, checkpoint.model, cue, probe.completion_steps, label
            )
        predicted = _predict(query, targets, labels)
        correct = float(predicted == label)
        correctness.append(correct)
        details.append({"label": label, "prediction": predicted, "correct": bool(correct)})
    return correctness, details


def _pool_mean(runs: dict[int, list[float]]) -> float:
    return float(np.mean([value for values in runs.values() for value in values]))


def held_out_benchmark(
    checkpoint: Checkpoint,
    sequences: list[FloatArray],
    labels: list[str],
    seeds: list[int],
    probe: ProbeConfig,
) -> dict[str, Any]:
    """Run every matched condition plus the 25% corruption and no-input false-recall audits over all seeds.

    Returns the ADR-0006 held-out decision report. Unlike ``experiment.benchmark`` this uses the disjoint
    held-out probe, so its gates are real, not fail-closed.
    """
    scores: dict[str, dict[int, list[float]]] = {name: {} for name in CONDITIONS}
    corrupted: dict[str, dict[int, list[float]]] = {"trained": {}, "shuffled": {}}
    false_recall: dict[int, list[float]] = {}
    for run_seed in seeds:
        for condition in CONDITIONS:
            values, _rows = probe_condition(
                checkpoint, sequences, labels, condition, run_seed, probe
            )
            scores[condition][run_seed] = values
        for condition in ("trained", "shuffled"):
            values, _rows = probe_condition(
                checkpoint,
                sequences,
                labels,
                condition,
                run_seed,
                probe,
                corruption=_CORRUPTION_FRACTION,
            )
            corrupted[condition][run_seed] = values
        blank, _rows = probe_condition(
            checkpoint, sequences, labels, "trained", run_seed, probe, blank_cue=True
        )
        false_recall[run_seed] = blank
    return held_out_decision_report(scores, corrupted, false_recall, seeds)


def held_out_decision_report(
    scores: dict[str, dict[int, list[float]]],
    corrupted: dict[str, dict[int, list[float]]],
    false_recall: dict[int, list[float]],
    seeds: list[int],
) -> dict[str, Any]:
    """Decide the ADR-0006 G1/G2 gates from real held-out recall.

    G1 (development): trained − shuffled ≥ 0.25 P@1, paired 95% lower bound > 0, and zero-recurrent does not
    reproduce the effect (trained > zero). G2 (locked): trained − shuffled ≥ 0.15 with paired 95% lower bound
    > 0.05; trained beats random AND zero; a positive trained-over-shuffled attractor margin survives 25% cue
    corruption; and no-input false recall stays below 0.05. This function does not decide G0, G3 or G4.
    """
    effect = paired_interval(scores["trained"], scores["shuffled"], seeds)
    corruption_margin = paired_interval(corrupted["trained"], corrupted["shuffled"], seeds)
    p_at_1 = {name: _pool_mean(runs) for name, runs in scores.items()}
    false_recall_rate = _pool_mean(false_recall)
    beats_random = p_at_1["trained"] > p_at_1["random"]
    beats_zero = p_at_1["trained"] > p_at_1["zero"]
    corruption_positive = corruption_margin.mean > 0.0
    false_recall_ok = false_recall_rate < _FALSE_RECALL_MAX
    g1_pass = effect.mean >= _G1_MIN_EFFECT and effect.lower > 0.0 and beats_zero
    g2_effect_pass = effect.mean >= _G2_MIN_EFFECT and effect.lower > _G2_LOWER_BOUND
    g2_pass = (
        g2_effect_pass and beats_random and beats_zero and corruption_positive and false_recall_ok
    )
    return {
        "schema_version": 1,
        "seeds": list(seeds),
        "method": (
            "held-out disjoint-suffix calibration, cue-prefix query; 25% corruption and no-input "
            "false-recall audits; not circular self-matching"
        ),
        "p_at_1": p_at_1,
        "trained_minus_shuffled": asdict(effect),
        "corruption_margin_25pct": asdict(corruption_margin),
        "false_recall_rate": false_recall_rate,
        "gates": {
            "g1_pass": bool(g1_pass),
            "g2_effect_threshold_pass": bool(g2_effect_pass),
            "g2_beats_random": bool(beats_random),
            "g2_beats_zero": bool(beats_zero),
            "g2_corruption_margin_positive": bool(corruption_positive),
            "g2_false_recall_ok": bool(false_recall_ok),
            "g2_pass": bool(g2_pass),
        },
    }
