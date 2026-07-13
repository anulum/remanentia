# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Paired temporal SNN statistics

"""Seed-complete paired effects and deterministic bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedInterval:
    """Mean paired effect and percentile confidence bounds."""

    mean: float
    lower: float
    upper: float
    pairs: int


def paired_interval(
    trained: dict[int, list[float]],
    control: dict[int, list[float]],
    expected_seeds: list[int],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260714,
) -> PairedInterval:
    """Compute a cue-level paired bootstrap while rejecting missing seeds."""
    if set(trained) != set(expected_seeds) or set(control) != set(expected_seeds):
        raise ValueError("result set does not contain every preregistered seed")
    differences: list[float] = []
    for run_seed in expected_seeds:
        left = trained[run_seed]
        right = control[run_seed]
        if not left or len(left) != len(right):
            raise ValueError("paired seed results must be non-empty and equal length")
        differences.extend(a - b for a, b in zip(left, right, strict=True))
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    boot = values[indices].mean(axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return PairedInterval(float(values.mean()), float(lower), float(upper), len(values))
