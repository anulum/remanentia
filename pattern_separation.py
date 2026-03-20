# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Dentate Gyrus Pattern Separation Layer

"""Condition 1 of Theorem 1: Pattern Separation.

Implements the Dasgupta fly algorithm (Science 2017) adapted for
text memory encoding:

1. Input: N-dim embedding-projected pattern (from encoding.py)
2. Expansion: 4x random sparse projection (each expansion neuron
   receives ~6 random inputs from N)
3. k-Winners-Take-All: keep top 5% of expanded neurons active
4. Output: 4N-dim sparse binary pattern with ~5% activation

Properties:
- Two inputs with cosine similarity 0.8 produce patterns with
  overlap ~0.05^2 = 0.25% (vs ~80% without separation)
- Deterministic (fixed seed projection matrix)
- O(N * expansion_ratio * fan_in) computation
"""
from __future__ import annotations

import numpy as np

_DG_PROJ: dict[tuple[int, int, int], np.ndarray] = {}
_DG_SEED = 1337


def _get_dg_projection(
    n_input: int,
    expansion: int = 4,
    fan_in: int = 6,
) -> np.ndarray:
    """Get or create the sparse random projection matrix.

    Each expansion neuron receives exactly `fan_in` random inputs.
    Matrix shape: (n_input * expansion, n_input), sparse binary.
    """
    key = (n_input, expansion, fan_in)
    if key in _DG_PROJ:
        return _DG_PROJ[key]

    n_expanded = n_input * expansion
    rng = np.random.default_rng(_DG_SEED)

    # Sparse binary: each row has exactly fan_in nonzero entries
    proj = np.zeros((n_expanded, n_input), dtype=np.float32)
    for i in range(n_expanded):
        inputs = rng.choice(n_input, size=fan_in, replace=False)
        proj[i, inputs] = 1.0 / np.sqrt(fan_in)

    _DG_PROJ[key] = proj
    return proj


def separate(
    pattern: np.ndarray,
    expansion: int = 4,
    fan_in: int = 6,
    sparsity: float = 0.05,
) -> np.ndarray:
    """Apply DG pattern separation to an input pattern.

    Args:
        pattern: Input pattern, shape (N,), values in [0, 1]
        expansion: Expansion ratio (output is expansion*N neurons)
        fan_in: Number of random inputs per expansion neuron
        sparsity: Fraction of expansion neurons to keep active

    Returns:
        Separated pattern, shape (expansion*N,), sparse binary-ish
    """
    n_input = len(pattern)
    proj = _get_dg_projection(n_input, expansion, fan_in)

    # Project: each expansion neuron computes weighted sum of fan_in inputs
    expanded = proj @ pattern

    # k-Winners-Take-All: keep top sparsity fraction
    n_expanded = len(expanded)
    k = max(1, int(sparsity * n_expanded))
    threshold = np.partition(expanded, -k)[-k]
    result = np.zeros(n_expanded, dtype=np.float32)
    active = expanded >= threshold
    result[active] = expanded[active]

    # Normalize
    mx = result.max()
    if mx > 0:
        result /= mx

    return result


def measure_separation(patterns: list[np.ndarray]) -> dict:
    """Measure how well pattern separation orthogonalizes inputs."""
    n = len(patterns)
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = patterns[i], patterns[j]
            # Overlap = fraction of co-active neurons
            a_active = a > 0
            b_active = b > 0
            if a_active.sum() > 0 and b_active.sum() > 0:
                overlap = (a_active & b_active).sum() / max(a_active.sum(), 1)
                overlaps.append(float(overlap))

    return {
        "mean_overlap": float(np.mean(overlaps)) if overlaps else 0,
        "max_overlap": float(np.max(overlaps)) if overlaps else 0,
        "min_overlap": float(np.min(overlaps)) if overlaps else 0,
        "n_pairs": len(overlaps),
        "sparsity": float(np.mean([float((p > 0).sum() / len(p)) for p in patterns])),
    }
