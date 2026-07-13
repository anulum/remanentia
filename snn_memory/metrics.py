# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN memory metrics

"""Trajectory metrics that preserve spike timing until the scoring boundary."""

from __future__ import annotations

import numpy as np

from snn_memory.state import FloatArray


def temporal_signature(spikes: np.ndarray, bins: int = 8) -> FloatArray:
    """Reduce a raster into neuron-by-time-bin counts at the output boundary."""
    if spikes.ndim != 2 or bins < 1:
        raise ValueError("spikes must be two-dimensional and bins positive")
    chunks = np.array_split(spikes.astype(np.float64), bins, axis=0)
    return np.concatenate([chunk.sum(axis=0) for chunk in chunks])


def cosine_scores(signature: FloatArray, candidates: FloatArray) -> FloatArray:
    """Return finite cosine similarity against stored temporal signatures."""
    if candidates.ndim != 2 or candidates.shape[1] != signature.size:
        raise ValueError("candidate signatures do not match query shape")
    query_norm = np.linalg.norm(signature)
    candidate_norm = np.linalg.norm(candidates, axis=1)
    denominator = candidate_norm * query_norm
    numerator = np.asarray(candidates @ signature, dtype=np.float64)
    return np.asarray(np.divide(
        numerator,
        denominator,
        out=np.zeros(candidates.shape[0], dtype=np.float64),
        where=denominator > 0.0,
    ), dtype=np.float64)


def recurrence_to_input_ratio(recurrent: FloatArray, inputs: FloatArray) -> float:
    """Return mean absolute recurrent current divided by external current."""
    denominator = float(np.mean(np.abs(inputs)))
    return float(np.mean(np.abs(recurrent)) / denominator) if denominator > 0.0 else 0.0
