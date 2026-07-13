# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real-corpus tests for sparse BM25 scoring

"""Exercise the production sparse scorer on concrete tokenized corpora."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from REMANENTIA.memory_sparse_scoring import (  # type: ignore[import]
    build_weight_index,
    score_uncached,
    score_weight_index,
)


def corpus() -> tuple[
    dict[str, list[int]],
    dict[str, float],
    list[dict[str, int]],
    NDArray[np.float32],
]:
    """Return a concrete two-paragraph sparse retrieval corpus."""
    return (
        {"plasma": [0, 1], "tokamak": [0], "memory": [1]},
        {"plasma": 0.7, "tokamak": 1.4, "memory": 1.4},
        [{"plasma": 2, "tokamak": 1}, {"plasma": 1, "memory": 2}],
        np.array([3, 3], dtype=np.float32),
    )


def test_cached_and_uncached_scoring_agree_on_a_real_token_corpus() -> None:
    """Precomputed and direct BM25 paths produce identical document rankings."""
    inverted, idf, counts, lengths = corpus()
    weights = build_weight_index(inverted, idf, counts, lengths, 3.0)

    cached = score_weight_index({"plasma", "tokamak"}, set(), weights)
    uncached = score_uncached({"plasma", "tokamak"}, set(), inverted, idf, counts, lengths, 3.0)

    assert cached == pytest.approx(uncached)
    assert cached[0] > cached[1]
    assert set(cached) == {0, 1}


def test_weight_index_skips_zero_idf_and_invalid_postings() -> None:
    """Precomputation ignores terms that cannot address a valid paragraph."""
    weights = build_weight_index(
        {"zero": [0], "negative": [-1], "stale": [4], "valid": [0]},
        {"zero": 0.0, "negative": 1.0, "stale": 1.0, "valid": 1.0},
        [{"valid": 1}],
        np.array([1], dtype=np.float32),
        0.0,
    )

    assert set(weights) == {"valid"}
    assert weights["valid"][0][0] == 0
    assert weights["valid"][0][1] > 0.0


def test_weight_index_uses_default_frequency_when_counts_are_absent() -> None:
    """A valid posting remains scoreable while count materialization catches up."""
    weights = build_weight_index(
        {"alpha": [0, 1]},
        {"alpha": 1.0},
        [{"alpha": 3}],
        np.array([3, 1], dtype=np.float32),
        2.0,
    )

    assert len(weights["alpha"]) == 2
    assert dict(weights["alpha"])[0] > dict(weights["alpha"])[1]


def test_cached_scoring_respects_filters_and_unknown_terms() -> None:
    """Filtering removes only the selected paragraph from accumulated postings."""
    inverted, idf, counts, lengths = corpus()
    weights = build_weight_index(inverted, idf, counts, lengths, 3.0)

    scores = score_weight_index({"plasma", "unknown"}, {0}, weights)

    assert set(scores) == {1}
    assert scores[1] > 0.0


def test_uncached_scoring_rejects_missing_zero_filtered_and_stale_postings() -> None:
    """Direct scoring fails closed on every non-addressable sparse posting."""
    scores = score_uncached(
        {"missing", "zero", "alpha"},
        {0},
        {"missing": [], "zero": [0], "alpha": [-1, 0, 1, 8]},
        {"zero": 0.0, "alpha": 1.0},
        [{"alpha": 2}],
        np.array([2, 1], dtype=np.float32),
        1.5,
    )

    assert set(scores) == {1}
    assert scores[1] > 0.0
