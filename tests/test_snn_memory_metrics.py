# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN metric tests

"""Real trajectory-metric tests over concrete rasters and current traces."""

from __future__ import annotations

import numpy as np
import pytest

from snn_memory.metrics import cosine_scores, recurrence_to_input_ratio, temporal_signature


def test_temporal_signature_bins_raster_by_time() -> None:
    raster = np.array([[True, False], [True, True], [False, True], [False, False]], dtype=np.bool_)
    signature = temporal_signature(raster, bins=2)
    assert signature.tolist() == [2.0, 1.0, 0.0, 1.0]


def test_temporal_signature_rejects_non_raster_or_zero_bins() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        temporal_signature(np.zeros(4, dtype=np.bool_), bins=2)
    with pytest.raises(ValueError, match="two-dimensional"):
        temporal_signature(np.zeros((2, 2), dtype=np.bool_), bins=0)


def test_cosine_scores_match_hand_computed_similarity() -> None:
    query = np.array([1.0, 0.0], dtype=np.float64)
    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    scores = cosine_scores(query, candidates)
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0


def test_cosine_scores_reject_shape_mismatch() -> None:
    query = np.array([1.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="do not match"):
        cosine_scores(query, np.array([1.0, 0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="do not match"):
        cosine_scores(query, np.zeros((2, 3), dtype=np.float64))


def test_recurrence_ratio_real_and_zero_input() -> None:
    recurrent = np.array([[1.0, -1.0]], dtype=np.float64)
    inputs = np.array([[2.0, 2.0]], dtype=np.float64)
    assert recurrence_to_input_ratio(recurrent, inputs) == 0.5
    assert recurrence_to_input_ratio(np.array([[1.0]]), np.zeros((1, 1))) == 0.0
