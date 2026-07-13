# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Paired-statistics tests

"""Public paired-effect tests including fail-closed seed completeness."""

from __future__ import annotations

import pytest

from snn_memory.statistics import paired_interval


def test_paired_interval_reports_positive_cue_effect() -> None:
    result = paired_interval({11: [1.0, 1.0], 29: [1.0, 0.0]}, {11: [0.0, 0.0], 29: [0.0, 0.0]}, [11, 29], bootstrap_samples=2000)
    assert result.mean == 0.75
    assert result.lower >= 0.25
    assert result.pairs == 4


def test_paired_interval_rejects_silently_missing_seed() -> None:
    with pytest.raises(ValueError, match="every preregistered seed"):
        paired_interval({11: [1.0]}, {11: [0.0]}, [11, 29])


def test_paired_interval_rejects_unequal_length_seed_results() -> None:
    with pytest.raises(ValueError, match="non-empty and equal length"):
        paired_interval(
            {11: [1.0, 1.0], 29: [1.0]}, {11: [0.0, 0.0], 29: [0.0, 0.0]}, [11, 29]
        )
