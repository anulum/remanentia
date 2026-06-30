# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for discipline_impact

from __future__ import annotations

import pytest

from discipline_impact import discipline_impact, worst_hit


def _agg(**qtypes: dict[int, float]) -> dict[str, dict[str, float]]:
    """Build a recall aggregate from {qtype: {n: recall}} into mean@N keys."""
    return {q: {f"mean@{n}": v for n, v in rec.items()} for q, rec in qtypes.items()}


class TestDisciplineImpact:
    def test_delta_and_overall_first(self) -> None:
        canonical = _agg(
            overall={1: 0.5, 10: 0.9},
            **{"temporal-reasoning": {1: 0.4, 10: 0.85}, "multi-session": {1: 0.6, 10: 0.95}},
        )
        degraded = _agg(
            overall={1: 0.4, 10: 0.8},
            **{"temporal-reasoning": {1: 0.2, 10: 0.6}, "multi-session": {1: 0.58, 10: 0.93}},
        )
        impacts = discipline_impact(canonical, degraded, ns=(1, 10))
        assert impacts[0].qtype == "overall"  # overall sorts first
        overall = impacts[0]
        assert overall.delta_at(10) == pytest.approx(0.1)
        assert overall.canonical[1] == 0.5
        assert overall.degraded[1] == 0.4

    def test_only_shared_qtypes_compared(self) -> None:
        canonical = _agg(overall={10: 0.9}, **{"only-in-canonical": {10: 0.5}})
        degraded = _agg(overall={10: 0.8}, **{"only-in-degraded": {10: 0.3}})
        impacts = discipline_impact(canonical, degraded, ns=(10,))
        assert [i.qtype for i in impacts] == ["overall"]  # only shared key

    def test_missing_mean_raises(self) -> None:
        canonical = _agg(overall={10: 0.9})
        degraded = _agg(overall={10: 0.8})
        with pytest.raises(KeyError):
            discipline_impact(canonical, degraded, ns=(5,))  # mean@5 absent


class TestWorstHit:
    def test_picks_largest_drop_excluding_overall(self) -> None:
        canonical = _agg(
            overall={10: 0.9},
            **{"temporal-reasoning": {10: 0.85}, "multi-session": {10: 0.95}},
        )
        degraded = _agg(
            overall={10: 0.8},
            **{"temporal-reasoning": {10: 0.55}, "multi-session": {10: 0.92}},
        )
        impacts = discipline_impact(canonical, degraded, ns=(10,))
        worst = worst_hit(impacts, n=10)
        assert worst is not None
        assert worst.qtype == "temporal-reasoning"  # biggest drop (0.30)
        assert worst.delta_at(10) == pytest.approx(0.30)

    def test_none_when_only_overall(self) -> None:
        canonical = _agg(overall={10: 0.9})
        degraded = _agg(overall={10: 0.8})
        impacts = discipline_impact(canonical, degraded, ns=(10,))
        assert worst_hit(impacts) is None
