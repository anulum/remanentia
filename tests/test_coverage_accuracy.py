# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for coverage_accuracy

from __future__ import annotations

import json

from coverage_accuracy import Outcome, _trapezoid_aurc, render_curve_jsonl, risk_coverage


def _calibrated() -> list[Outcome]:
    """High-confidence answers correct, low-confidence answers wrong."""
    return [
        Outcome(correct=True, confidence=0.9),
        Outcome(correct=True, confidence=0.8),
        Outcome(correct=False, confidence=0.3),
        Outcome(correct=False, confidence=0.2),
    ]


class TestRiskCoverageEmpty:
    def test_empty(self) -> None:
        rc = risk_coverage([])
        assert rc.total == 0
        assert rc.points == ()
        assert rc.accuracy_at_full_coverage == 0.0
        assert rc.aurc == 0.0
        assert rc.coverage_at_accuracy(1.0) == 0.0


class TestRiskCoverageCurve:
    def test_points_ranked_by_confidence(self) -> None:
        rc = risk_coverage(_calibrated())
        assert rc.total == 4
        # coverage grows 0.25 .. 1.0; accuracy declines as low-confidence wrongs enter
        accs = [round(p.accuracy, 3) for p in rc.points]
        assert accs == [1.0, 1.0, 0.667, 0.5]
        assert [round(p.coverage, 2) for p in rc.points] == [0.25, 0.5, 0.75, 1.0]
        assert rc.points[-1].risk == 0.5

    def test_accuracy_at_full_coverage(self) -> None:
        assert risk_coverage(_calibrated()).accuracy_at_full_coverage == 0.5

    def test_all_correct(self) -> None:
        rc = risk_coverage([Outcome(True, 0.5), Outcome(True, 0.1)])
        assert rc.accuracy_at_full_coverage == 1.0
        assert rc.aurc == 0.0
        assert rc.coverage_at_accuracy(1.0) == 1.0

    def test_aurc_is_low_for_calibrated(self) -> None:
        rc = risk_coverage(_calibrated())
        # trapezoid over (cov,risk): mostly zero risk at high confidence
        assert 0.13 < rc.aurc < 0.16


class TestCoverageAtAccuracy:
    def test_target_met_returns_max_coverage(self) -> None:
        rc = risk_coverage(_calibrated())
        assert rc.coverage_at_accuracy(1.0) == 0.5  # prefix of 2 all-correct
        assert rc.coverage_at_accuracy(0.6) == 0.75  # 2/3 still >= 0.6

    def test_target_never_met_returns_zero(self) -> None:
        rc = risk_coverage([Outcome(False, 0.9), Outcome(False, 0.1)])
        assert rc.coverage_at_accuracy(0.5) == 0.0


class TestTrapezoidAurc:
    def test_single_point_returns_risk(self) -> None:
        # len==1 branch: AURC degenerates to the single point's risk
        rc_correct = risk_coverage([Outcome(True, 0.5)])
        assert rc_correct.aurc == 0.0
        rc_wrong = risk_coverage([Outcome(False, 0.5)])
        assert rc_wrong.aurc == 1.0

    def test_direct_single_point(self) -> None:
        rc = risk_coverage([Outcome(True, 0.5)])
        assert _trapezoid_aurc(rc.points) == 0.0


class TestRenderCurveJsonl:
    def test_empty_curve_is_empty_string(self) -> None:
        assert render_curve_jsonl(risk_coverage([])) == ""

    def test_one_line_per_point_with_shape(self) -> None:
        rc = risk_coverage(_calibrated())
        lines = render_curve_jsonl(rc).splitlines()
        assert len(lines) == len(rc.points)
        first = json.loads(lines[0])
        assert set(first) == {"coverage", "answered", "accuracy", "risk"}

    def test_coverage_ascending_and_full_point_matches_summary(self) -> None:
        rc = risk_coverage(_calibrated())
        rows = [json.loads(x) for x in render_curve_jsonl(rc).splitlines()]
        coverages = [r["coverage"] for r in rows]
        assert coverages == sorted(coverages)  # ascending to full coverage
        assert rows[-1]["coverage"] == 1.0
        assert rows[-1]["accuracy"] == round(rc.accuracy_at_full_coverage, 6)

    def test_risk_is_one_minus_accuracy(self) -> None:
        rc = risk_coverage(_calibrated())
        for row in (json.loads(x) for x in render_curve_jsonl(rc).splitlines()):
            assert row["risk"] == round(1.0 - row["accuracy"], 6)
