# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — risk–coverage (calibrated-abstention) evaluation

"""Measure calibrated abstention: knowing what you don't know.

Final-answer accuracy alone is the field's only leaderboard metric, but a
world-class memory system must also know *when to abstain* — answer confidently
where it has support, and say "I don't have that" where it does not. No public
memory benchmark scores this. This module is the instrument for that
new-category axis (roadmap W1/W5): it turns per-question outcomes into a
risk–coverage curve and the summary numbers that make abstention comparable.

Given each answered question's correctness and the system's confidence, sorting
by confidence and sweeping a threshold yields, at every *coverage* (fraction of
questions answered rather than abstained), the *accuracy* over the answered
subset. A well-calibrated system's accuracy rises monotonically as coverage
falls — it abstains on exactly the questions it would get wrong. The headline
summaries are: accuracy at full coverage (answer everything), the area under the
risk–coverage curve (AURC; lower is better), and the maximum coverage attainable
at a target accuracy (how much you can answer while staying above a quality bar).

Pure, deterministic, no model calls — it scores result records the benchmark
already produces, so it composes into the honest evaluation harness without any
cloud dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Outcome:
    """One answered question's outcome for risk–coverage scoring."""

    correct: bool
    confidence: float


@dataclass(frozen=True)
class CoveragePoint:
    """Accuracy over the most-confident answered subset at one coverage level."""

    coverage: float  # fraction of questions answered (0, 1]
    answered: int
    accuracy: float  # accuracy over the answered subset
    risk: float  # 1 - accuracy (error rate over the answered subset)


@dataclass(frozen=True)
class RiskCoverage:
    """Risk–coverage curve and its calibrated-abstention summaries."""

    total: int
    points: tuple[CoveragePoint, ...]
    accuracy_at_full_coverage: float
    aurc: float  # area under the risk–coverage curve; lower = better calibrated

    def coverage_at_accuracy(self, target_accuracy: float) -> float:
        """Return the max coverage whose answered subset holds >= target accuracy.

        The honest "how much can I answer while staying above this quality bar?"
        number. Returns ``0.0`` when no non-empty prefix reaches the target.
        """
        best = 0.0
        for point in self.points:
            if point.accuracy >= target_accuracy and point.coverage > best:
                best = point.coverage
        return best


def risk_coverage(outcomes: Sequence[Outcome]) -> RiskCoverage:
    """Build the risk–coverage curve from per-question outcomes.

    Outcomes are ranked by confidence (highest first); each prefix is a coverage
    level whose accuracy is measured over that answered subset. Ties in
    confidence are kept in input order, so a stable upstream ordering gives a
    stable curve.
    """
    if not outcomes:
        return RiskCoverage(total=0, points=(), accuracy_at_full_coverage=0.0, aurc=0.0)

    ranked = sorted(outcomes, key=lambda o: o.confidence, reverse=True)
    total = len(ranked)
    points: list[CoveragePoint] = []
    correct_so_far = 0
    for i, outcome in enumerate(ranked, start=1):
        if outcome.correct:
            correct_so_far += 1
        accuracy = correct_so_far / i
        points.append(
            CoveragePoint(
                coverage=i / total,
                answered=i,
                accuracy=accuracy,
                risk=1.0 - accuracy,
            )
        )

    accuracy_at_full = points[-1].accuracy
    aurc = _trapezoid_aurc(points)
    return RiskCoverage(
        total=total,
        points=tuple(points),
        accuracy_at_full_coverage=accuracy_at_full,
        aurc=aurc,
    )


def _trapezoid_aurc(points: Sequence[CoveragePoint]) -> float:
    """Area under the risk–coverage curve via the trapezoidal rule over coverage."""
    if len(points) == 1:
        return points[0].risk
    area = 0.0
    prev = points[0]
    for point in points[1:]:
        width = point.coverage - prev.coverage
        area += width * (point.risk + prev.risk) / 2.0
        prev = point
    return area
