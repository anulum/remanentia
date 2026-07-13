# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — world-class scorecard

"""Compose the honest, comparable, category-defining scorecard for one run.

The field's headline numbers are non-comparable: every system uses a different
reader and a different LLM-as-judge, and most report the inflated oracle setting
(roadmap §1; the Zep 84 % → 58 % dispute is the proof). A world-class claim needs
a scorecard that pins what makes a number comparable and reports the axes no
leaderboard scores. This module is that scorecard (roadmap W1 capstone).

Each scorecard records the *setting* (oracle vs realistic full-S — never conflate
them), the *reader* and *judge* models (so two runs are only compared when both
match), and folds in the new-category metrics from their own modules: calibrated
abstention (:mod:`coverage_accuracy`), sovereign no-egress (:mod:`no_egress_audit`),
and lineage-of-belief completeness (:mod:`lineage_completeness`). The result is a
single deterministic, JSON-serialisable record — the artefact REMANENTIA publishes
to prove both a cloud-comparable and a pure-local number on equal footing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from coverage_accuracy import Outcome, risk_coverage
from fleet_recall_scorer import FleetRecallReport
from lineage_completeness import LineageReport
from no_egress_audit import EgressVerdict

Setting = Literal["oracle", "full_s"]


@dataclass(frozen=True)
class RunConfig:
    """What makes a score comparable: the setting and the reader/judge models."""

    setting: Setting
    reader: str  # the answer-generating LLM
    judge: str  # the LLM-as-judge used to mark correctness

    def comparable_to(self, other: RunConfig) -> bool:
        """Two runs are comparable only when setting, reader, and judge all match."""
        return (
            self.setting == other.setting
            and self.reader == other.reader
            and self.judge == other.judge
        )


@dataclass(frozen=True)
class Scorecard:
    """One run's accuracy plus the new-category axes, in comparable form."""

    config: RunConfig
    questions: int
    accuracy: float
    aurc: float
    coverage_at_target: float
    accuracy_target: float
    pure_local: bool
    cloud_calls: int
    lineage_completeness: float
    fleet: FleetRecallReport | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the scorecard."""
        return {
            "setting": self.config.setting,
            "reader": self.config.reader,
            "judge": self.config.judge,
            "questions": self.questions,
            "accuracy": round(self.accuracy, 4),
            "aurc": round(self.aurc, 4),
            "coverage_at_target": round(self.coverage_at_target, 4),
            "accuracy_target": self.accuracy_target,
            "pure_local": self.pure_local,
            "cloud_calls": self.cloud_calls,
            "lineage_completeness": round(self.lineage_completeness, 4),
            "fleet_recall": self.fleet.as_dict() if self.fleet else {"measured": False},
        }


def build_scorecard(
    config: RunConfig,
    outcomes: Sequence[Outcome],
    egress: EgressVerdict,
    lineage: LineageReport,
    *,
    accuracy_target: float = 0.90,
    fleet: FleetRecallReport | None = None,
) -> Scorecard:
    """Fold accuracy + calibrated abstention + no-egress + lineage + fleet recall.

    Parameters
    ----------
    config
        The comparability pin (setting, reader, judge).
    outcomes
        Per-question outcomes for the risk-coverage curve.
    egress
        The sovereign no-egress verdict for the run.
    lineage
        The lineage-of-belief completeness report.
    accuracy_target
        Accuracy level at which retained coverage is reported.
    fleet
        The fleet-fed recall axis (:func:`fleet_recall_scorer.score_fleet_recall`),
        or ``None`` when no query stream was captured — the axis then reports
        ``measured: false`` instead of a fabricated score.
    """
    rc = risk_coverage(outcomes)
    return Scorecard(
        config=config,
        questions=rc.total,
        accuracy=rc.accuracy_at_full_coverage,
        aurc=rc.aurc,
        coverage_at_target=rc.coverage_at_accuracy(accuracy_target),
        accuracy_target=accuracy_target,
        pure_local=egress.pure_local,
        cloud_calls=egress.cloud_calls,
        lineage_completeness=lineage.completeness,
        fleet=fleet,
    )
