# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — world-class scorecard from real benchmark results

"""Turn a real benchmark results file into the comparable world-class report.

This is the W1 harness wired to real output: it reads a LongMemEval results
JSONL (the rows the judge writes — `judge_label`, `judge_model`, `question_type`)
and produces a single comparable run report that pins what makes a number
comparable (setting, reader, judge) and records the sovereign no-egress axis
(:mod:`no_egress_audit`) from the reader endpoint.

The two new-category axes activate the moment the bench emits the data, and stay
honestly dark until then — no fabricated curve. When every judged row carries a
numeric ``confidence``, the calibrated-abstention axis is scored via
:mod:`coverage_accuracy` (accuracy at full coverage, AURC, coverage at a target
accuracy). When every judged row carries a ``cited_ids`` list, the
lineage-of-belief axis is scored via :mod:`lineage_completeness` (the fraction of
answers that rest on queryable provenance). A run missing a field reports that
axis ``not measured`` rather than guess.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from coverage_accuracy import Outcome, risk_coverage
from lineage_completeness import AnswerLineage, ProvenanceNode, lineage_completeness
from no_egress_audit import audit_endpoints

Setting = Literal["oracle", "full_s"]


@dataclass(frozen=True)
class ResultSummary:
    """Accuracy, judges, and (when present) the abstention + lineage inputs."""

    total: int
    correct: int
    accuracy: float
    judge_models: tuple[str, ...]
    outcomes: tuple[Outcome, ...]  # non-empty only when every judged row had confidence
    lineages: tuple[AnswerLineage, ...]  # non-empty only when every row had cited_ids
    provenance: tuple[ProvenanceNode, ...]  # origin nodes for the cited ids


def _confidence(value: object) -> float | None:
    """Return a numeric confidence, rejecting bools (``judge_label`` is bool)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_results(path: str | Path) -> ResultSummary:
    """Read a LongMemEval results JSONL into an accuracy + axis-input summary.

    Each row's boolean ``judge_label`` is the correctness verdict; rows without a
    ``judge_label`` (unjudged) are skipped. ``judge_model`` is collected for
    judge-matched comparison. Optional per-row ``confidence`` (float) feeds the
    calibrated-abstention axis and ``cited_ids`` (list) feeds the lineage axis —
    each axis activates only if *every* judged row carries its field.
    """
    total = 0
    correct = 0
    judges: set[str] = set()
    outcomes: list[Outcome] = []
    lineages: list[AnswerLineage] = []
    store: dict[str, ProvenanceNode] = {}
    has_confidence = True
    has_cited = True
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if "judge_label" not in row:
            continue
        total += 1
        is_correct = bool(row["judge_label"])
        if is_correct:
            correct += 1
        judge = row.get("judge_model")
        if isinstance(judge, str) and judge:
            judges.add(judge)

        conf = _confidence(row.get("confidence"))
        if conf is None:
            has_confidence = False
        else:
            outcomes.append(Outcome(correct=is_correct, confidence=conf))

        cited = row.get("cited_ids")
        if isinstance(cited, list):
            cited_ids = tuple(str(c) for c in cited)
            answer_id = str(row.get("question_id", f"q{total}"))
            lineages.append(AnswerLineage(answer_id=answer_id, cited_ids=cited_ids))
            for cid in cited_ids:
                store.setdefault(cid, ProvenanceNode(id=cid, origin=True, parent=None))
        else:
            has_cited = False

    accuracy = correct / total if total else 0.0
    return ResultSummary(
        total=total,
        correct=correct,
        accuracy=accuracy,
        judge_models=tuple(sorted(judges)),
        outcomes=tuple(outcomes) if (total and has_confidence) else (),
        lineages=tuple(lineages) if (total and has_cited) else (),
        provenance=tuple(store.values()),
    )


@dataclass(frozen=True)
class RunReport:
    """One run's comparable scorecard; unmeasured axes are flagged, not guessed."""

    setting: Setting
    reader: str
    judge: str
    questions: int
    accuracy: float
    pure_local: bool
    cloud_calls: int
    abstention_measured: bool
    aurc: float  # area under the risk-coverage curve (0 when not measured)
    coverage_at_target: float  # coverage retained at accuracy_target (0 when not measured)
    accuracy_target: float
    lineage_measured: bool
    lineage_completeness: (
        float  # fraction of answers with queryable provenance (0 when not measured)
    )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the run report."""
        return {
            "setting": self.setting,
            "reader": self.reader,
            "judge": self.judge,
            "questions": self.questions,
            "accuracy": round(self.accuracy, 4),
            "pure_local": self.pure_local,
            "cloud_calls": self.cloud_calls,
            "abstention_measured": self.abstention_measured,
            "aurc": round(self.aurc, 4),
            "coverage_at_target": round(self.coverage_at_target, 4),
            "accuracy_target": self.accuracy_target,
            "lineage_measured": self.lineage_measured,
            "lineage_completeness": round(self.lineage_completeness, 4),
        }


def build_run_report(
    results_path: str | Path,
    *,
    setting: Setting,
    reader: str,
    reader_endpoints: Sequence[object],
    accuracy_target: float = 0.90,
) -> RunReport:
    """Build the comparable run report from a results file and the reader endpoints."""
    summary = parse_results(results_path)
    egress = audit_endpoints(reader_endpoints)
    judge = summary.judge_models[0] if summary.judge_models else "unknown"

    aurc = 0.0
    coverage_at_target = 0.0
    abstention_measured = bool(summary.outcomes)
    if abstention_measured:
        rc = risk_coverage(summary.outcomes)
        aurc = rc.aurc
        coverage_at_target = rc.coverage_at_accuracy(accuracy_target)

    lineage_value = 0.0
    lineage_measured = bool(summary.lineages)
    if lineage_measured:
        store = {node.id: node for node in summary.provenance}
        lineage_value = lineage_completeness(summary.lineages, store).completeness

    return RunReport(
        setting=setting,
        reader=reader,
        judge=judge,
        questions=summary.total,
        accuracy=summary.accuracy,
        pure_local=egress.pure_local,
        cloud_calls=egress.cloud_calls,
        abstention_measured=abstention_measured,
        aurc=aurc,
        coverage_at_target=coverage_at_target,
        accuracy_target=accuracy_target,
        lineage_measured=lineage_measured,
        lineage_completeness=lineage_value,
    )


def main() -> int:  # pragma: no cover - CLI entry point
    """Build and print a world-class run report from a results JSONL."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="remanentia-scorecard",
        description="Build the comparable world-class scorecard from a benchmark results file.",
    )
    parser.add_argument("results", help="LongMemEval results JSONL (judged)")
    parser.add_argument("--setting", choices=["oracle", "full_s"], default="full_s")
    parser.add_argument("--reader", default="gpt-4o-mini", help="answer-generating model")
    parser.add_argument(
        "--reader-endpoint",
        action="append",
        default=[],
        help="reader endpoint(s) for the no-egress audit (repeatable)",
    )
    parser.add_argument("--accuracy-target", type=float, default=0.90)
    args = parser.parse_args()
    endpoints = args.reader_endpoint or ["https://api.openai.com/v1"]
    report = build_run_report(
        args.results,
        setting=args.setting,
        reader=args.reader,
        reader_endpoints=endpoints,
        accuracy_target=args.accuracy_target,
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
