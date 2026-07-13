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

The new-category axes activate the moment their data exists, and stay honestly
dark until then — no fabricated curve. When every judged row carries a numeric
``confidence``, the calibrated-abstention axis is scored via
:mod:`coverage_accuracy` (accuracy at full coverage, AURC, coverage at a target
accuracy). When every judged row carries a ``cited_ids`` list, *citation
presence* is scored: the fraction of answers that cited at least one memory.
The lineage-of-belief axis (:mod:`lineage_completeness` — every cited id
resolves to a record whose chain reaches an originating write) additionally
requires a real provenance store; a results file alone cannot prove
queryability, so without one the axis reports ``not measured`` rather than
score citations against a store synthesised from the citations themselves.
The fleet-fed recall axis (:mod:`fleet_recall_scorer`) requires the recall
query-stream ledger and follows the same rule: no ledger, no score.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from coverage_accuracy import Outcome, risk_coverage
from fleet_recall_scorer import FleetRecallReport, report_from_ledger
from lineage_completeness import AnswerLineage, ProvenanceNode, lineage_completeness
from no_egress_audit import audit_endpoints

Setting = Literal["oracle", "full_s"]


@dataclass(frozen=True)
class ResultSummary:
    """Accuracy, judges, and (when present) the abstention + citation inputs."""

    total: int
    correct: int
    accuracy: float
    judge_models: tuple[str, ...]
    outcomes: tuple[Outcome, ...]  # non-empty only when every judged row had confidence
    lineages: tuple[AnswerLineage, ...]  # non-empty only when every row had cited_ids


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
    calibrated-abstention axis and ``cited_ids`` (list) feeds the citation and
    lineage axes — each axis activates only if *every* judged row carries its
    field. The cited ids are collected as claims to verify, never turned into
    provenance records themselves: only a real store can prove they resolve.
    """
    total = 0
    correct = 0
    judges: set[str] = set()
    outcomes: list[Outcome] = []
    lineages: list[AnswerLineage] = []
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
    )


def load_provenance_store(path: str | Path) -> dict[str, ProvenanceNode]:
    """Load a real provenance store from a JSONL of provenance nodes.

    Each line is an object with a non-empty string ``id``, an optional boolean
    ``origin`` (default false), and an optional string ``parent`` (default
    null). This is the store the lineage axis verifies cited ids against —
    produced by the memory pipeline, not derived from the results file.
    """
    store: dict[str, ProvenanceNode] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        node_id = row.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"provenance node needs a non-empty string id: {line!r}")
        origin = row.get("origin", False)
        if not isinstance(origin, bool):
            raise ValueError(f"provenance node {node_id!r} origin must be a boolean")
        parent = row.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"provenance node {node_id!r} parent must be a string or null")
        store[node_id] = ProvenanceNode(id=node_id, origin=origin, parent=parent)
    return store


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
    citation_measured: bool
    citation_presence: float  # fraction of answers citing >= 1 memory (0 when not measured)
    lineage_measured: bool
    lineage_completeness: (
        float  # fraction of answers with queryable provenance (0 when not measured)
    )
    fleet: FleetRecallReport | None = None  # fleet-fed recall axis (None = not measured)

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
            "citation_measured": self.citation_measured,
            "citation_presence": round(self.citation_presence, 4),
            "lineage_measured": self.lineage_measured,
            "lineage_completeness": round(self.lineage_completeness, 4),
            "fleet_recall": self.fleet.as_dict() if self.fleet else {"measured": False},
        }


def build_run_report(
    results_path: str | Path,
    *,
    setting: Setting,
    reader: str,
    reader_endpoints: Sequence[object],
    accuracy_target: float = 0.90,
    provenance_store: Mapping[str, ProvenanceNode] | None = None,
    recall_ledger: str | Path | None = None,
) -> RunReport:
    """Build the comparable run report from a results file and the reader endpoints.

    Citation presence (did each answer cite at least one memory?) needs only the
    results file. Lineage completeness (does every cited id resolve to a record
    whose chain reaches an originating write?) additionally needs
    *provenance_store* — the real store the memory pipeline maintains. Without
    it the lineage axis stays honestly dark: scoring citations against a store
    built from the citations themselves would mark every cited id an origin and
    the axis could never fail, so a dangling citation would go undetected.
    The fleet-fed recall axis likewise needs *recall_ledger* — the query-stream
    JSONL the production recall path appends to; without it (or when the ledger
    holds no query records) the axis reports ``measured: false``.
    """
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

    citation_measured = bool(summary.lineages)
    citation_presence = 0.0
    if citation_measured:
        citing = sum(1 for lineage in summary.lineages if lineage.cited_ids)
        citation_presence = citing / len(summary.lineages)

    lineage_value = 0.0
    lineage_measured = citation_measured and provenance_store is not None
    if lineage_measured:
        assert provenance_store is not None  # narrowed by lineage_measured
        lineage_value = lineage_completeness(summary.lineages, provenance_store).completeness

    fleet: FleetRecallReport | None = None
    if recall_ledger is not None:
        candidate = report_from_ledger(recall_ledger)
        fleet = candidate if candidate.measured else None

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
        citation_measured=citation_measured,
        citation_presence=citation_presence,
        lineage_measured=lineage_measured,
        lineage_completeness=lineage_value,
        fleet=fleet,
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
    parser.add_argument(
        "--provenance-store",
        default=None,
        help="provenance-node JSONL to verify cited ids against (enables the lineage axis)",
    )
    parser.add_argument(
        "--recall-ledger",
        default=None,
        help="recall query-stream JSONL (recall_ledger) enabling the fleet-fed recall axis",
    )
    args = parser.parse_args()
    endpoints = args.reader_endpoint or ["https://api.openai.com/v1"]
    store = load_provenance_store(args.provenance_store) if args.provenance_store else None
    report = build_run_report(
        args.results,
        setting=args.setting,
        reader=args.reader,
        reader_endpoints=endpoints,
        accuracy_target=args.accuracy_target,
        provenance_store=store,
        recall_ledger=args.recall_ledger,
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
