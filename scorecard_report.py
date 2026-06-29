# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — world-class scorecard from real benchmark results

"""Turn a real benchmark results file into the comparable world-class report.

This is the W1 harness wired to real output: it reads a LongMemEval results
JSONL (the rows the judge already writes — `judge_label`, `judge_model`,
`question_type`) and produces a single comparable run report that pins what makes
a number comparable (setting, reader, judge) and records the sovereign no-egress
axis (:mod:`no_egress_audit`) from the reader endpoint.

Honesty matters more than a full grid: the current bench emits neither a
per-question confidence nor per-answer provenance, so calibrated abstention and
lineage-of-belief cannot yet be scored from this file — the report flags them
``not measured`` rather than fabricating an uncalibrated curve. Populating them
is a bench-instrumentation follow-up (emit confidence + the cited provenance ids
per answer), after which `coverage_accuracy` and `lineage_completeness` plug
straight in.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from no_egress_audit import audit_endpoints

Setting = Literal["oracle", "full_s"]


@dataclass(frozen=True)
class ResultSummary:
    """Accuracy and the judge models read off a results JSONL."""

    total: int
    correct: int
    accuracy: float
    judge_models: tuple[str, ...]


def parse_results(path: str | Path) -> ResultSummary:
    """Read a LongMemEval results JSONL into an accuracy + judge summary.

    Each row's boolean ``judge_label`` is the correctness verdict; rows without a
    ``judge_label`` (unjudged) are skipped. ``judge_model`` is collected so two
    runs are only compared judge-matched.
    """
    total = 0
    correct = 0
    judges: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if "judge_label" not in row:
            continue
        total += 1
        if bool(row["judge_label"]):
            correct += 1
        judge = row.get("judge_model")
        if isinstance(judge, str) and judge:
            judges.add(judge)
    accuracy = correct / total if total else 0.0
    return ResultSummary(
        total=total, correct=correct, accuracy=accuracy, judge_models=tuple(sorted(judges))
    )


@dataclass(frozen=True)
class RunReport:
    """One run's comparable scorecard, with unmeasured axes flagged honestly."""

    setting: Setting
    reader: str
    judge: str
    questions: int
    accuracy: float
    pure_local: bool
    cloud_calls: int
    abstention_measured: bool  # False until the bench emits per-question confidence
    lineage_measured: bool  # False until the bench emits per-answer provenance

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
            "lineage_measured": self.lineage_measured,
        }


def build_run_report(
    results_path: str | Path,
    *,
    setting: Setting,
    reader: str,
    reader_endpoints: Sequence[object],
) -> RunReport:
    """Build the comparable run report from a results file and the reader endpoints."""
    summary = parse_results(results_path)
    egress = audit_endpoints(reader_endpoints)
    judge = summary.judge_models[0] if summary.judge_models else "unknown"
    return RunReport(
        setting=setting,
        reader=reader,
        judge=judge,
        questions=summary.total,
        accuracy=summary.accuracy,
        pure_local=egress.pure_local,
        cloud_calls=egress.cloud_calls,
        abstention_measured=False,
        lineage_measured=False,
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
    args = parser.parse_args()
    endpoints = args.reader_endpoint or ["https://api.openai.com/v1"]
    report = build_run_report(
        args.results, setting=args.setting, reader=args.reader, reader_endpoints=endpoints
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
