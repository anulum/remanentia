# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — full-S retrieval diagnostic taxonomy

"""Classify full-S benchmark failures from recorded retrieval diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast


class FullSFailureType(str, Enum):
    """Failure classes derived from judged full-S retrieval diagnostics."""

    CORRECT = "correct"
    SYNTHESIS_FAILURE = "synthesis_failure"
    SESSION_LIMIT_MISS = "session_limit_miss"
    CHAR_BUDGET_MISS = "char_budget_miss"
    RETRIEVAL_RANKING_MISS = "retrieval_ranking_miss"
    MISSING_ANSWER_SESSIONS = "missing_answer_sessions"
    MISSING_DIAGNOSTICS = "missing_diagnostics"


@dataclass(frozen=True)
class FullSFailureOutcome:
    """Per-question full-S retrieval/synthesis classification.

    Parameters
    ----------
    question_id:
        LongMemEval question identifier.
    question_type:
        LongMemEval category for per-type grouping.
    failure_type:
        Retrieval/synthesis class assigned from judge and diagnostics.
    answer_session_recall:
        Fraction of gold answer sessions selected into the reader context, or
        ``None`` when the row has no answer sessions or no diagnostics.
    missing_answer_session_ids:
        Gold answer-session IDs absent from the reader context.
    """

    question_id: str
    question_type: str
    failure_type: FullSFailureType
    answer_session_recall: float | None
    missing_answer_session_ids: tuple[str, ...]

    @property
    def is_failure(self) -> bool:
        """Return ``True`` when the judged answer was incorrect."""
        return self.failure_type is not FullSFailureType.CORRECT


SummaryTable = dict[str, dict[FullSFailureType, int]]


def classify_full_s_record(record: Mapping[str, object]) -> FullSFailureOutcome:
    """Classify one judged benchmark result from its retrieval diagnostics.

    Correct rows remain ``correct`` regardless of diagnostics. Incorrect rows
    with complete answer-session recall are synthesis failures; incorrect rows
    with missed answer sessions are split by whether the gold session was
    removed by the session cap, removed by the character budget, or never
    retrieved into the candidate set.
    """
    question_id = str(record.get("question_id", "unknown"))
    question_type = str(record.get("question_type", "unknown"))
    if bool(record.get("judge_label")):
        return FullSFailureOutcome(
            question_id=question_id,
            question_type=question_type,
            failure_type=FullSFailureType.CORRECT,
            answer_session_recall=None,
            missing_answer_session_ids=(),
        )

    diagnostics_obj = record.get("retrieval_diagnostics")
    if not isinstance(diagnostics_obj, Mapping):
        return FullSFailureOutcome(
            question_id=question_id,
            question_type=question_type,
            failure_type=FullSFailureType.MISSING_DIAGNOSTICS,
            answer_session_recall=None,
            missing_answer_session_ids=(),
        )

    diagnostics = cast(Mapping[str, object], diagnostics_obj)
    answer_session_recall = _optional_float(diagnostics.get("answer_session_recall"))
    missing_answer_ids = tuple(_str_list(diagnostics.get("missing_answer_session_ids")))
    session_limit_ids = _str_list(diagnostics.get("session_limited_answer_session_ids"))
    budget_ids = _str_list(diagnostics.get("budget_dropped_answer_session_ids"))

    if answer_session_recall is None:
        failure_type = FullSFailureType.MISSING_ANSWER_SESSIONS
    elif answer_session_recall >= 1.0:
        failure_type = FullSFailureType.SYNTHESIS_FAILURE
    elif session_limit_ids:
        failure_type = FullSFailureType.SESSION_LIMIT_MISS
    elif budget_ids:
        failure_type = FullSFailureType.CHAR_BUDGET_MISS
    else:
        failure_type = FullSFailureType.RETRIEVAL_RANKING_MISS

    return FullSFailureOutcome(
        question_id=question_id,
        question_type=question_type,
        failure_type=failure_type,
        answer_session_recall=answer_session_recall,
        missing_answer_session_ids=missing_answer_ids,
    )


def summarize_full_s_outcomes(records: Sequence[Mapping[str, object]]) -> SummaryTable:
    """Aggregate full-S diagnostic outcomes by question type and overall."""
    table: dict[str, Counter[FullSFailureType]] = defaultdict(Counter)
    for record in records:
        outcome = classify_full_s_record(record)
        table[outcome.question_type][outcome.failure_type] += 1
        table["overall"][outcome.failure_type] += 1
    return {
        question_type: dict(counter)
        for question_type, counter in sorted(
            table.items(), key=lambda item: _summary_sort_key(item[0])
        )
    }


def format_full_s_summary(summary: SummaryTable) -> str:
    """Format a full-S diagnostic summary as an operator-readable table."""
    if not summary:
        return "No full-S retrieval diagnostics found."

    columns = [
        FullSFailureType.CORRECT,
        FullSFailureType.SYNTHESIS_FAILURE,
        FullSFailureType.RETRIEVAL_RANKING_MISS,
        FullSFailureType.SESSION_LIMIT_MISS,
        FullSFailureType.CHAR_BUDGET_MISS,
        FullSFailureType.MISSING_ANSWER_SESSIONS,
        FullSFailureType.MISSING_DIAGNOSTICS,
    ]
    header = f"{'qtype':<28}{'n':>5}" + "".join(f"{col.value[:12]:>14}" for col in columns)
    lines = ["Full-S retrieval diagnostic taxonomy", header]
    for question_type, counts in summary.items():
        total = sum(counts.values())
        cells = "".join(f"{counts.get(col, 0):>14}" for col in columns)
        lines.append(f"{question_type:<28}{total:>5}{cells}")
    return "\n".join(lines)


def print_full_s_diagnostic_report(records: Sequence[Mapping[str, object]]) -> None:
    """Print the full-S diagnostic taxonomy when result rows carry diagnostics."""
    diagnostic_rows = [
        record for record in records if isinstance(record.get("retrieval_diagnostics"), Mapping)
    ]
    if not diagnostic_rows:
        return
    print()
    print(format_full_s_summary(summarize_full_s_outcomes(diagnostic_rows)))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load non-empty JSONL objects from *path*."""
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(cast(dict[str, object], json.loads(line)))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full-S diagnostic taxonomy CLI."""
    parser = argparse.ArgumentParser(
        prog="remanentia-full-s-diagnostics",
        description="Summarise full-S LongMemEval retrieval diagnostics from JSONL results.",
    )
    parser.add_argument("results_jsonl", type=Path, help="judged LongMemEval results JSONL")
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = load_jsonl(args.results_jsonl)
    print(format_full_s_summary(summarize_full_s_outcomes(rows)))
    return 0


def _str_list(value: object) -> list[str]:
    """Return a string list only when *value* is a list-like JSON field."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_float(value: object) -> float | None:
    """Parse a JSON numeric field that may be ``null`` or absent."""
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _summary_sort_key(question_type: str) -> tuple[int, str]:
    """Keep the overall row first and sort question types alphabetically."""
    return (0 if question_type == "overall" else 1, question_type)


if __name__ == "__main__":
    raise SystemExit(main())
