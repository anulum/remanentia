# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for full-S retrieval diagnostic taxonomy

"""Behavioural tests for :mod:`full_s_diagnostics`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from full_s_diagnostics import (
    FullSFailureType,
    classify_full_s_record,
    format_full_s_summary,
    load_jsonl,
    main,
    print_full_s_diagnostic_report,
    summarize_full_s_outcomes,
)


def _record(
    *,
    qid: str = "q",
    qtype: str = "multi-session",
    judge_label: bool = False,
    recall: float | None = 1.0,
    selected: list[str] | None = None,
    missing: list[str] | None = None,
    session_limited: list[str] | None = None,
    budget_dropped: list[str] | None = None,
) -> dict[str, object]:
    """Build a judged result row with real ``retrieval_diagnostics`` shape."""
    return {
        "question_id": qid,
        "question_type": qtype,
        "hypothesis": "answer",
        "judge_label": judge_label,
        "retrieval_diagnostics": {
            "answer_session_recall": recall,
            "selected_answer_session_ids": selected or [],
            "missing_answer_session_ids": missing or [],
            "session_limited_answer_session_ids": session_limited or [],
            "budget_dropped_answer_session_ids": budget_dropped or [],
        },
    }


def test_classify_full_s_record() -> None:
    """Classifier separates correct, synthesis, retrieval, and missing-data rows."""
    cases: list[tuple[dict[str, object], FullSFailureType]] = [
        (_record(judge_label=True), FullSFailureType.CORRECT),
        (_record(recall=1.0), FullSFailureType.SYNTHESIS_FAILURE),
        (
            _record(recall=0.5, missing=["s2"], session_limited=["s2"]),
            FullSFailureType.SESSION_LIMIT_MISS,
        ),
        (
            _record(recall=0.5, missing=["s2"], budget_dropped=["s2"]),
            FullSFailureType.CHAR_BUDGET_MISS,
        ),
        (_record(recall=0.5, missing=["s2"]), FullSFailureType.RETRIEVAL_RANKING_MISS),
        (_record(recall=None), FullSFailureType.MISSING_ANSWER_SESSIONS),
        (
            {"question_id": "q", "question_type": "multi-session", "judge_label": False},
            FullSFailureType.MISSING_DIAGNOSTICS,
        ),
    ]
    for record, expected in cases:
        outcome = classify_full_s_record(record)

        assert outcome.failure_type is expected


def test_classify_handles_malformed_diagnostic_fields() -> None:
    """Malformed JSON fields fail into the missing-answer-session bucket."""
    outcome = classify_full_s_record(
        {
            "question_id": "q",
            "question_type": "multi-session",
            "judge_label": False,
            "retrieval_diagnostics": {
                "answer_session_recall": "not-a-number",
                "missing_answer_session_ids": "not-a-list",
            },
        }
    )

    assert outcome.failure_type is FullSFailureType.MISSING_ANSWER_SESSIONS
    assert outcome.missing_answer_session_ids == ()


def test_classify_preserves_failure_context() -> None:
    """Outcome keeps the question identity and missing answer-session IDs."""
    outcome = classify_full_s_record(
        _record(
            qid="q-miss",
            qtype="temporal-reasoning",
            recall=0.5,
            missing=["gold-a"],
        )
    )

    assert outcome.question_id == "q-miss"
    assert outcome.question_type == "temporal-reasoning"
    assert outcome.answer_session_recall == pytest.approx(0.5)
    assert outcome.missing_answer_session_ids == ("gold-a",)
    assert outcome.is_failure


def test_summarize_full_s_outcomes_groups_overall_and_qtype() -> None:
    """Summary table counts taxonomy buckets by qtype and overall."""
    records = [
        _record(qid="q1", qtype="multi-session", recall=1.0),
        _record(qid="q2", qtype="multi-session", recall=0.5, missing=["s2"]),
        _record(qid="q3", qtype="temporal-reasoning", judge_label=True),
    ]

    summary = summarize_full_s_outcomes(records)

    assert list(summary) == ["overall", "multi-session", "temporal-reasoning"]
    assert summary["overall"][FullSFailureType.SYNTHESIS_FAILURE] == 1
    assert summary["overall"][FullSFailureType.RETRIEVAL_RANKING_MISS] == 1
    assert summary["overall"][FullSFailureType.CORRECT] == 1
    assert summary["multi-session"][FullSFailureType.SYNTHESIS_FAILURE] == 1


def test_format_full_s_summary_renders_operator_table() -> None:
    """Formatter exposes the key taxonomy columns for operator output."""
    summary = summarize_full_s_outcomes(
        [
            _record(recall=1.0),
            _record(recall=0.5, missing=["s2"], budget_dropped=["s2"]),
        ]
    )

    rendered = format_full_s_summary(summary)

    assert "Full-S retrieval diagnostic taxonomy" in rendered
    assert "overall" in rendered
    assert "synthesis_fa" in rendered
    assert "char_budget_" in rendered


def test_format_empty_summary() -> None:
    """Empty inputs render as an explicit no-diagnostics message."""
    assert format_full_s_summary({}) == "No full-S retrieval diagnostics found."


def test_print_full_s_diagnostic_report_skips_rows_without_diagnostics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark reporter stays quiet for oracle or legacy result files."""
    print_full_s_diagnostic_report(
        [{"question_id": "q", "question_type": "multi-session", "judge_label": False}]
    )

    assert capsys.readouterr().out == ""


def test_print_full_s_diagnostic_report_outputs_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark reporter prints the taxonomy when diagnostics are present."""
    print_full_s_diagnostic_report([_record(recall=1.0)])

    assert "Full-S retrieval diagnostic taxonomy" in capsys.readouterr().out


def test_load_jsonl_and_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI reads real JSONL rows and prints the same operator table."""
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(_record(qid="q1", recall=1.0))
        + "\n\n"
        + json.dumps(_record(qid="q2", recall=0.5, missing=["s2"]))
        + "\n",
        encoding="utf-8",
    )

    loaded = load_jsonl(path)
    exit_code = main([str(path)])

    assert len(loaded) == 2
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Full-S retrieval diagnostic taxonomy" in output
    assert "retrieval_ra" in output
