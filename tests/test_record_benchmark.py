# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for the benchmark accuracy-history recorder
"""Behavioural tests for ``tools/record_benchmark.py``.

Covers aggregation maths (overall + per-category, rounding, empty/degenerate
inputs, missing fields), JSONL load/append round-trips, ledger creation and
absence, git-SHA fallback outside a repository, deterministic record assembly,
the end-to-end ``record_run`` path, and the CLI ``main`` entry point.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "record_benchmark.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("record_benchmark", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rb = _load_module()


def _records(*pairs: tuple[str, bool]) -> list[dict[str, Any]]:
    """Build judged records from ``(question_type, judge_label)`` pairs."""
    return [
        {"question_id": f"q{i}", "question_type": qt, "judge_label": label}
        for i, (qt, label) in enumerate(pairs)
    ]


# ── _pct ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("part", "whole", "expected"),
    [(1, 2, 50.0), (0, 0, 0.0), (3, 3, 100.0), (1, 3, 33.3), (2, 3, 66.7), (0, 5, 0.0)],
)
def test_pct(part: int, whole: int, expected: float) -> None:
    assert rb._pct(part, whole) == expected


# ── summarise ─────────────────────────────────────────────────────


def test_summarise_multi_category_mixed() -> None:
    records = _records(
        ("temporal-reasoning", True),
        ("temporal-reasoning", False),
        ("multi-session", True),
        ("multi-session", True),
        ("multi-session", False),
    )
    summary = rb.summarise(records)
    assert summary["n_total"] == 5
    assert summary["overall_correct"] == 3
    assert summary["overall_accuracy"] == 60.0
    assert summary["by_category"]["temporal-reasoning"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 50.0,
    }
    assert summary["by_category"]["multi-session"] == {
        "correct": 2,
        "total": 3,
        "accuracy": 66.7,
    }


def test_summarise_empty() -> None:
    summary = rb.summarise([])
    assert summary == {
        "n_total": 0,
        "overall_correct": 0,
        "overall_accuracy": 0.0,
        "by_category": {},
    }


def test_summarise_all_correct_and_all_wrong() -> None:
    assert rb.summarise(_records(("a", True), ("a", True)))["overall_accuracy"] == 100.0
    assert rb.summarise(_records(("a", False), ("a", False)))["overall_accuracy"] == 0.0


def test_summarise_missing_question_type_buckets_unknown() -> None:
    summary = rb.summarise([{"judge_label": True}, {"judge_label": False}])
    assert set(summary["by_category"]) == {"unknown"}
    assert summary["by_category"]["unknown"] == {"correct": 1, "total": 2, "accuracy": 50.0}


def test_summarise_absent_or_none_label_counts_incorrect() -> None:
    records = [
        {"question_type": "a"},  # no judge_label
        {"question_type": "a", "judge_label": None},
        {"question_type": "a", "judge_label": True},
    ]
    summary = rb.summarise(records)
    assert summary["overall_correct"] == 1
    assert summary["n_total"] == 3


def test_summarise_categories_sorted() -> None:
    summary = rb.summarise(_records(("zebra", True), ("alpha", True), ("mid", False)))
    assert list(summary["by_category"]) == ["alpha", "mid", "zebra"]


# ── load_results ──────────────────────────────────────────────────


def test_load_results_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"question_id": "x", "judge_label": True, "question_type": "a"})
        + "\n\n"
        + json.dumps({"question_id": "y", "judge_label": False, "question_type": "b"})
        + "\n",
        encoding="utf-8",
    )
    loaded = rb.load_results(path)
    assert len(loaded) == 2
    assert loaded[0]["question_id"] == "x"


def test_load_results_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert rb.load_results(path) == []


# ── git_sha ───────────────────────────────────────────────────────


def test_git_sha_in_repo_returns_string() -> None:
    sha = rb.git_sha()
    assert isinstance(sha, str) and sha


def test_git_sha_outside_repo_is_unknown(tmp_path: Path) -> None:
    assert rb.git_sha(tmp_path) == "unknown"


# ── build_record ──────────────────────────────────────────────────


def test_build_record_shape_and_passthrough() -> None:
    summary = rb.summarise(_records(("a", True), ("a", False)))
    record = rb.build_record(
        summary,
        timestamp="2026-06-20T18:00:00+00:00",
        round_label="R12",
        synth_model="gpt-4o-mini",
        judge_model="gpt-4o-mini",
        seed=42,
        sha="abc1234",
        note="unit",
    )
    assert record["timestamp"] == "2026-06-20T18:00:00+00:00"
    assert record["round"] == "R12"
    assert record["benchmark"] == "longmemeval"
    assert record["git_sha"] == "abc1234"
    assert record["config"] == {"synth_model": "gpt-4o-mini", "judge_model": "gpt-4o-mini", "seed": 42}
    assert record["overall_accuracy"] == 50.0
    assert record["note"] == "unit"


# ── append_record / read_history ──────────────────────────────────


def test_append_and_read_round_trip_in_order(tmp_path: Path) -> None:
    ledger = tmp_path / "nested" / "history.jsonl"
    rb.append_record({"round": "R1", "overall_accuracy": 70.0}, ledger)
    rb.append_record({"round": "R2", "overall_accuracy": 72.0}, ledger)
    history = rb.read_history(ledger)
    assert [r["round"] for r in history] == ["R1", "R2"]
    assert ledger.parent.is_dir()  # parent created on demand


def test_read_history_missing_ledger_returns_empty(tmp_path: Path) -> None:
    assert rb.read_history(tmp_path / "absent.jsonl") == []


# ── record_run (end-to-end) ───────────────────────────────────────


def test_record_run_end_to_end(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            json.dumps(r)
            for r in _records(("temporal-reasoning", True), ("multi-session", False))
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "history.jsonl"
    record = rb.record_run(
        results,
        round_label="R12",
        timestamp="2026-06-20T18:00:00+00:00",
        synth_model="gpt-4o-mini",
        judge_model="gpt-4o-mini",
        seed=42,
        note="P0.4 run 1/3",
        ledger=ledger,
    )
    assert record["overall_accuracy"] == 50.0
    assert record["n_total"] == 2
    persisted = rb.read_history(ledger)
    assert len(persisted) == 1
    assert persisted[0]["round"] == "R12"
    assert persisted[0]["note"] == "P0.4 run 1/3"


def test_record_run_default_timestamp_is_iso(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(_records(("a", True))[0]), encoding="utf-8")
    record = rb.record_run(results, round_label="R0", ledger=tmp_path / "h.jsonl")
    assert "T" in record["timestamp"]  # ISO-8601 date/time separator


# ── main (CLI) ────────────────────────────────────────────────────


def test_main_cli_records_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(json.dumps(r) for r in _records(("a", True), ("a", True), ("b", False))),
        encoding="utf-8",
    )
    ledger = tmp_path / "history.jsonl"
    rc = rb.main(
        [
            str(results),
            "--round",
            "R12",
            "--synth-model",
            "gpt-4o-mini",
            "--judge-model",
            "gpt-4o-mini",
            "--seed",
            "42",
            "--note",
            "cli",
            "--ledger",
            str(ledger),
            "--timestamp",
            "2026-06-20T18:00:00+00:00",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "recorded longmemeval R12" in out
    assert "66.7%" in out  # 2/3 correct
    assert len(rb.read_history(ledger)) == 1
