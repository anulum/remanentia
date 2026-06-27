# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — benchmark evidence report tests

"""Tests for benchmark score, latency, token, and prompt-hash reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, cast

import pytest

from benchmark_evidence import build_judge_evidence, prompt_sha256
from benchmark_report import load_benchmark_payload, main, report_from_path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write benchmark rows as JSONL for real loader/report tests."""
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _map(value: object) -> Mapping[str, object]:
    """Assert and cast nested JSON objects for strict-mypy test assertions."""
    assert isinstance(value, dict)
    return cast(Mapping[str, object], value)


class _UsageObject:
    """Attribute-style usage object matching hosted-client response metadata."""

    prompt_tokens = 5
    completion_tokens = 1
    total_tokens = 6


def test_build_judge_evidence_hashes_prompt_and_usage() -> None:
    """Judge metadata uses stable prompt hashes and preserves API usage counts."""
    evidence = build_judge_evidence(
        "Question: q\nCorrect Answer: a\nModel Response: b",
        model="gpt-4o-mini",
        max_tokens=10,
        latency_ms=42.5,
        usage={"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
    )
    assert evidence == {
        "judge_model": "gpt-4o-mini",
        "judge_max_tokens": 10,
        "judge_prompt_sha256": hashlib.sha256(
            b"Question: q\nCorrect Answer: a\nModel Response: b"
        ).hexdigest(),
        "judge_prompt_chars": 47,
        "judge_prompt_tokens_estimate": 8,
        "judge_latency_ms": 42.5,
        "judge_prompt_tokens": 11,
        "judge_completion_tokens": 2,
        "judge_total_tokens": 13,
    }


def test_build_judge_evidence_accepts_usage_objects() -> None:
    """Attribute-style usage objects are preserved like mapping usage data."""
    evidence = build_judge_evidence(
        "Prompt body",
        model="judge",
        max_tokens=3,
        latency_ms=1.23456,
        usage=_UsageObject(),
    )
    assert evidence["judge_prompt_tokens"] == 5
    assert evidence["judge_completion_tokens"] == 1
    assert evidence["judge_total_tokens"] == 6
    assert evidence["judge_latency_ms"] == 1.235


def test_build_judge_evidence_rejects_bool_usage_and_accepts_integral_float() -> None:
    """Usage coercion avoids bools while accepting integral float counters."""
    evidence = build_judge_evidence(
        "Prompt body",
        model="judge",
        max_tokens=3,
        latency_ms=1.0,
        usage={"prompt_tokens": True, "completion_tokens": 1.0, "total_tokens": 6.5},
    )
    assert "judge_prompt_tokens" not in evidence
    assert evidence["judge_completion_tokens"] == 1
    assert "judge_total_tokens" not in evidence


def test_report_from_longmemeval_jsonl_groups_scores_and_runtime(tmp_path: Path) -> None:
    """JSONL reports aggregate real judged rows by question type."""
    results = tmp_path / "longmemeval.results.jsonl"
    _write_jsonl(
        results,
        [
            {
                "question_id": "q1",
                "question_type": "temporal-reasoning",
                "judge_label": True,
                "judge_model": "gpt-4o-mini",
                "judge_prompt_sha256": "b" * 64,
                "judge_prompt_tokens_estimate": 20,
                "judge_completion_tokens": 1,
                "judge_latency_ms": 10.0,
            },
            {
                "question_id": "q2",
                "question_type": "temporal-reasoning",
                "judge_label": False,
                "judge_model": "gpt-4o-mini",
                "judge_prompt_sha256": "a" * 64,
                "judge_prompt_tokens_estimate": 40,
                "judge_completion_tokens": 1,
                "judge_latency_ms": 30.0,
            },
            {
                "question_id": "q3",
                "question_type": "multi-session",
                "judge_label": True,
                "judge_model": "gpt-4o-mini",
                "judge_prompt": "judge prompt without precomputed hash",
                "judge_prompt_tokens_estimate": 10,
                "judge_completion_tokens": 1,
                "judge_latency_ms": 20.0,
            },
        ],
    )

    report = report_from_path(
        results,
        benchmark="longmemeval",
        generated_at="2026-06-27T13:00:00+00:00",
    )

    assert report["schema_version"] == 1
    assert report["benchmark"] == "longmemeval"
    assert report["source_format"] == "jsonl"
    assert report["n_records"] == 3
    assert report["score"] == {"correct": 2, "total": 3, "accuracy": 66.7}
    by_category = _map(report["by_category"])
    runtime = _map(report["runtime"])
    latency = _map(runtime["latency_ms"])
    tokens = _map(report["tokens"])
    judge = _map(report["judge"])
    assert by_category["temporal-reasoning"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 50.0,
    }
    assert latency == {"mean": 20.0, "p50": 20.0, "p95": 30.0}
    assert tokens["judge_prompt_estimate"] == 70
    assert tokens["judge_completion"] == 3
    assert judge["models"] == ["gpt-4o-mini"]
    assert judge["prompt_sha256"] == sorted(
        [
            "a" * 64,
            "b" * 64,
            prompt_sha256("judge prompt without precomputed hash"),
        ]
    )


def test_jsonl_report_handles_empty_rows_even_median_and_prompt_fallback(
    tmp_path: Path,
) -> None:
    """Rows without exact usage still get deterministic prompt-token evidence."""
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n"
        + json.dumps(
            {
                "question_id": "q1",
                "question_type": "x",
                "judge_label": True,
                "judge_prompt": "one two three",
                "judge_latency_ms": 10,
                "judge_completion_tokens": 1.0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "question_id": "q2",
                "question_type": "x",
                "judge_label": False,
                "judge_prompt_tokens": 2.0,
                "judge_latency_ms": 30,
                "judge_completion_tokens": True,
            }
        )
        + "\n"
        + json.dumps(
            {
                "question_id": "q3",
                "category": "fallback-category",
                "judge_label": False,
                "judge_prompt_tokens": True,
                "judge_prompt_tokens_estimate": 4,
                "judge_latency_ms": True,
            }
        )
        + "\n"
        + json.dumps({"question_id": "q4", "judge_label": False})
        + "\n",
        encoding="utf-8",
    )

    report = report_from_path(results, benchmark="longmemeval", generated_at=None)
    runtime = _map(report["runtime"])
    latency = _map(runtime["latency_ms"])
    tokens = _map(report["tokens"])

    assert report["source_format"] == "jsonl"
    assert isinstance(report["generated_at"], str)
    assert latency == {"mean": 20.0, "p50": 20.0, "p95": 30.0}
    assert tokens == {"judge_prompt_estimate": 9, "judge_completion": 1}


def test_json_payload_can_be_a_row_list(tmp_path: Path) -> None:
    """JSON list payloads use the same row summariser as JSONL files."""
    source = tmp_path / "rows.json"
    source.write_text(
        json.dumps([{"question_id": "q", "question_type": "x", "judge_label": True}]),
        encoding="utf-8",
    )
    report = report_from_path(source, benchmark="row-json", generated_at="fixed")
    assert report["source_format"] == "json"
    assert report["score"] == {"correct": 1, "total": 1, "accuracy": 100.0}


def test_report_from_locomo_json_summary_preserves_method(tmp_path: Path) -> None:
    """LOCOMO's committed JSON summary is accepted without row-level samples."""
    source = tmp_path / "locomo_results.json"
    source.write_text(
        json.dumps(
            {
                "total_correct": 3,
                "total_tested": 4,
                "accuracy": 75.0,
                "by_category": {
                    "single-hop": {"correct": 2, "total": 2},
                    "temporal": {"correct": 1, "total": 2},
                },
                "elapsed_s": 8.0,
                "method": "BM25 + CE rerank + answer extraction + LLM synthesis",
                "llm_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    report = report_from_path(source, benchmark="locomo", generated_at="fixed")

    assert report["source_format"] == "json"
    assert report["score"] == {"correct": 3, "total": 4, "accuracy": 75.0}
    by_category = _map(report["by_category"])
    runtime = _map(report["runtime"])
    assert by_category["temporal"] == {"correct": 1, "total": 2, "accuracy": 50.0}
    assert runtime["elapsed_s"] == 8.0
    assert runtime["mean_per_record_ms"] == 2000.0
    assert report["method"] == "BM25 + CE rerank + answer extraction + LLM synthesis"
    assert report["llm_enabled"] is True


def test_invalid_json_payloads_fail_closed(tmp_path: Path) -> None:
    """Malformed JSON payloads fail closed with actionable messages."""
    cases: list[tuple[object, str]] = [
        (["not-an-object"], "JSON item 0 is not an object"),
        ({"total_correct": 1, "total_tested": "bad", "by_category": {}}, "integer"),
        ({"total_correct": 0, "total_tested": 0, "by_category": []}, "object by_category"),
        (
            {"total_correct": 0, "total_tested": 0, "by_category": {"x": []}},
            "Category x is not an object",
        ),
        (
            {"total_correct": 0, "total_tested": 0, "by_category": {"x": {"correct": 0}}},
            "Category x lacks integer correct/total",
        ),
    ]
    for index, (payload, match) in enumerate(cases):
        source = tmp_path / f"bad_{index}.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            report_from_path(source, benchmark="bad", generated_at="fixed")


def test_jsonl_non_object_row_fails_closed(tmp_path: Path) -> None:
    """JSONL rows must be objects because benchmark records are keyed fields."""
    source = tmp_path / "bad.jsonl"
    source.write_text('["not-object"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL row 1 is not an object"):
        load_benchmark_payload(source)


def test_locomo_zero_total_has_no_mean_latency(tmp_path: Path) -> None:
    """A zero-total summary avoids a fabricated per-record latency."""
    source = tmp_path / "empty_locomo.json"
    source.write_text(
        json.dumps({"total_correct": 0, "total_tested": 0, "by_category": {}}),
        encoding="utf-8",
    )
    report = report_from_path(source, benchmark="locomo", generated_at="fixed")
    runtime = _map(report["runtime"])
    assert runtime["elapsed_s"] is None
    assert runtime["mean_per_record_ms"] is None


def test_load_benchmark_payload_rejects_unknown_json_shape(tmp_path: Path) -> None:
    """Unknown JSON payloads fail closed instead of fabricating a report."""
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported benchmark JSON shape"):
        load_benchmark_payload(source)


def test_main_writes_report_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI writes the machine-readable report and prints its destination."""
    results = tmp_path / "results.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(
        results,
        [{"question_id": "q", "question_type": "x", "judge_label": True}],
    )

    rc = main(
        [
            str(results),
            "--benchmark",
            "longmemeval",
            "--generated-at",
            "2026-06-27T13:00:00+00:00",
            "--output",
            str(report_path),
        ]
    )

    assert rc == 0
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["score"] == {"correct": 1, "total": 1, "accuracy": 100.0}
    assert str(report_path) in capsys.readouterr().out
