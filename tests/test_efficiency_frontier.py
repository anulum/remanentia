# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — efficiency frontier report tests

"""Tests for benchmark efficiency-frontier reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, cast

import pytest

from efficiency_frontier import build_efficiency_report, main


def _map(value: object) -> Mapping[str, object]:
    """Assert and cast nested JSON objects for strict-mypy assertions."""
    assert isinstance(value, dict)
    return cast(Mapping[str, object], value)


def _list(value: object) -> list[object]:
    """Assert and cast nested JSON arrays for strict-mypy assertions."""
    assert isinstance(value, list)
    return value


def _write_json(path: Path, payload: object) -> None:
    """Write JSON payloads through the real file path used by the reporter."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _benchmark_report(
    path: Path,
    *,
    benchmark: str,
    accuracy: float,
    prompt_tokens: int,
    completion_tokens: int,
    p95_ms: float | None,
) -> None:
    """Create a benchmark-report-compatible JSON artefact for tests."""
    _write_json(
        path,
        {
            "schema_version": 1,
            "benchmark": benchmark,
            "source_path": f"data/{benchmark}.jsonl",
            "source_format": "jsonl",
            "generated_at": "2026-06-27T12:00:00+00:00",
            "n_records": 2,
            "score": {"correct": 1, "total": 2, "accuracy": accuracy},
            "runtime": {"latency_ms": {"mean": 15.0, "p50": 15.0, "p95": p95_ms}},
            "tokens": {
                "judge_prompt_estimate": prompt_tokens,
                "judge_completion": completion_tokens,
            },
            "judge": {"models": ["judge"], "prompt_sha256": ["a" * 64]},
        },
    )


def test_build_efficiency_report_merges_reports_and_baselines(tmp_path: Path) -> None:
    """The report computes comparable accuracy/token and p95-latency metrics."""
    report_path = tmp_path / "longmemeval_report.json"
    baselines_path = tmp_path / "baselines.json"
    _benchmark_report(
        report_path,
        benchmark="longmemeval",
        accuracy=72.2,
        prompt_tokens=1_000,
        completion_tokens=50,
        p95_ms=30.0,
    )
    _write_json(
        baselines_path,
        {
            "schema_version": 1,
            "baselines": [
                {
                    "benchmark": "longmemeval",
                    "label": "python-only reference",
                    "source": "committed historical baseline",
                    "accuracy": 56.6,
                    "tokens_total": 2_000,
                    "p95_latency_ms": 45.0,
                }
            ],
        },
    )

    frontier = build_efficiency_report(
        report_paths=[report_path],
        baseline_path=baselines_path,
        generated_at="2026-06-27T13:00:00+00:00",
    )

    assert frontier["schema_version"] == 1
    assert frontier["generated_at"] == "2026-06-27T13:00:00+00:00"
    candidates = _list(frontier["candidates"])
    assert len(candidates) == 2
    remanentia = _map(candidates[0])
    baseline = _map(candidates[1])
    assert remanentia["label"] == "Remanentia"
    assert remanentia["benchmark"] == "longmemeval"
    assert remanentia["tokens_total"] == 1050
    assert remanentia["accuracy_points_per_1k_tokens"] == 68.762
    assert remanentia["p95_latency_ms"] == 30.0
    assert remanentia["frontier"] is True
    assert baseline["label"] == "python-only reference"
    assert baseline["accuracy_points_per_1k_tokens"] == 28.3
    assert baseline["frontier"] is False


def test_build_efficiency_report_preserves_unknown_measurements(tmp_path: Path) -> None:
    """Unknown tokens or p95 latency remain null rather than being fabricated."""
    report_path = tmp_path / "locomo_report.json"
    _benchmark_report(
        report_path,
        benchmark="locomo",
        accuracy=83.1,
        prompt_tokens=0,
        completion_tokens=0,
        p95_ms=None,
    )

    frontier = build_efficiency_report(report_paths=[report_path], generated_at=None)

    candidate = _map(_list(frontier["candidates"])[0])
    assert candidate["tokens_total"] is None
    assert candidate["accuracy_points_per_1k_tokens"] is None
    assert candidate["p95_latency_ms"] is None
    assert candidate["frontier"] is False


def test_build_efficiency_report_rejects_malformed_baseline(tmp_path: Path) -> None:
    """Baseline files must be explicit object lists, not arbitrary JSON."""
    report_path = tmp_path / "report.json"
    baselines_path = tmp_path / "baselines.json"
    _benchmark_report(
        report_path,
        benchmark="longmemeval",
        accuracy=70.0,
        prompt_tokens=1,
        completion_tokens=0,
        p95_ms=10.0,
    )
    _write_json(baselines_path, {"schema_version": 1, "baselines": ["bad"]})

    with pytest.raises(ValueError, match="baseline item 0"):
        build_efficiency_report(report_paths=[report_path], baseline_path=baselines_path)


def test_build_efficiency_report_rejects_missing_report_list() -> None:
    """At least one committed benchmark report is required."""
    with pytest.raises(ValueError, match="At least one"):
        build_efficiency_report(report_paths=[])


def test_build_efficiency_report_rejects_non_object_json(tmp_path: Path) -> None:
    """Report paths must point to JSON objects."""
    report_path = tmp_path / "report.json"
    _write_json(report_path, [])

    with pytest.raises(ValueError, match="JSON file must contain an object"):
        build_efficiency_report(report_paths=[report_path])


def test_build_efficiency_report_rejects_missing_score_object(tmp_path: Path) -> None:
    """Benchmark reports must expose a score object."""
    report_path = tmp_path / "report.json"
    _write_json(report_path, {"benchmark": "longmemeval", "score": []})

    with pytest.raises(ValueError, match="object field score"):
        build_efficiency_report(report_paths=[report_path])


def test_build_efficiency_report_rejects_missing_string_fields(tmp_path: Path) -> None:
    """Benchmark reports must identify their benchmark."""
    report_path = tmp_path / "report.json"
    _write_json(report_path, {"benchmark": "", "score": {"accuracy": 1.0}})

    with pytest.raises(ValueError, match="non-empty string field benchmark"):
        build_efficiency_report(report_paths=[report_path])


def test_build_efficiency_report_rejects_missing_accuracy(tmp_path: Path) -> None:
    """Benchmark reports must include numeric accuracy."""
    report_path = tmp_path / "report.json"
    _write_json(report_path, {"benchmark": "longmemeval", "score": {"accuracy": True}})

    with pytest.raises(ValueError, match="numeric field accuracy"):
        build_efficiency_report(report_paths=[report_path])


def test_build_efficiency_report_accepts_partial_and_integral_float_fields(
    tmp_path: Path,
) -> None:
    """Optional bools are ignored while integral floats remain valid counters."""
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baselines.json"
    _write_json(
        report_path,
        {
            "benchmark": "longmemeval",
            "source_path": 123,
            "score": {"accuracy": 71.0, "correct": 1.0, "total": True},
            "tokens": [],
            "runtime": [],
        },
    )
    _write_json(
        baseline_path,
        {
            "baselines": [
                {
                    "benchmark": "longmemeval",
                    "label": "complete baseline",
                    "source": "measured baseline",
                    "accuracy": 70.0,
                    "tokens_total": 1.0,
                    "p95_latency_ms": True,
                }
            ]
        },
    )

    frontier = build_efficiency_report(report_paths=[report_path], baseline_path=baseline_path)

    remanentia, baseline = [_map(candidate) for candidate in _list(frontier["candidates"])]
    assert remanentia["source"] == str(report_path)
    assert remanentia["correct"] == 1
    assert remanentia["total"] is None
    assert remanentia["tokens_total"] is None
    assert baseline["tokens_total"] == 1
    assert baseline["p95_latency_ms"] is None
    assert baseline["frontier"] is False


def test_incomplete_baseline_does_not_dominate_complete_candidate(tmp_path: Path) -> None:
    """A baseline lacking p95/token evidence cannot dominate measured reports."""
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baselines.json"
    _benchmark_report(
        report_path,
        benchmark="longmemeval",
        accuracy=70.0,
        prompt_tokens=1_000,
        completion_tokens=0,
        p95_ms=10.0,
    )
    _write_json(
        baseline_path,
        {
            "baselines": [
                {
                    "benchmark": "longmemeval",
                    "label": "incomplete baseline",
                    "source": "historical accuracy-only row",
                    "accuracy": 99.0,
                    "tokens_total": None,
                    "p95_latency_ms": None,
                }
            ]
        },
    )

    frontier = build_efficiency_report(report_paths=[report_path], baseline_path=baseline_path)

    remanentia, baseline = [_map(candidate) for candidate in _list(frontier["candidates"])]
    assert remanentia["frontier"] is True
    assert baseline["frontier"] is False


def test_build_efficiency_report_rejects_baseline_without_array(tmp_path: Path) -> None:
    """Baseline JSON must use the documented baselines array."""
    report_path = tmp_path / "report.json"
    baselines_path = tmp_path / "baselines.json"
    _benchmark_report(
        report_path,
        benchmark="longmemeval",
        accuracy=70.0,
        prompt_tokens=1,
        completion_tokens=0,
        p95_ms=10.0,
    )
    _write_json(baselines_path, {"baselines": {}})

    with pytest.raises(ValueError, match="baselines array"):
        build_efficiency_report(report_paths=[report_path], baseline_path=baselines_path)


def test_main_writes_efficiency_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The real CLI writes a deterministic report JSON file."""
    report_path = tmp_path / "longmemeval_report.json"
    output = tmp_path / "efficiency_frontier.json"
    _benchmark_report(
        report_path,
        benchmark="longmemeval",
        accuracy=72.2,
        prompt_tokens=1_000,
        completion_tokens=50,
        p95_ms=30.0,
    )

    assert (
        main(
            [
                "--report",
                str(report_path),
                "--output",
                str(output),
                "--generated-at",
                "2026-06-27T13:00:00+00:00",
            ]
        )
        == 0
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["generated_at"] == "2026-06-27T13:00:00+00:00"
    assert "wrote efficiency frontier" in capsys.readouterr().out
