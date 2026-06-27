# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for retrieval benchmark suites

from __future__ import annotations

import gzip
import json
from pathlib import Path

import benchmark_suites
from benchmark_suites import current_operational_queries, historical_regression_queries


def _write_index(path: Path, *, timestamp: object = 1_774_654_400.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": ["a.md", "b.md"],
        "paragraph_index": [(0, 0), (1, 0), (1, 1)],
        "timestamp": timestamp,
    }
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(payload).encode("utf-8"))


def test_current_operational_queries_filter_missing_live_metrics(tmp_path: Path) -> None:
    queries = current_operational_queries(tmp_path)

    assert queries
    assert all(query["suite"] == "current_operational" for query in queries)
    assert "when was the current unified index built" not in {query["q"] for query in queries}
    assert "what was the recent full recall API p50 latency" not in {
        query["q"] for query in queries
    }


def test_current_operational_queries_read_index_and_latest_performance(tmp_path: Path) -> None:
    _write_index(tmp_path / "snn_state" / "memory_index.json.gz")
    report_dir = tmp_path / ".coordination" / "benchmarks" / "REMANENTIA"
    report_dir.mkdir(parents=True)
    (report_dir / "remanentia_performance_2026-01-01.json").write_text(
        json.dumps({"api": [{"name": "api_recall", "p50_ms": 999.0}]}),
        encoding="utf-8",
    )
    (report_dir / "remanentia_performance_2026-02-01.json").write_text(
        json.dumps(
            {
                "api": [
                    {"name": "api_recall", "p50_ms": 841.245},
                    {"name": "api_public_vector_search", "p50_ms": 95.347},
                ],
                "vector": {"index": {"count": 1103}},
            }
        ),
        encoding="utf-8",
    )

    queries = current_operational_queries(tmp_path)
    gold = " ".join(item for query in queries for item in query["gold"])

    assert "2" in gold
    assert "3" in gold
    assert "2026-03-27" in gold
    assert "841.245" in gold
    assert "95.347" in gold
    assert "1,103" in gold


def test_index_stats_returns_empty_values_for_bad_or_missing_index(tmp_path: Path) -> None:
    missing = benchmark_suites._index_stats(tmp_path / "missing.json.gz")
    assert missing == {"documents": 0, "paragraphs": 0, "date": ""}

    broken = tmp_path / "snn_state" / "memory_index.json.gz"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not gzip")
    assert benchmark_suites._index_stats(broken) == {
        "documents": 0,
        "paragraphs": 0,
        "date": "",
    }


def test_index_stats_ignores_non_numeric_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "snn_state" / "memory_index.json.gz"
    _write_index(path, timestamp="not numeric")

    stats = benchmark_suites._index_stats(path)

    assert stats["documents"] == 2
    assert stats["paragraphs"] == 3
    assert stats["date"] == ""


def test_latest_json_returns_empty_for_missing_or_invalid_report(tmp_path: Path) -> None:
    assert benchmark_suites._latest_json(tmp_path / "missing", "*.json") == {}

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "report.json").write_text("{", encoding="utf-8")

    assert benchmark_suites._latest_json(report_dir, "*.json") == {}


def test_latest_json_ignores_non_object_payload(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "report.json").write_text("[]", encoding="utf-8")

    assert benchmark_suites._latest_json(report_dir, "*.json") == {}


def test_named_row_and_format_helpers_handle_non_matching_inputs() -> None:
    assert benchmark_suites._named_row(None, "api_recall") == {}
    assert benchmark_suites._named_row([{"name": "other"}], "api_recall") == {}
    row = benchmark_suites._named_row(["bad", {"name": "api_recall", "p50_ms": 1}], "api_recall")
    assert row["p50_ms"] == 1
    assert benchmark_suites._fmt("plain") == "plain"
    assert benchmark_suites._fmt(1103) == "1,103"


def test_historical_regression_queries_are_stable() -> None:
    queries = historical_regression_queries()

    assert len(queries) == 4
    assert all(query["suite"] == "historical_regression" for query in queries)
    assert {"metric", "temporal", "debugging"} <= {query["cat"] for query in queries}
