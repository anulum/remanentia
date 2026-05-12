# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Performance benchmark tool tests

"""Unit tests for the operator performance benchmark helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "tools" / "benchmark_performance.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_performance", BENCHMARK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentile_empty_returns_zero():
    benchmark = _load_benchmark_module()
    assert benchmark.percentile([], 0.95) == 0.0


def test_measurement_summarises_latency_values():
    benchmark = _load_benchmark_module()
    result = benchmark.measurement("sample", [1.0, 3.0, 2.0])
    assert result["name"] == "sample"
    assert result["n"] == 3
    assert result["min_ms"] == 1.0
    assert result["p50_ms"] == 2.0
    assert result["avg_ms"] == 2.0
    assert result["p95_ms"] == 3.0
    assert result["max_ms"] == 3.0


def test_vector_storage_summary_counts_existing_files(tmp_path: Path):
    benchmark = _load_benchmark_module()
    vectors = tmp_path / "vectors.npy"
    metadata = tmp_path / "metadata.jsonl"
    vectors.write_bytes(b"v" * 12)
    metadata.write_bytes(b"m" * 5)
    result = benchmark.vector_storage_summary(
        {
            "index_dir": str(tmp_path),
            "count": 4,
            "vectors_path": str(vectors),
            "metadata_path": str(metadata),
            "manifest": {
                "dimension": 768,
                "elapsed_s": 1.23456,
                "corpus_fingerprint": "abc",
            },
        }
    )
    assert result == {
        "count": 4,
        "dimension": 768,
        "elapsed_s": 1.235,
        "vector_bytes": 12,
        "metadata_bytes": 5,
        "total_bytes": 17,
        "corpus_fingerprint": "abc",
        "index_dir": str(tmp_path),
    }


def test_summarise_public_report_selects_landing_page_fields():
    benchmark = _load_benchmark_module()
    report = {
        "api": [
            {"name": "api_health", "p95_ms": 10.0},
            {"name": "api_status", "p95_ms": 20.0},
            {"name": "api_recall", "p50_ms": 100.0, "p95_ms": 150.0},
            {"name": "api_public_vector_search", "p50_ms": 30.0, "p95_ms": 40.0},
        ],
        "vector": {
            "index": {"count": 1103, "dimension": 768, "total_bytes": 4326400},
            "refresh_skip": {"p50_ms": 800.0},
        },
        "direct_recall": {"p50_ms": 90.0},
    }
    assert benchmark.summarise_public_report(report) == {
        "chunks": 1103,
        "dimension": 768,
        "index_storage_bytes": 4326400,
        "refresh_skip_p50_ms": 800.0,
        "api_health_p95_ms": 10.0,
        "api_status_p95_ms": 20.0,
        "api_recall_p50_ms": 100.0,
        "api_recall_p95_ms": 150.0,
        "api_public_vector_search_p50_ms": 30.0,
        "api_public_vector_search_p95_ms": 40.0,
        "direct_recall_p50_ms": 90.0,
    }


def test_build_report_includes_reproducibility_manifest(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark, "hardware_snapshot", lambda: {"cpu_model": "cpu"})
    monkeypatch.setattr(benchmark, "api_benchmarks", lambda base_url, query, iterations: [])
    monkeypatch.setattr(
        benchmark,
        "vector_benchmarks",
        lambda iterations, embedding_model, embedding_base_url: {
            "index": {"count": 0, "dimension": 0, "total_bytes": 0},
            "refresh_skip": {"p50_ms": 0.0, "p95_ms": 0.0},
        },
    )
    monkeypatch.setattr(
        benchmark,
        "direct_recall_benchmark",
        lambda query, iterations: {"name": "direct_recall", "p50_ms": 0.0, "p95_ms": 0.0},
    )

    args = benchmark.build_parser().parse_args(
        [
            "--base-url",
            "http://127.0.0.1:8001",
            "--query",
            "q",
            "--seed",
            "123",
            "--api-iterations",
            "2",
            "--refresh-iterations",
            "1",
            "--direct-recall-iterations",
            "3",
        ]
    )
    report = benchmark.build_report(args)

    manifest = report["reproducibility"]
    assert manifest["seed"] == 123
    assert manifest["workload"] == "performance_benchmark"
    assert manifest["parameters"]["api_iterations"] == 2
    assert manifest["parameters"]["refresh_iterations"] == 1
    assert manifest["parameters"]["direct_recall_iterations"] == 3
