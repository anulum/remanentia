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
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


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


def test_time_call_runs_warmup_and_iterations():
    benchmark = _load_benchmark_module()
    calls: list[int] = []

    def call():
        calls.append(len(calls))
        return {"status": "ok", "results": [1, 2]}

    result = benchmark.time_call("call", call, iterations=2, warmups=1)

    assert len(calls) == 3
    assert result["name"] == "call"
    assert result["n"] == 2
    assert result["last_result_summary"] == {"status": "ok", "result_count": 2}


def test_summarise_result_handles_non_dict_and_semantic_memories():
    benchmark = _load_benchmark_module()

    assert benchmark.summarise_result(["x"]) == {"type": "list"}
    assert benchmark.summarise_result({"semantic_memories": [{"path": "a"}], "count": 3}) == {
        "semantic_memory_count": 1,
        "count": 3,
    }


def test_http_json_posts_payload(monkeypatch):
    benchmark = _load_benchmark_module()
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"status":"ok"}'

    def urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(benchmark.request, "urlopen", urlopen)

    assert benchmark.http_json("http://127.0.0.1:8001/", "POST", "/recall", {"query": "q"}) == {
        "status": "ok"
    }
    assert seen == {
        "url": "http://127.0.0.1:8001/recall",
        "method": "POST",
        "body": {"query": "q"},
        "timeout": 30,
    }


def test_run_json_command_parses_stdout_and_reports_errors(monkeypatch):
    benchmark = _load_benchmark_module()
    seen = {}

    def run_success(*args, **kwargs):
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", run_success)
    assert benchmark.run_json_command(["cmd"], env_extra={"EXTRA": "1"}) == {"ok": True}
    assert seen["EXTRA"] == "1"

    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        benchmark.run_json_command(["cmd"])


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


def test_api_benchmarks_measures_expected_endpoints(monkeypatch):
    benchmark = _load_benchmark_module()
    calls: list[tuple[str, str, str, dict | None]] = []

    def http_json(base_url, method, path, payload=None):
        calls.append((base_url, method, path, payload))
        return {"status": "ok"}

    monkeypatch.setattr(benchmark, "http_json", http_json)
    monkeypatch.setattr(
        benchmark,
        "time_call",
        lambda name, func, iterations, warmups=1: {"name": name, "result": func()},
    )

    result = benchmark.api_benchmarks("http://api", "query", iterations=1)

    assert [item["name"] for item in result] == [
        "api_health",
        "api_status",
        "api_recall",
        "api_public_vector_search",
    ]
    assert [call[2] for call in calls] == ["/health", "/status", "/recall", "/vector/search/public"]


def test_vector_benchmarks_refreshes_with_embedding_environment(monkeypatch):
    benchmark = _load_benchmark_module()
    env_seen = {}
    status = {"count": 0, "manifest": {}, "vectors_path": "", "metadata_path": ""}

    def run_json_command(args, env_extra=None, timeout=120):
        if args[-1] == "status":
            return status
        env_seen.update(env_extra or {})
        return {"status": "skipped"}

    monkeypatch.setattr(benchmark, "run_json_command", run_json_command)
    monkeypatch.setattr(
        benchmark,
        "time_call",
        lambda name, func, iterations, warmups=0: {"name": name, "last": func(), "p50_ms": 1.0},
    )

    result = benchmark.vector_benchmarks(1, "model", "http://embed")

    assert result["index"]["count"] == 0
    assert result["refresh_skip"]["last"] == {"status": "skipped"}
    assert env_seen == {
        "REMANENTIA_EMBEDDING_MODEL": "model",
        "REMANENTIA_EMBEDDING_BASE_URL": "http://embed",
    }


def test_direct_recall_benchmark_uses_memory_recall(monkeypatch):
    benchmark = _load_benchmark_module()

    class Context:
        elapsed_ms = 12.5
        semantic_memories = [{"path": "memory.md"}]

    fake_module = SimpleNamespace(recall=lambda query, top_k, include_content: Context())
    monkeypatch.setitem(sys.modules, "memory_recall", fake_module)
    monkeypatch.setattr(
        benchmark,
        "time_call",
        lambda name, func, iterations: {"name": name, "last": func()},
    )

    result = benchmark.direct_recall_benchmark("query", 1)

    assert result["name"] == "direct_recall"
    assert result["last"] == {"elapsed_ms": 12.5, "semantic_memories": [{"path": "memory.md"}]}


def test_pytest_performance_benchmark_captures_exit_and_tails(monkeypatch):
    benchmark = _load_benchmark_module()
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="\n".join(f"out{i}" for i in range(35)),
            stderr="\n".join(f"err{i}" for i in range(35)),
        ),
    )

    result = benchmark.pytest_performance_benchmark()

    assert result["exit_code"] == 1
    assert result["elapsed_ms"] == 250.0
    assert "out5" in result["stdout_tail"]
    assert "err5" in result["stderr_tail"]


def test_hardware_snapshot_reads_host_context_and_gpu(tmp_path: Path):
    benchmark = _load_benchmark_module()
    cpuinfo = tmp_path / "cpuinfo"
    meminfo = tmp_path / "meminfo"
    gpu_probe = tmp_path / "gpu-probe"
    cpuinfo.write_text("model name\t: Test CPU\n", encoding="utf-8")
    meminfo.write_text("MemTotal:       1048576 kB\n", encoding="utf-8")
    gpu_probe.write_text("#!/bin/sh\necho 'GPU 0 10%'\n", encoding="utf-8")
    gpu_probe.chmod(0o700)

    snapshot = benchmark.hardware_snapshot(
        cpuinfo_path=cpuinfo,
        meminfo_path=meminfo,
        gpu_command=(str(gpu_probe),),
    )

    assert snapshot == {
        "cpu_model": "Test CPU",
        "cpu_count": benchmark.os.cpu_count(),
        "mem_total_gb": 1.0,
        "gpu_use": "GPU 0 10%",
    }


def test_hardware_snapshot_handles_unavailable_files_and_gpu(tmp_path: Path):
    benchmark = _load_benchmark_module()
    snapshot = benchmark.hardware_snapshot(
        cpuinfo_path=tmp_path / "missing-cpuinfo",
        meminfo_path=tmp_path / "missing-meminfo",
        gpu_command=(str(tmp_path / "missing-gpu-probe"),),
    )

    assert snapshot["cpu_model"] == ""
    assert snapshot["cpu_count"] == benchmark.os.cpu_count()
    assert snapshot["mem_total_gb"] == 0
    assert snapshot["gpu_use"] == "unavailable"


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


def _sample_report() -> dict:
    return {
        "timestamp_utc": "2026-06-25T20:00:00+00:00",
        "base_url": "http://127.0.0.1:8001",
        "query": "q",
        "reproducibility": {"seed": 42},
        "hardware": {"cpu_model": "cpu", "cpu_count": 8, "mem_total_gb": 32},
        "api": [
            {"name": "api_health", "n": 1, "p50_ms": 1, "avg_ms": 1, "p95_ms": 1, "max_ms": 1},
            {"name": "api_status", "n": 1, "p50_ms": 2, "avg_ms": 2, "p95_ms": 2, "max_ms": 2},
        ],
        "vector": {
            "index": {"count": 10, "dimension": 3, "total_bytes": 100, "elapsed_s": 0.5},
            "refresh_skip": {"p50_ms": 4, "p95_ms": 5},
        },
        "direct_recall": {"p50_ms": 6, "p95_ms": 7},
        "pytest_performance": {"exit_code": 0, "elapsed_ms": 8, "stdout_tail": "ok"},
    }


def test_render_markdown_includes_all_report_sections():
    benchmark = _load_benchmark_module()

    markdown = benchmark.render_markdown(_sample_report())

    assert "# Remanentia Performance Benchmark" in markdown
    assert "## Live API" in markdown
    assert "## Vector Index" in markdown
    assert "## Direct Recall" in markdown
    assert "## Performance Regression Suite" in markdown


def test_write_reports_writes_json_and_markdown(tmp_path: Path):
    benchmark = _load_benchmark_module()

    paths = benchmark.write_reports(_sample_report(), tmp_path)

    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["query"] == "q"
    assert "Remanentia Performance Benchmark" in Path(paths["markdown"]).read_text(encoding="utf-8")


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


def test_build_report_can_skip_direct_and_include_pytest(monkeypatch):
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
    monkeypatch.setattr(benchmark, "direct_recall_benchmark", pytest.fail)
    monkeypatch.setattr(
        benchmark,
        "pytest_performance_benchmark",
        lambda: {"exit_code": 0, "elapsed_ms": 1, "stdout_tail": ""},
    )

    args = benchmark.build_parser().parse_args(
        ["--query", "q", "--skip-direct-recall", "--include-pytest"]
    )
    report = benchmark.build_report(args)

    assert "direct_recall" not in report
    assert report["pytest_performance"]["exit_code"] == 0


def test_main_writes_report_and_prints_summary(monkeypatch, tmp_path: Path, capsys):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark, "build_report", lambda args: _sample_report())

    code = benchmark.main(["--out-dir", str(tmp_path)])

    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert Path(output["report_paths"]["json"]).exists()
    assert output["summary"]["chunks"] == 10
