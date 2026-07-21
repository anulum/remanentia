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
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def test_http_json_posts_payload():
    benchmark = _load_benchmark_module()
    requests: list[tuple[str, dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            requests.append((self.path, json.loads(self.rfile.read(length))))
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}/"
        result = benchmark.http_json(base_url, "POST", "/recall", {"query": "q"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result == {"status": "ok"}
    assert requests == [("/recall", {"query": "q"})]


def test_run_json_command_parses_stdout_and_reports_errors():
    benchmark = _load_benchmark_module()
    success = [
        sys.executable,
        "-c",
        "import json,os; print(json.dumps({'extra': os.environ['EXTRA']}))",
    ]
    assert benchmark.run_json_command(success, env_extra={"EXTRA": "1"}) == {"extra": "1"}

    failure = [
        sys.executable,
        "-c",
        "import sys; print('boom', file=sys.stderr); raise SystemExit(2)",
    ]
    with pytest.raises(RuntimeError, match="boom"):
        benchmark.run_json_command(failure)


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


def test_api_benchmarks_measures_expected_endpoints():
    benchmark = _load_benchmark_module()
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, payload: dict[str, object] | None = None):
            requests.append((self.command, self.path, payload))
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._respond()

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            self._respond(json.loads(self.rfile.read(length)))

        def log_message(self, _format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        result = benchmark.api_benchmarks(base_url, "query", iterations=1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert [item["name"] for item in result] == [
        "api_health",
        "api_status",
        "api_recall",
        "api_public_vector_search",
    ]
    paths = [path for _, path, _ in requests]
    assert paths.count("/health") == 2
    assert paths.count("/status") == 2
    assert paths.count("/recall") == 2
    assert paths.count("/vector/search/public") == 2


def test_direct_recall_benchmark_uses_memory_recall(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, int, bool]] = []

    def fake_recall(query: str, *, top_k: int, include_content: bool):
        calls.append((query, top_k, include_content))
        return SimpleNamespace(elapsed_ms=1.25, semantic_memories=[{"path": "fixture.md"}])

    import memory_recall

    monkeypatch.setattr(memory_recall, "recall", fake_recall)
    benchmark = _load_benchmark_module()

    result = benchmark.direct_recall_benchmark("SNN removal decision", 1)

    assert calls == [
        ("SNN removal decision", 3, False),
        ("SNN removal decision", 3, False),
    ]
    assert result["name"] == "direct_recall"
    assert result["n"] == 1
    assert result["last_result_summary"] == {
        "elapsed_ms": 1.25,
        "semantic_memory_count": 1,
    }


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
