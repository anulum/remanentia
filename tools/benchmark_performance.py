# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Operator performance benchmark

"""Measure live Remanentia performance and write internal benchmark reports."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / ".coordination" / "benchmarks" / "REMANENTIA"
DEFAULT_QUERY = "What is Remanentia and how does its vector memory work?"


def percentile(values: list[float], pct: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[max(0, min(idx, len(ordered) - 1))]


def measurement(
    name: str, values_ms: list[float], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Summarise elapsed-millisecond samples."""
    return {
        "name": name,
        "n": len(values_ms),
        "min_ms": round(min(values_ms), 3) if values_ms else 0.0,
        "p50_ms": round(percentile(values_ms, 0.50), 3),
        "avg_ms": round(statistics.fmean(values_ms), 3) if values_ms else 0.0,
        "p95_ms": round(percentile(values_ms, 0.95), 3),
        "max_ms": round(max(values_ms), 3) if values_ms else 0.0,
        **(extra or {}),
    }


def time_call(
    name: str,
    func: Callable[[], Any],
    *,
    iterations: int,
    warmups: int = 1,
) -> dict[str, Any]:
    """Run a callable repeatedly and return a latency summary."""
    last: Any = None
    for _ in range(max(0, warmups)):
        last = func()

    values: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        last = func()
        values.append((time.perf_counter() - start) * 1000)

    extra = {"last_result_summary": summarise_result(last)}
    return measurement(name, values, extra)


def summarise_result(value: Any) -> dict[str, Any]:
    """Keep report summaries useful without storing full response payloads."""
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    summary: dict[str, Any] = {}
    for key in ("status", "daemon_kind", "daemon", "vector_worker", "version", "elapsed_ms"):
        if key in value:
            summary[key] = value[key]
    if "results" in value and isinstance(value["results"], list):
        summary["result_count"] = len(value["results"])
    if "semantic_memories" in value and isinstance(value["semantic_memories"], list):
        summary["semantic_memory_count"] = len(value["semantic_memories"])
    if "count" in value:
        summary["count"] = value["count"]
    return summary


def http_json(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    """Call a local JSON endpoint."""
    url = base_url.rstrip("/") + path
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = request.Request(url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=30) as response:  # nosec B310 - operator-supplied local URL.
        return json.loads(response.read().decode("utf-8"))


def run_json_command(
    args: list[str],
    *,
    env_extra: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run a repository command that emits JSON."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def vector_storage_summary(status: dict[str, Any]) -> dict[str, Any]:
    """Extract stable vector-index storage fields from CLI status JSON."""
    vectors = Path(str(status.get("vectors_path", "")))
    metadata = Path(str(status.get("metadata_path", "")))
    vector_bytes = vectors.stat().st_size if vectors.exists() else 0
    metadata_bytes = metadata.stat().st_size if metadata.exists() else 0
    manifest = status.get("manifest", {})
    return {
        "count": int(status.get("count", 0)),
        "dimension": int(manifest.get("dimension", 0) or 0),
        "elapsed_s": round(float(manifest.get("elapsed_s", 0.0) or 0.0), 3),
        "vector_bytes": vector_bytes,
        "metadata_bytes": metadata_bytes,
        "total_bytes": vector_bytes + metadata_bytes,
        "corpus_fingerprint": manifest.get("corpus_fingerprint", ""),
        "index_dir": status.get("index_dir", ""),
    }


def api_benchmarks(base_url: str, query: str, iterations: int) -> list[dict[str, Any]]:
    """Measure live HTTP endpoint latency."""
    endpoints: list[tuple[str, Callable[[], Any]]] = [
        ("api_health", lambda: http_json(base_url, "GET", "/health")),
        ("api_status", lambda: http_json(base_url, "GET", "/status")),
        (
            "api_recall",
            lambda: http_json(
                base_url,
                "POST",
                "/recall",
                {"query": query, "top_k": 3, "format": "summary", "include_content": False},
            ),
        ),
        (
            "api_public_vector_search",
            lambda: http_json(
                base_url,
                "POST",
                "/vector/search/public",
                {"query": query, "top_k": 5, "source": ""},
            ),
        ),
    ]
    return [time_call(name, call, iterations=iterations) for name, call in endpoints]


def vector_benchmarks(
    iterations: int, embedding_model: str, embedding_base_url: str
) -> dict[str, Any]:
    """Measure persistent vector-index status and unchanged-corpus refresh latency."""
    status = run_json_command([sys.executable, "-m", "vector_pipeline", "status"])
    env = {
        "REMANENTIA_EMBEDDING_MODEL": embedding_model,
        "REMANENTIA_EMBEDDING_BASE_URL": embedding_base_url,
    }

    def refresh() -> dict[str, Any]:
        return run_json_command([sys.executable, "-m", "vector_pipeline", "refresh"], env_extra=env)

    refresh_summary = time_call("vector_refresh_skip", refresh, iterations=iterations, warmups=0)
    return {
        "index": vector_storage_summary(status),
        "refresh_skip": refresh_summary,
    }


def direct_recall_benchmark(query: str, iterations: int) -> dict[str, Any]:
    """Measure the in-process recall path, excluding HTTP overhead."""
    sys.path.insert(0, str(REPO_ROOT))
    from memory_recall import recall

    def call() -> Any:
        ctx = recall(query, top_k=3, include_content=False)
        return {
            "elapsed_ms": ctx.elapsed_ms,
            "semantic_memories": ctx.semantic_memories,
        }

    return time_call("direct_recall", call, iterations=iterations)


def pytest_performance_benchmark() -> dict[str, Any]:
    """Run the focused performance regression suite and capture wall time."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_pipeline_performance.py",
        "tests/test_remanentia_retrieve.py::TestPerformance",
        "tests/test_vector_index.py",
        "-q",
        "--durations=20",
    ]
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return {
        "name": "pytest_performance_suite",
        "exit_code": result.returncode,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-30:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-30:]),
    }


def hardware_snapshot() -> dict[str, Any]:
    """Capture enough host context to make benchmark numbers interpretable."""
    cpu_model = ""
    cpu_count = os.cpu_count()
    mem_total_kb = 0
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
                break
    except OSError:
        pass

    gpu = ""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpu = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        gpu = "unavailable"

    return {
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "mem_total_gb": round(mem_total_kb / 1024 / 1024, 2) if mem_total_kb else 0,
        "gpu_use": gpu,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact internal Markdown benchmark report."""
    lines = [
        "# Remanentia Performance Benchmark",
        "",
        f"- Timestamp: `{report['timestamp_utc']}`",
        f"- Base URL: `{report['base_url']}`",
        f"- Query: `{report['query']}`",
        "",
        "## Hardware Context",
        "",
        f"- CPU: {report['hardware'].get('cpu_model') or 'unknown'}",
        f"- CPU count: {report['hardware'].get('cpu_count')}",
        f"- Memory: {report['hardware'].get('mem_total_gb')} GB",
        "",
        "## Live API",
        "",
        "| Endpoint | n | p50 ms | avg ms | p95 ms | max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["api"]:
        lines.append(
            f"| {item['name']} | {item['n']} | {item['p50_ms']} | {item['avg_ms']} | "
            f"{item['p95_ms']} | {item['max_ms']} |"
        )

    vector = report["vector"]["index"]
    refresh = report["vector"]["refresh_skip"]
    lines.extend(
        [
            "",
            "## Vector Index",
            "",
            f"- Chunks: {vector['count']:,}",
            f"- Dimension: {vector['dimension']}",
            f"- Index storage: {vector['total_bytes']:,} bytes",
            f"- Last build time: {vector['elapsed_s']} s",
            f"- Refresh skip p50: {refresh['p50_ms']} ms",
            f"- Refresh skip p95: {refresh['p95_ms']} ms",
        ]
    )
    if "direct_recall" in report:
        direct = report["direct_recall"]
        lines.extend(
            [
                "",
                "## Direct Recall",
                "",
                f"- p50: {direct['p50_ms']} ms",
                f"- p95: {direct['p95_ms']} ms",
            ]
        )
    if "pytest_performance" in report:
        pytest_result = report["pytest_performance"]
        lines.extend(
            [
                "",
                "## Performance Regression Suite",
                "",
                f"- Exit code: {pytest_result['exit_code']}",
                f"- Wall time: {pytest_result['elapsed_ms']} ms",
                "",
                "```text",
                pytest_result["stdout_tail"],
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Write timestamped JSON and Markdown reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"remanentia_performance_{stamp}.json"
    md_path = out_dir / f"remanentia_performance_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--embedding-model", default="nomic-embed")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--api-iterations", type=int, default=20)
    parser.add_argument("--refresh-iterations", type=int, default=5)
    parser.add_argument("--direct-recall-iterations", type=int, default=10)
    parser.add_argument("--skip-direct-recall", action="store_true")
    parser.add_argument("--include-pytest", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "query": args.query,
        "hardware": hardware_snapshot(),
        "api": api_benchmarks(args.base_url, args.query, args.api_iterations),
        "vector": vector_benchmarks(
            args.refresh_iterations, args.embedding_model, args.embedding_base_url
        ),
    }
    if not args.skip_direct_recall:
        report["direct_recall"] = direct_recall_benchmark(args.query, args.direct_recall_iterations)
    if args.include_pytest:
        report["pytest_performance"] = pytest_performance_benchmark()

    paths = write_reports(report, args.out_dir)
    print(json.dumps({"report_paths": paths, "summary": summarise_public_report(report)}, indent=2))
    return 0


def summarise_public_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a small safe summary for release notes or landing-page copy."""
    api_by_name = {item["name"]: item for item in report["api"]}
    vector = report["vector"]["index"]
    refresh = report["vector"]["refresh_skip"]
    summary: dict[str, Any] = {
        "chunks": vector["count"],
        "dimension": vector["dimension"],
        "index_storage_bytes": vector["total_bytes"],
        "refresh_skip_p50_ms": refresh["p50_ms"],
        "api_health_p95_ms": api_by_name.get("api_health", {}).get("p95_ms"),
        "api_status_p95_ms": api_by_name.get("api_status", {}).get("p95_ms"),
        "api_recall_p50_ms": api_by_name.get("api_recall", {}).get("p50_ms"),
        "api_recall_p95_ms": api_by_name.get("api_recall", {}).get("p95_ms"),
        "api_public_vector_search_p50_ms": api_by_name.get("api_public_vector_search", {}).get(
            "p50_ms"
        ),
        "api_public_vector_search_p95_ms": api_by_name.get("api_public_vector_search", {}).get(
            "p95_ms"
        ),
    }
    if "direct_recall" in report:
        summary["direct_recall_p50_ms"] = report["direct_recall"]["p50_ms"]
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
