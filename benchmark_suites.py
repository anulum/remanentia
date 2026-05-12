# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Retrieval benchmark suites

"""Current and historical retrieval benchmark query suites."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).parent
INDEX_PATH = BASE / "snn_state" / "memory_index.json.gz"
BENCHMARK_DIR = BASE / ".coordination" / "benchmarks" / "REMANENTIA"


def current_operational_queries(repo: Path = BASE) -> list[dict[str, Any]]:
    repo = repo.resolve()
    index_stats = _index_stats(repo / "snn_state" / "memory_index.json.gz")
    performance = _latest_json(
        repo / ".coordination" / "benchmarks" / "REMANENTIA", "remanentia_performance_*.json"
    )
    recall_api = _named_row(performance.get("api"), "api_recall")
    public_vector = _named_row(performance.get("api"), "api_public_vector_search")
    vector_index = (performance.get("vector") or {}).get("index") or {}
    queries = [
        {
            "q": "how many documents are in the current unified index",
            "gold": [_fmt(index_stats["documents"]), str(index_stats["documents"])],
            "cat": "metric",
            "suite": "current_operational",
        },
        {
            "q": "how many paragraphs does the current index have",
            "gold": [_fmt(index_stats["paragraphs"]), str(index_stats["paragraphs"])],
            "cat": "metric",
            "suite": "current_operational",
        },
        {
            "q": "when was the current unified index built",
            "gold": [index_stats["date"]],
            "cat": "temporal",
            "suite": "current_operational",
        },
        {
            "q": "what was the recent full recall API p50 latency",
            "gold": [str(recall_api.get("p50_ms", ""))],
            "cat": "continuity",
            "suite": "current_operational",
        },
        {
            "q": "what was the recent public vector search p50 latency",
            "gold": [str(public_vector.get("p50_ms", ""))],
            "cat": "continuity",
            "suite": "current_operational",
        },
        {
            "q": "what replaced the stale legacy daemon path in Remanentia",
            "gold": ["vector_worker", "scheduled vector", "vector worker"],
            "cat": "continuity",
            "suite": "current_operational",
        },
        {
            "q": "how many chunks were in the recent vector index benchmark",
            "gold": [_fmt(vector_index.get("count", 0)), str(vector_index.get("count", ""))],
            "cat": "continuity",
            "suite": "current_operational",
        },
        {
            "q": "what connects sc-neurocore and scpn-quantum-control",
            "gold": ["identity", "quantum", "classical"],
            "cat": "cross_project",
            "suite": "current_operational",
        },
        {
            "q": "how does the consolidation pipeline cluster traces",
            "gold": ["project", "date", "two days"],
            "cat": "factual",
            "suite": "current_operational",
        },
        {
            "q": "what GPU is available locally",
            "gold": ["GTX 1060", "6GB"],
            "cat": "factual",
            "suite": "current_operational",
        },
    ]
    return [q for q in queries if all(str(item) for item in q["gold"])]


def historical_regression_queries() -> list[dict[str, Any]]:
    return [
        {
            "q": "how many documents were expected by the historical March unified index benchmark",
            "gold": ["1,217", "1217"],
            "cat": "metric",
            "suite": "historical_regression",
        },
        {
            "q": "how many paragraphs were expected by the historical March unified index benchmark",
            "gold": ["15,938", "15938"],
            "cat": "metric",
            "suite": "historical_regression",
        },
        {
            "q": "when was the historical March unified index benchmark built",
            "gold": ["2026-03-20", "2026-03-22"],
            "cat": "temporal",
            "suite": "historical_regression",
        },
        {
            "q": "what happened with the historical identity coherence R metric incident",
            "gold": ["never called", "garbage", "theatre"],
            "cat": "debugging",
            "suite": "historical_regression",
        },
    ]


def _index_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"documents": 0, "paragraphs": 0, "date": ""}
    try:
        with gzip.open(path, "rb") as handle:
            data = json.loads(handle.read())
    except (OSError, json.JSONDecodeError):
        return {"documents": 0, "paragraphs": 0, "date": ""}
    timestamp = data.get("timestamp")
    date = ""
    if isinstance(timestamp, int | float):
        from datetime import UTC, datetime

        date = datetime.fromtimestamp(float(timestamp), UTC).date().isoformat()
    return {
        "documents": len(data.get("documents") or []),
        "paragraphs": len(data.get("paragraph_index") or []),
        "date": date,
    }


def _latest_json(directory: Path, pattern: str) -> dict[str, Any]:
    reports = sorted(directory.glob(pattern)) if directory.exists() else []
    if not reports:
        return {}
    try:
        return json.loads(reports[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _named_row(rows: Any, name: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return {}


def _fmt(value: Any) -> str:
    return f"{int(value):,}" if isinstance(value, int | float) else str(value)
