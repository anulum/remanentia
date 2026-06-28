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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

BASE = Path(__file__).parent
INDEX_PATH = BASE / "snn_state" / "memory_index.json.gz"
BENCHMARK_DIR = BASE / ".coordination" / "benchmarks" / "REMANENTIA"


JsonObject = dict[str, Any]


class BenchmarkQuery(TypedDict):
    """Retrieval benchmark query row consumed by local evaluation tools."""

    q: str
    gold: list[str]
    cat: str
    suite: str


class IndexStats(TypedDict):
    """Normalized unified-index counters used by benchmark queries."""

    documents: int
    paragraphs: int
    date: str


def current_operational_queries(repo: Path = BASE) -> list[BenchmarkQuery]:
    """Build the live operational retrieval benchmark query suite.

    Parameters
    ----------
    repo:
        Repository root containing ``snn_state/memory_index.json.gz`` and the
        benchmark report directory. The default points at this module's checkout.

    Returns
    -------
    list[BenchmarkQuery]
        Query rows whose expected answers are available from the current local
        artefacts. Rows with missing dynamic gold values are omitted.
    """

    repo = repo.resolve()
    index_stats = _index_stats(repo / "snn_state" / "memory_index.json.gz")
    performance = _latest_json(
        repo / ".coordination" / "benchmarks" / "REMANENTIA", "remanentia_performance_*.json"
    )
    recall_api = _named_row(performance.get("api"), "api_recall")
    public_vector = _named_row(performance.get("api"), "api_public_vector_search")
    vector_report = performance.get("vector")
    vector_index = (
        _json_object(vector_report.get("index")) if isinstance(vector_report, dict) else {}
    )
    queries: list[BenchmarkQuery] = [
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


def historical_regression_queries() -> list[BenchmarkQuery]:
    """Return fixed regression queries for historical retrieval incidents.

    Returns
    -------
    list[BenchmarkQuery]
        Stable query rows that preserve known benchmark expectations from older
        Remanentia index and identity-coherence incidents.
    """

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


def _index_stats(path: Path) -> IndexStats:
    """Read normalized document, paragraph, and build-date stats from an index.

    Parameters
    ----------
    path:
        Gzip-compressed JSON index path.

    Returns
    -------
    IndexStats
        Zero-valued counters and an empty date when the index is missing,
        unreadable, malformed, or not a JSON object.
    """

    if not path.exists():
        return {"documents": 0, "paragraphs": 0, "date": ""}
    try:
        with gzip.open(path, "rb") as handle:
            payload = json.loads(handle.read())
    except (OSError, json.JSONDecodeError):
        return {"documents": 0, "paragraphs": 0, "date": ""}
    data = _json_object(payload)
    timestamp = data.get("timestamp")
    date = ""
    if isinstance(timestamp, int | float):
        date = datetime.fromtimestamp(float(timestamp), timezone.utc).date().isoformat()
    documents = data.get("documents")
    paragraphs = data.get("paragraph_index")
    return {
        "documents": len(documents) if isinstance(documents, list) else 0,
        "paragraphs": len(paragraphs) if isinstance(paragraphs, list) else 0,
        "date": date,
    }


def _latest_json(directory: Path, pattern: str) -> JsonObject:
    """Load the newest JSON object report matching a glob pattern.

    Parameters
    ----------
    directory:
        Directory containing benchmark report artefacts.
    pattern:
        Glob pattern used to select candidate reports.

    Returns
    -------
    JsonObject
        Parsed report object, or an empty object for missing, unreadable,
        invalid, or non-object JSON content.
    """

    reports = sorted(directory.glob(pattern)) if directory.exists() else []
    if not reports:
        return {}
    try:
        payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _json_object(payload)


def _named_row(rows: object, name: str) -> JsonObject:
    """Find a named dictionary row inside a report table.

    Parameters
    ----------
    rows:
        Candidate list of benchmark result rows.
    name:
        Expected value of the row's ``name`` field.

    Returns
    -------
    JsonObject
        Matching row copied as a string-keyed object, or an empty object when
        no matching row is present.
    """

    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("name") == name:
            return {str(key): value for key, value in row.items()}
    return {}


def _fmt(value: object) -> str:
    """Format numeric benchmark gold values with thousands separators.

    Parameters
    ----------
    value:
        Dynamic benchmark value to render.

    Returns
    -------
    str
        Comma-formatted integer text for numeric input, otherwise ``str(value)``.
    """

    return f"{int(value):,}" if isinstance(value, int | float) else str(value)


def _json_object(payload: object) -> JsonObject:
    """Return ``payload`` when it is a JSON object, otherwise an empty object."""

    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}
