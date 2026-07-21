# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Retrieval history and trace catalog

"""Filesystem-backed history and catalog operations for query retrieval."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

try:
    from .retrieval_text import tokenize  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - top-level script compatibility
    from retrieval_text import tokenize

TIER_HOT_HOURS = 24
TIER_WARM_DAYS = 7
TIER_WEIGHTS: Mapping[str, float] = {"hot": 1.02, "warm": 1.0, "cold": 0.98}


def append_history(
    path: Path,
    query: str,
    results: Sequence[Mapping[str, object]],
    *,
    timestamp: float | None = None,
) -> None:
    """Append one bounded retrieval record to a JSONL history file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = [{"trace": result["trace"], "score": result["score"]} for result in results[:5]]
    entry = {
        "query": query,
        "timestamp": time.time() if timestamp is None else timestamp,
        "results": ranked,
    }
    with path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(entry) + "\n")


def read_history(path: Path, limit: int = 50) -> list[dict[str, object]]:
    """Read the newest valid object records from retrieval JSONL history."""
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            decoded = cast(object, json.loads(line))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            rows.append(cast(dict[str, object], decoded))
    return rows[-limit:]


def summarize_traces(traces_dir: Path) -> list[dict[str, str]]:
    """Extract the first content line from each trace, newest first."""
    if not traces_dir.exists():
        return []
    summaries: list[dict[str, str]] = []
    files = sorted(traces_dir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files:
        summary = ""
        past_heading = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                past_heading = True
                continue
            if not past_heading or not stripped or stripped.startswith("---"):
                continue
            if stripped.startswith("**") and ":**" in stripped:
                continue
            if stripped.startswith("- **") and ":**" in stripped:
                continue
            summary = stripped.lstrip("- ").strip()
            break
        summaries.append({"name": path.name, "summary": summary or "(no summary)"})
    return summaries


def classify_trace_tier(path: Path, *, timestamp: float | None = None) -> str:
    """Classify a trace as hot, warm, or cold from its real modification time."""
    current_time = time.time() if timestamp is None else timestamp
    age_hours = (current_time - path.stat().st_mtime) / 3600
    if age_hours <= TIER_HOT_HOURS:
        return "hot"
    if age_hours <= TIER_WARM_DAYS * 24:
        return "warm"
    return "cold"


def tier_boost(tier: str) -> float:
    """Return the bounded retrieval weight for a persistence tier."""
    return TIER_WEIGHTS.get(tier, 1.0)


def find_related_traces(
    traces_dir: Path, trace_name: str, top_k: int = 3
) -> list[dict[str, object]]:
    """Rank neighboring Markdown traces by lexical Jaccard similarity."""
    root = traces_dir.resolve()
    stem = trace_name[:-3] if trace_name.endswith(".md") else ""
    if (
        not stem
        or len(trace_name) > 255
        or not all(char.isascii() and (char.isalnum() or char in "_.-") for char in stem)
    ):
        return []
    target = (root / trace_name).resolve()
    if not target.is_relative_to(root) or not target.exists() or not target.is_file():
        return []

    target_text = target.read_text(encoding="utf-8")
    target_tokens = set(
        tokenize(target_text + " " + target.stem.replace("-", " ").replace("_", " "))
    )
    if not target_tokens:
        return []

    scored: list[dict[str, object]] = []
    for path in root.glob("*.md"):
        if path.name == trace_name:
            continue
        text = path.read_text(encoding="utf-8")
        tokens = set(tokenize(text + " " + path.stem.replace("-", " ").replace("_", " ")))
        overlap = target_tokens & tokens
        if not overlap:
            continue
        scored.append(
            {
                "trace": path.name,
                "similarity": round(len(overlap) / len(target_tokens | tokens), 4),
                "shared_terms": len(overlap),
            }
        )
    scored.sort(key=lambda result: cast(float, result["similarity"]), reverse=True)
    return scored[:top_k]


def suggest_queries(path: Path, prefix: str, limit: int = 8) -> list[str]:
    """Return newest distinct history queries matching a case-insensitive prefix."""
    if not path.exists() or len(prefix) < 2:
        return []
    prefix_lower = prefix.lower()
    seen: set[str] = set()
    suggestions: list[str] = []
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            decoded = cast(object, json.loads(line))
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        query = decoded.get("query", "")
        if not isinstance(query, str):
            continue
        if query.lower().startswith(prefix_lower) and query not in seen:
            seen.add(query)
            suggestions.append(query)
            if len(suggestions) >= limit:
                break
    return suggestions


def chunk_trace_catalog(traces_dir: Path) -> list[dict[str, object]]:
    """Group trace filenames by their project and date filename fields."""
    if not traces_dir.exists():
        return []
    groups: dict[str, list[Path]] = {}
    for path in sorted(traces_dir.glob("*.md")):
        parts = path.stem.split("_", 2)
        date = parts[0].split("T")[0]
        project = parts[1] if len(parts) > 1 else "general"
        groups.setdefault(f"{project} {date}", []).append(path)

    chunks: list[dict[str, object]] = []
    for key, files in groups.items():
        project, _, date = key.partition(" ")
        count = len(files)
        chunks.append(
            {
                "name": key,
                "project": project,
                "date": date,
                "traces": [path.name for path in files],
                "count": count,
                "summary": f"{project}: {count} trace{'s' if count > 1 else ''} ({date})",
            }
        )
    chunks.sort(key=lambda chunk: cast(str, chunk["date"]), reverse=True)
    return chunks
