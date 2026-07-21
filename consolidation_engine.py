# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Consolidation pipeline

"""Consolidation pipeline: episodic traces → semantic memories.

The SNN daemon calls consolidate() periodically. The engine:
1. Detects new/unprocessed traces
2. Clusters related traces into episodes
3. Extracts structured facts from each episode
4. Builds/updates entity relationship graph
5. Detects conflicts with existing memories
6. Writes semantic memory files

No LLM required — uses heuristic extraction from structured traces.
Can be upgraded to LLM extraction later.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np

from store_paths import default_base

BASE = default_base()
TRACES_DIR = BASE / "reasoning_traces"
SEMANTIC_DIR = BASE / "memory" / "semantic"
GRAPH_DIR = BASE / "memory" / "graph"
CONSOLIDATION_DIR = BASE / "consolidation"
PENDING_PATH = CONSOLIDATION_DIR / "pending.json"
LAST_RUN_PATH = CONSOLIDATION_DIR / "last_consolidation.json"
CONFLICTS_PATH = CONSOLIDATION_DIR / "conflicts.json"
ENTITIES_PATH = GRAPH_DIR / "entities.jsonl"
RELATIONS_PATH = GRAPH_DIR / "relations.jsonl"
CLUSTERS_PATH = GRAPH_DIR / "trace_clusters.json"


# ── Trace metadata extraction ────────────────────────────────────

from consolidation_trace_analysis import (
    JsonObject,
    PROJECT_NAMES,
    TraceData,
    cluster_traces as _cluster_traces,
    extract_entities as _extract_entities,
    extract_key_lines as _extract_key_lines,
    extract_metadata as _extract_metadata,
    extract_paragraphs as _extract_paragraphs,
    trace_hash as _trace_hash,  # noqa: F401 - retained as a compatibility export
)


# ── Semantic memory writing ──────────────────────────────────────


# ── Memory lifecycle states ──────────────────────────────────────
#
# Every semantic memory has a validity_state that progresses:
#   active → validated → stale → archived
#
# Transitions:
#   active → validated : when a later trace confirms the same fact
#   active/validated → stale : when age exceeds STALE_AFTER_DAYS without access
#   stale → archived : when age exceeds ARCHIVE_AFTER_DAYS
#   any → active : when a trace re-references the memory (reset)
#   any → archived : when contradicted by a newer fact
#
VALID_STATES = ("active", "validated", "stale", "archived")
STALE_AFTER_DAYS = 90
ARCHIVE_AFTER_DAYS = 365

# ── Bounded memory capacity ─────────────────────────────────────
#
# Each semantic memory category has a soft char limit. When a
# category exceeds CAPACITY_WARN_PERCENT, `capacity_report()`
# flags it for consolidation.  This mirrors Hermes Agent's
# bounded MEMORY.md pattern but at per-category granularity.
#
CATEGORY_CHAR_LIMITS = {
    "decision": 50_000,
    "finding": 100_000,
    "strategy": 30_000,
    "technical": 100_000,
    "continuity": 20_000,
    "personal": 20_000,
    "findings": 100_000,  # alias used by consolidate()
    "general": 50_000,
}
DEFAULT_CATEGORY_CHAR_LIMIT = 50_000
CAPACITY_WARN_PERCENT = 80


def _write_semantic_memory(
    category: str,
    topic: str,
    date: str,
    project: str,
    source_traces: list[str],
    entities: list[str],
    content: str,
    validity_state: str = "active",
    redact_pii: bool = True,
) -> Path:
    """Write a structured semantic memory file with lifecycle state.

    PII redaction runs on *content* by default so emails / phones /
    bank details / API keys do not get indexed into memories.
    """
    from file_utils import atomic_write_text

    if redact_pii:
        from pii_redactor import redact

        content = redact(content).text

    cat_dir = SEMANTIC_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    safe_topic = re.sub(r"[^a-zA-Z0-9_-]", "-", topic.lower())[:60]
    filename = f"{date}_{safe_topic}.md"
    path = cat_dir / filename

    frontmatter = {
        "type": category,
        "date": date,
        "project": project,
        "source_traces": source_traces,
        "entities": entities,
        "confidence": 0.8,
        "validity_state": validity_state,
        "last_validated": datetime.now().strftime("%Y-%m-%d"),
        "last_accessed": datetime.now().strftime("%Y-%m-%d"),
    }

    lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append(content)

    atomic_write_text(path, "\n".join(lines))
    return path


# ── Entity graph ─────────────────────────────────────────────────


def _load_entities() -> dict[str, JsonObject]:
    """Load persisted entity graph nodes keyed by entity id."""

    if not ENTITIES_PATH.exists():
        return {}
    entities: dict[str, JsonObject] = {}
    for line in ENTITIES_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            e = cast(JsonObject, json.loads(line))
            entities[e["id"]] = e
    return entities


def _save_entities(entities: dict[str, JsonObject]) -> None:
    """Persist entity graph nodes as JSONL records."""

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e) for e in entities.values()]
    ENTITIES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_relations() -> list[JsonObject]:
    """Load persisted weighted entity graph relations."""

    if not RELATIONS_PATH.exists():
        return []
    relations: list[JsonObject] = []
    for line in RELATIONS_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            relations.append(cast(JsonObject, json.loads(line)))
    return relations


def _save_relations(relations: list[JsonObject]) -> None:
    """Persist weighted entity graph relations as JSONL records."""

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in relations]
    RELATIONS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_graph(
    trace_name: str, entities: list[str], project: str, date: str, text: str = ""
) -> None:
    """Add entities and relations from a trace.

    Extracts typed relations (caused_by, fixed_by, replaced, etc.) when text
    is provided. Falls back to co_occurs for entity pairs without a detected
    typed relation.
    """
    entity_db = _load_entities()
    relations = _load_relations()

    for ent in entities:
        if ent not in entity_db:
            entity_db[ent] = {
                "id": ent,
                "type": "project" if ent in PROJECT_NAMES else "concept",
                "label": ent,
                "first_seen": date,
                "trace_count": 1,
            }
        else:
            entity_db[ent]["trace_count"] = entity_db[ent].get("trace_count", 0) + 1

    # Extract typed relations from text when available
    typed_pairs: dict[tuple[str, str], str] = {}
    if text:
        typed_pairs = _extract_typed_relations(text, entities)

    # Build relation edges
    existing_pairs = {(r["source"], r["target"]) for r in relations}
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1 :]:
            pair = _entity_pair(e1, e2)
            rel_type = typed_pairs.get(pair, typed_pairs.get((pair[1], pair[0]), "co_occurs"))
            if pair in existing_pairs:
                for r in relations:
                    if (r["source"], r["target"]) == pair or (r["target"], r["source"]) == pair:
                        r["weight"] = r.get("weight", 0) + 1
                        if trace_name not in r.get("evidence", []):
                            r.setdefault("evidence", []).append(trace_name)
                        # Upgrade co_occurs to typed if we now have a typed relation
                        if r["type"] == "co_occurs" and rel_type != "co_occurs":
                            r["type"] = rel_type
                        break
            else:
                relations.append(
                    {
                        "source": pair[0],
                        "target": pair[1],
                        "type": rel_type,
                        "weight": 1,
                        "evidence": [trace_name],
                    }
                )
                existing_pairs.add(pair)

    _save_entities(entity_db)
    _save_relations(relations)


_TYPED_RELATION_PATTERNS = [
    (re.compile(r"\bbecause\b|\bcaused by\b|\bdue to\b|\broot cause\b", re.I), "caused_by"),
    (re.compile(r"\bfixed\b|\brepaired\b|\bcorrected\b|\bpatched\b", re.I), "fixed_by"),
    (re.compile(r"\breplaced\b|\bsuperseded\b|\binstead of\b", re.I), "replaced"),
    (
        re.compile(r"\bcontradicts?\b|\binconsistent with\b|\bconflicts? with\b", re.I),
        "contradicts",
    ),
    (re.compile(r"\bv\d+\.\d+|\bversion\b", re.I), "version_of"),
    (re.compile(r"\bdepends on\b|\brequires\b|\bneeds\b", re.I), "depends_on"),
    (re.compile(r"\bimproved\b|\bfrom .+ to\b|\bincreased\b|\bdecreased\b", re.I), "improved"),
    (re.compile(r"\bproduced\b|\bcreated\b|\bgenerated\b|\bwrote\b", re.I), "produced"),
    (re.compile(r"\bused in\b|\bpart of\b|\bcomponent of\b", re.I), "used_in"),
    (re.compile(r"\btested\b|\bbenchmarked\b|\bevaluated\b|\bmeasured\b", re.I), "tested_with"),
]


def _extract_typed_relations(text: str, entities: list[str]) -> dict[tuple[str, str], str]:
    """Extract typed relations between entity pairs from their connecting text."""
    try:
        _rust_rels = import_module("remanentia_consolidation").extract_typed_relations

        rust_result = _rust_rels(text, entities)  # pragma: no cover
        return {(s, t): r for s, t, r in rust_result}  # pragma: no cover
    except ImportError:
        pass
    text_lower = text.lower()
    typed: dict[tuple[str, str], str] = {}
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1 :]:
            pos1 = text_lower.find(e1.lower())
            pos2 = text_lower.find(e2.lower())
            if pos1 < 0 or pos2 < 0:
                continue
            start = min(pos1, pos2)
            end = max(pos1 + len(e1), pos2 + len(e2))
            between = text[start:end]
            pair = _entity_pair(e1, e2)
            matched = False
            for pattern, rel_type in _TYPED_RELATION_PATTERNS:
                if pattern.search(between):
                    typed[pair] = rel_type
                    matched = True
                    break
            if not matched:
                typed[pair] = "co_occurs"
    return typed


def _entity_pair(first: str, second: str) -> tuple[str, str]:
    """Return a stable two-entity key."""
    return (first, second) if first <= second else (second, first)


# ── Novelty detection ────────────────────────────────────────────

_running_mean = None
_running_count = 0


def compute_novelty(spike_pattern: np.ndarray) -> float:
    """Novelty = 1 - cosine(pattern, running_mean).
    High novelty = this trace activates unusual neurons.
    """
    global _running_mean, _running_count
    if _running_mean is None:
        _running_mean = spike_pattern.copy()
        _running_count = 1
        return 1.0

    norm_p = np.linalg.norm(spike_pattern)
    norm_m = np.linalg.norm(_running_mean)
    if norm_p < 1e-12 or norm_m < 1e-12:
        return 1.0

    cosine = float(np.dot(spike_pattern, _running_mean) / (norm_p * norm_m))
    novelty = 1.0 - cosine

    # Update running mean
    _running_count += 1
    alpha = 1.0 / _running_count
    _running_mean = (1 - alpha) * _running_mean + alpha * spike_pattern

    return novelty


# ── Main consolidation pipeline ──────────────────────────────────


def get_pending_traces() -> list[str]:
    """Find traces not yet consolidated."""
    CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    if PENDING_PATH.exists():
        pending = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    else:
        pending = {"processed": []}

    all_traces = {f.name for f in TRACES_DIR.glob("*.md")}
    processed = set(pending.get("processed", []))
    return sorted(all_traces - processed)


def consolidate(force: bool = False) -> JsonObject:
    """Run one consolidation cycle.

    Returns stats about what was consolidated.
    """
    CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

    new_traces = get_pending_traces()
    if not new_traces and not force:
        return {"status": "nothing_to_consolidate", "pending": 0}

    if not new_traces:  # pragma: no cover
        new_traces = [f.name for f in sorted(TRACES_DIR.glob("*.md"))]

    # Load and analyze each trace
    trace_data: TraceData = {}
    for name in new_traces:
        path = TRACES_DIR / name
        if not path.exists():  # pragma: no cover
            continue
        text = path.read_text(encoding="utf-8")
        meta = _extract_metadata(name, text)
        entities = _extract_entities(text)
        key_lines = _extract_key_lines(text)
        paragraphs = _extract_paragraphs(text)
        trace_data[name] = {
            **meta,
            "entities": entities,
            "key_lines": key_lines,
            "paragraphs": paragraphs,
            "text": text,
        }

    # Cluster
    clusters = _cluster_traces(trace_data)

    # Process each cluster
    memories_written = 0
    entities_total = set()

    for cluster in clusters:
        if not cluster:  # pragma: no cover
            continue

        cluster_data = [trace_data[n] for n in cluster if n in trace_data]
        if not cluster_data:  # pragma: no cover
            continue

        # Aggregate metadata
        project = Counter(d["project"] for d in cluster_data).most_common(1)[0][0]
        date = cluster_data[0].get("date", "unknown")
        trace_type = Counter(d["type"] for d in cluster_data).most_common(1)[0][0]
        all_entities = sorted(set(e for d in cluster_data for e in d["entities"]))
        all_key_lines = [l for d in cluster_data for l in d["key_lines"]]

        entities_total.update(all_entities)

        # Write semantic memory — full text preserved, no information loss
        topic = f"{project}-{trace_type}"
        content_lines = [f"# {project} — {trace_type} ({date})\n"]
        content_lines.append(f"Consolidated from {len(cluster)} traces.\n")

        # Key findings (expanded extraction)
        if all_key_lines:
            content_lines.append("## Key Findings\n")
            seen = set()
            for line in all_key_lines:
                if line not in seen:
                    content_lines.append(f"- {line}")
                    seen.add(line)

        # Full content from all traces — zero information loss
        content_lines.append("\n## Full Content\n")
        for d in cluster_data:
            content_lines.append(f"### {d['filename']}\n")
            for para in d.get("paragraphs", []):
                content_lines.append(para)
                content_lines.append("")

        content = "\n".join(content_lines)

        _write_semantic_memory(
            category=trace_type if trace_type != "general" else "findings",
            topic=topic,
            date=date,
            project=project,
            source_traces=cluster,
            entities=all_entities,
            content=content,
        )
        memories_written += 1

        # Update entity graph with typed relations
        for name in cluster:
            if name in trace_data:
                _update_graph(
                    name,
                    trace_data[name]["entities"],
                    trace_data[name]["project"],
                    trace_data[name].get("date", ""),
                    text=trace_data[name].get("text", ""),
                )

    # Save clusters
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERS_PATH.write_text(json.dumps(clusters, indent=2) + "\n", encoding="utf-8")

    # Build and save hierarchical summary DAG
    dag_nodes = build_summary_dag(trace_data)
    if dag_nodes:
        SUMMARY_DAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_DAG_PATH.write_text(json.dumps(dag_nodes, indent=2) + "\n", encoding="utf-8")

    # Mark traces as processed
    if PENDING_PATH.exists():  # pragma: no cover
        pending = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    else:
        pending = {"processed": []}
    pending["processed"] = sorted(set(pending["processed"]) | set(new_traces))
    PENDING_PATH.write_text(json.dumps(pending, indent=2) + "\n", encoding="utf-8")

    # Save consolidation stats
    stats = {
        "timestamp": time.time(),
        "traces_processed": len(new_traces),
        "clusters_formed": len(clusters),
        "memories_written": memories_written,
        "entities_found": len(entities_total),
        "entity_list": sorted(entities_total),
    }
    LAST_RUN_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    return stats


def capacity_report() -> dict[str, JsonObject]:
    """Report memory capacity usage per semantic category.

    Returns a dict mapping category name to:
    - chars: total characters in that category
    - limit: configured char limit
    - usage_pct: percentage used (0-100+)
    - needs_consolidation: True if above CAPACITY_WARN_PERCENT
    - file_count: number of .md files
    - state_counts: breakdown by validity_state
    """
    report: dict[str, JsonObject] = {}
    if not SEMANTIC_DIR.exists():
        return report

    for cat_dir in sorted(SEMANTIC_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        total_chars = 0
        file_count = 0
        state_counts: dict[str, int] = {}

        for md_file in cat_dir.rglob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            total_chars += len(text)
            file_count += 1
            fm = _parse_frontmatter(text)
            if fm:
                state = fm.get("validity_state", "active")
                state_counts[state] = state_counts.get(state, 0) + 1

        limit = CATEGORY_CHAR_LIMITS.get(category, DEFAULT_CATEGORY_CHAR_LIMIT)
        usage_pct = round(100 * total_chars / limit, 1) if limit > 0 else 0.0

        report[category] = {
            "chars": total_chars,
            "limit": limit,
            "usage_pct": usage_pct,
            "needs_consolidation": usage_pct >= CAPACITY_WARN_PERCENT,
            "file_count": file_count,
            "state_counts": state_counts,
        }

    return report


# ── Hierarchical summary DAGs ────────────────────────────────────
#
# Inspired by Engram's Lossless Context Management (LCM).
# Multi-level compression of episodic traces:
#   Level 0 (leaf): individual traces (raw .md files)
#   Level 1: cluster summaries (~8 traces each)
#   Level 2: super-summaries (~32 traces = ~4 L1 nodes)
#   Level 3: meta-summaries (~128 traces = ~4 L2 nodes)
#
# Each node stores:
#   - summary text (heuristic extraction, no LLM needed)
#   - children (trace filenames or lower-level node IDs)
#   - date range
#   - entities mentioned
#
# The DAG enables efficient retrieval over long histories
# by searching at the appropriate depth first, then drilling
# down to leaf nodes for detail.

SUMMARY_DAG_PATH = CONSOLIDATION_DIR / "summary_dag.json"
from consolidation_summary_dag import (
    DAG_FANOUT,  # noqa: F401 - retained as a compatibility export
    DAGNode,  # noqa: F401 - retained as a compatibility export
    DAGNodeDict,  # noqa: F401 - retained as a compatibility export
    build_summary_dag,
    build_summary_dag_python,  # noqa: F401 - retained as a compatibility export
    search_summary_dag,  # noqa: F401 - retained as a compatibility export
)


def age_memories(reference_date: str | None = None) -> JsonObject:
    """Age semantic memories based on last_accessed and last_validated timestamps.

    Transitions:
    - active/validated → stale after STALE_AFTER_DAYS without access
    - stale → archived after ARCHIVE_AFTER_DAYS without access

    Returns stats: total scanned, transitioned counts per state.
    """
    if reference_date:
        ref = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        ref = datetime.now()

    stats = {"scanned": 0, "active_to_stale": 0, "validated_to_stale": 0, "stale_to_archived": 0}

    if not SEMANTIC_DIR.exists():
        return stats

    for md_file in SEMANTIC_DIR.rglob("*.md"):
        stats["scanned"] += 1
        text = md_file.read_text(encoding="utf-8")

        # Parse frontmatter
        fm = _parse_frontmatter(text)
        if not fm:
            continue

        state = fm.get("validity_state", "active")
        last_accessed = fm.get("last_accessed", fm.get("last_validated", ""))
        if not last_accessed:
            continue

        try:
            accessed = datetime.strptime(last_accessed[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        age_days = (ref - accessed).days
        new_state = state

        if state in ("active", "validated") and age_days > STALE_AFTER_DAYS:
            new_state = "stale"
            key = f"{state}_to_stale"
            stats[key] = stats.get(key, 0) + 1
        elif state == "stale" and age_days > ARCHIVE_AFTER_DAYS:
            new_state = "archived"
            stats["stale_to_archived"] += 1

        if new_state != state:
            _update_frontmatter_field(md_file, text, "validity_state", new_state)

    return stats


def _parse_frontmatter(text: str) -> JsonObject | None:
    """Parse YAML-ish frontmatter from a semantic memory file."""
    try:
        native_parse = import_module("remanentia_consolidation").parse_frontmatter
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return _parse_frontmatter_python(text)
    rust_result = native_parse(text)
    return cast(JsonObject | None, rust_result if rust_result else None)  # pragma: no cover


def _parse_frontmatter_python(text: str) -> JsonObject | None:
    """Parse semantic-memory frontmatter without the optional native engine."""
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end < 0:
        return None
    fm_text = text[3:end].strip()
    parsed: JsonObject = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("-"):
            key, _, val = line.partition(":")
            parsed[key.strip()] = val.strip()
    return parsed


def _update_frontmatter_field(path: Path, text: str, field_name: str, value: str) -> None:
    """Update a single field in a frontmatter block and write back."""
    pattern = re.compile(rf"^{re.escape(field_name)}:\s*.*$", re.MULTILINE)
    if pattern.search(text):
        new_text = pattern.sub(f"{field_name}: {value}", text)
    else:
        # Insert after the opening ---
        new_text = text.replace("---\n", f"---\n{field_name}: {value}\n", 1)
    path.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    print("Running initial consolidation...")
    result = consolidate(force=True)
    print(json.dumps(result, indent=2))
