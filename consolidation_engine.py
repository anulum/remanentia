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

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
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

_PROJECT_PATTERNS = [
    ("director-ai", ["director-ai", "director_ai"]),
    ("sc-neurocore", ["sc-neurocore", "neurocore"]),
    ("scpn-fusion-core", ["scpn-fusion", "fusion-core"]),
    ("scpn-phase-orchestrator", ["scpn-phase-orchestrator", "phase-orchestrator"]),
    ("scpn-control", ["scpn-control"]),
    ("scpn-quantum-control", ["scpn-quantum-control", "quantum-control"]),
    ("remanentia", ["remanentia", "arcane", "snn", "holographic"]),
    ("revenue", ["revenue", "pricing", "commercial"]),
]

_TYPE_PATTERNS = [
    ("decision", ["decision", "audit", "architecture"]),
    ("finding", ["breakthrough", "failure", "analysis", "experiment"]),
    ("strategy", ["strategy", "revenue", "sprint", "competitive"]),
    ("technical", ["daemon", "migration", "convergence", "deepening", "fix"]),
    ("continuity", ["continuity", "contribution", "identity"]),
    ("personal", ["personal", "partnership", "moment"]),
]


def _extract_metadata(filename: str, text: str) -> dict:
    ts_match = re.match(r"(\d{4}-\d{2}-\d{2})(T(\d{4}))?_", filename)
    timestamp = ""
    if ts_match:
        timestamp = ts_match.group(1)
        if ts_match.group(3):
            timestamp += "T" + ts_match.group(3)

    fname_lower = filename.lower()
    project = "general"
    for proj, patterns in _PROJECT_PATTERNS:
        if any(p in fname_lower for p in patterns):
            project = proj
            break

    trace_type = "general"
    for ttype, patterns in _TYPE_PATTERNS:
        if any(p in fname_lower for p in patterns):
            trace_type = ttype
            break

    return {"filename": filename, "date": timestamp, "project": project, "type": trace_type}


def _extract_entities(text: str) -> list[str]:
    """Extract entity names from trace text.

    Three extraction layers:
    1. Project names (from _PROJECT_PATTERNS)
    2. Known concepts (expanded list)
    3. Dynamic extraction: version numbers, file paths, function names,
       numeric results, person names
    """
    try:
        from remanentia_consolidation import extract_entities as _rust_ents

        return _rust_ents(text)  # pragma: no cover
    except ImportError:
        pass
    entities = set()
    text_lower = text.lower()

    # Layer 1: Projects
    for proj, patterns in _PROJECT_PATTERNS:
        if any(p in text_lower for p in patterns):
            entities.add(proj)

    # Layer 2: Known concepts (expanded from 22 to 60+)
    concepts = [
        "STDP",
        "LIF",
        "Kuramoto",
        "Hopfield",
        "TF-IDF",
        "BM25",
        "embedding",
        "PyTorch",
        "CUDA",
        "GPU",
        "CPU",
        "daemon",
        "holographic",
        "attractor",
        "inhibition",
        "spike",
        "neuron",
        "retrieval",
        "consolidation",
        "UPDE",
        "Stuart-Landau",
        "Dimits",
        "gyrokinetic",
        "tokamak",
        "VQE",
        "Heron",
        "BCPNN",
        "CSDP",
        "Hebbian",
        "Perron-Frobenius",
        "Marchenko-Pastur",
        "eigenvalue",
        "SVD",
        "MiniLM",
        "sentence-transformer",
        "FastAPI",
        "MCP",
        "Docker",
        "Prometheus",
        "Grafana",
        "CI",
        "pytest",
        "Rust",
        "PyO3",
        "maturin",
        "Rayon",
        "ArcaneNeuron",
        "chirp",
        "chimera",
        "bifurcation",
        "entropy",
        "Fisher",
        "Lyapunov",
        "Boltzmann",
        "hippocampus",
        "dentate gyrus",
        "pattern separation",
        "Dale's law",
        "E/I balance",
        "Mem0",
        "Letta",
        "Zep",
        "MemOS",
        "LangMem",
        "JOSS",
        "NeurIPS",
        "EMNLP",
        "arXiv",
        "Zenodo",
        "AGPL",
        "PyPI",
        "Loihi",
    ]
    for concept in concepts:
        if concept.lower() in text_lower:
            entities.add(concept.lower())

    # Layer 3: Dynamic extraction
    # Version numbers (v0.1.0, v3.9.0, etc.)
    for m in re.finditer(r"v\d+\.\d+(?:\.\d+)?", text):
        entities.add(m.group())

    # Percentages with context (e.g., "92.9%", "78.6%")
    for m in re.finditer(r"\d+\.?\d*%", text):
        entities.add(m.group())

    # File paths (.py, .rs, .md, .json)
    for m in re.finditer(r"[\w/\\]+\.(?:py|rs|md|json|yaml|toml)\b", text):
        name = m.group().split("/")[-1].split("\\")[-1]
        if len(name) > 3:
            entities.add(name)

    # Function/class names (word_word pattern or CamelCase)
    for m in re.finditer(r"\b[a-z][a-z_]+(?:_[a-z]+){2,}\b", text):
        if len(m.group()) > 8:
            entities.add(m.group())
    for m in re.finditer(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+){1,}\b", text):
        entities.add(m.group())

    return sorted(entities)


def _extract_paragraphs(text: str) -> list[str]:
    """Split text into meaningful paragraphs for indexing."""
    paragraphs = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        # Skip pure headers (but keep header + content blocks)
        lines = stripped.split("\n")
        content_lines = [l for l in lines if not l.strip().startswith("#") or len(lines) > 1]
        content = "\n".join(content_lines).strip()
        if len(content) > 30:
            paragraphs.append(content)
    return paragraphs


def _extract_key_lines(text: str) -> list[str]:
    """Extract decision/finding lines from trace text.

    Expanded trigger set + multi-line capture: when a trigger fires,
    grab the next 2 non-empty lines as context.
    """
    try:
        from remanentia_consolidation import extract_key_lines as _rust_kl

        return _rust_kl(text)  # pragma: no cover
    except ImportError:
        pass
    lines = text.split("\n")
    key_lines = []
    triggers = [
        "decided",
        "decision",
        "found",
        "finding",
        "result",
        "key insight",
        "conclusion",
        "fix",
        "resolved",
        "chose",
        "rejected",
        "confirmed",
        "measured",
        "p@1",
        "precision",
        "accuracy",
        "because",
        "therefore",
        "root cause",
        "the reason",
        "we proved",
        "this means",
        "critical",
        "important",
        "changed",
        "broke",
        "works",
        "doesn't work",
        "failed",
        "succeeded",
        "shipped",
        "version",
        "v0.",
        "v1.",
        "v2.",
        "v3.",
    ]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(t in stripped.lower() for t in triggers):
            clean = stripped.lstrip("- *>").strip()
            if len(clean) > 20:
                # Grab context: next 2 non-empty lines
                context = [clean]
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip().lstrip("- *>").strip()
                    if next_line and not next_line.startswith("#") and len(next_line) > 10:
                        context.append(next_line)
                key_lines.append(" ".join(context))
    return key_lines[:30]  # raised cap from 10 to 30


def _trace_hash(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:12]


# ── Clustering ───────────────────────────────────────────────────


def _cluster_traces(traces: dict[str, dict]) -> list[list[str]]:
    """Cluster traces by project + date proximity.

    Traces from the same project within 2 days of each other are grouped.
    A gap > 2 days starts a new cluster.
    """
    by_project = defaultdict(list)
    for name, meta in traces.items():
        by_project[meta["project"]].append(name)

    clusters = []
    for proj, names in by_project.items():
        names.sort(key=lambda n: traces[n].get("date", ""))
        if not names:  # pragma: no cover
            continue
        current_cluster = [names[0]]
        for i in range(1, len(names)):
            prev_date = traces[names[i - 1]].get("date", "")[:10]
            curr_date = traces[names[i]].get("date", "")[:10]
            # Parse dates and check gap
            try:
                from datetime import datetime as dt

                d1 = dt.strptime(prev_date, "%Y-%m-%d")
                d2 = dt.strptime(curr_date, "%Y-%m-%d")
                gap_days = abs((d2 - d1).days)
            except (ValueError, TypeError):
                gap_days = 0
            if gap_days > 2:
                clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(names[i])
        if current_cluster:
            clusters.append(current_cluster)

    return clusters


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
) -> Path:
    """Write a structured semantic memory file with lifecycle state."""
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

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Entity graph ─────────────────────────────────────────────────


def _load_entities() -> dict[str, dict]:
    if not ENTITIES_PATH.exists():
        return {}
    entities = {}
    for line in ENTITIES_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            e = json.loads(line)
            entities[e["id"]] = e
    return entities


def _save_entities(entities: dict[str, dict]):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e) for e in entities.values()]
    ENTITIES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_relations() -> list[dict]:
    if not RELATIONS_PATH.exists():
        return []
    relations = []
    for line in RELATIONS_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            relations.append(json.loads(line))
    return relations


def _save_relations(relations: list[dict]):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in relations]
    RELATIONS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_graph(trace_name: str, entities: list[str], project: str, date: str, text: str = ""):
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
                "type": "project" if ent in [p for p, _ in _PROJECT_PATTERNS] else "concept",
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
            pair = tuple(sorted([e1, e2]))
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
        from remanentia_consolidation import extract_typed_relations as _rust_rels

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
            for pattern, rel_type in _TYPED_RELATION_PATTERNS:
                if pattern.search(between):
                    pair = tuple(sorted([e1, e2]))
                    typed[pair] = rel_type
                    break
    return typed


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


def consolidate(force: bool = False) -> dict:
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
    trace_data = {}
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


def capacity_report() -> dict[str, dict]:
    """Report memory capacity usage per semantic category.

    Returns a dict mapping category name to:
    - chars: total characters in that category
    - limit: configured char limit
    - usage_pct: percentage used (0-100+)
    - needs_consolidation: True if above CAPACITY_WARN_PERCENT
    - file_count: number of .md files
    - state_counts: breakdown by validity_state
    """
    report: dict[str, dict] = {}
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
DAG_FANOUT = 4  # number of children per internal node


@dataclass
class DAGNode:
    """A node in the hierarchical summary DAG."""

    node_id: str
    level: int  # 0 = leaf (raw trace), 1+ = summary
    summary: str
    children: list[str]  # child node_ids or trace filenames
    date_range: tuple[str, str]  # (earliest, latest) ISO dates
    entities: list[str]
    project: str


def build_summary_dag(trace_data: dict[str, dict]) -> list[dict]:
    """Build a hierarchical summary DAG from trace data.

    Args:
        trace_data: dict mapping trace filename to metadata dict with
            keys: date, project, entities, key_lines, text

    Returns:
        List of DAGNode dicts (serialisable).
    """
    if not trace_data:
        return []

    # Level 0: leaf nodes from individual traces
    leaves: list[DAGNode] = []
    for name, data in sorted(trace_data.items(), key=lambda x: x[1].get("date", "")):
        summary_lines = data.get("key_lines", [])[:5]
        summary = " ".join(summary_lines) if summary_lines else data.get("text", "")[:200]
        date_str = data.get("date", "")[:10]
        leaves.append(
            DAGNode(
                node_id=f"L0_{name}",
                level=0,
                summary=summary,
                children=[name],
                date_range=(date_str, date_str),
                entities=data.get("entities", [])[:20],
                project=data.get("project", "general"),
            )
        )

    all_nodes = list(leaves)
    current_level_nodes = leaves
    level = 1

    # Build higher levels by grouping DAG_FANOUT nodes together
    while len(current_level_nodes) > 1:
        next_level: list[DAGNode] = []
        for i in range(0, len(current_level_nodes), DAG_FANOUT):
            group = current_level_nodes[i : i + DAG_FANOUT]
            if not group:  # pragma: no cover — range() never yields empty slice
                break

            # Merge summaries: take first sentence from each child
            merged_summary_parts = []
            all_entities: set[str] = set()
            earliest = "9999"
            latest = "0000"
            children_ids = []
            projects: list[str] = []

            for node in group:
                # Take first 100 chars of each child summary
                merged_summary_parts.append(node.summary[:100])
                all_entities.update(node.entities)
                if node.date_range[0] and node.date_range[0] < earliest:
                    earliest = node.date_range[0]
                if node.date_range[1] and node.date_range[1] > latest:
                    latest = node.date_range[1]
                children_ids.append(node.node_id)
                projects.append(node.project)

            # Most common project in group
            project = max(set(projects), key=projects.count) if projects else "general"

            merged_summary = " | ".join(merged_summary_parts)
            node_id = f"L{level}_{i // DAG_FANOUT}_{earliest}"

            parent = DAGNode(
                node_id=node_id,
                level=level,
                summary=merged_summary,
                children=children_ids,
                date_range=(
                    earliest if earliest != "9999" else "",
                    latest if latest != "0000" else "",
                ),
                entities=sorted(all_entities)[:30],
                project=project,
            )
            next_level.append(parent)

        all_nodes.extend(next_level)
        current_level_nodes = next_level
        level += 1

    return [_dag_node_to_dict(n) for n in all_nodes]


def _dag_node_to_dict(node: DAGNode) -> dict:
    return {
        "node_id": node.node_id,
        "level": node.level,
        "summary": node.summary,
        "children": node.children,
        "date_range": list(node.date_range),
        "entities": node.entities,
        "project": node.project,
    }


def search_summary_dag(
    dag_nodes: list[dict],
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Search the summary DAG top-down for relevant nodes.

    Starts at the highest level, finds matching nodes, then
    drills into their children for more detail.
    """
    if not dag_nodes:
        return []

    query_tokens = set(re.findall(r"\w{3,}", query.lower()))
    if not query_tokens:
        return []

    # Group nodes by level
    by_level: dict[int, list[dict]] = {}
    node_map: dict[str, dict] = {}
    for n in dag_nodes:
        level = n["level"]
        by_level.setdefault(level, []).append(n)
        node_map[n["node_id"]] = n

    max_level = max(by_level.keys()) if by_level else 0

    # Score at highest level first
    def _score(node: dict) -> float:
        text = (node["summary"] + " " + " ".join(node["entities"])).lower()
        text_tokens = set(re.findall(r"\w{3,}", text))
        overlap = len(query_tokens & text_tokens)
        return overlap

    # Top-down search: start from root, expand best matches
    candidates = by_level.get(max_level, [])
    scored = [(n, _score(n)) for n in candidates]
    scored.sort(key=lambda x: -x[1])

    results: list[dict] = []
    seen: set[str] = set()

    # Expand top matches down to leaves
    frontier = [n for n, s in scored[:top_k] if s > 0]
    while frontier:
        node = frontier.pop(0)
        if node["node_id"] in seen:  # pragma: no cover — dedup guard for overlapping DAG paths
            continue
        seen.add(node["node_id"])

        if node["level"] == 0:
            results.append(node)
        else:
            # Expand children, prioritise by score
            children = [node_map[cid] for cid in node["children"] if cid in node_map]
            children_scored = [(c, _score(c)) for c in children]
            children_scored.sort(key=lambda x: -x[1])
            for c, s in children_scored:
                if s > 0 and c["node_id"] not in seen:
                    frontier.append(c)

        if len(results) >= top_k:
            break

    return results[:top_k]


def age_memories(reference_date: str | None = None) -> dict:
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


def _parse_frontmatter(text: str) -> dict | None:
    """Parse YAML-ish frontmatter from a semantic memory file."""
    try:
        from remanentia_consolidation import parse_frontmatter as _rust_fm

        result = _rust_fm(text)  # pragma: no cover
        return result if result else None  # pragma: no cover
    except ImportError:
        pass
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end < 0:
        return None
    fm_text = text[3:end].strip()
    result = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("-"):
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _update_frontmatter_field(path: Path, text: str, field_name: str, value: str):
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
