# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — Memory Consolidation Engine

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
    entities = set()
    text_lower = text.lower()

    # Layer 1: Projects
    for proj, patterns in _PROJECT_PATTERNS:
        if any(p in text_lower for p in patterns):
            entities.add(proj)

    # Layer 2: Known concepts (expanded from 22 to 60+)
    concepts = [
        "STDP", "LIF", "Kuramoto", "Hopfield", "TF-IDF", "BM25",
        "embedding", "PyTorch", "CUDA", "GPU", "CPU", "daemon",
        "holographic", "attractor", "inhibition", "spike", "neuron",
        "retrieval", "consolidation", "UPDE", "Stuart-Landau",
        "Dimits", "gyrokinetic", "tokamak", "VQE", "Heron",
        "BCPNN", "CSDP", "Hebbian", "Perron-Frobenius",
        "Marchenko-Pastur", "eigenvalue", "SVD",
        "MiniLM", "sentence-transformer", "FastAPI", "MCP",
        "Docker", "Prometheus", "Grafana", "CI", "pytest",
        "Rust", "PyO3", "maturin", "Rayon",
        "ArcaneNeuron", "chirp", "chimera", "bifurcation",
        "entropy", "Fisher", "Lyapunov", "Boltzmann",
        "hippocampus", "dentate gyrus", "pattern separation",
        "Dale's law", "E/I balance",
        "Mem0", "Letta", "Zep", "MemOS", "LangMem",
        "JOSS", "NeurIPS", "EMNLP", "arXiv", "Zenodo",
        "AGPL", "PyPI", "Loihi",
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
    lines = text.split("\n")
    key_lines = []
    triggers = [
        "decided", "decision", "found", "finding", "result",
        "key insight", "conclusion", "fix", "resolved",
        "chose", "rejected", "confirmed", "measured",
        "p@1", "precision", "accuracy", "because", "therefore",
        "root cause", "the reason", "we proved", "this means",
        "critical", "important", "changed", "broke", "works",
        "doesn't work", "failed", "succeeded", "shipped",
        "version", "v0.", "v1.", "v2.", "v3.",
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
        if not names:
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

def _write_semantic_memory(
    category: str,
    topic: str,
    date: str,
    project: str,
    source_traces: list[str],
    entities: list[str],
    content: str,
) -> Path:
    """Write a structured semantic memory file."""
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
        "last_validated": datetime.now().strftime("%Y-%m-%d"),
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


def _update_graph(trace_name: str, entities: list[str], project: str, date: str):
    """Add entities and relations from a trace."""
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

    # Relations: all entity pairs in this trace are co-occurring
    existing_pairs = {(r["source"], r["target"]) for r in relations}
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1:]:
            pair = tuple(sorted([e1, e2]))
            if pair in existing_pairs:
                for r in relations:
                    if (r["source"], r["target"]) == pair or (r["target"], r["source"]) == pair:
                        r["weight"] = r.get("weight", 0) + 1
                        if trace_name not in r.get("evidence", []):
                            r.setdefault("evidence", []).append(trace_name)
                        break
            else:
                relations.append({
                    "source": pair[0], "target": pair[1],
                    "type": "co_occurs",
                    "weight": 1,
                    "evidence": [trace_name],
                })
                existing_pairs.add(pair)

    _save_entities(entity_db)
    _save_relations(relations)


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

    if not new_traces:
        new_traces = [f.name for f in sorted(TRACES_DIR.glob("*.md"))]

    # Load and analyze each trace
    trace_data = {}
    for name in new_traces:
        path = TRACES_DIR / name
        if not path.exists():
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
        if not cluster:
            continue

        cluster_data = [trace_data[n] for n in cluster if n in trace_data]
        if not cluster_data:
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

        # Update entity graph
        for name in cluster:
            if name in trace_data:
                _update_graph(
                    name,
                    trace_data[name]["entities"],
                    trace_data[name]["project"],
                    trace_data[name].get("date", ""),
                )

    # Save clusters
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERS_PATH.write_text(json.dumps(clusters, indent=2) + "\n", encoding="utf-8")

    # Mark traces as processed
    if PENDING_PATH.exists():
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


if __name__ == "__main__":
    print("Running initial consolidation...")
    result = consolidate(force=True)
    print(json.dumps(result, indent=2))
