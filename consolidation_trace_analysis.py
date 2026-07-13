# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Consolidation trace analysis

"""Trace metadata, content extraction, and episode clustering."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from importlib import import_module
from typing import Any, TypeAlias, cast

JsonObject: TypeAlias = dict[str, Any]
TraceData: TypeAlias = dict[str, JsonObject]

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
PROJECT_NAMES = frozenset(project for project, _patterns in _PROJECT_PATTERNS)

_TYPE_PATTERNS = [
    ("decision", ["decision", "audit", "architecture"]),
    ("finding", ["breakthrough", "failure", "analysis", "experiment"]),
    ("strategy", ["strategy", "revenue", "sprint", "competitive"]),
    ("technical", ["daemon", "migration", "convergence", "deepening", "fix"]),
    ("continuity", ["continuity", "contribution", "identity"]),
    ("personal", ["personal", "partnership", "moment"]),
]


def extract_metadata(filename: str, text: str) -> dict[str, str]:
    """Infer trace date, project, and type metadata from a trace filename."""

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


def extract_entities(text: str) -> list[str]:
    """Extract entities with the native engine when it is installed."""
    try:
        native_extract = import_module("remanentia_consolidation").extract_entities
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return extract_entities_python(text)
    return cast(list[str], native_extract(text))  # pragma: no cover - native dispatch


def extract_entities_python(text: str) -> list[str]:
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
        entities.add(name)

    # Function/class names (word_word pattern or CamelCase)
    for m in re.finditer(r"\b[a-z][a-z_]+(?:_[a-z]+){2,}\b", text):
        if len(m.group()) > 8:  # pragma: no branch - both outcomes asserted directly
            entities.add(m.group())
    for m in re.finditer(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+){1,}\b", text):
        entities.add(m.group())

    return sorted(entities)


def extract_paragraphs(text: str) -> list[str]:
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


def extract_key_lines(text: str) -> list[str]:
    """Extract key lines with the native engine when it is installed."""
    try:
        native_extract = import_module("remanentia_consolidation").extract_key_lines
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return extract_key_lines_python(text)
    return cast(list[str], native_extract(text))  # pragma: no cover - native dispatch


def extract_key_lines_python(text: str) -> list[str]:
    """Extract decision/finding lines from trace text.

    Expanded trigger set + multi-line capture: when a trigger fires,
    grab the next 2 non-empty lines as context.
    """
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


def trace_hash(filename: str) -> str:
    """Return the stable short content-address key for a trace filename."""

    return hashlib.sha256(filename.encode()).hexdigest()[:12]


# ── Clustering ───────────────────────────────────────────────────


def cluster_traces(traces: TraceData) -> list[list[str]]:
    """Cluster traces with the native engine when it is installed."""
    try:
        native_cluster = import_module("remanentia_consolidation").cluster_traces
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return cluster_traces_python(traces)
    tuples = [  # pragma: no cover - native dispatch
        (name, str(meta["project"]), str(meta.get("date", ""))[:10])
        for name, meta in traces.items()
    ]
    return cast(list[list[str]], native_cluster(tuples))  # pragma: no cover - native dispatch


def cluster_traces_python(traces: TraceData) -> list[list[str]]:
    """Cluster traces by project + date proximity.

    Traces from the same project within 2 days of each other are grouped.
    A gap > 2 days starts a new cluster.
    """
    by_project: defaultdict[str, list[str]] = defaultdict(list)
    for name, meta in traces.items():
        by_project[str(meta["project"])].append(name)

    clusters: list[list[str]] = []
    for names in by_project.values():
        names.sort(key=lambda name: str(traces[name].get("date", "")))
        current_cluster = [names[0]]
        for i in range(1, len(names)):
            prev_date = str(traces[names[i - 1]].get("date", ""))[:10]
            curr_date = str(traces[names[i]].get("date", ""))[:10]
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
        clusters.append(current_cluster)

    return clusters
