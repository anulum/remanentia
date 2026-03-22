# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Test fixtures

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the parent directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_traces(tmp_path):
    """Create a temporary traces directory with sample traces."""
    traces = tmp_path / "reasoning_traces"
    traces.mkdir()

    (traces / "2026-03-15_decision_stdp_removal.md").write_text(
        "# Decision: Remove SNN from retrieval\n\n"
        "## Context\n\n"
        "SNN retrieval adds zero signal across 70+ experiments.\n"
        "Weight was 0.30, now set to 0.00.\n\n"
        "## Decision\n\n"
        "We decided to remove SNN scoring from the retrieval pipeline.\n"
        "The embedding weight increased from 0.25 to 0.45.\n\n"
        "## Result\n\n"
        "P@1 improved from 85.7% to 100% on internal benchmark (14 queries).\n",
        encoding="utf-8",
    )

    (traces / "2026-03-17_finding_locomo_benchmark.md").write_text(
        "# LOCOMO Benchmark Results\n\n"
        "## Summary\n\n"
        "External benchmark on 1,986 questions from LOCOMO dataset.\n"
        "BM25-only: 48.9%. With embedding: 50.0%. With query intelligence: 66.4%.\n\n"
        "## Breakdown\n\n"
        "- Multi-hop: 75.4%\n"
        "- Adversarial: 73.6%\n"
        "- Open: 72.4%\n"
        "- Single-hop: 42.6%\n"
        "- Temporal: 15.6%\n\n"
        "## Finding\n\n"
        "The gap is answer extraction, not retrieval.\n"
        "Hindsight (SOTA with LLM): 91.4%.\n",
        encoding="utf-8",
    )

    (traces / "2026-03-10_technical_rust_bm25.md").write_text(
        "# Rust BM25 Engine\n\n"
        "Built a Rust BM25 engine with PyO3 + Rayon parallel scoring.\n"
        "Compiled and installed in .venv312.\n\n"
        "## Finding\n\n"
        "Python is faster at 10K paragraphs (FFI overhead).\n"
        "Rust would win at 100K+ with an inverted index.\n"
        "Built for future scale.\n",
        encoding="utf-8",
    )
    return traces


@pytest.fixture
def tmp_semantic(tmp_path):
    """Create a temporary semantic memories directory."""
    sem = tmp_path / "memory" / "semantic" / "decision"
    sem.mkdir(parents=True)

    (sem / "2026-03-15_remanentia-decision.md").write_text(
        "---\n"
        "type: decision\n"
        "date: 2026-03-15\n"
        "project: remanentia\n"
        "source_traces:\n"
        "  - 2026-03-15_decision_stdp_removal.md\n"
        "entities:\n"
        "  - stdp\n"
        "  - bm25\n"
        "  - embedding\n"
        "confidence: 0.8\n"
        "last_validated: 2026-03-15\n"
        "---\n\n"
        "# remanentia — decision (2026-03-15)\n\n"
        "- Removed SNN from retrieval scoring\n"
        "- Embedding weight increased to 0.45\n",
        encoding="utf-8",
    )
    return tmp_path / "memory" / "semantic"


@pytest.fixture
def tmp_graph(tmp_path):
    """Create a temporary entity graph."""
    graph = tmp_path / "memory" / "graph"
    graph.mkdir(parents=True)

    entities = [
        {"id": "stdp", "type": "concept", "label": "STDP", "first_seen": "2026-03-10", "trace_count": 5},
        {"id": "bm25", "type": "concept", "label": "BM25", "first_seen": "2026-03-10", "trace_count": 3},
        {"id": "remanentia", "type": "project", "label": "remanentia", "first_seen": "2026-03-10", "trace_count": 8},
        {"id": "embedding", "type": "concept", "label": "embedding", "first_seen": "2026-03-12", "trace_count": 4},
    ]
    (graph / "entities.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entities) + "\n",
        encoding="utf-8",
    )

    relations = [
        {"source": "bm25", "target": "remanentia", "type": "used_in", "weight": 5, "evidence": ["trace1"]},
        {"source": "embedding", "target": "remanentia", "type": "used_in", "weight": 4, "evidence": ["trace2"]},
        {"source": "stdp", "target": "remanentia", "type": "co_occurs", "weight": 6, "evidence": ["trace1", "trace3"]},
        {"source": "bm25", "target": "embedding", "type": "co_occurs", "weight": 3, "evidence": ["trace2"]},
    ]
    (graph / "relations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in relations) + "\n",
        encoding="utf-8",
    )
    return graph


@pytest.fixture
def tmp_consolidation(tmp_path):
    """Create a temporary consolidation directory."""
    cdir = tmp_path / "consolidation"
    cdir.mkdir()
    return cdir


@pytest.fixture
def sample_code_text():
    """Sample Python code for code splitting tests."""
    return '''"""Module docstring for testing."""

import numpy as np


def compute_score(query, doc):
    """Compute BM25 score between query and document."""
    tokens = query.lower().split()
    score = 0.0
    for t in tokens:
        if t in doc.lower():
            score += 1.0
    return score


class SearchEngine:
    """Simple search engine."""

    def __init__(self):
        self.index = {}

    def add(self, name, text):
        self.index[name] = text

    def search(self, query):
        results = []
        for name, text in self.index.items():
            if query.lower() in text.lower():
                results.append(name)
        return results
'''
