# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for consolidation_engine.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from consolidation_engine import (
    _cluster_traces,
    _extract_entities,
    _extract_key_lines,
    _extract_metadata,
    _extract_paragraphs,
    _trace_hash,
    _update_graph,
    _write_semantic_memory,
    compute_novelty,
    consolidate,
    get_pending_traces,
)


# ── Metadata extraction ──────────────────────────────────────────


class TestExtractMetadata:
    def test_date_from_filename(self):
        meta = _extract_metadata("2026-03-15_decision_stdp.md", "")
        assert meta["date"] == "2026-03-15"

    def test_date_with_time(self):
        meta = _extract_metadata("2026-03-15T0415_trace.md", "")
        assert meta["date"] == "2026-03-15T0415"

    def test_project_detection(self):
        meta = _extract_metadata("director-ai_audit.md", "")
        assert meta["project"] == "director-ai"

    def test_project_neurocore(self):
        meta = _extract_metadata("neurocore_migration.md", "")
        assert meta["project"] == "sc-neurocore"

    def test_project_remanentia(self):
        meta = _extract_metadata("remanentia_index.md", "")
        assert meta["project"] == "remanentia"

    def test_project_general_fallback(self):
        meta = _extract_metadata("random_notes.md", "")
        assert meta["project"] == "general"

    def test_type_decision(self):
        meta = _extract_metadata("decision_stdp.md", "")
        assert meta["type"] == "decision"

    def test_type_finding(self):
        meta = _extract_metadata("breakthrough_results.md", "")
        assert meta["type"] == "finding"

    def test_type_strategy(self):
        meta = _extract_metadata("revenue_plan.md", "")
        assert meta["type"] == "strategy"

    def test_type_technical(self):
        meta = _extract_metadata("daemon_fix.md", "")
        assert meta["type"] == "technical"

    def test_type_general_fallback(self):
        meta = _extract_metadata("random_notes.md", "")
        assert meta["type"] == "general"


# ── Entity extraction ────────────────────────────────────────────


class TestExtractEntities:
    def test_project_detection(self):
        entities = _extract_entities("The director-ai pipeline scored 78%.")
        assert "director-ai" in entities

    def test_concept_detection(self):
        entities = _extract_entities("We used STDP learning rule with PyTorch.")
        assert "stdp" in entities
        assert "pytorch" in entities

    def test_version_numbers(self):
        entities = _extract_entities("Released v3.9.0 and v0.2.0.")
        assert "v3.9.0" in entities
        assert "v0.2.0" in entities

    def test_percentages(self):
        entities = _extract_entities("Accuracy reached 66.4% on LOCOMO.")
        assert "66.4%" in entities

    def test_file_paths(self):
        entities = _extract_entities("Fixed the bug in snn_backend.py.")
        assert "snn_backend.py" in entities

    def test_camelcase_names(self):
        entities = _extract_entities("The ArcaneNeuron class handles encoding.")
        assert "ArcaneNeuron" in entities

    def test_snake_case_names(self):
        entities = _extract_entities("The compute_order_parameter function is key.")
        assert "compute_order_parameter" in entities

    def test_empty_text(self):
        assert _extract_entities("") == []


# ── Key line extraction ──────────────────────────────────────────


class TestExtractKeyLines:
    def test_decision_trigger(self):
        text = "# Notes\n\n- We decided to remove SNN scoring\n- Other stuff"
        lines = _extract_key_lines(text)
        assert any("decided" in l.lower() for l in lines)

    def test_finding_trigger(self):
        text = "## Results\n\nWe found that BM25 outperforms TF-IDF by 20%."
        lines = _extract_key_lines(text)
        assert any("found" in l.lower() for l in lines)

    def test_metric_trigger(self):
        text = "P@1 accuracy improved from 85.7% to 100% on internal benchmark."
        lines = _extract_key_lines(text)
        assert len(lines) > 0

    def test_context_capture(self):
        text = "We decided to remove SNN.\nThe reason was zero signal.\nAcross 70 experiments."
        lines = _extract_key_lines(text)
        # Should capture the trigger line + context
        assert any("zero signal" in l or "70 experiments" in l for l in lines)

    def test_skip_headers(self):
        text = "# Decision\n\nActual content here with a decision made."
        lines = _extract_key_lines(text)
        assert not any(l.startswith("#") for l in lines)

    def test_cap_at_30(self):
        text = "\n".join(f"We decided thing {i} is important." for i in range(50))
        lines = _extract_key_lines(text)
        assert len(lines) <= 30


# ── Paragraph extraction ─────────────────────────────────────────


class TestExtractParagraphs:
    def test_splits_on_blank_lines(self):
        text = "First block with enough content here for splitting.\n\nSecond block also has enough content for splitting."
        paras = _extract_paragraphs(text)
        assert len(paras) == 2

    def test_filters_short(self):
        text = "Short\n\nThis paragraph has more than enough content to be indexed properly by the engine."
        paras = _extract_paragraphs(text)
        assert len(paras) == 1

    def test_strips_pure_headers(self):
        text = "# Just a header\n\n# Header\nWith content following the header line."
        paras = _extract_paragraphs(text)
        assert len(paras) == 1
        assert "With content" in paras[0]


# ── Trace hash ───────────────────────────────────────────────────


class TestTraceHash:
    def test_deterministic(self):
        h1 = _trace_hash("2026-03-15_decision.md")
        h2 = _trace_hash("2026-03-15_decision.md")
        assert h1 == h2

    def test_different_inputs(self):
        h1 = _trace_hash("trace_a.md")
        h2 = _trace_hash("trace_b.md")
        assert h1 != h2

    def test_length(self):
        h = _trace_hash("test.md")
        assert len(h) == 12


# ── Clustering ───────────────────────────────────────────────────


class TestClusterTraces:
    def test_same_project_same_date(self):
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-15"},
            "b.md": {"project": "remanentia", "date": "2026-03-15"},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a.md", "b.md"}

    def test_different_projects(self):
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-15"},
            "b.md": {"project": "director-ai", "date": "2026-03-15"},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 2

    def test_date_gap_splits(self):
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-10"},
            "b.md": {"project": "remanentia", "date": "2026-03-15"},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 2

    def test_within_2_day_window(self):
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-14"},
            "b.md": {"project": "remanentia", "date": "2026-03-15"},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 1


# ── Novelty computation ──────────────────────────────────────────


class TestComputeNovelty:
    def test_first_pattern_max_novelty(self):
        # Reset global state
        import consolidation_engine
        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0

        pattern = np.array([1.0, 0.0, 0.0, 1.0])
        novelty = compute_novelty(pattern)
        assert novelty == 1.0

    def test_identical_pattern_low_novelty(self):
        import consolidation_engine
        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0

        pattern = np.array([1.0, 0.0, 1.0, 0.0])
        compute_novelty(pattern)  # first
        novelty = compute_novelty(pattern)  # same pattern
        assert novelty < 0.1

    def test_orthogonal_pattern_high_novelty(self):
        import consolidation_engine
        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0

        p1 = np.array([1.0, 0.0, 0.0, 0.0])
        compute_novelty(p1)
        p2 = np.array([0.0, 0.0, 0.0, 1.0])
        novelty = compute_novelty(p2)
        assert novelty > 0.5


# ── Semantic memory writing ──────────────────────────────────────


class TestWriteSemanticMemory:
    def test_writes_file(self, tmp_path):
        with patch("consolidation_engine.SEMANTIC_DIR", tmp_path / "semantic"):
            path = _write_semantic_memory(
                category="decision",
                topic="test-decision",
                date="2026-03-15",
                project="remanentia",
                source_traces=["trace1.md"],
                entities=["stdp", "bm25"],
                content="# Test\n\nSome content.",
            )
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "---" in text
        assert "decision" in text
        assert "remanentia" in text
        assert "Some content" in text


# ── Entity graph update ──────────────────────────────────────────


class TestUpdateGraph:
    def test_creates_entities(self, tmp_path):
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with patch("consolidation_engine.GRAPH_DIR", graph_dir), \
             patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"), \
             patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"):
            _update_graph("trace1.md", ["stdp", "bm25"], "remanentia", "2026-03-15")

        entities_text = (graph_dir / "entities.jsonl").read_text(encoding="utf-8")
        assert "stdp" in entities_text
        assert "bm25" in entities_text

    def test_creates_relations(self, tmp_path):
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with patch("consolidation_engine.GRAPH_DIR", graph_dir), \
             patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"), \
             patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"):
            _update_graph("trace1.md", ["stdp", "bm25", "embedding"], "remanentia", "2026-03-15")

        rels_text = (graph_dir / "relations.jsonl").read_text(encoding="utf-8")
        rels = [json.loads(l) for l in rels_text.strip().split("\n") if l.strip()]
        # 3 entities → 3 pairs
        assert len(rels) == 3

    def test_increments_weight(self, tmp_path):
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with patch("consolidation_engine.GRAPH_DIR", graph_dir), \
             patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"), \
             patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"):
            _update_graph("trace1.md", ["stdp", "bm25"], "remanentia", "2026-03-15")
            _update_graph("trace2.md", ["stdp", "bm25"], "remanentia", "2026-03-16")

        rels_text = (graph_dir / "relations.jsonl").read_text(encoding="utf-8")
        rel = json.loads(rels_text.strip().split("\n")[0])
        assert rel["weight"] == 2
        assert len(rel["evidence"]) == 2


# ── Full consolidation pipeline ──────────────────────────────────


class TestConsolidate:
    def test_full_pipeline(self, tmp_traces, tmp_path):
        consol_dir = tmp_path / "consolidation"
        consol_dir.mkdir()
        semantic_dir = tmp_path / "memory" / "semantic"
        graph_dir = tmp_path / "memory" / "graph"

        with patch("consolidation_engine.TRACES_DIR", tmp_traces), \
             patch("consolidation_engine.CONSOLIDATION_DIR", consol_dir), \
             patch("consolidation_engine.PENDING_PATH", consol_dir / "pending.json"), \
             patch("consolidation_engine.LAST_RUN_PATH", consol_dir / "last_consolidation.json"), \
             patch("consolidation_engine.SEMANTIC_DIR", semantic_dir), \
             patch("consolidation_engine.GRAPH_DIR", graph_dir), \
             patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"), \
             patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"), \
             patch("consolidation_engine.CLUSTERS_PATH", graph_dir / "trace_clusters.json"):

            result = consolidate(force=True)

        assert result["traces_processed"] == 3
        assert result["clusters_formed"] > 0
        assert result["memories_written"] > 0
        assert result["entities_found"] > 0
        assert semantic_dir.exists()

    def test_nothing_to_consolidate(self, tmp_path):
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        consol_dir = tmp_path / "consolidation"

        with patch("consolidation_engine.TRACES_DIR", traces_dir), \
             patch("consolidation_engine.CONSOLIDATION_DIR", consol_dir), \
             patch("consolidation_engine.PENDING_PATH", consol_dir / "pending.json"):
            result = consolidate(force=False)

        assert result["status"] == "nothing_to_consolidate"

    def test_pending_tracks_processed(self, tmp_traces, tmp_path):
        consol_dir = tmp_path / "consolidation"
        consol_dir.mkdir()
        semantic_dir = tmp_path / "memory" / "semantic"
        graph_dir = tmp_path / "memory" / "graph"

        with patch("consolidation_engine.TRACES_DIR", tmp_traces), \
             patch("consolidation_engine.CONSOLIDATION_DIR", consol_dir), \
             patch("consolidation_engine.PENDING_PATH", consol_dir / "pending.json"), \
             patch("consolidation_engine.LAST_RUN_PATH", consol_dir / "last_consolidation.json"), \
             patch("consolidation_engine.SEMANTIC_DIR", semantic_dir), \
             patch("consolidation_engine.GRAPH_DIR", graph_dir), \
             patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"), \
             patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"), \
             patch("consolidation_engine.CLUSTERS_PATH", graph_dir / "trace_clusters.json"):

            consolidate(force=True)
            # Second run should find nothing pending
            result = consolidate(force=False)

        assert result["status"] == "nothing_to_consolidate"


# ── Edge cases ──────────────────────────────────────────────────


class TestConsolidationEdge:
    def test_zero_norm_novelty(self):
        import consolidation_engine
        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0
        pattern = np.array([1.0, 0.0, 1.0])
        compute_novelty(pattern)
        zero = np.array([0.0, 0.0, 0.0])
        novelty = compute_novelty(zero)
        assert novelty == 1.0

    def test_get_pending_traces_empty(self, tmp_path):
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        consol_dir = tmp_path / "consolidation"
        with patch("consolidation_engine.TRACES_DIR", traces_dir), \
             patch("consolidation_engine.CONSOLIDATION_DIR", consol_dir), \
             patch("consolidation_engine.PENDING_PATH", consol_dir / "pending.json"):
            pending = get_pending_traces()
        assert pending == []

    def test_cluster_missing_date(self):
        traces = {
            "a.md": {"project": "test", "date": ""},
            "b.md": {"project": "test", "date": ""},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 1

    def test_extract_paragraphs_empty(self):
        assert _extract_paragraphs("") == []

    def test_extract_entities_short_file(self):
        entities = _extract_entities("The a.py file is short.")
        file_names = [e for e in entities if e.endswith(".py")]
        assert not any(len(e) <= 3 for e in file_names)
