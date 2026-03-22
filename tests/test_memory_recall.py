# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for memory_recall.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_recall import (
    MemoryContext,
    _assess_novelty,
    _entities_for_query,
    _find_related,
    _search_semantic,
    _temporal_context,
)


# ── MemoryContext ────────────────────────────────────────────────


class TestMemoryContext:
    def test_summary_basic(self):
        ctx = MemoryContext(
            query="test query",
            trace="trace.md",
            trace_score=0.85,
            trace_snippet="Some snippet text",
            elapsed_ms=42.0,
            sources_consulted=3,
        )
        s = ctx.summary
        assert "test query" in s
        assert "trace.md" in s
        assert "0.850" in s
        assert "42ms" in s

    def test_summary_with_entities(self):
        ctx = MemoryContext(
            query="test",
            entities=["stdp", "bm25"],
            related_entities=[
                {"entity": "embedding", "weight": 4, "relation": "co_occurs"},
            ],
        )
        s = ctx.summary
        assert "stdp" in s
        assert "bm25" in s
        assert "embedding" in s

    def test_summary_with_timeline(self):
        ctx = MemoryContext(
            query="test",
            before=["trace1.md", "trace2.md"],
            after=["trace3.md"],
        )
        s = ctx.summary
        assert "2 before" in s
        assert "1 after" in s

    def test_to_llm_context_trace(self):
        ctx = MemoryContext(
            query="test",
            trace="trace.md",
            trace_snippet="Important finding here.",
        )
        llm = ctx.to_llm_context()
        assert "trace.md" in llm
        assert "Important finding" in llm

    def test_to_llm_context_semantic(self):
        ctx = MemoryContext(
            query="test",
            semantic_memories=[
                {"path": "memory/semantic/decision.md", "content": "Decision content"},
            ],
        )
        llm = ctx.to_llm_context()
        assert "Consolidated" in llm
        assert "Decision content" in llm

    def test_to_llm_context_related(self):
        ctx = MemoryContext(
            query="test",
            related_entities=[
                {"entity": "bm25", "weight": 5},
                {"entity": "stdp", "weight": 3},
            ],
        )
        llm = ctx.to_llm_context()
        assert "bm25" in llm

    def test_to_llm_context_empty(self):
        ctx = MemoryContext(query="test")
        assert ctx.to_llm_context() == ""


# ── Entity graph queries ─────────────────────────────────────────


class TestFindRelated:
    def test_finds_connected(self):
        relations = [
            {"source": "stdp", "target": "remanentia", "type": "co_occurs", "weight": 6, "evidence": []},
            {"source": "bm25", "target": "remanentia", "type": "used_in", "weight": 5, "evidence": []},
        ]
        result = _find_related("remanentia", relations, top_k=5)
        assert len(result) == 2
        assert result[0]["entity"] == "stdp"  # higher weight
        assert result[1]["entity"] == "bm25"

    def test_bidirectional(self):
        relations = [
            {"source": "a", "target": "b", "type": "co_occurs", "weight": 3, "evidence": []},
        ]
        result_a = _find_related("a", relations)
        result_b = _find_related("b", relations)
        assert len(result_a) == 1 and result_a[0]["entity"] == "b"
        assert len(result_b) == 1 and result_b[0]["entity"] == "a"

    def test_top_k_limit(self):
        relations = [
            {"source": "a", "target": f"b{i}", "type": "co_occurs", "weight": i, "evidence": []}
            for i in range(20)
        ]
        result = _find_related("a", relations, top_k=5)
        assert len(result) == 5

    def test_empty_relations(self):
        assert _find_related("a", [], top_k=5) == []


class TestEntitiesForQuery:
    def test_finds_by_id(self):
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
            "bm25": {"id": "bm25", "label": "BM25"},
        }
        found = _entities_for_query("how does stdp work", entities)
        assert "stdp" in found

    def test_finds_by_label(self):
        entities = {
            "pytorch": {"id": "pytorch", "label": "PyTorch"},
        }
        found = _entities_for_query("using pytorch for training", entities)
        assert "pytorch" in found

    def test_no_match(self):
        entities = {"stdp": {"id": "stdp", "label": "STDP"}}
        found = _entities_for_query("unrelated query", entities)
        assert found == []


# ── Semantic memory search ───────────────────────────────────────


class TestSearchSemantic:
    def test_finds_matching(self, tmp_semantic):
        with patch("memory_recall.SEMANTIC_DIR", tmp_semantic), \
             patch("memory_recall.BASE", tmp_semantic.parent):
            results = _search_semantic("SNN removal decision", top_k=5)
        assert len(results) > 0
        assert any("decision" in r["path"].lower() for r in results)

    def test_no_match(self, tmp_semantic):
        with patch("memory_recall.SEMANTIC_DIR", tmp_semantic), \
             patch("memory_recall.BASE", tmp_semantic.parent):
            results = _search_semantic("xyznonexistent_zzz", top_k=5)
        assert results == []

    def test_no_semantic_dir(self, tmp_path):
        with patch("memory_recall.SEMANTIC_DIR", tmp_path / "nonexistent"):
            results = _search_semantic("anything", top_k=5)
        assert results == []

    def test_parses_frontmatter(self, tmp_semantic):
        with patch("memory_recall.SEMANTIC_DIR", tmp_semantic), \
             patch("memory_recall.BASE", tmp_semantic.parent):
            results = _search_semantic("remanentia decision", top_k=5)
        if results:
            assert results[0]["project"] == "remanentia"
            assert results[0]["type"] == "decision"


# ── Temporal context ─────────────────────────────────────────────


class TestTemporalContext:
    def test_finds_neighbors(self, tmp_traces):
        with patch("memory_recall.TRACES_DIR", tmp_traces):
            before, after = _temporal_context("2026-03-15_decision_stdp_removal.md")
        assert "2026-03-10_technical_rust_bm25.md" in before
        assert "2026-03-17_finding_locomo_benchmark.md" in after

    def test_first_trace_no_before(self, tmp_traces):
        with patch("memory_recall.TRACES_DIR", tmp_traces):
            before, after = _temporal_context("2026-03-10_technical_rust_bm25.md")
        assert before == []
        assert len(after) > 0

    def test_last_trace_no_after(self, tmp_traces):
        with patch("memory_recall.TRACES_DIR", tmp_traces):
            before, after = _temporal_context("2026-03-17_finding_locomo_benchmark.md")
        assert len(before) > 0
        assert after == []

    def test_nonexistent_trace(self, tmp_traces):
        with patch("memory_recall.TRACES_DIR", tmp_traces):
            before, after = _temporal_context("nonexistent.md")
        assert before == []
        assert after == []


# ── Novelty assessment ───────────────────────────────────────────


class TestAssessNovelty:
    def test_known_entities_low_novelty(self):
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
            "bm25": {"id": "bm25", "label": "BM25"},
        }
        novelty = _assess_novelty("stdp bm25 scoring", entities)
        assert novelty < 0.5

    def test_unknown_terms_high_novelty(self):
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
        }
        novelty = _assess_novelty("quantum teleportation decoherence fidelity", entities)
        assert novelty > 0.5

    def test_empty_query(self):
        assert _assess_novelty("", {}) == 0.0

    def test_mixed_known_unknown(self):
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
        }
        novelty = _assess_novelty("stdp quantum fidelity", entities)
        assert 0.0 < novelty < 1.0
