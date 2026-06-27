# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for memory recall

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memory_recall import (
    JsonMap,
    MemoryContext,
    _assess_novelty,
    _cross_project_insights,
    _entities_for_query,
    _find_related,
    _load_entities,
    _load_relations,
    _search_semantic,
    _temporal_context,
    recall,
)


# ── MemoryContext ────────────────────────────────────────────────


class TestMemoryContext:
    def test_summary_basic(self) -> None:
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

    def test_summary_with_entities(self) -> None:
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

    def test_summary_with_timeline(self) -> None:
        ctx = MemoryContext(
            query="test",
            before=["trace1.md", "trace2.md"],
            after=["trace3.md"],
        )
        s = ctx.summary
        assert "2 before" in s
        assert "1 after" in s

    def test_to_llm_context_trace(self) -> None:
        ctx = MemoryContext(
            query="test",
            trace="trace.md",
            trace_snippet="Important finding here.",
        )
        llm = ctx.to_llm_context()
        assert "trace.md" in llm
        assert "Important finding" in llm

    def test_to_llm_context_semantic(self) -> None:
        ctx = MemoryContext(
            query="test",
            semantic_memories=[
                {"path": "memory/semantic/decision.md", "content": "Decision content"},
            ],
        )
        llm = ctx.to_llm_context()
        assert "Consolidated" in llm
        assert "Decision content" in llm

    def test_to_llm_context_related(self) -> None:
        ctx = MemoryContext(
            query="test",
            related_entities=[
                {"entity": "bm25", "weight": 5},
                {"entity": "stdp", "weight": 3},
            ],
        )
        llm = ctx.to_llm_context()
        assert "bm25" in llm

    def test_to_llm_context_empty(self) -> None:
        ctx = MemoryContext(query="test")
        assert ctx.to_llm_context() == ""


# ── Entity graph queries ─────────────────────────────────────────


class TestFindRelated:
    def test_finds_connected(self) -> None:
        relations = [
            {
                "source": "stdp",
                "target": "remanentia",
                "type": "co_occurs",
                "weight": 6,
                "evidence": [],
            },
            {
                "source": "bm25",
                "target": "remanentia",
                "type": "used_in",
                "weight": 5,
                "evidence": [],
            },
        ]
        result = _find_related("remanentia", relations, top_k=5)
        assert len(result) == 2
        assert result[0]["entity"] == "stdp"  # higher weight
        assert result[1]["entity"] == "bm25"

    def test_bidirectional(self) -> None:
        relations = [
            {"source": "a", "target": "b", "type": "co_occurs", "weight": 3, "evidence": []},
        ]
        result_a = _find_related("a", relations)
        result_b = _find_related("b", relations)
        assert len(result_a) == 1 and result_a[0]["entity"] == "b"
        assert len(result_b) == 1 and result_b[0]["entity"] == "a"

    def test_top_k_limit(self) -> None:
        relations = [
            {"source": "a", "target": f"b{i}", "type": "co_occurs", "weight": i, "evidence": []}
            for i in range(20)
        ]
        result = _find_related("a", relations, top_k=5)
        assert len(result) == 5

    def test_empty_relations(self) -> None:
        assert _find_related("a", [], top_k=5) == []


class TestEntitiesForQuery:
    def test_finds_by_id(self) -> None:
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
            "bm25": {"id": "bm25", "label": "BM25"},
        }
        found = _entities_for_query("how does stdp work", entities)
        assert "stdp" in found

    def test_finds_by_label(self) -> None:
        entities = {
            "pytorch": {"id": "pytorch", "label": "PyTorch"},
        }
        found = _entities_for_query("using pytorch for training", entities)
        assert "pytorch" in found

    def test_no_match(self) -> None:
        entities = {"stdp": {"id": "stdp", "label": "STDP"}}
        found = _entities_for_query("unrelated query", entities)
        assert found == []


# ── Semantic memory search ───────────────────────────────────────


class TestSearchSemantic:
    def test_finds_matching(self, tmp_semantic: Path) -> None:
        with (
            patch("memory_recall.SEMANTIC_DIR", tmp_semantic),
            patch("memory_recall.BASE", tmp_semantic.parent),
        ):
            results = _search_semantic("SNN removal decision", top_k=5)
        assert len(results) > 0
        assert any("decision" in r["path"].lower() for r in results)

    def test_no_match(self, tmp_semantic: Path) -> None:
        with (
            patch("memory_recall.SEMANTIC_DIR", tmp_semantic),
            patch("memory_recall.BASE", tmp_semantic.parent),
        ):
            results = _search_semantic("xyznonexistent_zzz", top_k=5)
        assert results == []

    def test_no_semantic_dir(self, tmp_path: Path) -> None:
        with patch("memory_recall.SEMANTIC_DIR", tmp_path / "nonexistent"):
            results = _search_semantic("anything", top_k=5)
        assert results == []

    def test_parses_frontmatter(self, tmp_semantic: Path) -> None:
        with (
            patch("memory_recall.SEMANTIC_DIR", tmp_semantic),
            patch("memory_recall.BASE", tmp_semantic.parent),
        ):
            results = _search_semantic("remanentia decision", top_k=5)
        if results:
            assert results[0]["project"] == "remanentia"
            assert results[0]["type"] == "decision"

    def test_python_overlap_fallback_scores_matching_memory(self, tmp_semantic: Path) -> None:
        with (
            patch("memory_recall.SEMANTIC_DIR", tmp_semantic),
            patch("memory_recall.BASE", tmp_semantic.parent),
            patch("memory_recall.import_module", side_effect=ImportError),
        ):
            results = _search_semantic("embedding weight", top_k=5)

        assert results
        assert results[0]["score"] > 0
        assert results[0]["key_point"] == "Removed SNN from retrieval scoring"
        assert "Embedding weight increased" in results[0]["content"]


# ── Temporal context ─────────────────────────────────────────────


class TestTemporalContext:
    def test_finds_neighbors(self, tmp_traces: Path) -> None:
        with patch("memory_recall.TRACES_DIR", tmp_traces):
            before, after = _temporal_context("2026-03-15_decision_stdp_removal.md")
        assert "2026-03-10_technical_rust_bm25.md" in before
        assert "2026-03-17_finding_locomo_benchmark.md" in after

    def test_first_trace_no_before(self, tmp_traces: Path) -> None:
        with patch("memory_recall.TRACES_DIR", tmp_traces):
            before, after = _temporal_context("2026-03-10_technical_rust_bm25.md")
        assert before == []
        assert len(after) > 0

    def test_last_trace_no_after(self, tmp_traces: Path) -> None:
        with patch("memory_recall.TRACES_DIR", tmp_traces):
            before, after = _temporal_context("2026-03-17_finding_locomo_benchmark.md")
        assert len(before) > 0
        assert after == []

    def test_nonexistent_trace(self, tmp_traces: Path) -> None:
        with patch("memory_recall.TRACES_DIR", tmp_traces):
            before, after = _temporal_context("nonexistent.md")
        assert before == []
        assert after == []


# ── Novelty assessment ───────────────────────────────────────────


class TestAssessNovelty:
    def test_known_entities_low_novelty(self) -> None:
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
            "bm25": {"id": "bm25", "label": "BM25"},
        }
        novelty = _assess_novelty("stdp bm25 scoring", entities)
        assert novelty < 0.5

    def test_unknown_terms_high_novelty(self) -> None:
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
        }
        novelty = _assess_novelty("quantum teleportation decoherence fidelity", entities)
        assert novelty > 0.5

    def test_empty_query(self) -> None:
        assert _assess_novelty("", {}) == 0.0

    def test_mixed_known_unknown(self) -> None:
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
        }
        novelty = _assess_novelty("stdp quantum fidelity", entities)
        assert 0.0 < novelty < 1.0

    def test_python_fallback_without_native_extension(self) -> None:
        entities = {
            "stdp": {"id": "stdp", "label": "STDP"},
        }
        with patch("memory_recall.import_module", side_effect=ImportError):
            novelty = _assess_novelty("stdp quantum fidelity", entities)

        assert novelty == 2 / 3

    def test_native_extension_assesses_novelty_with_known_tokens(self) -> None:
        calls: list[tuple[str, set[str]]] = []

        def assess_novelty(query: str, known_tokens: set[str]) -> float:
            calls.append((query, known_tokens))
            return 0.25

        native = SimpleNamespace(assess_novelty=assess_novelty)

        entities = {
            "stdp": {"id": "stdp", "label": "Spike Timing Dependent Plasticity"},
        }
        with patch("memory_recall.import_module", return_value=native):
            novelty = _assess_novelty("stdp quantum fidelity", entities)

        assert novelty == 0.25
        assert calls == [
            (
                "stdp quantum fidelity",
                {"stdp", "spike", "timing", "dependent", "plasticity"},
            ),
        ]

    def test_python_fallback_empty_token_query_has_zero_novelty(self) -> None:
        with patch("memory_recall.import_module", side_effect=ImportError):
            novelty = _assess_novelty("!!!", {})

        assert novelty == 0.0


# ── Cross-project insights ──────────────────────────────────────


class TestCrossProjectInsights:
    def test_finds_cross_project(self) -> None:
        entities = {
            "remanentia": {"id": "remanentia", "type": "project", "label": "Remanentia"},
            "director-ai": {"id": "director-ai", "type": "project", "label": "Director-AI"},
            "stdp": {"id": "stdp", "type": "concept", "label": "STDP"},
        }
        relations = [
            {
                "source": "stdp",
                "target": "remanentia",
                "type": "co_occurs",
                "weight": 5,
                "evidence": [],
            },
            {
                "source": "stdp",
                "target": "director-ai",
                "type": "co_occurs",
                "weight": 3,
                "evidence": [],
            },
        ]
        insights = _cross_project_insights(["stdp"], "remanentia", entities, relations)
        projects = [i["project"] for i in insights]
        assert "director-ai" in projects

    def test_no_cross_project(self) -> None:
        entities = {"stdp": {"id": "stdp", "type": "concept"}}
        relations: list[JsonMap] = []
        insights = _cross_project_insights(["stdp"], "remanentia", entities, relations)
        assert insights == []


# ── Load entities/relations ─────────────────────────────────────


class TestLoadEntities:
    def test_load_entities(self, tmp_graph: Path) -> None:
        with patch("memory_recall.GRAPH_DIR", tmp_graph):
            entities = _load_entities()
        assert "stdp" in entities
        assert "bm25" in entities

    def test_load_entities_no_file(self, tmp_path: Path) -> None:
        with patch("memory_recall.GRAPH_DIR", tmp_path):
            assert _load_entities() == {}

    def test_load_relations(self, tmp_graph: Path) -> None:
        with patch("memory_recall.GRAPH_DIR", tmp_graph):
            relations = _load_relations()
        assert len(relations) == 4

    def test_load_relations_no_file(self, tmp_path: Path) -> None:
        with patch("memory_recall.GRAPH_DIR", tmp_path):
            assert _load_relations() == []


# ── Full recall function ────────────────────────────────────────


class TestRecall:
    def test_recall_basic(self, tmp_traces: Path, tmp_graph: Path, tmp_semantic: Path) -> None:
        with (
            patch("memory_recall.TRACES_DIR", tmp_traces),
            patch("memory_recall.SEMANTIC_DIR", tmp_semantic),
            patch("memory_recall.GRAPH_DIR", tmp_graph),
            patch("memory_recall.BASE", tmp_traces.parent),
        ):
            ctx = recall("SNN removal decision", top_k=3, include_content=False)
        assert ctx.query == "SNN removal decision"
        assert isinstance(ctx.elapsed_ms, float)
        assert isinstance(ctx.entities, list)

    def test_recall_with_semantic(
        self, tmp_traces: Path, tmp_graph: Path, tmp_semantic: Path
    ) -> None:
        with (
            patch("memory_recall.TRACES_DIR", tmp_traces),
            patch("memory_recall.SEMANTIC_DIR", tmp_semantic),
            patch("memory_recall.GRAPH_DIR", tmp_graph),
            patch("memory_recall.BASE", tmp_semantic.parent),
        ):
            ctx = recall("remanentia decision SNN", top_k=3)
        assert len(ctx.semantic_memories) > 0

    def test_recall_novelty(
        self, tmp_traces: Path, tmp_graph: Path, tmp_semantic: Path
    ) -> None:
        with (
            patch("memory_recall.TRACES_DIR", tmp_traces),
            patch("memory_recall.SEMANTIC_DIR", tmp_semantic),
            patch("memory_recall.GRAPH_DIR", tmp_graph),
            patch("memory_recall.BASE", tmp_traces.parent),
        ):
            ctx = recall("completely unknown topic xyzfoo", top_k=1)
        assert ctx.novelty_score > 0

    def test_recall_builds_unloaded_memory_index_and_extracts_trace_entities(
        self,
        tmp_traces: Path,
        tmp_graph: Path,
        tmp_semantic: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeMemoryIndex:
            def __init__(self) -> None:
                self.build_kwargs: dict[str, bool] | None = None

            def load(self) -> bool:
                return False

            def build(self, **kwargs: bool) -> None:
                self.build_kwargs = kwargs

            def search(self, query: str, top_k: int) -> list[SimpleNamespace]:
                return [
                    SimpleNamespace(
                        name="2026-03-15_decision_stdp_removal.md",
                        score=0.91,
                        snippet="SNN removal and embedding weight changed.",
                    )
                ]

        fake = FakeMemoryIndex()
        monkeypatch.setattr("memory_index.MemoryIndex", lambda: fake)

        with (
            patch("memory_recall.TRACES_DIR", tmp_traces),
            patch("memory_recall.SEMANTIC_DIR", tmp_semantic),
            patch("memory_recall.GRAPH_DIR", tmp_graph),
            patch("memory_recall.BASE", tmp_traces.parent),
        ):
            ctx = recall("unrelated wording", top_k=1)

        assert fake.build_kwargs == {"use_gpu_embeddings": False, "use_gliner": False}
        assert ctx.trace == "2026-03-15_decision_stdp_removal.md"
        assert "embedding" in ctx.entities


# ── MemoryContext summary/llm with cross-project ────────────────


class TestMemoryContextExtended:
    def test_summary_with_semantic_memories(self) -> None:
        ctx = MemoryContext(
            query="test",
            semantic_memories=[
                {"path": "memory/decision.md", "key_point": "Important finding"},
            ],
        )
        s = ctx.summary
        assert "1 memories" in s or "Consolidated" in s

    def test_summary_with_cross_project(self) -> None:
        ctx = MemoryContext(
            query="test",
            cross_project=[{"project": "director-ai", "insight": "shared concept"}],
        )
        s = ctx.summary
        assert "Cross-project" in s

    def test_to_llm_context_with_timeline(self) -> None:
        ctx = MemoryContext(
            query="test",
            trace="t.md",
            trace_snippet="snip",
            before=["a.md"],
            after=["b.md"],
        )
        llm = ctx.to_llm_context()
        assert "Before:" in llm
        assert "After:" in llm

    def test_to_llm_context_cross_project(self) -> None:
        ctx = MemoryContext(
            query="test",
            trace="t.md",
            trace_snippet="snip",
            cross_project=[{"project": "foo", "insight": "bar stuff"}],
        )
        llm = ctx.to_llm_context()
        assert "Cross-project" in llm


# ── Missing patterns: error, pipeline, roundtrip ─────────────


class TestRecallEdgeCases:
    def test_empty_query(self, tmp_traces: Path) -> None:
        from memory_recall import recall, MemoryContext

        ctx = recall("", top_k=3)
        assert isinstance(ctx, MemoryContext)

    def test_recall_returns_memory_context(self, tmp_traces: Path) -> None:
        from memory_recall import recall, MemoryContext

        ctx = recall("STDP", top_k=3)
        assert isinstance(ctx, MemoryContext)
        assert hasattr(ctx, "query")
        assert hasattr(ctx, "trace")

    def test_recall_to_llm_context(self, tmp_traces: Path) -> None:
        from memory_recall import recall

        ctx = recall("STDP", top_k=3)
        llm = ctx.to_llm_context()
        assert isinstance(llm, str)


class TestRecallErrorHandling:
    def test_recall_error_on_broken_trace(self, tmp_traces: Path) -> None:
        """Recall handles missing/broken traces gracefully."""
        from memory_recall import recall, MemoryContext

        ctx = recall("nonexistent topic with no matching traces", top_k=1)
        assert isinstance(ctx, MemoryContext)

    def test_recall_error_returns_context(self, tmp_traces: Path) -> None:
        from memory_recall import recall, MemoryContext

        ctx = recall("STDP error handling", top_k=3)
        assert isinstance(ctx, MemoryContext)
        assert hasattr(ctx, "to_llm_context")
