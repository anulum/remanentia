# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for consolidation engine

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import consolidation_engine
from consolidation_trace_analysis import (
    cluster_traces_python,
    extract_entities_python,
    extract_key_lines_python,
)
import numpy as np
import pytest

from consolidation_engine import (
    _cluster_traces,
    _extract_entities,
    _extract_key_lines,
    _extract_metadata,
    _extract_paragraphs,
    _extract_typed_relations,
    _parse_frontmatter,
    _trace_hash,
    _update_frontmatter_field,
    _update_graph,
    _write_semantic_memory,
    age_memories,
    build_summary_dag,
    capacity_report,
    compute_novelty,
    consolidate,
    get_pending_traces,
    search_summary_dag,
)


# ── Metadata extraction ──────────────────────────────────────────


class TestExtractMetadata:
    def test_date_from_filename(self) -> None:
        meta = _extract_metadata("2026-03-15_decision_stdp.md", "")
        assert meta["date"] == "2026-03-15"

    def test_date_with_time(self) -> None:
        meta = _extract_metadata("2026-03-15T0415_trace.md", "")
        assert meta["date"] == "2026-03-15T0415"

    def test_project_detection(self) -> None:
        meta = _extract_metadata("director-ai_audit.md", "")
        assert meta["project"] == "director-ai"

    def test_project_neurocore(self) -> None:
        meta = _extract_metadata("neurocore_migration.md", "")
        assert meta["project"] == "sc-neurocore"

    def test_project_remanentia(self) -> None:
        meta = _extract_metadata("remanentia_index.md", "")
        assert meta["project"] == "remanentia"

    def test_project_general_fallback(self) -> None:
        meta = _extract_metadata("random_notes.md", "")
        assert meta["project"] == "general"

    def test_type_decision(self) -> None:
        meta = _extract_metadata("decision_stdp.md", "")
        assert meta["type"] == "decision"

    def test_type_finding(self) -> None:
        meta = _extract_metadata("breakthrough_results.md", "")
        assert meta["type"] == "finding"

    def test_type_strategy(self) -> None:
        meta = _extract_metadata("revenue_plan.md", "")
        assert meta["type"] == "strategy"

    def test_type_technical(self) -> None:
        meta = _extract_metadata("daemon_fix.md", "")
        assert meta["type"] == "technical"

    def test_type_general_fallback(self) -> None:
        meta = _extract_metadata("random_notes.md", "")
        assert meta["type"] == "general"


# ── Entity extraction ────────────────────────────────────────────


class TestExtractEntities:
    def test_project_detection(self) -> None:
        entities = _extract_entities("The director-ai pipeline scored 78%.")
        assert "director-ai" in entities

    def test_concept_detection(self) -> None:
        entities = _extract_entities("We used STDP learning rule with PyTorch.")
        assert "stdp" in entities
        assert "pytorch" in entities

    def test_version_numbers(self) -> None:
        entities = _extract_entities("Released v3.9.0 and v0.2.0.")
        assert "v3.9.0" in entities
        assert "v0.2.0" in entities

    def test_percentages(self) -> None:
        entities = _extract_entities("Accuracy reached 66.4% on LOCOMO.")
        assert "66.4%" in entities

    def test_file_paths(self) -> None:
        entities = _extract_entities("Fixed the bug in snn_backend.py.")
        assert "snn_backend.py" in entities

    def test_camelcase_names(self) -> None:
        entities = _extract_entities("The ArcaneNeuron class handles encoding.")
        assert "ArcaneNeuron" in entities

    def test_snake_case_names(self) -> None:
        entities = _extract_entities("The compute_order_parameter function is key.")
        assert "compute_order_parameter" in entities

    def test_empty_text(self) -> None:
        assert _extract_entities("") == []


# ── Key line extraction ──────────────────────────────────────────


class TestExtractKeyLines:
    def test_decision_trigger(self) -> None:
        text = "# Notes\n\n- We decided to remove SNN scoring\n- Other stuff"
        lines = _extract_key_lines(text)
        assert any("decided" in l.lower() for l in lines)

    def test_finding_trigger(self) -> None:
        text = "## Results\n\nWe found that BM25 outperforms TF-IDF by 20%."
        lines = _extract_key_lines(text)
        assert any("found" in l.lower() for l in lines)

    def test_metric_trigger(self) -> None:
        text = "P@1 accuracy improved from 85.7% to 100% on internal benchmark."
        lines = _extract_key_lines(text)
        assert len(lines) > 0

    def test_context_capture(self) -> None:
        text = "We decided to remove SNN.\nThe reason was zero signal.\nAcross 70 experiments."
        lines = _extract_key_lines(text)
        # Should capture the trigger line + context
        assert any("zero signal" in l or "70 experiments" in l for l in lines)

    def test_skip_headers(self) -> None:
        text = "# Decision\n\nActual content here with a decision made."
        lines = _extract_key_lines(text)
        assert not any(l.startswith("#") for l in lines)

    def test_cap_at_30(self) -> None:
        text = "\n".join(f"We decided thing {i} is important." for i in range(50))
        lines = _extract_key_lines(text)
        assert len(lines) <= 30


# ── Paragraph extraction ─────────────────────────────────────────


class TestExtractParagraphs:
    def test_splits_on_blank_lines(self) -> None:
        text = "First block with enough content here for splitting.\n\nSecond block also has enough content for splitting."
        paras = _extract_paragraphs(text)
        assert len(paras) == 2

    def test_filters_short(self) -> None:
        text = "Short\n\nThis paragraph has more than enough content to be indexed properly by the engine."
        paras = _extract_paragraphs(text)
        assert len(paras) == 1

    def test_strips_pure_headers(self) -> None:
        text = "# Just a header\n\n# Header\nWith content following the header line."
        paras = _extract_paragraphs(text)
        assert len(paras) == 1
        assert "With content" in paras[0]


# ── Trace hash ───────────────────────────────────────────────────


class TestTraceHash:
    def test_deterministic(self) -> None:
        h1 = _trace_hash("2026-03-15_decision.md")
        h2 = _trace_hash("2026-03-15_decision.md")
        assert h1 == h2

    def test_different_inputs(self) -> None:
        h1 = _trace_hash("trace_a.md")
        h2 = _trace_hash("trace_b.md")
        assert h1 != h2

    def test_length(self) -> None:
        h = _trace_hash("test.md")
        assert len(h) == 12


# ── Clustering ───────────────────────────────────────────────────


class TestClusterTraces:
    def test_same_project_same_date(self) -> None:
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-15"},
            "b.md": {"project": "remanentia", "date": "2026-03-15"},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a.md", "b.md"}

    def test_different_projects(self) -> None:
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-15"},
            "b.md": {"project": "director-ai", "date": "2026-03-15"},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 2

    def test_date_gap_splits(self) -> None:
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-10"},
            "b.md": {"project": "remanentia", "date": "2026-03-15"},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 2

    def test_within_2_day_window(self) -> None:
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-14"},
            "b.md": {"project": "remanentia", "date": "2026-03-15"},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 1


# ── Novelty computation ──────────────────────────────────────────


class TestComputeNovelty:
    def test_first_pattern_max_novelty(self) -> None:
        # Reset global state
        import consolidation_engine

        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0

        pattern = np.array([1.0, 0.0, 0.0, 1.0])
        novelty = compute_novelty(pattern)
        assert novelty == 1.0

    def test_identical_pattern_low_novelty(self) -> None:
        import consolidation_engine

        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0

        pattern = np.array([1.0, 0.0, 1.0, 0.0])
        compute_novelty(pattern)  # first
        novelty = compute_novelty(pattern)  # same pattern
        assert novelty < 0.1

    def test_orthogonal_pattern_high_novelty(self) -> None:
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
    def test_writes_file(self, tmp_path: Path) -> None:
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
    def test_creates_entities(self, tmp_path: Path) -> None:
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with (
            patch("consolidation_engine.GRAPH_DIR", graph_dir),
            patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"),
        ):
            _update_graph("trace1.md", ["stdp", "bm25"], "remanentia", "2026-03-15")

        entities_text = (graph_dir / "entities.jsonl").read_text(encoding="utf-8")
        assert "stdp" in entities_text
        assert "bm25" in entities_text

    def test_creates_relations(self, tmp_path: Path) -> None:
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with (
            patch("consolidation_engine.GRAPH_DIR", graph_dir),
            patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"),
        ):
            _update_graph("trace1.md", ["stdp", "bm25", "embedding"], "remanentia", "2026-03-15")

        rels_text = (graph_dir / "relations.jsonl").read_text(encoding="utf-8")
        rels = [json.loads(l) for l in rels_text.strip().split("\n") if l.strip()]
        # 3 entities → 3 pairs
        assert len(rels) == 3

    def test_increments_weight(self, tmp_path: Path) -> None:
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with (
            patch("consolidation_engine.GRAPH_DIR", graph_dir),
            patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"),
        ):
            _update_graph("trace1.md", ["stdp", "bm25"], "remanentia", "2026-03-15")
            _update_graph("trace2.md", ["stdp", "bm25"], "remanentia", "2026-03-16")

        rels_text = (graph_dir / "relations.jsonl").read_text(encoding="utf-8")
        rel = json.loads(rels_text.strip().split("\n")[0])
        assert rel["weight"] == 2
        assert len(rel["evidence"]) == 2

    def test_upgrades_co_occurs_to_typed(self, tmp_path: Path) -> None:
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with (
            patch("consolidation_engine.GRAPH_DIR", graph_dir),
            patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"),
        ):
            _update_graph("trace1.md", ["stdp", "bm25"], "remanentia", "2026-03-15")
            _update_graph(
                "trace2.md",
                ["stdp", "bm25"],
                "remanentia",
                "2026-03-16",
                text="stdp depends on bm25 for scoring",
            )

        rels_text = (graph_dir / "relations.jsonl").read_text(encoding="utf-8")
        rel = json.loads(rels_text.strip().split("\n")[0])
        assert rel["type"] == "depends_on"


# ── Full consolidation pipeline ──────────────────────────────────


class TestConsolidate:
    def test_full_pipeline(self, tmp_traces: Path, tmp_path: Path) -> None:
        console_dir = tmp_path / "consolidation"
        console_dir.mkdir()
        semantic_dir = tmp_path / "memory" / "semantic"
        graph_dir = tmp_path / "memory" / "graph"

        with (
            patch("consolidation_engine.TRACES_DIR", tmp_traces),
            patch("consolidation_engine.CONSOLIDATION_DIR", console_dir),
            patch("consolidation_engine.PENDING_PATH", console_dir / "pending.json"),
            patch("consolidation_engine.LAST_RUN_PATH", console_dir / "last_consolidation.json"),
            patch("consolidation_engine.SEMANTIC_DIR", semantic_dir),
            patch("consolidation_engine.GRAPH_DIR", graph_dir),
            patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"),
            patch("consolidation_engine.CLUSTERS_PATH", graph_dir / "trace_clusters.json"),
        ):
            result = consolidate(force=True)

        assert result["traces_processed"] == 3
        assert result["clusters_formed"] > 0
        assert result["memories_written"] > 0
        assert result["entities_found"] > 0
        assert semantic_dir.exists()

    def test_nothing_to_consolidate(self, tmp_path: Path) -> None:
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        console_dir = tmp_path / "consolidation"

        with (
            patch("consolidation_engine.TRACES_DIR", traces_dir),
            patch("consolidation_engine.CONSOLIDATION_DIR", console_dir),
            patch("consolidation_engine.PENDING_PATH", console_dir / "pending.json"),
        ):
            result = consolidate(force=False)

        assert result["status"] == "nothing_to_consolidate"

    def test_pending_tracks_processed(self, tmp_traces: Path, tmp_path: Path) -> None:
        console_dir = tmp_path / "consolidation"
        console_dir.mkdir()
        semantic_dir = tmp_path / "memory" / "semantic"
        graph_dir = tmp_path / "memory" / "graph"

        with (
            patch("consolidation_engine.TRACES_DIR", tmp_traces),
            patch("consolidation_engine.CONSOLIDATION_DIR", console_dir),
            patch("consolidation_engine.PENDING_PATH", console_dir / "pending.json"),
            patch("consolidation_engine.LAST_RUN_PATH", console_dir / "last_consolidation.json"),
            patch("consolidation_engine.SEMANTIC_DIR", semantic_dir),
            patch("consolidation_engine.GRAPH_DIR", graph_dir),
            patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"),
            patch("consolidation_engine.CLUSTERS_PATH", graph_dir / "trace_clusters.json"),
        ):
            consolidate(force=True)
            # Second run should find nothing pending
            result = consolidate(force=False)

        assert result["status"] == "nothing_to_consolidate"


# ── Edge cases ──────────────────────────────────────────────────


class TestConsolidationEdge:
    def test_zero_norm_novelty(self) -> None:
        import consolidation_engine

        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0
        pattern = np.array([1.0, 0.0, 1.0])
        compute_novelty(pattern)
        zero = np.array([0.0, 0.0, 0.0])
        novelty = compute_novelty(zero)
        assert novelty == 1.0

    def test_get_pending_traces_empty(self, tmp_path: Path) -> None:
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        console_dir = tmp_path / "consolidation"
        with (
            patch("consolidation_engine.TRACES_DIR", traces_dir),
            patch("consolidation_engine.CONSOLIDATION_DIR", console_dir),
            patch("consolidation_engine.PENDING_PATH", console_dir / "pending.json"),
        ):
            pending = get_pending_traces()
        assert pending == []

    def test_cluster_missing_date(self) -> None:
        traces = {
            "a.md": {"project": "test", "date": ""},
            "b.md": {"project": "test", "date": ""},
        }
        clusters = _cluster_traces(traces)
        assert len(clusters) == 1

    def test_extract_paragraphs_empty(self) -> None:
        assert _extract_paragraphs("") == []

    def test_extract_entities_short_file(self) -> None:
        entities = _extract_entities("The a.py file is short.")
        file_names = [e for e in entities if e.endswith(".py")]
        assert not any(len(e) <= 3 for e in file_names)


# ── Typed relation extraction ────────────────────────────────


class TestExtractTypedRelations:
    def test_caused_by(self) -> None:
        rels = _extract_typed_relations(
            "BM25 error was caused by the wrong tokeniser in memory_index.py.",
            ["bm25", "memory_index.py"],
        )
        assert ("bm25", "memory_index.py") in rels or ("memory_index.py", "bm25") in rels

    def test_fixed_by(self) -> None:
        rels = _extract_typed_relations(
            "The mask bug was fixed in snn_backend.py by correcting the sign.",
            ["mask", "snn_backend.py"],
        )
        assert rels[("mask", "snn_backend.py")] == "fixed_by"

    def test_replaced(self) -> None:
        rels = _extract_typed_relations(
            "BM25 replaced TF-IDF for all queries.",
            ["bm25", "tf-idf"],
        )
        assert rels[("bm25", "tf-idf")] == "replaced"

    def test_empty_entities(self) -> None:
        rels = _extract_typed_relations("Some text.", [])
        assert rels == {}

    def test_single_entity(self) -> None:
        rels = _extract_typed_relations("BM25 is fast.", ["bm25"])
        assert rels == {}

    def test_no_relation_signal(self) -> None:
        rels = _extract_typed_relations(
            "BM25 and TF-IDF are both retrieval methods.",
            ["bm25", "tf-idf"],
        )
        assert rels[("bm25", "tf-idf")] == "co_occurs"


# ── Additional edge cases ────────────────────────────────────


class TestConsolidationEdgeCases:
    def test_extract_metadata_no_date(self) -> None:
        meta = _extract_metadata("notes.md", "Just some text without dates.")
        assert meta["date"] == ""

    def test_extract_metadata_multiple_projects(self) -> None:
        meta = _extract_metadata(
            "2026-03-15_cross.md",
            "Project: remanentia\nAlso mentions director-ai.",
        )
        assert meta["project"] in ("remanentia", "director-ai", "general")

    def test_extract_key_lines_empty(self) -> None:
        assert _extract_key_lines("") == []

    def test_extract_key_lines_no_triggers(self) -> None:
        lines = _extract_key_lines("Nothing interesting here at all.")
        assert lines == []

    def test_trace_hash_deterministic(self) -> None:
        h1 = _trace_hash("test.md")
        h2 = _trace_hash("test.md")
        assert h1 == h2

    def test_trace_hash_different_inputs(self) -> None:
        h1 = _trace_hash("a.md")
        h2 = _trace_hash("b.md")
        assert h1 != h2

    def test_cluster_single_trace(self) -> None:
        traces = {"a.md": {"project": "test", "date": "2026-03-15"}}
        clusters = _cluster_traces(traces)
        assert len(clusters) == 1
        assert clusters[0] == ["a.md"]

    def test_cluster_empty(self) -> None:
        clusters = _cluster_traces({})
        assert clusters == []

    def test_compute_novelty_first_call(self) -> None:
        """First call always returns 1.0 (maximum novelty)."""
        import numpy as np
        import consolidation_engine

        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0
        result = compute_novelty(np.ones(3))
        assert result == 1.0

    def test_compute_novelty_second_call_identical(self) -> None:
        """Second call with same pattern returns low novelty."""
        import numpy as np
        import consolidation_engine

        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0
        v = np.array([1.0, 0.0, 0.0])
        compute_novelty(v)  # first
        result = compute_novelty(v.copy())  # second, identical
        assert result < 0.5

    def test_extract_entities_all_types(self) -> None:
        text = (
            "Project remanentia v3.14.0 uses BM25 and STDP with PyTorch on GPU. "
            "The snn_backend.py handles LIF neurons at 85.7% accuracy."
        )
        ents = _extract_entities(text)
        assert len(ents) >= 5
        assert "remanentia" in ents
        assert "v3.14.0" in ents or any("3.14" in e for e in ents)

    def test_extract_paragraphs_strips_headers(self) -> None:
        text = "# Header\n\nReal content with enough text to pass the length filter."
        paras = _extract_paragraphs(text)
        for p in paras:
            assert not p.startswith("#")

    def test_write_semantic_memory_unicode(self, tmp_path: Path) -> None:
        p = patch

        semantic_dir = tmp_path / "semantic"
        with p("consolidation_engine.SEMANTIC_DIR", semantic_dir):
            _write_semantic_memory(
                category="finding",
                topic="coupling-test",
                date="2026-03-15",
                project="remanentia",
                source_traces=["trace_test.md"],
                entities=["kuramoto"],
                content="Šotek–Kuramoto coupling validated with r=0.951.",
            )
        files = list(semantic_dir.rglob("*.md"))
        assert len(files) >= 1
        text = files[0].read_text(encoding="utf-8")
        assert "Šotek" in text


# ── Pipeline integration ─────────────────────────────────────


class TestConsolidationPipeline:
    """Consolidation engine integrated with knowledge store and observer."""

    @staticmethod
    def _patch_all_paths(tmp_path: Path) -> ExitStack:
        """Return ExitStack context manager patching ALL consolidation paths."""
        p = patch

        graph_dir = tmp_path / "graph"
        console_dir = tmp_path / "consolidation"
        stack = ExitStack()
        for attr, val in [
            ("TRACES_DIR", tmp_path / "traces"),
            ("CONSOLIDATION_DIR", console_dir),
            ("SEMANTIC_DIR", tmp_path / "semantic"),
            ("GRAPH_DIR", graph_dir),
            ("ENTITIES_PATH", graph_dir / "entities.jsonl"),
            ("RELATIONS_PATH", graph_dir / "relations.jsonl"),
            ("CLUSTERS_PATH", graph_dir / "trace_clusters.json"),
            ("PENDING_PATH", console_dir / "pending.json"),
            ("LAST_RUN_PATH", console_dir / "last_consolidation.json"),
            ("SUMMARY_DAG_PATH", console_dir / "summary_dag.json"),
        ]:
            stack.enter_context(p(f"consolidation_engine.{attr}", val))
        return stack

    def test_full_consolidation_creates_semantic_memories(self, tmp_path: Path) -> None:
        """Write traces → consolidate → verify semantic memories created."""

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "2026-03-15_decision.md").write_text(
            "# Decision\n\n"
            "We decided to remove SNN from retrieval because experiments showed no signal.\n",
            encoding="utf-8",
        )
        (traces_dir / "2026-03-15_finding.md").write_text(
            "# Finding\n\nBM25 accuracy measured at 85.7% P@1 on the LOCOMO benchmark.\n",
            encoding="utf-8",
        )

        with self._patch_all_paths(tmp_path):
            result = consolidate(force=True)

        assert result["traces_processed"] >= 1
        semantic_dir = tmp_path / "semantic"
        if semantic_dir.exists():
            files = list(semantic_dir.rglob("*.md"))
            assert len(files) >= 1

    def test_consolidation_updates_entity_graph(self, tmp_path: Path) -> None:
        """Consolidation creates entity graph files."""

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "2026-03-15_trace.md").write_text(
            "STDP was replaced by BM25 in remanentia retrieval pipeline v3.14.0.\n",
            encoding="utf-8",
        )

        with self._patch_all_paths(tmp_path):
            consolidate(force=True)

        graph_dir = tmp_path / "graph"
        assert (graph_dir / "entities.jsonl").exists()
        assert (graph_dir / "relations.jsonl").exists()

    def test_consolidation_idempotent(self, tmp_path: Path) -> None:
        """Running consolidate twice doesn't duplicate results."""

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "2026-03-15_trace.md").write_text(
            "We decided to use BM25 because it scored 85.7% accuracy on benchmarks.\n",
            encoding="utf-8",
        )

        with self._patch_all_paths(tmp_path):
            consolidate(force=True)
            r2 = consolidate(force=False)

        assert r2.get("status") == "nothing_to_consolidate"


# ── Missing patterns: error ───────────────────────────────────


class TestConsolidationErrors:
    def test_corrupt_trace_file(self, tmp_path: Path) -> None:
        """Consolidation handles binary/corrupt trace files."""

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "corrupt.md").write_bytes(b"\x00\x01\x02" * 100)

        with TestConsolidationPipeline._patch_all_paths(tmp_path):
            result = consolidate(force=True)
        assert result["traces_processed"] == 1
        assert (tmp_path / "consolidation" / "last_consolidation.json").exists()

    def test_empty_trace_dir(self, tmp_path: Path) -> None:
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        with TestConsolidationPipeline._patch_all_paths(tmp_path):
            result = consolidate(force=True)
        assert result["traces_processed"] == 0
        assert result["memories_written"] == 0


# ── Memory lifecycle / aging ────────────────────────────────────


class TestMemoryLifecycle:
    """Tests for validity_state transitions (active→validated→stale→archived)."""

    def test_write_semantic_memory_has_validity_state(self, tmp_path: Path) -> None:
        """New memories should have validity_state=active in frontmatter."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        ce.SEMANTIC_DIR = tmp_path / "semantic"
        try:
            path = _write_semantic_memory(
                "decision",
                "test-topic",
                "2024-01-01",
                "remanentia",
                ["trace1.md"],
                ["bm25"],
                "Test content.",
            )
            text = path.read_text(encoding="utf-8")
            assert "validity_state: active" in text
            assert "last_accessed:" in text
        finally:
            ce.SEMANTIC_DIR = orig

    def test_age_memories_active_to_stale(self, tmp_path: Path) -> None:
        """Memories older than STALE_AFTER_DAYS without access → stale."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic" / "decision"
        sem_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = tmp_path / "semantic"
        try:
            (sem_dir / "old.md").write_text(
                "---\nvalidity_state: active\nlast_accessed: 2023-01-01\n---\nOld fact.",
                encoding="utf-8",
            )
            stats = age_memories(reference_date="2024-06-01")
            assert stats["active_to_stale"] == 1
            text = (sem_dir / "old.md").read_text(encoding="utf-8")
            assert "validity_state: stale" in text
        finally:
            ce.SEMANTIC_DIR = orig

    def test_age_memories_stale_to_archived(self, tmp_path: Path) -> None:
        """Memories stale for longer than ARCHIVE_AFTER_DAYS → archived."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic" / "finding"
        sem_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = tmp_path / "semantic"
        try:
            (sem_dir / "ancient.md").write_text(
                "---\nvalidity_state: stale\nlast_accessed: 2022-01-01\n---\nAncient.",
                encoding="utf-8",
            )
            stats = age_memories(reference_date="2024-06-01")
            assert stats["stale_to_archived"] == 1
            text = (sem_dir / "ancient.md").read_text(encoding="utf-8")
            assert "validity_state: archived" in text
        finally:
            ce.SEMANTIC_DIR = orig

    def test_age_memories_recent_stays_active(self, tmp_path: Path) -> None:
        """Recently accessed memories should remain active."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic" / "decision"
        sem_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = tmp_path / "semantic"
        try:
            (sem_dir / "fresh.md").write_text(
                "---\nvalidity_state: active\nlast_accessed: 2024-05-20\n---\nFresh.",
                encoding="utf-8",
            )
            stats = age_memories(reference_date="2024-06-01")
            assert stats["active_to_stale"] == 0
            text = (sem_dir / "fresh.md").read_text(encoding="utf-8")
            assert "validity_state: active" in text
        finally:
            ce.SEMANTIC_DIR = orig

    def test_age_memories_empty_dir(self, tmp_path: Path) -> None:
        """age_memories reports zero scanned files when semantic memory is absent."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        ce.SEMANTIC_DIR = tmp_path / "nonexistent"
        try:
            stats = age_memories()
            assert stats["scanned"] == 0
        finally:
            ce.SEMANTIC_DIR = orig

    def test_parse_frontmatter_valid(self) -> None:
        text = "---\ntype: decision\ndate: 2024-01-01\n---\nContent."
        fm = _parse_frontmatter(text)
        assert fm is not None
        assert fm["type"] == "decision"

    def test_parse_frontmatter_no_delimiters(self) -> None:
        assert _parse_frontmatter("Just plain text") is None

    def test_parse_frontmatter_no_closing_delimiter(self) -> None:
        assert _parse_frontmatter("---\ntype: decision\nno closing") is None

    def test_age_memories_no_frontmatter(self, tmp_path: Path) -> None:
        """Files without frontmatter should be skipped, not crash."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic" / "decision"
        sem_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = tmp_path / "semantic"
        try:
            (sem_dir / "no_fm.md").write_text("Just plain text, no frontmatter.", encoding="utf-8")
            stats = age_memories()
            assert stats["scanned"] == 1
            assert stats["active_to_stale"] == 0
        finally:
            ce.SEMANTIC_DIR = orig

    def test_age_memories_no_last_accessed(self, tmp_path: Path) -> None:
        """Files with frontmatter but no last_accessed should be skipped."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic" / "decision"
        sem_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = tmp_path / "semantic"
        try:
            (sem_dir / "no_date.md").write_text(
                "---\nvalidity_state: active\n---\nNo dates.", encoding="utf-8"
            )
            stats = age_memories()
            assert stats["active_to_stale"] == 0
        finally:
            ce.SEMANTIC_DIR = orig

    def test_age_memories_corrupt_date(self, tmp_path: Path) -> None:
        """Files with unparsable dates should be skipped."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic" / "decision"
        sem_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = tmp_path / "semantic"
        try:
            (sem_dir / "bad_date.md").write_text(
                "---\nvalidity_state: active\nlast_accessed: not-a-date\n---\nBad.",
                encoding="utf-8",
            )
            stats = age_memories()
            assert stats["active_to_stale"] == 0
        finally:
            ce.SEMANTIC_DIR = orig

    def test_update_frontmatter_field_existing(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("---\nvalidity_state: active\n---\nContent.", encoding="utf-8")
        _update_frontmatter_field(f, f.read_text(), "validity_state", "stale")
        assert "validity_state: stale" in f.read_text()

    def test_update_frontmatter_field_new(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("---\ntype: decision\n---\nContent.", encoding="utf-8")
        _update_frontmatter_field(f, f.read_text(), "validity_state", "active")
        assert "validity_state: active" in f.read_text()


# ── Capacity tracking ───────────────────────────────────────────


class TestCapacityReport:
    """Tests for bounded memory capacity monitoring."""

    def test_capacity_report_basic(self, tmp_path: Path) -> None:
        """Capacity report should return per-category stats."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic"
        dec_dir = sem_dir / "decision"
        dec_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = sem_dir
        try:
            (dec_dir / "d1.md").write_text(
                "---\nvalidity_state: active\n---\n" + "x" * 1000, encoding="utf-8"
            )
            report = capacity_report()
            assert "decision" in report
            assert report["decision"]["file_count"] == 1
            assert report["decision"]["chars"] > 0
            assert isinstance(report["decision"]["usage_pct"], float)
        finally:
            ce.SEMANTIC_DIR = orig

    def test_capacity_report_over_threshold(self, tmp_path: Path) -> None:
        """Categories exceeding CAPACITY_WARN_PERCENT should be flagged."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic"
        dec_dir = sem_dir / "decision"
        dec_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = sem_dir
        try:
            # Write enough to exceed 80% of 50_000 char limit
            (dec_dir / "big.md").write_text(
                "---\nvalidity_state: active\n---\n" + "x" * 45_000, encoding="utf-8"
            )
            report = capacity_report()
            assert report["decision"]["needs_consolidation"] is True
            assert report["decision"]["usage_pct"] >= 80.0
        finally:
            ce.SEMANTIC_DIR = orig

    def test_capacity_report_empty(self, tmp_path: Path) -> None:
        """Empty or nonexistent dir returns empty report."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        ce.SEMANTIC_DIR = tmp_path / "nonexistent"
        try:
            report = capacity_report()
            assert report == {}
        finally:
            ce.SEMANTIC_DIR = orig

    def test_capacity_report_skips_non_dirs(self, tmp_path: Path) -> None:
        """Non-directory items in SEMANTIC_DIR should be skipped."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic"
        sem_dir.mkdir()
        (sem_dir / "stray_file.md").write_text("Not a category dir.")
        ce.SEMANTIC_DIR = sem_dir
        try:
            report = capacity_report()
            assert "stray_file.md" not in report
        finally:
            ce.SEMANTIC_DIR = orig

    def test_capacity_tracks_state_counts(self, tmp_path: Path) -> None:
        """Report should include per-state breakdown."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem_dir = tmp_path / "semantic"
        dec_dir = sem_dir / "decision"
        dec_dir.mkdir(parents=True)
        ce.SEMANTIC_DIR = sem_dir
        try:
            (dec_dir / "a.md").write_text("---\nvalidity_state: active\n---\nA.", encoding="utf-8")
            (dec_dir / "s.md").write_text("---\nvalidity_state: stale\n---\nS.", encoding="utf-8")
            report = capacity_report()
            assert report["decision"]["state_counts"]["active"] == 1
            assert report["decision"]["state_counts"]["stale"] == 1
        finally:
            ce.SEMANTIC_DIR = orig


# ── Hierarchical summary DAGs ───────────────────────────────────


class TestSummaryDAG:
    """Tests for multi-level summary DAG construction and search."""

    def _make_trace_data(self, count: int = 8) -> dict[str, dict[str, Any]]:
        return {
            f"2024-0{(i % 9) + 1}-{(i % 28) + 1:02d}_trace_{i}.md": {
                "date": f"2024-0{(i % 9) + 1}-{(i % 28) + 1:02d}",
                "project": "remanentia" if i % 2 == 0 else "director-ai",
                "entities": [f"entity_{i}", "bm25", "retrieval"],
                "key_lines": [f"Finding {i}: measured accuracy at {50 + i}%"],
                "text": f"Trace {i}: We measured accuracy at {50 + i}% for BM25 retrieval.",
            }
            for i in range(count)
        }

    def test_build_dag_creates_leaf_nodes(self) -> None:
        data = self._make_trace_data(4)
        dag = build_summary_dag(data)
        leaves = [n for n in dag if n["level"] == 0]
        assert len(leaves) == 4
        for leaf in leaves:
            assert leaf["node_id"].startswith("L0_")
            assert len(leaf["children"]) == 1

    def test_build_dag_creates_hierarchy(self) -> None:
        """With 8+ traces, DAG should have multiple levels."""
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        levels = {n["level"] for n in dag}
        assert len(levels) >= 2  # at least L0 and L1

    def test_build_dag_preserves_all_traces(self) -> None:
        """Every trace must appear as a leaf node."""
        data = self._make_trace_data(12)
        dag = build_summary_dag(data)
        leaves = [n for n in dag if n["level"] == 0]
        leaf_traces = set()
        for leaf in leaves:
            leaf_traces.update(leaf["children"])
        assert leaf_traces == set(data.keys())

    def test_build_dag_empty(self) -> None:
        assert build_summary_dag({}) == []

    def test_build_dag_single_trace(self) -> None:
        data = self._make_trace_data(1)
        dag = build_summary_dag(data)
        assert len(dag) == 1
        assert dag[0]["level"] == 0

    def test_search_dag_finds_relevant_leaves(self) -> None:
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        results = search_summary_dag(dag, "accuracy BM25 retrieval")
        assert len(results) > 0
        for r in results:
            assert r["level"] == 0  # should drill down to leaves

    def test_search_dag_empty_query(self) -> None:
        data = self._make_trace_data(4)
        dag = build_summary_dag(data)
        results = search_summary_dag(dag, "")
        assert results == []

    def test_search_dag_empty_dag(self) -> None:
        results = search_summary_dag([], "query")
        assert results == []

    def test_search_dag_no_match(self) -> None:
        data = self._make_trace_data(4)
        dag = build_summary_dag(data)
        results = search_summary_dag(dag, "quantum chromodynamics plasma")
        assert results == []

    def test_search_dag_deduplicates_seen_nodes(self) -> None:
        """Search should not revisit already-seen nodes."""
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        # Search twice with same query — should return same results
        r1 = search_summary_dag(dag, "accuracy BM25 retrieval", top_k=5)
        r2 = search_summary_dag(dag, "accuracy BM25 retrieval", top_k=5)
        assert len(r1) == len(r2)
        assert [n["node_id"] for n in r1] == [n["node_id"] for n in r2]

    def test_dag_date_ranges_correct(self) -> None:
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        # L1+ nodes should have date ranges spanning their children
        for node in dag:
            if node["level"] > 0:
                earliest, latest = node["date_range"]
                assert earliest <= latest or earliest == "" or latest == ""

    def test_dag_performance(self) -> None:
        """Building and searching a 100-trace DAG should be fast."""
        import time

        data = self._make_trace_data(100)
        t0 = time.perf_counter()
        dag = build_summary_dag(data)
        build_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(100):
            search_summary_dag(dag, "accuracy retrieval BM25")
        search_ms = (time.perf_counter() - t0) * 1000 / 100

        assert build_ms < 100, f"DAG build too slow: {build_ms:.1f}ms for 100 traces"
        assert search_ms < 10, f"DAG search too slow: {search_ms:.1f}ms"

    def test_dag_roundtrip_serialisation(self, tmp_path: Path) -> None:
        """DAG should survive JSON serialisation and remain searchable."""
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        path = tmp_path / "dag.json"
        path.write_text(json.dumps(dag), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        results = search_summary_dag(loaded, "accuracy BM25")
        assert len(results) > 0


class TestConsolidationPythonFallbackContracts:
    def test_explicit_python_extraction_helpers_cover_dynamic_trace_features(self) -> None:
        text = (
            "# Header\n\n"
            "We decided to fix compute_order_parameter in src/solver.py because BM25 "
            "accuracy improved to 92.5% for Remanentia v3.9.0 on GPU.\n"
            "The next context line explains the root cause and measured result.\n"
            "Another context line keeps the decision grounded.\n"
            "ArcaneNeuron replaced OldRetriever and depends on FastAPI."
        )

        entities = extract_entities_python(text)
        key_lines = extract_key_lines_python(text)
        paragraphs = _extract_paragraphs(text)
        relations = _extract_typed_relations(text, ["ArcaneNeuron", "OldRetriever", "FastAPI"])
        depends_relations = _extract_typed_relations(
            "ArcaneNeuron depends on FastAPI.",
            ["ArcaneNeuron", "FastAPI"],
        )
        co_occurs_relations = _extract_typed_relations(
            "ArcaneNeuron and FastAPI are mentioned together.",
            ["ArcaneNeuron", "MissingEntity", "FastAPI"],
        )

        assert {"remanentia", "bm25", "gpu", "v3.9.0", "92.5%", "solver.py"} <= set(entities)
        assert "compute_order_parameter" in entities
        assert key_lines and "root cause" in key_lines[0]
        assert paragraphs and not paragraphs[0].startswith("#")
        assert relations[("ArcaneNeuron", "OldRetriever")] == "replaced"
        assert depends_relations[("ArcaneNeuron", "FastAPI")] == "depends_on"
        assert co_occurs_relations[("ArcaneNeuron", "FastAPI")] == "co_occurs"

    def test_explicit_python_clustering_splits_by_project_and_date_gap(self) -> None:
        traces = {
            "a.md": {"project": "remanentia", "date": "2026-03-01"},
            "b.md": {"project": "remanentia", "date": "2026-03-02"},
            "c.md": {"project": "remanentia", "date": "2026-03-10"},
            "d.md": {"project": "director-ai", "date": "not-a-date"},
            "e.md": {"project": "director-ai", "date": ""},
        }

        clusters = cluster_traces_python(traces)

        assert ["a.md", "b.md"] in clusters
        assert ["c.md"] in clusters
        assert {"d.md", "e.md"} in [set(cluster) for cluster in clusters]

    def test_native_free_summary_dag_builds_hierarchy_and_frontmatter_parses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = cast(Any, consolidation_engine).import_module

        def reject_native(name: str) -> Any:
            if name == "remanentia_consolidation":
                raise ImportError(name)
            return real_import(name)

        monkeypatch.setattr(consolidation_engine, "import_module", reject_native)
        trace_data = {
            f"trace_{idx}.md": {
                "date": f"2026-03-{idx + 1:02d}",
                "project": "remanentia" if idx < 5 else "director-ai",
                "entities": ["bm25", f"entity{idx}"],
                "key_lines": [f"BM25 retrieval finding {idx}"],
                "text": f"Full trace text {idx} about retrieval accuracy.",
            }
            for idx in range(9)
        }

        dag = build_summary_dag(trace_data)
        parent_nodes = [node for node in dag if node["level"] > 0]

        assert len(dag) > len(trace_data)
        assert parent_nodes
        assert parent_nodes[0]["date_range"][0] <= parent_nodes[0]["date_range"][1]
        assert _parse_frontmatter("plain text") is None
        assert _parse_frontmatter("---\nmissing end") is None
        assert _parse_frontmatter("---\nvalidity_state: active\n- ignored\n---\nbody") == {
            "validity_state": "active"
        }

    def test_forced_consolidation_processes_temp_traces_and_updates_outputs(
        self, tmp_path: Path
    ) -> None:
        traces = tmp_path / "traces"
        semantic = tmp_path / "semantic"
        graph = tmp_path / "graph"
        consolidation = tmp_path / "consolidation"
        traces.mkdir()
        (traces / "2026-03-15_remanentia_decision.md").write_text(
            "# Decision\n\n"
            "We decided BM25 retrieval fixed the Remanentia daemon because accuracy improved.\n\n"
            "The fix produced a measured 92.5% result in retrieval.py.",
            encoding="utf-8",
        )

        patches = (
            patch("consolidation_engine.TRACES_DIR", traces),
            patch("consolidation_engine.SEMANTIC_DIR", semantic),
            patch("consolidation_engine.GRAPH_DIR", graph),
            patch("consolidation_engine.CONSOLIDATION_DIR", consolidation),
            patch("consolidation_engine.PENDING_PATH", consolidation / "pending.json"),
            patch(
                "consolidation_engine.LAST_RUN_PATH",
                consolidation / "last_consolidation.json",
            ),
            patch("consolidation_engine.CONFLICTS_PATH", consolidation / "conflicts.json"),
            patch("consolidation_engine.ENTITIES_PATH", graph / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", graph / "relations.jsonl"),
            patch("consolidation_engine.CLUSTERS_PATH", graph / "trace_clusters.json"),
            patch("consolidation_engine.SUMMARY_DAG_PATH", consolidation / "summary_dag.json"),
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
        ):
            stats = consolidate(force=True)

        assert stats["traces_processed"] == 1
        assert stats["memories_written"] == 1
        assert list(semantic.rglob("*.md"))
        assert (graph / "entities.jsonl").exists()
        assert (graph / "relations.jsonl").exists()
        assert (consolidation / "pending.json").exists()
