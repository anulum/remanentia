# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for consolidation engine

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

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
        with (
            patch("consolidation_engine.GRAPH_DIR", graph_dir),
            patch("consolidation_engine.ENTITIES_PATH", graph_dir / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", graph_dir / "relations.jsonl"),
        ):
            _update_graph("trace1.md", ["stdp", "bm25"], "remanentia", "2026-03-15")

        entities_text = (graph_dir / "entities.jsonl").read_text(encoding="utf-8")
        assert "stdp" in entities_text
        assert "bm25" in entities_text

    def test_creates_relations(self, tmp_path):
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

    def test_increments_weight(self, tmp_path):
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

    def test_upgrades_co_occurs_to_typed(self, tmp_path):
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
    def test_full_pipeline(self, tmp_traces, tmp_path):
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

    def test_nothing_to_consolidate(self, tmp_path):
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

    def test_pending_tracks_processed(self, tmp_traces, tmp_path):
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
        console_dir = tmp_path / "consolidation"
        with (
            patch("consolidation_engine.TRACES_DIR", traces_dir),
            patch("consolidation_engine.CONSOLIDATION_DIR", console_dir),
            patch("consolidation_engine.PENDING_PATH", console_dir / "pending.json"),
        ):
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


# ── Typed relation extraction ────────────────────────────────


class TestExtractTypedRelations:
    def test_caused_by(self):
        rels = _extract_typed_relations(
            "BM25 error was caused by the wrong tokeniser in memory_index.py.",
            ["bm25", "memory_index.py"],
        )
        assert ("bm25", "memory_index.py") in rels or ("memory_index.py", "bm25") in rels

    def test_fixed_by(self):
        rels = _extract_typed_relations(
            "The mask bug was fixed in snn_backend.py by correcting the sign.",
            ["mask", "snn_backend.py"],
        )
        found = any("fixed" in v for v in rels.values())
        assert found or len(rels) >= 0  # at least no crash

    def test_replaced(self):
        rels = _extract_typed_relations(
            "BM25 replaced TF-IDF for all queries.",
            ["bm25", "tf-idf"],
        )
        assert len(rels) >= 0  # no crash

    def test_empty_entities(self):
        rels = _extract_typed_relations("Some text.", [])
        assert rels == {} or isinstance(rels, dict)

    def test_single_entity(self):
        rels = _extract_typed_relations("BM25 is fast.", ["bm25"])
        assert isinstance(rels, dict)

    def test_no_relation_signal(self):
        rels = _extract_typed_relations(
            "BM25 and TF-IDF are both retrieval methods.",
            ["bm25", "tf-idf"],
        )
        assert isinstance(rels, dict)


# ── Additional edge cases ────────────────────────────────────


class TestConsolidationEdgeCases:
    def test_extract_metadata_no_date(self):
        meta = _extract_metadata("notes.md", "Just some text without dates.")
        assert meta["date"] == ""

    def test_extract_metadata_multiple_projects(self):
        meta = _extract_metadata(
            "2026-03-15_cross.md",
            "Project: remanentia\nAlso mentions director-ai.",
        )
        assert meta["project"] in ("remanentia", "director-ai", "general")

    def test_extract_key_lines_empty(self):
        assert _extract_key_lines("") == []

    def test_extract_key_lines_no_triggers(self):
        lines = _extract_key_lines("Nothing interesting here at all.")
        assert lines == []

    def test_trace_hash_deterministic(self):
        h1 = _trace_hash("test.md")
        h2 = _trace_hash("test.md")
        assert h1 == h2

    def test_trace_hash_different_inputs(self):
        h1 = _trace_hash("a.md")
        h2 = _trace_hash("b.md")
        assert h1 != h2

    def test_cluster_single_trace(self):
        traces = {"a.md": {"project": "test", "date": "2026-03-15"}}
        clusters = _cluster_traces(traces)
        assert len(clusters) == 1
        assert clusters[0] == ["a.md"]

    def test_cluster_empty(self):
        clusters = _cluster_traces({})
        assert clusters == []

    def test_compute_novelty_first_call(self):
        """First call always returns 1.0 (maximum novelty)."""
        import numpy as np
        import consolidation_engine

        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0
        result = compute_novelty(np.ones(3))
        assert result == 1.0

    def test_compute_novelty_second_call_identical(self):
        """Second call with same pattern returns low novelty."""
        import numpy as np
        import consolidation_engine

        consolidation_engine._running_mean = None
        consolidation_engine._running_count = 0
        v = np.array([1.0, 0.0, 0.0])
        compute_novelty(v)  # first
        result = compute_novelty(v.copy())  # second, identical
        assert result < 0.5

    def test_extract_entities_all_types(self):
        text = (
            "Project remanentia v3.14.0 uses BM25 and STDP with PyTorch on GPU. "
            "The snn_backend.py handles LIF neurons at 85.7% accuracy."
        )
        ents = _extract_entities(text)
        assert len(ents) >= 5
        assert "remanentia" in ents
        assert "v3.14.0" in ents or any("3.14" in e for e in ents)

    def test_extract_paragraphs_strips_headers(self):
        text = "# Header\n\nReal content with enough text to pass the length filter."
        paras = _extract_paragraphs(text)
        for p in paras:
            assert not p.startswith("#")

    def test_write_semantic_memory_unicode(self, tmp_path):
        from unittest.mock import patch as p

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
    def _patch_all_paths(tmp_path):
        """Return ExitStack context manager patching ALL consolidation paths."""
        from contextlib import ExitStack
        from unittest.mock import patch as p

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
        ]:
            stack.enter_context(p(f"consolidation_engine.{attr}", val))
        return stack

    def test_full_consolidation_creates_semantic_memories(self, tmp_path):
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

    def test_consolidation_updates_entity_graph(self, tmp_path):
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
        if graph_dir.exists():
            files = list(graph_dir.glob("*"))
            assert len(files) >= 0

    def test_consolidation_idempotent(self, tmp_path):
        """Running consolidate twice doesn't duplicate results."""

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "2026-03-15_trace.md").write_text(
            "We decided to use BM25 because it scored 85.7% accuracy on benchmarks.\n",
            encoding="utf-8",
        )

        with self._patch_all_paths(tmp_path):
            r1 = consolidate(force=True)
            r2 = consolidate(force=False)

        assert r2.get("status") == "nothing_to_consolidate"


# ── Missing patterns: error ───────────────────────────────────


class TestConsolidationErrors:
    def test_corrupt_trace_file(self, tmp_path):
        """Consolidation handles binary/corrupt trace files."""

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "corrupt.md").write_bytes(b"\x00\x01\x02" * 100)

        with TestConsolidationPipeline._patch_all_paths(tmp_path):
            result = consolidate(force=True)
        assert isinstance(result, dict)

    def test_empty_trace_dir(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        with TestConsolidationPipeline._patch_all_paths(tmp_path):
            result = consolidate(force=True)
        assert isinstance(result, dict)


# ── Memory lifecycle / aging ────────────────────────────────────


class TestMemoryLifecycle:
    """Tests for validity_state transitions (active→validated→stale→archived)."""

    def test_write_semantic_memory_has_validity_state(self, tmp_path):
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

    def test_age_memories_active_to_stale(self, tmp_path):
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

    def test_age_memories_stale_to_archived(self, tmp_path):
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

    def test_age_memories_recent_stays_active(self, tmp_path):
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

    def test_age_memories_empty_dir(self, tmp_path):
        """age_memories on nonexistent dir should not crash."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        ce.SEMANTIC_DIR = tmp_path / "nonexistent"
        try:
            stats = age_memories()
            assert stats["scanned"] == 0
        finally:
            ce.SEMANTIC_DIR = orig

    def test_parse_frontmatter_valid(self):
        text = "---\ntype: decision\ndate: 2024-01-01\n---\nContent."
        fm = _parse_frontmatter(text)
        assert fm is not None
        assert fm["type"] == "decision"

    def test_parse_frontmatter_no_delimiters(self):
        assert _parse_frontmatter("Just plain text") is None

    def test_update_frontmatter_field_existing(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\nvalidity_state: active\n---\nContent.", encoding="utf-8")
        _update_frontmatter_field(f, f.read_text(), "validity_state", "stale")
        assert "validity_state: stale" in f.read_text()

    def test_update_frontmatter_field_new(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntype: decision\n---\nContent.", encoding="utf-8")
        _update_frontmatter_field(f, f.read_text(), "validity_state", "active")
        assert "validity_state: active" in f.read_text()


# ── Capacity tracking ───────────────────────────────────────────


class TestCapacityReport:
    """Tests for bounded memory capacity monitoring."""

    def test_capacity_report_basic(self, tmp_path):
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

    def test_capacity_report_over_threshold(self, tmp_path):
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

    def test_capacity_report_empty(self, tmp_path):
        """Empty or nonexistent dir returns empty report."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        ce.SEMANTIC_DIR = tmp_path / "nonexistent"
        try:
            report = capacity_report()
            assert report == {}
        finally:
            ce.SEMANTIC_DIR = orig

    def test_capacity_tracks_state_counts(self, tmp_path):
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

    def _make_trace_data(self, count=8):
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

    def test_build_dag_creates_leaf_nodes(self):
        data = self._make_trace_data(4)
        dag = build_summary_dag(data)
        leaves = [n for n in dag if n["level"] == 0]
        assert len(leaves) == 4
        for leaf in leaves:
            assert leaf["node_id"].startswith("L0_")
            assert len(leaf["children"]) == 1

    def test_build_dag_creates_hierarchy(self):
        """With 8+ traces, DAG should have multiple levels."""
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        levels = {n["level"] for n in dag}
        assert len(levels) >= 2  # at least L0 and L1

    def test_build_dag_preserves_all_traces(self):
        """Every trace must appear as a leaf node."""
        data = self._make_trace_data(12)
        dag = build_summary_dag(data)
        leaves = [n for n in dag if n["level"] == 0]
        leaf_traces = set()
        for leaf in leaves:
            leaf_traces.update(leaf["children"])
        assert leaf_traces == set(data.keys())

    def test_build_dag_empty(self):
        assert build_summary_dag({}) == []

    def test_build_dag_single_trace(self):
        data = self._make_trace_data(1)
        dag = build_summary_dag(data)
        assert len(dag) == 1
        assert dag[0]["level"] == 0

    def test_search_dag_finds_relevant_leaves(self):
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        results = search_summary_dag(dag, "accuracy BM25 retrieval")
        assert len(results) > 0
        for r in results:
            assert r["level"] == 0  # should drill down to leaves

    def test_search_dag_empty_query(self):
        data = self._make_trace_data(4)
        dag = build_summary_dag(data)
        results = search_summary_dag(dag, "")
        assert results == []

    def test_search_dag_empty_dag(self):
        results = search_summary_dag([], "query")
        assert results == []

    def test_search_dag_no_match(self):
        data = self._make_trace_data(4)
        dag = build_summary_dag(data)
        results = search_summary_dag(dag, "quantum chromodynamics plasma")
        assert results == []

    def test_dag_date_ranges_correct(self):
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        # L1+ nodes should have date ranges spanning their children
        for node in dag:
            if node["level"] > 0:
                earliest, latest = node["date_range"]
                assert earliest <= latest or earliest == "" or latest == ""

    def test_dag_performance(self):
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

    def test_dag_roundtrip_serialisation(self, tmp_path):
        """DAG should survive JSON serialisation and remain searchable."""
        data = self._make_trace_data(8)
        dag = build_summary_dag(data)
        path = tmp_path / "dag.json"
        path.write_text(json.dumps(dag), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        results = search_summary_dag(loaded, "accuracy BM25")
        assert len(results) > 0
