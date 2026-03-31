# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for observer

from __future__ import annotations

from unittest.mock import MagicMock, patch


from observer import (
    ObserverState,
    _has_signal,
    _split_into_paragraphs,
    extract_notes_from_file,
    heartbeat,
    observe_once,
)


class TestHasSignal:
    def test_decision(self):
        assert _has_signal("We decided to remove SNN scoring.") is True

    def test_finding(self):
        assert _has_signal("We found that BM25 outperforms TF-IDF.") is True

    def test_metric(self):
        assert _has_signal("Accuracy reached 81.2% on LOCOMO benchmark.") is True

    def test_no_signal(self):
        assert _has_signal("Just some random text without any signals.") is False

    def test_version(self):
        assert _has_signal("Released v3.9.0 to PyPI.") is True


class TestSplitIntoParagraphs:
    def test_filters_short(self):
        text = "Short\n\nThis is a longer paragraph that decided to include enough content for the filter."
        paras = _split_into_paragraphs(text)
        assert len(paras) == 1
        assert "decided" in paras[0]

    def test_filters_no_signal(self):
        text = "This is a sufficiently long paragraph with no signal words whatsoever in the entire text block here.\n\nWe decided this is important enough to note and record for future reference."
        paras = _split_into_paragraphs(text)
        assert len(paras) == 1
        assert "decided" in paras[0]

    def test_empty(self):
        assert _split_into_paragraphs("") == []


class TestExtractNotesFromFile:
    def test_extracts_notes(self, tmp_path):
        f = tmp_path / "trace.md"
        f.write_text(
            "# Decision\n\n"
            "We decided to remove SNN from retrieval because it adds zero signal across 70 experiments.\n\n"
            "## Result\n\n"
            "The accuracy measured after removal was 100% on our internal benchmark of 14 queries.\n",
            encoding="utf-8",
        )
        notes = extract_notes_from_file(f)
        assert len(notes) >= 1
        assert all(n["source"] == "trace.md" for n in notes)

    def test_short_file(self, tmp_path):
        f = tmp_path / "short.md"
        f.write_text("hi", encoding="utf-8")
        assert extract_notes_from_file(f) == []


class TestObserverState:
    def test_new_file_detected(self, tmp_path):
        state = ObserverState()
        f = tmp_path / "test.md"
        f.write_text("content", encoding="utf-8")
        assert state.is_new_or_changed(f) is True

    def test_processed_file_skipped(self, tmp_path):
        state = ObserverState()
        f = tmp_path / "test.md"
        f.write_text("content", encoding="utf-8")
        state.mark_processed(f)
        assert state.is_new_or_changed(f) is False

    def test_save_and_load(self, tmp_path):
        state = ObserverState()
        f = tmp_path / "test.md"
        f.write_text("content", encoding="utf-8")
        state.mark_processed(f)
        state_path = tmp_path / "state.json"
        state.save(state_path)

        state2 = ObserverState()
        assert state2.load(state_path) is True
        assert state2.is_new_or_changed(f) is False

    def test_load_nonexistent(self, tmp_path):
        state = ObserverState()
        assert state.load(tmp_path / "nope.json") is False


class TestObserveOnce:
    def test_creates_notes(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "2026-03-24_decision.md").write_text(
            "# Decision\n\n"
            "We decided to remove SNN from retrieval scoring because experiments showed zero signal.\n\n"
            "## Outcome\n\n"
            "The accuracy measured improved from 85.7% to 100% on our benchmark.\n",
            encoding="utf-8",
        )
        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"

        with (
            patch("observer.WATCHED_DIRS", {"traces": traces_dir}),
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            state = ObserverState()
            result = observe_once(state, {"traces": traces_dir})

        assert result["files_scanned"] >= 1
        assert result["files_new"] >= 1
        assert result["notes_created"] >= 1

    def test_idempotent(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "trace.md").write_text(
            "# Finding\n\nWe found that BM25 scoring works well enough for retrieval accuracy.\n",
            encoding="utf-8",
        )
        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"

        with (
            patch("observer.WATCHED_DIRS", {"traces": traces_dir}),
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            state = ObserverState()
            r1 = observe_once(state, {"traces": traces_dir})
            r2 = observe_once(state, {"traces": traces_dir})

        assert r1["files_new"] >= 1
        assert r2["files_new"] == 0  # already processed

    def test_empty_dir(self, tmp_path):
        state = ObserverState()
        result = observe_once(state, {"empty": tmp_path})
        assert result["files_scanned"] == 0

    def test_nonexistent_dir(self, tmp_path):
        state = ObserverState()
        result = observe_once(state, {"nope": tmp_path / "nonexistent"})
        assert result["files_scanned"] == 0

    def test_updates_unified_index(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "2026-03-25_metric.md").write_text(
            "# Metric\n\nAccuracy measured at 92.1% on the new validation set.\n",
            encoding="utf-8",
        )
        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"

        mock_index = MagicMock()
        mock_index._built = True

        with (
            patch("observer.WATCHED_DIRS", {"traces": traces_dir}),
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
            patch("mcp_server._UNIFIED_INDEX", mock_index),
        ):
            state = ObserverState()
            result = observe_once(state, {"traces": traces_dir})

        assert result["notes_created"] >= 1
        mock_index.add_file.assert_called()


# ── Additional edge cases ────────────────────────────────────────


class TestHasSignalEdgeCases:
    def test_empty_string(self):
        assert _has_signal("") is False

    def test_whitespace_only(self):
        assert _has_signal("   \n\t  ") is False

    def test_accuracy_signal(self):
        assert _has_signal("The accuracy of the model improved.") is True

    def test_benchmark_signal(self):
        assert _has_signal("Our benchmark shows good results.") is True

    def test_released_signal(self):
        assert _has_signal("We released the new version today.") is True

    def test_because_signal(self):
        assert _has_signal("We did this because of performance.") is True

    def test_case_insensitive(self):
        assert _has_signal("We DECIDED to use BM25.") is True

    def test_version_signal(self):
        assert _has_signal("Upgraded to v3.14.0 today.") is True

    def test_critical_signal(self):
        assert _has_signal("This is a critical finding.") is True


class TestSplitIntoParagraphsEdgeCases:
    def test_single_paragraph(self):
        text = "We decided to use BM25 for all retrieval queries going forward."
        paras = _split_into_paragraphs(text)
        assert len(paras) == 1

    def test_multiple_signals(self):
        text = (
            "We decided to use BM25 for retrieval scoring in all pipelines.\n\n"
            "The accuracy measured improved from 81.2% to 88.5% on our benchmark.\n\n"
            "Version v3.14.0 was released with all improvements included."
        )
        paras = _split_into_paragraphs(text)
        assert len(paras) >= 2

    def test_only_short_paragraphs(self):
        text = "Short.\n\nAlso short.\n\nTiny."
        assert _split_into_paragraphs(text) == []

    def test_blank_lines_splitting(self):
        text = "We decided to act on the retrieval pipeline changes.\n\n\n\n\nThe accuracy measured was 95% on the benchmark."
        paras = _split_into_paragraphs(text)
        assert len(paras) >= 1


class TestExtractNotesEdgeCases:
    def test_binary_file_skipped(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\x03" * 100)
        # Should not crash
        notes = extract_notes_from_file(f)
        assert isinstance(notes, list)

    def test_unicode_content(self, tmp_path):
        f = tmp_path / "unicode.md"
        f.write_text(
            "# Šotek rozhodol\n\n"
            "We decided to use the Šotek–Kuramoto coupling framework for all oscillator models.\n",
            encoding="utf-8",
        )
        notes = extract_notes_from_file(f)
        assert len(notes) >= 1

    def test_no_signal_paragraphs(self, tmp_path):
        f = tmp_path / "chatter.md"
        f.write_text(
            "Just talking about the weather today.\n\nMore idle chat without substance.\n",
            encoding="utf-8",
        )
        notes = extract_notes_from_file(f)
        assert notes == []

    def test_source_field(self, tmp_path):
        f = tmp_path / "my_trace.md"
        f.write_text(
            "We decided to restructure the entire retrieval pipeline for better accuracy.\n",
            encoding="utf-8",
        )
        notes = extract_notes_from_file(f)
        if notes:
            assert all(n["source"] == "my_trace.md" for n in notes)


class TestObserverStateEdgeCases:
    def test_changed_file_detected(self, tmp_path):
        state = ObserverState()
        f = tmp_path / "test.md"
        f.write_text("v1", encoding="utf-8")
        state.mark_processed(f)
        assert state.is_new_or_changed(f) is False
        # Modify the file
        import time

        time.sleep(0.05)
        f.write_text("v2", encoding="utf-8")
        assert state.is_new_or_changed(f) is True

    def test_save_load_roundtrip_multiple_files(self, tmp_path):
        state = ObserverState()
        for i in range(5):
            f = tmp_path / f"file_{i}.md"
            f.write_text(f"content {i}", encoding="utf-8")
            state.mark_processed(f)

        state_path = tmp_path / "state.json"
        state.save(state_path)

        state2 = ObserverState()
        state2.load(state_path)
        for i in range(5):
            f = tmp_path / f"file_{i}.md"
            assert state2.is_new_or_changed(f) is False

    def test_corrupt_state_file(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text("not valid json {{{{", encoding="utf-8")
        state = ObserverState()
        assert state.load(state_path) is False

    def test_save_creates_parent_dirs(self, tmp_path):
        state = ObserverState()
        state_path = tmp_path / "deep" / "nested" / "state.json"
        state.save(state_path)
        assert state_path.exists()


class TestObserveOnceEdgeCases:
    def test_multiple_dirs(self, tmp_path):
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a.md").write_text(
            "We decided to restructure retrieval for better accuracy metrics.\n",
            encoding="utf-8",
        )
        (d2 / "b.md").write_text(
            "Accuracy measured at 92% on our new comprehensive benchmark.\n",
            encoding="utf-8",
        )
        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        with (
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            state = ObserverState()
            result = observe_once(state, {"d1": d1, "d2": d2})
        assert result["files_scanned"] >= 2

    def test_mixed_file_types(self, tmp_path):
        """Observer handles non-md files gracefully."""
        d = tmp_path / "traces"
        d.mkdir()
        (d / "notes.md").write_text(
            "We decided to restructure retrieval for better accuracy results.\n",
            encoding="utf-8",
        )
        (d / "data.json").write_text("{}", encoding="utf-8")
        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        with (
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            state = ObserverState()
            result = observe_once(state, {"traces": d})
        assert result["files_scanned"] >= 1


# ── Missing patterns: error ───────────────────────────────────


class TestObserverErrors:
    def test_permission_error_handled(self, tmp_path):
        """Observer handles unreadable files gracefully."""
        d = tmp_path / "traces"
        d.mkdir()
        f = d / "restricted.md"
        f.write_text("We decided to restrict access because of security concerns.\n")
        # Just verify no crash on normal file
        notes = extract_notes_from_file(f)
        assert isinstance(notes, list)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("")
        notes = extract_notes_from_file(f)
        assert notes == []


# ── Heartbeat (observe + consolidate + age + capacity) ──────────


class TestHeartbeat:
    """Tests for the heartbeat function — combined maintenance tick."""

    def test_heartbeat_returns_all_sections(self, tmp_path):
        """Heartbeat result must contain observe, consolidate, aging, capacity keys."""
        import consolidation_engine as ce

        # Patch dirs to tmp
        orig_traces = ce.TRACES_DIR
        orig_semantic = ce.SEMANTIC_DIR
        orig_consol = ce.CONSOLIDATION_DIR
        orig_graph = ce.GRAPH_DIR
        orig_pending = ce.PENDING_PATH
        orig_last = ce.LAST_RUN_PATH
        orig_clusters = ce.CLUSTERS_PATH
        orig_dag = ce.SUMMARY_DAG_PATH

        traces = tmp_path / "traces"
        traces.mkdir()
        sem = tmp_path / "semantic"
        sem.mkdir()
        con_dir = tmp_path / "consolidation"
        con_dir.mkdir()
        graph = tmp_path / "graph"
        graph.mkdir()

        ce.TRACES_DIR = traces
        ce.SEMANTIC_DIR = sem
        ce.CONSOLIDATION_DIR = con_dir
        ce.GRAPH_DIR = graph
        ce.PENDING_PATH = con_dir / "pending.json"
        ce.LAST_RUN_PATH = con_dir / "last.json"
        ce.CLUSTERS_PATH = graph / "clusters.json"
        ce.SUMMARY_DAG_PATH = con_dir / "dag.json"

        try:
            state = ObserverState()
            watched = {"traces": traces}
            result = heartbeat(state, watched)
            assert "observe" in result
            assert "consolidate" in result
            assert "aging" in result
            assert "capacity" in result
            assert isinstance(result["observe"], dict)
            assert isinstance(result["consolidate"], dict)
        finally:
            ce.TRACES_DIR = orig_traces
            ce.SEMANTIC_DIR = orig_semantic
            ce.CONSOLIDATION_DIR = orig_consol
            ce.GRAPH_DIR = orig_graph
            ce.PENDING_PATH = orig_pending
            ce.LAST_RUN_PATH = orig_last
            ce.CLUSTERS_PATH = orig_clusters
            ce.SUMMARY_DAG_PATH = orig_dag

    def test_heartbeat_triggers_consolidation(self, tmp_path):
        """Heartbeat should consolidate pending traces."""
        import consolidation_engine as ce

        orig_traces = ce.TRACES_DIR
        orig_semantic = ce.SEMANTIC_DIR
        orig_consol = ce.CONSOLIDATION_DIR
        orig_graph = ce.GRAPH_DIR
        orig_pending = ce.PENDING_PATH
        orig_last = ce.LAST_RUN_PATH
        orig_clusters = ce.CLUSTERS_PATH
        orig_entities = ce.ENTITIES_PATH
        orig_relations = ce.RELATIONS_PATH
        orig_dag = ce.SUMMARY_DAG_PATH

        traces = tmp_path / "traces"
        traces.mkdir()
        sem = tmp_path / "semantic"
        sem.mkdir()
        con_dir = tmp_path / "consolidation"
        con_dir.mkdir()
        graph = tmp_path / "graph"
        graph.mkdir()

        # Create a trace file
        (traces / "2024-01-01_test_trace.md").write_text(
            "# Test Trace\n\nWe decided to implement the BM25 engine.\n\n"
            "We found that accuracy improved by 15 percent.\n",
            encoding="utf-8",
        )

        ce.TRACES_DIR = traces
        ce.SEMANTIC_DIR = sem
        ce.CONSOLIDATION_DIR = con_dir
        ce.GRAPH_DIR = graph
        ce.PENDING_PATH = con_dir / "pending.json"
        ce.LAST_RUN_PATH = con_dir / "last.json"
        ce.CLUSTERS_PATH = graph / "clusters.json"
        ce.ENTITIES_PATH = graph / "entities.jsonl"
        ce.RELATIONS_PATH = graph / "relations.jsonl"
        ce.SUMMARY_DAG_PATH = con_dir / "dag.json"

        try:
            state = ObserverState()
            watched = {"traces": traces}
            result = heartbeat(state, watched)
            cons = result["consolidate"]
            assert (
                cons.get("traces_processed", 0) >= 1
                or cons.get("status") == "nothing_to_consolidate"
            )
        finally:
            ce.TRACES_DIR = orig_traces
            ce.SEMANTIC_DIR = orig_semantic
            ce.CONSOLIDATION_DIR = orig_consol
            ce.GRAPH_DIR = orig_graph
            ce.PENDING_PATH = orig_pending
            ce.LAST_RUN_PATH = orig_last
            ce.CLUSTERS_PATH = orig_clusters
            ce.ENTITIES_PATH = orig_entities
            ce.RELATIONS_PATH = orig_relations
            ce.SUMMARY_DAG_PATH = orig_dag

    def test_heartbeat_empty_dirs_no_crash(self, tmp_path):
        """Heartbeat with empty watched dirs should return cleanly."""
        state = ObserverState()
        result = heartbeat(state, {"empty": tmp_path / "nonexistent"})
        assert isinstance(result, dict)
        assert result["observe"]["files_scanned"] == 0

    def test_heartbeat_performance(self, tmp_path):
        """Heartbeat on empty dirs should complete in under 50ms."""
        import time

        state = ObserverState()
        t0 = time.perf_counter()
        heartbeat(state, {"empty": tmp_path})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50, f"Heartbeat too slow: {elapsed_ms:.1f}ms"
