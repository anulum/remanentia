# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for observer.py

from __future__ import annotations

from unittest.mock import patch


from observer import (
    ObserverState,
    _has_signal,
    _split_into_paragraphs,
    extract_notes_from_file,
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
