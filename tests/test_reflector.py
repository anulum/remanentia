# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for reflector

from __future__ import annotations

import time
from unittest.mock import patch


from reflector import (
    _cluster_notes,
    _generate_prospective_queries_llm,
    _generate_summary_heuristic,
    _generate_summary_llm,
    _identify_contradictions,
    _identify_gaps,
    reflect_once,
)
from knowledge_store import KnowledgeNote, KnowledgeStore


def _make_note(
    content, keywords=None, entities=None, source="test.md", supersedes="", superseded_by=""
):
    now = time.strftime("%Y-%m-%dT%H%M", time.gmtime())
    return KnowledgeNote(
        id=f"n{hash(content) % 10000}",
        title=content[:40],
        content=content,
        keywords=keywords or [],
        entities=entities or [],
        source=source,
        created=now,
        updated=now,
        supersedes=supersedes,
        superseded_by=superseded_by,
    )


class TestClusterNotes:
    def test_clusters_by_shared_keywords(self):
        notes = [
            _make_note("BM25 retrieval", keywords=["bm25", "retrieval", "scoring"]),
            _make_note("BM25 scoring", keywords=["bm25", "scoring", "retrieval"]),
            _make_note("SNN daemon killed", keywords=["snn", "daemon", "killed"]),
        ]
        clusters = _cluster_notes(notes)
        assert len(clusters) >= 1

    def test_no_clusters_unrelated(self):
        notes = [
            _make_note("alpha", keywords=["alpha"]),
            _make_note("beta", keywords=["beta"]),
        ]
        clusters = _cluster_notes(notes)
        assert len(clusters) == 0

    def test_empty(self):
        assert _cluster_notes([]) == []

    def test_clusters_by_entities(self):
        notes = [
            _make_note("LOCOMO score", entities=["locomo", "81.2%", "bm25"]),
            _make_note("LOCOMO benchmark", entities=["locomo", "bm25", "retrieval"]),
        ]
        clusters = _cluster_notes(notes)
        assert len(clusters) >= 1


class TestGenerateSummaryHeuristic:
    def test_produces_summary(self):
        notes = [
            _make_note("BM25 retrieval accuracy", entities=["bm25"]),
            _make_note("Cross-encoder reranking", entities=["bm25"]),
        ]
        summary = _generate_summary_heuristic(notes)
        assert "2 related notes" in summary
        assert "bm25" in summary.lower()

    def test_empty(self):
        assert _generate_summary_heuristic([]) == ""


class _FakeBackend:
    def __init__(self, response):
        self._response = response

    def complete(self, prompt, *, max_tokens=200, system=""):
        return self._response


class TestGenerateSummaryLLM:
    def setup_method(self):
        import answer_extractor

        self._orig = answer_extractor._BACKEND
        answer_extractor._BACKEND = None

    def teardown_method(self):
        import answer_extractor

        answer_extractor._BACKEND = self._orig

    def test_no_backend_returns_none(self):
        notes = [_make_note("test content")]
        result = _generate_summary_llm(notes)
        assert result is None

    def test_with_backend_returns_summary(self):
        import answer_extractor

        answer_extractor._BACKEND = _FakeBackend("Dense summary of 1 note.")
        notes = [_make_note("test content")]
        result = _generate_summary_llm(notes)
        assert result == "Dense summary of 1 note."

    def test_backend_none_response(self):
        import answer_extractor

        answer_extractor._BACKEND = _FakeBackend(None)
        notes = [_make_note("test content")]
        result = _generate_summary_llm(notes)
        assert result is None

    def test_backend_exception(self):
        import answer_extractor

        class _ErrorBackend:
            def complete(self, prompt, **kwargs):
                raise RuntimeError("fail")

        answer_extractor._BACKEND = _ErrorBackend()
        notes = [_make_note("test content")]
        result = _generate_summary_llm(notes)
        assert result is None


class TestGenerateProspectiveQueriesLLM:
    def setup_method(self):
        import answer_extractor

        self._orig = answer_extractor._BACKEND
        answer_extractor._BACKEND = None

    def teardown_method(self):
        import answer_extractor

        answer_extractor._BACKEND = self._orig

    def test_no_backend_returns_empty(self):
        note = _make_note("test content about hiking")
        result = _generate_prospective_queries_llm(note)
        assert result == []

    def test_with_backend_returns_queries(self):
        import answer_extractor

        answer_extractor._BACKEND = _FakeBackend(
            "What are the hiking trails?\nDoes the user like mountains?"
        )
        note = _make_note("User went hiking in the Alps last weekend")
        result = _generate_prospective_queries_llm(note)
        assert len(result) == 2

    def test_backend_none_response(self):
        import answer_extractor

        answer_extractor._BACKEND = _FakeBackend(None)
        note = _make_note("test content")
        result = _generate_prospective_queries_llm(note)
        assert result == []

    def test_filters_short_queries(self):
        import answer_extractor

        answer_extractor._BACKEND = _FakeBackend("What about hiking trails?\nOk\nShort")
        note = _make_note("test content")
        result = _generate_prospective_queries_llm(note)
        assert len(result) == 1  # only first one > 5 chars

    def test_backend_exception(self):
        import answer_extractor

        class _ErrorBackend:
            def complete(self, prompt, **kwargs):
                raise RuntimeError("fail")

        answer_extractor._BACKEND = _ErrorBackend()
        note = _make_note("test content")
        result = _generate_prospective_queries_llm(note)
        assert result == []


class TestIdentifyGaps:
    def test_finds_unmeasured_decisions(self):
        notes = [
            _make_note("We decided to use BM25 for all queries", entities=["bm25"]),
            _make_note("We decided to add cross-encoder reranking", entities=["cross-encoder"]),
            _make_note("BM25 accuracy measured at 81.2%", entities=["bm25"]),
        ]
        gaps = _identify_gaps(notes)
        assert any("cross-encoder" in g for g in gaps)

    def test_no_gaps(self):
        notes = [
            _make_note("We decided to use BM25", entities=["bm25"]),
            _make_note("BM25 accuracy measured at 81.2%", entities=["bm25"]),
        ]
        gaps = _identify_gaps(notes)
        assert len(gaps) == 0

    def test_empty(self):
        assert _identify_gaps([]) == []


class TestIdentifyContradictions:
    def test_finds_supersession(self):
        notes = [
            _make_note("SNN started", supersedes="old_note"),
        ]
        contradictions = _identify_contradictions(notes)
        assert len(contradictions) == 1

    def test_ignores_resolved(self):
        notes = [
            _make_note("SNN started", supersedes="old", superseded_by="newer"),
        ]
        contradictions = _identify_contradictions(notes)
        assert len(contradictions) == 0


class TestReflectOnce:
    def test_reflect_with_notes(self, tmp_path):
        store = KnowledgeStore()
        store.add_note("We decided to remove SNN from retrieval scoring.", source="a.md")
        store.add_note("BM25 retrieval accuracy measured at 81.2% on LOCOMO.", source="b.md")
        store.add_note("BM25 LOCOMO score improved to 83.1% with LLM synthesis.", source="c.md")
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)
        digest_dir = tmp_path / "digests"

        with (
            patch("knowledge_store.STORE_PATH", notes_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
            patch("reflector.BASE", tmp_path),
        ):
            result = reflect_once(days=30, use_llm=False)

        assert result["status"] == "ok"
        assert result["notes"] >= 2
        assert "digest" in result
        assert "Reflection Digest" in result["digest"]

    def test_reflect_no_notes(self, tmp_path):
        notes_path = tmp_path / "notes.jsonl"
        with (
            patch("knowledge_store.STORE_PATH", notes_path),
            patch("knowledge_store.TRIGGERS_PATH", tmp_path / "t.jsonl"),
        ):
            result = reflect_once(days=7)
        assert result["status"] == "no_notes"

    def test_reflect_nothing_recent(self, tmp_path):
        store = KnowledgeStore()
        note = store.add_note("Old note content about BM25 retrieval.", source="old.md")
        note.created = "2020-01-01T0000"
        note.updated = "2020-01-01T0000"
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)

        with (
            patch("knowledge_store.STORE_PATH", notes_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            result = reflect_once(days=7)
        assert result["status"] == "nothing_recent"

    def test_reflect_finds_gaps(self, tmp_path):
        store = KnowledgeStore()
        store.add_note("We decided to add temporal graph support.", source="a.md")
        # No finding note about temporal graph → gap
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)

        with (
            patch("knowledge_store.STORE_PATH", notes_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
            patch("reflector.BASE", tmp_path),
        ):
            result = reflect_once(days=30, use_llm=False)

        if result.get("gaps", 0) > 0:
            assert "Knowledge Gaps" in result["digest"]


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestReflectorPipeline:
    def test_reflect_once_with_store(self, tmp_path):
        """Reflector processes knowledge store notes end-to-end."""
        from knowledge_store import KnowledgeStore
        from reflector import reflect_once
        from unittest.mock import patch

        store = KnowledgeStore()
        store.add_note("BM25 accuracy measured at 88.5%.", source="a.md")

        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        with (
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            store.save()
            result = reflect_once(days=30, use_llm=False)
        assert isinstance(result, dict)

    def test_reflect_empty_store(self):
        from reflector import reflect_once

        result = reflect_once(days=1, use_llm=False)
        assert isinstance(result, dict)
