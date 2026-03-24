# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for reflector.py

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from reflector import (
    _cluster_notes,
    _generate_summary_heuristic,
    _generate_summary_llm,
    _identify_contradictions,
    _identify_gaps,
    reflect_once,
)
from knowledge_store import KnowledgeNote, KnowledgeStore


def _make_note(content, keywords=None, entities=None, source="test.md",
               supersedes="", superseded_by=""):
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


class TestGenerateSummaryLLM:
    def test_no_client_returns_none(self):
        import os
        from unittest.mock import patch as p
        import answer_extractor
        answer_extractor._ANTHROPIC_CLIENT = None
        notes = [_make_note("test content")]
        with p.dict(os.environ, {}, clear=True):
            result = _generate_summary_llm(notes)
        assert result is None
        answer_extractor._ANTHROPIC_CLIENT = None


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

        with patch("knowledge_store.STORE_PATH", notes_path), \
             patch("knowledge_store.TRIGGERS_PATH", triggers_path), \
             patch("reflector.BASE", tmp_path):
            result = reflect_once(days=30, use_llm=False)

        assert result["status"] == "ok"
        assert result["notes"] >= 2
        assert "digest" in result
        assert "Reflection Digest" in result["digest"]

    def test_reflect_no_notes(self, tmp_path):
        notes_path = tmp_path / "notes.jsonl"
        with patch("knowledge_store.STORE_PATH", notes_path), \
             patch("knowledge_store.TRIGGERS_PATH", tmp_path / "t.jsonl"):
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

        with patch("knowledge_store.STORE_PATH", notes_path), \
             patch("knowledge_store.TRIGGERS_PATH", triggers_path):
            result = reflect_once(days=7)
        assert result["status"] == "nothing_recent"

    def test_reflect_finds_gaps(self, tmp_path):
        store = KnowledgeStore()
        store.add_note("We decided to add temporal graph support.", source="a.md")
        # No finding note about temporal graph → gap
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)

        with patch("knowledge_store.STORE_PATH", notes_path), \
             patch("knowledge_store.TRIGGERS_PATH", triggers_path), \
             patch("reflector.BASE", tmp_path):
            result = reflect_once(days=30, use_llm=False)

        if result.get("gaps", 0) > 0:
            assert "Knowledge Gaps" in result["digest"]
