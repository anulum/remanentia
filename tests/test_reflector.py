# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for reflector

from __future__ import annotations

import builtins
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import answer_extractor
from answer_extractor import LLMBackend
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
    content: str,
    keywords: list[str] | None = None,
    entities: list[str] | None = None,
    source: str = "test.md",
    supersedes: str = "",
    superseded_by: str = "",
) -> KnowledgeNote:
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
    def test_clusters_by_shared_keywords(self) -> None:
        notes = [
            _make_note("BM25 retrieval", keywords=["bm25", "retrieval", "scoring"]),
            _make_note("BM25 scoring", keywords=["bm25", "scoring", "retrieval"]),
            _make_note("SNN daemon killed", keywords=["snn", "daemon", "killed"]),
        ]
        clusters = _cluster_notes(notes)
        assert len(clusters) >= 1

    def test_no_clusters_unrelated(self) -> None:
        notes = [
            _make_note("alpha", keywords=["alpha"]),
            _make_note("beta", keywords=["beta"]),
        ]
        clusters = _cluster_notes(notes)
        assert len(clusters) == 0

    def test_empty(self) -> None:
        assert _cluster_notes([]) == []

    def test_python_cluster_fallback_without_native_extension(self) -> None:
        original_import = builtins.__import__

        def import_without_native(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name == "remanentia_consolidation":
                raise ImportError(name)
            return original_import(name, globals, locals, fromlist, level)

        notes = [
            _make_note("first", keywords=["bm25", "retrieval"], entities=["remanentia"]),
            _make_note("second", keywords=["bm25", "retrieval"], entities=["index"]),
            _make_note("third", keywords=["temporal"], entities=["graph"]),
        ]

        with patch("builtins.__import__", side_effect=import_without_native):
            clusters = _cluster_notes(notes)

        assert clusters == [[0, 1]]

    def test_clusters_by_entities(self) -> None:
        notes = [
            _make_note("LOCOMO score", entities=["locomo", "81.2%", "bm25"]),
            _make_note("LOCOMO benchmark", entities=["locomo", "bm25", "retrieval"]),
        ]
        clusters = _cluster_notes(notes)
        assert len(clusters) >= 1


class TestGenerateSummaryHeuristic:
    def test_produces_summary(self) -> None:
        notes = [
            _make_note("BM25 retrieval accuracy", entities=["bm25"]),
            _make_note("Cross-encoder reranking", entities=["bm25"]),
        ]
        summary = _generate_summary_heuristic(notes)
        assert "2 related notes" in summary
        assert "bm25" in summary.lower()

    def test_empty(self) -> None:
        assert _generate_summary_heuristic([]) == ""


class _FakeBackend:
    def __init__(self, response: str | None) -> None:
        self._response = response

    def complete(self, prompt: str, *, max_tokens: int = 200, system: str = "") -> str | None:
        return self._response


class TestGenerateSummaryLLM:
    def setup_method(self) -> None:
        self._orig = cast(LLMBackend | None, answer_extractor._BACKEND)
        answer_extractor._BACKEND = None

    def teardown_method(self) -> None:
        answer_extractor._BACKEND = self._orig

    def test_no_backend_returns_none(self) -> None:
        notes = [_make_note("test content")]
        result = _generate_summary_llm(notes)
        assert result is None

    def test_with_backend_returns_summary(self) -> None:
        answer_extractor._BACKEND = _FakeBackend("Dense summary of 1 note.")
        notes = [_make_note("test content")]
        result = _generate_summary_llm(notes)
        assert result == "Dense summary of 1 note."

    def test_backend_none_response(self) -> None:
        answer_extractor._BACKEND = _FakeBackend(None)
        notes = [_make_note("test content")]
        result = _generate_summary_llm(notes)
        assert result is None

    def test_backend_exception(self) -> None:
        class _ErrorBackend:
            def complete(self, prompt: str, **kwargs: object) -> str:
                raise RuntimeError("fail")

        answer_extractor._BACKEND = _ErrorBackend()
        notes = [_make_note("test content")]
        result = _generate_summary_llm(notes)
        assert result is None


class TestGenerateProspectiveQueriesLLM:
    def setup_method(self) -> None:
        self._orig = cast(LLMBackend | None, answer_extractor._BACKEND)
        answer_extractor._BACKEND = None

    def teardown_method(self) -> None:
        answer_extractor._BACKEND = self._orig

    def test_no_backend_returns_empty(self) -> None:
        note = _make_note("test content about hiking")
        result = _generate_prospective_queries_llm(note)
        assert result == []

    def test_with_backend_returns_queries(self) -> None:
        answer_extractor._BACKEND = _FakeBackend(
            "What are the hiking trails?\nDoes the user like mountains?"
        )
        note = _make_note("User went hiking in the Alps last weekend")
        result = _generate_prospective_queries_llm(note)
        assert len(result) == 2

    def test_backend_none_response(self) -> None:
        answer_extractor._BACKEND = _FakeBackend(None)
        note = _make_note("test content")
        result = _generate_prospective_queries_llm(note)
        assert result == []

    def test_filters_short_queries(self) -> None:
        answer_extractor._BACKEND = _FakeBackend("What about hiking trails?\nOk\nShort")
        note = _make_note("test content")
        result = _generate_prospective_queries_llm(note)
        assert len(result) == 1  # only first one > 5 chars

    def test_backend_exception(self) -> None:
        class _ErrorBackend:
            def complete(self, prompt: str, **kwargs: object) -> str:
                raise RuntimeError("fail")

        answer_extractor._BACKEND = _ErrorBackend()
        note = _make_note("test content")
        result = _generate_prospective_queries_llm(note)
        assert result == []


class TestIdentifyGaps:
    def test_finds_unmeasured_decisions(self) -> None:
        notes = [
            _make_note("We decided to use BM25 for all queries", entities=["bm25"]),
            _make_note("We decided to add cross-encoder reranking", entities=["cross-encoder"]),
            _make_note("BM25 accuracy measured at 81.2%", entities=["bm25"]),
        ]
        gaps = _identify_gaps(notes)
        assert any("cross-encoder" in g for g in gaps)

    def test_no_gaps(self) -> None:
        notes = [
            _make_note("We decided to use BM25", entities=["bm25"]),
            _make_note("BM25 accuracy measured at 81.2%", entities=["bm25"]),
        ]
        gaps = _identify_gaps(notes)
        assert len(gaps) == 0

    def test_empty(self) -> None:
        assert _identify_gaps([]) == []


class TestIdentifyContradictions:
    def test_finds_supersession(self) -> None:
        notes = [
            _make_note("SNN started", supersedes="old_note"),
        ]
        contradictions = _identify_contradictions(notes)
        assert len(contradictions) == 1

    def test_ignores_resolved(self) -> None:
        notes = [
            _make_note("SNN started", supersedes="old", superseded_by="newer"),
        ]
        contradictions = _identify_contradictions(notes)
        assert len(contradictions) == 0


class TestReflectOnce:
    def test_reflect_with_notes(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        store.add_note("We decided to remove SNN from retrieval scoring.", source="a.md")
        store.add_note("BM25 retrieval accuracy measured at 81.2% on LOCOMO.", source="b.md")
        store.add_note("BM25 LOCOMO score improved to 83.1% with LLM synthesis.", source="c.md")
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)
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

    def test_reflect_no_notes(self, tmp_path: Path) -> None:
        notes_path = tmp_path / "notes.jsonl"
        with (
            patch("knowledge_store.STORE_PATH", notes_path),
            patch("knowledge_store.TRIGGERS_PATH", tmp_path / "t.jsonl"),
        ):
            result = reflect_once(days=7)
        assert result["status"] == "no_notes"

    def test_reflect_nothing_recent(self, tmp_path: Path) -> None:
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

    def test_reflect_finds_gaps(self, tmp_path: Path) -> None:
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
    def test_reflect_once_with_store(self, tmp_path: Path) -> None:
        """Reflector processes knowledge store notes end-to-end."""
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
        assert result["status"] == "ok"
        assert result["notes"] == 1
        assert Path(result["digest_path"]).exists()

    def test_reflect_empty_store(self, tmp_path: Path) -> None:
        with (
            patch("knowledge_store.STORE_PATH", tmp_path / "missing-notes.jsonl"),
            patch("knowledge_store.TRIGGERS_PATH", tmp_path / "missing-triggers.jsonl"),
        ):
            result = reflect_once(days=1, use_llm=False)
        assert result == {"status": "no_notes", "notes": 0}
