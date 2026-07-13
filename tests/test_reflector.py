# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for reflector

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import answer_extractor
from llm_backend import LocalLLMBackend
from reflector import (
    _cluster_notes,
    _cluster_notes_python,
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
        assert _cluster_notes_python([]) == []

    def test_python_cluster_fallback_without_native_extension(self) -> None:
        notes = [
            _make_note("first", keywords=["bm25", "retrieval"], entities=["remanentia"]),
            _make_note("second", keywords=["bm25", "retrieval"], entities=["index"]),
            _make_note("third", keywords=["temporal"], entities=["graph"]),
        ]

        assert _cluster_notes_python(notes) == [[0, 1]]

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

    def test_summary_without_entities_omits_entity_line(self) -> None:
        summary = _generate_summary_heuristic([_make_note("Plain note")])
        assert "Entities:" not in summary


class _CompletionServer(ThreadingHTTPServer):
    responses: list[str | None]
    requests: list[dict[str, Any]]


class _CompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = cast(_CompletionServer, self.server)
        length = int(self.headers.get("Content-Length", "0"))
        server.requests.append(cast(dict[str, Any], json.loads(self.rfile.read(length))))
        response = server.responses.pop(0)
        body = json.dumps({"choices": [{"message": {"content": response}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


@contextmanager
def _local_backend(*responses: str | None) -> Iterator[tuple[LocalLLMBackend, _CompletionServer]]:
    server = _CompletionServer(("127.0.0.1", 0), _CompletionHandler)
    server.responses = list(responses)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    backend = LocalLLMBackend(base_url=f"http://{host}:{port}/v1", model="test", timeout=2)
    try:
        yield backend, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _closed_backend() -> LocalLLMBackend:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()
    probe.close()
    return LocalLLMBackend(base_url=f"http://{host}:{port}/v1", timeout=0.1)


@contextmanager
def _configured(backend: LocalLLMBackend | None) -> Iterator[None]:
    original = answer_extractor.get_llm_backend()
    answer_extractor.set_llm_backend(backend)
    try:
        yield
    finally:
        answer_extractor.set_llm_backend(original)


class TestGenerateSummaryLLM:
    def test_no_backend_returns_none(self) -> None:
        notes = [_make_note("test content")]
        with _configured(None):
            assert _generate_summary_llm(notes) is None

    def test_summary_crosses_real_http(self) -> None:
        notes = [_make_note("test content")]
        with _local_backend("Dense summary of 1 note.") as (backend, server), _configured(backend):
            result = _generate_summary_llm(notes)
        assert result == "Dense summary of 1 note."
        assert server.requests[0]["max_tokens"] == 200

    def test_missing_content_and_connection_refusal_return_none(self) -> None:
        notes = [_make_note("test content")]
        with _local_backend(None) as (backend, _server), _configured(backend):
            assert _generate_summary_llm(notes) is None
        with _configured(_closed_backend()):
            assert _generate_summary_llm(notes) is None


class TestGenerateProspectiveQueriesLLM:
    def test_no_backend_returns_empty(self) -> None:
        note = _make_note("test content about hiking")
        with _configured(None):
            assert _generate_prospective_queries_llm(note) == []

    def test_queries_cross_real_http_and_filter_short_lines(self) -> None:
        note = _make_note("User went hiking in the Alps last weekend")
        response = "What are the hiking trails?\nDoes the user like mountains?\nOk\nShort"
        with _local_backend(response) as (backend, server), _configured(backend):
            result = _generate_prospective_queries_llm(note)
        assert len(result) == 2
        assert server.requests[0]["max_tokens"] == 150

    def test_missing_content_and_connection_refusal_return_empty(self) -> None:
        note = _make_note("test content")
        with _local_backend(None) as (backend, _server), _configured(backend):
            assert _generate_prospective_queries_llm(note) == []
        with _configured(_closed_backend()):
            assert _generate_prospective_queries_llm(note) == []


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
    @staticmethod
    def _clustered_store(tmp_path: Path) -> tuple[Path, Path]:
        store = KnowledgeStore()
        first = store.add_note("BM25 retrieval measured at 81 percent.", source="a.md")
        second = store.add_note("BM25 retrieval improved after tuning.", source="b.md")
        first.keywords = ["bm25", "retrieval"]
        second.keywords = ["bm25", "retrieval"]
        first.entities = []
        second.entities = []
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)
        return notes_path, triggers_path

    def test_reflect_with_notes(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        store.add_note("We decided to remove SNN from retrieval scoring.", source="a.md")
        store.add_note("BM25 retrieval accuracy measured at 81.2% on LOCOMO.", source="b.md")
        store.add_note("BM25 LOCOMO score improved to 83.1% with LLM synthesis.", source="c.md")
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)
        result = reflect_once(
            days=30,
            use_llm=False,
            notes_path=notes_path,
            triggers_path=triggers_path,
            digest_dir=tmp_path / "digests",
        )

        assert result["status"] == "ok"
        assert result["notes"] >= 2
        assert "digest" in result
        assert "Reflection Digest" in result["digest"]

    def test_reflect_no_notes(self, tmp_path: Path) -> None:
        notes_path = tmp_path / "notes.jsonl"
        result = reflect_once(
            days=7,
            notes_path=notes_path,
            triggers_path=tmp_path / "t.jsonl",
            digest_dir=tmp_path / "digests",
        )
        assert result["status"] == "no_notes"

    def test_reflect_nothing_recent(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        note = store.add_note("Old note content about BM25 retrieval.", source="old.md")
        note.created = "2020-01-01T0000"
        note.updated = "2020-01-01T0000"
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)

        result = reflect_once(
            days=7,
            notes_path=notes_path,
            triggers_path=triggers_path,
            digest_dir=tmp_path / "digests",
        )
        assert result["status"] == "nothing_recent"

    def test_reflect_finds_gaps(self, tmp_path: Path) -> None:
        store = KnowledgeStore()
        store.add_note("We decided to add temporal graph support.", source="a.md")
        # No finding note about temporal graph → gap
        notes_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(notes_path, triggers_path)

        result = reflect_once(
            days=30,
            use_llm=False,
            notes_path=notes_path,
            triggers_path=triggers_path,
            digest_dir=tmp_path / "digests",
        )

        if result.get("gaps", 0) > 0:
            assert "Knowledge Gaps" in result["digest"]

    def test_llm_reflection_crosses_http_and_persists_queries(self, tmp_path: Path) -> None:
        notes_path, triggers_path = self._clustered_store(tmp_path)
        responses = (
            "Dense BM25 summary.",
            "What is BM25 recall?\nWhen was retrieval measured?",
            "How did BM25 improve?\nWhat tuning was used?",
        )
        with _local_backend(*responses) as (backend, server), _configured(backend):
            result = reflect_once(
                days=30,
                use_llm=True,
                notes_path=notes_path,
                triggers_path=triggers_path,
                digest_dir=tmp_path / "digests",
            )

        assert result["status"] == "ok"
        assert result["prospective_queries"] == 4
        assert "Dense BM25 summary." in result["digest"]
        assert len(server.requests) == 3
        reloaded = KnowledgeStore()
        assert reloaded.load(notes_path, triggers_path)
        assert any("What is BM25 recall?" in note.keywords for note in reloaded.notes.values())

    def test_llm_mode_without_backend_uses_heuristic(self, tmp_path: Path) -> None:
        notes_path, triggers_path = self._clustered_store(tmp_path)
        with _configured(None):
            result = reflect_once(
                days=30,
                use_llm=True,
                notes_path=notes_path,
                triggers_path=triggers_path,
                digest_dir=tmp_path / "digests",
            )
        assert result["status"] == "ok"
        assert result["prospective_queries"] == 0
        assert "Summary of 2 related notes" in result["digest"]


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestReflectorPipeline:
    def test_reflect_once_with_store(self, tmp_path: Path) -> None:
        """Reflector processes knowledge store notes end-to-end."""
        store = KnowledgeStore()
        store.add_note("BM25 accuracy measured at 88.5%.", source="a.md")

        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        store.save(store_path, triggers_path)
        result = reflect_once(
            days=30,
            use_llm=False,
            notes_path=store_path,
            triggers_path=triggers_path,
            digest_dir=tmp_path / "digests",
        )
        assert result["status"] == "ok"
        assert result["notes"] == 1
        assert Path(result["digest_path"]).exists()

    def test_reflect_empty_store(self, tmp_path: Path) -> None:
        result = reflect_once(
            days=1,
            use_llm=False,
            notes_path=tmp_path / "missing-notes.jsonl",
            triggers_path=tmp_path / "missing-triggers.jsonl",
            digest_dir=tmp_path / "digests",
        )
        assert result == {"status": "no_notes", "notes": 0}
