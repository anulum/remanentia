# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for persistent vector index

from __future__ import annotations

import json
import threading
from hashlib import sha256
from io import StringIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from memory_index import Document, MemoryIndex
from vector_pipeline import (
    PublicVectorResultPolicy,
    VectorRefreshWorkerConfig,
    build_memory_vector_index,
    chunks_from_memory_index,
    public_vector_results,
    refresh_memory_vector_index,
    run_vector_refresh_worker,
    run_vector_cli,
    search_memory_vector_index,
    vector_index_status,
)
from vector_index import (
    HttpEmbeddingClient,
    PersistentVectorIndex,
    VectorChunk,
    VectorIndexError,
    VectorSearchResult,
)


class KeywordEmbeddingProvider:
    def embed_texts(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            rows.append(
                [
                    1.0 if "alpha" in lower else 0.0,
                    1.0 if "beta" in lower else 0.0,
                    1.0 if "gamma" in lower else 0.0,
                    0.25,
                ]
            )
        return np.asarray(rows, dtype=np.float32)


def _chunks() -> list[VectorChunk]:
    return [
        VectorChunk(
            chunk_id="a",
            text="alpha memory retrieval",
            source="trace",
            metadata={"project": "remanentia"},
        ),
        VectorChunk(
            chunk_id="b",
            text="beta factual verifier",
            source="director",
            metadata={"project": "director"},
        ),
        VectorChunk(
            chunk_id="c",
            text="gamma temporal entity graph",
            source="trace",
            metadata={"project": "remanentia"},
        ),
    ]


class TestPersistentVectorIndex:
    def test_build_persists_vectors_metadata_and_manifest(self, tmp_path: Path):
        index = PersistentVectorIndex(tmp_path / "vec")
        stats = index.build(_chunks(), KeywordEmbeddingProvider(), batch_size=2)

        assert stats.count == 3
        assert stats.dimension == 4
        assert stats.vector_bytes == 3 * 4 * 4
        assert stats.metadata_bytes > 0
        assert index.count() == 3
        assert (tmp_path / "vec" / "vectors.npz").exists()
        assert (tmp_path / "vec" / "chunks.sqlite").exists()

        manifest = json.loads((tmp_path / "vec" / "manifest.json").read_text())
        assert manifest["count"] == 3
        assert manifest["dimension"] == 4
        assert manifest["total_bytes"] >= stats.vector_bytes

    def test_search_returns_best_semantic_match(self, tmp_path: Path):
        index = PersistentVectorIndex(tmp_path / "vec")
        provider = KeywordEmbeddingProvider()
        index.build(_chunks(), provider)

        results = index.search("alpha question", provider, top_k=2)

        assert [r.chunk_id for r in results][:1] == ["a"]
        assert results[0].metadata == {"project": "remanentia"}
        assert 0.0 < results[0].score <= 1.0

    def test_search_filters_source(self, tmp_path: Path):
        index = PersistentVectorIndex(tmp_path / "vec")
        provider = KeywordEmbeddingProvider()
        index.build(_chunks(), provider)

        results = index.search("beta question", provider, top_k=3, source="trace")

        assert results
        assert {r.source for r in results} == {"trace"}
        assert "director" not in {r.source for r in results}

    def test_dimension_mismatch_is_rejected(self, tmp_path: Path):
        class WrongDimProvider:
            def embed_texts(self, texts):
                return np.ones((len(texts), 2), dtype=np.float32)

        index = PersistentVectorIndex(tmp_path / "vec")
        provider = KeywordEmbeddingProvider()
        index.build(_chunks(), provider)

        with pytest.raises(VectorIndexError, match="dimension"):
            index.search("alpha", WrongDimProvider())

    def test_duplicate_chunk_ids_are_rejected(self, tmp_path: Path):
        index = PersistentVectorIndex(tmp_path / "vec")
        chunks = [
            VectorChunk(chunk_id="same", text="alpha"),
            VectorChunk(chunk_id="same", text="beta"),
        ]

        with pytest.raises(VectorIndexError, match="duplicate"):
            index.build(chunks, KeywordEmbeddingProvider())

    def test_storage_estimate_scales_linearly(self):
        small = PersistentVectorIndex.estimate_storage(100, dimension=768)
        large = PersistentVectorIndex.estimate_storage(200_000, dimension=768)

        assert large.vector_bytes == small.vector_bytes * 2_000
        assert large.metadata_bytes == small.metadata_bytes * 2_000
        assert large.total_bytes > 700_000_000


class _EmbeddingHandler(BaseHTTPRequestHandler):
    def handle_embedding_post(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        texts = payload["input"]
        data: list[dict[str, Any]] = []
        for idx, text in enumerate(texts):
            data.append(
                {
                    "index": idx,
                    "embedding": [
                        float(len(text)),
                        float(text.count("a")),
                        1.0,
                    ],
                }
            )
        body = json.dumps({"data": data}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = handle_embedding_post

    def log_message(self, fmt, *args):
        return


@pytest.fixture
def embedding_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestHttpEmbeddingClient:
    def test_client_parses_embedding_response(self, embedding_server: str):
        client = HttpEmbeddingClient(base_url=embedding_server, model="local-test")

        vectors = client.embed_texts(["alpha", "beta"])

        assert vectors.shape == (2, 3)
        assert vectors.dtype == np.float32
        assert vectors[0, 0] == 5.0

    def test_client_rejects_empty_batch(self, embedding_server: str):
        client = HttpEmbeddingClient(base_url=embedding_server, model="local-test")

        with pytest.raises(VectorIndexError, match="empty"):
            client.embed_texts([])


class TestMemoryVectorPipeline:
    def test_chunks_from_memory_index_use_stable_metadata(self):
        memory = _built_memory_index()

        chunks = chunks_from_memory_index(memory)

        assert len(chunks) == 2
        assert chunks[0].source == "trace"
        assert chunks[0].metadata["document"] == "decision.md"
        assert chunks[0].metadata["paragraph_idx"] == 0
        assert chunks[0].chunk_id == chunks_from_memory_index(memory)[0].chunk_id

    def test_chunks_do_not_store_private_absolute_paths(self):
        memory = _built_memory_index()
        memory.documents[0].path = str(Path.cwd() / "reasoning_traces" / "decision.md")

        chunks = chunks_from_memory_index(memory)

        assert chunks[0].metadata["path"] == "reasoning_traces/decision.md"

    def test_chunks_skip_private_sources_by_default(self):
        memory = _built_memory_index()
        memory.documents.append(
            Document(
                name="internal.md",
                source="sessions",
                path=".coordination/sessions/internal.md",
                paragraphs=["gamma private operational note"],
                tokens={"gamma"},
                date="2026-04-27",
                doc_type="session",
            )
        )
        memory.paragraph_index.append((1, 0))

        public_chunks = chunks_from_memory_index(memory)
        private_chunks = chunks_from_memory_index(memory, include_private=True)

        assert [chunk.metadata["document"] for chunk in public_chunks] == ["decision.md"] * 2
        assert [chunk.metadata["document"] for chunk in private_chunks] == [
            "decision.md",
            "decision.md",
            "internal.md",
        ]

    def test_chunks_deduplicate_duplicate_paragraph_references(self):
        memory = _built_memory_index()
        memory.paragraph_index.append((0, 1))

        chunks = chunks_from_memory_index(memory)

        assert len(chunks) == 2
        assert [chunk.text for chunk in chunks] == [
            "alpha memory retrieval paragraph",
            "beta factual verifier paragraph",
        ]

    def test_chunks_cap_long_paragraphs_for_embedding_context(self):
        memory = _built_memory_index()
        memory.documents[0].paragraphs[0] = " ".join(f"token{i}" for i in range(600))

        chunks = chunks_from_memory_index(memory)

        assert len(chunks[0].text.split()) < 600
        assert len(chunks[0].text.split()) <= 96
        assert len(chunks[0].text) <= 600

    def test_build_and_search_memory_vector_index(self, tmp_path: Path):
        memory = _built_memory_index()
        provider = KeywordEmbeddingProvider()

        stats = build_memory_vector_index(memory, tmp_path / "memory_vec", provider)
        results = search_memory_vector_index(
            tmp_path / "memory_vec",
            "beta verifier",
            provider,
            top_k=1,
        )

        assert stats.count == 2
        assert results[0].text == "beta factual verifier paragraph"

    def test_build_memory_vector_index_tolerates_duplicate_paragraph_references(
        self, tmp_path: Path
    ):
        memory = _built_memory_index()
        memory.paragraph_index.append((0, 1))
        provider = KeywordEmbeddingProvider()

        stats = build_memory_vector_index(memory, tmp_path / "memory_vec", provider)

        assert stats.count == 2

    def test_refresh_rebuilds_once_then_skips_unchanged_corpus(self, tmp_path: Path):
        memory = _built_memory_index()
        provider = KeywordEmbeddingProvider()
        index_dir = tmp_path / "memory_vec"

        first = refresh_memory_vector_index(memory, index_dir, provider)
        second = refresh_memory_vector_index(memory, index_dir, provider)

        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        assert first["action"] == "rebuilt"
        assert second["action"] == "skipped"
        assert first["corpus_fingerprint"] == second["corpus_fingerprint"]
        assert manifest["corpus_fingerprint"] == first["corpus_fingerprint"]
        assert manifest["corpus_chunk_count"] == 2

    def test_refresh_force_rebuilds_unchanged_corpus(self, tmp_path: Path):
        memory = _built_memory_index()
        provider = KeywordEmbeddingProvider()
        index_dir = tmp_path / "memory_vec"

        refresh_memory_vector_index(memory, index_dir, provider)
        forced = refresh_memory_vector_index(memory, index_dir, provider, force=True)

        assert forced["action"] == "rebuilt"
        assert forced["changed"] is True

    def test_refresh_worker_writes_heartbeat(self, tmp_path: Path):
        memory = _built_memory_index()
        provider = KeywordEmbeddingProvider()
        heartbeat = tmp_path / "worker.json"
        output = StringIO()
        config = VectorRefreshWorkerConfig(
            index_dir=tmp_path / "memory_vec",
            heartbeat_path=heartbeat,
            max_cycles=1,
        )

        result = run_vector_refresh_worker(
            config,
            provider,
            memory_index_factory=lambda **_: memory,
            output=output,
        )

        stored = json.loads(heartbeat.read_text(encoding="utf-8"))
        emitted = json.loads(output.getvalue())
        assert result["status"] == "ok"
        assert stored["status"] == "ok"
        assert emitted["result"]["action"] == "rebuilt"

    def test_public_results_require_allowlist(self):
        result = VectorSearchResult(
            chunk_id="one",
            text="public text",
            source="trace",
            score=0.9,
            metadata={"path": "reasoning_traces/decision.md"},
        )

        public = public_vector_results([result], PublicVectorResultPolicy())

        assert public == []

    def test_public_results_filter_and_redact_text_and_metadata(self):
        result = VectorSearchResult(
            chunk_id="one",
            text="alpha private-token public summary",
            source="trace",
            score=0.9,
            metadata={
                "document": "private-token-note.md",
                "document_type": "trace",
                "paragraph_idx": 0,
                "path": "reasoning_traces/private-token-note.md",
            },
        )
        policy = PublicVectorResultPolicy(
            allowed_sources=("trace",),
            allowed_path_prefixes=("reasoning_traces",),
            redacted_terms=("private-token",),
        )

        public = public_vector_results([result], policy)

        assert public == [
            {
                "chunk_id": "one",
                "text": "alpha [redacted] public summary",
                "source": "trace",
                "score": 0.9,
                "metadata": {
                    "document": "[redacted]-note.md",
                    "document_type": "trace",
                    "paragraph_idx": 0,
                    "path": "reasoning_traces/[redacted]-note.md",
                },
                "redactions": 3,
            }
        ]

    def test_public_results_reject_nonmatching_path_prefix(self):
        result = VectorSearchResult(
            chunk_id="one",
            text="public text",
            source="trace",
            score=0.9,
            metadata={"path": "paper/decision.md"},
        )
        policy = PublicVectorResultPolicy(
            allowed_sources=("trace",),
            allowed_path_prefixes=("reasoning_traces",),
        )

        public = public_vector_results([result], policy)

        assert public == []

    def test_public_results_accept_trailing_slash_path_prefix(self):
        result = VectorSearchResult(
            chunk_id="one",
            text="public text",
            source="paper",
            score=0.9,
            metadata={"path": "paper/decision.md"},
        )
        policy = PublicVectorResultPolicy(
            allowed_sources=("paper",),
            allowed_path_prefixes=("paper/",),
        )

        public = public_vector_results([result], policy)

        assert public[0]["metadata"]["path"] == "paper/decision.md"

    def test_status_reports_missing_index(self, tmp_path: Path):
        status = vector_index_status(tmp_path / "missing")

        assert status["exists"] is False
        assert status["count"] == 0
        assert status["manifest"] == {}

    def test_cli_estimate_outputs_storage_json(self):
        output = StringIO()

        code = run_vector_cli(
            ["estimate", "200000", "--dimension", "768"],
            output=output,
        )

        payload = json.loads(output.getvalue())
        assert code == 0
        assert payload["count"] == 200000
        assert payload["dimension"] == 768
        assert payload["total_bytes"] > 700_000_000

    def test_cli_build_and_search_use_injected_dependencies(self, tmp_path: Path):
        memory = _built_memory_index()
        provider = KeywordEmbeddingProvider()
        index_dir = tmp_path / "memory_vec"
        build_output = StringIO()
        search_output = StringIO()

        build_code = run_vector_cli(
            ["build", "--index-dir", str(index_dir), "--batch-size", "1"],
            provider=provider,
            memory_index=memory,
            output=build_output,
        )
        search_code = run_vector_cli(
            ["search", "beta verifier", "--index-dir", str(index_dir), "--top", "1"],
            provider=provider,
            output=search_output,
        )

        build_payload = json.loads(build_output.getvalue())
        search_payload = json.loads(search_output.getvalue())
        assert build_code == 0
        assert search_code == 0
        assert build_payload["count"] == 2
        assert search_payload[0]["text"] == "beta factual verifier paragraph"

    def test_cli_public_search_uses_allowlist_and_redaction(self, tmp_path: Path):
        memory = _built_memory_index()
        memory.documents[0].paragraphs[1] = "beta private-token verifier paragraph"
        provider = KeywordEmbeddingProvider()
        index_dir = tmp_path / "memory_vec"
        build_output = StringIO()
        search_output = StringIO()

        run_vector_cli(
            ["build", "--index-dir", str(index_dir), "--batch-size", "1"],
            provider=provider,
            memory_index=memory,
            output=build_output,
        )
        search_code = run_vector_cli(
            [
                "search",
                "beta verifier",
                "--index-dir",
                str(index_dir),
                "--top",
                "1",
                "--public",
                "--public-source",
                "trace",
                "--public-path-prefix",
                "reasoning_traces",
                "--redact-term",
                "private-token",
            ],
            provider=provider,
            output=search_output,
        )

        payload = json.loads(search_output.getvalue())
        assert search_code == 0
        assert payload[0]["text"] == "beta [redacted] verifier paragraph"
        assert payload[0]["redactions"] == 1

    def test_cli_refresh_skips_unchanged_index(self, tmp_path: Path):
        memory = _built_memory_index()
        provider = KeywordEmbeddingProvider()
        index_dir = tmp_path / "memory_vec"
        output = StringIO()

        assert (
            run_vector_cli(
                ["refresh", "--index-dir", str(index_dir)],
                provider=provider,
                memory_index=memory,
                output=StringIO(),
            )
            == 0
        )
        code = run_vector_cli(
            ["refresh", "--index-dir", str(index_dir)],
            provider=provider,
            memory_index=memory,
            output=output,
        )

        payload = json.loads(output.getvalue())
        assert code == 0
        assert payload["action"] == "skipped"


def _built_memory_index() -> MemoryIndex:
    memory = MemoryIndex()
    memory.documents = [
        Document(
            name="decision.md",
            source="trace",
            path="reasoning_traces/decision.md",
            paragraphs=[
                "alpha memory retrieval paragraph",
                "beta factual verifier paragraph",
            ],
            tokens={"alpha", "beta"},
            date="2026-04-27",
            doc_type="trace",
        )
    ]
    memory.paragraph_index = [(0, 0), (0, 1)]
    memory._built = True
    return memory


class CountingEmbeddingProvider:
    """Deterministic provider that records every text it is asked to embed.

    The vector for a text depends only on that text, so a reused stored vector
    is bit-identical to what re-embedding would produce — letting the tests
    assert both that unchanged chunks are *not* re-sent and that the index they
    land in is identical to a full rebuild.
    """

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.embedded: list[str] = []
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        self.embedded.extend(texts)
        rows = []
        for text in texts:
            seed = int(sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            rows.append(np.random.default_rng(seed).standard_normal(self.dimension))
        return np.asarray(rows, dtype=np.float32)


def _ic(chunk_id: str, text: str) -> VectorChunk:
    return VectorChunk(chunk_id=chunk_id, text=text, source="trace")


class TestIncrementalBuild:
    """Reuse stored embeddings for unchanged chunks; embed only the delta."""

    def _seed(self, tmp_path: Path, provider) -> PersistentVectorIndex:
        index = PersistentVectorIndex(tmp_path / "vi")
        index.build([_ic("a", "alpha"), _ic("b", "beta"), _ic("c", "gamma")], provider)
        return index

    def test_only_new_chunks_are_embedded(self, tmp_path: Path):
        provider = CountingEmbeddingProvider()
        index = self._seed(tmp_path, provider)
        before = index._load_vectors().copy()

        provider.embedded.clear()
        stats = index.build(
            [_ic("a", "alpha"), _ic("b", "beta"), _ic("c", "gamma"), _ic("d", "delta")],
            provider,
        )

        assert provider.embedded == ["delta"]  # only the new chunk's text
        assert stats.reused == 3
        assert stats.embedded == 1
        assert stats.count == 4
        # The three unchanged rows are byte-identical to the prior store.
        after = index._load_vectors()
        np.testing.assert_array_equal(after[:3], before)

    def test_identical_corpus_calls_provider_never(self, tmp_path: Path):
        provider = CountingEmbeddingProvider()
        index = self._seed(tmp_path, provider)

        provider.calls = 0
        provider.embedded.clear()
        stats = index.build([_ic("a", "alpha"), _ic("b", "beta"), _ic("c", "gamma")], provider)

        assert provider.calls == 0
        assert provider.embedded == []
        assert stats.reused == 3
        assert stats.embedded == 0

    def test_edited_text_is_reembedded_others_reused(self, tmp_path: Path):
        provider = CountingEmbeddingProvider()
        index = self._seed(tmp_path, provider)

        provider.embedded.clear()
        stats = index.build(
            [_ic("a", "alpha"), _ic("b", "beta EDITED"), _ic("c", "gamma")],
            provider,
        )

        assert provider.embedded == ["beta EDITED"]
        assert stats.reused == 2
        assert stats.embedded == 1

    def test_reuse_false_forces_full_reembed(self, tmp_path: Path):
        provider = CountingEmbeddingProvider()
        index = self._seed(tmp_path, provider)

        provider.embedded.clear()
        stats = index.build(
            [_ic("a", "alpha"), _ic("b", "beta"), _ic("c", "gamma")],
            provider,
            reuse=False,
        )

        assert sorted(provider.embedded) == ["alpha", "beta", "gamma"]
        assert stats.reused == 0

    def test_dimension_change_triggers_full_reembed(self, tmp_path: Path):
        index = self._seed(tmp_path, CountingEmbeddingProvider(dimension=4))
        wider = CountingEmbeddingProvider(dimension=8)

        stats = index.build(
            [_ic("a", "alpha"), _ic("b", "beta"), _ic("c", "gamma"), _ic("d", "delta")],
            wider,
        )

        # Mixing geometries would corrupt search, so every chunk is re-embedded
        # at the new width (the new chunk is also touched by the dimension probe
        # before the mismatch is detected — acceptable on a rare model change).
        assert stats.reused == 0
        assert stats.embedded == 4
        assert stats.dimension == 8
        assert set(wider.embedded) == {"alpha", "beta", "gamma", "delta"}

    def test_fresh_index_embeds_all_with_zero_reuse(self, tmp_path: Path):
        provider = CountingEmbeddingProvider()
        index = PersistentVectorIndex(tmp_path / "vi")

        stats = index.build([_ic("a", "alpha"), _ic("b", "beta")], provider)

        assert stats.reused == 0
        assert stats.embedded == 2
        assert sorted(provider.embedded) == ["alpha", "beta"]

    def test_rejects_non_positive_batch_size(self, tmp_path: Path):
        index = PersistentVectorIndex(tmp_path / "vi")
        with pytest.raises(ValueError, match="batch_size must be positive"):
            index.build([_ic("a", "alpha")], CountingEmbeddingProvider(), batch_size=0)

    def test_reused_index_still_searches_correctly(self, tmp_path: Path):
        provider = KeywordEmbeddingProvider()
        index = PersistentVectorIndex(tmp_path / "vi")
        index.build([_ic("a", "alpha one"), _ic("b", "beta two")], provider)
        index.build(
            [_ic("a", "alpha one"), _ic("b", "beta two"), _ic("g", "gamma three")],
            provider,
        )

        results = index.search("gamma", provider, top_k=1)
        assert results[0].chunk_id == "g"
