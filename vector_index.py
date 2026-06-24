# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Persistent vector index

"""Persistent dense-vector index with SQLite metadata.

This module is the storage layer for large local RAG indexes. It keeps
vectors in a contiguous NumPy matrix, keeps chunk metadata in SQLite, and
uses a generic HTTP embedding service for local or remote embedding
providers. The search path is exact cosine similarity; this is deliberate
for the first production slice because it is deterministic, inspectable,
and already practical at hundreds of thousands of vectors on workstation
hardware.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from http import client as http_client
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np


class VectorIndexError(RuntimeError):
    """Raised when vector index input, persistence, or provider output is invalid."""


class EmbeddingProvider(Protocol):
    """Interface required by ``PersistentVectorIndex``."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return one dense vector per input text as a two-dimensional array."""


@dataclass(frozen=True)
class VectorChunk:
    """One retrievable text chunk."""

    chunk_id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VectorSearchResult:
    """Search result returned from the vector index."""

    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorIndexStats:
    """Build or storage statistics."""

    count: int
    dimension: int
    vector_bytes: int
    metadata_bytes: int
    elapsed_s: float = 0.0
    reused: int = 0
    """Chunks whose stored vector was kept rather than re-embedded. ``count``
    minus ``reused`` is the number of texts actually sent to the provider."""

    @property
    def total_bytes(self) -> int:
        return self.vector_bytes + self.metadata_bytes

    @property
    def embedded(self) -> int:
        """Chunks sent to the embedding provider this build."""
        return self.count - self.reused


class HttpEmbeddingClient:
    """Embedding provider backed by a generic JSON-over-HTTP endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        timeout_s: float = 120.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        parsed = urlparse(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an http or https URL")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra_headers = dict(extra_headers or {})

    @classmethod
    def from_env(cls, prefix: str = "REMANENTIA_EMBEDDING") -> HttpEmbeddingClient:
        """Create a client from environment variables.

        Required:
        - ``{prefix}_BASE_URL``
        - ``{prefix}_MODEL``

        Optional:
        - ``{prefix}_API_KEY``
        - ``{prefix}_TIMEOUT_S``
        """
        base_url = os.environ.get(f"{prefix}_BASE_URL", "").strip()
        model = os.environ.get(f"{prefix}_MODEL", "").strip()
        api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        timeout_s = float(os.environ.get(f"{prefix}_TIMEOUT_S", "120"))
        return cls(base_url=base_url, model=model, api_key=api_key, timeout_s=timeout_s)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise VectorIndexError("cannot embed an empty text batch")
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        path = f"{self._base_path}/embeddings" if self._base_path else "/embeddings"
        connection_cls = (
            http_client.HTTPSConnection if self._scheme == "https" else http_client.HTTPConnection
        )
        connection = connection_cls(self._host, self._port, timeout=self.timeout_s)
        try:
            connection.request("POST", path, body=payload, headers=headers)
            response = connection.getresponse()
            raw_body = response.read().decode("utf-8")
            if response.status < 200 or response.status >= 300:
                raise VectorIndexError(
                    f"embedding request failed with HTTP {response.status}: {raw_body[:200]}"
                )
            body = json.loads(raw_body)
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorIndexError(f"embedding request failed: {exc}") from exc
        finally:
            connection.close()
        return _parse_embedding_response(body, expected_count=len(texts))


class PersistentVectorIndex:
    """Disk-backed exact vector search index.

    Layout under ``root``:

    - ``vectors.npz`` — normalised float32 vector matrix
    - ``chunks.sqlite`` — chunk text, source, hash, metadata
    - ``manifest.json`` — count, dimension, byte estimates, build timestamp
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.vectors_path = self.root / "vectors.npz"
        self.metadata_path = self.root / "chunks.sqlite"
        self.manifest_path = self.root / "manifest.json"

    def exists(self) -> bool:
        return self.vectors_path.exists() and self.metadata_path.exists()

    def count(self) -> int:
        if not self.metadata_path.exists():
            return 0
        with sqlite3.connect(self.metadata_path) as conn:
            row = conn.execute("select count(*) from chunks").fetchone()
        return int(row[0] if row else 0)

    def build(
        self,
        chunks: Iterable[VectorChunk],
        provider: EmbeddingProvider,
        *,
        batch_size: int = 64,
        reuse: bool = True,
    ) -> VectorIndexStats:
        """Build the index, reusing prior embeddings for unchanged chunks.

        Embedding is the expensive step and depends only on chunk text, so when
        a prior index exists every chunk whose text is byte-identical to one
        already embedded — matched by content hash — keeps its stored vector and
        only genuinely new or edited chunks are sent to the provider. A corpus
        that grew by a handful of records re-embeds a handful, not the whole
        store, which is the difference between seconds and re-embedding hundreds
        of thousands of vectors that did not change.

        Pass ``reuse=False`` to force a full re-embed — for instance after an
        embedding-model change that the content hash cannot see. A dimension
        change between the stored vectors and the provider's output is detected
        and triggers the same full re-embed automatically, since mixing vector
        geometries would corrupt search.
        """
        started = time.perf_counter()
        chunk_list = list(chunks)
        _validate_chunks(chunk_list)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        prior = self._prior_vectors_by_hash() if reuse else {}
        pending = [chunk for chunk in chunk_list if chunk.content_hash not in prior]
        fresh = self._embed_chunks(pending, provider, batch_size) if pending else None

        if fresh is not None and prior:
            prior_dim = next(iter(prior.values())).shape[0]
            if fresh.shape[1] != prior_dim:
                prior = {}
                fresh = self._embed_chunks(chunk_list, provider, batch_size)

        vectors = self._assemble_vectors(chunk_list, prior, fresh)
        reused = len(chunk_list) - (0 if fresh is None else fresh.shape[0])

        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_save_vectors(self.vectors_path, vectors)
        metadata_bytes = _write_metadata(self.metadata_path, chunk_list)

        elapsed = time.perf_counter() - started
        stats = VectorIndexStats(
            count=len(chunk_list),
            dimension=int(vectors.shape[1]),
            vector_bytes=int(vectors.nbytes),
            metadata_bytes=metadata_bytes,
            elapsed_s=round(elapsed, 6),
            reused=reused,
        )
        self._write_manifest(stats)
        return stats

    def _prior_vectors_by_hash(self) -> dict[str, np.ndarray]:
        """Map content hash → stored vector from the existing index, for reuse.

        Returns an empty map when no index exists yet, so a first build embeds
        every chunk. Both stores are read fully into memory before the build
        overwrites them, so reusing the index it is rebuilding is safe.
        """
        if not self.exists():
            return {}
        vectors = self._load_vectors()
        mapping: dict[str, np.ndarray] = {}
        with sqlite3.connect(self.metadata_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("select ordinal, content_hash from chunks").fetchall()
        for row in rows:
            ordinal = int(row["ordinal"])
            if 0 <= ordinal < vectors.shape[0]:
                mapping[str(row["content_hash"])] = vectors[ordinal]
        return mapping

    def _assemble_vectors(
        self,
        chunk_list: list[VectorChunk],
        prior: dict[str, np.ndarray],
        fresh: np.ndarray | None,
    ) -> np.ndarray:
        """Interleave reused and freshly embedded vectors into chunk order."""
        if not prior:
            assert fresh is not None  # a non-empty chunk list always embeds when no reuse
            return fresh
        dimension = fresh.shape[1] if fresh is not None else next(iter(prior.values())).shape[0]
        vectors = np.empty((len(chunk_list), dimension), dtype=np.float32)
        fresh_cursor = 0
        for ordinal, chunk in enumerate(chunk_list):
            reused_vector = prior.get(chunk.content_hash)
            if reused_vector is not None:
                vectors[ordinal] = reused_vector
            else:
                assert fresh is not None
                vectors[ordinal] = fresh[fresh_cursor]
                fresh_cursor += 1
        return vectors

    def search(
        self,
        query: str,
        provider: EmbeddingProvider,
        *,
        top_k: int = 5,
        source: str = "",
    ) -> list[VectorSearchResult]:
        """Return top-k chunks by exact cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.exists():
            raise VectorIndexError("vector index has not been built")
        query_vec = _normalise_matrix(provider.embed_texts([query]))
        vectors = self._load_vectors()
        if vectors.shape[1] != query_vec.shape[1]:
            raise VectorIndexError(
                f"query vector dimension {query_vec.shape[1]} does not match "
                f"index dimension {vectors.shape[1]}"
            )
        rows = _load_metadata_rows(self.metadata_path, source=source)
        if not rows:
            return []

        ordinals = np.asarray([row["ordinal"] for row in rows], dtype=np.int64)
        scores = vectors[ordinals] @ query_vec[0]
        order = np.argsort(-scores)[:top_k]
        results = []
        for idx in order:
            row = rows[int(idx)]
            score = float(np.clip(scores[int(idx)], -1.0, 1.0))
            results.append(
                VectorSearchResult(
                    chunk_id=str(row["chunk_id"]),
                    text=str(row["text"]),
                    source=str(row["source"]),
                    score=score,
                    metadata=json.loads(str(row["metadata_json"] or "{}")),
                )
            )
        return results

    @staticmethod
    def estimate_storage(
        count: int,
        *,
        dimension: int = 768,
        vector_bytes_per_value: int = 4,
        metadata_bytes_per_chunk: int = 512,
        index_overhead_ratio: float = 1.25,
    ) -> VectorIndexStats:
        """Estimate disk budget for a vector corpus.

        ``index_overhead_ratio`` covers ANN graph files or database index
        overhead. Exact NumPy storage uses a ratio close to 1.0; HNSW-style
        indexes use more.
        """
        if count < 0:
            raise ValueError("count must not be negative")
        raw_vector_bytes = count * dimension * vector_bytes_per_value
        vector_bytes = int(raw_vector_bytes * index_overhead_ratio)
        metadata_bytes = count * metadata_bytes_per_chunk
        return VectorIndexStats(
            count=count,
            dimension=dimension,
            vector_bytes=vector_bytes,
            metadata_bytes=metadata_bytes,
        )

    def _embed_chunks(
        self,
        chunks: list[VectorChunk],
        provider: EmbeddingProvider,
        batch_size: int,
    ) -> np.ndarray:
        batches = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = provider.embed_texts([chunk.text for chunk in batch])
            batches.append(vectors)
        matrix = np.vstack(batches).astype(np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise VectorIndexError("embedding provider returned invalid vector matrix")
        return _normalise_matrix(matrix)

    def _load_vectors(self) -> np.ndarray:
        with np.load(self.vectors_path, allow_pickle=False) as data:
            vectors = data["vectors"].astype(np.float32)
        if vectors.ndim != 2:
            raise VectorIndexError("stored vector matrix must be two-dimensional")
        return vectors

    def _write_manifest(self, stats: VectorIndexStats) -> None:
        payload = {
            "count": stats.count,
            "dimension": stats.dimension,
            "vector_bytes": stats.vector_bytes,
            "metadata_bytes": stats.metadata_bytes,
            "total_bytes": stats.total_bytes,
            "elapsed_s": stats.elapsed_s,
            "reused": stats.reused,
            "embedded": stats.embedded,
            "created_at_unix": int(time.time()),
        }
        _atomic_write_text(self.manifest_path, json.dumps(payload, indent=2) + "\n")


def _parse_embedding_response(body: Any, expected_count: int) -> np.ndarray:
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise VectorIndexError("embedding response must contain a data list")
    rows = sorted(body["data"], key=lambda item: int(item.get("index", 0)))
    if len(rows) != expected_count:
        raise VectorIndexError(
            f"embedding response returned {len(rows)} rows, expected {expected_count}"
        )
    vectors = []
    for row in rows:
        embedding = row.get("embedding") if isinstance(row, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise VectorIndexError("embedding row must contain a non-empty embedding list")
        vectors.append(embedding)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise VectorIndexError("embedding response must form a two-dimensional matrix")
    return matrix


def _validate_chunks(chunks: list[VectorChunk]) -> None:
    if not chunks:
        raise VectorIndexError("cannot build vector index from zero chunks")
    seen = set()
    for chunk in chunks:
        if not chunk.chunk_id.strip():
            raise VectorIndexError("chunk_id must not be empty")
        if chunk.chunk_id in seen:
            raise VectorIndexError(f"duplicate chunk_id: {chunk.chunk_id}")
        if not chunk.text.strip():
            raise VectorIndexError(f"chunk {chunk.chunk_id} has empty text")
        seen.add(chunk.chunk_id)


def _normalise_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise VectorIndexError("vectors must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _atomic_save_vectors(path: Path, vectors: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, vectors=vectors.astype(np.float32))
    tmp.replace(path)


def _write_metadata(path: Path, chunks: list[VectorChunk]) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with sqlite3.connect(tmp) as conn:
        conn.execute(
            """
            create table chunks (
                ordinal integer primary key,
                chunk_id text not null unique,
                source text not null,
                text text not null,
                content_hash text not null,
                metadata_json text not null
            )
            """
        )
        conn.execute("create index chunks_source_idx on chunks(source)")
        rows = [
            (
                ordinal,
                chunk.chunk_id,
                chunk.source,
                chunk.text,
                chunk.content_hash,
                json.dumps(chunk.metadata, sort_keys=True, separators=(",", ":")),
            )
            for ordinal, chunk in enumerate(chunks)
        ]
        conn.executemany("insert into chunks values (?, ?, ?, ?, ?, ?)", rows)
    tmp.replace(path)
    return path.stat().st_size


def _load_metadata_rows(path: Path, *, source: str = "") -> list[sqlite3.Row]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        if source:
            rows = conn.execute(
                """
                select ordinal, chunk_id, source, text, metadata_json
                from chunks
                where source = ?
                order by ordinal
                """,
                (source,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                select ordinal, chunk_id, source, text, metadata_json
                from chunks
                order by ordinal
                """
            ).fetchall()
    return list(rows)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
