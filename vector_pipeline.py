# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Memory vector pipeline bridge

"""Bridge between ``MemoryIndex`` paragraphs and the persistent vector index."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TextIO

from vector_index import (
    EmbeddingProvider,
    HttpEmbeddingClient,
    PersistentVectorIndex,
    VectorChunk,
    VectorIndexStats,
    VectorSearchResult,
)

BASE = Path(__file__).parent
DEFAULT_VECTOR_INDEX_DIR = BASE / "snn_state" / "vector_index"
DEFAULT_VECTOR_REFRESH_HEARTBEAT = BASE / "snn_state" / "vector_refresh_worker.json"
VECTOR_CHUNK_MAX_WORDS = 96
VECTOR_CHUNK_MAX_CHARS = 600
PRIVATE_PATH_MARKERS = (
    ".coordination",
    ".git",
    ".venv",
    "__pycache__",
    "ARCHIVE",
    "BACKUP",
    "MODELS",
    "docs/internal",
    "snn_state",
    "web/backups",
)


@dataclass(frozen=True)
class PublicVectorResultPolicy:
    """Filtering and redaction policy for public vector result views."""

    allowed_sources: tuple[str, ...] = ()
    allowed_path_prefixes: tuple[str, ...] = ()
    redacted_terms: tuple[str, ...] = ()
    replacement: str = "[redacted]"
    max_text_chars: int = 800
    metadata_keys: tuple[str, ...] = (
        "date",
        "document",
        "document_type",
        "paragraph_idx",
        "path",
    )


@dataclass(frozen=True)
class VectorRefreshWorkerConfig:
    """Configuration for the scheduled vector-index refresh worker."""

    index_dir: Path = DEFAULT_VECTOR_INDEX_DIR
    heartbeat_path: Path = DEFAULT_VECTOR_REFRESH_HEARTBEAT
    interval_s: float = 900.0
    batch_size: int = 64
    include_private: bool = False
    use_gpu_embeddings: bool = False
    force_first: bool = False
    max_cycles: int | None = None


def chunks_from_memory_index(
    memory_index: Any, *, include_private: bool = False
) -> list[VectorChunk]:
    """Convert a built ``MemoryIndex`` into stable vector chunks.

    The bridge intentionally reads only public attributes on the existing
    index object. This keeps the vector subsystem independent from the
    current BM25 implementation while still wiring it into the retrieval
    pipeline.
    """
    if not getattr(memory_index, "_built", False):
        memory_index.build(use_gpu_embeddings=False)

    chunks = []
    seen_chunk_ids: set[str] = set()
    for ordinal, pair in enumerate(memory_index.paragraph_index):
        doc_idx, paragraph_idx = pair
        doc = memory_index.documents[doc_idx]
        if not include_private and _is_private_source_path(doc.path):
            continue
        text = _vector_chunk_text(doc.paragraphs[paragraph_idx])
        chunk_id = _chunk_id(doc.source, doc.name, paragraph_idx, text)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        chunks.append(
            VectorChunk(
                chunk_id=chunk_id,
                text=text,
                source=doc.source,
                metadata={
                    "ordinal": ordinal,
                    "document": doc.name,
                    "document_type": doc.doc_type,
                    "date": doc.date,
                    "paragraph_idx": paragraph_idx,
                    "path": _safe_source_path(doc.path),
                },
            )
        )
    return chunks


def build_memory_vector_index(
    memory_index: Any,
    index_dir: Path | str,
    provider: EmbeddingProvider,
    *,
    batch_size: int = 64,
    include_private: bool = False,
) -> VectorIndexStats:
    """Build a persistent vector index from a ``MemoryIndex`` instance."""
    chunks = chunks_from_memory_index(memory_index, include_private=include_private)
    return PersistentVectorIndex(index_dir).build(chunks, provider, batch_size=batch_size)


def refresh_memory_vector_index(
    memory_index: Any,
    index_dir: Path | str,
    provider: EmbeddingProvider,
    *,
    batch_size: int = 64,
    include_private: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Rebuild the vector index only when the source corpus changed."""
    chunks = chunks_from_memory_index(memory_index, include_private=include_private)
    fingerprint = corpus_fingerprint(chunks)
    index = PersistentVectorIndex(index_dir)
    manifest = _read_manifest(index.manifest_path)
    unchanged = (
        index.exists()
        and manifest.get("corpus_fingerprint") == fingerprint
        and manifest.get("corpus_chunk_count") == len(chunks)
    )
    if unchanged and not force:
        return {
            "action": "skipped",
            "changed": False,
            "corpus_chunk_count": len(chunks),
            "corpus_fingerprint": fingerprint,
            "index_dir": str(index.root),
        }

    stats = index.build(chunks, provider, batch_size=batch_size)
    _merge_manifest(
        index.manifest_path,
        {
            "corpus_fingerprint": fingerprint,
            "corpus_chunk_count": len(chunks),
            "include_private": include_private,
            "refreshed_at_unix": int(time.time()),
        },
    )
    return {
        "action": "rebuilt",
        "changed": True,
        "corpus_chunk_count": len(chunks),
        "corpus_fingerprint": fingerprint,
        "index_dir": str(index.root),
        "stats": _stats_to_dict(stats),
    }


def search_memory_vector_index(
    index_dir: Path | str,
    query: str,
    provider: EmbeddingProvider,
    *,
    top_k: int = 5,
    source: str = "",
) -> list[VectorSearchResult]:
    """Search a persistent vector index produced from ``MemoryIndex``."""
    return PersistentVectorIndex(index_dir).search(
        query,
        provider,
        top_k=top_k,
        source=source,
    )


def run_vector_refresh_worker(
    config: VectorRefreshWorkerConfig,
    provider: EmbeddingProvider,
    *,
    memory_index_factory: Any = None,
    sleeper: Any = time.sleep,
    output: TextIO | None = None,
) -> dict[str, Any]:
    """Run the scheduled vector refresh loop and return the last heartbeat."""
    import sys

    out = output or sys.stdout
    active_factory = memory_index_factory or load_or_build_memory_index
    cycle = 0
    last_heartbeat: dict[str, Any] = {}
    while config.max_cycles is None or cycle < config.max_cycles:
        cycle += 1
        try:
            memory_index = active_factory(use_gpu_embeddings=config.use_gpu_embeddings)
            result = refresh_memory_vector_index(
                memory_index,
                config.index_dir,
                provider,
                batch_size=config.batch_size,
                include_private=config.include_private,
                force=config.force_first and cycle == 1,
            )
            last_heartbeat = _worker_heartbeat(cycle, "ok", result)
        except Exception as exc:
            last_heartbeat = _worker_heartbeat(
                cycle,
                "error",
                {"error": str(exc), "error_type": type(exc).__name__},
            )
        _write_json(out, last_heartbeat)
        _write_heartbeat(config.heartbeat_path, last_heartbeat)
        if config.max_cycles is not None and cycle >= config.max_cycles:
            break
        sleeper(config.interval_s)
    return last_heartbeat


def public_vector_results(
    results: Sequence[VectorSearchResult],
    policy: PublicVectorResultPolicy,
) -> list[dict[str, Any]]:
    """Return a filtered, redacted result view safe for public API surfaces.

    Public output must be explicitly allowlisted. If neither source nor
    path-prefix allowlists are configured, no raw result is returned.
    """
    public_results = []
    for result in results:
        if not _is_public_allowed(result, policy):
            continue
        redacted_text, text_redactions = _redact_text(result.text, policy)
        metadata, metadata_redactions = _public_metadata(result.metadata, policy)
        public_results.append(
            {
                "chunk_id": result.chunk_id,
                "text": _truncate_text(redacted_text, policy.max_text_chars),
                "source": result.source,
                "score": result.score,
                "metadata": metadata,
                "redactions": text_redactions + metadata_redactions,
            }
        )
    return public_results


def load_or_build_memory_index(*, use_gpu_embeddings: bool = False) -> Any:
    """Return a MemoryIndex that reflects the current source tree.

    The persisted cache is reused only when it is still current; if any source
    file is newer than the cache (``needs_rebuild``) — or no cache exists — the
    index is rebuilt and saved. The earlier version returned the cache whenever
    it merely *existed*, so the refresh worker fed the vector index a frozen
    corpus and the index stalled on an April-2026 worldview while new memories
    accumulated unseen. Rebuilding is incremental, so an unchanged corpus stays
    cheap; the staleness check is what makes new memory actually reach search.
    """
    from memory_index import MemoryIndex, needs_rebuild

    memory_index = MemoryIndex()
    if memory_index.load() and not needs_rebuild():
        return memory_index
    memory_index.build(use_gpu_embeddings=use_gpu_embeddings)
    memory_index.save()
    return memory_index


def vector_index_status(index_dir: Path | str = DEFAULT_VECTOR_INDEX_DIR) -> dict[str, Any]:
    """Return operator-facing status for a persistent vector index."""
    index = PersistentVectorIndex(index_dir)
    manifest = _read_manifest(index.manifest_path)
    return {
        "index_dir": str(index.root),
        "exists": index.exists(),
        "count": index.count(),
        "vectors_path": str(index.vectors_path),
        "metadata_path": str(index.metadata_path),
        "manifest": manifest,
    }


def run_vector_cli(
    argv: list[str] | None = None,
    *,
    provider: EmbeddingProvider | None = None,
    memory_index: Any | None = None,
    output: TextIO | None = None,
) -> int:
    """Run the vector-index operator CLI.

    ``provider`` and ``memory_index`` are injectable so tests can exercise
    the production command path without reaching any real endpoint.
    """
    import sys

    out = output or sys.stdout
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        _write_json(out, vector_index_status(args.index_dir))
        return 0

    if args.command == "estimate":
        stats = PersistentVectorIndex.estimate_storage(
            args.count,
            dimension=args.dimension,
            metadata_bytes_per_chunk=args.metadata_bytes_per_chunk,
            index_overhead_ratio=args.index_overhead_ratio,
        )
        _write_json(out, _stats_to_dict(stats))
        return 0

    active_provider = provider or HttpEmbeddingClient.from_env(args.env_prefix)

    if args.command == "build":
        active_memory = memory_index or load_or_build_memory_index(
            use_gpu_embeddings=args.use_gpu_embeddings
        )
        stats = build_memory_vector_index(
            active_memory,
            args.index_dir,
            active_provider,
            batch_size=args.batch_size,
            include_private=args.include_private,
        )
        _write_json(out, _stats_to_dict(stats))
        return 0

    if args.command == "refresh":
        active_memory = memory_index or load_or_build_memory_index(
            use_gpu_embeddings=args.use_gpu_embeddings
        )
        result = refresh_memory_vector_index(
            active_memory,
            args.index_dir,
            active_provider,
            batch_size=args.batch_size,
            include_private=args.include_private,
            force=args.force,
        )
        _write_json(out, result)
        return 0

    if args.command == "watch":
        cycles = args.cycles if args.cycles > 0 else None
        config = VectorRefreshWorkerConfig(
            index_dir=args.index_dir,
            heartbeat_path=args.heartbeat_path,
            interval_s=args.interval_s,
            batch_size=args.batch_size,
            include_private=args.include_private,
            use_gpu_embeddings=args.use_gpu_embeddings,
            force_first=args.force_first,
            max_cycles=cycles,
        )
        run_vector_refresh_worker(
            config,
            active_provider,
            memory_index_factory=(
                (lambda **_: memory_index)
                if memory_index is not None
                else load_or_build_memory_index
            ),
            output=out,
        )
        return 0

    if args.command == "search":
        results = search_memory_vector_index(
            args.index_dir,
            args.query,
            active_provider,
            top_k=args.top,
            source=args.source,
        )
        if args.public:
            policy = PublicVectorResultPolicy(
                allowed_sources=tuple(args.public_source or ()),
                allowed_path_prefixes=tuple(args.public_path_prefix or ()),
                redacted_terms=tuple(args.redact_term or ())
                + tuple(_load_redaction_terms(args.redaction_file)),
                max_text_chars=args.max_text_chars,
            )
            _write_json(out, public_vector_results(results, policy))
            return 0
        _write_json(out, [_result_to_dict(result) for result in results])
        return 0

    parser.print_help(file=out)  # pragma: no cover - argparse requires a subcommand.
    return 1  # pragma: no cover - paired with unreachable argparse fallback.


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m vector_pipeline``."""
    return run_vector_cli(argv)


def _chunk_id(source: str, document: str, paragraph_idx: int, text: str) -> str:
    payload = "\0".join([source, document, str(paragraph_idx), text])
    return sha256(payload.encode("utf-8")).hexdigest()


def _vector_chunk_text(text: str) -> str:
    words = text.split()
    trimmed = " ".join(words[:VECTOR_CHUNK_MAX_WORDS])
    if len(trimmed) <= VECTOR_CHUNK_MAX_CHARS:
        return trimmed
    return (
        trimmed[:VECTOR_CHUNK_MAX_CHARS].rsplit(" ", 1)[0].strip()
        or trimmed[:VECTOR_CHUNK_MAX_CHARS]
    )


def corpus_fingerprint(chunks: Sequence[VectorChunk]) -> str:
    """Return a deterministic fingerprint for vector-index refresh decisions."""
    digest = sha256()
    for chunk in chunks:
        payload = {
            "chunk_id": chunk.chunk_id,
            "content_hash": chunk.content_hash,
            "metadata": chunk.metadata,
            "source": chunk.source,
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_source_path(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return path_value
    for root in (BASE, BASE.parent):
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def _is_private_source_path(path_value: str) -> bool:
    normalised = path_value.replace("\\", "/")
    return any(marker in normalised for marker in PRIVATE_PATH_MARKERS)


def _is_public_allowed(result: VectorSearchResult, policy: PublicVectorResultPolicy) -> bool:
    if not policy.allowed_sources and not policy.allowed_path_prefixes:
        return False
    if policy.allowed_sources and result.source not in policy.allowed_sources:
        return False
    path_value = str(result.metadata.get("path", "")).replace("\\", "/")
    if policy.allowed_path_prefixes:
        return any(
            _path_matches_prefix(path_value, prefix) for prefix in policy.allowed_path_prefixes
        )
    return True


def _path_matches_prefix(path_value: str, prefix: str) -> bool:
    normalised_prefix = prefix.replace("\\", "/").strip("/")
    normalised_path = path_value.strip("/")
    return normalised_path == normalised_prefix or normalised_path.startswith(
        f"{normalised_prefix}/"
    )


def _public_metadata(
    metadata: dict[str, Any],
    policy: PublicVectorResultPolicy,
) -> tuple[dict[str, Any], int]:
    public_metadata = {}
    redactions = 0
    for key in policy.metadata_keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, str):
            public_metadata[key], count = _redact_text(value, policy)
            redactions += count
        else:
            public_metadata[key] = value
    return public_metadata, redactions


def _redact_text(text: str, policy: PublicVectorResultPolicy) -> tuple[str, int]:
    redacted = text
    redactions = 0
    for term in policy.redacted_terms:
        clean_term = term.strip()
        if not clean_term:
            continue
        redacted, count = re.subn(
            re.escape(clean_term),
            policy.replacement,
            redacted,
            flags=re.IGNORECASE,
        )
        redactions += count
    return redacted, redactions


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _load_redaction_terms(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"redaction file does not exist: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vector_pipeline",
        description="Build and query the persistent Remanentia vector index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show vector-index status")
    p_status.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)

    p_estimate = sub.add_parser("estimate", help="Estimate vector storage")
    p_estimate.add_argument("count", type=int, help="Number of chunks")
    p_estimate.add_argument("--dimension", type=int, default=768)
    p_estimate.add_argument("--metadata-bytes-per-chunk", type=int, default=512)
    p_estimate.add_argument("--index-overhead-ratio", type=float, default=1.25)

    p_build = sub.add_parser("build", help="Build vector index from MemoryIndex")
    p_build.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--env-prefix", default="REMANENTIA_EMBEDDING")
    p_build.add_argument("--use-gpu-embeddings", action="store_true")
    p_build.add_argument(
        "--include-private",
        action="store_true",
        help="Include ignored coordination/runtime paths in the vector index",
    )

    p_refresh = sub.add_parser("refresh", help="Refresh vector index only when corpus changed")
    p_refresh.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)
    p_refresh.add_argument("--batch-size", type=int, default=64)
    p_refresh.add_argument("--env-prefix", default="REMANENTIA_EMBEDDING")
    p_refresh.add_argument("--use-gpu-embeddings", action="store_true")
    p_refresh.add_argument("--include-private", action="store_true")
    p_refresh.add_argument("--force", action="store_true", help="Rebuild even when unchanged")

    p_watch = sub.add_parser("watch", help="Run scheduled vector-index refresh worker")
    p_watch.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)
    p_watch.add_argument("--heartbeat-path", type=Path, default=DEFAULT_VECTOR_REFRESH_HEARTBEAT)
    p_watch.add_argument("--interval-s", type=float, default=900.0)
    p_watch.add_argument("--batch-size", type=int, default=64)
    p_watch.add_argument("--env-prefix", default="REMANENTIA_EMBEDDING")
    p_watch.add_argument("--use-gpu-embeddings", action="store_true")
    p_watch.add_argument("--include-private", action="store_true")
    p_watch.add_argument("--force-first", action="store_true")
    p_watch.add_argument("--cycles", type=int, default=0, help="0 means run until stopped")

    p_search = sub.add_parser("search", help="Search a built vector index")
    p_search.add_argument("query")
    p_search.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)
    p_search.add_argument("--top", type=int, default=5)
    p_search.add_argument("--source", default="")
    p_search.add_argument("--env-prefix", default="REMANENTIA_EMBEDDING")
    p_search.add_argument("--public", action="store_true", help="Emit public-safe result views")
    p_search.add_argument(
        "--public-source",
        action="append",
        default=[],
        help="Allow one result source in public output; repeat for multiple sources",
    )
    p_search.add_argument(
        "--public-path-prefix",
        action="append",
        default=[],
        help="Allow one metadata path prefix in public output; repeat for multiple prefixes",
    )
    p_search.add_argument(
        "--redaction-file",
        type=Path,
        default=None,
        help="Read one sensitive public-output redaction term per line",
    )
    p_search.add_argument(
        "--redact-term",
        action="append",
        default=[],
        help="Add one sensitive public-output redaction term; repeat for multiple terms",
    )
    p_search.add_argument("--max-text-chars", type=int, default=800)

    return parser


def _stats_to_dict(stats: VectorIndexStats) -> dict[str, int | float]:
    return {
        "count": stats.count,
        "dimension": stats.dimension,
        "vector_bytes": stats.vector_bytes,
        "metadata_bytes": stats.metadata_bytes,
        "total_bytes": stats.total_bytes,
        "elapsed_s": stats.elapsed_s,
        "reused": stats.reused,
        "embedded": stats.embedded,
    }


def _result_to_dict(result: VectorSearchResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "text": result.text,
        "source": result.source,
        "score": result.score,
        "metadata": result.metadata,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # The manifest is a rebuildable derived artefact, not source memory.
        # A torn or unreadable manifest (a crash or disk-full mid-write) must
        # not raise here: every reader runs *before* the rebuild that would
        # repair it, so a raise would permanently stall the self-healing
        # refresh worker on a file it can never get past. Treat it as absent
        # instead — the corpus-fingerprint gate then sees a mismatch and
        # rebuilds, and the rebuild rewrites a clean manifest atomically.
        return {}
    # A manifest that parsed to a non-object (list/scalar) is equally unusable
    # for the ``.get(...)`` gate; treat it as absent for the same reason.
    return loaded if isinstance(loaded, dict) else {}


def _merge_manifest(path: Path, fields: dict[str, Any]) -> None:
    from file_utils import atomic_write_text

    manifest = _read_manifest(path)
    manifest.update(fields)
    # Atomic write (tmpfile + fsync + os.replace): the refresh gate reads this
    # manifest before it can rebuild, so a torn write here would brick the
    # worker. This was the lone non-atomic writer in the store/persist stack.
    atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _worker_heartbeat(cycle: int, status: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "cycle": cycle,
        "pid": os.getpid(),
        "result": result,
        "status": status,
        "timestamp_unix": int(time.time()),
    }


def _write_heartbeat(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_json(output: TextIO, payload: Any) -> None:
    output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
