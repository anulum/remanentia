# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Unified memory index persistence

"""Secure JSON/gzip and NumPy sidecar persistence for the memory index."""

from __future__ import annotations

import gzip
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)
Array = NDArray[np.generic]
FloatArray = NDArray[np.float32]


class PersistedDocument(Protocol):
    """Document fields stored in index metadata."""

    name: str
    source: str
    path: str
    paragraphs: list[str]
    date: str
    doc_type: str


def load_content_hashes(path: Path) -> dict[str, str]:
    """Load a string-to-string content hash map from JSON."""
    if not path.exists():
        return {}
    try:
        decoded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {
        key: value
        for key, value in decoded.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def save_content_hashes(path: Path, hashes: Mapping[str, str]) -> None:
    """Persist content hashes for the next incremental build."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(hashes)), encoding="utf-8")


def embedding_sidecar_path(path: Path) -> Path:
    """Derive the NPZ sidecar path from a gzip metadata path."""
    stem = path.stem.replace(".json", "")
    return path.with_name(stem + "_embeddings.npz")


def save_index(
    path: Path,
    documents: Sequence[PersistedDocument],
    paragraph_index: Sequence[tuple[int, int]],
    paragraph_tokens: Sequence[set[str]],
    paragraph_token_counts: Sequence[Mapping[str, int]],
    paragraph_types: Sequence[str],
    idf: Mapping[str, float],
    document_frequency: Mapping[str, int],
    embeddings: Array | None,
    *,
    quantize: bool = True,
    timestamp: float | None = None,
) -> None:
    """Atomically persist metadata and an optional embedding sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    embedding_data: Array | None = None
    embedding_scale: FloatArray | None = None
    if embeddings is not None:
        if quantize:
            scale = np.max(np.abs(embeddings), axis=1, keepdims=True)
            scale = np.where(scale == 0, 1.0, scale)
            embedding_data = (embeddings / scale * 127).astype(np.int8)
            embedding_scale = scale.astype(np.float32)
        else:
            embedding_data = embeddings

    metadata: dict[str, Any] = {
        "documents": [
            (doc.name, doc.source, doc.path, doc.paragraphs, doc.date, doc.doc_type)
            for doc in documents
        ],
        "paragraph_index": list(paragraph_index),
        "paragraph_tokens": [list(tokens) for tokens in paragraph_tokens],
        "paragraph_token_counts": [dict(counts) for counts in paragraph_token_counts],
        "paragraph_types": list(paragraph_types),
        "idf": dict(idf),
        "_df": dict(document_frequency),
        "quantized": quantize and embeddings is not None,
        "timestamp": time.time() if timestamp is None else timestamp,
    }
    temporary_path = path.with_suffix(".tmp")
    raw = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(temporary_path, "wb") as metadata_file:
        metadata_file.write(raw)
    temporary_path.replace(path)

    sidecar = embedding_sidecar_path(path)
    if embedding_data is None:
        sidecar.unlink(missing_ok=True)
        return
    arrays: dict[str, Array] = {"embeddings": embedding_data}
    if embedding_scale is not None:
        arrays["emb_scale"] = embedding_scale
    cast(Any, np.savez_compressed)(sidecar, **arrays)


def load_index_data(
    path: Path,
    *,
    default_path: Path | None = None,
    legacy_path: Path | None = None,
) -> dict[str, Any] | None:
    """Load gzip JSON metadata and its safe, pickle-free NPZ sidecar."""
    if not path.exists():
        if (
            default_path is not None
            and path == default_path
            and legacy_path is not None
            and legacy_path.exists()
        ):
            path = legacy_path
        else:
            return None
    try:
        with path.open("rb") as source:
            magic = source.read(2)
    except OSError:
        return None
    if magic != b"\x1f\x8b":
        return None
    try:
        with gzip.open(path, "rb") as metadata_file:
            decoded = cast(object, json.loads(metadata_file.read()))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    metadata = cast(dict[str, Any], decoded)
    sidecar = embedding_sidecar_path(path)
    if sidecar.exists():
        try:
            with np.load(sidecar, allow_pickle=False) as arrays:
                metadata["embeddings"] = arrays.get("embeddings")
                metadata["emb_scale"] = arrays.get("emb_scale")
        except (OSError, ValueError):
            log.debug("Embedding sidecar load failed: %s", sidecar, exc_info=True)
    return metadata


def validate_loaded_embeddings(
    embedding_data: Any,
    embedding_scale: Any,
    *,
    quantized: bool,
    paragraph_count: int,
) -> FloatArray | None:
    """Decode and validate an embedding sidecar against index dimensions."""
    if embedding_data is None:
        return None
    try:
        embeddings = np.asarray(embedding_data)
        if quantized:
            if embedding_scale is None:
                return None
            scale = np.asarray(embedding_scale, dtype=np.float32)
            embeddings = (embeddings.astype(np.float32) / 127.0) * scale
        else:
            embeddings = embeddings.astype(np.float32, copy=False)
    except (TypeError, ValueError):
        log.debug("Loaded embedding sidecar has invalid array data", exc_info=True)
        return None
    if embeddings.ndim != 2 or embeddings.shape[0] != paragraph_count:
        log.debug(
            "Ignoring embedding sidecar with shape %s for %d paragraphs",
            embeddings.shape,
            paragraph_count,
        )
        return None
    if not np.isfinite(embeddings).all():
        log.debug("Ignoring embedding sidecar containing non-finite values")
        return None
    return cast(FloatArray, embeddings)
