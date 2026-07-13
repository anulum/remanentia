# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real persistence tests for unified memory index storage

"""Exercise gzip, JSON, and NPZ production persistence on real files."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from REMANENTIA.memory_index_storage import (  # type: ignore[import]
    embedding_sidecar_path,
    load_content_hashes,
    load_index_data,
    save_content_hashes,
    save_index,
    validate_loaded_embeddings,
)


@dataclass
class StoredDocument:
    """Concrete document passed through the production persistence protocol."""

    name: str = "trace.md"
    source: str = "traces"
    path: str = "/memory/trace.md"
    paragraphs: list[str] = field(default_factory=lambda: ["persistent neural memory"])
    date: str = "2026-07-13"
    doc_type: str = "markdown"


def save_sample(
    path: Path,
    embeddings: NDArray[np.generic] | None,
    *,
    quantize: bool,
    timestamp: float | None = 42.0,
) -> None:
    """Persist one real metadata record through the production writer."""
    save_index(
        path,
        [StoredDocument()],
        [(0, 0)],
        [{"persistent", "memory"}],
        [{"persistent": 1, "memory": 1}],
        ["finding"],
        {"persistent": 0.7},
        {"persistent": 1},
        embeddings,
        quantize=quantize,
        timestamp=timestamp,
    )


def test_content_hash_cache_roundtrip_filters_invalid_payloads(tmp_path: Path) -> None:
    """Hash storage uses real JSON and rejects malformed or mistyped entries."""
    path = tmp_path / "state" / "hashes.json"
    assert load_content_hashes(path) == {}
    save_content_hashes(path, {"trace.md": "abc123"})
    assert load_content_hashes(path) == {"trace.md": "abc123"}

    path.write_text(json.dumps({"good": "hash", "bad": 7}), encoding="utf-8")
    assert load_content_hashes(path) == {"good": "hash"}
    path.write_text("[]", encoding="utf-8")
    assert load_content_hashes(path) == {}
    path.write_text("not-json", encoding="utf-8")
    assert load_content_hashes(path) == {}
    assert load_content_hashes(tmp_path) == {}


def test_unquantized_index_roundtrip_uses_real_gzip_and_npz(tmp_path: Path) -> None:
    """Unquantized embeddings survive the complete metadata/sidecar roundtrip."""
    path = tmp_path / "index.json.gz"
    embeddings = np.array([[0.25, -0.5, 0.75]], dtype=np.float32)
    save_sample(path, embeddings, quantize=False)

    data = load_index_data(path)

    assert data is not None
    assert data["timestamp"] == 42.0
    assert data["documents"][0][0] == "trace.md"
    assert data["quantized"] is False
    assert np.array_equal(data["embeddings"], embeddings)
    assert data["emb_scale"] is None
    loaded = validate_loaded_embeddings(
        data["embeddings"], data["emb_scale"], quantized=False, paragraph_count=1
    )
    assert np.array_equal(loaded, embeddings)


def test_quantized_index_roundtrip_handles_zero_scale_rows(tmp_path: Path) -> None:
    """Quantization writes int8 values and safely reconstructs zero and nonzero rows."""
    path = tmp_path / "quantized.json.gz"
    embeddings = np.array([[0.0, 0.0], [1.0, -0.5]], dtype=np.float32)
    save_index(
        path,
        [StoredDocument(), StoredDocument(name="second.md")],
        [(0, 0), (1, 0)],
        [{"zero"}, {"signal"}],
        [{"zero": 1}, {"signal": 1}],
        ["finding", "finding"],
        {"signal": 1.0},
        {"signal": 1},
        embeddings,
        quantize=True,
    )

    data = load_index_data(path)
    assert data is not None
    assert data["quantized"] is True
    assert data["embeddings"].dtype == np.int8
    assert isinstance(data["timestamp"], float)
    loaded = validate_loaded_embeddings(
        data["embeddings"], data["emb_scale"], quantized=True, paragraph_count=2
    )
    assert loaded is not None
    assert np.allclose(loaded, embeddings, atol=0.01)


def test_save_without_embeddings_removes_a_real_stale_sidecar(tmp_path: Path) -> None:
    """A metadata-only save deletes an obsolete NPZ sidecar on disk."""
    path = tmp_path / "index.json.gz"
    save_sample(path, np.ones((1, 2), dtype=np.float32), quantize=False)
    sidecar = embedding_sidecar_path(path)
    assert sidecar.exists()

    save_sample(path, None, quantize=True)

    assert path.exists()
    assert not sidecar.exists()
    data = load_index_data(path)
    assert data is not None
    assert data["quantized"] is False


def test_loader_rejects_missing_legacy_corrupt_and_non_object_metadata(tmp_path: Path) -> None:
    """Unsupported pickle and malformed gzip files fail closed without substitutes."""
    missing = tmp_path / "missing.json.gz"
    assert load_index_data(missing) is None

    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(b"\x80\x04pickle")
    assert load_index_data(missing, default_path=missing, legacy_path=legacy) is None
    assert load_index_data(legacy) is None
    assert load_index_data(tmp_path) is None

    corrupt = tmp_path / "corrupt.json.gz"
    corrupt.write_bytes(b"\x1f\x8bnot-a-gzip-stream")
    assert load_index_data(corrupt) is None
    non_object = tmp_path / "list.json.gz"
    with gzip.open(non_object, "wb") as stream:
        stream.write(b"[]")
    assert load_index_data(non_object) is None


def test_loader_ignores_a_real_corrupt_embedding_sidecar(tmp_path: Path) -> None:
    """Valid gzip metadata remains usable when its NPZ sidecar is corrupt."""
    path = tmp_path / "index.json.gz"
    save_sample(path, None, quantize=False)
    embedding_sidecar_path(path).write_bytes(b"not-an-npz")

    data = load_index_data(path)

    assert data is not None
    assert "embeddings" not in data


def test_embedding_validation_rejects_real_shape_scale_and_finiteness_errors() -> None:
    """Sidecar validation rejects invalid arrays before they enter search state."""
    assert validate_loaded_embeddings(None, None, quantized=False, paragraph_count=1) is None
    assert (
        validate_loaded_embeddings(np.ones((1, 2)), None, quantized=True, paragraph_count=1) is None
    )
    assert validate_loaded_embeddings(object(), None, quantized=False, paragraph_count=1) is None
    assert validate_loaded_embeddings(np.ones(2), None, quantized=False, paragraph_count=1) is None
    assert (
        validate_loaded_embeddings(np.ones((2, 2)), None, quantized=False, paragraph_count=1)
        is None
    )
    assert (
        validate_loaded_embeddings(
            np.array([[np.nan, 1.0]]), None, quantized=False, paragraph_count=1
        )
        is None
    )
