# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Model-free CLI manifest I/O tests

"""Hermetic real-file tests for the model-free JSON and corpus-manifest loaders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from snn_memory.cli_io import load_corpus, read_json

_VALID_DIGEST = "a" * 64
_OMIT = object()


def _entry(source: Path, label: str = "a", path: str | None = None) -> dict[str, object]:
    """Build a corpus entry whose sha256 matches the on-disk source bytes."""
    return {
        "label": label,
        "path": source.name if path is None else path,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _manifest(directory: Path, entries: object, **overrides: object) -> Path:
    """Write a corpus manifest with valid defaults; overrides tune one boundary.

    Passing ``key=_OMIT`` drops that key entirely so a missing-field case is exercised.
    """
    payload: dict[str, object] = {
        "schema_version": 1,
        "encoder_checkpoint": "enc",
        "encoder_digest": _VALID_DIGEST,
        "entries": entries,
    }
    payload.update(overrides)
    payload = {key: value for key, value in payload.items() if value is not _OMIT}
    manifest = directory / "corpus.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _source(directory: Path, name: str, text: str) -> Path:
    source = directory / name
    source.write_text(text, encoding="utf-8")
    return source


def test_read_json_rejects_a_non_object_root(tmp_path: Path) -> None:
    document = tmp_path / "x.json"
    document.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        read_json(document)


def test_read_json_returns_a_real_object_document(tmp_path: Path) -> None:
    document = tmp_path / "x.json"
    document.write_text('{"value": 7}', encoding="utf-8")
    assert read_json(document) == {"value": 7}


def test_load_corpus_rejects_a_non_object_root(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        load_corpus(manifest)


def test_load_corpus_reads_verifies_and_returns_unicode(tmp_path: Path) -> None:
    first = _source(tmp_path, "a.md", "alpha — Ünïcödé αβγ ✓")
    second = _source(tmp_path, "b.md", "beta content")
    manifest = _manifest(tmp_path, [_entry(first, "alpha"), _entry(second, "beta")])
    labels, texts, encoder_path, digest, corpus_digest = load_corpus(manifest)
    assert labels == ["alpha", "beta"]
    assert texts == ["alpha — Ünïcödé αβγ ✓", "beta content"]
    assert encoder_path == (tmp_path / "enc").resolve()
    assert digest == _VALID_DIGEST
    assert corpus_digest == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_load_corpus_rejects_a_missing_schema_version(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [], schema_version=_OMIT)
    with pytest.raises(ValueError, match="integer schema_version 1"):
        load_corpus(manifest)


def test_load_corpus_rejects_a_wrong_schema_version(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [], schema_version=2)
    with pytest.raises(ValueError, match="integer schema_version 1"):
        load_corpus(manifest)


def test_load_corpus_rejects_a_boolean_schema_version(tmp_path: Path) -> None:
    # JSON ``true`` deserialises to Python ``True`` and ``True == 1``; the exact-int
    # guard must still reject it.
    manifest = _manifest(tmp_path, [], schema_version=True)
    with pytest.raises(ValueError, match="integer schema_version 1"):
        load_corpus(manifest)


def test_load_corpus_rejects_every_non_integer_schema_version(tmp_path: Path) -> None:
    values: tuple[object, ...] = ("1", 1.0, None, [], {})
    for value in values:
        manifest = _manifest(tmp_path, [], schema_version=value)
        with pytest.raises(ValueError, match="integer schema_version 1"):
            load_corpus(manifest)


def test_load_corpus_requires_entries_and_encoder_checkpoint(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "not-a-list")
    with pytest.raises(ValueError, match="requires entries and encoder_checkpoint"):
        load_corpus(manifest)


def test_load_corpus_rejects_a_missing_encoder_digest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [], encoder_digest=_OMIT)
    with pytest.raises(ValueError, match="encoder_digest must be 64 lowercase"):
        load_corpus(manifest)


def test_load_corpus_rejects_an_uppercase_encoder_digest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [], encoder_digest="A" * 64)
    with pytest.raises(ValueError, match="encoder_digest must be 64 lowercase"):
        load_corpus(manifest)


def test_load_corpus_rejects_a_short_encoder_digest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [], encoder_digest="abc")
    with pytest.raises(ValueError, match="encoder_digest must be 64 lowercase"):
        load_corpus(manifest)


def test_load_corpus_rejects_all_malformed_encoder_digests(tmp_path: Path) -> None:
    for value in (None, 7, "g" * 64, "a" * 63, "a" * 65):
        manifest = _manifest(tmp_path, [], encoder_digest=value)
        with pytest.raises(ValueError, match="encoder_digest must be 64 lowercase"):
            load_corpus(manifest)


def test_load_corpus_rejects_a_non_string_path(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [{"label": "a", "path": 123, "sha256": _VALID_DIGEST}])
    with pytest.raises(ValueError, match="string path and label"):
        load_corpus(manifest)


def test_load_corpus_rejects_an_empty_label(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [{"label": "", "path": "a.md", "sha256": _VALID_DIGEST}])
    with pytest.raises(ValueError, match="non-empty label without surrounding whitespace"):
        load_corpus(manifest)


def test_load_corpus_rejects_every_non_string_label(tmp_path: Path) -> None:
    labels: tuple[object, ...] = (None, 7, True, [], {})
    for label in labels:
        manifest = _manifest(
            tmp_path,
            [{"label": label, "path": "a.md", "sha256": _VALID_DIGEST}],
        )
        with pytest.raises(ValueError, match="non-empty label without surrounding whitespace"):
            load_corpus(manifest)


def test_load_corpus_rejects_a_whitespace_only_label(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [{"label": "   ", "path": "a.md", "sha256": _VALID_DIGEST}])
    with pytest.raises(ValueError, match="non-empty label without surrounding whitespace"):
        load_corpus(manifest)


def test_load_corpus_rejects_a_label_with_surrounding_whitespace(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [{"label": " a ", "path": "a.md", "sha256": _VALID_DIGEST}])
    with pytest.raises(ValueError, match="non-empty label without surrounding whitespace"):
        load_corpus(manifest)


def test_load_corpus_rejects_a_duplicate_label(tmp_path: Path) -> None:
    first = _source(tmp_path, "a.md", "alpha")
    second = _source(tmp_path, "b.md", "beta")
    manifest = _manifest(tmp_path, [_entry(first, "same"), _entry(second, "same")])
    with pytest.raises(ValueError, match="duplicate corpus label: same"):
        load_corpus(manifest)


def test_load_corpus_rejects_an_uppercase_entry_sha256(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [{"label": "a", "path": "a.md", "sha256": "A" * 64}])
    with pytest.raises(ValueError, match="corpus entry a sha256 must be 64 lowercase"):
        load_corpus(manifest)


def test_load_corpus_rejects_all_malformed_entry_digests(tmp_path: Path) -> None:
    for value in (None, 7, "g" * 64, "a" * 63, "a" * 65):
        manifest = _manifest(
            tmp_path,
            [{"label": "a", "path": "a.md", "sha256": value}],
        )
        with pytest.raises(ValueError, match="corpus entry a sha256 must be 64 lowercase"):
            load_corpus(manifest)


def test_load_corpus_rejects_a_source_digest_mismatch(tmp_path: Path) -> None:
    _source(tmp_path, "a.md", "real content")
    manifest = _manifest(tmp_path, [{"label": "a", "path": "a.md", "sha256": "0" * 64}])
    with pytest.raises(ValueError, match="source digest mismatch"):
        load_corpus(manifest)


def test_load_corpus_propagates_invalid_utf8_from_a_real_source(tmp_path: Path) -> None:
    source = tmp_path / "bad.bin"
    source.write_bytes(b"\xff\xfe not valid utf-8")
    entry = {
        "label": "a",
        "path": "bad.bin",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    manifest = _manifest(tmp_path, [entry])
    with pytest.raises(UnicodeDecodeError):
        load_corpus(manifest)
