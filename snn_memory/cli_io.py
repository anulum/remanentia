# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Model-free CLI manifest I/O

"""Model-free JSON and corpus-manifest loading shared by the CLI surfaces.

Kept in its own module (imported by both the model-free CLI and the model run
adapter) so the dependency graph is strictly one-way — the run adapter depends on
this, never the reverse — and so no scientific/model import is pulled in to read a
config or corpus file.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

CORPUS_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a UTF-8 file, rejecting a non-object document root."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def _require_sha256(value: object, context: str) -> str:
    """Return ``value`` if it is exactly 64 lowercase hex characters, else raise."""
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def load_corpus(manifest_path: Path) -> tuple[list[str], list[str], Path, str, str]:
    """Load a schema-versioned corpus manifest, verifying each source byte digest.

    Requires an integer ``schema_version == 1`` (a JSON boolean is rejected); requires
    the ``encoder_digest`` and every entry ``sha256`` to be exactly 64 lowercase hex
    characters; requires a unique ``label`` per entry that is a non-empty string with
    no surrounding whitespace, plus a string ``path``; reads every source as UTF-8.
    """
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: root must be an object")
    corpus_digest = hashlib.sha256(manifest_bytes).hexdigest()
    version = manifest.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CORPUS_SCHEMA_VERSION
    ):
        raise ValueError(f"corpus manifest requires integer schema_version {CORPUS_SCHEMA_VERSION}")
    entries = manifest.get("entries")
    checkpoint = manifest.get("encoder_checkpoint")
    if not isinstance(entries, list) or not isinstance(checkpoint, str):
        raise ValueError("corpus manifest requires entries and encoder_checkpoint")
    encoder_digest = _require_sha256(manifest.get("encoder_digest"), "encoder_digest")
    labels: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("each corpus entry requires a string path and label")
        label = entry.get("label")
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            raise ValueError(
                "each corpus entry requires a non-empty label without surrounding whitespace"
            )
        if label in seen:
            raise ValueError(f"duplicate corpus label: {label}")
        seen.add(label)
        digest = _require_sha256(entry.get("sha256"), f"corpus entry {label} sha256")
        source = (manifest_path.parent / entry["path"]).resolve()
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"corpus source digest mismatch: {source}")
        labels.append(label)
        texts.append(content.decode("utf-8"))
    encoder_path = (manifest_path.parent / checkpoint).resolve()
    return labels, texts, encoder_path, encoder_digest, corpus_digest
