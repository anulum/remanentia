# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Independent pinned-model gate precondition

"""Hard precondition shared by every explicitly selected model gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".snn_models" / "all-MiniLM-L6-v2"
MANIFESTS = (
    ROOT / "experiments/snn_memory/development_corpus.json",
    ROOT / "experiments/snn_memory/locked_evaluation_corpus.json",
)


def _framed_directory_digest(path: Path) -> str:
    """Independently compute the versioned encoder-tree digest contract."""
    digest = hashlib.sha256()
    digest.update(b"remanentia.encoder-directory.v1\x00")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise AssertionError(f"pinned local encoder is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        content = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def require_pinned_model() -> str:
    """Return the live digest or hard-fail on absence or tracked pin drift."""
    if not MODEL.is_dir():
        raise AssertionError(f"pinned local encoder not provisioned: {MODEL}")
    declared = tuple(
        json.loads(path.read_text(encoding="utf-8"))["encoder_digest"] for path in MANIFESTS
    )
    if len(set(declared)) != 1:
        raise AssertionError(f"tracked corpus manifests disagree on encoder digest: {declared}")
    live = _framed_directory_digest(MODEL)
    if live != declared[0]:
        raise AssertionError(
            f"pinned local encoder digest drift: tracked={declared[0]} live={live}"
        )
    return live
