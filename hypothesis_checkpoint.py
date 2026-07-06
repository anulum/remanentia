# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Resumable checkpoint for benchmark hypothesis records

"""Persist benchmark hypotheses incrementally so a long run survives a kill.

A full-S LongMemEval pass over a slow local reader takes tens of hours; the first
sovereign run measured ~30.5 h on one card. Holding every hypothesis in memory and
flushing only at the end means a single interruption — an OOM reap, a power blip, a
mistaken kill — discards the whole run. This module writes each record to a JSONL
file the moment it is produced and reloads the completed records on resume, skipping
a blank or half-written final line left by an interrupted append rather than
aborting the load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["append_record", "completed_ids", "load_completed"]


def load_completed(path: Path) -> list[dict[str, Any]]:
    """Return the hypothesis records already persisted at *path*.

    A missing file yields an empty list. Each line is parsed independently so a
    blank line or an unparsable trailing fragment — the signature of a write
    interrupted mid-record — is skipped instead of failing the whole load. Only
    JSON objects carrying a ``question_id`` are returned; anything else (a bare
    list, a number, an object without the key) is ignored as not-a-record.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "question_id" in obj:
            records.append(obj)
    return records


def completed_ids(records: list[dict[str, Any]]) -> set[str]:
    """The set of ``question_id`` values (as strings) across *records*.

    Records missing the key are silently excluded so the caller can pass raw
    loaded data without pre-filtering.
    """
    return {str(r["question_id"]) for r in records if "question_id" in r}


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append *record* as one JSONL line to *path*, forcing it to disk.

    The parent directory is created on demand. The stream is flushed and
    ``fsync``-ed so a record that returns from this call survives a crash of the
    process or the machine — the durability guarantee the resume path relies on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
