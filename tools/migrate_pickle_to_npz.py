#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — One-shot legacy pickle → npz/json migrator

"""Convert legacy pickle state files to npz / gzip-JSON.

Before v0.4, SNN state and the memory index were serialised with
``pickle``, which makes load() an arbitrary-code-execution primitive.
All production loaders have since moved to ``numpy.savez`` / gzipped
JSON; the pickle path was kept as a fallback for backwards compat.

This tool closes that fallback: it scans known state paths, reads
each legacy ``*.pkl``, writes the equivalent ``*.npz`` (dict of
ndarrays) or ``*.json.gz`` beside it, and renames the original to
``*.pkl.bak`` so operators can verify before deletion.

The migrator does **not** delete the backup. Run this script once,
verify the new files load, then remove the ``.pkl.bak`` files manually.

Usage::

    python tools/migrate_pickle_to_npz.py                # default paths
    python tools/migrate_pickle_to_npz.py --dry-run      # no writes
    python tools/migrate_pickle_to_npz.py --path FILE    # one file
    python tools/migrate_pickle_to_npz.py --path DIR     # recursive
    python tools/migrate_pickle_to_npz.py --restore      # undo migration

Exit codes:
    0 — success, or nothing to migrate
    1 — at least one file failed to migrate (details on stderr)
    2 — invalid arguments
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle  # noqa: S403  # nosec B403 — migrator consumes legacy format by design
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

DEFAULT_SCAN_DIRS: tuple[Path, ...] = (
    REPO / "snn_state",
    REPO / "memory",
    REPO / "checkpoints",
    REPO / "data",
)


def _looks_like_dict_of_arrays(obj: object) -> bool:
    return isinstance(obj, dict) and all(
        isinstance(v, (np.ndarray, list, tuple, int, float, str, bytes)) for v in obj.values()
    )


def _pickle_load(path: Path) -> object:
    """Trusted load. Caller must have vetted the source path.

    The whole point of this migrator is to consume legacy pickle once
    and emit a safe format; bandit is suppressed on the specific line.
    """
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301  # nosec B301


def _write_npz(path: Path, data: dict) -> None:
    """Write dict to npz, coercing everything to np.ndarray.

    Scalars are stored as 0-d arrays (``np.asarray(v)``) so consumers
    that do ``int(data["n"])`` keep working — wrapping in a list would
    produce a 1-d array and break scalar coercion.
    """
    arrays: dict[str, np.ndarray] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            arrays[k] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def _write_json_gz(path: Path, data: object) -> None:
    """Write arbitrary JSON-serialisable data to gzipped JSON."""

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(f"not JSON-serialisable: {type(o).__name__}")

    payload = json.dumps(data, default=_default, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(payload)


def _migrate_one(pkl: Path, *, dry_run: bool) -> str:
    """Migrate a single pickle file. Return a one-line status string."""
    if pkl.suffix != ".pkl":
        return f"skip (not .pkl): {pkl}"

    try:
        obj = _pickle_load(pkl)
    except Exception as e:
        return f"FAIL   load    {pkl}: {type(e).__name__}: {e}"

    # Decide target format by shape of payload.
    if _looks_like_dict_of_arrays(obj):
        target = pkl.with_suffix(".npz")
        kind = "npz"
    else:
        target = pkl.with_suffix(".json.gz")
        kind = "json.gz"

    if dry_run:
        return f"would  {kind:<7} {pkl} -> {target.name}"

    try:
        if kind == "npz":
            _write_npz(target, obj)
        else:
            _write_json_gz(target, obj)
    except Exception as e:
        return f"FAIL   write   {pkl}: {type(e).__name__}: {e}"

    backup = pkl.with_suffix(".pkl.bak")
    try:
        pkl.rename(backup)
    except Exception as e:
        return f"FAIL   backup  {pkl}: {type(e).__name__}: {e}"

    return f"ok     {kind:<7} {pkl} -> {target.name} (backup: {backup.name})"


def _restore_one(bak: Path, *, dry_run: bool) -> str:
    """Undo a migration by renaming *.pkl.bak back to *.pkl."""
    if not bak.name.endswith(".pkl.bak"):
        return f"skip (not .pkl.bak): {bak}"
    target = bak.with_suffix("")  # strips .bak → leaves .pkl
    if dry_run:
        return f"would  restore {bak} -> {target.name}"
    target = Path(str(bak)[:-4])  # drop ".bak"
    try:
        bak.rename(target)
    except Exception as e:
        return f"FAIL   restore {bak}: {type(e).__name__}: {e}"
    return f"ok     restore {bak} -> {target.name}"


def _iter_targets(paths: list[Path], *, suffix: str) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_file() and p.name.endswith(suffix):
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob(f"*{suffix}")))
    return found


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], add_help=True)
    p.add_argument(
        "--path",
        action="append",
        type=Path,
        help="extra file or directory to scan (default: snn_state/, memory/, checkpoints/, data/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be migrated without writing anything",
    )
    p.add_argument(
        "--restore",
        action="store_true",
        help="rename *.pkl.bak back to *.pkl (reverse of a migration)",
    )
    args = p.parse_args(argv)

    roots: list[Path] = list(args.path) if args.path else list(DEFAULT_SCAN_DIRS)

    suffix = ".pkl.bak" if args.restore else ".pkl"
    targets = _iter_targets(roots, suffix=suffix)

    if not targets:
        print(f"no legacy {suffix} files found under {', '.join(str(r) for r in roots)}")
        return 0

    print(f"{'action':<11}{'kind':<8}path")
    print("-" * 72)
    fail = 0
    for t in targets:
        line = (
            _restore_one(t, dry_run=args.dry_run)
            if args.restore
            else _migrate_one(t, dry_run=args.dry_run)
        )
        print(line)
        if line.startswith("FAIL"):
            fail += 1
    print("-" * 72)
    print(f"processed={len(targets)}  failures={fail}  dry_run={args.dry_run}")
    return 1 if fail else 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
