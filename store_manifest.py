# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — canonical store selection manifest

"""Build and persist the operator-visible memory-store selection manifest.

MS.0 needs an explicit record of which corpus is selected before backlog
reconsolidation mutates any live store. This module resolves the same
``REMANENTIA_BASE`` / ``REMANENTIA_STIMULI_DIR`` contract as ingest,
freshness checks, and CLI status; it then records the selected paths and current
artifact counts in a small JSON manifest under ``snn_state``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from store_paths import StorePaths, resolve_store_paths

DEFAULT_STORE_SELECTION_NAME = "store_selection.json"
"""Default filename for the selected-store manifest under ``snn_state``."""


@dataclass(frozen=True)
class ArtifactReading:
    """Artifact count, byte total, and newest mtime for one store stage."""

    root: Path
    patterns: tuple[str, ...]
    count: int
    bytes: int
    newest: float | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable artifact reading."""
        return {
            "root": str(self.root),
            "patterns": list(self.patterns),
            "count": self.count,
            "bytes": self.bytes,
            "newest": self.newest,
        }


@dataclass(frozen=True)
class StoreSelectionManifest:
    """Resolved store selection plus artifact readings for reconsolidation."""

    paths: StorePaths
    checked_at_unix: float
    artifacts: dict[str, ArtifactReading]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable selected-store manifest."""
        return {
            "selected_base": str(self.paths.base),
            "stimuli_dir": str(self.paths.stimuli_dir),
            "semantic_dir": str(self.paths.semantic_dir),
            "findings_dir": str(self.paths.findings_dir),
            "digests_dir": str(self.paths.digests_dir),
            "graph_dir": str(self.paths.graph_dir),
            "traces_dir": str(self.paths.traces_dir),
            "state_dir": str(self.paths.state_dir),
            "vector_index_dir": str(self.paths.vector_index_dir),
            "memory_index": str(self.paths.memory_index),
            "memory_sources_config": str(self.paths.memory_sources_config),
            "freshness_report": str(self.paths.freshness_report),
            "freshness_report_present": self.paths.freshness_report.is_file(),
            "checked_at_unix": self.checked_at_unix,
            "artifacts": {name: reading.as_dict() for name, reading in self.artifacts.items()},
        }


def _artifact_reading(root: Path, patterns: Sequence[str]) -> ArtifactReading:
    """Scan one real store stage without mutating it."""
    count = 0
    total_bytes = 0
    newest: float | None = None
    if root.is_dir():
        for pattern in patterns:
            for path in root.glob(pattern):
                if not path.is_file():
                    continue
                stat = path.stat()
                count += 1
                total_bytes += stat.st_size
                if newest is None or stat.st_mtime > newest:
                    newest = stat.st_mtime
    return ArtifactReading(
        root=root,
        patterns=tuple(patterns),
        count=count,
        bytes=total_bytes,
        newest=newest,
    )


def build_store_manifest(
    *,
    base: str | Path | None = None,
    stimuli_dir: str | Path | None = None,
    checked_at: float | None = None,
) -> StoreSelectionManifest:
    """Resolve the selected store and read its current artifact inventory."""
    paths = resolve_store_paths(base=base, stimuli_dir=stimuli_dir)
    artifacts = {
        "stimuli": _artifact_reading(paths.stimuli_dir, ("*.json",)),
        "traces": _artifact_reading(paths.traces_dir, ("*.md",)),
        "semantic": _artifact_reading(paths.semantic_dir, ("**/*.md",)),
        "findings": _artifact_reading(paths.findings_dir, ("*.md",)),
        "digests": _artifact_reading(paths.digests_dir, ("*.md",)),
        "memory_index": _artifact_reading(paths.state_dir, ("memory_index.json.gz",)),
        "graph": _artifact_reading(paths.graph_dir, ("*.json", "*.jsonl")),
        "vector_index": _artifact_reading(paths.vector_index_dir, ("*.npz", "*.sqlite")),
    }
    return StoreSelectionManifest(
        paths=paths,
        checked_at_unix=time.time() if checked_at is None else checked_at,
        artifacts=artifacts,
    )


def default_manifest_path(manifest: StoreSelectionManifest) -> Path:
    """Return the default write path for a selected-store manifest."""
    return manifest.paths.state_dir / DEFAULT_STORE_SELECTION_NAME


def write_store_manifest(
    manifest: StoreSelectionManifest,
    path: str | Path | None = None,
) -> Path:
    """Persist a selected-store manifest atomically and return its path."""
    out = default_manifest_path(manifest) if path is None else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(out)
    return out


def render_store_manifest(manifest: StoreSelectionManifest) -> str:
    """Render the selected store manifest for an operator terminal."""
    lines = [
        f"Selected store: {manifest.paths.base}",
        f"Stimuli dir: {manifest.paths.stimuli_dir}",
        f"freshness report: {manifest.paths.freshness_report}",
        "Artifacts:",
    ]
    for name, reading in manifest.artifacts.items():
        newest = "none" if reading.newest is None else f"{reading.newest:.0f}"
        lines.append(
            f"  {name:<12} {reading.count:>5} file(s), {reading.bytes:>8} bytes, newest={newest}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected-store manifest CLI."""
    parser = argparse.ArgumentParser(
        prog="remanentia-store-manifest",
        description="Show or persist the resolved Remanentia memory-store selection.",
    )
    parser.add_argument("--base", default=None, help="Override REMANENTIA_BASE for this run")
    parser.add_argument(
        "--stimuli-dir",
        default=None,
        help="Override REMANENTIA_STIMULI_DIR for this run",
    )
    parser.add_argument("--write", action="store_true", help="Write the manifest JSON")
    parser.add_argument("--output", default=None, help="Manifest output path when writing")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args(argv)

    manifest = build_store_manifest(base=args.base, stimuli_dir=args.stimuli_dir)
    if args.write:
        write_store_manifest(manifest, args.output)
    if args.json:
        print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
    else:
        print(render_store_manifest(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
