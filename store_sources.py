# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — selected-store MemoryIndex source configuration

"""Write MemoryIndex source configs for selected memory-store backfills.

MS.0 needs the Mar-Jun stimulus firehose to become a real retrievable source
without reviving the legacy SNN daemon. This module emits the JSON configuration
consumed by :mod:`memory_sources` so ``MemoryIndex`` indexes the selected
``reasoning_traces``, semantic memory, compiled facts, and external
``snn_stimuli`` firehose through its normal build path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from store_paths import StorePaths, resolve_store_paths

DEFAULT_STORE_SOURCE_CONFIG_NAME = "memory_sources.json"
"""Default selected-source config filename under ``snn_state``."""


class SourceSpec(TypedDict):
    """JSON shape for one ``memory_sources`` source entry."""

    path: str
    extensions: list[str]


class StoreSourceConfig(TypedDict):
    """JSON shape consumed by ``memory_sources.build_source_config``."""

    extends_defaults: bool
    sources: dict[str, SourceSpec]


def build_store_source_config(
    *,
    base: str | Path | None = None,
    stimuli_dir: str | Path | None = None,
) -> StoreSourceConfig:
    """Build a MemoryIndex config for the selected store and firehose."""
    paths = resolve_store_paths(base=base, stimuli_dir=stimuli_dir)
    return {
        "extends_defaults": False,
        "sources": {
            "traces": _source(paths.traces_dir, [".md"]),
            "semantic": _source(
                paths.semantic_dir,
                [".md", ".txt", ".json", ".jsonl", ".yaml", ".yml"],
            ),
            "compiled": _source(paths.base / "memory" / "compiled", [".md", ".jsonl"]),
            "arcane_stimuli": _source(paths.stimuli_dir, [".json"]),
        },
    }


def default_source_config_path(paths: StorePaths) -> Path:
    """Return the default selected-source config path."""
    return paths.memory_sources_config


def write_store_source_config(
    *,
    base: str | Path | None = None,
    stimuli_dir: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    """Persist a selected-store MemoryIndex source config atomically."""
    paths = resolve_store_paths(base=base, stimuli_dir=stimuli_dir)
    config = build_store_source_config(base=paths.base, stimuli_dir=paths.stimuli_dir)
    out = default_source_config_path(paths) if output is None else Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def render_store_source_config(config: StoreSourceConfig) -> str:
    """Render a selected-source config for an operator terminal."""
    lines = ["MemoryIndex selected sources:"]
    for label, spec in config["sources"].items():
        extensions = ", ".join(spec["extensions"])
        lines.append(f"  {label:<15} {spec['path']} ({extensions})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the selected-source config CLI."""
    parser = argparse.ArgumentParser(
        prog="remanentia-store-sources",
        description="Show or persist MemoryIndex sources for the selected Remanentia store.",
    )
    parser.add_argument("--base", default=None, help="Override REMANENTIA_BASE for this run")
    parser.add_argument(
        "--stimuli-dir",
        default=None,
        help="Override REMANENTIA_STIMULI_DIR for this run",
    )
    parser.add_argument("--write", action="store_true", help="Write the source config JSON")
    parser.add_argument("--output", default=None, help="Config output path when writing")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args(argv)

    config = build_store_source_config(base=args.base, stimuli_dir=args.stimuli_dir)
    if args.write:
        write_store_source_config(base=args.base, stimuli_dir=args.stimuli_dir, output=args.output)
    if args.json:
        print(json.dumps(config, indent=2, sort_keys=True))
    else:
        print(render_store_source_config(config))
    return 0


def _source(path: Path, extensions: list[str]) -> SourceSpec:
    """Return one normalised source specification."""
    return {"path": str(path), "extensions": extensions}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
