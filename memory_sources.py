# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Memory source configuration

"""Configuration loader for MemoryIndex source roots.

The public package ships with neutral, repository-local source roots. Operators
can add deployment-specific archives with a JSON file or inline JSON without
editing :mod:`memory_index`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast


CONFIG_PATH_ENV = "REMANENTIA_MEMORY_SOURCES_CONFIG"
INLINE_CONFIG_ENV = "REMANENTIA_MEMORY_SOURCES_JSON"
SOURCE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DEFAULT_TEXT_EXTENSIONS = frozenset({".md", ".txt", ".json", ".jsonl", ".yaml", ".yml"})


@dataclass(frozen=True)
class SourceConfig:
    """Resolved MemoryIndex source roots and file-extension allow lists.

    Attributes
    ----------
    sources:
        Mapping from stable source labels to resolved filesystem roots.
    extensions:
        Mapping from source labels to allowed file suffixes. Missing labels use
        the broad text-document default in :mod:`memory_index`.
    """

    sources: dict[str, Path]
    extensions: dict[str, frozenset[str]]


@dataclass(frozen=True)
class _DefaultSourceSpec:
    """Repository-local default source specification."""

    label: str
    relative_path: str
    extensions: frozenset[str]


DEFAULT_SOURCE_SPECS = (
    _DefaultSourceSpec("traces", "reasoning_traces", frozenset({".md"})),
    _DefaultSourceSpec("paper", "paper", frozenset({".md", ".tex", ".bib", ".txt"})),
    _DefaultSourceSpec("semantic", "memory/semantic", DEFAULT_TEXT_EXTENSIONS),
    _DefaultSourceSpec("compiled", "memory/compiled", frozenset({".md", ".jsonl"})),
    _DefaultSourceSpec("code_remanentia", ".", frozenset({".py"})),
)


def load_source_config(base: Path) -> SourceConfig:
    """Load source configuration from the process environment.

    Parameters
    ----------
    base:
        Repository root used to resolve neutral defaults and inline relative
        paths.

    Returns
    -------
    SourceConfig
        Resolved source roots and source-specific file suffix allow lists.
    """

    return build_source_config(base=base, environ=os.environ)


def build_source_config(
    *,
    base: Path,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> SourceConfig:
    """Build a source configuration from defaults plus optional JSON.

    Parameters
    ----------
    base:
        Repository root used for default roots and inline relative paths.
    config_path:
        Optional JSON configuration file. Relative paths inside that file are
        resolved from the file's parent directory.
    environ:
        Environment mapping. ``REMANENTIA_MEMORY_SOURCES_CONFIG`` points to a
        JSON file and ``REMANENTIA_MEMORY_SOURCES_JSON`` carries inline JSON.

    Returns
    -------
    SourceConfig
        The merged and resolved MemoryIndex source configuration.
    """

    env = environ or {}
    resolved_base = base.resolve()
    config_entries: list[tuple[dict[str, object], Path]] = []

    env_config_path = env.get(CONFIG_PATH_ENV, "").strip()
    if config_path is None and env_config_path:
        config_path = Path(env_config_path)
    if config_path is not None:
        config_path = config_path.expanduser().resolve()
        config_entries.append(
            (
                _require_config_object(json.loads(config_path.read_text(encoding="utf-8"))),
                config_path.parent,
            )
        )

    inline = env.get(INLINE_CONFIG_ENV, "").strip()
    if inline:
        config_entries.append((_require_config_object(json.loads(inline)), resolved_base))

    sources: dict[str, Path] = {}
    extensions: dict[str, frozenset[str]] = {}
    extends_defaults = all(_extends_defaults(raw_config) for raw_config, _ in config_entries)
    if extends_defaults:
        sources.update(_default_sources(resolved_base))
        extensions.update(_default_extensions())

    for raw_config, relative_root in config_entries:
        configured_sources, configured_extensions = _configured_sources(raw_config, relative_root)
        sources.update(configured_sources)
        extensions.update(configured_extensions)

    return SourceConfig(sources=sources, extensions=extensions)


def _default_sources(base: Path) -> dict[str, Path]:
    """Resolve repository-local default roots from a base directory."""

    return {spec.label: (base / spec.relative_path).resolve() for spec in DEFAULT_SOURCE_SPECS}


def _default_extensions() -> dict[str, frozenset[str]]:
    """Return the file suffix allow lists for neutral defaults."""

    return {spec.label: spec.extensions for spec in DEFAULT_SOURCE_SPECS}


def _require_config_object(raw_config: object) -> dict[str, object]:
    """Validate and return a raw JSON config object."""

    if not isinstance(raw_config, dict):
        raise ValueError("memory source config must be an object")
    return cast(dict[str, object], raw_config)


def _extends_defaults(raw_config: Mapping[str, object]) -> bool:
    """Return whether a raw JSON config keeps the neutral defaults."""

    value = raw_config.get("extends_defaults", True)
    if not isinstance(value, bool):
        raise ValueError("extends_defaults must be a boolean")
    return value


def _configured_sources(
    raw_config: Mapping[str, object],
    relative_root: Path,
) -> tuple[dict[str, Path], dict[str, frozenset[str]]]:
    """Parse configured sources from a raw JSON object."""

    raw_sources = raw_config.get("sources", {})
    if not isinstance(raw_sources, dict):
        raise ValueError("sources must be an object")

    sources: dict[str, Path] = {}
    extensions: dict[str, frozenset[str]] = {}
    for label, spec in raw_sources.items():
        if not isinstance(label, str) or not SOURCE_LABEL_PATTERN.fullmatch(label):
            raise ValueError(f"source label {label!r} must match {SOURCE_LABEL_PATTERN.pattern}")
        path, suffixes = _parse_source_spec(label, spec, relative_root)
        sources[label] = path
        extensions[label] = suffixes

    return sources, extensions


def _parse_source_spec(
    label: str,
    raw_spec: object,
    relative_root: Path,
) -> tuple[Path, frozenset[str]]:
    """Parse one configured source entry."""

    if isinstance(raw_spec, str):
        raw_path = raw_spec
        raw_extensions: object = None
    elif isinstance(raw_spec, dict):
        raw_path_value = raw_spec.get("path")
        if not isinstance(raw_path_value, str) or not raw_path_value.strip():
            raise ValueError(f"source {label!r} must define a non-empty path")
        raw_path = raw_path_value
        raw_extensions = raw_spec.get("extensions")
    else:
        raise ValueError(f"source {label!r} must be a string path or object")

    path = _resolve_source_path(raw_path, relative_root)
    suffixes = _parse_extensions(label, raw_extensions)
    return path, suffixes


def _resolve_source_path(raw_path: str, relative_root: Path) -> Path:
    """Resolve a configured path from a config-relative root."""

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = relative_root / candidate
    return candidate.resolve()


def _parse_extensions(label: str, raw_extensions: object) -> frozenset[str]:
    """Parse and normalise file suffixes for one source."""

    if raw_extensions is None:
        return DEFAULT_TEXT_EXTENSIONS
    if not isinstance(raw_extensions, list):
        raise ValueError(f"source {label!r} extensions must be a list")

    suffixes: set[str] = set()
    for raw_suffix in raw_extensions:
        if not isinstance(raw_suffix, str) or not raw_suffix.strip():
            raise ValueError(f"source {label!r} extensions must contain non-empty strings")
        suffix = raw_suffix.strip().lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        suffixes.add(suffix)

    if not suffixes:
        raise ValueError(f"source {label!r} extensions must not be empty")
    return frozenset(suffixes)
