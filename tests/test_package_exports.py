# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Persistent neural memory for AI agents

"""Tests for the top-level public export facade."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import sys
from pathlib import Path
from typing import cast
from types import ModuleType

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility branch.
    import tomli as tomllib


EXPECTED_EXPORTS: tuple[str, ...] = (
    "retrieve",
    "retrieve_context",
    "retrieval_history",
    "related_traces",
    "query_suggestions",
    "trace_summaries",
    "chunk_traces",
    "drop_stimulus",
    "heartbeat",
    "extract_skills",
    "query_skills",
    "load_skills",
    "snapshot_save",
    "snapshot_load",
    "continuity_score",
    "reconstruction_prompt",
    "consult_memory",
    "decision_guard",
)


def _load_public_facade() -> ModuleType:
    """Import the distribution's flat public facade module."""
    return importlib.import_module("__init__")


def test_public_facade_exports_expected_symbols() -> None:
    """Verify the advertised facade export list stays stable and complete."""
    module = _load_public_facade()

    assert tuple(module.__all__) == EXPECTED_EXPORTS


def test_public_facade_exported_symbols_resolve_to_callables() -> None:
    """Resolve every advertised export against the real runtime imports."""
    module = _load_public_facade()
    exported_names = tuple(module.__all__)

    missing = [name for name in exported_names if not hasattr(module, name)]
    non_callables = [name for name in exported_names if not callable(getattr(module, name))]

    assert missing == []
    assert non_callables == []


def test_public_facade_version_is_present() -> None:
    """Expose the same version as the project distribution metadata."""
    module = _load_public_facade()
    pyproject = cast(
        Mapping[str, object],
        tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8")),
    )
    project = pyproject["project"]
    assert isinstance(project, dict)
    version = project["version"]
    assert isinstance(version, str)

    assert module.__version__ == version
