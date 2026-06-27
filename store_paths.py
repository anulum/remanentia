# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — canonical memory store path selection

"""Resolve Remanentia's canonical memory-store filesystem layout.

The MS.0 reconciliation lane has several production surfaces that must agree on
one store: hub-backed finding ingest, feed-backed finding ingest, and index
freshness monitoring. This module owns that path contract. Operators may select
the repository-local store with ``REMANENTIA_BASE`` and may point the stimuli
firehose at an external directory with ``REMANENTIA_STIMULI_DIR``; all remaining
paths derive from the same base so cursors, findings, digests, and vector-index
artifacts cannot silently diverge.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FINDING_CURSOR_NAME = "finding_ingest_cursor.json"
"""Default hub-event cursor filename under ``memory/semantic``."""

DEFAULT_FEED_CURSOR_NAME = "synapse_feed_cursor.json"
"""Default feed-line cursor filename under ``memory/semantic``."""


@dataclass(frozen=True)
class StageRoot:
    """Filesystem selection for one freshness-monitor pipeline stage.

    Parameters
    ----------
    name
        Stable stage label emitted in freshness reports.
    root
        Directory scanned for artifacts in this stage.
    patterns
        Glob patterns, relative to ``root``, selecting artifacts for the stage.
    """

    name: str
    root: Path
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class StorePaths:
    """Resolved filesystem contract for one Remanentia memory store.

    Parameters
    ----------
    base
        Repository or deployment root that owns the durable memory store.
    stimuli_dir
        Source firehose directory. It may live outside ``base``.
    semantic_dir
        Semantic-memory root under ``base``.
    findings_dir
        Markdown finding sink used by hub and feed ingest.
    digests_dir
        Consolidated digest root.
    vector_index_dir
        Retriever vector-index artifact root.
    graph_dir
        Entity/relation graph artifact root.
    traces_dir
        Episodic reasoning-trace root.
    state_dir
        Runtime state root used by workers and watchdog reports.
    consolidation_dir
        Consolidation bookkeeping root.
    finding_cursor
        Hub-event cursor file.
    feed_cursor
        Feed-line cursor file.
    freshness_report
        JSON report written by the index-freshness watchdog.
    """

    base: Path
    stimuli_dir: Path
    semantic_dir: Path
    findings_dir: Path
    digests_dir: Path
    vector_index_dir: Path
    graph_dir: Path
    traces_dir: Path
    state_dir: Path
    consolidation_dir: Path
    finding_cursor: Path
    feed_cursor: Path
    freshness_report: Path

    def freshness_stage_roots(self) -> tuple[StageRoot, ...]:
        """Return the canonical source-to-sink freshness-monitor chain."""
        return (
            StageRoot("stimuli", self.stimuli_dir, ("*.json",)),
            StageRoot("findings", self.semantic_dir, ("**/*.md",)),
            StageRoot("digests", self.digests_dir, ("*.md",)),
            StageRoot("vector-index", self.vector_index_dir, ("*.npz", "*.sqlite")),
        )


def default_base(base: str | Path | None = None) -> Path:
    """Return the selected memory-store base directory.

    The explicit ``base`` argument wins, then ``REMANENTIA_BASE``, then the
    directory containing this module for a repository-local default.
    """
    if base is not None:
        return Path(base)
    env_base = os.environ.get("REMANENTIA_BASE")
    if env_base:
        return Path(env_base)
    return Path(__file__).resolve().parent


def default_stimuli_dir(
    *,
    base: str | Path | None = None,
    stimuli_dir: str | Path | None = None,
) -> Path:
    """Return the selected stimuli firehose directory."""
    if stimuli_dir is not None:
        return Path(stimuli_dir)
    env_stimuli = os.environ.get("REMANENTIA_STIMULI_DIR")
    if env_stimuli:
        return Path(env_stimuli)
    return default_base(base) / "snn_stimuli"


def resolve_store_paths(
    *,
    base: str | Path | None = None,
    stimuli_dir: str | Path | None = None,
) -> StorePaths:
    """Resolve all canonical memory-store paths from operator selections."""
    root = default_base(base)
    semantic_dir = root / "memory" / "semantic"
    return StorePaths(
        base=root,
        stimuli_dir=default_stimuli_dir(base=root, stimuli_dir=stimuli_dir),
        semantic_dir=semantic_dir,
        findings_dir=semantic_dir / "findings",
        digests_dir=root / "memory" / "digests",
        vector_index_dir=root / "snn_state" / "vector_index",
        graph_dir=root / "memory" / "graph",
        traces_dir=root / "reasoning_traces",
        state_dir=root / "snn_state",
        consolidation_dir=root / "consolidation",
        finding_cursor=semantic_dir / DEFAULT_FINDING_CURSOR_NAME,
        feed_cursor=semantic_dir / DEFAULT_FEED_CURSOR_NAME,
        freshness_report=root / "snn_state" / "index_freshness.json",
    )


def default_findings_dir(base: str | Path | None = None) -> Path:
    """Return the canonical Markdown finding sink directory."""
    return resolve_store_paths(base=base).findings_dir


def default_finding_cursor(base: str | Path | None = None) -> Path:
    """Return the canonical hub-event cursor path."""
    return resolve_store_paths(base=base).finding_cursor


def default_feed_cursor(base: str | Path | None = None) -> Path:
    """Return the canonical feed-line cursor path."""
    return resolve_store_paths(base=base).feed_cursor
