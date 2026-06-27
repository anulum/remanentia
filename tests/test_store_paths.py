# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for canonical memory store path selection

"""Tests for :mod:`store_paths`.

The resolver is the shared production contract for the MS.0 canonical memory
store: feed ingest, hub ingest, and freshness monitoring must all point at the
same corpus roots unless an operator explicitly overrides the source firehose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from store_paths import (
    DEFAULT_FEED_CURSOR_NAME,
    DEFAULT_FINDING_CURSOR_NAME,
    StageRoot,
    default_base,
    default_feed_cursor,
    default_finding_cursor,
    default_findings_dir,
    resolve_store_paths,
)


def test_resolver_uses_explicit_base_and_default_corpus_layout(tmp_path: Path) -> None:
    """The canonical repo-local store layout is derived from one base root."""
    paths = resolve_store_paths(base=tmp_path)

    assert paths.base == tmp_path
    assert paths.stimuli_dir == tmp_path / "snn_stimuli"
    assert paths.semantic_dir == tmp_path / "memory" / "semantic"
    assert paths.findings_dir == tmp_path / "memory" / "semantic" / "findings"
    assert paths.digests_dir == tmp_path / "memory" / "digests"
    assert paths.vector_index_dir == tmp_path / "snn_state" / "vector_index"
    assert paths.memory_index == tmp_path / "snn_state" / "memory_index.json.gz"
    assert paths.memory_sources_config == tmp_path / "snn_state" / "memory_sources.json"
    assert paths.finding_cursor == tmp_path / "memory" / "semantic" / DEFAULT_FINDING_CURSOR_NAME
    assert paths.feed_cursor == tmp_path / "memory" / "semantic" / DEFAULT_FEED_CURSOR_NAME


def test_default_base_falls_back_to_module_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh checkout has a repository-local store without env configuration."""
    monkeypatch.delenv("REMANENTIA_BASE", raising=False)

    assert default_base() == Path(__file__).resolve().parents[1]


def test_resolver_accepts_external_stimuli_firehose(tmp_path: Path) -> None:
    """A live firehose can sit outside the repository without moving the store."""
    firehose = tmp_path / "external" / "stimuli"
    paths = resolve_store_paths(base=tmp_path / "repo", stimuli_dir=firehose)

    assert paths.base == tmp_path / "repo"
    assert paths.stimuli_dir == firehose
    assert paths.findings_dir == tmp_path / "repo" / "memory" / "semantic" / "findings"


def test_environment_overrides_match_cli_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolver honours the existing REMANENTIA_* operator variables."""
    base = tmp_path / "runtime"
    firehose = tmp_path / "firehose"
    monkeypatch.setenv("REMANENTIA_BASE", str(base))
    monkeypatch.setenv("REMANENTIA_STIMULI_DIR", str(firehose))

    paths = resolve_store_paths()

    assert paths.base == base
    assert paths.stimuli_dir == firehose
    assert default_findings_dir() == base / "memory" / "semantic" / "findings"
    assert default_finding_cursor() == base / "memory" / "semantic" / DEFAULT_FINDING_CURSOR_NAME
    assert default_feed_cursor() == base / "memory" / "semantic" / DEFAULT_FEED_CURSOR_NAME


def test_freshness_stage_roots_are_ordered_source_to_sink(tmp_path: Path) -> None:
    """Freshness monitoring consumes the same canonical source-to-sink chain."""
    paths = resolve_store_paths(base=tmp_path)

    assert paths.freshness_stage_roots() == (
        StageRoot("stimuli", tmp_path / "snn_stimuli", ("*.json",)),
        StageRoot("findings", tmp_path / "memory" / "semantic", ("**/*.md",)),
        StageRoot("digests", tmp_path / "memory" / "digests", ("*.md",)),
        StageRoot("memory-index", tmp_path / "snn_state", ("memory_index.json.gz",)),
        StageRoot("vector-index", tmp_path / "snn_state" / "vector_index", ("*.npz", "*.sqlite")),
    )
