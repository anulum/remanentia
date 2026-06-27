# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for canonical store selection manifests

"""Tests for :mod:`store_manifest`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest

from store_manifest import build_store_manifest, main, render_store_manifest, write_store_manifest


def _touch(path: Path, *, text: str = "x", mtime: float = 1_700_000_000.0) -> Path:
    """Create a real artifact and pin its mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_manifest_records_selected_store_paths_and_artifacts(tmp_path: Path) -> None:
    """The manifest makes the selected corpus and its current artifacts explicit."""
    store = tmp_path / "store"
    firehose = tmp_path / "stimuli"
    _touch(firehose / "a.json", text="aa", mtime=100.0)
    _touch(store / "memory" / "semantic" / "findings" / "f.md", text="fff", mtime=200.0)
    _touch(store / "memory" / "graph" / "entities.jsonl", text="{}\n", mtime=300.0)
    (store / "memory" / "graph" / "not-an-artifact.json").mkdir(parents=True)
    _touch(store / "snn_state" / "memory_index.json.gz", text="idx", mtime=350.0)
    _touch(store / "snn_state" / "vector_index" / "index.npz", text="vec", mtime=400.0)
    _touch(store / "snn_state" / "index_freshness.json", text='{"stale": true}', mtime=500.0)

    manifest = build_store_manifest(base=store, stimuli_dir=firehose, checked_at=1234.0)
    data = manifest.as_dict()

    assert data["selected_base"] == str(store)
    assert data["stimuli_dir"] == str(firehose)
    assert data["freshness_report"] == str(store / "snn_state" / "index_freshness.json")
    assert data["checked_at_unix"] == 1234.0
    artifacts = cast(dict[str, dict[str, object]], data["artifacts"])
    assert artifacts["stimuli"]["count"] == 1
    assert artifacts["stimuli"]["bytes"] == 2
    assert artifacts["semantic"]["count"] == 1
    assert artifacts["graph"]["count"] == 1
    assert artifacts["memory_index"]["count"] == 1
    assert artifacts["vector_index"]["count"] == 1
    assert data["freshness_report_present"] is True


def test_write_manifest_uses_state_dir_by_default(tmp_path: Path) -> None:
    """Writing the manifest leaves a durable selection record in the store state."""
    manifest = build_store_manifest(base=tmp_path, checked_at=50.0)

    path = write_store_manifest(manifest)

    assert path == tmp_path / "snn_state" / "store_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected_base"] == str(tmp_path)
    assert not path.with_suffix(".json.tmp").exists()


def test_render_manifest_is_operator_readable(tmp_path: Path) -> None:
    """The text renderer names the selected store and each audited stage."""
    _touch(tmp_path / "reasoning_traces" / "trace.md")

    text = render_store_manifest(build_store_manifest(base=tmp_path, checked_at=60.0))

    assert f"Selected store: {tmp_path}" in text
    assert "traces" in text
    assert "freshness report" in text


def test_main_prints_json_and_writes_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The standalone CLI writes the selected-store manifest and prints JSON."""
    out_path = tmp_path / "selection.json"

    assert main(["--base", str(tmp_path), "--write", "--output", str(out_path), "--json"]) == 0

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert printed["selected_base"] == str(tmp_path)
    assert written["selected_base"] == str(tmp_path)


def test_main_prints_text_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The standalone CLI has a human-readable default output."""
    assert main(["--base", str(tmp_path)]) == 0

    assert f"Selected store: {tmp_path}" in capsys.readouterr().out
