# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — CLI tests for canonical store paths

"""CLI coverage for the canonical memory-store path contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

import cli
from cli import cmd_status


def _raise_dashboard_down(*_args: object, **_kwargs: object) -> None:
    """HTTPConnection replacement that makes the dashboard probe fail closed."""
    raise OSError("dashboard unavailable")


def test_status_uses_remanentia_base_for_store_counts_and_freshness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The operator status view reads the selected store, not import-time paths."""
    selected = tmp_path / "selected-store"
    stale_constants = tmp_path / "stale-constants"
    (selected / "reasoning_traces").mkdir(parents=True)
    (selected / "reasoning_traces" / "trace.md").write_text("trace", encoding="utf-8")
    (selected / "memory" / "semantic").mkdir(parents=True)
    (selected / "memory" / "semantic" / "fact.md").write_text("fact", encoding="utf-8")
    (selected / "memory" / "graph").mkdir(parents=True)
    (selected / "snn_state").mkdir(parents=True)
    (selected / "snn_state" / "index_freshness.json").write_text(
        json.dumps({"stale": True, "drift_days": 12.5, "checked_at_unix": time.time()}),
        encoding="utf-8",
    )
    stale_constants.mkdir()

    monkeypatch.setenv("REMANENTIA_BASE", str(selected))
    monkeypatch.setattr(cli, "BASE", stale_constants)
    monkeypatch.setattr(cli, "STATE_DIR", stale_constants / "snn_state")
    monkeypatch.setattr(cli, "GRAPH_DIR", stale_constants / "memory" / "graph")
    monkeypatch.setattr("http.client.HTTPConnection", _raise_dashboard_down)
    monkeypatch.setattr(cli, "_runtime_attr", lambda *_args: lambda: {})

    cmd_status(argparse.Namespace())

    out = capsys.readouterr().out
    assert "Index freshness: STALE (drift 12.5d)" in out
    assert "Episodic traces: 1" in out
    assert "Semantic memories: 1" in out


def test_status_uses_patched_base_when_no_operator_base_is_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Existing tests and local embedders can still patch the CLI base root."""
    base = tmp_path / "patched-base"
    (base / "reasoning_traces").mkdir(parents=True)
    (base / "reasoning_traces" / "trace.md").write_text("trace", encoding="utf-8")
    (base / "memory" / "semantic").mkdir(parents=True)
    (base / "memory" / "semantic" / "fact.md").write_text("fact", encoding="utf-8")
    (base / "memory" / "graph").mkdir(parents=True)
    (base / "snn_state").mkdir(parents=True)

    monkeypatch.delenv("REMANENTIA_BASE", raising=False)
    monkeypatch.setattr(cli, "BASE", base)
    monkeypatch.setattr(cli, "STATE_DIR", base / "snn_state")
    monkeypatch.setattr(cli, "GRAPH_DIR", base / "memory" / "graph")
    monkeypatch.setattr("http.client.HTTPConnection", _raise_dashboard_down)
    monkeypatch.setattr(cli, "_runtime_attr", lambda *_args: lambda: {})

    cmd_status(argparse.Namespace())

    out = capsys.readouterr().out
    assert "Episodic traces: 1" in out
    assert "Semantic memories: 1" in out


def test_store_manifest_command_writes_selected_store_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The top-level CLI exposes the store-selection manifest."""
    output = tmp_path / "selection.json"
    firehose = tmp_path / "firehose"
    monkeypatch.delenv("REMANENTIA_BASE", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "remanentia",
            "store-manifest",
            "--base",
            str(tmp_path),
            "--stimuli-dir",
            str(firehose),
            "--write",
            "--output",
            str(output),
            "--json",
        ],
    )

    cli.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed["selected_base"] == str(tmp_path)
    assert printed["stimuli_dir"] == str(firehose)
    assert written["selected_base"] == str(tmp_path)


def test_store_manifest_command_prints_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The top-level CLI has a human-readable store manifest view."""
    monkeypatch.setenv("REMANENTIA_BASE", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["remanentia", "store-manifest"])

    cli.main()

    assert f"Selected store: {tmp_path}" in capsys.readouterr().out


def test_store_sources_command_writes_selected_memory_index_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The top-level CLI exposes selected MemoryIndex source configuration."""
    output = tmp_path / "sources.json"
    firehose = tmp_path / "firehose"
    monkeypatch.delenv("REMANENTIA_BASE", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "remanentia",
            "store-sources",
            "--base",
            str(tmp_path),
            "--stimuli-dir",
            str(firehose),
            "--write",
            "--output",
            str(output),
            "--json",
        ],
    )

    cli.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written
    assert printed["sources"]["arcane_stimuli"]["path"] == str(firehose)


def test_store_sources_command_prints_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The top-level CLI has a human-readable source config view."""
    monkeypatch.setenv("REMANENTIA_BASE", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["remanentia", "store-sources"])

    cli.main()

    out = capsys.readouterr().out
    assert "MemoryIndex selected sources:" in out
    assert "arcane_stimuli" in out
