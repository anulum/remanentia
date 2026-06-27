# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — CLI store command tests

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from cli import cmd_init, cmd_store_manifest, cmd_store_sources


def test_cmd_init_respects_environment_store_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Create the operator-selected store layout through `REMANENTIA_BASE`."""
    selected = tmp_path / "selected"
    monkeypatch.setenv("REMANENTIA_BASE", str(selected))

    cmd_init(argparse.Namespace())

    assert (selected / "reasoning_traces").is_dir()
    assert (selected / "memory" / "semantic").is_dir()
    assert (selected / "memory" / "graph").is_dir()
    assert (selected / "consolidation").is_dir()
    assert (selected / "snn_state").is_dir()
    assert "Ready to use" in capsys.readouterr().out


def test_cmd_store_manifest_writes_and_renders_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write a selected-store manifest and render the text operator view."""
    output = tmp_path / "store_selection.json"
    args = argparse.Namespace(
        base=str(tmp_path),
        stimuli_dir=None,
        write=True,
        output=str(output),
        json=False,
    )

    cmd_store_manifest(args)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_base"] == str(tmp_path)
    assert "Selected store" in capsys.readouterr().out


def test_cmd_store_manifest_renders_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Render the selected-store manifest as JSON."""
    args = argparse.Namespace(
        base=str(tmp_path),
        stimuli_dir=None,
        write=False,
        output=None,
        json=True,
    )

    cmd_store_manifest(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_base"] == str(tmp_path)


def test_cmd_store_sources_writes_and_renders_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write selected-store MemoryIndex sources and render the text view."""
    output = tmp_path / "memory_sources.json"
    stimuli = tmp_path / "stimuli"
    args = argparse.Namespace(
        base=str(tmp_path),
        stimuli_dir=str(stimuli),
        write=True,
        output=str(output),
        json=False,
    )

    cmd_store_sources(args)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["extends_defaults"] is False
    assert payload["sources"]["traces"]["path"] == str(tmp_path / "reasoning_traces")
    assert "MemoryIndex selected sources" in capsys.readouterr().out


def test_cmd_store_sources_renders_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Render selected-store MemoryIndex sources as JSON."""
    args = argparse.Namespace(
        base=str(tmp_path),
        stimuli_dir=None,
        write=False,
        output=None,
        json=True,
    )

    cmd_store_sources(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["extends_defaults"] is False
    assert set(payload["sources"]) == {"arcane_stimuli", "compiled", "semantic", "traces"}
