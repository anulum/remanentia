# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for selected-store MemoryIndex source configs

"""Tests for selected-store MemoryIndex source configuration."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from store_sources import (
    build_store_source_config,
    main,
    render_store_source_config,
    write_store_source_config,
)


def test_build_config_routes_memory_sources_to_selected_store(tmp_path: Path) -> None:
    """The config should turn the selected store and firehose into index sources."""
    store = tmp_path / "store"
    firehose = tmp_path / "stimuli"

    config = build_store_source_config(base=store, stimuli_dir=firehose)

    assert config["extends_defaults"] is False
    sources = config["sources"]
    assert sources["traces"]["path"] == str(store / "reasoning_traces")
    assert sources["semantic"]["path"] == str(store / "memory" / "semantic")
    assert sources["compiled"]["path"] == str(store / "memory" / "compiled")
    assert sources["arcane_stimuli"]["path"] == str(firehose)
    assert sources["arcane_stimuli"]["extensions"] == [".json"]


def test_write_config_uses_selected_state_dir(tmp_path: Path) -> None:
    """Writing the config should leave the operational file under ``snn_state``."""
    store = tmp_path / "store"
    firehose = tmp_path / "stimuli"

    out = write_store_source_config(base=store, stimuli_dir=firehose)

    assert out == store / "snn_state" / "memory_sources.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["sources"]["arcane_stimuli"]["path"] == str(firehose)
    assert not out.with_suffix(".json.tmp").exists()


def test_config_drives_real_memory_index_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MemoryIndex should index the selected firehose through the written config."""
    store = tmp_path / "store"
    firehose = tmp_path / "stimuli"
    firehose.mkdir(parents=True)
    (firehose / "stimulus.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-06-27T02:12:29Z",
                "project": "REMANENTIA",
                "text": "Finding: selected store firehose backlog is indexed for recall.",
            }
        ),
        encoding="utf-8",
    )
    config_path = write_store_source_config(base=store, stimuli_dir=firehose)
    monkeypatch.setenv("REMANENTIA_MEMORY_SOURCES_CONFIG", str(config_path))

    import memory_index as loaded_memory_index

    memory_index = importlib.reload(loaded_memory_index)
    monkeypatch.setattr("compiled_memory.compile_facts", lambda _repo: [])
    monkeypatch.setattr(memory_index, "HASH_CACHE_PATH", store / "snn_state" / "hashes.json")

    try:
        idx = memory_index.MemoryIndex()
        stats = idx.build(use_gpu_embeddings=False, use_gliner=False, incremental=False)
    finally:
        monkeypatch.delenv("REMANENTIA_MEMORY_SOURCES_CONFIG", raising=False)
        importlib.reload(memory_index)

    assert stats["sources"] == {"arcane_stimuli": 1}
    assert idx.documents[0].source == "arcane_stimuli"
    result = idx.search("firehose backlog indexed for recall", top_k=1)[0]
    assert result.source == "arcane_stimuli"


def test_main_writes_and_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI should persist and print the selected source config."""
    store = tmp_path / "store"
    firehose = tmp_path / "stimuli"

    assert main(["--base", str(store), "--stimuli-dir", str(firehose), "--write", "--json"]) == 0

    printed = json.loads(capsys.readouterr().out)
    written = json.loads((store / "snn_state" / "memory_sources.json").read_text(encoding="utf-8"))
    assert printed == written
    assert printed["sources"]["arcane_stimuli"]["path"] == str(firehose)


def test_main_prints_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The standalone CLI should have a human-readable default view."""
    assert main(["--base", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "MemoryIndex selected sources:" in out
    assert "arcane_stimuli" in out


def test_render_config_lists_sources(tmp_path: Path) -> None:
    """The text renderer should name every selected source."""
    config = build_store_source_config(base=tmp_path)

    text = render_store_source_config(config)

    assert "traces" in text
    assert "semantic" in text
    assert "compiled" in text
    assert "arcane_stimuli" in text
