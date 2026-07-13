# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real recall CLI tests

"""End-to-end recall command tests over isolated production stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
import pytest

import answer_extractor
import memory_index
import memory_recall
from cli import cmd_recall, main
from llm_backend import NullBackend


def _configure_recall_stores(monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    traces = tmp_path / "reasoning_traces"
    semantic = tmp_path / "memory" / "semantic"
    graph = tmp_path / "memory" / "graph"
    state = tmp_path / "snn_state"
    traces.mkdir()
    semantic.mkdir(parents=True)
    graph.mkdir(parents=True)
    state.mkdir()
    (traces / "2026-07-13_alpha.md").write_text(
        "# Alpha retrieval decision\n\n"
        "Alpha retrieval started on 2026-07-13. The production memory index keeps "
        "alpha retrieval evidence for real CLI recall.\n",
        encoding="utf-8",
    )
    (traces / "omega.md").write_text(
        "# Omega retrieval note\n\n"
        "Omega evidence remains a qualitative production memory without numeric candidates.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(memory_index, "SOURCES", {"test": traces})
    monkeypatch.setattr(memory_index, "SOURCE_EXTENSIONS", {"test": {".md"}})
    monkeypatch.setattr(memory_index, "INDEX_PATH", state / "memory_index.json.gz")
    monkeypatch.setattr(memory_index, "_LEGACY_INDEX_PATH", state / "legacy.pkl")
    monkeypatch.setattr(memory_index, "HASH_CACHE_PATH", state / "content_hashes.json")
    monkeypatch.setattr(memory_index, "GRAPH_DIR", graph)
    monkeypatch.setattr(memory_recall, "TRACES_DIR", traces)
    monkeypatch.setattr(memory_recall, "SEMANTIC_DIR", semantic)
    monkeypatch.setattr(memory_recall, "GRAPH_DIR", graph)
    monkeypatch.setattr(memory_recall, "HISTORY_PATH", state / "retrieval_history.jsonl")
    answer_extractor.set_llm_backend(None)
    return traces


def _recall_args(*, output_format: str, project: str = "", llm: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        query="alpha retrieval",
        top=3,
        format=output_format,
        content=False,
        project=project,
        after="",
        before="",
        llm=llm,
        llm_backend="none",
    )


def test_filtered_recall_builds_searches_and_renders_real_index(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    _configure_recall_stores(monkeypatch, tmp_path)

    cmd_recall(_recall_args(output_format="summary", project="test"))

    output = capsys.readouterr().out
    assert "2026-07-13_alpha.md" in output
    assert "production memory index" in output.lower()
    assert memory_index.INDEX_PATH.exists()

    no_answer_args = _recall_args(output_format="summary", project="test")
    no_answer_args.query = "omega qualitative evidence"
    no_answer_args.top = 1
    cmd_recall(no_answer_args)
    no_answer_output = capsys.readouterr().out
    assert "omega.md" in no_answer_output
    assert "Answer:" not in no_answer_output

    answer_args = _recall_args(output_format="summary", project="test")
    answer_args.query = "when did alpha retrieval start"
    cmd_recall(answer_args)
    assert "Answer: 2026-07-13" in capsys.readouterr().out


def test_structured_recall_renders_summary_context_and_json_from_real_store(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    _configure_recall_stores(monkeypatch, tmp_path)

    cmd_recall(_recall_args(output_format="summary"))
    summary = capsys.readouterr().out
    assert "Query: alpha retrieval" in summary
    assert "2026-07-13_alpha.md" in summary

    cmd_recall(_recall_args(output_format="context"))
    context = capsys.readouterr().out
    assert "[Matched trace: 2026-07-13_alpha.md]" in context
    assert "production memory index" in context.lower()

    cmd_recall(_recall_args(output_format="json"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "alpha retrieval"
    assert payload["trace"] == "2026-07-13_alpha.md"

    invalid_args = _recall_args(output_format="yaml")
    with pytest.raises(ValueError, match="unsupported recall format"):
        cmd_recall(invalid_args)


def test_empty_query_runs_real_cli_pipeline(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    _configure_recall_stores(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["remanentia", "recall", "", "--top", "1"])

    main()

    assert "Query:" in capsys.readouterr().out


def test_llm_none_backend_runs_real_filtered_cli_pipeline(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    _configure_recall_stores(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "remanentia",
            "recall",
            "alpha retrieval",
            "--project",
            "test",
            "--llm",
            "--llm-backend",
            "none",
        ],
    )

    main()

    assert isinstance(answer_extractor.get_llm_backend(), NullBackend)
    assert "2026-07-13_alpha.md" in capsys.readouterr().out
