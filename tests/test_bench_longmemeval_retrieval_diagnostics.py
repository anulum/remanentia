# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — LongMemEval full-S retrieval diagnostic artefact tests

"""Tests for full-S benchmark retrieval diagnostics written to JSONL."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import bench_longmemeval as bench


@dataclass
class _Fact:
    """Minimal retrieved fact carrying the production session-index contract."""

    session_idx: int
    text: str = "retrieved fact"
    date_mentions: list[str] = field(default_factory=list)


@dataclass
class _Result:
    """Minimal retrieval result exposing ``fact`` like ``FusedResult``."""

    fact: _Fact


class _FakeArcaneRetriever:
    """Deterministic ArcaneRetriever replacement for benchmark writer tests."""

    def __init__(
        self,
        sessions: list[list[dict[str, Any]]],
        session_dates: list[str] | None = None,
    ) -> None:
        self.sessions = sessions
        self.session_dates = session_dates

    def retrieve(
        self,
        question: str,
        qtype: str,
        *,
        top_k: int,
        max_iterations: int,
    ) -> list[_Result]:
        """Return three distinct ranked sessions so the cap excludes one gold."""
        assert question
        assert qtype == "multi-session"
        assert top_k >= 3
        assert max_iterations == 2
        return [_Result(_Fact(0)), _Result(_Fact(2)), _Result(_Fact(3))]

    def build_context(
        self,
        question: str,
        results: list[_Result],
        *,
        max_facts: int,
        sort_chronologically: bool,
    ) -> str:
        """Return the ranked-fact context consumed by the real prompt builder."""
        assert question
        assert len(results) == 3
        assert max_facts == 15
        assert not sort_chronologically
        return "ranked fact context"


def _install_fake_arcane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``arcane_retriever`` module for the import inside the runner."""
    module = ModuleType("arcane_retriever")
    module.__dict__["ArcaneRetriever"] = _FakeArcaneRetriever
    monkeypatch.setitem(sys.modules, "arcane_retriever", module)


def _answer(prompt: str, max_tokens: int = 400) -> str:
    """Return a deterministic answer after asserting the full prompt boundary."""
    assert max_tokens == 400
    assert "FULL CONVERSATION HISTORY:" in prompt
    assert "ranked fact context" in prompt
    return "The selected answer is in session s2."


def test_full_s_arcane_run_writes_retrieval_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The full-S benchmark JSONL records selected and missing answer sessions."""
    _install_fake_arcane(monkeypatch)
    dataset = tmp_path / "longmemeval_s.json"
    output = tmp_path / "longmemeval_hypotheses.jsonl"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-diagnostics",
                    "question_type": "multi-session",
                    "question": "Which medical visits should be counted?",
                    "answer": "two visits",
                    "answer_session_ids": ["s2", "s3"],
                    "haystack_session_ids": ["s0", "s1", "s2", "s3"],
                    "haystack_dates": [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                    ],
                    "haystack_sessions": [
                        [{"role": "user", "content": "opening context"}],
                        [{"role": "user", "content": "unused middle context"}],
                        [{"role": "user", "content": "first answer session"}],
                        [{"role": "user", "content": "second answer session"}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(bench, "DATA_PATH", dataset)
    monkeypatch.setattr(bench, "OUTPUT_PATH", output)
    monkeypatch.setattr(bench, "_DATA_FILE", dataset.name)
    monkeypatch.setattr(bench, "_LIMIT", None)
    monkeypatch.setattr(bench, "_USE_ARCANE", True)
    monkeypatch.setattr(bench, "_RETRIEVED_CONTEXT", True)
    monkeypatch.setattr(bench, "_FULL_MAX_SESSIONS", 2)
    monkeypatch.setattr(bench, "_FULL_CHAR_BUDGET", 20_000)
    monkeypatch.setattr(bench, "_FULL_RETRIEVE_K", 50)
    monkeypatch.setattr(bench, "_PROGRESS_EVERY", 1)
    monkeypatch.setattr(bench, "_hypothesis_complete", _answer)

    bench.run_benchmark()

    capsys.readouterr()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    diagnostics = rows[0]["retrieval_diagnostics"]
    assert diagnostics["selected_session_idxs"] == [0, 2]
    assert diagnostics["candidate_session_count"] == 3
    assert diagnostics["dropped_to_session_limit"] == [3]
    assert diagnostics["dropped_to_budget"] == []
    assert diagnostics["answer_session_recall"] == pytest.approx(0.5)
    assert diagnostics["selected_answer_session_ids"] == ["s2"]
    assert diagnostics["missing_answer_session_ids"] == ["s3"]
    assert diagnostics["session_limited_answer_session_ids"] == ["s3"]
    assert diagnostics["budget_dropped_answer_session_ids"] == []
