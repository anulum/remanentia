# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — LongMemEval judge evidence writer tests

"""Tests for benchmark judge metadata written by the real evaluation path."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

import bench_longmemeval as bench
from benchmark_evidence import prompt_sha256


@dataclass
class _Usage:
    """OpenAI-compatible usage object returned by the fake judge client."""

    prompt_tokens: int = 9
    completion_tokens: int = 1
    total_tokens: int = 10


@dataclass
class _Message:
    """OpenAI-compatible message object returned by the fake judge client."""

    content: str = "yes"


@dataclass
class _Choice:
    """OpenAI-compatible choice object returned by the fake judge client."""

    message: _Message


@dataclass
class _Response:
    """OpenAI-compatible response object returned by the fake judge client."""

    choices: list[_Choice]
    usage: _Usage


class _Completions:
    """Fake completions endpoint that verifies the production call shape."""

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, str]],
    ) -> _Response:
        """Return a deterministic affirmative judge response."""
        assert model == "gpt-4o-mini"
        assert max_tokens == 10
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Is the model response correct?" in messages[0]["content"]
        return _Response(choices=[_Choice(_Message())], usage=_Usage())


@dataclass
class _Chat:
    """Fake chat namespace exposing completions."""

    completions: _Completions


class _OpenAI:
    """Fake OpenAI client installed into ``sys.modules`` for run_evaluation."""

    def __init__(self, *, api_key: str, timeout: float) -> None:
        assert api_key == "test-key"
        assert timeout == bench._OPENAI_TIMEOUT
        self.chat = _Chat(_Completions())


def test_run_evaluation_writes_judge_evidence_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real evaluator persists prompt hash, usage, and latency evidence."""
    data_path = tmp_path / "longmemeval_oracle.json"
    output_path = tmp_path / "longmemeval_hypotheses.jsonl"
    data_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "multi-session",
                    "question": "What did I decide?",
                    "answer": "Use the auditable benchmark report.",
                    "answer_session_ids": [],
                    "haystack_session_ids": [],
                    "haystack_sessions": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "hypothesis": "Use the auditable benchmark report.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    fake_openai = ModuleType("openai")
    fake_openai.__dict__["OpenAI"] = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(bench, "DATA_PATH", data_path)
    monkeypatch.setattr(bench, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(bench, "_PROGRESS_EVERY", 1)

    bench.run_evaluation()

    capsys.readouterr()
    result_path = output_path.with_suffix(".results.jsonl")
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    prompt = bench._judge_prompt(
        "multi-session",
        "What did I decide?",
        "Use the auditable benchmark report.",
        "Use the auditable benchmark report.",
    )
    assert row["judge_label"] is True
    assert row["judge_model"] == "gpt-4o-mini"
    assert row["judge_max_tokens"] == 10
    assert row["judge_prompt_sha256"] == prompt_sha256(prompt)
    assert row["judge_prompt_chars"] == len(prompt)
    assert row["judge_prompt_tokens_estimate"] > 0
    assert row["judge_prompt_tokens"] == 9
    assert row["judge_completion_tokens"] == 1
    assert row["judge_total_tokens"] == 10
    assert isinstance(row["judge_latency_ms"], float)
    assert row["judge_latency_ms"] >= 0.0
