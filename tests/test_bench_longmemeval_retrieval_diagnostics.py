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
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from collections.abc import Iterator

import pytest

import bench_longmemeval as bench


@contextmanager
def _local_reader(answer: str) -> Iterator[tuple[str, list[dict[str, object]]]]:
    requests: list[dict[str, object]] = []

    class ReaderHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/models"
            body = json.dumps({"data": [{"id": "diagnostic-reader"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ReaderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_full_s_arcane_run_writes_retrieval_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The full-S benchmark JSONL records selected and missing answer sessions."""
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
                        [
                            {
                                "role": "user",
                                "content": (
                                    "Medical visit counting protocol medical visit count "
                                    "includes clinic appointments and hospital visits. Which "
                                    "medical visits should be counted?"
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "The garden contains blue flowers and a wooden bench "
                                    "unrelated to healthcare."
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "First answer: count the cardiology medical visit and "
                                    "clinic appointment in the visit total."
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "Second answer: count the dermatology medical visit in "
                                    "the visit total."
                                ),
                            }
                        ],
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
    monkeypatch.setattr(bench, "_EMIT_CONFIDENCE", False)
    monkeypatch.setattr(bench, "_USE_LOCAL_LLM", True)
    monkeypatch.setattr(bench, "_LOCAL_MODEL", "diagnostic-reader")
    monkeypatch.setenv("REMANENTIA_ARCANE_CE_DISABLE", "1")
    with _local_reader("The selected answer is in session s2.") as (url, requests):
        monkeypatch.setattr(bench, "_LOCAL_URL", url)
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
    assert len(requests) == 1
    assert requests[0]["model"] == "diagnostic-reader"
    assert requests[0]["max_tokens"] == 400
    prompt = requests[0]["messages"][0]["content"]
    assert "FULL CONVERSATION HISTORY:" in prompt
    assert "cardiology medical visit" in prompt


def test_full_s_arcane_confidence_run_stamps_abstention_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REMANENTIA_EMIT_CONFIDENCE on the arcane path emits the scorecard fields.

    Regression: the arcane branch read ``results[0].score`` (FusedResult has
    ``rrf_score``) — an AttributeError that would have crashed every question
    of the first confidence-bearing sovereign run.
    """
    dataset = tmp_path / "longmemeval_s.json"
    output = tmp_path / "longmemeval_hypotheses.jsonl"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-confidence",
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
                        [
                            {
                                "role": "user",
                                "content": (
                                    "Medical visit counting protocol medical visit count "
                                    "includes clinic appointments and hospital visits. Which "
                                    "medical visits should be counted?"
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "The garden contains blue flowers and a wooden bench "
                                    "unrelated to healthcare."
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "First answer: count the cardiology medical visit and "
                                    "clinic appointment in the visit total."
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "Second answer: count the dermatology medical visit in "
                                    "the visit total."
                                ),
                            }
                        ],
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
    monkeypatch.setattr(bench, "_USE_FULL", True)
    monkeypatch.setattr(bench, "_RETRIEVED_CONTEXT", True)
    monkeypatch.setattr(bench, "_FULL_MAX_SESSIONS", 2)
    monkeypatch.setattr(bench, "_FULL_CHAR_BUDGET", 20_000)
    monkeypatch.setattr(bench, "_FULL_RETRIEVE_K", 50)
    monkeypatch.setattr(bench, "_PROGRESS_EVERY", 1)
    monkeypatch.setattr(bench, "_EMIT_CONFIDENCE", True)
    monkeypatch.setattr(bench, "_USE_LOCAL_LLM", True)
    monkeypatch.setattr(bench, "_LOCAL_MODEL", "diagnostic-reader")
    monkeypatch.setenv("REMANENTIA_ARCANE_CE_DISABLE", "1")
    with _local_reader("The selected answer is in session s2.\nCONFIDENCE: 0.8") as (url, requests):
        monkeypatch.setattr(bench, "_LOCAL_URL", url)
        bench.run_benchmark()

    capsys.readouterr()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    # The rating line is split off the stored hypothesis and surfaced as a field.
    assert row["hypothesis"] == "The selected answer is in session s2."
    assert row["confidence"] == pytest.approx(0.8)
    # rrf_score 0.05 through the logistic squash — a monotone retrieval proxy.
    assert 0.5 < row["retrieval_confidence"] < 0.6
    assert len(row["cited_ids"]) == 4  # one 12-hex id per real retrieved fact
    assert all(len(cid) == 12 for cid in row["cited_ids"])
    # Run-provenance stamps for the manifest headline gates.
    assert row["reader"] == bench._READER_MODEL
    assert row["setting"] == "full_s"
    assert row["seed"] == bench._EFFECTIVE_SEED
    assert len(requests) == 1
    prompt = requests[0]["messages"][0]["content"]
    assert "CONFIDENCE:" in prompt
