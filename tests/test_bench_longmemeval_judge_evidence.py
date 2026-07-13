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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import bench_longmemeval as bench
from benchmark_evidence import prompt_sha256


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

    requests: list[dict[str, object]] = []
    request_times: list[float] = []

    class JudgeHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer test-key"
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            request_times.append(time.monotonic())
            body = json.dumps(
                {
                    "id": "chatcmpl-local-judge",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "yes"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 1,
                        "total_tokens": 10,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), JudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        f"http://127.0.0.1:{server.server_address[1]}/v1",
    )
    monkeypatch.setattr(bench, "DATA_PATH", data_path)
    monkeypatch.setattr(bench, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(bench, "_PROGRESS_EVERY", 1)
    try:
        bench.run_evaluation()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

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
    assert len(request_times) == 1
    assert requests == [
        {
            "messages": [{"role": "user", "content": prompt}],
            "model": "gpt-4o-mini",
            "max_tokens": 10,
        }
    ]
