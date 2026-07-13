# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for bench_longmemeval CLI flags and timeout handling

"""Verify the bench script picks up CLI flags / env vars correctly.

The script reads `sys.argv` and `os.environ` at import time (legacy
pattern pre-dating the P3-16 argparse migration). These tests import
the module under various sys.argv / env combinations using the
subprocess + module-reimport pattern so they do not mutate the
importer's state.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def _probe(argv_extra: list[str], env: dict[str, str] | None = None) -> dict[str, str]:
    """Import bench_longmemeval under custom argv/env and read its module-level constants.

    Returns a dict of ``{name: repr(value)}`` for the probed constants.
    """
    code = (
        "import sys; sys.argv = ['bench_longmemeval.py'] + " + repr(argv_extra) + "\n"
        "import bench_longmemeval as b\n"
        "print('_OPENAI_TIMEOUT=', b._OPENAI_TIMEOUT)\n"
        "print('_PROGRESS_EVERY=', b._PROGRESS_EVERY)\n"
        "print('_USE_LLM=', b._USE_LLM)\n"
        "print('_LIMIT=', b._LIMIT)\n"
        "print('_EFFECTIVE_SEED=', b._EFFECTIVE_SEED)\n"
        "print('_LOCAL_MODEL=', b._LOCAL_MODEL)\n"
        "print('_LOCAL_URL=', b._LOCAL_URL)\n"
        "print('_READER_MODEL=', b._READER_MODEL)\n"
    )
    r = subprocess.run(
        [PY, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PATH": str(REPO / ".venv" / "bin") + ":/usr/bin:/bin", **(env or {})},
        timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"probe failed: {r.stderr}")
    out: dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class TestOpenAITimeout:
    def test_default_timeout_is_30(self):
        vals = _probe([])
        assert vals["_OPENAI_TIMEOUT"] == "30.0"

    def test_env_override(self):
        vals = _probe([], env={"REMANENTIA_OPENAI_TIMEOUT": "12.5"})
        assert vals["_OPENAI_TIMEOUT"] == "12.5"

    def test_env_invalid_raises(self):
        # Non-numeric env var should crash early rather than silently ignore
        r = subprocess.run(
            [PY, "-c", "import bench_longmemeval"],
            cwd=REPO,
            capture_output=True,
            text=True,
            env={
                "PATH": str(REPO / ".venv" / "bin") + ":/usr/bin:/bin",
                "REMANENTIA_OPENAI_TIMEOUT": "not-a-number",
            },
            timeout=15,
        )
        assert r.returncode != 0
        assert "ValueError" in r.stderr or "could not convert" in r.stderr


class TestProgressEvery:
    def test_default_is_25(self):
        vals = _probe([])
        assert vals["_PROGRESS_EVERY"] == "25"

    def test_flag_override(self):
        vals = _probe(["--progress-every", "7"])
        assert vals["_PROGRESS_EVERY"] == "7"

    def test_flag_minimum_is_1(self):
        vals = _probe(["--progress-every", "0"])
        assert vals["_PROGRESS_EVERY"] == "1"

    def test_negative_flag_clamped(self):
        vals = _probe(["--progress-every", "-5"])
        assert vals["_PROGRESS_EVERY"] == "1"


class TestLegacyFlags:
    def test_llm_flag(self):
        assert _probe(["--llm"])["_USE_LLM"] == "True"
        assert _probe([])["_USE_LLM"] == "False"

    def test_limit_flag(self):
        assert _probe(["--limit", "42"])["_LIMIT"] == "42"
        assert _probe([])["_LIMIT"] == "None"


class TestReaderModel:
    """The per-row reader stamp must mirror the completion-routing decision."""

    def test_local_llm_flag_stamps_local_model(self):
        vals = _probe(["--local-llm"], env={"OPENAI_API_KEY": "sk-test"})
        assert vals["_READER_MODEL"] == vals["_LOCAL_MODEL"]

    def test_missing_hosted_key_stamps_local_model(self):
        # No OPENAI_API_KEY in the probe env → completions route locally.
        vals = _probe([])
        assert vals["_READER_MODEL"] == vals["_LOCAL_MODEL"]

    def test_hosted_key_without_local_flag_stamps_hosted_model(self):
        vals = _probe([], env={"OPENAI_API_KEY": "sk-test"})
        assert vals["_READER_MODEL"] == "gpt-4o-mini"

    def test_local_model_env_override_propagates_to_stamp(self):
        vals = _probe(["--local-llm"], env={"REMANENTIA_LOCAL_MODEL": "gemma3:12b"})
        assert vals["_READER_MODEL"] == "gemma3:12b"

    def test_hosted_model_env_override_propagates_to_stamp(self):
        vals = _probe(
            [],
            env={"OPENAI_API_KEY": "sk-test", "REMANENTIA_HOSTED_MODEL": "gpt-5.4-nano"},
        )
        assert vals["_READER_MODEL"] == "gpt-5.4-nano"


class TestSeed:
    def test_default_seed_is_42(self):
        # seed defaults to env REMANENTIA_SEED or 42; tests run without env.
        assert _probe([])["_EFFECTIVE_SEED"] == "42"

    def test_flag_override(self):
        assert _probe(["--seed", "777"])["_EFFECTIVE_SEED"] == "777"

    def test_env_override(self):
        assert _probe([], env={"REMANENTIA_SEED": "333"})["_EFFECTIVE_SEED"] == "333"

    def test_flag_beats_env(self):
        assert _probe(["--seed", "111"], env={"REMANENTIA_SEED": "222"})["_EFFECTIVE_SEED"] == "111"


class TestLocalReaderSelection:
    """The pure-local W3 number is attributable only if the model/endpoint the
    reader actually queried is resolved from config and printed verbatim — a
    literal baked into the banner could drift from the model that answered."""

    def test_defaults(self):
        vals = _probe([])
        assert vals["_LOCAL_MODEL"] == "gemma3:4b"
        assert vals["_LOCAL_URL"] == "http://localhost:11434/v1"

    def test_model_env_override(self):
        vals = _probe([], env={"REMANENTIA_LOCAL_MODEL": "gemma3:12b"})
        assert vals["_LOCAL_MODEL"] == "gemma3:12b"

    def test_url_env_override(self):
        vals = _probe([], env={"REMANENTIA_LOCAL_URL": "http://localhost:8080/v1"})
        assert vals["_LOCAL_URL"] == "http://localhost:8080/v1"

    def test_resolved_reader_reaches_the_backend(self, monkeypatch):
        # The honesty contract is not that the constants resolve, but that the
        # model/endpoint the reader actually queried is the resolved one — a
        # local score is attributable only if the backend was built from it.
        import bench_longmemeval as b
        requests: list[dict[str, object]] = []

        class ReaderHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                assert self.path == "/v1/models"
                body = json.dumps({"data": [{"id": "qwen2.5:7b"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                assert self.path == "/v1/chat/completions"
                length = int(self.headers["Content-Length"])
                requests.append(json.loads(self.rfile.read(length)))
                body = json.dumps(
                    {"choices": [{"message": {"content": "answer"}}]}
                ).encode()
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
        monkeypatch.setattr(b, "_USE_LOCAL_LLM", True)
        monkeypatch.setattr(b, "_LOCAL_MODEL", "qwen2.5:7b")
        monkeypatch.setattr(
            b,
            "_LOCAL_URL",
            f"http://127.0.0.1:{server.server_address[1]}/v1",
        )
        try:
            result = b._hypothesis_complete("q", max_tokens=73)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert result == "answer"
        assert requests == [
            {
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "q"}],
                "max_tokens": 73,
                "temperature": 0.1,
            }
        ]
