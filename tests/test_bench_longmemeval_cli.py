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

import subprocess
import sys
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
        import llm_backend

        captured: dict[str, object] = {}

        class _FakeBackend:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def is_available(self):
                return True

            def complete(self, prompt, max_tokens=400):
                return "answer"

        monkeypatch.setattr(llm_backend, "LocalLLMBackend", _FakeBackend)
        monkeypatch.setattr(b, "_USE_LOCAL_LLM", True)
        monkeypatch.setattr(b, "_LOCAL_MODEL", "qwen2.5:7b")
        monkeypatch.setattr(b, "_LOCAL_URL", "http://localhost:9999/v1")

        result = b._hypothesis_complete("q")

        assert result == "answer"
        assert captured["model"] == "qwen2.5:7b"
        assert captured["base_url"] == "http://localhost:9999/v1"
