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
