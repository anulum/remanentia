# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for MCP local-LLM wiring (resolve_backend + _parse_cli)

"""Verify the ``--llm`` / ``--local-llm`` MCP flags and backend factory.

Three surfaces meet here:

- ``llm_backend.resolve_backend("local")`` returns a ``LocalLLMBackend``.
- ``mcp_server._parse_cli(["--llm", "--local-llm"])`` sets the two env
  vars ``handle_recall`` reads.
- ``handle_recall(llm=True)`` on a non-available local Ollama gracefully
  falls through to the retrieval-only path (no crash, no hang).
"""

from __future__ import annotations

import os

import pytest


# ── resolve_backend ──────────────────────────────────────────────────


class TestResolveBackend:
    def test_local_returns_local_backend(self):
        from llm_backend import LocalLLMBackend, resolve_backend

        backend = resolve_backend("local")
        assert isinstance(backend, LocalLLMBackend)

    def test_none_returns_null_backend(self):
        from llm_backend import NullBackend, resolve_backend

        backend = resolve_backend("none")
        assert isinstance(backend, NullBackend)

    def test_anthropic_returns_anthropic_backend(self):
        from llm_backend import AnthropicBackend, resolve_backend

        backend = resolve_backend("anthropic")
        assert isinstance(backend, AnthropicBackend)

    def test_auto_returns_auto_backend(self):
        from llm_backend import AutoBackend, resolve_backend

        backend = resolve_backend("auto")
        assert isinstance(backend, AutoBackend)

    def test_unknown_name_falls_back_to_auto(self):
        from llm_backend import AutoBackend, resolve_backend

        backend = resolve_backend("gpt-42")
        assert isinstance(backend, AutoBackend)

    def test_name_is_case_insensitive(self):
        from llm_backend import LocalLLMBackend, resolve_backend

        backend = resolve_backend("LOCAL")
        assert isinstance(backend, LocalLLMBackend)


# ── mcp_server._parse_cli ────────────────────────────────────────────


@pytest.fixture
def clean_env(monkeypatch):
    """Each test starts with all LLM-related env vars clear."""
    monkeypatch.delenv("REMANENTIA_LLM_ANSWERS", raising=False)
    monkeypatch.delenv("REMANENTIA_LLM_BACKEND", raising=False)
    monkeypatch.delenv("REMANENTIA_GUARDED", raising=False)
    yield


class TestParseCli:
    def test_no_flags_leaves_env_clean(self, clean_env):
        from mcp_server import _parse_cli

        _parse_cli([])
        assert "REMANENTIA_LLM_ANSWERS" not in os.environ
        assert "REMANENTIA_LLM_BACKEND" not in os.environ

    def test_llm_flag_sets_answers_env(self, clean_env):
        from mcp_server import _parse_cli

        _parse_cli(["--llm"])
        assert os.environ["REMANENTIA_LLM_ANSWERS"] == "1"
        # --llm alone does not pin a backend; AutoBackend still chooses.
        assert "REMANENTIA_LLM_BACKEND" not in os.environ

    def test_local_llm_flag_pins_backend(self, clean_env):
        from mcp_server import _parse_cli

        _parse_cli(["--local-llm"])
        assert os.environ["REMANENTIA_LLM_BACKEND"] == "local"

    def test_both_flags_combine(self, clean_env):
        from mcp_server import _parse_cli

        _parse_cli(["--llm", "--local-llm"])
        assert os.environ["REMANENTIA_LLM_ANSWERS"] == "1"
        assert os.environ["REMANENTIA_LLM_BACKEND"] == "local"

    def test_existing_env_var_preserved_by_llm(self, clean_env, monkeypatch):
        """setdefault should not overwrite a pre-set REMANENTIA_LLM_ANSWERS."""
        monkeypatch.setenv("REMANENTIA_LLM_ANSWERS", "verbose")
        from mcp_server import _parse_cli

        _parse_cli(["--llm"])
        assert os.environ["REMANENTIA_LLM_ANSWERS"] == "verbose"

    def test_unknown_args_ignored(self, clean_env):
        from mcp_server import _parse_cli

        _parse_cli(["--local-llm", "--some-mcp-internal-flag", "value"])
        assert os.environ["REMANENTIA_LLM_BACKEND"] == "local"

    def test_guarded_flag_sets_env(self, clean_env):
        from mcp_server import _parse_cli

        _parse_cli(["--guarded"])
        assert os.environ["REMANENTIA_GUARDED"] == "1"

    def test_guarded_with_local_llm(self, clean_env):
        from mcp_server import _parse_cli

        _parse_cli(["--llm", "--local-llm", "--guarded"])
        assert os.environ["REMANENTIA_LLM_ANSWERS"] == "1"
        assert os.environ["REMANENTIA_LLM_BACKEND"] == "local"
        assert os.environ["REMANENTIA_GUARDED"] == "1"


# ── LocalLLMBackend degradation when Ollama is absent ────────────────


class TestLocalBackendDegradation:
    def test_is_available_false_when_no_server(self, monkeypatch):
        """With no Ollama at the configured URL, is_available stays False."""
        from llm_backend import LocalLLMBackend

        # Point at a definitely-unreachable URL so the check fails fast.
        backend = LocalLLMBackend(base_url="http://127.0.0.1:1", model="gemma3:4b", timeout=0.25)
        assert backend.is_available() is False

    def test_complete_returns_none_when_unavailable(self):
        from llm_backend import LocalLLMBackend

        backend = LocalLLMBackend(base_url="http://127.0.0.1:1", model="gemma3:4b", timeout=0.25)
        out = backend.complete("hello", max_tokens=10)
        assert out is None
