# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for LLM backend

# Repository: https://github.com/anulum/remanentia
"""Tests for llm_backend — pluggable LLM backend abstraction."""

from __future__ import annotations

import json
import os
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import Literal, cast
from unittest import mock

import pytest

from llm_backend import (
    AnthropicBackend,
    AutoBackend,
    LLMBackend,
    LLMConfig,
    LocalLLMBackend,
    NullBackend,
    _parse_toml,
    load_config,
    resolve_backend,
)


class _FakeHTTPResponse:
    """Context-manager response used by urllib-backed local LLM tests."""

    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        """Store deterministic response bytes and status code."""
        self._body = body
        self.status = status

    def read(self) -> bytes:
        """Return the encoded response body."""
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        """Enter the urllib response context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Propagate exceptions from the response context."""
        return False


# ── Protocol conformance ─────────────────────────────────────────


class TestProtocolConformance:
    """All backends must satisfy the LLMBackend protocol."""

    def test_null_is_backend(self) -> None:
        assert isinstance(NullBackend(), LLMBackend)

    def test_anthropic_is_backend(self) -> None:
        assert isinstance(AnthropicBackend(api_key="fake"), LLMBackend)

    def test_local_is_backend(self) -> None:
        assert isinstance(LocalLLMBackend(), LLMBackend)

    def test_auto_is_backend(self) -> None:
        assert isinstance(AutoBackend(), LLMBackend)


# ── NullBackend ──���────────────────────────────────────────────────


class TestNullBackend:
    def test_complete_returns_none(self) -> None:
        b = NullBackend()
        assert b.complete("hello") is None

    def test_complete_with_kwargs_returns_none(self) -> None:
        b = NullBackend()
        assert b.complete("hello", max_tokens=500, system="be brief") is None


# ── AnthropicBackend ──────────────────────────────────────────────


class TestAnthropicBackend:
    def test_no_api_key_returns_none(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            b = AnthropicBackend(api_key="")
            assert b.complete("hello") is None

    def test_missing_api_key_env(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            b = AnthropicBackend()
            assert b._api_key == ""
            assert b.complete("hello") is None

    def test_client_cached(self) -> None:
        mock_anthropic = types.ModuleType("anthropic")
        mock_client = mock.MagicMock()
        anthropic_ctor = mock.MagicMock(return_value=mock_client)
        mock_anthropic.__dict__["Anthropic"] = anthropic_ctor

        with mock.patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            b = AnthropicBackend(api_key="test-key")
            client1 = b._get_client()
            client2 = b._get_client()
            assert client1 is client2
            assert anthropic_ctor.call_count == 1

    def test_missing_anthropic_factory_returns_none(self) -> None:
        mock_anthropic = types.ModuleType("anthropic")
        with mock.patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            assert AnthropicBackend(api_key="test-key").complete("hello") is None

    def test_env_key_used_when_no_explicit_key(self) -> None:
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            b = AnthropicBackend()
            assert b._api_key == "env-key"


# ── LocalLLMBackend ───────────────────────────────────────────────


def _mock_openai_response(content: str = "test answer") -> _FakeHTTPResponse:
    """Create a fake OpenAI-compatible JSON response."""
    body = json.dumps(
        {
            "choices": [{"message": {"content": content}}],
        }
    ).encode("utf-8")
    return _FakeHTTPResponse(body=body)


def _mock_json_response(payload: object) -> _FakeHTTPResponse:
    """Create a fake JSON response from an arbitrary decoded payload."""
    return _FakeHTTPResponse(body=json.dumps(payload).encode("utf-8"))


class TestLocalLLMBackend:
    def test_complete_success(self) -> None:
        b = LocalLLMBackend(base_url="http://localhost:9999/v1")
        with mock.patch(
            "urllib.request.urlopen", return_value=_mock_openai_response("hello world")
        ):
            result = b.complete("test prompt")
        assert result == "hello world"

    def test_complete_with_system(self) -> None:
        b = LocalLLMBackend(api_key="test-key")
        calls: list[dict[str, object]] = []

        def capture_urlopen(req: urllib.request.Request, **kwargs: object) -> _FakeHTTPResponse:
            assert isinstance(req.data, bytes)
            calls.append(cast(dict[str, object], json.loads(req.data.decode("utf-8"))))
            assert req.headers["Authorization"] == "Bearer test-key"
            return _mock_openai_response("ok")

        with mock.patch("urllib.request.urlopen", side_effect=capture_urlopen):
            b.complete("test", system="be brief")

        assert len(calls) == 1
        messages = cast(list[dict[str, str]], calls[0]["messages"])
        assert messages[0] == {"role": "system", "content": "be brief"}
        assert messages[1] == {"role": "user", "content": "test"}

    def test_complete_no_system(self) -> None:
        b = LocalLLMBackend()
        calls: list[dict[str, object]] = []

        def capture_urlopen(req: urllib.request.Request, **kwargs: object) -> _FakeHTTPResponse:
            assert isinstance(req.data, bytes)
            calls.append(cast(dict[str, object], json.loads(req.data.decode("utf-8"))))
            return _mock_openai_response("ok")

        with mock.patch("urllib.request.urlopen", side_effect=capture_urlopen):
            b.complete("test")

        messages = cast(list[dict[str, str]], calls[0]["messages"])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_complete_connection_error(self) -> None:
        b = LocalLLMBackend()
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = b.complete("test")
        assert result is None

    def test_is_available_true(self) -> None:
        b = LocalLLMBackend()
        with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(status=200)):
            assert b.is_available() is True

    def test_is_available_false(self) -> None:
        b = LocalLLMBackend()
        with mock.patch("urllib.request.urlopen", side_effect=ConnectionError):
            assert b.is_available() is False

    def test_max_tokens_passed(self) -> None:
        b = LocalLLMBackend()
        calls: list[dict[str, object]] = []

        def capture_urlopen(req: urllib.request.Request, **kwargs: object) -> _FakeHTTPResponse:
            assert isinstance(req.data, bytes)
            calls.append(cast(dict[str, object], json.loads(req.data.decode("utf-8"))))
            return _mock_openai_response("ok")

        with mock.patch("urllib.request.urlopen", side_effect=capture_urlopen):
            b.complete("test", max_tokens=42)

        assert calls[0]["max_tokens"] == 42

    def test_strips_whitespace(self) -> None:
        b = LocalLLMBackend()
        with mock.patch("urllib.request.urlopen", return_value=_mock_openai_response("  padded  ")):
            assert b.complete("test") == "padded"

    def test_complete_returns_none_for_missing_choices(self) -> None:
        b = LocalLLMBackend()
        with mock.patch(
            "urllib.request.urlopen", return_value=_mock_json_response({"choices": []})
        ):
            assert b.complete("test") is None

    def test_complete_returns_none_for_non_mapping_choice(self) -> None:
        b = LocalLLMBackend()
        with mock.patch(
            "urllib.request.urlopen", return_value=_mock_json_response({"choices": [1]})
        ):
            assert b.complete("test") is None

    def test_complete_returns_none_for_non_string_content(self) -> None:
        b = LocalLLMBackend()
        payload = {"choices": [{"message": {"content": {"nested": "text"}}}]}
        with mock.patch("urllib.request.urlopen", return_value=_mock_json_response(payload)):
            assert b.complete("test") is None

    def test_base_url_trailing_slash_stripped(self) -> None:
        b = LocalLLMBackend(base_url="http://localhost:11434/v1/")
        assert b._base_url == "http://localhost:11434/v1"

    def test_rejects_non_http_base_url(self) -> None:
        with pytest.raises(ValueError, match="http or https"):
            LocalLLMBackend(base_url="file:///tmp/local-llm.sock")

    def test_accepts_https_base_url(self) -> None:
        b = LocalLLMBackend(base_url="https://llm.example.test/v1/")
        assert b._base_url == "https://llm.example.test/v1"

    def test_default_url_is_ollama(self) -> None:
        b = LocalLLMBackend()
        assert b._base_url == "http://localhost:11434/v1"

    def test_default_model_is_gemma3_4b(self) -> None:
        b = LocalLLMBackend()
        assert b._model == "gemma3:4b"

    def test_default_timeout_60s(self) -> None:
        b = LocalLLMBackend()
        assert b._timeout == 60.0

    def test_api_key_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"REMANENTIA_LOCAL_LLM_API_KEY": "env-key"}):
            b = LocalLLMBackend()
        assert b._api_key == "env-key"

    def test_no_authorization_header_without_key(self) -> None:
        b = LocalLLMBackend()
        assert "Authorization" not in b._headers()


# ── AutoBackend ───────────────────────────────────────────────────


class TestAutoBackend:
    def test_resolves_to_local_when_available(self) -> None:
        ab = AutoBackend()
        with mock.patch.object(LocalLLMBackend, "is_available", return_value=True):
            with mock.patch.object(LocalLLMBackend, "complete", return_value="local answer"):
                result = ab.complete("test")
        assert result == "local answer"

    def test_resolves_to_anthropic_when_local_unavailable(self) -> None:
        ab = AutoBackend()
        with mock.patch.object(LocalLLMBackend, "is_available", return_value=False):
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key123"}):
                ab._resolve()
                assert isinstance(ab._resolved, AnthropicBackend)

    def test_resolves_to_null_when_nothing_available(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            ab = AutoBackend()
            with mock.patch.object(LocalLLMBackend, "is_available", return_value=False):
                ab._resolve()
                assert isinstance(ab._resolved, NullBackend)

    def test_caches_resolved_backend(self) -> None:
        ab = AutoBackend()
        with mock.patch.object(LocalLLMBackend, "is_available", return_value=False):
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            with mock.patch.dict(os.environ, env, clear=True):
                ab._resolve()
                first = ab._resolved
                ab._resolve()
                assert ab._resolved is first

    def test_uses_config(self) -> None:
        cfg = LLMConfig(
            local_url="http://myhost:1234/v1",
            local_model="llama-8b",
            local_api_key="local-key",
            local_timeout=180.0,
        )
        ab = AutoBackend(config=cfg)
        with mock.patch.object(LocalLLMBackend, "is_available", return_value=True):
            ab._resolve()
            resolved = ab._resolved
            assert isinstance(resolved, LocalLLMBackend)
            assert resolved._base_url == "http://myhost:1234/v1"
            assert resolved._model == "llama-8b"
            assert resolved._api_key == "local-key"
            assert resolved._timeout == 180.0


# ── LLMConfig ─────────────────────────────────────────────────────


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.backend == "auto"
        assert cfg.local_url == "http://localhost:11434/v1"
        assert cfg.local_model == "gemma3:4b"
        assert cfg.local_api_key == ""
        assert cfg.local_timeout == 60.0
        assert cfg.anthropic_model == "claude-haiku-4-5-20251001"
        assert cfg.max_tokens_extract == 100
        assert cfg.max_tokens_generate == 200
        assert cfg.max_tokens_synthesise == 200


# ── load_config ───────────────────────────────────────────────────


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.backend == "auto"

    def test_loads_from_toml(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "llm.toml"
        toml_path.write_text(
            '[llm]\nbackend = "local"\nlocal_url = "http://gpu:9090/v1"\n'
            'local_model = "llama-3b"\nlocal_api_key = "test-key"\n'
            "local_timeout = 240\n\n"
            "[llm.tokens]\nextract = 50\ngenerate = 100\nsynthesise = 150\n",
            encoding="utf-8",
        )
        cfg = load_config(toml_path)
        assert cfg.backend == "local"
        assert cfg.local_url == "http://gpu:9090/v1"
        assert cfg.local_model == "llama-3b"
        assert cfg.local_api_key == "test-key"
        assert cfg.local_timeout == 240.0
        assert cfg.max_tokens_extract == 50
        assert cfg.max_tokens_generate == 100
        assert cfg.max_tokens_synthesise == 150

    def test_env_var_override(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "custom.toml"
        toml_path.write_text('[llm]\nbackend = "anthropic"\n', encoding="utf-8")
        with mock.patch.dict(os.environ, {"REMANENTIA_LLM_CONFIG": str(toml_path)}):
            cfg = load_config()
        assert cfg.backend == "anthropic"

    def test_default_path_when_no_env(self, tmp_path: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "REMANENTIA_LLM_CONFIG"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("llm_backend._DEFAULT_CONFIG_DIR", tmp_path):
                cfg = load_config()
        assert cfg.backend == "auto"  # defaults because file doesn't exist

    def test_partial_config(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "llm.toml"
        toml_path.write_text('[llm]\nbackend = "none"\n', encoding="utf-8")
        cfg = load_config(toml_path)
        assert cfg.backend == "none"
        assert cfg.local_url == "http://localhost:11434/v1"  # default preserved


# ── _parse_toml ───────────────────────────────────────────────────


class TestParseToml:
    def test_parses_valid_toml(self) -> None:
        result = _parse_toml('[section]\nkey = "value"\n')
        assert result == {"section": {"key": "value"}}

    def test_no_parser_returns_empty(self) -> None:
        with mock.patch.dict(sys.modules, {"tomllib": None, "tomli": None}):
            # Force reimport failure
            with mock.patch("builtins.__import__", side_effect=ModuleNotFoundError):
                result = _parse_toml('[llm]\nbackend = "local"\n')
        assert result == {}


# ── resolve_backend ───────────────────────────────────────────────


class TestResolveBackend:
    def test_none(self) -> None:
        b = resolve_backend("none")
        assert isinstance(b, NullBackend)

    def test_local(self) -> None:
        cfg = LLMConfig(local_url="http://myhost:5000/v1", local_model="test-model")
        b = resolve_backend("local", config=cfg)
        assert isinstance(b, LocalLLMBackend)
        assert b._base_url == "http://myhost:5000/v1"
        assert b._model == "test-model"

    def test_local_passes_timeout_and_api_key(self) -> None:
        cfg = LLMConfig(
            local_url="http://myhost:5000/v1",
            local_model="test-model",
            local_api_key="test-key",
            local_timeout=300.0,
        )
        b = resolve_backend("local", config=cfg)
        assert isinstance(b, LocalLLMBackend)
        assert b._api_key == "test-key"
        assert b._timeout == 300.0

    def test_anthropic(self) -> None:
        cfg = LLMConfig(anthropic_model="claude-opus-4-6")
        b = resolve_backend("anthropic", config=cfg)
        assert isinstance(b, AnthropicBackend)
        assert b._model == "claude-opus-4-6"

    def test_auto(self) -> None:
        b = resolve_backend("auto")
        assert isinstance(b, AutoBackend)

    def test_unknown_falls_back_to_auto(self) -> None:
        b = resolve_backend("mystery")
        assert isinstance(b, AutoBackend)

    def test_case_insensitive(self) -> None:
        b = resolve_backend("  LOCAL  ")
        assert isinstance(b, LocalLLMBackend)

    def test_loads_config_when_none_given(self) -> None:
        with mock.patch("llm_backend.load_config", return_value=LLMConfig()) as m:
            resolve_backend("none")
            m.assert_called_once()


# ── Missing patterns: pipeline ────────────────────────────────


class TestLLMBackendPipeline:
    def test_auto_backend_feeds_answer_extractor(self) -> None:
        """AutoBackend → set_llm_backend → answer_extractor uses it."""
        from answer_extractor import set_llm_backend, get_llm_backend
        from llm_backend import NullBackend

        backend = NullBackend()
        set_llm_backend(backend)
        assert get_llm_backend() is backend
        set_llm_backend(None)

    def test_resolve_backend_feeds_mcp(self) -> None:
        """resolve_backend output is compatible with MCP server path."""
        from llm_backend import resolve_backend, NullBackend

        backend = resolve_backend("none")
        assert isinstance(backend, NullBackend)
        result = backend.complete("test")
        assert result is None
