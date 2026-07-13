# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for LLM backend

"""Real HTTP and filesystem tests for :mod:`llm_backend`."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

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


class _LLMServer(ThreadingHTTPServer):
    requests: list[dict[str, object]]


class _LLMHandler(BaseHTTPRequestHandler):
    def _server(self) -> _LLMServer:
        return cast(_LLMServer, self.server)

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:
        self._server().requests.append(
            {"method": "GET", "path": self.path, "headers": dict(self.headers.items())}
        )
        if self.path.startswith("/unhealthy/"):
            self._send_json(503, {"error": "not ready"})
            return
        if self.path.endswith("/models"):
            self._send_json(200, {"data": []})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = cast(dict[str, Any], json.loads(raw.decode("utf-8")))
        self._server().requests.append(
            {
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers.items()),
                "payload": payload,
            }
        )
        messages = cast(list[dict[str, str]], payload.get("messages", []))
        prompt = messages[-1]["content"] if messages else ""
        if self.path.startswith("/error/"):
            self._send_json(500, {"error": "provider down"})
            return
        if prompt == "malformed-json":
            self._send(200, b"{")
            return
        if self.path.endswith("/messages"):
            if prompt == "missing-content":
                self._send_json(200, {"content": []})
            elif prompt == "non-string-content":
                self._send_json(200, {"content": [{"text": {"nested": True}}]})
            else:
                self._send_json(200, {"content": [{"type": "text", "text": " hosted answer "}]})
            return
        if self.path.endswith("/chat/completions"):
            if prompt == "missing-choices":
                response: object = {"choices": []}
            elif prompt == "non-mapping-choice":
                response = {"choices": [1]}
            elif prompt == "non-string-content":
                response = {"choices": [{"message": {"content": {"nested": True}}}]}
            elif prompt == "padded":
                response = {"choices": [{"message": {"content": "  padded  "}}]}
            else:
                response = {"choices": [{"message": {"content": "hello world"}}]}
            self._send_json(200, response)
            return
        self._send_json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: object) -> None:
        return


@contextmanager
def _llm_server() -> Iterator[tuple[str, _LLMServer]]:
    server = _LLMServer(("127.0.0.1", 0), _LLMHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = cast(str, server.server_address[0])
        port = server.server_address[1]
        yield f"http://{host}:{port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestProtocolConformance:
    def test_all_backends_satisfy_protocol(self) -> None:
        assert isinstance(NullBackend(), LLMBackend)
        assert isinstance(AnthropicBackend(api_key="test"), LLMBackend)
        assert isinstance(LocalLLMBackend(), LLMBackend)
        assert isinstance(AutoBackend(), LLMBackend)


class TestNullBackend:
    def test_complete_returns_none(self) -> None:
        backend = NullBackend()
        assert backend.complete("hello") is None
        assert backend.complete("hello", max_tokens=500, system="be brief") is None


class TestAnthropicBackend:
    def test_no_api_key_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert AnthropicBackend(api_key="").complete("hello") is None

    def test_env_key_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        assert AnthropicBackend()._api_key == "env-key"

    def test_real_messages_request_and_response(self) -> None:
        with _llm_server() as (url, server):
            backend = AnthropicBackend(
                model="claude-local",
                api_key="secret",
                base_url=f"{url}/v1",
            )
            result = backend.complete("hello", max_tokens=42, system="be brief")

        assert result == "hosted answer"
        request = server.requests[0]
        headers = {
            key.lower(): value
            for key, value in cast(dict[str, str], request["headers"]).items()
        }
        payload = cast(dict[str, object], request["payload"])
        assert request["path"] == "/v1/messages"
        assert headers["x-api-key"] == "secret"
        assert headers["anthropic-version"] == "2023-06-01"
        assert payload["model"] == "claude-local"
        assert payload["max_tokens"] == 42
        assert payload["system"] == "be brief"

    def test_real_error_and_invalid_payload_paths(self) -> None:
        with _llm_server() as (url, _server):
            assert (
                AnthropicBackend(api_key="key", base_url=f"{url}/error/v1").complete("hello")
                is None
            )
            backend = AnthropicBackend(api_key="key", base_url=f"{url}/v1")
            assert backend.complete("malformed-json") is None
            assert backend.complete("missing-content") is None
            assert backend.complete("non-string-content") is None

    def test_rejects_non_http_base_url(self) -> None:
        with pytest.raises(ValueError, match="http or https"):
            AnthropicBackend(api_key="key", base_url="file:///tmp/anthropic.sock")


class TestLocalLLMBackend:
    def test_real_complete_with_system_auth_and_tokens(self) -> None:
        with _llm_server() as (url, server):
            backend = LocalLLMBackend(
                base_url=f"{url}/v1",
                model="local-model",
                api_key="test-key",
            )
            result = backend.complete("test prompt", max_tokens=42, system="be brief")

        assert result == "hello world"
        request = server.requests[0]
        headers = cast(dict[str, str], request["headers"])
        payload = cast(dict[str, object], request["payload"])
        messages = cast(list[dict[str, str]], payload["messages"])
        assert headers["Authorization"] == "Bearer test-key"
        assert payload["model"] == "local-model"
        assert payload["max_tokens"] == 42
        assert messages == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "test prompt"},
        ]

    def test_real_complete_without_system(self) -> None:
        with _llm_server() as (url, server):
            backend = LocalLLMBackend(base_url=f"{url}/v1")
            assert backend.complete("test") == "hello world"
        payload = cast(dict[str, object], server.requests[0]["payload"])
        assert payload["messages"] == [{"role": "user", "content": "test"}]

    def test_real_transport_and_payload_failures(self) -> None:
        with _llm_server() as (url, _server):
            assert LocalLLMBackend(base_url=f"{url}/error/v1").complete("test") is None
            backend = LocalLLMBackend(base_url=f"{url}/v1")
            assert backend.complete("malformed-json") is None
            assert backend.complete("missing-choices") is None
            assert backend.complete("non-mapping-choice") is None
            assert backend.complete("non-string-content") is None
            assert backend.complete("padded") == "padded"

    def test_real_availability_status(self) -> None:
        with _llm_server() as (url, _server):
            assert LocalLLMBackend(base_url=f"{url}/v1").is_available() is True
            assert LocalLLMBackend(base_url=f"{url}/unhealthy/v1").is_available() is False

    def test_configuration_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REMANENTIA_LOCAL_LLM_API_KEY", "env-key")
        backend = LocalLLMBackend(base_url="http://localhost:11434/v1/")
        assert backend._base_url == "http://localhost:11434/v1"
        assert backend._model == "gemma3:4b"
        assert backend._timeout == 60.0
        assert backend._api_key == "env-key"
        assert backend._headers()["Authorization"] == "Bearer env-key"

    def test_no_authorization_header_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REMANENTIA_LOCAL_LLM_API_KEY", raising=False)
        assert "Authorization" not in LocalLLMBackend()._headers()

    def test_rejects_non_http_base_url(self) -> None:
        with pytest.raises(ValueError, match="http or https"):
            LocalLLMBackend(base_url="file:///tmp/local-llm.sock")

    def test_accepts_https_base_url(self) -> None:
        assert LocalLLMBackend(base_url="https://llm.example.test/v1/")._base_url.endswith("/v1")


class TestAutoBackend:
    def test_resolves_real_local_and_caches(self) -> None:
        with _llm_server() as (url, _server):
            config = LLMConfig(local_url=f"{url}/v1", local_model="local-model")
            backend = AutoBackend(config)
            assert backend.complete("test") == "hello world"
            resolved = backend._resolved
            assert backend._resolve() is resolved
            assert isinstance(resolved, LocalLLMBackend)

    def test_resolves_hosted_when_real_local_unhealthy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "hosted-key")
        with _llm_server() as (url, _server):
            backend = AutoBackend(LLMConfig(local_url=f"{url}/unhealthy/v1"))
            assert isinstance(backend._resolve(), AnthropicBackend)

    def test_resolves_null_when_nothing_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with _llm_server() as (url, _server):
            backend = AutoBackend(LLMConfig(local_url=f"{url}/unhealthy/v1"))
            assert isinstance(backend._resolve(), NullBackend)

    def test_uses_config_on_real_health_check(self) -> None:
        with _llm_server() as (url, _server):
            config = LLMConfig(
                local_url=f"{url}/v1",
                local_model="llama-8b",
                local_api_key="local-key",
                local_timeout=180.0,
            )
            resolved = AutoBackend(config)._resolve()
        assert isinstance(resolved, LocalLLMBackend)
        assert resolved._model == "llama-8b"
        assert resolved._api_key == "local-key"
        assert resolved._timeout == 180.0


class TestConfiguration:
    def test_defaults(self) -> None:
        config = LLMConfig()
        assert config == LLMConfig(
            backend="auto",
            local_url="http://localhost:11434/v1",
            local_model="gemma3:4b",
            local_api_key="",
            local_timeout=60.0,
            anthropic_model="claude-haiku-4-5-20251001",
            max_tokens_extract=100,
            max_tokens_generate=200,
            max_tokens_synthesise=200,
        )

    def test_defaults_when_file_missing(self, tmp_path: Path) -> None:
        assert load_config(tmp_path / "missing.toml") == LLMConfig()

    def test_loads_full_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "llm.toml"
        path.write_text(
            '[llm]\nbackend = "local"\nlocal_url = "http://gpu:9090/v1"\n'
            'local_model = "llama-3b"\nlocal_api_key = "test-key"\n'
            'anthropic_model = "claude-local"\nlocal_timeout = "240"\n\n'
            '[llm.tokens]\nextract = 50.0\ngenerate = "100"\nsynthesise = 150\n',
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.backend == "local"
        assert config.local_url == "http://gpu:9090/v1"
        assert config.local_model == "llama-3b"
        assert config.local_api_key == "test-key"
        assert config.anthropic_model == "claude-local"
        assert config.local_timeout == 240.0
        assert config.max_tokens_extract == 50
        assert config.max_tokens_generate == 100
        assert config.max_tokens_synthesise == 150

    def test_env_config_and_partial_defaults(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "custom.toml"
        path.write_text('[llm]\nbackend = "anthropic"\n', encoding="utf-8")
        monkeypatch.setenv("REMANENTIA_LLM_CONFIG", str(path))
        config = load_config()
        assert config.backend == "anthropic"
        assert config.local_url == LLMConfig().local_url

    def test_default_path_when_no_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("REMANENTIA_LLM_CONFIG", raising=False)
        monkeypatch.setattr("llm_backend._DEFAULT_CONFIG_DIR", tmp_path)
        assert load_config() == LLMConfig()

    def test_parse_toml_and_non_mapping_sections(self, tmp_path: Path) -> None:
        assert _parse_toml('[section]\nkey = "value"\n') == {"section": {"key": "value"}}
        path = tmp_path / "invalid-shape.toml"
        path.write_text('llm = "local"\n', encoding="utf-8")
        assert load_config(path) == LLMConfig()


class TestResolveBackend:
    def test_named_backends(self) -> None:
        config = LLMConfig(
            local_url="http://myhost:5000/v1",
            local_model="test-model",
            local_api_key="test-key",
            local_timeout=300.0,
            anthropic_model="claude-opus-4-6",
        )
        assert isinstance(resolve_backend("none", config), NullBackend)
        local = resolve_backend(" LOCAL ", config)
        assert isinstance(local, LocalLLMBackend)
        assert local._model == "test-model"
        assert local._api_key == "test-key"
        assert local._timeout == 300.0
        anthropic = resolve_backend("anthropic", config)
        assert isinstance(anthropic, AnthropicBackend)
        assert anthropic._model == "claude-opus-4-6"
        assert isinstance(resolve_backend("auto", config), AutoBackend)
        assert isinstance(resolve_backend("mystery", config), AutoBackend)

    def test_loads_real_config_when_not_injected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "llm.toml"
        path.write_text(
            '[llm]\nlocal_url = "http://configured:8123/v1"\nlocal_model = "configured"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("REMANENTIA_LLM_CONFIG", str(path))
        backend = resolve_backend("local")
        assert isinstance(backend, LocalLLMBackend)
        assert backend._base_url == "http://configured:8123/v1"
        assert backend._model == "configured"


class TestLLMBackendPipeline:
    def test_null_backend_feeds_answer_extractor(self) -> None:
        from answer_extractor import get_llm_backend, set_llm_backend

        backend = NullBackend()
        set_llm_backend(backend)
        assert get_llm_backend() is backend
        set_llm_backend(None)

    def test_resolved_backend_is_mcp_compatible(self) -> None:
        backend = resolve_backend("none", LLMConfig())
        assert isinstance(backend, NullBackend)
        assert backend.complete("test") is None
