# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for HTTP API server

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api_security import DEFAULT_BODY_LIMIT, BearerAuth, RequestAuditLogger, TokenBucketLimiter
from api_server import RemanentiaHandler, _json_default


class _HandlerServer:
    """Minimal stand-in for the HTTPServer attributes the handler reads."""

    def __init__(
        self,
        *,
        auth: BearerAuth | None = None,
        limiter: TokenBucketLimiter | None = None,
        body_limit: int = DEFAULT_BODY_LIMIT,
        audit_logger: RequestAuditLogger | None = None,
    ) -> None:
        self.auth = auth or BearerAuth(None, warn_on_disabled=False)
        self.limiter = limiter or TokenBucketLimiter(rate_per_minute=6000, burst=1000)
        self.body_limit = body_limit
        self.audit_logger = audit_logger or RequestAuditLogger(None)


# ── JSON default handler ─────────────────────────────────────────


class TestJsonDefault:
    def test_np_float32(self) -> None:
        assert _json_default(np.float32(3.14)) == pytest.approx(3.14, abs=0.01)

    def test_np_float64(self) -> None:
        assert isinstance(_json_default(np.float64(2.71)), float)

    def test_np_int32(self) -> None:
        assert _json_default(np.int32(42)) == 42

    def test_np_int64(self) -> None:
        assert _json_default(np.int64(99)) == 99

    def test_np_array(self) -> None:
        result = _json_default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError):
            _json_default(object())


# ── Handler helpers ──────────────────────────────────────────────


def _make_handler(
    path: str = "/",
    method: str = "GET",
    body: object | None = None,
    *,
    server: _HandlerServer | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """Create a RemanentiaHandler with mocked I/O.

    Parameters
    ----------
    server:
        A :class:`_HandlerServer` providing ``auth``, ``limiter``, ``body_limit``.
        Defaults to an open-auth server that permits any request.
    headers:
        Dict of header overrides. ``Content-Length`` defaults to body size
        and ``Authorization`` defaults to absent.
    """
    handler = RemanentiaHandler.__new__(RemanentiaHandler)
    handler_any: Any = handler
    handler_any.path = path
    handler_any.command = method
    handler_any.requestline = f"{method} {path} HTTP/1.1"
    handler_any.request_version = "HTTP/1.1"
    handler_any.client_address = ("127.0.0.1", 12345)
    handler_any.server = server or _HandlerServer()

    body_bytes = b""
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")

    handler_any.rfile = io.BytesIO(body_bytes)
    handler_any.wfile = io.BytesIO()
    header_map = {"Content-Length": str(len(body_bytes))}
    if headers:
        header_map.update(headers)
    handler_any.headers = MagicMock()

    def header_get(key: str, default: str | None = None) -> str | None:
        return header_map.get(key, default)

    handler_any.headers.get = MagicMock(side_effect=header_get)
    handler_any.send_response = MagicMock()
    handler_any.send_header = MagicMock()
    handler_any.end_headers = MagicMock()
    return handler_any


def _get_response_body(handler: Any) -> dict[str, Any]:
    """Read the JSON body written to handler.wfile."""
    handler.wfile.seek(0)
    return cast(dict[str, Any], json.loads(handler.wfile.read()))


# ── GET endpoints ────────────────────────────────────────────────


class TestDoGet:
    def test_health(self) -> None:
        h = _make_handler("/health")
        h.do_GET()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["status"] == "ok"
        assert "timestamp" in body

    def test_status_empty(self, tmp_path: Path) -> None:
        h = _make_handler("/status")
        with patch("api_server.BASE", tmp_path):
            h.do_GET()
        body = _get_response_body(h)
        assert body["entities"] == 0
        assert body["traces"] == 0

    def test_status_with_data(self, tmp_path: Path) -> None:
        # Create graph data
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (graph_dir / "entities.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
        (graph_dir / "relations.jsonl").write_text('{"src":"a","tgt":"b"}\n')
        # Create traces
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        (traces_dir / "t1.md").write_text("trace")
        # Create semantic
        sem_dir = tmp_path / "memory" / "semantic"
        sem_dir.mkdir(parents=True)
        (sem_dir / "s1.md").write_text("mem")

        h = _make_handler("/status")
        with patch("api_server.BASE", tmp_path):
            h._handle_status()
        body = _get_response_body(h)
        assert body["entities"] == 2
        assert body["relations"] == 1
        assert body["traces"] == 1
        assert body["memories"] == 1

    def test_unknown_get(self) -> None:
        h = _make_handler("/unknown")
        h.do_GET()
        h.send_response.assert_called_with(404)
        body = _get_response_body(h)
        assert "error" in body


# ── POST endpoints ───────────────────────────────────────────────


class TestDoPost:
    def test_recall_empty_query(self) -> None:
        h = _make_handler("/recall", "POST", {"query": ""})
        h.do_POST()
        h.send_response.assert_called_with(400)
        body = _get_response_body(h)
        assert "error" in body

    def test_recall_top_k_zero(self) -> None:
        h = _make_handler("/recall", "POST", {"query": "test", "top_k": 0})
        h.do_POST()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["results"] == []

    def test_recall_rejects_non_integer_top_k(self) -> None:
        h = _make_handler("/recall", "POST", {"query": "test", "top_k": "five"})
        h.do_POST()
        h.send_response.assert_called_with(400)
        body = _get_response_body(h)
        assert body["error"] == "top_k must be an integer"

    def test_recall_success(self) -> None:
        mock_ctx = MagicMock()
        mock_ctx.trace = "trace.md"
        mock_ctx.trace_score = 0.8
        mock_ctx.trace_snippet = "Snippet text"
        mock_ctx.semantic_memories = [
            {"file": "mem.md", "score": 0.5, "content": "Memory content"},
        ]
        mock_ctx.entities = ["stdp", "bm25"]
        mock_ctx.novelty_score = 0.3

        h = _make_handler("/recall", "POST", {"query": "test"})
        with patch("memory_recall.recall", return_value=mock_ctx):
            h.do_POST()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert len(body["results"]) == 2  # trace + semantic
        assert body["results"][0]["type"] == "trace"
        assert body["results"][1]["type"] == "semantic"

    def test_recall_exception(self) -> None:
        h = _make_handler("/recall", "POST", {"query": "test"})
        with patch("memory_recall.recall", side_effect=RuntimeError("boom")):
            h.do_POST()
        h.send_response.assert_called_with(500)
        body = _get_response_body(h)
        assert "error" in body

    def test_recall_no_trace(self) -> None:
        mock_ctx = MagicMock()
        mock_ctx.trace = ""
        mock_ctx.trace_score = 0.0
        mock_ctx.trace_snippet = ""
        mock_ctx.semantic_memories = []
        mock_ctx.entities = []
        mock_ctx.novelty_score = 0.5

        h = _make_handler("/recall", "POST", {"query": "test"})
        with patch("memory_recall.recall", return_value=mock_ctx):
            h.do_POST()
        body = _get_response_body(h)
        assert body["results"] == []

    def test_consolidate_success(self) -> None:
        mock_result = {"new_memories": 3}
        h = _make_handler("/consolidate", "POST", {"force": True})
        with patch("consolidation_engine.consolidate", return_value=mock_result):
            h.do_POST()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["new_memories"] == 3

    def test_consolidate_rejects_non_boolean_force(self) -> None:
        h = _make_handler("/consolidate", "POST", {"force": "yes"})
        h.do_POST()
        h.send_response.assert_called_with(400)
        body = _get_response_body(h)
        assert body["error"] == "force must be a boolean"

    def test_consolidate_exception(self) -> None:
        h = _make_handler("/consolidate", "POST", {})
        with patch("consolidation_engine.consolidate", side_effect=RuntimeError("fail")):
            h.do_POST()
        h.send_response.assert_called_with(500)

    def test_remember_empty_content(self) -> None:
        h = _make_handler("/remember", "POST", {"content": ""})
        h.do_POST()
        h.send_response.assert_called_with(400)

    def test_remember_rejects_non_string_trigger(self) -> None:
        h = _make_handler("/remember", "POST", {"content": "Test fact", "trigger": 5})
        h.do_POST()
        h.send_response.assert_called_with(400)
        body = _get_response_body(h)
        assert body["error"] == "trigger must be a string"

    def test_remember_success(self) -> None:
        mock_ks = MagicMock()
        h = _make_handler("/remember", "POST", {"content": "Test fact", "trigger": "test"})
        with patch("knowledge_store.KnowledgeStore", return_value=mock_ks):
            h.do_POST()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["status"] == "ok"

    def test_remember_exception(self) -> None:
        h = _make_handler("/remember", "POST", {"content": "Test"})
        with patch("knowledge_store.KnowledgeStore", side_effect=RuntimeError("fail")):
            h.do_POST()
        h.send_response.assert_called_with(500)

    def test_unknown_post(self) -> None:
        h = _make_handler("/unknown", "POST", {})
        h.do_POST()
        h.send_response.assert_called_with(404)


# ── Read body ────────────────────────────────────────────────────


class TestReadBody:
    def test_no_body(self) -> None:
        h = _make_handler("/test")
        h.headers.get = MagicMock(return_value="0")
        result = h._read_body()
        assert result == {}

    def test_valid_json(self) -> None:
        body = {"key": "value"}
        h = _make_handler("/test", "POST", body)
        result = h._read_body()
        assert result == body

    def test_invalid_json(self) -> None:
        h = _make_handler("/test")
        h.rfile = io.BytesIO(b"not json")
        h.headers.get = MagicMock(return_value="8")
        result = h._read_body()
        assert result == {}


# ── Log message ──────────────────────────────────────────────────


class TestLogMessage:
    def test_404_logged(self) -> None:
        h = _make_handler("/test")
        with patch("http.server.BaseHTTPRequestHandler.log_message") as mock_log:
            h.log_message("%s", "404 Not Found")
        mock_log.assert_called_once()

    def test_non_404_suppressed(self) -> None:
        h = _make_handler("/test")
        with patch("http.server.BaseHTTPRequestHandler.log_message") as mock_log:
            h.log_message("%s", "200 OK")
        mock_log.assert_not_called()


# ── JSON response ────────────────────────────────────────────────


class TestJsonResponse:
    def test_sets_headers(self) -> None:
        h = _make_handler("/test")
        h._json_response({"ok": True}, 200)
        h.send_response.assert_called_with(200)
        h.send_header.assert_any_call("Content-Type", "application/json")
        h.send_header.assert_any_call("Access-Control-Allow-Origin", "*")

    def test_custom_status(self) -> None:
        h = _make_handler("/test")
        h._json_response({"error": "bad"}, 400)
        h.send_response.assert_called_with(400)

    def test_cors_scoped_echoes_listed_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REMANENTIA_CORS_ORIGINS", "https://ok.example")
        h = _make_handler("/test", headers={"Origin": "https://ok.example"})
        h._json_response({"ok": True}, 200)
        h.send_header.assert_any_call("Access-Control-Allow-Origin", "https://ok.example")

    def test_cors_scoped_omits_header_for_unlisted_origin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REMANENTIA_CORS_ORIGINS", "https://ok.example")
        h = _make_handler("/test", headers={"Origin": "https://evil.example"})
        h._json_response({"ok": True}, 200)
        cors_calls = [
            c
            for c in h.send_header.call_args_list
            if c.args and c.args[0] == "Access-Control-Allow-Origin"
        ]
        assert cors_calls == []


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestAPIServerPipeline:
    def test_recall_handler_returns_json_from_request_body(self) -> None:
        """HTTP handler exercises JSON request parsing through POST /recall."""
        mock_ctx = MagicMock()
        mock_ctx.trace = "pipeline_trace.md"
        mock_ctx.trace_score = 0.9
        mock_ctx.trace_snippet = "Pipeline trace snippet"
        mock_ctx.semantic_memories = []
        mock_ctx.entities = ["pipeline"]
        mock_ctx.novelty_score = 0.2

        h = _make_handler("/recall", "POST", {"query": "pipeline", "top_k": 1})

        with patch("memory_recall.recall", return_value=mock_ctx) as recall:
            h.do_POST()

        recall.assert_called_once_with("pipeline", top_k=1)
        h.send_response.assert_called_with(200)
        h.send_header.assert_any_call("Content-Type", "application/json")
        body = _get_response_body(h)
        assert body["query"] == "pipeline"
        assert body["entities"] == ["pipeline"]
        assert body["results"][0]["name"] == "pipeline_trace.md"

    def test_recall_handler_preserves_real_semantic_memory_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real semantic recall result must expose its canonical non-empty name."""
        import memory_index
        import memory_recall

        semantic_dir = tmp_path / "memory" / "semantic"
        semantic_dir.mkdir(parents=True)
        (semantic_dir / "known-memory.md").write_text(
            "---\nproject: integration\ntype: finding\n---\n"
            "- Known semantic memory survives the HTTP mapping.\n",
            encoding="utf-8",
        )
        empty_index = MagicMock()
        empty_index.load.return_value = True
        empty_index.search.return_value = []
        monkeypatch.setattr(memory_index, "MemoryIndex", lambda: empty_index)
        monkeypatch.setattr(memory_recall, "BASE", tmp_path)
        monkeypatch.setattr(memory_recall, "SEMANTIC_DIR", semantic_dir)
        monkeypatch.setattr(memory_recall, "TRACES_DIR", tmp_path / "reasoning_traces")
        monkeypatch.setattr(memory_recall, "GRAPH_DIR", tmp_path / "memory" / "graph")

        handler = _make_handler(
            "/recall",
            "POST",
            {"query": "known semantic memory", "top_k": 1},
        )
        handler.do_POST()

        handler.send_response.assert_called_with(200)
        body = _get_response_body(handler)
        assert body["results"] == [
            {
                "name": "memory/semantic/known-memory.md",
                "score": 1.0,
                "snippet": "- Known semantic memory survives the HTTP mapping.",
                "type": "semantic",
            }
        ]

    def test_malformed_recall_json_returns_missing_query_error(self) -> None:
        h = _make_handler("/recall", "POST")
        h.rfile = io.BytesIO(b'{"query":')
        h.headers.get = MagicMock(
            side_effect=lambda k, default=None: "9" if k == "Content-Length" else default
        )

        h.do_POST()

        h.send_response.assert_called_with(400)
        body = _get_response_body(h)
        assert body["error"] == "query required"

    def test_non_object_recall_json_returns_missing_query_error(self) -> None:
        h = _make_handler("/recall", "POST")
        h.rfile = io.BytesIO(b'["not", "an", "object"]')
        h.headers.get = MagicMock(
            side_effect=lambda k, default=None: "23" if k == "Content-Length" else default
        )

        h.do_POST()

        h.send_response.assert_called_with(400)
        body = _get_response_body(h)
        assert body["error"] == "query required"


# ── Security gate integration ────────────────────────────────────────


class TestSecurityGates:
    """Exercise BearerAuth / limiter / body-size through the handler path."""

    def test_health_is_public_no_auth_required(self) -> None:
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler("/health", server=server)
        h.do_GET()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["status"] == "ok"

    def test_status_requires_auth_when_token_set(self) -> None:
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler("/status", server=server)  # no Authorization header
        h.do_GET()
        h.send_response.assert_called_with(401)
        body = _get_response_body(h)
        assert "authentication" in body["error"].lower()

    def test_status_passes_with_valid_bearer(self, tmp_path: Path) -> None:
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler(
            "/status",
            server=server,
            headers={"Authorization": "Bearer valid-value"},
        )
        with patch("api_server.BASE", tmp_path):
            h.do_GET()
        h.send_response.assert_called_with(200)

    def test_status_rejects_wrong_bearer(self) -> None:
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler(
            "/status",
            server=server,
            headers={"Authorization": "Bearer wrong"},
        )
        h.do_GET()
        h.send_response.assert_called_with(401)

    def test_non_ascii_bearer_returns_401_not_crash(self) -> None:
        # A crafted non-ASCII Authorization header reaches the handler as
        # latin-1-decoded str. The auth gate must return a clean 401,
        # never raise a TypeError out of do_GET into a 500/connection reset.
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler(
            "/status",
            server=server,
            headers={"Authorization": "Bearer \xa3\xa3"},
        )
        h.do_GET()
        h.send_response.assert_called_with(401)

    def test_rate_limit_triggers_429(self) -> None:
        # burst=1, rate=60/min, same IP → second request immediately denied
        server = _HandlerServer(limiter=TokenBucketLimiter(rate_per_minute=60, burst=1))
        h1 = _make_handler("/status", server=server)
        h1.do_GET()
        assert h1.send_response.call_args[0][0] == 200

        h2 = _make_handler("/status", server=server)  # same client_address
        h2.do_GET()
        h2.send_response.assert_called_with(429)
        h2.send_header.assert_any_call("Retry-After", "1")
        assert "rate limit" in _get_response_body(h2)["error"].lower()

    def test_body_size_limit_rejects_413(self) -> None:
        # body limit 10 bytes, sending JSON ~20 bytes
        server = _HandlerServer(body_limit=10)
        h = _make_handler(
            "/recall",
            method="POST",
            body={"query": "hello world"},
            server=server,
        )
        h.do_POST()
        h.send_response.assert_called_with(413)
        assert "exceeds" in _get_response_body(h)["error"].lower()

    def test_body_size_permits_small_post(self) -> None:
        server = _HandlerServer()  # default 1 MiB
        h = _make_handler(
            "/recall",
            method="POST",
            body={"query": "hi", "top_k": 0},  # top_k=0 short-circuits
            server=server,
        )
        h.do_POST()
        h.send_response.assert_called_with(200)

    def test_health_bypasses_rate_limit(self) -> None:
        # Exhaust the bucket with /status, then /health still works
        server = _HandlerServer(limiter=TokenBucketLimiter(rate_per_minute=60, burst=1))
        h1 = _make_handler("/status", server=server)
        h1.do_GET()
        h2 = _make_handler("/status", server=server)
        h2.do_GET()  # 429
        h3 = _make_handler("/health", server=server)
        h3.do_GET()
        h3.send_response.assert_called_with(200)

    def test_health_bypasses_body_size(self) -> None:
        # Health is GET so body size doesn't apply, but /health shouldn't even
        # reach gates — check it stays public even with a hostile Content-Length
        server = _HandlerServer(body_limit=0)
        h = _make_handler(
            "/health",
            headers={"Content-Length": "999999999"},
            server=server,
        )
        h.do_GET()
        h.send_response.assert_called_with(200)

    def test_disabled_auth_allows_all(self) -> None:
        server = _HandlerServer(auth=BearerAuth(None, warn_on_disabled=False))
        h = _make_handler("/status", server=server)  # no Authorization
        h.do_GET()
        h.send_response.assert_called_with(200)

    def test_private_endpoint_writes_audit_record(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        server = _HandlerServer(audit_logger=RequestAuditLogger(audit_path))
        h = _make_handler("/status", server=server)
        with patch("api_server.BASE", tmp_path):
            h.do_GET()

        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        assert payload["server"] == "stdlib"
        assert payload["method"] == "GET"
        assert payload["path"] == "/status"
        assert payload["client"] == "127.0.0.1"
        assert payload["status"] == 200
        assert payload["outcome"] == "ok"
        assert "authorization" not in payload
        assert "body" not in payload

    def test_public_health_does_not_write_audit_record(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        server = _HandlerServer(audit_logger=RequestAuditLogger(audit_path))
        h = _make_handler("/health", server=server)
        h.do_GET()

        assert not audit_path.exists()


# ── build_server factory ─────────────────────────────────────────────


class TestBuildServer:
    def test_build_server_uses_env_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from api_server import build_server

        monkeypatch.setenv("REMANENTIA_API_TOKEN", "env-tok")
        srv = build_server("127.0.0.1", 0)  # port 0 = kernel pick, not bound
        try:
            assert srv.auth.enabled
            assert srv.auth.check_header("Bearer env-tok") is True
            assert srv.body_limit == DEFAULT_BODY_LIMIT
        finally:
            srv.server_close()

    def test_build_server_custom_limits(self) -> None:
        from api_server import build_server

        auth = BearerAuth("x", warn_on_disabled=False)
        lim = TokenBucketLimiter(rate_per_minute=120, burst=20)
        srv = build_server("127.0.0.1", 0, auth=auth, limiter=lim, body_limit=512)
        try:
            assert srv.body_limit == 512
            assert srv.auth is auth
            assert srv.limiter is lim
        finally:
            srv.server_close()
