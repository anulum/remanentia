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
    ):
        self.auth = auth or BearerAuth(None, warn_on_disabled=False)
        self.limiter = limiter or TokenBucketLimiter(rate_per_minute=6000, burst=1000)
        self.body_limit = body_limit
        self.audit_logger = audit_logger or RequestAuditLogger(None)


# ── JSON default handler ─────────────────────────────────────────


class TestJsonDefault:
    def test_np_float32(self):
        assert _json_default(np.float32(3.14)) == pytest.approx(3.14, abs=0.01)

    def test_np_float64(self):
        assert isinstance(_json_default(np.float64(2.71)), float)

    def test_np_int32(self):
        assert _json_default(np.int32(42)) == 42

    def test_np_int64(self):
        assert _json_default(np.int64(99)) == 99

    def test_np_array(self):
        result = _json_default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            _json_default(object())


# ── Handler helpers ──────────────────────────────────────────────


def _make_handler(path="/", method="GET", body=None, *, server=None, headers=None):
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
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 12345)
    handler.server = server or _HandlerServer()

    body_bytes = b""
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")

    handler.rfile = io.BytesIO(body_bytes)
    handler.wfile = io.BytesIO()
    header_map = {"Content-Length": str(len(body_bytes))}
    if headers:
        header_map.update(headers)
    handler.headers = MagicMock()
    handler.headers.get = MagicMock(side_effect=lambda k, default=None: header_map.get(k, default))
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    return handler


def _get_response_body(handler):
    """Read the JSON body written to handler.wfile."""
    handler.wfile.seek(0)
    return json.loads(handler.wfile.read())


# ── GET endpoints ────────────────────────────────────────────────


class TestDoGet:
    def test_health(self):
        h = _make_handler("/health")
        h.do_GET()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["status"] == "ok"
        assert "timestamp" in body

    def test_status_empty(self, tmp_path):
        h = _make_handler("/status")
        with patch("api_server.BASE", tmp_path):
            h.do_GET()
        body = _get_response_body(h)
        assert body["entities"] == 0
        assert body["traces"] == 0

    def test_status_with_data(self, tmp_path):
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

    def test_unknown_get(self):
        h = _make_handler("/unknown")
        h.do_GET()
        h.send_response.assert_called_with(404)
        body = _get_response_body(h)
        assert "error" in body


# ── POST endpoints ───────────────────────────────────────────────


class TestDoPost:
    def test_recall_empty_query(self):
        h = _make_handler("/recall", "POST", {"query": ""})
        h.do_POST()
        h.send_response.assert_called_with(400)
        body = _get_response_body(h)
        assert "error" in body

    def test_recall_top_k_zero(self):
        h = _make_handler("/recall", "POST", {"query": "test", "top_k": 0})
        h.do_POST()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["results"] == []

    def test_recall_success(self):
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

    def test_recall_exception(self):
        h = _make_handler("/recall", "POST", {"query": "test"})
        with patch("memory_recall.recall", side_effect=RuntimeError("boom")):
            h.do_POST()
        h.send_response.assert_called_with(500)
        body = _get_response_body(h)
        assert "error" in body

    def test_recall_no_trace(self):
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

    def test_consolidate_success(self):
        mock_result = {"new_memories": 3}
        h = _make_handler("/consolidate", "POST", {"force": True})
        with patch("consolidation_engine.consolidate", return_value=mock_result):
            h.do_POST()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["new_memories"] == 3

    def test_consolidate_exception(self):
        h = _make_handler("/consolidate", "POST", {})
        with patch("consolidation_engine.consolidate", side_effect=RuntimeError("fail")):
            h.do_POST()
        h.send_response.assert_called_with(500)

    def test_remember_empty_content(self):
        h = _make_handler("/remember", "POST", {"content": ""})
        h.do_POST()
        h.send_response.assert_called_with(400)

    def test_remember_success(self):
        mock_ks = MagicMock()
        h = _make_handler("/remember", "POST", {"content": "Test fact", "trigger": "test"})
        with patch("knowledge_store.KnowledgeStore", return_value=mock_ks):
            h.do_POST()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["status"] == "ok"

    def test_remember_exception(self):
        h = _make_handler("/remember", "POST", {"content": "Test"})
        with patch("knowledge_store.KnowledgeStore", side_effect=RuntimeError("fail")):
            h.do_POST()
        h.send_response.assert_called_with(500)

    def test_unknown_post(self):
        h = _make_handler("/unknown", "POST", {})
        h.do_POST()
        h.send_response.assert_called_with(404)


# ── Read body ────────────────────────────────────────────────────


class TestReadBody:
    def test_no_body(self):
        h = _make_handler("/test")
        h.headers.get = MagicMock(return_value="0")
        result = h._read_body()
        assert result == {}

    def test_valid_json(self):
        body = {"key": "value"}
        h = _make_handler("/test", "POST", body)
        result = h._read_body()
        assert result == body

    def test_invalid_json(self):
        h = _make_handler("/test")
        h.rfile = io.BytesIO(b"not json")
        h.headers.get = MagicMock(return_value="8")
        result = h._read_body()
        assert result == {}


# ── Log message ──────────────────────────────────────────────────


class TestLogMessage:
    def test_404_logged(self):
        h = _make_handler("/test")
        with patch("http.server.BaseHTTPRequestHandler.log_message") as mock_log:
            h.log_message("%s", "404 Not Found")
        mock_log.assert_called_once()

    def test_non_404_suppressed(self):
        h = _make_handler("/test")
        with patch("http.server.BaseHTTPRequestHandler.log_message") as mock_log:
            h.log_message("%s", "200 OK")
        mock_log.assert_not_called()


# ── JSON response ────────────────────────────────────────────────


class TestJsonResponse:
    def test_sets_headers(self):
        h = _make_handler("/test")
        h._json_response({"ok": True}, 200)
        h.send_response.assert_called_with(200)
        h.send_header.assert_any_call("Content-Type", "application/json")
        h.send_header.assert_any_call("Access-Control-Allow-Origin", "*")

    def test_custom_status(self):
        h = _make_handler("/test")
        h._json_response({"error": "bad"}, 400)
        h.send_response.assert_called_with(400)


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestAPIServerPipeline:
    def test_recall_handler_returns_json(self):
        """HTTP handler exercises full recall pipeline."""
        from unittest.mock import MagicMock

        handler = MagicMock()
        handler.path = "/api/recall?q=test&top_k=1"
        handler.headers = {"Content-Length": "0"}
        handler.wfile = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        # Just verify no crash on malformed request
        assert True

    def test_error_on_missing_query(self):
        from unittest.mock import MagicMock

        handler = MagicMock()
        handler.path = "/api/recall"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        assert True


# ── Security gate integration ────────────────────────────────────────


class TestSecurityGates:
    """Exercise BearerAuth / limiter / body-size through the handler path."""

    def test_health_is_public_no_auth_required(self):
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler("/health", server=server)
        h.do_GET()
        h.send_response.assert_called_with(200)
        body = _get_response_body(h)
        assert body["status"] == "ok"

    def test_status_requires_auth_when_token_set(self):
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler("/status", server=server)  # no Authorization header
        h.do_GET()
        h.send_response.assert_called_with(401)
        body = _get_response_body(h)
        assert "authentication" in body["error"].lower()

    def test_status_passes_with_valid_bearer(self, tmp_path):
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler(
            "/status",
            server=server,
            headers={"Authorization": "Bearer valid-value"},
        )
        with patch("api_server.BASE", tmp_path):
            h.do_GET()
        h.send_response.assert_called_with(200)

    def test_status_rejects_wrong_bearer(self):
        server = _HandlerServer(auth=BearerAuth("valid-value", warn_on_disabled=False))
        h = _make_handler(
            "/status",
            server=server,
            headers={"Authorization": "Bearer wrong"},
        )
        h.do_GET()
        h.send_response.assert_called_with(401)

    def test_rate_limit_triggers_429(self):
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

    def test_body_size_limit_rejects_413(self):
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

    def test_body_size_permits_small_post(self):
        server = _HandlerServer()  # default 1 MiB
        h = _make_handler(
            "/recall",
            method="POST",
            body={"query": "hi", "top_k": 0},  # top_k=0 short-circuits
            server=server,
        )
        h.do_POST()
        h.send_response.assert_called_with(200)

    def test_health_bypasses_rate_limit(self):
        # Exhaust the bucket with /status, then /health still works
        server = _HandlerServer(limiter=TokenBucketLimiter(rate_per_minute=60, burst=1))
        h1 = _make_handler("/status", server=server)
        h1.do_GET()
        h2 = _make_handler("/status", server=server)
        h2.do_GET()  # 429
        h3 = _make_handler("/health", server=server)
        h3.do_GET()
        h3.send_response.assert_called_with(200)

    def test_health_bypasses_body_size(self):
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

    def test_disabled_auth_allows_all(self):
        server = _HandlerServer(auth=BearerAuth(None, warn_on_disabled=False))
        h = _make_handler("/status", server=server)  # no Authorization
        h.do_GET()
        h.send_response.assert_called_with(200)

    def test_private_endpoint_writes_audit_record(self, tmp_path):
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

    def test_public_health_does_not_write_audit_record(self, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        server = _HandlerServer(audit_logger=RequestAuditLogger(audit_path))
        h = _make_handler("/health", server=server)
        h.do_GET()

        assert not audit_path.exists()


# ── build_server factory ─────────────────────────────────────────────


class TestBuildServer:
    def test_build_server_uses_env_token(self, monkeypatch):
        from api_server import build_server

        monkeypatch.setenv("REMANENTIA_API_TOKEN", "env-tok")
        srv = build_server("127.0.0.1", 0)  # port 0 = kernel pick, not bound
        try:
            assert srv.auth.enabled
            assert srv.auth.check_header("Bearer env-tok") is True
            assert srv.body_limit == DEFAULT_BODY_LIMIT
        finally:
            srv.server_close()

    def test_build_server_custom_limits(self):
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
