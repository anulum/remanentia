# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for HTTP API server

from __future__ import annotations

import http.client
import json
from contextlib import contextmanager
from collections.abc import Generator
from pathlib import Path
from threading import Thread
from typing import Any, cast

import numpy as np
import pytest

import api_server
import consolidation_engine
import knowledge_store
import memory_index
import memory_recall
from api_security import DEFAULT_BODY_LIMIT, BearerAuth, RequestAuditLogger, TokenBucketLimiter
from api_server import RemanentiaHTTPServer, _json_default, build_server


@contextmanager
def _running_server(
    *,
    auth: BearerAuth | None = None,
    limiter: TokenBucketLimiter | None = None,
    body_limit: int = DEFAULT_BODY_LIMIT,
) -> Generator[RemanentiaHTTPServer, None, None]:
    server = build_server(
        "127.0.0.1",
        0,
        auth=auth or BearerAuth(None, warn_on_disabled=False),
        limiter=limiter or TokenBucketLimiter(rate_per_minute=6000, burst=1000),
        body_limit=body_limit,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _http_json(
    server: RemanentiaHTTPServer,
    method: str,
    path: str,
    payload: object | None = None,
    *,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = raw_body
    request_headers = dict(headers or {})
    if body is None and payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        response_body = response.read()
        parsed = json.loads(response_body) if response_body else {}
        return response.status, cast(dict[str, Any], parsed), dict(response.getheaders())
    finally:
        connection.close()


def _configure_real_service_stores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_traces: Path,
    tmp_semantic: Path,
    tmp_graph: Path,
) -> None:
    state_dir = tmp_path / "snn_state"
    consolidation_dir = tmp_path / "consolidation"
    state_dir.mkdir()
    monkeypatch.setattr(api_server, "BASE", tmp_path)
    monkeypatch.setattr(memory_index, "SOURCES", {"test": tmp_traces})
    monkeypatch.setattr(memory_index, "SOURCE_EXTENSIONS", {"test": {".md"}})
    monkeypatch.setattr(memory_index, "INDEX_PATH", state_dir / "memory_index.json.gz")
    monkeypatch.setattr(memory_index, "_LEGACY_INDEX_PATH", state_dir / "legacy.pkl")
    monkeypatch.setattr(memory_index, "HASH_CACHE_PATH", state_dir / "content_hashes.json")
    monkeypatch.setattr(memory_index, "GRAPH_DIR", tmp_graph)
    monkeypatch.setattr(memory_recall, "BASE", tmp_path)
    monkeypatch.setattr(memory_recall, "TRACES_DIR", tmp_traces)
    monkeypatch.setattr(memory_recall, "SEMANTIC_DIR", tmp_semantic)
    monkeypatch.setattr(memory_recall, "GRAPH_DIR", tmp_graph)
    monkeypatch.setattr(memory_recall, "HISTORY_PATH", state_dir / "retrieval_history.jsonl")
    monkeypatch.setattr(consolidation_engine, "TRACES_DIR", tmp_traces)
    monkeypatch.setattr(consolidation_engine, "SEMANTIC_DIR", tmp_semantic)
    monkeypatch.setattr(consolidation_engine, "GRAPH_DIR", tmp_graph)
    monkeypatch.setattr(consolidation_engine, "CONSOLIDATION_DIR", consolidation_dir)
    monkeypatch.setattr(consolidation_engine, "PENDING_PATH", consolidation_dir / "pending.json")
    monkeypatch.setattr(
        consolidation_engine,
        "LAST_RUN_PATH",
        consolidation_dir / "last_consolidation.json",
    )
    monkeypatch.setattr(
        consolidation_engine,
        "CONFLICTS_PATH",
        consolidation_dir / "conflicts.json",
    )
    monkeypatch.setattr(consolidation_engine, "ENTITIES_PATH", tmp_graph / "entities.jsonl")
    monkeypatch.setattr(consolidation_engine, "RELATIONS_PATH", tmp_graph / "relations.jsonl")
    monkeypatch.setattr(consolidation_engine, "CLUSTERS_PATH", tmp_graph / "trace_clusters.json")
    monkeypatch.setattr(
        consolidation_engine,
        "SUMMARY_DAG_PATH",
        consolidation_dir / "summary_dag.json",
    )
    monkeypatch.setattr(knowledge_store, "STORE_PATH", tmp_path / "memory" / "knowledge.jsonl")
    monkeypatch.setattr(knowledge_store, "TRIGGERS_PATH", tmp_path / "memory" / "triggers.jsonl")


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


# ── GET endpoints ────────────────────────────────────────────────


class TestDoGet:
    def test_health(self) -> None:
        with _running_server() as server:
            status, body, headers = _http_json(server, "GET", "/health")

        assert status == 200
        assert body["status"] == "ok"
        assert "timestamp" in body
        assert headers["Content-Type"] == "application/json"
        assert int(headers["Content-Length"]) > 0

    def test_status_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api_server, "BASE", tmp_path)
        with _running_server() as server:
            status, body, _ = _http_json(server, "GET", "/status")

        assert status == 200
        assert body["entities"] == 0
        assert body["traces"] == 0

    def test_status_with_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (graph_dir / "entities.jsonl").write_text(
            '{"id":"a"}\n{"id":"b"}\n', encoding="utf-8"
        )
        (graph_dir / "relations.jsonl").write_text(
            '{"src":"a","tgt":"b"}\n', encoding="utf-8"
        )
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        (traces_dir / "t1.md").write_text("trace", encoding="utf-8")
        sem_dir = tmp_path / "memory" / "semantic"
        sem_dir.mkdir(parents=True)
        (sem_dir / "s1.md").write_text("mem", encoding="utf-8")
        monkeypatch.setattr(api_server, "BASE", tmp_path)

        with _running_server() as server:
            status, body, _ = _http_json(server, "GET", "/status")

        assert status == 200
        assert body["entities"] == 2
        assert body["relations"] == 1
        assert body["traces"] == 1
        assert body["memories"] == 1

    def test_unknown_get(self) -> None:
        with _running_server() as server:
            status, body, _ = _http_json(server, "GET", "/unknown")

        assert status == 404
        assert "error" in body


# ── POST endpoints ───────────────────────────────────────────────


class TestDoPost:
    def test_recall_empty_query(self) -> None:
        with _running_server() as server:
            status, body, _ = _http_json(server, "POST", "/recall", {"query": ""})

        assert status == 400
        assert "error" in body

    def test_recall_top_k_zero(self) -> None:
        with _running_server() as server:
            status, body, _ = _http_json(
                server,
                "POST",
                "/recall",
                {"query": "test", "top_k": 0},
            )

        assert status == 200
        assert body["results"] == []

    def test_recall_rejects_non_integer_top_k(self) -> None:
        with _running_server() as server:
            status, body, _ = _http_json(
                server,
                "POST",
                "/recall",
                {"query": "test", "top_k": "five"},
            )

        assert status == 400
        assert body["error"] == "top_k must be an integer"

    def test_consolidate_rejects_non_boolean_force(self) -> None:
        with _running_server() as server:
            status, body, _ = _http_json(
                server,
                "POST",
                "/consolidate",
                {"force": "yes"},
            )

        assert status == 400
        assert body["error"] == "force must be a boolean"

    def test_remember_empty_content(self) -> None:
        with _running_server() as server:
            status, _, _ = _http_json(server, "POST", "/remember", {"content": ""})

        assert status == 400

    def test_remember_rejects_non_string_trigger(self) -> None:
        with _running_server() as server:
            status, body, _ = _http_json(
                server,
                "POST",
                "/remember",
                {"content": "Test fact", "trigger": 5},
            )

        assert status == 400
        assert body["error"] == "trigger must be a string"

    def test_unknown_post(self) -> None:
        with _running_server() as server:
            status, body, _ = _http_json(server, "POST", "/unknown", {})

        assert status == 404
        assert "Unknown path" in body["error"]


# ── JSON response ────────────────────────────────────────────────


class TestJsonResponse:
    def test_cors_scoped_echoes_listed_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REMANENTIA_CORS_ORIGINS", "https://ok.example")
        with _running_server() as server:
            status, _, headers = _http_json(
                server,
                "GET",
                "/health",
                headers={"Origin": "https://ok.example"},
            )

        assert status == 200
        assert headers["Access-Control-Allow-Origin"] == "https://ok.example"

    def test_cors_scoped_omits_header_for_unlisted_origin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REMANENTIA_CORS_ORIGINS", "https://ok.example")
        with _running_server() as server:
            status, _, headers = _http_json(
                server,
                "GET",
                "/health",
                headers={"Origin": "https://evil.example"},
            )

        assert status == 200
        assert "Access-Control-Allow-Origin" not in headers


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestAPIServerPipeline:
    def test_real_http_recall_returns_empty_results_for_empty_store(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        traces = tmp_path / "reasoning_traces"
        semantic = tmp_path / "memory" / "semantic"
        graph = tmp_path / "memory" / "graph"
        traces.mkdir()
        semantic.mkdir(parents=True)
        graph.mkdir(parents=True)
        _configure_real_service_stores(monkeypatch, tmp_path, traces, semantic, graph)

        with _running_server() as server:
            status, body, _ = _http_json(
                server,
                "POST",
                "/recall",
                {"query": "uniquely absent memory", "top_k": 1},
            )

        assert status == 200
        assert body["results"] == []

    def test_real_http_recall_returns_trace_and_semantic_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_traces: Path,
        tmp_semantic: Path,
        tmp_graph: Path,
    ) -> None:
        _configure_real_service_stores(
            monkeypatch,
            tmp_path,
            tmp_traces,
            tmp_semantic,
            tmp_graph,
        )
        with _running_server() as server:
            status, body, headers = _http_json(
                server,
                "POST",
                "/recall",
                {"query": "STDP retrieval", "top_k": 2},
            )

        assert status == 200
        assert headers["Content-Type"] == "application/json"
        assert body["query"] == "STDP retrieval"
        assert body["results"][0]["name"] == "2026-03-15_decision_stdp_removal.md"
        assert any(result["type"] == "semantic" for result in body["results"])
        assert "stdp" in body["entities"]

    def test_real_http_consolidate_and_remember_persist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_traces: Path,
        tmp_semantic: Path,
        tmp_graph: Path,
    ) -> None:
        _configure_real_service_stores(
            monkeypatch,
            tmp_path,
            tmp_traces,
            tmp_semantic,
            tmp_graph,
        )
        with _running_server() as server:
            consolidate_status, consolidate_body, _ = _http_json(
                server,
                "POST",
                "/consolidate",
                {"force": False},
            )
            remember_status, remember_body, _ = _http_json(
                server,
                "POST",
                "/remember",
                {
                    "content": "Remanentia stores this real production knowledge note.",
                    "trigger": "remanentia",
                },
            )

        assert consolidate_status == 200
        assert consolidate_body["status"] == "ok"
        assert consolidate_body["new_memories"] > 0
        assert remember_status == 200
        assert remember_body["status"] == "ok"
        stored_rows = knowledge_store.STORE_PATH.read_text(encoding="utf-8")
        assert "real production knowledge note" in stored_rows

    def test_real_http_rejects_malformed_and_non_object_json(self) -> None:
        with _running_server() as server:
            empty_status, empty_body, _ = _http_json(server, "POST", "/recall")
            malformed_status, malformed_body, _ = _http_json(
                server,
                "POST",
                "/recall",
                raw_body=b'{"query":',
                headers={"Content-Type": "application/json"},
            )
            non_object_status, non_object_body, _ = _http_json(
                server,
                "POST",
                "/recall",
                raw_body=b'["not", "an", "object"]',
                headers={"Content-Type": "application/json"},
            )

        assert empty_status == 400
        assert empty_body["error"] == "query required"
        assert malformed_status == 400
        assert malformed_body["error"] == "query required"
        assert non_object_status == 400
        assert non_object_body["error"] == "query required"

    def test_real_service_failures_return_500(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_traces: Path,
        tmp_semantic: Path,
        tmp_graph: Path,
    ) -> None:
        _configure_real_service_stores(
            monkeypatch,
            tmp_path,
            tmp_traces,
            tmp_semantic,
            tmp_graph,
        )
        (tmp_graph / "entities.jsonl").write_text("{invalid json", encoding="utf-8")
        blocked_consolidation = tmp_path / "blocked-consolidation"
        blocked_consolidation.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(consolidation_engine, "CONSOLIDATION_DIR", blocked_consolidation)
        blocked_store = tmp_path / "blocked-store"
        blocked_store.mkdir()
        monkeypatch.setattr(knowledge_store, "STORE_PATH", blocked_store)

        with _running_server() as server:
            recall_status, _, _ = _http_json(
                server,
                "POST",
                "/recall",
                {"query": "STDP retrieval"},
            )
            consolidate_status, _, _ = _http_json(
                server,
                "POST",
                "/consolidate",
                {"force": False},
            )
            remember_status, _, _ = _http_json(
                server,
                "POST",
                "/remember",
                {"content": "This write must fail on a real directory conflict."},
            )

        assert recall_status == 500
        assert consolidate_status == 500
        assert remember_status == 500


# ── Security gate integration ────────────────────────────────────────


class TestSecurityGates:
    """Exercise auth, rate, size and audit gates over a real TCP server."""

    def test_auth_accepts_valid_and_rejects_missing_wrong_and_non_ascii(self) -> None:
        auth = BearerAuth("valid-value", warn_on_disabled=False)
        with _running_server(auth=auth) as server:
            missing_status, missing_body, _ = _http_json(server, "GET", "/status")
            valid_status, _, _ = _http_json(
                server,
                "GET",
                "/status",
                headers={"Authorization": "Bearer valid-value"},
            )
            wrong_status, _, _ = _http_json(
                server,
                "GET",
                "/status",
                headers={"Authorization": "Bearer wrong"},
            )
            non_ascii_status, _, _ = _http_json(
                server,
                "GET",
                "/status",
                headers={"Authorization": "Bearer ££"},
            )
            health_status, health_body, _ = _http_json(server, "GET", "/health")

        assert missing_status == 401
        assert "authentication" in missing_body["error"].lower()
        assert valid_status == 200
        assert wrong_status == 401
        assert non_ascii_status == 401
        assert health_status == 200
        assert health_body["status"] == "ok"

    def test_rate_limit_and_health_exemption(self) -> None:
        limiter = TokenBucketLimiter(rate_per_minute=60, burst=1)
        with _running_server(limiter=limiter) as server:
            first_status, _, _ = _http_json(server, "GET", "/status")
            second_status, second_body, second_headers = _http_json(server, "GET", "/status")
            health_status, _, _ = _http_json(server, "GET", "/health")

        assert first_status == 200
        assert second_status == 429
        assert "rate limit" in second_body["error"].lower()
        assert second_headers["Retry-After"] == "1"
        assert health_status == 200

    def test_body_limit_rejects_large_and_allows_small_post(self) -> None:
        with _running_server(body_limit=20) as server:
            large_status, large_body, _ = _http_json(
                server,
                "POST",
                "/recall",
                {"query": "hello world beyond limit"},
            )
            health_status, _, _ = _http_json(
                server,
                "GET",
                "/health",
                headers={"Content-Length": "999999999"},
            )

        with _running_server() as server:
            small_status, _, _ = _http_json(
                server,
                "POST",
                "/recall",
                {"query": "hi", "top_k": 0},
            )

        assert large_status == 413
        assert "exceeds" in large_body["error"].lower()
        assert health_status == 200
        assert small_status == 200

    def test_disabled_auth_and_audit_boundaries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_server, "BASE", tmp_path)
        audit_path = tmp_path / "audit.jsonl"
        with _running_server(auth=BearerAuth(None, warn_on_disabled=False)) as server:
            server.audit_logger = RequestAuditLogger(audit_path)
            status, _, _ = _http_json(server, "GET", "/status")
            health_status, _, _ = _http_json(server, "GET", "/health")

        assert status == 200
        assert health_status == 200
        rows = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 1
        payload = json.loads(rows[0])
        assert payload["server"] == "stdlib"
        assert payload["method"] == "GET"
        assert payload["path"] == "/status"
        assert payload["client"] == "127.0.0.1"
        assert payload["status"] == 200
        assert payload["outcome"] == "ok"
        assert "authorization" not in payload
        assert "body" not in payload


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
