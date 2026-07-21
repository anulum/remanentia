# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for API module

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from unittest.mock import patch

import pytest

import api
import consolidation_engine
import memory_index
import memory_recall
from api import app
from api_security import BearerAuth, RequestAuditLogger, TokenBucketLimiter
from vector_index import HttpEmbeddingClient, PersistentVectorIndex, VectorChunk

try:
    from fastapi import Request
    from fastapi.testclient import TestClient

    HAS_TEST_CLIENT = True
except ImportError:
    Request = Any
    HAS_TEST_CLIENT = False

pytestmark = pytest.mark.skipif(not HAS_TEST_CLIENT, reason="fastapi/httpx not installed")


@pytest.fixture  # type: ignore[untyped-decorator]
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        api,
        "LIMITER",
        TokenBucketLimiter(rate_per_minute=60000, burst=1000),
        raising=False,
    )
    monkeypatch.setattr(api, "AUDIT_LOGGER", RequestAuditLogger(None), raising=False)
    return TestClient(app)


@pytest.fixture  # type: ignore[untyped-decorator]
def real_recall_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_traces: Path,
    tmp_semantic: Path,
    tmp_graph: Path,
) -> None:
    state_dir = tmp_path / "snn_state"
    state_dir.mkdir()
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


class _EmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        vectors = []
        for index, text in enumerate(payload["input"]):
            lower = str(text).lower()
            vectors.append(
                {
                    "index": index,
                    "embedding": [
                        1.0 if "alpha" in lower else 0.0,
                        1.0 if "beta" in lower else 0.0,
                        0.25,
                    ],
                }
            )
        body = json.dumps({"data": vectors}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture  # type: ignore[untyped-decorator]
def embedding_endpoint() -> Generator[str, None, None]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _build_public_vector_index(
    index_dir: Path,
    embedding_endpoint: str,
    *,
    text: str,
    source: str,
    path: str,
) -> None:
    provider = HttpEmbeddingClient(embedding_endpoint, "test-model")
    PersistentVectorIndex(index_dir).build(
        [
            VectorChunk(
                chunk_id="one",
                text=text,
                source=source,
                metadata={
                    "document": Path(path).name,
                    "document_type": source,
                    "paragraph_idx": 0,
                    "path": path,
                },
            )
        ],
        provider,
    )


def _configure_embedding_environment(
    monkeypatch: pytest.MonkeyPatch,
    embedding_endpoint: str,
    index_dir: Path,
) -> None:
    monkeypatch.setenv("REMANENTIA_EMBEDDING_BASE_URL", embedding_endpoint)
    monkeypatch.setenv("REMANENTIA_EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("REMANENTIA_VECTOR_INDEX_DIR", str(index_dir))


# ── Health ───────────────────────────────────────────────────────


class TestHealth:
    def test_health_no_daemon(self, client: Any, tmp_path: Path) -> None:
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        with patch("api.STATE_DIR", state_dir):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["daemon"] == "stale"
        assert data["vector_worker"] == "missing"
        assert data["version"] == "0.5.0"

    def test_health_daemon_alive(self, client: Any, tmp_path: Path) -> None:
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        state = {"timestamp": time.time(), "cycle": 10}
        (state_dir / "current_state.json").write_text(json.dumps(state), encoding="utf-8")

        with patch("api.STATE_DIR", state_dir):
            resp = client.get("/health")
        assert resp.json()["daemon"] == "alive"

    def test_health_vector_worker_alive(self, client: Any, tmp_path: Path) -> None:
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        worker_state = {
            "cycle": 3,
            "pid": 1234,
            "result": {"action": "skipped"},
            "status": "ok",
            "timestamp_unix": time.time(),
        }
        (state_dir / "vector_refresh_worker.json").write_text(
            json.dumps(worker_state), encoding="utf-8"
        )

        with patch("api.STATE_DIR", state_dir):
            resp = client.get("/health")

        payload = resp.json()
        assert payload["daemon"] == "alive"
        assert payload["daemon_kind"] == "vector_worker"
        assert payload["legacy_daemon"] == "stale"
        assert payload["vector_worker"] == "alive"


# ── Security boundary ──────────────────────────────────────────


class TestAPISecurityBoundary:
    def test_private_endpoint_requires_bearer_token_when_configured(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            api, "AUTH", BearerAuth("secret", warn_on_disabled=False), raising=False
        )
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)

        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            resp = client.get("/status")

        assert resp.status_code == 401
        assert resp.json()["detail"] == "authentication required"

    def test_private_endpoint_accepts_matching_bearer_token(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            api, "AUTH", BearerAuth("secret", warn_on_disabled=False), raising=False
        )
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)

        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            resp = client.get("/status", headers={"Authorization": "Bearer secret"})

        assert resp.status_code == 200
        assert resp.json()["episodic_traces"] == 0

    def test_post_body_limit_rejects_before_handler(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api, "BODY_LIMIT", 8, raising=False)
        resp = client.post("/recall", json={"query": "body is too large"})

        assert resp.status_code == 413
        assert resp.json()["detail"] == "request body too large"

    def test_private_endpoint_rate_limit_is_enforced(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            api,
            "LIMITER",
            TokenBucketLimiter(rate_per_minute=60, burst=1),
            raising=False,
        )
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)

        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            first = client.get("/status")
            second = client.get("/status")

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["detail"] == "rate limit exceeded"
        assert second.headers["retry-after"] == "1"

    def test_default_cors_origin_allows_local_development(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("REMANENTIA_CORS_ORIGINS", raising=False)

        assert api._cors_origins_from_env() == ["*"]

    def test_cors_origins_can_be_scoped_by_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "REMANENTIA_CORS_ORIGINS",
            "https://remanentia.com, https://www.remanentia.com",
        )

        assert api._cors_origins_from_env() == [
            "https://remanentia.com",
            "https://www.remanentia.com",
        ]

    def test_retry_after_ceilings_match_rate_limit(self) -> None:
        assert TokenBucketLimiter(rate_per_minute=60).retry_after_seconds() == "1"
        assert TokenBucketLimiter(rate_per_minute=30).retry_after_seconds() == "2"
        assert TokenBucketLimiter(rate_per_minute=7).retry_after_seconds() == "9"

    def test_private_endpoint_writes_audit_record(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audit_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(api, "AUDIT_LOGGER", RequestAuditLogger(audit_path), raising=False)
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)

        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            resp = client.get("/status")

        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        assert resp.status_code == 200
        assert payload["server"] == "fastapi"
        assert payload["method"] == "GET"
        assert payload["path"] == "/status"
        assert payload["status"] == 200
        assert payload["outcome"] == "ok"
        assert "authorization" not in payload
        assert "body" not in payload

    def test_public_health_does_not_write_audit_record(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audit_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(api, "AUDIT_LOGGER", RequestAuditLogger(audit_path), raising=False)

        resp = client.get("/health")

        assert resp.status_code == 200
        assert not audit_path.exists()


# ── Status ───────────────────────────────────────────────────────


class TestStatus:
    def test_status_empty(self, client: Any, tmp_path: Path) -> None:
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        semantic_dir = tmp_path / "memory" / "semantic"
        semantic_dir.mkdir(parents=True)
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()

        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            resp = client.get("/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["episodic_traces"] == 0
        assert data["semantic_memories"] == 0

    def test_status_with_data(self, client: Any, tmp_path: Path) -> None:
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        (traces_dir / "t1.md").write_text("x", encoding="utf-8")
        (traces_dir / "t2.md").write_text("x", encoding="utf-8")
        semantic_dir = tmp_path / "memory" / "semantic"
        semantic_dir.mkdir(parents=True)
        (semantic_dir / "s1.md").write_text("x", encoding="utf-8")
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        entities = [{"id": "a"}, {"id": "b"}]
        (graph_dir / "entities.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entities) + "\n",
            encoding="utf-8",
        )
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()

        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            resp = client.get("/status")

        data = resp.json()
        assert data["episodic_traces"] == 2
        assert data["semantic_memories"] == 1
        assert data["entities"] == 2


# ── Entities ─────────────────────────────────────────────────────


class TestEntities:
    def test_no_entities(self, client: Any, tmp_path: Path) -> None:
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with patch("api.GRAPH_DIR", graph_dir):
            resp = client.get("/entities")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_entities(self, client: Any, tmp_graph: Path) -> None:
        with patch("api.GRAPH_DIR", tmp_graph):
            resp = client.get("/entities")
        data = resp.json()
        assert len(data) == 4
        ids = {e["id"] for e in data}
        assert "stdp" in ids


# ── Graph ────────────────────────────────────────────────────────


class TestGraph:
    def test_no_relations(self, client: Any, tmp_path: Path) -> None:
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with patch("api.GRAPH_DIR", graph_dir):
            resp = client.get("/graph")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_top_relations(self, client: Any, tmp_graph: Path) -> None:
        with patch("api.GRAPH_DIR", tmp_graph):
            resp = client.get("/graph?top=2")
        data = resp.json()
        assert len(data) == 2
        # Sorted by weight descending
        assert data[0]["weight"] >= data[1]["weight"]

    def test_entity_detail(self, client: Any, tmp_graph: Path) -> None:
        with patch("api.GRAPH_DIR", tmp_graph):
            resp = client.get("/graph/entity/stdp")
        data = resp.json()
        assert data["entity"]["id"] == "stdp"
        assert len(data["connections"]) > 0

    def test_entity_not_found(self, client: Any, tmp_graph: Path) -> None:
        with patch("api.GRAPH_DIR", tmp_graph):
            resp = client.get("/graph/entity/nonexistent")
        data = resp.json()
        assert "error" in data

    def test_entity_detail_handles_missing_graph_files(self, client: Any, tmp_path: Path) -> None:
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()

        with patch("api.GRAPH_DIR", graph_dir):
            resp = client.get("/graph/entity/stdp")

        assert resp.json() == {"error": "Entity 'stdp' not found"}

    def test_entity_detail_handles_blank_rows_and_missing_relations(
        self, client: Any, tmp_path: Path
    ) -> None:
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "entities.jsonl").write_text(
            '{"id":"other","label":"Other"}\n\n{"id":"stdp","label":"STDP"}\n',
            encoding="utf-8",
        )

        with patch("api.GRAPH_DIR", graph_dir):
            without_relations = client.get("/graph/entity/stdp")

        assert without_relations.json()["connections"] == []

        (graph_dir / "relations.jsonl").write_text(
            '{"source":"other","target":"third","type":"related","weight":1}\n\n'
            '{"source":"stdp","target":"other","type":"uses","weight":2}\n',
            encoding="utf-8",
        )
        with patch("api.GRAPH_DIR", graph_dir):
            with_relations = client.get("/graph/entity/stdp")

        assert with_relations.json()["connections"] == [
            {"entity": "other", "weight": 2, "relation": "uses", "evidence": []}
        ]


# ── Recall ───────────────────────────────────────────────────────


class TestRecall:
    def test_recall_summary_uses_real_memory_pipeline(
        self, client: Any, real_recall_store: None
    ) -> None:
        resp = client.post(
            "/recall",
            json={"query": "STDP retrieval", "top_k": 2, "include_content": True},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "STDP retrieval"
        assert data["trace"] == "2026-03-15_decision_stdp_removal.md"
        assert data["score"] > 0
        assert "Decision: Remove SNN" in data["snippet"]
        assert data["semantic_memories"][0]["path"].endswith("2026-03-15_remanentia-decision.md")
        assert "stdp" in data["entities"]
        assert data["elapsed_ms"] >= 0

    def test_recall_context_uses_real_memory_pipeline(
        self, client: Any, real_recall_store: None
    ) -> None:
        resp = client.post(
            "/recall",
            json={
                "query": "STDP retrieval",
                "format": "context",
                "include_content": True,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "[Matched trace: 2026-03-15_decision_stdp_removal.md]" in data["context"]
        assert "[Consolidated:" in data["context"]
        assert data["elapsed_ms"] >= 0


class TestPublicVectorSearch:
    def test_public_vector_search_requires_query(self, client: Any) -> None:
        resp = client.post("/vector/search/public", json={"query": ""})

        assert resp.status_code == 400

    def test_public_vector_search_top_k_zero_short_circuits(self, client: Any) -> None:
        resp = client.post("/vector/search/public", json={"query": "beta", "top_k": 0})

        assert resp.status_code == 200
        assert resp.json() == {"query": "beta", "results": []}

    def test_public_vector_search_no_allowlist_returns_no_results(
        self,
        client: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        embedding_endpoint: str,
    ) -> None:
        index_dir = tmp_path / "vector"
        _build_public_vector_index(
            index_dir,
            embedding_endpoint,
            text="beta public paragraph",
            source="paper",
            path="paper/remanentia.md",
        )
        _configure_embedding_environment(monkeypatch, embedding_endpoint, index_dir)
        monkeypatch.delenv("REMANENTIA_PUBLIC_VECTOR_SOURCES", raising=False)
        monkeypatch.delenv("REMANENTIA_PUBLIC_VECTOR_PATH_PREFIXES", raising=False)
        monkeypatch.delenv("REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE", raising=False)

        resp = client.post("/vector/search/public", json={"query": "beta", "top_k": 1})

        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_public_vector_search_uses_server_policy_and_redaction(
        self,
        client: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        embedding_endpoint: str,
    ) -> None:
        redaction_file = tmp_path / "terms.txt"
        redaction_file.write_text("private-token\n", encoding="utf-8")
        index_dir = tmp_path / "vector"
        _build_public_vector_index(
            index_dir,
            embedding_endpoint,
            text="beta private-token public paragraph",
            source="paper",
            path="paper/private-token-paper.md",
        )
        _configure_embedding_environment(monkeypatch, embedding_endpoint, index_dir)
        monkeypatch.setenv("REMANENTIA_PUBLIC_VECTOR_SOURCES", "paper")
        monkeypatch.setenv("REMANENTIA_PUBLIC_VECTOR_PATH_PREFIXES", "paper")
        monkeypatch.setenv("REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE", str(redaction_file))

        resp = client.post("/vector/search/public", json={"query": "beta", "top_k": 1})

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["results"][0]["text"] == "beta [redacted] public paragraph"
        assert payload["results"][0]["metadata"]["document"] == "[redacted]-paper.md"
        assert payload["results"][0]["redactions"] == 3

    def test_public_vector_search_backend_error_returns_503(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("REMANENTIA_EMBEDDING_BASE_URL", raising=False)
        monkeypatch.delenv("REMANENTIA_EMBEDDING_MODEL", raising=False)
        resp = client.post("/vector/search/public", json={"query": "beta", "top_k": 1})

        assert resp.status_code == 503
        assert "base_url must not be empty" in resp.json()["detail"]

    def test_missing_redaction_file_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = tmp_path / "missing_terms.txt"
        monkeypatch.setenv("REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE", str(missing))

        with pytest.raises(ValueError, match="REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE"):
            api._read_terms_from_env("REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE")

    def test_json_number_falls_back_for_invalid_values(self) -> None:
        assert api._json_number("not-a-number", default=12.0) == 12.0
        assert api._json_number(["not", "scalar"], default=7.0) == 7.0

    def test_consolidate_endpoint_runs_real_store_pipeline(
        self,
        client: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        tmp_traces: Path,
    ) -> None:
        semantic_dir = tmp_path / "memory" / "semantic"
        graph_dir = tmp_path / "memory" / "graph"
        consolidation_dir = tmp_path / "consolidation"
        monkeypatch.setattr(consolidation_engine, "TRACES_DIR", tmp_traces)
        monkeypatch.setattr(consolidation_engine, "SEMANTIC_DIR", semantic_dir)
        monkeypatch.setattr(consolidation_engine, "GRAPH_DIR", graph_dir)
        monkeypatch.setattr(consolidation_engine, "CONSOLIDATION_DIR", consolidation_dir)
        monkeypatch.setattr(
            consolidation_engine, "PENDING_PATH", consolidation_dir / "pending.json"
        )
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
        monkeypatch.setattr(consolidation_engine, "ENTITIES_PATH", graph_dir / "entities.jsonl")
        monkeypatch.setattr(consolidation_engine, "RELATIONS_PATH", graph_dir / "relations.jsonl")
        monkeypatch.setattr(
            consolidation_engine,
            "CLUSTERS_PATH",
            graph_dir / "trace_clusters.json",
        )
        monkeypatch.setattr(
            consolidation_engine,
            "SUMMARY_DAG_PATH",
            consolidation_dir / "summary_dag.json",
        )

        resp = client.post("/consolidate", json={"force": False})

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["traces_processed"] == 3
        assert payload["memories_written"] > 0
        assert list(semantic_dir.rglob("*.md"))
        assert (consolidation_dir / "last_consolidation.json").exists()

    def test_status_with_daemon_state(self, client: Any, tmp_path: Path) -> None:
        import time

        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        state = {
            "timestamp": time.time(),
            "cycle": 5,
            "n_neurons": 200,
            "vram_mb": 100,
            "live_retrieval_available": True,
        }
        (state_dir / "current_state.json").write_text(json.dumps(state), encoding="utf-8")
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)
        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            resp = client.get("/status")
        data = resp.json()
        assert "daemon" in data
        assert data["daemon"]["cycle"] == 5

    def test_health_reports_unreadable_legacy_daemon_state(
        self, client: Any, tmp_path: Path
    ) -> None:
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        (state_dir / "current_state.json").write_text("{", encoding="utf-8")

        with patch("api.STATE_DIR", state_dir):
            resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["legacy_daemon"] == "unreadable"

    def test_status_with_vector_worker_state(self, client: Any, tmp_path: Path) -> None:
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        worker_state = {
            "cycle": 7,
            "pid": 1234,
            "result": {"action": "skipped"},
            "status": "ok",
            "timestamp_unix": time.time(),
        }
        (state_dir / "vector_refresh_worker.json").write_text(
            json.dumps(worker_state), encoding="utf-8"
        )
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)

        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            resp = client.get("/status")

        data = resp.json()
        assert data["vector_worker"]["state"] == "alive"
        assert data["vector_worker"]["last_action"] == "skipped"

    def test_status_reports_unreadable_vector_worker_state(
        self, client: Any, tmp_path: Path
    ) -> None:
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        (state_dir / "vector_refresh_worker.json").write_text("{", encoding="utf-8")
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)

        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            resp = client.get("/status")

        assert resp.status_code == 200
        assert resp.json()["vector_worker"]["state"] == "unreadable"


class TestAPIHelpers:
    def test_json_safe_keeps_value_when_item_conversion_rejects(self) -> None:
        class RejectingScalar:
            def item(self) -> object:
                raise ValueError("not scalar")

        value = RejectingScalar()

        assert api._json_safe(value) is value

    def test_middleware_audits_handler_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audit_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(api, "AUTH", BearerAuth(None, warn_on_disabled=False), raising=False)
        monkeypatch.setattr(api, "AUDIT_LOGGER", RequestAuditLogger(audit_path), raising=False)
        monkeypatch.setattr(
            api,
            "LIMITER",
            TokenBucketLimiter(rate_per_minute=60000, burst=1000),
            raising=False,
        )
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/status",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
                "scheme": "http",
                "query_string": b"",
            }
        )

        async def failing_handler(_request: Request) -> None:
            raise RuntimeError("handler boom")

        with pytest.raises(RuntimeError, match="handler boom"):
            asyncio.run(api.require_bearer_token(request, failing_handler))

        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        assert payload["path"] == "/status"
        assert payload["status"] == 500
        assert payload["outcome"] == "exception"


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestAPIPipeline:
    def test_status_endpoint_pipeline(self, tmp_path: Path) -> None:
        """Status endpoint exercises the full pipeline."""
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from api import app

        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with (
            patch("api.BASE", tmp_path),
            patch("api.STATE_DIR", state_dir),
            patch("api.GRAPH_DIR", graph_dir),
        ):
            client = TestClient(app)
            resp = client.get("/status")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)


class TestRecallCorrectnessEndpoint:
    """The HTTP write seam a verifier posts its correctness verdict to."""

    def test_records_verdict(
        self, client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from recall_ledger import RecallLedger

        led_path = tmp_path / "rl.jsonl"
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", str(led_path))
        RecallLedger(led_path).record("how to reuse vectors", ["s:n"], top_k=1)

        resp = client.post(
            "/recall/correctness",
            json={"query": "how to reuse vectors", "was_correct": False},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["was_correct"] is False
        assert body["event_id"]
        (rq,) = list(RecallLedger(led_path).queries())
        assert rq.was_correct is False
        assert rq.was_used is None  # correctness must not touch usage

    def test_unknown_query_returns_404(
        self, client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", str(tmp_path / "empty.jsonl"))
        resp = client.post(
            "/recall/correctness", json={"query": "never asked", "was_correct": True}
        )
        assert resp.status_code == 404
        assert "no prior recall" in resp.json()["detail"]

    def test_missing_was_correct_is_422(
        self, client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", str(tmp_path / "rl.jsonl"))
        resp = client.post("/recall/correctness", json={"query": "q"})
        assert resp.status_code == 422  # pydantic validation: was_correct required
