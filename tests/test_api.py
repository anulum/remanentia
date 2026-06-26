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
from unittest.mock import patch

import numpy as np
import pytest

import api
from api_security import BearerAuth, RequestAuditLogger, TokenBucketLimiter
from api import app
from vector_index import VectorSearchResult

try:
    from fastapi.testclient import TestClient

    HAS_TEST_CLIENT = True
except ImportError:
    HAS_TEST_CLIENT = False

pytestmark = pytest.mark.skipif(not HAS_TEST_CLIENT, reason="fastapi/httpx not installed")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api,
        "LIMITER",
        TokenBucketLimiter(rate_per_minute=60000, burst=1000),
        raising=False,
    )
    monkeypatch.setattr(api, "AUDIT_LOGGER", RequestAuditLogger(None), raising=False)
    return TestClient(app)


# ── Health ───────────────────────────────────────────────────────


class TestHealth:
    def test_health_no_daemon(self, client, tmp_path):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        with patch("api.STATE_DIR", state_dir):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["daemon"] == "stale"
        assert data["vector_worker"] == "missing"
        assert data["version"] == "0.2.0"

    def test_health_daemon_alive(self, client, tmp_path):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        state = {"timestamp": time.time(), "cycle": 10}
        (state_dir / "current_state.json").write_text(json.dumps(state), encoding="utf-8")

        with patch("api.STATE_DIR", state_dir):
            resp = client.get("/health")
        assert resp.json()["daemon"] == "alive"

    def test_health_vector_worker_alive(self, client, tmp_path):
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
        self, client, monkeypatch, tmp_path
    ):
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

    def test_private_endpoint_accepts_matching_bearer_token(self, client, monkeypatch, tmp_path):
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

    def test_post_body_limit_rejects_before_handler(self, client, monkeypatch):
        monkeypatch.setattr(api, "BODY_LIMIT", 8, raising=False)
        resp = client.post("/recall", json={"query": "body is too large"})

        assert resp.status_code == 413
        assert resp.json()["detail"] == "request body too large"

    def test_private_endpoint_rate_limit_is_enforced(self, client, monkeypatch, tmp_path):
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

    def test_default_cors_origin_allows_local_development(self, monkeypatch):
        monkeypatch.delenv("REMANENTIA_CORS_ORIGINS", raising=False)

        assert api._cors_origins_from_env() == ["*"]

    def test_cors_origins_can_be_scoped_by_environment(self, monkeypatch):
        monkeypatch.setenv(
            "REMANENTIA_CORS_ORIGINS",
            "https://remanentia.com, https://www.remanentia.com",
        )

        assert api._cors_origins_from_env() == [
            "https://remanentia.com",
            "https://www.remanentia.com",
        ]

    def test_retry_after_ceilings_match_rate_limit(self):
        assert TokenBucketLimiter(rate_per_minute=60).retry_after_seconds() == "1"
        assert TokenBucketLimiter(rate_per_minute=30).retry_after_seconds() == "2"
        assert TokenBucketLimiter(rate_per_minute=7).retry_after_seconds() == "9"

    def test_private_endpoint_writes_audit_record(self, client, monkeypatch, tmp_path):
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

    def test_public_health_does_not_write_audit_record(self, client, monkeypatch, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(api, "AUDIT_LOGGER", RequestAuditLogger(audit_path), raising=False)

        resp = client.get("/health")

        assert resp.status_code == 200
        assert not audit_path.exists()


# ── Status ───────────────────────────────────────────────────────


class TestStatus:
    def test_status_empty(self, client, tmp_path):
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

    def test_status_with_data(self, client, tmp_path):
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
    def test_no_entities(self, client, tmp_path):
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with patch("api.GRAPH_DIR", graph_dir):
            resp = client.get("/entities")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_entities(self, client, tmp_graph):
        with patch("api.GRAPH_DIR", tmp_graph):
            resp = client.get("/entities")
        data = resp.json()
        assert len(data) == 4
        ids = {e["id"] for e in data}
        assert "stdp" in ids


# ── Graph ────────────────────────────────────────────────────────


class TestGraph:
    def test_no_relations(self, client, tmp_path):
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        with patch("api.GRAPH_DIR", graph_dir):
            resp = client.get("/graph")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_top_relations(self, client, tmp_graph):
        with patch("api.GRAPH_DIR", tmp_graph):
            resp = client.get("/graph?top=2")
        data = resp.json()
        assert len(data) == 2
        # Sorted by weight descending
        assert data[0]["weight"] >= data[1]["weight"]

    def test_entity_detail(self, client, tmp_graph):
        with patch("api.GRAPH_DIR", tmp_graph):
            resp = client.get("/graph/entity/stdp")
        data = resp.json()
        assert data["entity"]["id"] == "stdp"
        assert len(data["connections"]) > 0

    def test_entity_not_found(self, client, tmp_graph):
        with patch("api.GRAPH_DIR", tmp_graph):
            resp = client.get("/graph/entity/nonexistent")
        data = resp.json()
        assert "error" in data


# ── Recall ───────────────────────────────────────────────────────


class TestRecall:
    def test_recall_basic(self, client):
        mock_ctx = type(
            "Ctx",
            (),
            {
                "query": "test",
                "trace": "trace.md",
                "trace_score": 0.8,
                "trace_snippet": "Some snippet content here",
                "semantic_memories": [
                    {"path": "memory/semantic/test.md", "key_point": "Stored fact"}
                ],
                "entities": ["stdp"],
                "related_entities": [],
                "before": [],
                "after": [],
                "cross_project": [],
                "novelty_score": 0.3,
                "elapsed_ms": 15.0,
                "to_llm_context": lambda self: "context string",
            },
        )()

        with patch("memory_recall.recall", return_value=mock_ctx) as mock_recall:
            resp = client.post(
                "/recall",
                json={"query": "test", "top_k": 2, "include_content": True},
            )

        mock_recall.assert_called_once_with("test", top_k=2, include_content=True)
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert data["trace"] == "trace.md"
        assert data["score"] == pytest.approx(0.8)
        assert data["snippet"] == "Some snippet content here"
        assert data["semantic_memories"] == [
            {"path": "memory/semantic/test.md", "key_point": "Stored fact"}
        ]
        assert data["entities"] == ["stdp"]
        assert data["elapsed_ms"] == 15.0

    def test_recall_summary_format(self, client, tmp_traces, tmp_path):
        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_ctx.query = "test"
        mock_ctx.trace = "trace.md"
        mock_ctx.trace_score = 0.8
        mock_ctx.trace_snippet = "Some snippet"
        mock_ctx.semantic_memories = []
        mock_ctx.entities = ["stdp"]
        mock_ctx.related_entities = []
        mock_ctx.before = []
        mock_ctx.after = []
        mock_ctx.cross_project = []
        mock_ctx.novelty_score = 0.3
        mock_ctx.elapsed_ms = 15.0
        with patch("memory_recall.recall", return_value=mock_ctx):
            resp = client.post("/recall", json={"query": "test", "format": "summary"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert data["trace"] == "trace.md"

    def test_recall_context_format(self, client):
        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_ctx.to_llm_context.return_value = "LLM context here"
        mock_ctx.elapsed_ms = 10.0
        with patch("memory_recall.recall", return_value=mock_ctx):
            resp = client.post("/recall", json={"query": "test", "format": "context"})
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data
        assert data["context"] == "LLM context here"

    def test_recall_serializes_numpy_scalars(self, client):
        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_ctx.query = "test"
        mock_ctx.trace = "trace.md"
        mock_ctx.trace_score = np.float32(0.8)
        mock_ctx.trace_snippet = "Some snippet"
        mock_ctx.semantic_memories = []
        mock_ctx.entities = ["stdp"]
        mock_ctx.related_entities = [{"entity": "graph", "weight": np.float32(0.5)}]
        mock_ctx.before = []
        mock_ctx.after = []
        mock_ctx.cross_project = []
        mock_ctx.novelty_score = np.float32(0.3)
        mock_ctx.elapsed_ms = np.float32(15.0)
        with patch("memory_recall.recall", return_value=mock_ctx):
            resp = client.post("/recall", json={"query": "test", "format": "summary"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] == pytest.approx(0.8)
        assert data["related"][0]["weight"] == pytest.approx(0.5)
        assert data["novelty"] == pytest.approx(0.3)


class TestPublicVectorSearch:
    def test_public_vector_search_requires_query(self, client):
        resp = client.post("/vector/search/public", json={"query": ""})

        assert resp.status_code == 400

    def test_public_vector_search_top_k_zero_short_circuits(self, client):
        resp = client.post("/vector/search/public", json={"query": "beta", "top_k": 0})

        assert resp.status_code == 200
        assert resp.json() == {"query": "beta", "results": []}

    def test_public_vector_search_no_allowlist_returns_no_results(self, client):
        raw_result = VectorSearchResult(
            chunk_id="one",
            text="beta public paragraph",
            source="paper",
            score=0.8,
            metadata={"path": "paper/remanentia.md"},
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("vector_index.HttpEmbeddingClient.from_env", return_value=object()),
            patch("vector_pipeline.search_memory_vector_index", return_value=[raw_result]),
        ):
            resp = client.post("/vector/search/public", json={"query": "beta", "top_k": 1})

        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_public_vector_search_uses_server_policy_and_redaction(self, client, tmp_path):
        redaction_file = tmp_path / "terms.txt"
        redaction_file.write_text("private-token\n", encoding="utf-8")
        raw_result = VectorSearchResult(
            chunk_id="one",
            text="beta private-token public paragraph",
            source="paper",
            score=0.8,
            metadata={
                "document": "private-token-paper.md",
                "document_type": "paper",
                "paragraph_idx": 0,
                "path": "paper/private-token-paper.md",
            },
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "REMANENTIA_PUBLIC_VECTOR_SOURCES": "paper",
                    "REMANENTIA_PUBLIC_VECTOR_PATH_PREFIXES": "paper",
                    "REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE": str(redaction_file),
                },
                clear=True,
            ),
            patch("vector_index.HttpEmbeddingClient.from_env", return_value=object()),
            patch("vector_pipeline.search_memory_vector_index", return_value=[raw_result]),
        ):
            resp = client.post("/vector/search/public", json={"query": "beta", "top_k": 1})

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["results"][0]["text"] == "beta [redacted] public paragraph"
        assert payload["results"][0]["metadata"]["document"] == "[redacted]-paper.md"
        assert payload["results"][0]["redactions"] == 3

    def test_public_vector_search_backend_error_returns_503(self, client):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "vector_index.HttpEmbeddingClient.from_env", side_effect=ValueError("no endpoint")
            ),
        ):
            resp = client.post("/vector/search/public", json={"query": "beta", "top_k": 1})

        assert resp.status_code == 503
        assert resp.json()["detail"] == "no endpoint"

    def test_missing_redaction_file_is_rejected(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing_terms.txt"
        monkeypatch.setenv("REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE", str(missing))

        with pytest.raises(ValueError, match="REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE"):
            api._read_terms_from_env("REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE")

    def test_consolidate_endpoint(self, client):
        mock_result = {"status": "ok", "traces_processed": 5}
        with patch("consolidation_engine.consolidate", return_value=mock_result):
            resp = client.post("/consolidate", json={"force": False})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_status_with_daemon_state(self, client, tmp_path):
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

    def test_health_reports_unreadable_legacy_daemon_state(self, client, tmp_path):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        (state_dir / "current_state.json").write_text("{", encoding="utf-8")

        with patch("api.STATE_DIR", state_dir):
            resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["legacy_daemon"] == "unreadable"

    def test_status_with_vector_worker_state(self, client, tmp_path):
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

    def test_status_reports_unreadable_vector_worker_state(self, client, tmp_path):
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
    def test_json_safe_keeps_value_when_item_conversion_rejects(self):
        class RejectingScalar:
            def item(self):
                raise ValueError("not scalar")

        value = RejectingScalar()

        assert api._json_safe(value) is value

    def test_middleware_audits_handler_exception(self, monkeypatch, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(api, "AUTH", BearerAuth(None, warn_on_disabled=False), raising=False)
        monkeypatch.setattr(api, "AUDIT_LOGGER", RequestAuditLogger(audit_path), raising=False)
        monkeypatch.setattr(
            api,
            "LIMITER",
            TokenBucketLimiter(rate_per_minute=60000, burst=1000),
            raising=False,
        )
        request = api.Request(
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

        async def failing_handler(_request):
            raise RuntimeError("handler boom")

        with pytest.raises(RuntimeError, match="handler boom"):
            asyncio.run(api.require_bearer_token(request, failing_handler))

        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        assert payload["path"] == "/status"
        assert payload["status"] == 500
        assert payload["outcome"] == "exception"


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestAPIPipeline:
    def test_status_endpoint_pipeline(self, tmp_path):
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

    def test_records_verdict(self, client, tmp_path, monkeypatch):
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

    def test_unknown_query_returns_404(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", str(tmp_path / "empty.jsonl"))
        resp = client.post(
            "/recall/correctness", json={"query": "never asked", "was_correct": True}
        )
        assert resp.status_code == 404
        assert "no prior recall" in resp.json()["detail"]

    def test_missing_was_correct_is_422(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", str(tmp_path / "rl.jsonl"))
        resp = client.post("/recall/correctness", json={"query": "q"})
        assert resp.status_code == 422  # pydantic validation: was_correct required
