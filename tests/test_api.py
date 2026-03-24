# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for api.py

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from api import app

try:
    from fastapi.testclient import TestClient
    HAS_TEST_CLIENT = True
except ImportError:
    HAS_TEST_CLIENT = False

pytestmark = pytest.mark.skipif(not HAS_TEST_CLIENT, reason="fastapi/httpx not installed")


@pytest.fixture
def client():
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
        assert data["version"] == "0.2.0"

    def test_health_daemon_alive(self, client, tmp_path):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        state = {"timestamp": time.time(), "cycle": 10}
        (state_dir / "current_state.json").write_text(json.dumps(state), encoding="utf-8")

        with patch("api.STATE_DIR", state_dir):
            resp = client.get("/health")
        assert resp.json()["daemon"] == "alive"


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

        with patch("api.BASE", tmp_path), \
             patch("api.STATE_DIR", state_dir), \
             patch("api.GRAPH_DIR", graph_dir):
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
            "\n".join(json.dumps(e) for e in entities) + "\n", encoding="utf-8",
        )
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()

        with patch("api.BASE", tmp_path), \
             patch("api.STATE_DIR", state_dir), \
             patch("api.GRAPH_DIR", graph_dir):
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
        mock_ctx = type("Ctx", (), {
            "query": "test",
            "trace": "trace.md",
            "trace_score": 0.8,
            "trace_snippet": "Some snippet content here",
            "semantic_memories": [],
            "entities": ["stdp"],
            "related_entities": [],
            "before": [],
            "after": [],
            "cross_project": [],
            "novelty_score": 0.3,
            "elapsed_ms": 15.0,
            "to_llm_context": lambda self: "context string",
        })()

        with patch("api.recall_endpoint") as mock_endpoint:
            # Actually patch the import inside
            pass

        # Direct approach: patch the recall import
        from unittest.mock import MagicMock
        mock_recall = MagicMock(return_value=mock_ctx)
        with patch.dict("sys.modules", {}):
            with patch("memory_recall.recall", mock_recall, create=True):
                # Test the endpoint format
                pass

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
        state = {"timestamp": time.time(), "cycle": 5, "n_neurons": 200,
                 "vram_mb": 100, "live_retrieval_available": True}
        (state_dir / "current_state.json").write_text(json.dumps(state), encoding="utf-8")
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)
        with patch("api.BASE", tmp_path), \
             patch("api.STATE_DIR", state_dir), \
             patch("api.GRAPH_DIR", graph_dir):
            resp = client.get("/status")
        data = resp.json()
        assert "daemon" in data
        assert data["daemon"]["cycle"] == 5
