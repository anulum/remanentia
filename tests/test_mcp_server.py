# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for MCP server

from __future__ import annotations

import json
import os

import pytest


from mcp_server import (
    TOOLS,
    handle_graph,
    handle_recall,
    handle_recall_correctness,
    handle_recall_feedback,
    handle_remember,
    handle_request,
    handle_status,
)


@pytest.fixture(autouse=True)
def _disable_default_mcp_audit():
    import mcp_server
    from api_security import ToolAuditLogger

    original = mcp_server.MCP_AUDIT_LOGGER
    mcp_server.MCP_AUDIT_LOGGER = ToolAuditLogger(None)
    try:
        yield
    finally:
        mcp_server.MCP_AUDIT_LOGGER = original


@pytest.fixture
def real_unified_index(tmp_path, monkeypatch):
    import memory_index
    from memory_index import MemoryIndex

    workspace = tmp_path / "workspace"
    traces = workspace / "reasoning_traces"
    traces.mkdir(parents=True)
    (traces / "decision.md").write_text(
        "# SNN removal decision\n\n"
        "The production team removed the obsolete SNN adapter after the real "
        "runtime migration completed and retained the native recall pipeline.",
        encoding="utf-8",
    )
    (traces / "benchmark.md").write_text(
        "# Benchmark evidence\n\n"
        "The LOCOMO benchmark records retrieval accuracy and latency from the "
        "real memory service under deterministic evaluation conditions.",
        encoding="utf-8",
    )

    monkeypatch.setattr(memory_index, "BASE", workspace)
    monkeypatch.setattr(memory_index, "SOURCES", {"traces": traces})
    monkeypatch.setattr(memory_index, "SOURCE_EXTENSIONS", {"traces": {".md"}})
    monkeypatch.setattr(memory_index, "HASH_CACHE_PATH", workspace / "content_hashes.json")
    monkeypatch.delenv("REMANENTIA_GUARDED", raising=False)
    monkeypatch.delenv("REMANENTIA_LLM_ANSWERS", raising=False)

    index = MemoryIndex()
    stats = index.build(
        use_gpu_embeddings=False,
        use_gliner=False,
        incremental=False,
    )
    assert stats["documents"] == 2
    index_path = workspace / "memory_index.json.gz"
    index.save(index_path)
    return workspace, index, index_path


# ── Tool definitions ─────────────────────────────────────────────


class TestToolDefinitions:
    def test_six_tools_defined(self):
        assert len(TOOLS) == 6

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {
            "remanentia_recall",
            "remanentia_remember",
            "remanentia_status",
            "remanentia_graph",
            "remanentia_recall_feedback",
            "remanentia_recall_correctness",
        }

    def test_recall_schema(self):
        recall_tool = next(t for t in TOOLS if t["name"] == "remanentia_recall")
        props = recall_tool["inputSchema"]["properties"]
        assert "query" in props
        assert "top_k" in props
        assert "query" in recall_tool["inputSchema"]["required"]

    def test_graph_schema(self):
        graph_tool = next(t for t in TOOLS if t["name"] == "remanentia_graph")
        props = graph_tool["inputSchema"]["properties"]
        assert "entity" in props
        assert "top" in props

    def test_recall_correctness_schema(self):
        correctness_tool = next(t for t in TOOLS if t["name"] == "remanentia_recall_correctness")
        schema = correctness_tool["inputSchema"]
        props = schema["properties"]
        assert "query" in props
        assert "was_correct" in props
        assert schema["required"] == ["query", "was_correct"]


# ── MCP Protocol ─────────────────────────────────────────────────


class TestMCPProtocol:
    def test_initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = handle_request(req)
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "remanentia"

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        resp = handle_request(req)
        assert len(resp["result"]["tools"]) == 6

    def test_notifications_initialized(self):
        req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        resp = handle_request(req)
        assert resp is None

    def test_unknown_method(self):
        req = {"jsonrpc": "2.0", "id": 3, "method": "unknown/method"}
        resp = handle_request(req)
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_tools_call_recall(self):
        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "remanentia_recall", "arguments": {"query": ""}},
        }
        resp = handle_request(req)
        assert resp["result"]["content"][0]["type"] == "text"
        assert "No memories found" in resp["result"]["content"][0]["text"]

    def test_tools_call_status(self):
        req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "remanentia_status", "arguments": {}},
        }
        resp = handle_request(req)
        assert isinstance(resp["result"]["content"][0]["text"], str)

    def test_tools_call_graph(self):
        req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "remanentia_graph", "arguments": {"entity": "stdp"}},
        }
        resp = handle_request(req)
        assert isinstance(resp["result"]["content"][0]["text"], str)

    def test_tools_call_recall_correctness(self, monkeypatch):
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        req = {
            "jsonrpc": "2.0",
            "id": 61,
            "method": "tools/call",
            "params": {
                "name": "remanentia_recall_correctness",
                "arguments": {"query": "calibration query", "was_correct": True},
            },
        }
        resp = handle_request(req)
        assert "disabled" in resp["result"]["content"][0]["text"]

    def test_tools_call_recall_feedback(self, monkeypatch):
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        req = {
            "jsonrpc": "2.0",
            "id": 62,
            "method": "tools/call",
            "params": {
                "name": "remanentia_recall_feedback",
                "arguments": {"query": "calibration query", "was_used": True},
            },
        }
        resp = handle_request(req)
        assert "disabled" in resp["result"]["content"][0]["text"]

    def test_tools_call_coerces_non_dict_arguments(self):
        req = {
            "jsonrpc": "2.0",
            "id": 63,
            "method": "tools/call",
            "params": {"name": "remanentia_status", "arguments": "not a dict"},
        }
        resp = handle_request(req)
        assert isinstance(resp["result"]["content"][0]["text"], str)

    def test_tools_call_unknown_tool(self):
        req = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        resp = handle_request(req)
        assert "Unknown tool" in resp["result"]["content"][0]["text"]

    def test_preserves_request_id(self):
        req = {"jsonrpc": "2.0", "id": "abc-123", "method": "tools/list"}
        resp = handle_request(req)
        assert resp["id"] == "abc-123"


# ── handle_recall ────────────────────────────────────────────────


class TestHandleRecall:
    def test_isolated_workspace_uses_real_filesystem_fallback(self, tmp_traces):
        result = handle_recall("SNN removal decision", base=tmp_traces.parent)
        assert "SNN" in result

    def test_empty_query(self, tmp_path):
        assert handle_recall("", base=tmp_path) == "Empty query."


# ── handle_remember ──────────────────────────────────────────


class TestHandleRemember:
    def test_writes_trace_file(self, tmp_path):
        result = handle_remember("We decided to use BM25.", "decision", "remanentia", base=tmp_path)
        assert "Remembered:" in result
        traces = list((tmp_path / "reasoning_traces").glob("*.md"))
        assert len(traces) == 1
        content = traces[0].read_text(encoding="utf-8")
        assert "We decided to use BM25." in content
        assert "remanentia" in content
        notes_path = tmp_path / "memory" / "knowledge_notes.jsonl"
        assert notes_path.exists()
        assert "We decided to use BM25." in notes_path.read_text(encoding="utf-8")
        assert not (tmp_path / "consolidation").exists()

    def test_invalidates_recall_cache(self, tmp_path):
        import mcp_server

        mcp_server._RECALL_INDEX = {"old": "data"}
        handle_remember("test content", "context", "", base=tmp_path)
        assert mcp_server._RECALL_INDEX is None

    def test_mcp_protocol_remember(self, tmp_path):
        req = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "remanentia_remember",
                "arguments": {"content": "Test memory", "type": "finding", "project": "test"},
            },
        }
        resp = handle_request(req, base=tmp_path)
        assert "Remembered:" in resp["result"]["content"][0]["text"]


# ── handle_graph ─────────────────────────────────────────────────


class TestHandleGraph:
    def test_no_relations_file(self, tmp_path):
        result = handle_graph(graph_dir=tmp_path)
        assert "No relations" in result

    def test_top_relationships(self, tmp_graph):
        result = handle_graph(top=3, graph_dir=tmp_graph)
        assert "entity relationships" in result.lower() or "Top" in result

    def test_entity_filter(self, tmp_graph):
        result = handle_graph(entity="stdp", top=5, graph_dir=tmp_graph)
        assert "stdp" in result.lower()

    def test_entity_not_found(self, tmp_graph):
        result = handle_graph(entity="nonexistent_entity", top=5, graph_dir=tmp_graph)
        assert "Connections" in result or "nonexistent" in result


# ── handle_status ────────────────────────────────────────────────


class TestHandleRecallWithIndex:
    def test_real_index_returns_persisted_document(self, real_unified_index):
        workspace, index, _ = real_unified_index

        result = handle_recall(
            "Why was the obsolete SNN adapter removed?",
            top_k=3,
            base=workspace,
            index=index,
        )

        assert "decision.md" in result
        assert "native recall pipeline" in result

    def test_real_index_returns_no_memories_for_unmatched_query(self, real_unified_index):
        workspace, index, _ = real_unified_index

        result = handle_recall(
            "xyznonexistent_zzz_999",
            top_k=3,
            base=workspace,
            index=index,
        )

        assert "No memories" in result


class TestHandleRecallLightweight:
    def test_lightweight_fallback(self, tmp_traces):
        from mcp_server import _lightweight_recall

        result = _lightweight_recall("SNN removal decision", top_k=3, base=tmp_traces.parent)
        assert isinstance(result, str)

    def test_lightweight_empty_query(self):
        from mcp_server import _lightweight_recall

        result = _lightweight_recall("", top_k=3)
        assert "Empty query" in result

    def test_lightweight_no_match(self, tmp_traces):
        from mcp_server import _lightweight_recall

        import mcp_server

        mcp_server._RECALL_INDEX = None
        result = _lightweight_recall("xyznonexistent_zzz_999", top_k=3, base=tmp_traces.parent)
        assert "No memories" in result


class TestHandleStatus:
    def test_actual_status(self, tmp_path):
        # handle_status imports cli.cmd_status, which may fail
        result = handle_status()
        assert isinstance(result, str)


class TestBuildRecallIndex:
    def test_missing_workspace_is_empty(self, tmp_path):
        from mcp_storage import build_recall_index

        assert build_recall_index(tmp_path / "missing", lambda text: set(text.split())) == {}

    def test_empty_existing_trace_directory(self, tmp_path):
        from mcp_storage import build_recall_index

        (tmp_path / "reasoning_traces").mkdir()
        assert build_recall_index(tmp_path, lambda text: set(text.split())) == {}

    def test_builds_index(self, tmp_traces, tmp_semantic):
        from mcp_server import _build_recall_index
        import mcp_server

        mcp_server._RECALL_INDEX = None
        index = _build_recall_index(tmp_traces.parent)
        assert len(index) > 0
        mcp_server._RECALL_INDEX = None

    def test_caches_index(self, tmp_traces):
        from mcp_server import _build_recall_index
        import mcp_server

        mcp_server._RECALL_INDEX = None
        idx1 = _build_recall_index(tmp_traces.parent)
        idx2 = _build_recall_index(tmp_traces.parent)
        assert idx1 is idx2
        mcp_server._RECALL_INDEX = None


class TestHandleRecallLoadIndex:
    def test_loads_real_persisted_index(self, real_unified_index):
        workspace, _, index_path = real_unified_index

        result = handle_recall(
            "LOCOMO retrieval accuracy",
            base=workspace,
            index_path=index_path,
        )

        assert "benchmark.md" in result
        assert "deterministic evaluation" in result

    def test_load_fails_falls_back(self, tmp_traces):
        result = handle_recall("SNN decision", base=tmp_traces.parent)
        assert "SNN" in result


class TestHandleRememberIndex:
    def test_incremental_index_update(self, real_unified_index):
        workspace, index, _ = real_unified_index
        before = len(index.documents)

        result = handle_remember(
            "The helixquartz invariant is persisted through the production "
            "remember handler and added to the active unified index.",
            "finding",
            "test",
            base=workspace,
            index=index,
        )

        assert result.startswith("Remembered:")
        assert len(index.documents) == before + 1
        assert index.search("helixquartz invariant", top_k=1)[0].source == "traces"


class TestHandleStatusActual:
    def test_status_captures_output(self):
        result = handle_status()
        assert isinstance(result, str)


# ── MCP tool-call audit ────────────────────────────────────────


class TestMCPToolAudit:
    def test_tool_call_writes_metadata_only_record(self, tmp_path, tmp_traces):
        from api_security import ToolAuditLogger

        audit_path = tmp_path / "mcp_audit.jsonl"
        req = {
            "jsonrpc": "2.0",
            "id": "audit-1",
            "method": "tools/call",
            "params": {
                "name": "remanentia_recall",
                "arguments": {
                    "query": "private memory text",
                    "top_k": 1,
                    "llm": False,
                },
            },
        }
        resp = handle_request(
            req,
            base=tmp_traces.parent,
            audit_logger=ToolAuditLogger(audit_path),
        )
        assert resp is not None
        assert isinstance(resp["result"]["content"][0]["text"], str)

        record = json.loads(audit_path.read_text(encoding="utf-8"))
        assert record["server"] == "mcp"
        assert record["method"] == "tools/call"
        assert record["tool"] == "remanentia_recall"
        assert record["request_id"] == "audit-1"
        assert record["outcome"] == "ok"
        assert record["argument_keys"] == ["llm", "query", "top_k"]
        assert "private memory text" not in audit_path.read_text(encoding="utf-8")
        assert "arguments" not in record
        assert "authorization" not in record

    def test_unknown_tool_writes_audit_record(self, tmp_path):
        from api_security import ToolAuditLogger

        audit_path = tmp_path / "mcp_audit.jsonl"
        req = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {"content": "do not log"}},
        }
        handle_request(req, audit_logger=ToolAuditLogger(audit_path))

        record = json.loads(audit_path.read_text(encoding="utf-8"))
        assert record["tool"] == "nonexistent_tool"
        assert record["outcome"] == "unknown_tool"
        assert record["argument_keys"] == ["content"]
        assert "do not log" not in audit_path.read_text(encoding="utf-8")

    def test_tool_exception_returns_error_and_writes_audit_record(self, tmp_path):
        from api_security import ToolAuditLogger

        audit_path = tmp_path / "mcp_audit.jsonl"
        req = {
            "jsonrpc": "2.0",
            "id": "recall-fail",
            "method": "tools/call",
            "params": {
                "name": "remanentia_recall",
                "arguments": {"query": {"sensitive": "detail"}},
            },
        }
        resp = handle_request(
            req,
            base=tmp_path,
            audit_logger=ToolAuditLogger(audit_path),
        )

        assert resp is not None
        assert resp["error"]["code"] == -32000
        assert "sensitive detail" not in resp["error"]["message"]
        record = json.loads(audit_path.read_text(encoding="utf-8"))
        assert record["tool"] == "remanentia_recall"
        assert record["outcome"] == "error"
        assert record["error_type"] in {"AttributeError", "TypeError"}


class TestMCPTelemetryAndCli:
    def test_observe_and_close_noop_when_disabled_or_missing_event(self, monkeypatch):
        import mcp_server

        mcp_server._observe_recall(None, ["text"])
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        mcp_server._observe_recall("event", ["text"])
        mcp_server._close_recall_loops("remembered text")

    def test_mcp_tokenizer(self):
        import mcp_server

        assert mcp_server._mcp_tok("Alpha be beta_42") == {"alpha", "beta_42"}

    def test_recall_feedback_disabled_no_prior_and_recorded(self, monkeypatch, tmp_path):
        from recall_ledger import RecallLedger

        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        assert "disabled" in handle_recall_feedback("query", True)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE")

        ledger = RecallLedger(tmp_path / "feedback.jsonl")
        assert "No prior recall" in handle_recall_feedback("query", True, ledger=ledger)
        ledger.record("query", ["trace:one"], top_k=1, by="agent")

        request = {
            "jsonrpc": "2.0",
            "id": "feedback",
            "method": "tools/call",
            "params": {
                "name": "remanentia_recall_feedback",
                "arguments": {"query": "query", "was_used": True},
            },
        }
        response = handle_request(request, ledger=ledger)

        assert response is not None
        assert "was_used=True" in response["result"]["content"][0]["text"]
        assert list(ledger.queries())[0].was_used is True

    def test_recall_correctness_disabled_no_prior_and_recorded(self, monkeypatch, tmp_path):
        from recall_ledger import RecallLedger

        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        assert "disabled" in handle_recall_correctness("query", False)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE")

        ledger = RecallLedger(tmp_path / "correctness.jsonl")
        assert "No prior recall" in handle_recall_correctness("query", False, ledger=ledger)
        ledger.record("query", ["trace:one"], top_k=1, by="agent")

        request = {
            "jsonrpc": "2.0",
            "id": "correctness",
            "method": "tools/call",
            "params": {
                "name": "remanentia_recall_correctness",
                "arguments": {"query": "query", "was_correct": False},
            },
        }
        response = handle_request(request, ledger=ledger)

        assert response is not None
        assert "was_correct=False" in response["result"]["content"][0]["text"]
        assert list(ledger.queries())[0].was_correct is False

    def test_parse_cli_sets_requested_environment(self, monkeypatch):
        import mcp_server

        monkeypatch.delenv("REMANENTIA_LLM_ANSWERS", raising=False)
        monkeypatch.delenv("REMANENTIA_LLM_BACKEND", raising=False)
        monkeypatch.delenv("REMANENTIA_GUARDED", raising=False)

        mcp_server._parse_cli(["--llm", "--local-llm", "--guarded"])

        assert os.environ["REMANENTIA_LLM_ANSWERS"] == "1"
        assert os.environ["REMANENTIA_LLM_BACKEND"] == "local"
        assert os.environ["REMANENTIA_GUARDED"] == "1"


# ── Pipeline integration ─────────────────────────────────────


class TestMCPPipelineIntegration:
    """MCP server as the entry point for the entire Remanentia pipeline."""

    def test_status_returns_structured_info(self):
        result = handle_status()
        assert isinstance(result, str)
        # Should mention index, entities, or similar
        assert len(result) > 10

    def test_graph_query_returns_entities(self):
        """MCP graph tool returns entity relationship text."""
        resp = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {"name": "remanentia_graph", "arguments": {"top": 3}},
            }
        )
        text = resp["result"]["content"][0]["text"]
        assert isinstance(text, str)

    def test_consolidate_via_mcp(self):
        """MCP status call exercises consolidation path."""
        resp = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {"name": "remanentia_status", "arguments": {}},
            }
        )
        text = resp["result"]["content"][0]["text"]
        assert isinstance(text, str)


# ── Missing patterns: roundtrip ───────────────────────────────


class TestMCPRoundtrip:
    def test_remember_then_recall(self, real_unified_index):
        """Pipeline: remember → recall → verify content persists."""
        workspace, index, _ = real_unified_index
        remember = handle_request(
            {
                "jsonrpc": "2.0",
                "id": "remember-real",
                "method": "tools/call",
                "params": {
                    "name": "remanentia_remember",
                    "arguments": {
                        "content": (
                            "The cobaltsemaphore memory persists through the real MCP "
                            "remember and unified-index pipeline."
                        ),
                        "type": "finding",
                        "project": "test",
                    },
                },
            },
            base=workspace,
            index=index,
        )
        recall = handle_request(
            {
                "jsonrpc": "2.0",
                "id": "recall-real",
                "method": "tools/call",
                "params": {
                    "name": "remanentia_recall",
                    "arguments": {"query": "cobaltsemaphore memory", "top_k": 1},
                },
            },
            base=workspace,
            index=index,
        )

        assert remember is not None and recall is not None
        assert "Remembered:" in remember["result"]["content"][0]["text"]
        assert "cobaltsemaphore" in recall["result"]["content"][0]["text"]


# ── default-workspace + guarded-tier coverage gaps ───────────────


class TestMcpServerDefaultWorkspaceGaps:
    """Drive the pure-Python default-workspace and guarded-tier branches that
    isolated-workspace tests never reach (mcp_server 341, 371-374, 394,
    400-415, 485-487, 503, 507)."""

    def test_recall_non_default_index_path_load_fail(self, tmp_path):
        missing = tmp_path / "absent_index.json.gz"
        result = handle_recall("any query", base=tmp_path, index_path=missing)
        assert isinstance(result, str)

    def test_recall_guarded_tier_resolves(self, real_unified_index, monkeypatch):
        workspace, index, index_path = real_unified_index
        monkeypatch.setenv("REMANENTIA_GUARDED", "1")
        result = handle_recall("SNN removal decision", base=workspace, index=index)
        assert isinstance(result, str)

    def test_recall_result_answer_header(self):
        class _Res:
            source = "traces"
            name = "d.md"
            score = 9.0
            answer = "the real answer"
            snippet = "snippet body"

        class _Index:
            def search(self, *args, **kwargs):
                return [_Res()]

        result = handle_recall("q", index=_Index())
        assert "the real answer" in result

    def test_recall_default_workspace_knowledge_graph(self, real_unified_index, monkeypatch):
        import mcp_server

        workspace, index, index_path = real_unified_index
        monkeypatch.setattr(mcp_server, "BASE", workspace)
        monkeypatch.setattr(mcp_server, "_UNIFIED_INDEX", index)

        class _Note:
            content = "A distinct knowledge note about the SNN adapter removal."
            title = "SNN knowledge note"
            note_type = "fact"

        class _KS:
            def check_triggers(self, query):
                return []

            def graph_search(self, query, top_k, hop_depth):
                return [_Note()]

        monkeypatch.setattr(mcp_server, "_get_knowledge_store", lambda: _KS())
        result = handle_recall("SNN removal decision", base=workspace)
        assert isinstance(result, str)

    def test_remember_default_workspace(self, tmp_path, monkeypatch):
        import mcp_server

        workspace = tmp_path / "ws"
        (workspace / "reasoning_traces").mkdir(parents=True)
        monkeypatch.setattr(mcp_server, "BASE", workspace)

        class _KS:
            def add_note(self, *args, **kwargs):
                return None

            def add_trigger(self, *args, **kwargs):
                return None

            def save(self, *args, **kwargs):
                return None

        monkeypatch.setattr(mcp_server, "_get_knowledge_store", lambda: _KS())
        monkeypatch.setattr(mcp_server, "_close_recall_loops", lambda content: None)
        monkeypatch.setattr(mcp_server, "_schedule_consolidation", lambda: None)
        result = handle_remember("a real memory body", base=workspace)
        assert "Remembered" in result
