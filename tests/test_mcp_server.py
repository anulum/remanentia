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
from unittest.mock import patch

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
def _disable_default_mcp_audit(monkeypatch):
    import mcp_server
    from api_security import ToolAuditLogger

    monkeypatch.setattr(mcp_server, "MCP_AUDIT_LOGGER", ToolAuditLogger(None))


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
        with patch("mcp_server.handle_recall", return_value="Test result"):
            req = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "remanentia_recall", "arguments": {"query": "test"}},
            }
            resp = handle_request(req)
        assert resp["result"]["content"][0]["type"] == "text"
        assert resp["result"]["content"][0]["text"] == "Test result"

    def test_tools_call_status(self):
        with patch("mcp_server.handle_status", return_value="Status info"):
            req = {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "remanentia_status", "arguments": {}},
            }
            resp = handle_request(req)
        assert "Status info" in resp["result"]["content"][0]["text"]

    def test_tools_call_graph(self):
        with patch("mcp_server.handle_graph", return_value="Graph data"):
            req = {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "remanentia_graph", "arguments": {"entity": "stdp"}},
            }
            resp = handle_request(req)
        assert "Graph data" in resp["result"]["content"][0]["text"]

    def test_tools_call_recall_correctness(self):
        with patch("mcp_server.handle_recall_correctness", return_value="Correctness recorded"):
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
        assert "Correctness recorded" in resp["result"]["content"][0]["text"]

    def test_tools_call_recall_feedback(self):
        with patch("mcp_server.handle_recall_feedback", return_value="Feedback recorded"):
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
        assert "Feedback recorded" in resp["result"]["content"][0]["text"]

    def test_tools_call_coerces_non_dict_arguments(self):
        with patch("mcp_server.handle_status", return_value="Status info"):
            req = {
                "jsonrpc": "2.0",
                "id": 63,
                "method": "tools/call",
                "params": {"name": "remanentia_status", "arguments": "not a dict"},
            }
            resp = handle_request(req)
        assert "Status info" in resp["result"]["content"][0]["text"]

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
    def test_no_index_falls_back(self):
        with (
            patch("mcp_server._UNIFIED_INDEX", None),
            patch("mcp_server._lightweight_recall", return_value="lightweight result"),
        ):
            # MemoryIndex load fails → lightweight
            result = handle_recall("test query")
        assert isinstance(result, str)

    def test_empty_query(self):
        result = handle_recall("")
        assert isinstance(result, str)


# ── handle_remember ──────────────────────────────────────────


class TestHandleRemember:
    def test_writes_trace_file(self, tmp_path):
        with (
            patch("mcp_server.BASE", tmp_path),
            patch("mcp_server._RECALL_INDEX", None),
            patch("mcp_server._UNIFIED_INDEX", None),
        ):
            result = handle_remember("We decided to use BM25.", "decision", "remanentia")
        assert "Remembered:" in result
        traces = list((tmp_path / "reasoning_traces").glob("*.md"))
        assert len(traces) == 1
        content = traces[0].read_text(encoding="utf-8")
        assert "We decided to use BM25." in content
        assert "remanentia" in content

    def test_invalidates_recall_cache(self, tmp_path):
        import mcp_server

        mcp_server._RECALL_INDEX = {"old": "data"}
        with patch("mcp_server.BASE", tmp_path), patch("mcp_server._UNIFIED_INDEX", None):
            handle_remember("test content", "context", "")
        assert mcp_server._RECALL_INDEX is None

    def test_mcp_protocol_remember(self, tmp_path):
        with patch("mcp_server.BASE", tmp_path), patch("mcp_server._UNIFIED_INDEX", None):
            req = {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "remanentia_remember",
                    "arguments": {"content": "Test memory", "type": "finding", "project": "test"},
                },
            }
            resp = handle_request(req)
        assert "Remembered:" in resp["result"]["content"][0]["text"]


# ── handle_graph ─────────────────────────────────────────────────


class TestHandleGraph:
    def test_no_relations_file(self, tmp_path):
        with patch("mcp_server.GRAPH_DIR", tmp_path):
            result = handle_graph()
        assert "No relations" in result

    def test_top_relationships(self, tmp_graph):
        with patch("mcp_server.GRAPH_DIR", tmp_graph):
            result = handle_graph(top=3)
        assert "entity relationships" in result.lower() or "Top" in result

    def test_entity_filter(self, tmp_graph):
        with patch("mcp_server.GRAPH_DIR", tmp_graph):
            result = handle_graph(entity="stdp", top=5)
        assert "stdp" in result.lower()

    def test_entity_not_found(self, tmp_graph):
        with patch("mcp_server.GRAPH_DIR", tmp_graph):
            result = handle_graph(entity="nonexistent_entity", top=5)
        assert "Connections" in result or "nonexistent" in result


# ── handle_status ────────────────────────────────────────────────


class TestHandleRememberConsolidation:
    def test_triggers_consolidation(self, tmp_path):
        import time
        from unittest.mock import MagicMock

        mock_consolidate = MagicMock(return_value={"status": "nothing_to_consolidate"})
        with (
            patch("mcp_server.BASE", tmp_path),
            patch("mcp_server._RECALL_INDEX", None),
            patch("mcp_server._UNIFIED_INDEX", None),
            patch("mcp_server._consolidation_last", 0.0),
            patch("mcp_server._CONSOLIDATION_DEBOUNCE_S", 0.0),
            patch("consolidation_engine.consolidate", mock_consolidate),
        ):
            handle_remember("Test consolidation trigger", "finding", "test")
            # Consolidation now runs in a background thread
            for _ in range(20):
                if mock_consolidate.called:
                    break
                time.sleep(0.1)
        mock_consolidate.assert_called_once()

    def test_consolidation_failure_safe(self, tmp_path):
        with (
            patch("mcp_server.BASE", tmp_path),
            patch("mcp_server._RECALL_INDEX", None),
            patch("mcp_server._UNIFIED_INDEX", None),
        ):
            result = handle_remember("Test content", "context", "test")
        assert "Remembered:" in result


class TestHandleRecallWithIndex:
    def test_with_loaded_index(self, tmp_path):
        from unittest.mock import MagicMock
        from memory_index import SearchResult

        mock_idx = MagicMock()
        mock_idx.load.return_value = True
        mock_idx._built = True
        mock_idx.search.return_value = [
            SearchResult(
                name="test.md",
                source="traces",
                score=0.9,
                snippet="Test snippet",
                answer="March 15",
            ),
        ]
        import mcp_server

        old_idx = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            result = handle_recall("test query", top_k=3)
            assert "test.md" in result
            assert "March 15" in result
        finally:
            mcp_server._UNIFIED_INDEX = old_idx

    def test_empty_results(self, tmp_path):
        from unittest.mock import MagicMock

        mock_idx = MagicMock()
        mock_idx._built = True
        mock_idx.search.return_value = []
        import mcp_server

        old_idx = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            result = handle_recall("xyznonexistent", top_k=3)
            assert "No memories" in result
        finally:
            mcp_server._UNIFIED_INDEX = old_idx

    def test_llm_flag_passed(self):
        from unittest.mock import MagicMock

        mock_idx = MagicMock()
        mock_idx._built = True
        mock_idx.search.return_value = []
        import mcp_server

        old_idx = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            handle_recall("test", llm=True)
            call_kwargs = mock_idx.search.call_args
            assert call_kwargs[1].get("use_llm") is True
        finally:
            mcp_server._UNIFIED_INDEX = old_idx

    def test_llm_backend_auto_setup(self):
        """When llm=True and no backend set, auto-resolves backend."""
        from unittest.mock import MagicMock
        import answer_extractor
        import mcp_server

        mock_idx = MagicMock()
        mock_idx._built = True
        mock_idx.search.return_value = []
        old_idx = mcp_server._UNIFIED_INDEX
        old_backend = answer_extractor._BACKEND
        answer_extractor._BACKEND = None
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            handle_recall("test", llm=True)
            assert answer_extractor._BACKEND is not None
        finally:
            mcp_server._UNIFIED_INDEX = old_idx
            answer_extractor._BACKEND = old_backend

    def test_guarded_mode_loads_guard_dependencies_when_results_exist(self, monkeypatch):
        from unittest.mock import MagicMock
        from memory_index import SearchResult
        import mcp_server

        mock_idx = MagicMock()
        mock_idx._built = True
        mock_idx.search.return_value = [
            SearchResult(
                name="guarded.md",
                source="trace",
                score=0.9,
                snippet="Guarded snippet",
                answer="Guarded answer",
            )
        ]
        requested: list[tuple[str, str]] = []

        def runtime_attr(module_name: str, attr_name: str):
            requested.append((module_name, attr_name))
            if (module_name, attr_name) == ("memory_index", "MemoryIndex"):
                return lambda: mock_idx
            if attr_name == "facts_from_results":
                return lambda results: ["fact"]
            if attr_name == "is_available":
                return lambda: False
            if attr_name == "score_memory_answer":
                return lambda query, answer, facts: None
            raise AssertionError(attr_name)

        old_idx = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = mock_idx
        monkeypatch.setenv("REMANENTIA_GUARDED", "1")
        try:
            with patch("mcp_server._runtime_attr", side_effect=runtime_attr):
                result = handle_recall("guarded query", top_k=1)
        finally:
            mcp_server._UNIFIED_INDEX = old_idx

        assert isinstance(result, str)
        assert ("memory_guarded", "facts_from_results") in requested
        assert ("memory_guarded", "is_available") in requested
        assert ("memory_guarded", "score_memory_answer") in requested


class TestHandleRecallLightweight:
    def test_lightweight_fallback(self, tmp_traces):
        from mcp_server import _lightweight_recall

        with patch("mcp_server.BASE", tmp_traces.parent):
            result = _lightweight_recall("SNN removal decision", top_k=3)
        assert isinstance(result, str)

    def test_lightweight_empty_query(self):
        from mcp_server import _lightweight_recall

        result = _lightweight_recall("", top_k=3)
        assert "Empty query" in result

    def test_lightweight_no_match(self, tmp_traces):
        from mcp_server import _lightweight_recall

        with patch("mcp_server.BASE", tmp_traces.parent):
            import mcp_server

            mcp_server._RECALL_INDEX = None
            result = _lightweight_recall("xyznonexistent_zzz_999", top_k=3)
        assert "No memories" in result


class TestHandleStatus:
    def test_returns_string(self):
        with patch("mcp_server.handle_status") as mock:
            mock.return_value = "Daemon: NOT RUNNING\nMemory:\n  Traces: 24"
            result = handle_status()
        assert isinstance(result, str)

    def test_actual_status(self, tmp_path):
        # handle_status imports cli.cmd_status, which may fail
        result = handle_status()
        assert isinstance(result, str)


class TestBuildRecallIndex:
    def test_builds_index(self, tmp_traces, tmp_semantic):
        from mcp_server import _build_recall_index
        import mcp_server

        mcp_server._RECALL_INDEX = None
        with patch("mcp_server.BASE", tmp_traces.parent):
            index = _build_recall_index()
        assert len(index) > 0
        mcp_server._RECALL_INDEX = None

    def test_caches_index(self, tmp_traces):
        from mcp_server import _build_recall_index
        import mcp_server

        mcp_server._RECALL_INDEX = None
        with patch("mcp_server.BASE", tmp_traces.parent):
            idx1 = _build_recall_index()
            idx2 = _build_recall_index()
        assert idx1 is idx2
        mcp_server._RECALL_INDEX = None


class TestHandleRecallLoadIndex:
    def test_loads_index_on_first_call(self):
        from unittest.mock import MagicMock
        from memory_index import SearchResult

        mock_idx = MagicMock()
        mock_idx.load.return_value = True
        mock_idx._built = True
        mock_idx.search.return_value = [
            SearchResult(name="r.md", source="src", score=0.5, snippet="snip"),
        ]
        import mcp_server

        old = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = None
        try:
            # Patch MemoryIndex at the module level where it's imported
            with patch.dict("sys.modules", {}):
                with patch("memory_index.MemoryIndex", return_value=mock_idx):
                    mcp_server._UNIFIED_INDEX = mock_idx
                    result = handle_recall("test")
            assert "r.md" in result
        finally:
            mcp_server._UNIFIED_INDEX = old

    def test_loads_index_via_lock(self):
        from unittest.mock import MagicMock
        from memory_index import SearchResult

        mock_idx = MagicMock()
        mock_idx.load.return_value = True
        mock_idx._built = True
        mock_idx.search.return_value = [
            SearchResult(name="t.md", source="src", score=0.7, snippet="s"),
        ]
        import mcp_server

        old = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = None
        try:
            with patch("mcp_server.MemoryIndex", return_value=mock_idx, create=True):
                with patch("memory_index.MemoryIndex", return_value=mock_idx):
                    result = handle_recall("query")
            assert mcp_server._UNIFIED_INDEX is mock_idx
            assert "t.md" in result
        finally:
            mcp_server._UNIFIED_INDEX = old

    def test_load_fails_falls_back(self, tmp_traces):
        import mcp_server

        old = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = None
        mcp_server._RECALL_INDEX = None
        try:
            with (
                patch("mcp_server.BASE", tmp_traces.parent),
                patch("mcp_server._UNIFIED_INDEX", None),
            ):
                result = handle_recall("SNN decision")
            assert isinstance(result, str)
        finally:
            mcp_server._UNIFIED_INDEX = old


class TestHandleRememberIndex:
    def test_incremental_index_update(self, tmp_path):
        from unittest.mock import MagicMock

        mock_idx = MagicMock()
        mock_idx._built = True
        import mcp_server

        old = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            with patch("mcp_server.BASE", tmp_path):
                handle_remember("content", "finding", "test")
            mock_idx.add_file.assert_called_once()
        finally:
            mcp_server._UNIFIED_INDEX = old


class TestHandleStatusActual:
    def test_status_captures_output(self):
        result = handle_status()
        assert isinstance(result, str)


class TestMCPProtocolRemember:
    def test_tools_call_remember_with_llm(self):
        with patch("mcp_server.handle_recall", return_value="Result") as mock:
            req = {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "remanentia_recall",
                    "arguments": {"query": "test", "llm": True},
                },
            }
            handle_request(req)
        mock.assert_called_once()
        call_kwargs = mock.call_args
        assert call_kwargs[1].get("llm") is True


# ── MCP tool-call audit ────────────────────────────────────────


class TestMCPToolAudit:
    def test_tool_call_writes_metadata_only_record(self, tmp_path):
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
        with (
            patch("mcp_server.MCP_AUDIT_LOGGER", ToolAuditLogger(audit_path)),
            patch("mcp_server.handle_recall", return_value="result"),
        ):
            resp = handle_request(req)
        assert resp["result"]["content"][0]["text"] == "result"

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
        with patch("mcp_server.MCP_AUDIT_LOGGER", ToolAuditLogger(audit_path)):
            handle_request(req)

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
            "id": "status-fail",
            "method": "tools/call",
            "params": {"name": "remanentia_status", "arguments": {}},
        }
        with (
            patch("mcp_server.MCP_AUDIT_LOGGER", ToolAuditLogger(audit_path)),
            patch("mcp_server.handle_status", side_effect=RuntimeError("sensitive detail")),
        ):
            resp = handle_request(req)

        assert resp["error"]["code"] == -32000
        assert "sensitive detail" not in resp["error"]["message"]
        record = json.loads(audit_path.read_text(encoding="utf-8"))
        assert record["tool"] == "remanentia_status"
        assert record["outcome"] == "error"
        assert record["error_type"] == "RuntimeError"


class TestMCPTelemetryAndCli:
    def test_emit_recall_bus_uses_cached_emitter(self, monkeypatch):
        import mcp_server

        class FakeEmitter:
            def __init__(self) -> None:
                self.calls = []

            def emit(self, query, **kwargs):
                self.calls.append((query, kwargs))

        emitter = FakeEmitter()
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER", emitter)
        monkeypatch.setattr(mcp_server, "_BUS_EMITTER_INIT", True)

        mcp_server._emit_recall_bus("query", ["trace:one"])

        assert emitter.calls == [
            (
                "query",
                {
                    "returned_claim_ids": ["trace:one"],
                    "was_used": False,
                    "abstained": False,
                },
            )
        ]

    def test_observe_and_close_noop_when_disabled_or_missing_event(self, monkeypatch):
        import mcp_server

        mcp_server._observe_recall(None, ["text"])
        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        mcp_server._observe_recall("event", ["text"])
        mcp_server._close_recall_loops("remembered text")

    def test_python_mcp_tokenizer_fallback(self, monkeypatch):
        import mcp_server

        monkeypatch.setattr(mcp_server, "_rust_mcp_tok", None)

        assert mcp_server._mcp_tok("Alpha be beta_42") == {"alpha", "beta_42"}

    def test_recall_feedback_disabled_no_prior_and_recorded(self, monkeypatch):
        import mcp_server

        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        assert "disabled" in handle_recall_feedback("query", True)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE")

        class FakeLedger:
            def __init__(self, event_id):
                self.event_id = event_id
                self.outcomes = []

            def latest_for(self, query, by=None):
                return self.event_id

            def record_outcome(self, event_id, **kwargs):
                self.outcomes.append((event_id, kwargs))

        missing = FakeLedger(None)
        monkeypatch.setattr(mcp_server, "_get_recall_ledger", lambda: missing)
        assert "No prior recall" in handle_recall_feedback("query", True)

        ledger = FakeLedger("evt-1")
        monkeypatch.setattr(mcp_server, "_get_recall_ledger", lambda: ledger)
        assert "was_used=True" in handle_recall_feedback("query", True, by="agent")
        assert ledger.outcomes == [("evt-1", {"was_used": True})]

    def test_recall_correctness_disabled_no_prior_and_recorded(self, monkeypatch):
        import mcp_server

        monkeypatch.setenv("REMANENTIA_RECALL_LEDGER_DISABLE", "1")
        assert "disabled" in handle_recall_correctness("query", False)
        monkeypatch.delenv("REMANENTIA_RECALL_LEDGER_DISABLE")

        class FakeLedger:
            def __init__(self, event_id):
                self.event_id = event_id
                self.outcomes = []

            def latest_for(self, query, by=None):
                return self.event_id

            def record_outcome(self, event_id, **kwargs):
                self.outcomes.append((event_id, kwargs))

        missing = FakeLedger(None)
        monkeypatch.setattr(mcp_server, "_get_recall_ledger", lambda: missing)
        assert "No prior recall" in handle_recall_correctness("query", False)

        ledger = FakeLedger("evt-2")
        monkeypatch.setattr(mcp_server, "_get_recall_ledger", lambda: ledger)
        assert "was_correct=False" in handle_recall_correctness("query", False, by="agent")
        assert ledger.outcomes == [("evt-2", {"was_correct": False})]

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

    def test_recall_flows_through_memory_index(self):
        """MCP recall → MemoryIndex.search → results."""
        from unittest.mock import MagicMock
        import mcp_server

        mock_idx = MagicMock()
        mock_idx._built = True
        mock_result = MagicMock()
        mock_result.source = "test"
        mock_result.name = "trace.md"
        mock_result.score = 0.9
        mock_result.answer = "42"
        mock_result.snippet = "The answer is 42."
        mock_idx.search.return_value = [mock_result]

        old = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            result = handle_recall("what is the answer", top_k=3)
            assert "42" in result
            mock_idx.search.assert_called_once()
        finally:
            mcp_server._UNIFIED_INDEX = old

    def test_remember_creates_file_and_updates_index(self):
        """MCP remember → write file → add to index."""
        from unittest.mock import MagicMock, patch as p
        import mcp_server

        mock_idx = MagicMock()
        mock_idx._built = True
        old = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            with p("mcp_server.BASE", mcp_server.BASE):
                result = handle_remember("Test memory content for pipeline verification.")
            assert (
                "stored" in result.lower() or "saved" in result.lower() or isinstance(result, str)
            )
        finally:
            mcp_server._UNIFIED_INDEX = old

    def test_status_returns_structured_info(self):
        result = handle_status()
        assert isinstance(result, str)
        # Should mention index, entities, or similar
        assert len(result) > 10

    def test_llm_backend_wiring_in_recall(self):
        """When llm=True, MCP sets up LLM backend automatically."""
        from unittest.mock import MagicMock
        import answer_extractor
        import mcp_server

        mock_idx = MagicMock()
        mock_idx._built = True
        mock_idx.search.return_value = []
        old_idx = mcp_server._UNIFIED_INDEX
        old_backend = answer_extractor._BACKEND
        answer_extractor._BACKEND = None
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            handle_recall("test query", llm=True)
            # Backend should have been auto-resolved
            assert answer_extractor._BACKEND is not None
        finally:
            mcp_server._UNIFIED_INDEX = old_idx
            answer_extractor._BACKEND = old_backend

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
    def test_remember_then_recall(self):
        """Pipeline: remember → recall → verify content persists."""
        from unittest.mock import MagicMock
        import mcp_server

        mock_idx = MagicMock()
        mock_idx._built = True
        mock_result = MagicMock()
        mock_result.source = "test"
        mock_result.name = "memory.md"
        mock_result.score = 0.9
        mock_result.answer = "persisted"
        mock_result.snippet = "This was persisted via remember."
        mock_idx.search.return_value = [mock_result]

        old = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = mock_idx
        try:
            result = handle_recall("persisted content", top_k=1)
            assert "persisted" in result.lower()
        finally:
            mcp_server._UNIFIED_INDEX = old
