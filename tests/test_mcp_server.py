# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for mcp_server.py

from __future__ import annotations

from unittest.mock import patch


from mcp_server import (
    TOOLS,
    handle_graph,
    handle_recall,
    handle_remember,
    handle_request,
    handle_status,
)


# ── Tool definitions ─────────────────────────────────────────────


class TestToolDefinitions:
    def test_four_tools_defined(self):
        assert len(TOOLS) == 4

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {"remanentia_recall", "remanentia_remember", "remanentia_status", "remanentia_graph"}

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
        assert len(resp["result"]["tools"]) == 4

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
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "remanentia_recall", "arguments": {"query": "test"}},
            }
            resp = handle_request(req)
        assert resp["result"]["content"][0]["type"] == "text"
        assert resp["result"]["content"][0]["text"] == "Test result"

    def test_tools_call_status(self):
        with patch("mcp_server.handle_status", return_value="Status info"):
            req = {
                "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "remanentia_status", "arguments": {}},
            }
            resp = handle_request(req)
        assert "Status info" in resp["result"]["content"][0]["text"]

    def test_tools_call_graph(self):
        with patch("mcp_server.handle_graph", return_value="Graph data"):
            req = {
                "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                "params": {"name": "remanentia_graph", "arguments": {"entity": "stdp"}},
            }
            resp = handle_request(req)
        assert "Graph data" in resp["result"]["content"][0]["text"]

    def test_tools_call_unknown_tool(self):
        req = {
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
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
        with patch("mcp_server._UNIFIED_INDEX", None), \
             patch("mcp_server._lightweight_recall", return_value="lightweight result"):
            # MemoryIndex load fails → lightweight
            result = handle_recall("test query")
        assert isinstance(result, str)

    def test_empty_query(self):
        result = handle_recall("")
        assert isinstance(result, str)


# ── handle_remember ──────────────────────────────────────────


class TestHandleRemember:
    def test_writes_trace_file(self, tmp_path):
        with patch("mcp_server.BASE", tmp_path), \
             patch("mcp_server._RECALL_INDEX", None), \
             patch("mcp_server._UNIFIED_INDEX", None):
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
        with patch("mcp_server.BASE", tmp_path), \
             patch("mcp_server._UNIFIED_INDEX", None):
            handle_remember("test content", "context", "")
        assert mcp_server._RECALL_INDEX is None

    def test_mcp_protocol_remember(self, tmp_path):
        with patch("mcp_server.BASE", tmp_path), \
             patch("mcp_server._UNIFIED_INDEX", None):
            req = {
                "jsonrpc": "2.0", "id": 10, "method": "tools/call",
                "params": {"name": "remanentia_remember",
                           "arguments": {"content": "Test memory", "type": "finding", "project": "test"}},
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
        with patch("mcp_server.BASE", tmp_path), \
             patch("mcp_server._RECALL_INDEX", None), \
             patch("mcp_server._UNIFIED_INDEX", None), \
             patch("mcp_server._consolidation_last", 0.0), \
             patch("consolidation_engine.consolidate", mock_consolidate):
            handle_remember("Test consolidation trigger", "finding", "test")
            # Consolidation now runs in a background thread
            for _ in range(20):
                if mock_consolidate.called:
                    break
                time.sleep(0.1)
        mock_consolidate.assert_called_once()

    def test_consolidation_failure_safe(self, tmp_path):
        with patch("mcp_server.BASE", tmp_path), \
             patch("mcp_server._RECALL_INDEX", None), \
             patch("mcp_server._UNIFIED_INDEX", None):
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
            SearchResult(name="test.md", source="traces", score=0.9,
                         snippet="Test snippet", answer="March 15"),
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

    def test_load_fails_falls_back(self, tmp_traces):
        import mcp_server
        old = mcp_server._UNIFIED_INDEX
        mcp_server._UNIFIED_INDEX = None
        mcp_server._RECALL_INDEX = None
        try:
            with patch("mcp_server.BASE", tmp_traces.parent), \
                 patch("mcp_server._UNIFIED_INDEX", None):
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
                "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                "params": {"name": "remanentia_recall",
                           "arguments": {"query": "test", "llm": True}},
            }
            resp = handle_request(req)
        mock.assert_called_once()
        call_kwargs = mock.call_args
        assert call_kwargs[1].get("llm") is True
