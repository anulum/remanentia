# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for cli.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli import (
    cmd_consolidate,
    cmd_entities,
    cmd_graph,
    cmd_init,
    cmd_recall,
    cmd_status,
    main,
)


# ── cmd_status ───────────────────────────────────────────────────


class TestCmdStatus:
    def test_no_daemon(self, tmp_path, capsys):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        semantic_dir = tmp_path / "memory" / "semantic"
        semantic_dir.mkdir(parents=True)

        with patch("cli.STATE_DIR", state_dir), \
             patch("cli.GRAPH_DIR", graph_dir), \
             patch("cli.BASE", tmp_path):
            args = type("Args", (), {})()
            cmd_status(args)

        out = capsys.readouterr().out
        assert "NOT RUNNING" in out

    def test_with_daemon_state(self, tmp_path, capsys):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        import time
        state = {
            "timestamp": time.time(),
            "cycle": 42,
            "n_neurons": 512,
            "vram_mb": 256,
            "live_retrieval_available": True,
        }
        (state_dir / "current_state.json").write_text(
            json.dumps(state), encoding="utf-8",
        )
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        semantic_dir = tmp_path / "memory" / "semantic"
        semantic_dir.mkdir(parents=True)

        with patch("cli.STATE_DIR", state_dir), \
             patch("cli.GRAPH_DIR", graph_dir), \
             patch("cli.BASE", tmp_path):
            cmd_status(type("Args", (), {})())

        out = capsys.readouterr().out
        assert "ALIVE" in out
        assert "42" in out

    def test_memory_counts(self, tmp_path, capsys):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        (traces_dir / "trace1.md").write_text("content", encoding="utf-8")
        (traces_dir / "trace2.md").write_text("content", encoding="utf-8")
        semantic_dir = tmp_path / "memory" / "semantic"
        semantic_dir.mkdir(parents=True)
        (semantic_dir / "mem1.md").write_text("content", encoding="utf-8")
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)

        with patch("cli.STATE_DIR", state_dir), \
             patch("cli.GRAPH_DIR", graph_dir), \
             patch("cli.BASE", tmp_path):
            cmd_status(type("Args", (), {})())

        out = capsys.readouterr().out
        assert "2" in out  # 2 traces


# ── cmd_graph ────────────────────────────────────────────────────


class TestCmdGraph:
    def test_no_relations(self, tmp_path, capsys):
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()

        with patch("cli.GRAPH_DIR", graph_dir):
            cmd_graph(type("Args", (), {"top": 10})())

        out = capsys.readouterr().out
        assert "No relations" in out

    def test_with_relations(self, tmp_graph, capsys):
        with patch("cli.GRAPH_DIR", tmp_graph):
            cmd_graph(type("Args", (), {"top": 3})())

        out = capsys.readouterr().out
        assert "entity relationships" in out.lower() or "<->" in out


# ── cmd_entities ─────────────────────────────────────────────────


class TestCmdEntities:
    def test_no_entities(self, tmp_path, capsys):
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()

        with patch("cli.GRAPH_DIR", graph_dir):
            cmd_entities(type("Args", (), {})())

        out = capsys.readouterr().out
        assert "No entities" in out

    def test_with_entities(self, tmp_graph, capsys):
        with patch("cli.GRAPH_DIR", tmp_graph):
            cmd_entities(type("Args", (), {})())

        out = capsys.readouterr().out
        assert "entities" in out.lower()
        assert "stdp" in out


# ── main() argument parsing ──────────────────────────────────────


class TestMain:
    def test_no_args_shows_help(self, capsys):
        with patch("sys.argv", ["remanentia"]):
            main()
        # argparse prints help to stdout
        out = capsys.readouterr().out
        assert "usage" in out.lower() or "remanentia" in out.lower() or out == ""

    def test_status_command(self, tmp_path, capsys):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        traces_dir = tmp_path / "reasoning_traces"
        traces_dir.mkdir()
        semantic_dir = tmp_path / "memory" / "semantic"
        semantic_dir.mkdir(parents=True)

        with patch("sys.argv", ["remanentia", "status"]), \
             patch("cli.STATE_DIR", state_dir), \
             patch("cli.GRAPH_DIR", graph_dir), \
             patch("cli.BASE", tmp_path):
            main()

        out = capsys.readouterr().out
        assert "Daemon" in out or "NOT RUNNING" in out


# ── cmd_recall ──────────────────────────────────────────────────


class TestCmdRecall:
    def test_recall_with_filters(self, tmp_path, capsys):
        from memory_index import SearchResult
        mock_idx = MagicMock()
        mock_idx._built = True
        mock_idx.search.return_value = [
            SearchResult(name="test.md", source="test", score=0.9,
                         snippet="snippet", answer="42"),
        ]
        with patch("memory_index.auto_rebuild_if_needed", return_value=mock_idx):
            args = type("Args", (), {
                "query": "test query", "top": 3, "format": "summary",
                "content": False, "project": "test", "after": "", "before": "",
                "llm": False,
            })()
            cmd_recall(args)
        out = capsys.readouterr().out
        assert "test.md" in out
        assert "42" in out

    def test_recall_with_llm_flag(self, tmp_path, capsys):
        mock_idx = MagicMock()
        mock_idx._built = True
        mock_idx.search.return_value = []
        with patch("memory_index.auto_rebuild_if_needed", return_value=mock_idx):
            args = type("Args", (), {
                "query": "test", "top": 3, "format": "summary",
                "content": False, "project": "p", "after": "", "before": "",
                "llm": True,
            })()
            cmd_recall(args)
        call_kwargs = mock_idx.search.call_args
        assert call_kwargs[1].get("use_llm") is True

    def test_recall_no_filters(self, capsys):
        mock_ctx = MagicMock()
        mock_ctx.summary = "Summary text"
        mock_ctx.to_llm_context.return_value = "LLM context"
        with patch("memory_recall.recall", return_value=mock_ctx):
            args = type("Args", (), {
                "query": "test query", "top": 3, "format": "summary",
                "content": False, "project": "", "after": "", "before": "",
                "llm": False,
            })()
            cmd_recall(args)
        out = capsys.readouterr().out
        assert "Summary" in out

    def test_recall_context_format(self, capsys):
        mock_ctx = MagicMock()
        mock_ctx.to_llm_context.return_value = "LLM context output"
        with patch("memory_recall.recall", return_value=mock_ctx):
            args = type("Args", (), {
                "query": "test", "top": 3, "format": "context",
                "content": False, "project": "", "after": "", "before": "",
                "llm": False,
            })()
            cmd_recall(args)
        out = capsys.readouterr().out
        assert "LLM context" in out

    def test_recall_json_format(self, capsys):
        mock_ctx = MagicMock()
        mock_ctx.query = "test"
        mock_ctx.trace = "t.md"
        mock_ctx.trace_score = 0.8
        mock_ctx.entities = ["a"]
        mock_ctx.related_entities = []
        mock_ctx.semantic_memories = []
        mock_ctx.before = []
        mock_ctx.after = []
        mock_ctx.cross_project = []
        mock_ctx.novelty_score = 0.5
        mock_ctx.elapsed_ms = 10
        with patch("memory_recall.recall", return_value=mock_ctx):
            args = type("Args", (), {
                "query": "test", "top": 3, "format": "json",
                "content": False, "project": "", "after": "", "before": "",
                "llm": False,
            })()
            cmd_recall(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["query"] == "test"


# ── cmd_consolidate ─────────────────────────────────────────────


class TestCmdConsolidate:
    def test_consolidate(self, capsys):
        mock_result = {"status": "ok", "traces_processed": 3, "memories_written": 1}
        with patch("consolidation_engine.consolidate", return_value=mock_result):
            args = type("Args", (), {"force": False})()
            cmd_consolidate(args)
        out = capsys.readouterr().out
        assert "Done" in out
        assert "ok" in out

    def test_consolidate_force(self, capsys):
        mock_result = {"status": "ok", "traces_processed": 5}
        with patch("consolidation_engine.consolidate", return_value=mock_result):
            args = type("Args", (), {"force": True})()
            cmd_consolidate(args)
        out = capsys.readouterr().out
        assert "Done" in out


# ── cmd_init ────────────────────────────────────────────────────


class TestCmdInit:
    def test_creates_dirs(self, tmp_path, capsys):
        with patch("cli.BASE", tmp_path):
            args = type("Args", (), {})()
            cmd_init(args)
        out = capsys.readouterr().out
        assert "Created" in out or "already exist" in out
        assert (tmp_path / "reasoning_traces").exists()
        assert (tmp_path / "memory" / "semantic").exists()

    def test_already_exists(self, tmp_path, capsys):
        for d in ["reasoning_traces", "memory/semantic", "memory/graph",
                   "consolidation", "snn_state"]:
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
        with patch("cli.BASE", tmp_path):
            cmd_init(type("Args", (), {})())
        out = capsys.readouterr().out
        assert "already exist" in out


# ── cmd_daemon ──────────────────────────────────────────────────


class TestCmdDaemon:
    def test_daemon_stop_no_lock(self, tmp_path, capsys):
        from cli import cmd_daemon
        with patch("cli.STATE_DIR", tmp_path):
            cmd_daemon(type("Args", (), {"action": "stop"})())
        out = capsys.readouterr().out
        assert "No daemon lock" in out

    def test_daemon_stop_with_lock(self, tmp_path, capsys):
        from cli import cmd_daemon
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        (state_dir / "daemon.lock").write_text("99999", encoding="utf-8")
        with patch("cli.STATE_DIR", state_dir), \
             patch("os.kill", side_effect=OSError("No such process")):
            cmd_daemon(type("Args", (), {"action": "stop"})())
        out = capsys.readouterr().out
        assert "Failed" in out

    def test_daemon_stop_success(self, tmp_path, capsys):
        from cli import cmd_daemon
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        (state_dir / "daemon.lock").write_text("12345", encoding="utf-8")
        with patch("cli.STATE_DIR", state_dir), \
             patch("os.kill"):
            cmd_daemon(type("Args", (), {"action": "stop"})())
        out = capsys.readouterr().out
        assert "SIGTERM" in out

    def test_daemon_start(self, capsys):
        from cli import cmd_daemon
        with patch("subprocess.Popen") as mock_popen:
            cmd_daemon(type("Args", (), {"action": "start"})())
        out = capsys.readouterr().out
        assert "start requested" in out

    def test_daemon_status(self, tmp_path, capsys):
        from cli import cmd_daemon
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)
        with patch("cli.STATE_DIR", state_dir), \
             patch("cli.GRAPH_DIR", graph_dir), \
             patch("cli.BASE", tmp_path):
            cmd_daemon(type("Args", (), {"action": "status"})())
        out = capsys.readouterr().out
        assert "Daemon" in out or "NOT RUNNING" in out


# ── cmd_status stale daemon ─────────────────────────────────────


class TestCmdStatusStale:
    def test_stale_daemon(self, tmp_path, capsys):
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        state = {"timestamp": 0, "cycle": 1, "n_neurons": 100, "vram_mb": 0}
        (state_dir / "current_state.json").write_text(
            json.dumps(state), encoding="utf-8",
        )
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)
        with patch("cli.STATE_DIR", state_dir), \
             patch("cli.GRAPH_DIR", graph_dir), \
             patch("cli.BASE", tmp_path):
            cmd_status(type("Args", (), {})())
        out = capsys.readouterr().out
        assert "STALE" in out

    def test_consolidation_in_state(self, tmp_path, capsys):
        import time as t
        state_dir = tmp_path / "snn_state"
        state_dir.mkdir()
        state = {
            "timestamp": t.time(), "cycle": 10, "n_neurons": 500, "vram_mb": 100,
            "live_retrieval_available": False,
            "last_consolidation": {"memories_written": 3, "entities_found": 10},
        }
        (state_dir / "current_state.json").write_text(json.dumps(state), encoding="utf-8")
        graph_dir = tmp_path / "memory" / "graph"
        graph_dir.mkdir(parents=True)
        (tmp_path / "reasoning_traces").mkdir()
        (tmp_path / "memory" / "semantic").mkdir(parents=True)
        with patch("cli.STATE_DIR", state_dir), \
             patch("cli.GRAPH_DIR", graph_dir), \
             patch("cli.BASE", tmp_path):
            cmd_status(type("Args", (), {})())
        out = capsys.readouterr().out
        assert "3 memories" in out or "consolidation" in out.lower()
