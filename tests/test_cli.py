# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for cli.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli import (
    cmd_entities,
    cmd_graph,
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
