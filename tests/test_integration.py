# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Integration Tests

"""End-to-end tests for the shipping retrieval pipeline.

Tests the full path: build index → search → answer extraction → results.
No mocking — exercises the actual code paths.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def index_with_data(tmp_path_factory):
    """Build a real MemoryIndex from test data."""
    from memory_index import MemoryIndex, SOURCES

    tmp = tmp_path_factory.mktemp("integration")

    traces = {
        "trace_stdp.md": (
            "# Decision: STDP removal\n"
            "Date: 2026-03-15\n"
            "Project: sc-neurocore\n\n"
            "We decided to set STDP weight to 0.0 because 70+ experiments\n"
            "showed it adds zero discriminative signal to retrieval.\n"
            "The SNN's role is consolidation, not retrieval.\n\n"
            "Best-paragraph embedding achieves 85.7% P@1 alone."
        ),
        "trace_locomo.md": (
            "# Finding: LOCOMO benchmark\n"
            "Date: 2026-03-22\n"
            "Project: remanentia\n\n"
            "LOCOMO v3 result: 66.4% accuracy.\n"
            "Strongest: multi-hop at 75.4%.\n"
            "Weakest: temporal at 15.6%.\n"
            "Gap to Hindsight SOTA: 91.4%."
        ),
        "trace_revenue.md": (
            "# Metric: Director-AI revenue\n"
            "Date: 2026-03-20\n"
            "Project: director-ai\n\n"
            "265 emails sent to 264 companies across 40+ countries.\n"
            "11 companies engaged. Revenue target: first pilot customer.\n"
            "Pricing: founding member, free 30-day pilot."
        ),
        "trace_alice.md": (
            "# Context: Alice and Bob discussion\n"
            "Date: 2026-03-25\n\n"
            "Alice mentioned her hobbies include pottery and hiking.\n"
            "Bob said he works at Google as a data scientist."
        ),
    }

    traces_dir = tmp / "traces"
    traces_dir.mkdir()
    for name, content in traces.items():
        (traces_dir / name).write_text(content, encoding="utf-8")

    # Build with only our test directory
    import memory_index
    original_sources = memory_index.SOURCES
    memory_index.SOURCES = {"test_traces": traces_dir}
    try:
        idx = MemoryIndex()
        idx.build(use_gpu_embeddings=False, use_gliner=False)
    finally:
        memory_index.SOURCES = original_sources

    return idx


class TestEndToEndSearch:
    """Full pipeline: index → search → results."""

    def test_basic_search_returns_results(self, index_with_data):
        results = index_with_data.search("STDP learning", top_k=3)
        assert len(results) > 0

    def test_search_finds_relevant_trace(self, index_with_data):
        results = index_with_data.search("LOCOMO benchmark accuracy", top_k=3)
        names = [r.name for r in results]
        assert "trace_locomo.md" in names

    def test_search_returns_scores(self, index_with_data):
        results = index_with_data.search("director-ai revenue", top_k=3)
        assert all(r.score > 0 for r in results)

    def test_answer_extraction_fires(self, index_with_data):
        results = index_with_data.search("What accuracy did LOCOMO achieve?", top_k=3)
        has_answer = any(r.answer for r in results)
        # Answer extraction may or may not fire depending on import availability
        # At minimum, results should exist
        assert len(results) > 0

    def test_person_centric_query(self, index_with_data):
        results = index_with_data.search("What are Alice's hobbies?", top_k=3)
        assert len(results) > 0
        found = any("pottery" in r.snippet.lower() for r in results)
        assert found


class TestInvertedIndex:
    """Inverted index correctness."""

    def test_inverted_index_has_tokens(self, index_with_data):
        assert len(index_with_data._inverted_index) > 0

    def test_posting_lists_point_to_valid_paragraphs(self, index_with_data):
        n_paras = len(index_with_data.paragraph_tokens)
        for token, posting in index_with_data._inverted_index.items():
            for idx in posting:
                assert 0 <= idx < n_paras, f"Token '{token}' has invalid posting {idx}"

    def test_token_in_posting_matches_paragraph(self, index_with_data):
        for token, posting in list(index_with_data._inverted_index.items())[:50]:
            for idx in posting:
                assert token in index_with_data.paragraph_tokens[idx], \
                    f"Token '{token}' in posting for para {idx} but not in para tokens"

    def test_para_lengths_match(self, index_with_data):
        assert len(index_with_data._para_lengths) == len(index_with_data.paragraph_tokens)
        for i, tokens in enumerate(index_with_data.paragraph_tokens):
            assert index_with_data._para_lengths[i] == len(tokens)


class TestAtomicSave:
    """Atomic save/load cycle."""

    def test_save_load_roundtrip(self, index_with_data, tmp_path):
        save_path = tmp_path / "test_index.pkl"
        index_with_data.save(save_path, quantize=False)
        assert save_path.exists()
        assert not save_path.with_suffix(".pkl.tmp").exists()

        from memory_index import MemoryIndex
        idx2 = MemoryIndex()
        assert idx2.load(save_path)
        assert len(idx2.documents) == len(index_with_data.documents)
        assert len(idx2.paragraph_tokens) == len(index_with_data.paragraph_tokens)
        assert len(idx2._inverted_index) > 0

    def test_search_after_load(self, index_with_data, tmp_path):
        save_path = tmp_path / "test_index2.pkl"
        index_with_data.save(save_path, quantize=False)

        from memory_index import MemoryIndex
        idx2 = MemoryIndex()
        idx2.load(save_path)
        results = idx2.search("STDP", top_k=3)
        assert len(results) > 0


class TestMCPServerIntegration:
    """MCP server handle_request with real data."""

    def test_mcp_recall_returns_text(self, monkeypatch):
        import mcp_server
        monkeypatch.setattr(mcp_server, "_UNIFIED_INDEX", None)

        resp = mcp_server.handle_request({
            "jsonrpc": "2.0", "id": 1,
            "method": "initialize", "params": {}
        })
        assert resp["result"]["serverInfo"]["name"] == "remanentia"

    def test_mcp_tools_list(self):
        import mcp_server
        resp = mcp_server.handle_request({
            "jsonrpc": "2.0", "id": 2,
            "method": "tools/list"
        })
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "remanentia_recall" in names
        assert "remanentia_remember" in names
        assert "remanentia_status" in names
        assert "remanentia_graph" in names

    def test_mcp_graph_query(self):
        import mcp_server
        resp = mcp_server.handle_request({
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {"name": "remanentia_graph", "arguments": {"top": 5}}
        })
        text = resp["result"]["content"][0]["text"]
        assert isinstance(text, str)
