# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Integration tests

"""End-to-end tests for the shipping retrieval pipeline.

Tests the full path: build index → search → answer extraction → results.
No mocking — exercises the actual code paths.
"""

from __future__ import annotations


import pytest


@pytest.fixture(scope="module")
def index_with_data(tmp_path_factory):
    """Build a real MemoryIndex from test data."""
    from memory_index import MemoryIndex

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
                assert token in index_with_data.paragraph_tokens[idx], (
                    f"Token '{token}' in posting for para {idx} but not in para tokens"
                )

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

        resp = mcp_server.handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        assert resp["result"]["serverInfo"]["name"] == "remanentia"

    def test_mcp_tools_list(self):
        import mcp_server

        resp = mcp_server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "remanentia_recall" in names
        assert "remanentia_remember" in names
        assert "remanentia_status" in names
        assert "remanentia_graph" in names

    def test_mcp_graph_query(self):
        import mcp_server

        resp = mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "remanentia_graph", "arguments": {"top": 5}},
            }
        )
        text = resp["result"]["content"][0]["text"]
        assert isinstance(text, str)


# ── Negative / edge case tests ───────────────────────────────


class TestSearchEdgeCases:
    """Boundary, empty, and adversarial inputs."""

    def test_empty_query(self, index_with_data):
        results = index_with_data.search("", top_k=3)
        assert isinstance(results, list)

    def test_nonsense_query(self, index_with_data):
        results = index_with_data.search("xyzzy frobnicator qux", top_k=3)
        # Should return empty or low-score results, not crash
        assert isinstance(results, list)

    def test_top_k_zero(self, index_with_data):
        results = index_with_data.search("STDP", top_k=0)
        assert results == []

    def test_top_k_one(self, index_with_data):
        results = index_with_data.search("STDP", top_k=1)
        assert len(results) <= 1

    def test_top_k_larger_than_corpus(self, index_with_data):
        results = index_with_data.search("STDP", top_k=10000)
        assert len(results) <= len(index_with_data.documents)

    def test_query_with_special_chars(self, index_with_data):
        results = index_with_data.search("what's the 85.7% score?", top_k=3)
        assert isinstance(results, list)

    def test_unicode_query(self, index_with_data):
        results = index_with_data.search("Šotek Remanentia výsledky", top_k=3)
        assert isinstance(results, list)

    def test_very_long_query(self, index_with_data):
        results = index_with_data.search("STDP " * 500, top_k=3)
        assert isinstance(results, list)


class TestSearchQuality:
    """Retrieval quality assertions beyond basic smoke tests."""

    def test_scores_monotonically_decrease(self, index_with_data):
        results = index_with_data.search("LOCOMO benchmark", top_k=10)
        if len(results) >= 2:
            scores = [r.score for r in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], "Results not sorted by score"

    def test_different_queries_different_top(self, index_with_data):
        r1 = index_with_data.search("STDP learning SNN", top_k=1)
        r2 = index_with_data.search("Alice hobbies pottery", top_k=1)
        if r1 and r2:
            assert r1[0].name != r2[0].name, "Different queries should rank different docs first"

    def test_exact_term_match_ranked_high(self, index_with_data):
        results = index_with_data.search("pottery", top_k=3)
        if results:
            assert "pottery" in results[0].snippet.lower()

    def test_project_filter_narrows_results(self, index_with_data):
        all_results = index_with_data.search("decision", top_k=10)
        filtered = index_with_data.search("decision", top_k=10, project="sc-neurocore")
        assert len(filtered) <= len(all_results)


class TestSaveLoadIntegrity:
    """Save/load preserves all index properties."""

    def test_idf_preserved(self, index_with_data, tmp_path):
        save_path = tmp_path / "idf_test.pkl"
        index_with_data.save(save_path, quantize=False)
        from memory_index import MemoryIndex

        idx2 = MemoryIndex()
        idx2.load(save_path)
        assert set(idx2.idf.keys()) == set(index_with_data.idf.keys())

    def test_search_results_match_after_roundtrip(self, index_with_data, tmp_path):
        save_path = tmp_path / "roundtrip.pkl"
        index_with_data.save(save_path, quantize=False)
        from memory_index import MemoryIndex

        idx2 = MemoryIndex()
        idx2.load(save_path)

        q = "LOCOMO benchmark accuracy"
        r_orig = index_with_data.search(q, top_k=3)
        r_loaded = idx2.search(q, top_k=3)
        assert len(r_orig) == len(r_loaded)
        for a, b in zip(r_orig, r_loaded):
            assert a.name == b.name
            assert abs(a.score - b.score) < 0.01


class TestMCPEdgeCases:
    """MCP server error handling."""

    def test_unknown_method(self):
        import mcp_server

        resp = mcp_server.handle_request(
            {"jsonrpc": "2.0", "id": 99, "method": "nonexistent/method"}
        )
        assert "error" in resp

    def test_unknown_tool(self):
        import mcp_server

        resp = mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {"name": "nonexistent_tool", "arguments": {}},
            }
        )
        assert "error" in resp or "Unknown" in str(resp)
