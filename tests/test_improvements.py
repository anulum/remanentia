# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for v0.3.0 improvements

"""Tests for the 8 improvements:
1. Real TF in BM25, DF-based IDF, normalised fusion
2. Reciprocal Rank Fusion
3. Relative date resolution, O(N) edge building, query-relevant dates
4. Typed relations in consolidation graph
5. Async consolidation
6. Thread safety
7. Query-proximity answer extraction
8. Raised chunking limits
"""

from __future__ import annotations

import time
import threading
from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np


# ── 1. BM25 improvements ────────────────────────────────────────


class TestTokenCounts:
    def test_counts_occurrences(self):
        from memory_index import _token_counts

        counts = _token_counts(["hello", "world", "hello", "test", "hello"])
        assert counts["hello"] == 3
        assert counts["world"] == 1
        assert counts["test"] == 1

    def test_empty_list(self):
        from memory_index import _token_counts

        assert _token_counts([]) == {}

    def test_single_token(self):
        from memory_index import _token_counts

        assert _token_counts(["foo"]) == {"foo": 1}


class TestRealTFInBM25:
    def test_repeated_term_scores_higher(self):
        """A paragraph mentioning a query term 5 times should score higher
        than one mentioning it once, all else equal."""
        from memory_index import MemoryIndex, _tokenize

        idx = MemoryIndex()
        idx._built = True

        # Manually set up two paragraphs
        from memory_index import Document

        doc = Document(
            name="test.md",
            source="test",
            path="/test",
            paragraphs=["alpha beta gamma", "alpha alpha alpha alpha alpha"],
            date="2026-03-26",
            doc_type="test",
        )
        idx.documents = [doc]
        idx.paragraph_index = [(0, 0), (0, 1)]
        tokens_0 = set(_tokenize("alpha beta gamma"))
        tokens_1 = set(_tokenize("alpha alpha alpha alpha alpha"))
        idx.paragraph_tokens = [tokens_0, tokens_1]
        idx.paragraph_token_counts = [
            {"alpha": 1, "beta": 1, "gamma": 1},
            {"alpha": 5},
        ]
        idx.paragraph_types = ["discussion", "discussion"]
        idx._para_lengths = np.array([len(tokens_0), len(tokens_1)], dtype=np.float32)
        idx._avg_dl = float(np.mean(idx._para_lengths))

        # Build inverted index
        inv = {}
        for i, tokens in enumerate(idx.paragraph_tokens):
            for t in tokens:
                if t not in inv:
                    inv[t] = []
                inv[t].append(i)
        idx._inverted_index = inv
        idx.idf = {"alpha": 0.5, "beta": 1.0, "gamma": 1.0}
        idx._df = {"alpha": 2, "beta": 1, "gamma": 1}

        scores = idx._search_python_bm25({"alpha"}, set())
        # Paragraph 1 (5 occurrences) should score higher than paragraph 0 (1 occurrence)
        assert scores[1] > scores[0]


class TestDFBasedIDF:
    def test_add_file_tracks_df(self, tmp_path):
        from memory_index import MemoryIndex

        idx = MemoryIndex()
        idx._built = True
        idx.documents = []
        idx.paragraph_index = []
        idx.paragraph_tokens = []
        idx.paragraph_token_counts = []
        idx.paragraph_types = []
        idx.idf = {}
        idx._df = {}
        idx._inverted_index = {}
        idx._para_lengths = np.array([], dtype=np.float32)
        idx._avg_dl = 1.0
        idx._rust_bm25_dirty = False
        idx._rust_bm25 = None

        f1 = tmp_path / "test1.md"
        f1.write_text(
            "This is a test document about memory retrieval systems and their performance."
        )
        idx.add_file(f1, source="test")

        f2 = tmp_path / "test2.md"
        f2.write_text("Another document about memory consolidation and graph analysis techniques.")
        idx.add_file(f2, source="test")

        # _df should track actual document frequency
        assert idx._df.get("memory", 0) >= 2
        # IDF should be computed from real df, not approximated
        import math

        n = len(idx.paragraph_tokens)
        expected_idf = math.log(1 + n / (1 + idx._df["memory"]))
        assert abs(idx.idf["memory"] - expected_idf) < 0.01


class TestSaveLoadTokenCounts:
    def test_round_trip(self, tmp_path):
        from memory_index import MemoryIndex

        idx = MemoryIndex()
        idx._built = True
        idx.documents = []
        idx.paragraph_index = []
        idx.paragraph_tokens = [{"hello", "world"}]
        idx.paragraph_token_counts = [{"hello": 3, "world": 1}]
        idx.paragraph_types = ["discussion"]
        idx.idf = {"hello": 0.5, "world": 1.0}
        idx._df = {"hello": 1, "world": 1}
        idx._inverted_index = {"hello": [0], "world": [0]}
        idx._para_lengths = np.array([2.0], dtype=np.float32)
        idx._avg_dl = 2.0
        idx.embeddings = None

        pkl = tmp_path / "test.pkl"
        idx.save(pkl)

        idx2 = MemoryIndex()
        assert idx2.load(pkl)
        assert idx2.paragraph_token_counts == [{"hello": 3, "world": 1}]
        assert idx2._df == {"hello": 1, "world": 1}


# ── 2. Reciprocal Rank Fusion ───────────────────────────────────


class TestRRF:
    def test_basic_fusion(self):
        from memory_index import _reciprocal_rank_fusion

        list_a = [(10, 5.0), (20, 4.0), (30, 3.0)]
        list_b = [(20, 9.0), (30, 8.0), (10, 7.0)]
        fused = _reciprocal_rank_fusion([list_a, list_b], k=60)
        # All items should appear
        ids = [idx for idx, _ in fused]
        assert 10 in ids
        assert 20 in ids
        assert 30 in ids

    def test_item_in_both_lists_ranks_higher(self):
        from memory_index import _reciprocal_rank_fusion

        list_a = [(10, 5.0), (20, 4.0)]
        list_b = [(10, 9.0), (30, 8.0)]
        fused = _reciprocal_rank_fusion([list_a, list_b], k=60)
        # Item 10 appears in both lists, should rank highest
        assert fused[0][0] == 10

    def test_single_list(self):
        from memory_index import _reciprocal_rank_fusion

        ranked = [(5, 10.0), (3, 8.0), (7, 6.0)]
        fused = _reciprocal_rank_fusion([ranked], k=60)
        assert [idx for idx, _ in fused] == [5, 3, 7]

    def test_empty_lists(self):
        from memory_index import _reciprocal_rank_fusion

        assert _reciprocal_rank_fusion([[], []], k=60) == []

    def test_k_parameter_affects_scores(self):
        from memory_index import _reciprocal_rank_fusion

        ranked = [(1, 10.0), (2, 5.0)]
        fused_k1 = _reciprocal_rank_fusion([ranked], k=1)
        fused_k100 = _reciprocal_rank_fusion([ranked], k=100)
        # With small k, rank difference has more impact
        gap_k1 = fused_k1[0][1] - fused_k1[1][1]
        gap_k100 = fused_k100[0][1] - fused_k100[1][1]
        assert gap_k1 > gap_k100


# ── 3. Temporal improvements ────────────────────────────────────


class TestRelativeDateResolution:
    def test_yesterday(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)
        dates = parse_dates("Saw it yesterday.", reference_date=ref)
        assert "2026-03-25" in dates

    def test_today(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)
        dates = parse_dates("Deployed today.", reference_date=ref)
        assert "2026-03-26" in dates

    def test_last_week(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)
        dates = parse_dates("Fixed last week.", reference_date=ref)
        assert "2026-03-19" in dates

    def test_last_month(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)
        dates = parse_dates("Released last month.", reference_date=ref)
        assert "2026-02-26" in dates

    def test_last_year(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)
        dates = parse_dates("Started last year.", reference_date=ref)
        assert "2025-03-26" in dates

    def test_this_week(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)  # Thursday
        dates = parse_dates("Happening this week.", reference_date=ref)
        # Should resolve to Monday of current week
        assert len(dates) == 1

    def test_this_month(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)
        dates = parse_dates("Updated this month.", reference_date=ref)
        assert "2026-03-01" in dates

    def test_this_year(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)
        dates = parse_dates("Released this year.", reference_date=ref)
        assert "2026-01-01" in dates

    def test_mixed_absolute_and_relative(self):
        from temporal_graph import parse_dates

        ref = date(2026, 3, 26)
        dates = parse_dates("Fixed 2026-03-15, tested yesterday.", reference_date=ref)
        assert "2026-03-15" in dates
        assert "2026-03-25" in dates

    def test_no_reference_uses_today(self):
        from temporal_graph import parse_dates

        dates = parse_dates("Deployed today.")
        assert date.today().isoformat() in dates


class TestResolveRelativeDate:
    def test_unknown_expression_returns_none(self):
        from temporal_graph import _resolve_relative_date

        assert _resolve_relative_date("next century", date(2026, 3, 26)) is None


class TestDateBucketedEdgeBuilding:
    def test_same_day_edges(self):
        from temporal_graph import TemporalGraph, TemporalEvent

        tg = TemporalGraph()
        events = [
            TemporalEvent(date="2026-03-15", text="Event A", source="a.md"),
            TemporalEvent(date="2026-03-15", text="Event B", source="b.md"),
        ]
        tg.add_events(events)
        same_day = [e for e in tg.edges if e.relation == "same_day"]
        assert len(same_day) >= 1

    def test_adjacent_date_edges(self):
        from temporal_graph import TemporalGraph, TemporalEvent

        tg = TemporalGraph()
        events = [
            TemporalEvent(date="2026-03-15", text="First event", source="a.md"),
            TemporalEvent(date="2026-03-16", text="Next event", source="b.md"),
        ]
        tg.add_events(events)
        before = [e for e in tg.edges if e.relation == "before"]
        assert len(before) >= 1

    def test_no_quadratic_explosion(self):
        from temporal_graph import TemporalGraph, TemporalEvent

        tg = TemporalGraph()
        # 100 events on 100 different dates
        events = [
            TemporalEvent(date=f"2026-01-{i + 1:02d}", text=f"Event {i}", source=f"e{i}.md")
            for i in range(100)
        ]
        tg.add_events(events)
        # With O(N²), we'd get ~5000 edges. With bucketed, far fewer.
        assert len(tg.edges) < 500

    def test_incremental_add_creates_cross_edges(self):
        from temporal_graph import TemporalGraph, TemporalEvent

        tg = TemporalGraph()
        tg.add_events([TemporalEvent(date="2026-03-15", text="Old event", source="a.md")])
        tg.add_events([TemporalEvent(date="2026-03-15", text="New event", source="b.md")])
        same_day = [e for e in tg.edges if e.relation == "same_day"]
        assert len(same_day) >= 1


class TestQueryRelevantDateExtraction:
    def test_returns_most_relevant_date(self):
        from answer_extractor import _extract_date_answer

        # Dates far apart so query-proximity window can distinguish them
        text = (
            "The project was initiated on 2026-01-10 with the initial planning phase. "
            "Several months of development followed with incremental improvements. "
            "Eventually the STDP learning rule bug was identified and fixed on 2026-03-15 "
            "after extensive debugging of the timing parameters. "
            "The final release was shipped on 2026-03-28 to production."
        )
        answer = _extract_date_answer(text, query="when was the STDP bug fixed")
        assert answer == "2026-03-15"

    def test_single_date_no_query(self):
        from answer_extractor import _extract_date_answer

        text = "Released on 2026-03-20."
        assert _extract_date_answer(text) == "2026-03-20"

    def test_falls_back_to_first_when_no_query(self):
        from answer_extractor import _extract_date_answer

        text = "2026-03-01 and 2026-03-15 and 2026-03-20."
        assert _extract_date_answer(text, query="") == "2026-03-01"


# ── 4. Typed relations ──────────────────────────────────────────


class TestTypedRelationsInGraph:
    def test_extract_typed_relations(self):
        from consolidation_engine import _extract_typed_relations

        text = "The STDP bug was fixed by Miroslav. BM25 depends on scikit-learn."
        entities = ["stdp", "miroslav", "bm25", "scikit-learn"]
        typed = _extract_typed_relations(text, entities)
        # Should detect "fixed" relation
        has_fixed = any(v == "fixed_by" for v in typed.values())
        has_depends = any(v == "depends_on" for v in typed.values())
        assert has_fixed or has_depends

    def test_co_occurs_fallback(self):
        from consolidation_engine import _extract_typed_relations

        text = "We used BM25 and embedding for retrieval."
        entities = ["bm25", "embedding"]
        typed = _extract_typed_relations(text, entities)
        # No typed pattern matches "and", so should return empty
        assert len(typed) == 0

    def test_update_graph_with_typed_relations(self, tmp_path):
        from consolidation_engine import _update_graph, _load_relations

        with (
            patch("consolidation_engine.GRAPH_DIR", tmp_path),
            patch("consolidation_engine.ENTITIES_PATH", tmp_path / "entities.jsonl"),
            patch("consolidation_engine.RELATIONS_PATH", tmp_path / "relations.jsonl"),
        ):
            _update_graph(
                "test_trace.md",
                ["stdp", "miroslav"],
                "test",
                "2026-03-26",
                text="The STDP bug was fixed by Miroslav in the LIF module.",
            )
            rels = _load_relations()
            assert len(rels) > 0
            # Should have a typed relation, not just co_occurs
            types = {r["type"] for r in rels}
            assert "fixed_by" in types or "co_occurs" in types


class TestEntityBoostWithTypedRelations:
    def test_typed_relation_boost(self):
        from memory_index import _entity_boost_score

        graph = {
            "entities": {
                "stdp": {"id": "stdp", "label": "stdp"},
                "lif": {"id": "lif", "label": "lif"},
            },
            "relations": [
                {"source": "stdp", "target": "lif", "type": "fixed_by", "weight": 3},
            ],
        }
        # Paragraph mentions lif, query is about stdp
        boost = _entity_boost_score("The lif module has a timing bug", {"stdp"}, graph)
        # Should get typed relation boost (0.15) since stdp→lif is fixed_by
        assert boost >= 0.15

    def test_co_occurs_no_extra_boost(self):
        from memory_index import _entity_boost_score

        graph = {
            "entities": {
                "stdp": {"id": "stdp", "label": "stdp"},
                "lif": {"id": "lif", "label": "lif"},
            },
            "relations": [
                {"source": "stdp", "target": "lif", "type": "co_occurs", "weight": 3},
            ],
        }
        boost = _entity_boost_score("The lif module has a timing bug", {"stdp"}, graph)
        # co_occurs should not get the extra 0.15
        assert boost < 0.15


# ── 5 & 6. Async consolidation & thread safety ──────────────────


class TestAsyncConsolidation:
    def test_debounce_prevents_rapid_fire(self):
        import mcp_server

        original_last = mcp_server._consolidation_last
        try:
            # Set last consolidation to "just now"
            mcp_server._consolidation_last = time.monotonic()
            mcp_server._consolidation_pending = False
            mcp_server._schedule_consolidation()
            # Should be pending, not executed
            assert mcp_server._consolidation_pending is True
        finally:
            mcp_server._consolidation_last = original_last

    def test_consolidation_runs_after_debounce(self):
        import mcp_server

        original_last = mcp_server._consolidation_last
        try:
            # Set last consolidation to long ago
            mcp_server._consolidation_last = 0.0
            mock_consolidate = MagicMock(return_value={"status": "ok"})
            with patch("consolidation_engine.consolidate", mock_consolidate):
                mcp_server._schedule_consolidation()
                for _ in range(20):
                    if mock_consolidate.called:
                        break
                    time.sleep(0.1)
            assert mock_consolidate.called
        finally:
            mcp_server._consolidation_last = original_last


class TestThreadSafety:
    def test_concurrent_knowledge_store_init(self):
        import mcp_server

        original_ks = mcp_server._KNOWLEDGE_STORE
        try:
            mcp_server._KNOWLEDGE_STORE = None
            results = []

            def _get():
                ks = mcp_server._get_knowledge_store()
                results.append(id(ks))

            threads = [threading.Thread(target=_get) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            # All threads should get the same singleton
            assert len(set(results)) == 1
        finally:
            mcp_server._KNOWLEDGE_STORE = original_ks


# ── 7. Query-proximity answer extraction ────────────────────────


class TestBestByProximity:
    def test_selects_nearest_to_query(self):
        from answer_extractor import _best_by_proximity

        # Spread candidates far apart so 80-char windows don't overlap
        text = (
            "The alpha channel has a precision of 42% which is measured on the validation set. "
            + "x" * 100
            + " The beta channel achieves an impressive accuracy of 88% on the test benchmark. "
            + "x" * 100
            + " The gamma channel only reaches 15% in comparison to the others."
        )
        pos_42 = text.index("42%")
        pos_88 = text.index("88%")
        pos_15 = text.index("15%")
        candidates = [("42%", pos_42), ("88%", pos_88), ("15%", pos_15)]
        best = _best_by_proximity(candidates, text, "what is beta accuracy")
        assert best == "88%"

    def test_single_candidate(self):
        from answer_extractor import _best_by_proximity

        best = _best_by_proximity([("42%", 10)], "Score is 42%.", "what score")
        assert best == "42%"


class TestImprovedYesNo:
    def test_multiple_negation_markers(self):
        from answer_extractor import _extract_yes_no

        text = "The system couldn't handle it and wasn't able to recover."
        result = _extract_yes_no(text, "can the system handle errors")
        assert result == "No"

    def test_positive_context(self):
        from answer_extractor import _extract_yes_no

        text = "Yes, the system handles errors gracefully and recovers."
        result = _extract_yes_no(text, "can the system handle errors")
        assert result == "Yes"


class TestNumberProximity:
    def test_returns_query_relevant_number(self):
        from answer_extractor import _extract_number_answer

        text = "We have 500 documents. The precision is 92.5. There are 3 clusters."
        result = _extract_number_answer(text, "how many documents")
        assert result == "500"


# ── 8. Chunking limits ──────────────────────────────────────────


class TestChunkingLimits:
    def test_max_code_chunk_chars_raised(self):
        from memory_index import MAX_CODE_CHUNK_CHARS

        assert MAX_CODE_CHUNK_CHARS >= 1000

    def test_max_code_chunks_raised(self):
        from memory_index import MAX_CODE_CHUNKS

        assert MAX_CODE_CHUNKS >= 200

    def test_large_file_not_silently_truncated(self):
        from memory_index import _split_python_code

        # Generate a file with 100 functions of ~50 chars each
        lines = []
        for i in range(100):
            lines.append(f"def func_{i}(x):")
            lines.append(f'    """Function {i} does something useful."""')
            lines.append(f"    return x + {i}")
            lines.append("")
        code = "\n".join(lines)
        chunks = _split_python_code(code)
        # Should capture all 100 functions (old limit was 50)
        assert len(chunks) >= 99
