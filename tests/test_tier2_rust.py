# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for Tier 2 Rust crate extensions

"""Multi-angle tests for Tier 2 Rust functions:
- RustFactIndex (pyclass) + fact_index_query
- build_temporal_edges + score_temporal_query
- knowledge_search + get_related_ids + graph_search + RustKnowledgeIndex
"""

from __future__ import annotations

import time

import pytest

fd = pytest.importorskip("remanentia_fact_decomposer")
tp = pytest.importorskip("remanentia_temporal")
ks = pytest.importorskip("remanentia_knowledge_store")


# ── RustFactIndex ──────────────────────────────────────────────


class TestRustFactIndex:
    def _make_index(self, n=50):
        texts = [f"Plasma temperature reached {100 + i} degrees at tokamak." for i in range(n)]
        entities = [["tokamak", "plasma", f"entity_{i}"] for i in range(n)]
        valid_until = ["" for _ in range(n)]
        session_idx = [float(i // 10) for i in range(n)]
        supersedes = [i > 40 for i in range(n)]
        return fd.RustFactIndex(texts, entities, valid_until, session_idx, supersedes)

    def test_basic_query(self):
        idx = self._make_index()
        results = idx.query("plasma temperature at tokamak", "", False, 5)
        assert len(results) <= 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        for fact_idx, score in results:
            assert isinstance(fact_idx, int)
            assert isinstance(score, float)
            assert score > 0

    def test_entity_boost(self):
        idx = fd.RustFactIndex(
            ["Alpha works at Beta Corp.", "Gamma uses Delta."],
            [["Alpha", "Beta Corp"], ["Gamma", "Delta"]],
            ["", ""], [0.0, 0.0], [False, False],
        )
        results = idx.query("What does Alpha do at Beta Corp?", "", False, 5)
        assert len(results) > 0
        top_idx = results[0][0]
        assert top_idx == 0

    def test_temporal_filtering(self):
        idx = fd.RustFactIndex(
            ["Old fact.", "New fact."],
            [[], []],
            ["2024-01-01", ""],  # first fact expired
            [0.0, 1.0], [False, False],
        )
        results = idx.query("fact", "2025-01-01", True, 10)
        fact_indices = [r[0] for r in results]
        assert 0 not in fact_indices
        assert 1 in fact_indices

    def test_recency_boost(self):
        idx = fd.RustFactIndex(
            ["Job at company A.", "Job at company B."],
            [[], []],
            ["", ""],
            [0.0, 5.0],  # second fact from later session
            [False, True],
        )
        results = idx.query("current job", "", False, 10)
        if len(results) >= 2:
            assert results[0][0] == 1

    def test_empty_query(self):
        idx = self._make_index()
        results = idx.query("", "", False, 5)
        assert results == []

    def test_no_match(self):
        idx = self._make_index()
        results = idx.query("xyzzy frobnicator zilch", "", False, 5)
        assert results == []

    def test_top_k_respected(self):
        idx = self._make_index(100)
        results = idx.query("plasma temperature tokamak degrees", "", False, 3)
        assert len(results) <= 3

    def test_performance(self):
        idx = self._make_index(1000)
        t0 = time.perf_counter()
        for _ in range(500):
            idx.query("plasma temperature at tokamak facility", "", False, 10)
        elapsed_us = (time.perf_counter() - t0) / 500 * 1e6
        assert elapsed_us < 5000, f"RustFactIndex.query too slow: {elapsed_us:.0f} µs"


# ── fact_index_query (stateless) ──────────────────────────────


class TestFactIndexQuery:
    def test_basic(self):
        kw = {"plasma": [0, 1], "reactor": [1, 2]}
        ent = {"tokamak": [0]}
        results = fd.fact_index_query(
            kw, ent, "plasma reactor at tokamak",
            ["", "", ""], [0.0, 0.0, 0.0], [False, False, False],
            "", False, 5,
        )
        assert len(results) > 0
        indices = [r[0] for r in results]
        assert 1 in indices  # "plasma" + "reactor" both hit idx 1

    def test_empty(self):
        results = fd.fact_index_query({}, {}, "test", [], [], [], "", False, 5)
        assert results == []


# ── build_temporal_edges ──────────────────────────────────────


class TestBuildTemporalEdges:
    def test_same_day_edges(self):
        by_date = {"2025-01-15": [0]}
        new_events = [("2025-01-15", "New event on Jan 15")]
        edges = tp.build_temporal_edges(by_date, new_events, 1, {0: "Old event on Jan 15"})
        same_day = [e for e in edges if e[2] == "same_day"]
        assert len(same_day) >= 1
        assert same_day[0][3] == "2025-01-15"

    def test_before_edges(self):
        by_date = {"2025-01-15": [0]}
        new_events = [("2025-01-10", "Earlier event")]
        edges = tp.build_temporal_edges(by_date, new_events, 1, {0: "Later event"})
        before_edges = [e for e in edges if e[2] == "before"]
        assert len(before_edges) >= 1

    def test_new_new_same_day(self):
        new_events = [("2025-03-01", "Event A"), ("2025-03-01", "Event B")]
        edges = tp.build_temporal_edges({}, new_events, 0, {})
        same_day = [e for e in edges if e[2] == "same_day"]
        assert len(same_day) == 1

    def test_empty_inputs(self):
        edges = tp.build_temporal_edges({}, [], 0, {})
        assert edges == []

    def test_no_cross_date_without_adjacent(self):
        new_events = [("2025-01-01", "A"), ("2025-12-31", "B")]
        edges = tp.build_temporal_edges({}, new_events, 0, {})
        before_edges = [e for e in edges if e[2] == "before"]
        assert len(before_edges) >= 1


# ── score_temporal_query ──────────────────────────────────────


class TestScoreTemporalQuery:
    def _events(self):
        return [
            ("2025-01-15", "Plasma density measured at reactor", "doc1", 0),
            ("2025-03-20", "Temperature peaked at 200 degrees", "doc2", 1),
            ("2025-06-10", "New plasma experiment started", "doc3", 2),
            ("2025-09-01", "Reactor maintenance completed", "doc4", 3),
        ]

    def test_relevance_scoring(self):
        events = self._events()
        indices = tp.score_temporal_query(events, "plasma density reactor", [], 5)
        assert len(indices) > 0
        assert 0 in indices  # "Plasma density measured at reactor" matches best

    def test_latest_sorting(self):
        events = self._events()
        indices = tp.score_temporal_query(events, "latest experiment", [], 5)
        if indices:
            assert events[indices[0]][0] >= events[indices[-1]][0]

    def test_earliest_sorting(self):
        events = self._events()
        indices = tp.score_temporal_query(events, "first plasma experiment", [], 5)
        if len(indices) >= 2:
            assert events[indices[0]][0] <= events[indices[1]][0]

    def test_date_filter_after(self):
        events = self._events()
        indices = tp.score_temporal_query(
            events, "experiments after 2025-06-01", ["2025-06-01"], 10,
        )
        for i in indices:
            assert events[i][0] >= "2025-06-01"

    def test_date_filter_before(self):
        events = self._events()
        indices = tp.score_temporal_query(
            events, "events before 2025-04-01", ["2025-04-01"], 10,
        )
        for i in indices:
            assert events[i][0] <= "2025-04-01"

    def test_empty_events(self):
        indices = tp.score_temporal_query([], "test query", [], 5)
        assert indices == []

    def test_top_k(self):
        events = self._events()
        indices = tp.score_temporal_query(events, "plasma reactor temperature", [], 2)
        assert len(indices) <= 2

    def test_performance(self):
        events = [
            (f"2025-{(i%12)+1:02d}-{(i%28)+1:02d}", f"Event {i} plasma tokamak density", "doc", i)
            for i in range(1000)
        ]
        t0 = time.perf_counter()
        for _ in range(500):
            tp.score_temporal_query(events, "latest plasma density at tokamak", [], 5)
        elapsed_us = (time.perf_counter() - t0) / 500 * 1e6
        assert elapsed_us < 20000, f"score_temporal_query too slow: {elapsed_us:.0f} µs"


# ── knowledge_search ──────────────────────────────────────────


class TestKnowledgeSearch:
    def _token_index(self):
        return {
            "n1": {"plasma", "temperature", "tokamak", "density"},
            "n2": {"reactor", "maintenance", "schedule", "tokamak"},
            "n3": {"plasma", "experiment", "new", "density"},
            "n4": {"unrelated", "topic", "cooking", "recipes"},
        }

    def test_basic_search(self):
        results = ks.knowledge_search(
            self._token_index(), set(), {"plasma", "density"}, 5, True,
        )
        assert len(results) >= 2
        ids = [r[0] for r in results]
        assert "n1" in ids
        assert "n3" in ids

    def test_exclude_superseded(self):
        results = ks.knowledge_search(
            self._token_index(), {"n1"}, {"plasma", "density"}, 5, True,
        )
        ids = [r[0] for r in results]
        assert "n1" not in ids
        assert "n3" in ids

    def test_include_superseded(self):
        results = ks.knowledge_search(
            self._token_index(), {"n1"}, {"plasma", "density"}, 5, False,
        )
        ids = [r[0] for r in results]
        assert "n1" in ids

    def test_no_match(self):
        results = ks.knowledge_search(
            self._token_index(), set(), {"xyzzy", "frobnicator"}, 5, True,
        )
        assert results == []

    def test_top_k(self):
        results = ks.knowledge_search(
            self._token_index(), set(), {"plasma", "tokamak"}, 1, True,
        )
        assert len(results) == 1

    def test_score_ordering(self):
        results = ks.knowledge_search(
            self._token_index(), set(), {"plasma", "density"}, 5, True,
        )
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ── get_related_ids ───────────────────────────────────────────


class TestGetRelatedIds:
    def _links(self):
        return {
            "n1": [("n2", "related"), ("n3", "temporal")],
            "n2": [("n4", "related")],
            "n3": [("n4", "supersedes")],
            "n4": [],
        }

    def test_depth_1(self):
        ids = ks.get_related_ids(
            self._links(), "n1", 1, set(), {"n1", "n2", "n3", "n4"},
        )
        assert set(ids) == {"n2", "n3"}

    def test_depth_2(self):
        ids = ks.get_related_ids(
            self._links(), "n1", 2, set(), {"n1", "n2", "n3", "n4"},
        )
        assert "n4" in ids

    def test_edge_type_filter(self):
        ids = ks.get_related_ids(
            self._links(), "n1", 1, {"related"}, {"n1", "n2", "n3", "n4"},
        )
        assert "n2" in ids
        assert "n3" not in ids

    def test_invalid_start(self):
        ids = ks.get_related_ids(
            self._links(), "nonexistent", 1, set(), {"n1", "n2", "n3", "n4"},
        )
        assert ids == []

    def test_no_links(self):
        ids = ks.get_related_ids(
            self._links(), "n4", 2, set(), {"n1", "n2", "n3", "n4"},
        )
        assert ids == []


# ── graph_search ──────────────────────────────────────────────


class TestGraphSearch:
    def test_basic(self):
        token_index = {
            "n1": {"plasma", "temperature", "tokamak"},
            "n2": {"reactor", "tokamak", "maintenance"},
            "n3": {"plasma", "experiment"},
        }
        note_links = {
            "n1": [("n2", "related")],
            "n2": [("n3", "temporal")],
            "n3": [],
        }
        results = ks.graph_search(
            token_index, set(), note_links, {"n1", "n2", "n3"},
            {"plasma", "tokamak"}, 5, 2,
        )
        assert len(results) >= 1
        ids = [r[0] for r in results]
        assert "n1" in ids

    def test_empty_query(self):
        results = ks.graph_search(
            {"n1": {"a", "b"}}, set(), {"n1": []}, {"n1"},
            set(), 5, 2,
        )
        assert results == []

    def test_no_match(self):
        results = ks.graph_search(
            {"n1": {"a", "b"}}, set(), {"n1": []}, {"n1"},
            {"xyz"}, 5, 2,
        )
        assert results == []


# ── RustKnowledgeIndex (pyclass) ──────────────────────────────


class TestRustKnowledgeIndex:
    def _make_index(self):
        token_index = {
            "n1": {"plasma", "temperature", "tokamak"},
            "n2": {"reactor", "tokamak", "maintenance"},
            "n3": {"plasma", "experiment", "new"},
        }
        note_links = {
            "n1": [("n2", "related")],
            "n2": [("n3", "temporal")],
            "n3": [],
        }
        return ks.RustKnowledgeIndex(
            token_index, set(), note_links, {"n1", "n2", "n3"},
        )

    def test_search(self):
        idx = self._make_index()
        results = idx.search({"plasma", "tokamak"}, 5, True)
        assert len(results) >= 1
        ids = [r[0] for r in results]
        assert "n1" in ids

    def test_get_related(self):
        idx = self._make_index()
        ids = idx.get_related("n1", 1, set())
        assert "n2" in ids

    def test_graph_search(self):
        idx = self._make_index()
        results = idx.graph_search({"plasma", "tokamak"}, 5, 2)
        ids = [r[0] for r in results]
        assert "n1" in ids

    def test_search_performance(self):
        token_index = {f"n{i}": {"plasma", "tokamak", f"kw_{i}"} for i in range(500)}
        note_links = {f"n{i}": [(f"n{(i+1)%500}", "related")] for i in range(500)}
        valid_ids = {f"n{i}" for i in range(500)}
        idx = ks.RustKnowledgeIndex(token_index, set(), note_links, valid_ids)
        t0 = time.perf_counter()
        for _ in range(1000):
            idx.search({"plasma", "tokamak"}, 5, True)
        elapsed_us = (time.perf_counter() - t0) / 1000 * 1e6
        assert elapsed_us < 5000, f"RustKnowledgeIndex.search too slow: {elapsed_us:.0f} µs"
