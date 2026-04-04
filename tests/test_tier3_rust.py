# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tier 3 rustification tests (consolidation + reflection + SNN)

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

# ── Rust crate imports (skip entire file if unavailable) ───────

rc = pytest.importorskip("remanentia_consolidation")
stdp = pytest.importorskip("arcane_stdp")


# ── Helpers ────────────────────────────────────────────────────


@dataclass
class FakeNote:
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    title: str = ""
    content: str = ""
    source: str = ""


# ═══════════════════════════════════════════════════════════════
# cluster_traces
# ═══════════════════════════════════════════════════════════════


class TestClusterTraces:
    def test_same_project_close_dates(self):
        traces = [
            ("t1.md", "proj-a", "2026-03-01"),
            ("t2.md", "proj-a", "2026-03-02"),
            ("t3.md", "proj-a", "2026-03-03"),
        ]
        clusters = rc.cluster_traces(traces)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == ["t1.md", "t2.md", "t3.md"]

    def test_same_project_gap_splits(self):
        traces = [
            ("t1.md", "proj-a", "2026-03-01"),
            ("t2.md", "proj-a", "2026-03-02"),
            ("t3.md", "proj-a", "2026-03-10"),
        ]
        clusters = rc.cluster_traces(traces)
        assert len(clusters) == 2

    def test_different_projects_separate(self):
        traces = [
            ("t1.md", "proj-a", "2026-03-01"),
            ("t2.md", "proj-b", "2026-03-01"),
        ]
        clusters = rc.cluster_traces(traces)
        assert len(clusters) == 2

    def test_empty_input(self):
        assert rc.cluster_traces([]) == []

    def test_single_trace(self):
        clusters = rc.cluster_traces([("t1.md", "proj-a", "2026-03-01")])
        assert len(clusters) == 1
        assert clusters[0] == ["t1.md"]

    def test_invalid_dates_no_crash(self):
        traces = [
            ("t1.md", "proj-a", "bad-date"),
            ("t2.md", "proj-a", "also-bad"),
        ]
        clusters = rc.cluster_traces(traces)
        assert len(clusters) >= 1

    def test_matches_python_path(self):
        """Rust result should match Python _cluster_traces logic."""
        from consolidation_engine import _cluster_traces

        trace_data = {
            "a.md": {"project": "proj-x", "date": "2026-03-01"},
            "b.md": {"project": "proj-x", "date": "2026-03-02"},
            "c.md": {"project": "proj-x", "date": "2026-03-10"},
            "d.md": {"project": "proj-y", "date": "2026-03-01"},
        }
        # Run via Python path (Rust is used if available, but let's compare structure)
        py_result = _cluster_traces(trace_data)
        rust_result = rc.cluster_traces(
            [(n, m["project"], m.get("date", "")[:10]) for n, m in trace_data.items()]
        )
        # Both should produce 3 clusters: 2 from proj-x split, 1 from proj-y
        assert len(py_result) == len(rust_result)

    def test_performance(self):
        """1000 traces should cluster in < 5 ms."""
        traces = [
            (f"t{i}.md", f"proj-{i % 10}", f"2026-{1 + i // 30:02d}-{1 + i % 28:02d}")
            for i in range(1000)
        ]
        start = time.perf_counter()
        for _ in range(100):
            rc.cluster_traces(traces)
        elapsed_us = (time.perf_counter() - start) / 100 * 1e6
        assert elapsed_us < 5000, f"cluster_traces too slow: {elapsed_us:.0f} µs"


# ═══════════════════════════════════════════════════════════════
# build_summary_dag
# ═══════════════════════════════════════════════════════════════


class TestBuildSummaryDag:
    def _make_data(self, n: int) -> list[tuple]:
        return [
            (
                f"trace_{i}.md",
                f"2026-03-{1 + i % 28:02d}",
                f"proj-{i % 3}",
                [f"entity_{i}", "shared"],
                [f"Key finding {i}: something important"],
                f"Full text of trace {i} with detailed content.",
            )
            for i in range(n)
        ]

    def test_basic_structure(self):
        data = self._make_data(5)
        nodes = rc.build_summary_dag(data, 4)
        assert len(nodes) > 5  # leaves + at least 1 internal
        # All leaves should be level 0
        leaves = [n for n in nodes if n["level"] == 0]
        assert len(leaves) == 5
        for leaf in leaves:
            assert leaf["node_id"].startswith("L0_")
            assert len(leaf["children"]) == 1

    def test_empty_input(self):
        assert rc.build_summary_dag([], 4) == []

    def test_single_trace(self):
        data = self._make_data(1)
        nodes = rc.build_summary_dag(data, 4)
        assert len(nodes) == 1
        assert nodes[0]["level"] == 0

    def test_fanout_respected(self):
        data = self._make_data(16)
        nodes = rc.build_summary_dag(data, 4)
        # L0: 16, L1: 4, L2: 1 → 21 nodes
        assert len(nodes) == 21

    def test_date_range_propagation(self):
        data = self._make_data(8)
        nodes = rc.build_summary_dag(data, 4)
        root = [n for n in nodes if n["level"] == max(n2["level"] for n2 in nodes)][0]
        assert root["date_range"][0] <= root["date_range"][1]

    def test_entities_truncated(self):
        data = [
            (
                "t.md",
                "2026-03-01",
                "proj",
                [f"e{i}" for i in range(50)],
                ["key line"],
                "text",
            )
        ]
        nodes = rc.build_summary_dag(data, 4)
        assert len(nodes[0]["entities"]) <= 20

    def test_performance(self):
        """100 traces DAG in < 5 ms."""
        data = self._make_data(100)
        start = time.perf_counter()
        for _ in range(50):
            rc.build_summary_dag(data, 4)
        elapsed_us = (time.perf_counter() - start) / 50 * 1e6
        assert elapsed_us < 5000, f"build_summary_dag too slow: {elapsed_us:.0f} µs"


# ═══════════════════════════════════════════════════════════════
# cluster_notes
# ═══════════════════════════════════════════════════════════════


class TestClusterNotes:
    def test_basic_clustering(self):
        notes = [
            (["rust", "pyo3", "speed"], ["remanentia"]),
            (["rust", "pyo3", "maturin"], ["arcane"]),
            (["python", "django"], ["web"]),
        ]
        clusters = rc.cluster_notes(notes, 2)
        assert len(clusters) == 1  # notes 0 and 1 share rust+pyo3
        assert sorted(clusters[0]) == [0, 1]

    def test_no_overlap(self):
        notes = [
            (["a", "b"], ["x"]),
            (["c", "d"], ["y"]),
        ]
        clusters = rc.cluster_notes(notes, 2)
        assert clusters == []

    def test_empty(self):
        assert rc.cluster_notes([], 2) == []

    def test_single_note(self):
        clusters = rc.cluster_notes([(["a"], ["b"])], 2)
        assert clusters == []  # need 2+ for a cluster

    def test_min_overlap_3(self):
        notes = [
            (["a", "b", "c"], []),
            (["a", "b", "d"], []),  # overlap=2, below min_overlap=3
        ]
        clusters = rc.cluster_notes(notes, 3)
        assert clusters == []

    def test_greedy_expansion(self):
        """When note 0 absorbs note 1, its keywords expand to catch note 2."""
        notes = [
            (["a", "b"], []),  # shares a,b with note 1
            (["a", "b", "c"], []),  # absorbed → keywords now include c
            (["c", "b"], []),  # shares c,b with expanded set
        ]
        clusters = rc.cluster_notes(notes, 2)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_matches_python_reflector(self):
        """Verify Rust matches Python _cluster_notes."""
        from reflector import _cluster_notes

        fake_notes = [
            FakeNote(keywords=["stdp", "lif", "spike"], entities=["remanentia"]),
            FakeNote(keywords=["stdp", "lif", "neuron"], entities=["arcane"]),
            FakeNote(keywords=["django", "web"], entities=["parazit"]),
        ]
        py_result = _cluster_notes(fake_notes)
        rust_result = rc.cluster_notes(
            [(list(n.keywords), list(n.entities)) for n in fake_notes], 2
        )
        assert len(py_result) == len(rust_result)

    def test_performance(self):
        """500 notes clustering in < 10 ms."""
        notes = [([f"kw{i % 20}", f"kw{(i + 7) % 20}"], [f"ent{i % 15}"]) for i in range(500)]
        start = time.perf_counter()
        for _ in range(20):
            rc.cluster_notes(notes, 2)
        elapsed_us = (time.perf_counter() - start) / 20 * 1e6
        assert elapsed_us < 10000, f"cluster_notes too slow: {elapsed_us:.0f} µs"


# ═══════════════════════════════════════════════════════════════
# homeostatic_scaling
# ═══════════════════════════════════════════════════════════════


class TestHomeostaticScaling:
    def test_basic_scaling(self):
        w = np.array([[0.5, 0.5, 0.0], [0.1, 0.1, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        w_before = w.copy()
        stdp.homeostatic_scaling(w, 0.3, 0.01)
        # Row 0: mean=0.5, should scale down
        assert w[0, 0] < w_before[0, 0]
        # Row 1: mean=0.1, should scale up
        assert w[1, 0] > w_before[1, 0]
        # Row 2: all zeros, unchanged
        np.testing.assert_array_equal(w[2], [0.0, 0.0, 0.0])

    def test_clipping(self):
        w = np.array([[1.9, 1.9, 0.0]], dtype=np.float32)
        stdp.homeostatic_scaling(w, 10.0, 1.0)
        assert w.max() <= 2.0

    def test_skip_inactive_rows(self):
        w = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
        w_before = w.copy()
        stdp.homeostatic_scaling(w, 0.3, 0.01)
        # Row 0: all zero → unchanged
        np.testing.assert_array_equal(w[0], w_before[0])
        # Row 1: only 1 active → skip (need at least 2)
        np.testing.assert_array_equal(w[1], w_before[1])

    def test_matches_python(self):
        """Verify Rust matches Python _homeostatic_scaling."""
        rng = np.random.RandomState(42)
        w_rust = rng.rand(50, 50).astype(np.float32) * 0.6
        w_py = w_rust.copy()

        stdp.homeostatic_scaling(w_rust, 0.3, 0.01)
        # Python fallback (force it by temporarily hiding import)
        for i in range(w_py.shape[0]):
            row = w_py[i]
            active = row > 0.001
            if active.sum() < 2:
                continue
            current_mean = row[active].mean()
            if current_mean < 0.001:
                continue
            scale = 1.0 + 0.01 * (0.3 / current_mean - 1.0)
            w_py[i] = np.clip(w_py[i] * scale, 0, 2.0)

        np.testing.assert_allclose(w_rust, w_py, rtol=1e-5)

    def test_large_matrix(self):
        w = np.random.rand(500, 500).astype(np.float32) * 0.5
        stdp.homeostatic_scaling(w, 0.3, 0.01)
        assert w.max() <= 2.0
        assert w.min() >= 0.0

    def test_performance(self):
        """500x500 homeostatic scaling in < 10 ms."""
        w = np.random.rand(500, 500).astype(np.float32) * 0.5
        start = time.perf_counter()
        for _ in range(100):
            stdp.homeostatic_scaling(w, 0.3, 0.01)
        elapsed_us = (time.perf_counter() - start) / 100 * 1e6
        assert elapsed_us < 10000, f"homeostatic_scaling too slow: {elapsed_us:.0f} µs"

    def test_zero_rate(self):
        w = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
        w_before = w.copy()
        stdp.homeostatic_scaling(w, 0.3, 0.0)
        np.testing.assert_array_equal(w, w_before)

    def test_idempotent_at_target(self):
        """If mean already equals target, weights should barely change."""
        w = np.array([[0.3, 0.3, 0.0]], dtype=np.float32)
        w_before = w.copy()
        stdp.homeostatic_scaling(w, 0.3, 0.01)
        np.testing.assert_allclose(w, w_before, atol=1e-6)
