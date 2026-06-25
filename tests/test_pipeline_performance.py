# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Pipeline performance tests

"""End-to-end pipeline performance tests with documented budgets.

Each test measures wall-clock time for a realistic workload and asserts
it stays within budget. Budgets are generous (10× headroom over measured
local performance) to avoid CI flakiness while still catching regressions.

Pipeline architecture under test:

    text → parse_dates (temporal_graph)
         → _regex_entities (entity_extractor)
         → extract_answer (answer_extractor)
         → normalize_answer (answer_normalizer)
         → fuzzy_match (answer_extractor)
         → classify_fact_type (fact_decomposer)
         → split_sentences (fact_decomposer)
         → KnowledgeStore.add_note (knowledge_store)
         → TemporalGraph.query_temporal (temporal_graph)
         → MemoryIndex.search (memory_index)
         → recall (memory_recall)
         → consolidate (consolidation_engine)
         → observe_once (observer)
         → reflect_once (reflector)
         → handle_recall (mcp_server)
         → resolve_backend (llm_backend)
         → write_config (llm_setup)
"""

from __future__ import annotations

import math
import os
import time
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

# Slow local hardware (NTFS-backed venv, shared laptop disks) can blow
# the 10× headroom CI uses. Operators can set
# ``REMANENTIA_PERF_BUDGET_SCALE=<N>`` to multiply every budget by N;
# CI keeps the default 1.0. This does not hide regressions — CI is the
# authoritative check — it only skips failures attributable to
# local-HW jitter. The scale is clamped to [0.1, 100] for safety.
_BUDGET_SCALE = max(0.1, min(100.0, float(os.environ.get("REMANENTIA_PERF_BUDGET_SCALE", "1.0"))))


def _budget(ms: float) -> float:
    """Return the effective budget ceiling after applying BUDGET_SCALE."""
    return ms * _BUDGET_SCALE


# ── Test data ────────────────────────────────────────────────


SHORT_TEXT = "The review was on March 15, 2026. Score improved to 88.5%."

LONG_TEXT = (
    "The team had a quarterly review on March 15, 2026. "
    "The LOCOMO score improved from 81.2% to 88.5% after implementing "
    "hybrid retrieval with BM25 v3.14.0. Fixed STDP bug in snn_backend.py "
    "for sc-neurocore because the LTD mask was wrong. I love hiking. "
) * 100  # ~47K chars

QUERY = "When did the score improve and what was the new value?"
REF_DATE = date(2026, 3, 30)


def _timed(fn, *args, n=100, **kwargs):
    """Run fn n times, return (avg_ms, result_of_last_call)."""
    # warmup
    result = fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n):
        result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000 / n
    return elapsed, result


# ── Per-module performance ───────────────────────────────────


class TestParseDatesPerformance:
    """temporal_graph.parse_dates — budget: <5ms for 47K chars."""

    def test_parse_dates_budget(self):
        from temporal_graph import parse_dates

        ms, dates = _timed(parse_dates, LONG_TEXT, REF_DATE)
        assert ms < _budget(50), f"parse_dates: {ms:.2f}ms exceeds 50ms budget"
        assert len(dates) >= 1
        assert "2026-03-15" in dates


class TestExtractAnswerPerformance:
    """answer_extractor.extract_answer — budget: <2ms for 47K chars."""

    def test_extract_answer_budget(self):
        from answer_extractor import extract_answer

        ms, answer = _timed(extract_answer, QUERY, LONG_TEXT)
        assert ms < _budget(20), f"extract_answer: {ms:.2f}ms exceeds 20ms budget"
        assert answer is not None

    def test_fuzzy_match_budget(self):
        from answer_extractor import fuzzy_match

        ms, result = _timed(fuzzy_match, "March 15, 2026", "march 15 2026", 0.5)
        assert ms < _budget(1), f"fuzzy_match: {ms:.2f}ms exceeds 1ms budget"

    def test_normalize_number_budget(self):
        from answer_extractor import normalize_number

        ms, result = _timed(normalize_number, "forty-two")
        assert ms < _budget(1), f"normalize_number: {ms:.2f}ms exceeds 1ms budget"
        assert result == "42"


class TestAnswerNormalizerPerformance:
    """answer_normalizer — budget: <1ms per call."""

    def test_normalize_answer_budget(self):
        from answer_normalizer import normalize_answer

        ms, result = _timed(normalize_answer, "Likely yes, because she enjoys reading")
        assert ms < _budget(1), f"normalize_answer: {ms:.2f}ms exceeds 1ms budget"
        assert result == "likely yes"

    def test_answers_match_budget(self):
        from answer_normalizer import answers_match

        ms, result = _timed(answers_match, "Likely yes", "Yes", 0.25)
        assert ms < _budget(2), f"answers_match: {ms:.2f}ms exceeds 2ms budget"
        assert result is True


class TestEntityExtractorPerformance:
    """entity_extractor._regex_entities — budget: <5ms for 47K chars."""

    def test_regex_entities_budget(self):
        from entity_extractor import _regex_entities

        ms, entities = _timed(_regex_entities, LONG_TEXT)
        assert ms < _budget(50), f"regex_entities: {ms:.2f}ms exceeds 50ms budget"
        assert len(entities) >= 1

    def test_extract_relations_budget(self):
        from entity_extractor import _regex_entities, extract_relations

        entities = _regex_entities(SHORT_TEXT)
        if len(entities) >= 2:
            ms, rels = _timed(extract_relations, SHORT_TEXT, entities)
            assert ms < _budget(5), f"extract_relations: {ms:.2f}ms exceeds 5ms budget"


class TestFactDecomposerPerformance:
    """fact_decomposer.decompose_sessions — budget: <50ms for 10 turns."""

    def test_decompose_budget(self):
        from fact_decomposer import decompose_sessions

        sessions = [
            [
                {"role": "user", "content": f"I did thing {i} on March {i + 1}, 2026."}
                for i in range(10)
            ]
        ]
        # First call may trigger lazy model loading; measure warm path
        decompose_sessions(sessions)  # warmup (may hit network)
        ms, facts = _timed(decompose_sessions, sessions, n=10)
        # Budget generous: model loading + network retries can spike first runs
        assert ms < _budget(5000), f"decompose_sessions (warm): {ms:.2f}ms exceeds 5000ms budget"
        assert len(facts) >= 5


class TestKnowledgeStorePerformance:
    """knowledge_store.add_note — budget: <5ms per note."""

    def test_add_note_budget(self):
        from knowledge_store import KnowledgeStore

        store = KnowledgeStore()
        ms, note = _timed(
            store.add_note,
            "BM25 accuracy measured at 88.5% on LOCOMO benchmark.",
            source="perf.md",
            n=50,
        )
        assert ms < _budget(50), f"add_note: {ms:.2f}ms exceeds 50ms budget"

    def test_search_budget(self):
        from knowledge_store import KnowledgeStore

        store = KnowledgeStore()
        for i in range(20):
            store.add_note(f"Fact {i}: unique content about topic {i}.", source=f"s{i}.md")
        ms, results = _timed(store.search, "topic 5", n=50)
        assert ms < _budget(20), f"search: {ms:.2f}ms exceeds 20ms budget"


class TestTemporalGraphPerformance:
    """temporal_graph operations — budget: <10ms for 100 events."""

    def test_add_events_budget(self):
        from temporal_graph import TemporalGraph, TemporalEvent

        tg = TemporalGraph()
        events = [
            TemporalEvent(date=f"2026-03-{i % 28 + 1:02d}", text=f"Event {i}", source="p.md")
            for i in range(100)
        ]
        ms, _ = _timed(tg.add_events, events, n=10)
        assert ms < _budget(100), f"add_events(100): {ms:.2f}ms exceeds 100ms budget"

    def test_query_temporal_budget(self):
        from temporal_graph import TemporalGraph, TemporalEvent

        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(
                    date=f"2026-03-{i % 28 + 1:02d}", text=f"Event {i} with data", source="p.md"
                )
                for i in range(100)
            ]
        )
        ms, results = _timed(tg.query_temporal, "data after 2026-03-15", 5)
        assert ms < _budget(10), f"query_temporal: {ms:.2f}ms exceeds 10ms budget"


class TestLLMBackendPerformance:
    """llm_backend.resolve_backend — budget: <5ms."""

    def test_resolve_backend_budget(self):
        from llm_backend import resolve_backend

        ms, backend = _timed(resolve_backend, "none")
        assert ms < _budget(5), f"resolve_backend: {ms:.2f}ms exceeds 5ms budget"

    def test_null_backend_budget(self):
        from llm_backend import NullBackend

        b = NullBackend()
        ms, result = _timed(b.complete, "test")
        assert ms < _budget(0.5), f"NullBackend.complete: {ms:.2f}ms exceeds 0.5ms budget"
        assert result is None


class TestLLMSetupPerformance:
    """llm_setup.write_config — budget: <5ms."""

    def test_write_config_budget(self, tmp_path):
        from llm_setup import write_config

        path = tmp_path / "perf.toml"
        ms, result = _timed(write_config, path=path, n=50)
        assert ms < _budget(10), f"write_config: {ms:.2f}ms exceeds 10ms budget"


class TestObserverPerformance:
    """observer.observe_once — budget: <100ms for 10 files."""

    def test_observe_once_budget(self, tmp_path):
        from observer import ObserverState, observe_once

        d = tmp_path / "traces"
        d.mkdir()
        for i in range(10):
            (d / f"trace_{i}.md").write_text(
                f"We decided to implement feature {i} because of accuracy improvements.\n",
                encoding="utf-8",
            )
        store_path = tmp_path / "notes.jsonl"
        triggers_path = tmp_path / "triggers.jsonl"
        with (
            patch.dict(os.environ, {"REMANENTIA_ARCANE_CE_DISABLE": "1"}),
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            state = ObserverState()
            t0 = time.perf_counter()
            observe_once(state, {"traces": d})
            ms = (time.perf_counter() - t0) * 1000
        assert ms < _budget(500), f"observe_once(10 files): {ms:.2f}ms exceeds 500ms budget"


class TestReflectorPerformance:
    """reflector.reflect_once — budget: <100ms (no LLM)."""

    def test_reflect_once_budget(self):
        from reflector import reflect_once

        t0 = time.perf_counter()
        reflect_once(days=1, use_llm=False)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < _budget(500), f"reflect_once: {ms:.2f}ms exceeds 500ms budget"


# ── Full pipeline performance ─────────────────────────────────


class TestFullPipelinePerformance:
    """End-to-end pipeline: text → all modules → answer.

    Budget: <100ms for regex-only path (no LLM).
    """

    def test_regex_pipeline_budget(self):
        """Full regex pipeline on 47K char document."""
        from temporal_graph import parse_dates
        from entity_extractor import _regex_entities
        from answer_extractor import extract_answer
        from answer_normalizer import normalize_answer, answers_match

        t0 = time.perf_counter()
        for _ in range(10):
            dates = parse_dates(LONG_TEXT, REF_DATE)
            entities = _regex_entities(LONG_TEXT)
            answer = extract_answer(QUERY, LONG_TEXT)
            if answer:
                normalize_answer(answer)
                answers_match(answer, "88.5%", 0.25)
        ms = (time.perf_counter() - t0) * 1000 / 10

        assert ms < _budget(200), f"Full regex pipeline: {ms:.2f}ms exceeds 200ms budget"
        assert len(dates) >= 1
        assert len(entities) >= 1
        assert answer is not None

    def test_memory_index_search_budget(self, tmp_path):
        """Build index + search on small corpus."""
        from memory_index import MemoryIndex, Document, _tokenize, _token_counts

        idx = MemoryIndex()
        idx.documents = []
        idx.paragraph_index = []
        idx.paragraph_tokens = []
        idx.paragraph_token_counts = []
        idx.paragraph_types = []
        idx._inverted_index = {}
        idx._df = {}
        idx.idf = {}
        idx._built = True
        idx._rust_bm25_dirty = True

        # Build small corpus
        texts = [
            "STDP learning was set to 0.0 weight after experiments.",
            "LOCOMO scored 81.2% accuracy on the benchmark.",
            "Alice loves pottery and hiking on weekends.",
            "BM25 retrieval achieves 85.7% P@1.",
            "Director-AI sent 265 emails to companies.",
        ] * 20  # 100 paragraphs

        for i, text in enumerate(texts):
            doc = Document(name=f"doc_{i}.md", source="test", path=f"doc_{i}.md", paragraphs=[text])
            idx.documents.append(doc)
            tokens = set(_tokenize(text))
            token_counts = _token_counts(_tokenize(text))
            p_idx = len(idx.paragraph_tokens)
            idx.paragraph_index.append((i, 0))
            idx.paragraph_tokens.append(tokens)
            idx.paragraph_token_counts.append(token_counts)
            idx.paragraph_types.append("discussion")
            for t in tokens:
                idx._df[t] = idx._df.get(t, 0) + 1
                if t not in idx._inverted_index:
                    idx._inverted_index[t] = []
                idx._inverted_index[t].append(p_idx)

        n = len(idx.paragraph_tokens)
        idx.idf = {t: math.log(1 + n / (1 + c)) for t, c in idx._df.items()}
        idx._para_lengths = np.array([len(t) for t in idx.paragraph_tokens], dtype=np.float32)
        idx._avg_dl = float(np.mean(idx._para_lengths))

        # Measure search
        ms, results = _timed(idx.search, "LOCOMO accuracy benchmark", 3, n=50)
        assert ms < _budget(20), f"MemoryIndex.search(100 paras): {ms:.2f}ms exceeds 20ms budget"
        assert len(results) > 0


# ── Rust vs Python benchmarks ─────────────────────────────────


class TestRustVsPythonBenchmarks:
    """Measure Rust-accelerated path vs Python fallback for every wired module.

    Rust modules:
    - remanentia_temporal → temporal_graph.parse_dates, date_normalizer
    - remanentia_answer_extractor → answer_extractor.extract_answer, fuzzy_match, normalize_number
    - remanentia_answer_normalizer → answer_normalizer.normalize_answer
    - remanentia_entity_extractor → entity_extractor._regex_entities
    - remanentia_search → memory_index BM25 (if installed)
    """

    @staticmethod
    def _measure_both(py_fn, rust_fn, py_args, rust_args=None, n=200):
        """Measure Python and Rust paths, return (py_ms, rust_ms, speedup)."""
        if rust_args is None:
            rust_args = py_args
        # Warmup
        py_fn(*py_args)
        rust_fn(*rust_args)
        # Python
        t0 = time.perf_counter()
        for _ in range(n):
            py_fn(*py_args)
        py_ms = (time.perf_counter() - t0) * 1000 / n
        # Rust
        t0 = time.perf_counter()
        for _ in range(n):
            rust_fn(*rust_args)
        rust_ms = (time.perf_counter() - t0) * 1000 / n
        speedup = py_ms / rust_ms if rust_ms > 0 else 0
        return py_ms, rust_ms, speedup

    def test_parse_dates_rust_vs_python(self):
        """temporal_graph.parse_dates: Rust remanentia_temporal vs Python."""
        try:
            import remanentia_temporal as rt
        except ImportError:
            pytest.skip("remanentia_temporal not installed")

        # Python path — reimport with Rust blocked
        import sys
        import importlib

        saved = sys.modules.get("remanentia_temporal")
        sys.modules["remanentia_temporal"] = None
        import temporal_graph

        importlib.reload(temporal_graph)
        py_parse = temporal_graph.parse_dates

        sys.modules["remanentia_temporal"] = saved
        importlib.reload(temporal_graph)

        py_ms, rust_ms, speedup = self._measure_both(
            py_parse,
            lambda t, r: rt.parse_dates(t, r),
            (LONG_TEXT, REF_DATE),
            (LONG_TEXT, REF_DATE.isoformat()),
        )
        print(f"\n  parse_dates: Python={py_ms:.3f}ms Rust={rust_ms:.3f}ms Speedup={speedup:.1f}×")
        # On cached/warm paths, Rust and Python are close; just verify no major regression
        assert rust_ms < py_ms * 3, f"Rust unexpectedly slow (py={py_ms:.3f} rust={rust_ms:.3f})"

    def test_extract_answer_rust_vs_python(self):
        """answer_extractor.extract_answer: Rust vs Python."""
        try:
            import remanentia_answer_extractor as rae
        except ImportError:
            pytest.skip("remanentia_answer_extractor not installed")

        import sys
        import importlib

        saved = sys.modules.get("remanentia_answer_extractor")
        sys.modules["remanentia_answer_extractor"] = None
        import answer_extractor

        importlib.reload(answer_extractor)
        py_extract = answer_extractor.extract_answer

        sys.modules["remanentia_answer_extractor"] = saved
        importlib.reload(answer_extractor)

        py_ms, rust_ms, speedup = self._measure_both(
            py_extract,
            rae.extract_answer,
            (QUERY, LONG_TEXT),
        )
        print(
            f"\n  extract_answer: Python={py_ms:.3f}ms Rust={rust_ms:.3f}ms Speedup={speedup:.1f}×"
        )
        assert rust_ms < py_ms * 3, f"Rust unexpectedly slow (py={py_ms:.3f} rust={rust_ms:.3f})"

    def test_normalize_answer_rust_vs_python(self):
        """answer_normalizer.normalize_answer: Rust vs Python."""
        try:
            import remanentia_answer_normalizer as ran
        except ImportError:
            pytest.skip("remanentia_answer_normalizer not installed")

        import sys
        import importlib

        saved = sys.modules.get("remanentia_answer_normalizer")
        sys.modules["remanentia_answer_normalizer"] = None
        import answer_normalizer

        importlib.reload(answer_normalizer)
        py_norm = answer_normalizer.normalize_answer

        sys.modules["remanentia_answer_normalizer"] = saved
        importlib.reload(answer_normalizer)

        text = "Likely yes, because she enjoys reading and outdoor activities"
        py_ms, rust_ms, speedup = self._measure_both(
            py_norm,
            ran.normalize_answer,
            (text,),
            n=5000,
        )
        print(
            f"\n  normalize_answer: Python={py_ms:.4f}ms Rust={rust_ms:.4f}ms Speedup={speedup:.1f}×"
        )
        # Small strings may not show speedup due to PyO3 overhead
        assert rust_ms < 1, f"Rust should be under 1ms (got {rust_ms:.4f}ms)"

    def test_regex_entities_rust_vs_python(self):
        """entity_extractor._regex_entities: Rust vs Python."""
        try:
            import remanentia_entity_extractor as ree
        except ImportError:
            pytest.skip("remanentia_entity_extractor not installed")

        import sys
        import importlib

        saved = sys.modules.get("remanentia_entity_extractor")
        sys.modules["remanentia_entity_extractor"] = None
        import entity_extractor

        importlib.reload(entity_extractor)
        py_ents = entity_extractor._regex_entities

        sys.modules["remanentia_entity_extractor"] = saved
        importlib.reload(entity_extractor)

        py_ms, rust_ms, speedup = self._measure_both(
            lambda t: py_ents(t),
            lambda t: ree.regex_entities(t),
            (LONG_TEXT,),
        )
        print(
            f"\n  regex_entities: Python={py_ms:.3f}ms Rust={rust_ms:.3f}ms Speedup={speedup:.1f}×"
        )

    def test_full_pipeline_rust_vs_python(self):
        """Full regex pipeline: Rust-accelerated vs pure Python."""
        try:
            import remanentia_temporal as rt
            import remanentia_answer_extractor as rae
            import remanentia_answer_normalizer as ran
            import remanentia_entity_extractor as ree
        except ImportError:
            pytest.skip("Rust modules not installed")

        ref_str = REF_DATE.isoformat()

        def rust_pipeline():
            dates = rt.parse_dates(LONG_TEXT, ref_str)
            ents = ree.regex_entities(LONG_TEXT)
            ans = rae.extract_answer(QUERY, LONG_TEXT)
            if ans:
                ran.normalize_answer(ans)
            return dates, ents, ans

        # Python pipeline — block all Rust imports
        import sys
        import importlib

        rust_mods = [
            "remanentia_temporal",
            "remanentia_answer_extractor",
            "remanentia_answer_normalizer",
            "remanentia_entity_extractor",
        ]
        saved = {m: sys.modules.get(m) for m in rust_mods}
        for m in rust_mods:
            sys.modules[m] = None

        import temporal_graph
        import answer_extractor
        import answer_normalizer
        import entity_extractor

        importlib.reload(temporal_graph)
        importlib.reload(answer_extractor)
        importlib.reload(answer_normalizer)
        importlib.reload(entity_extractor)

        py_parse = temporal_graph.parse_dates
        py_extract = answer_extractor.extract_answer
        py_norm = answer_normalizer.normalize_answer
        py_ents = entity_extractor._regex_entities

        for m in rust_mods:
            sys.modules[m] = saved[m]
        importlib.reload(temporal_graph)
        importlib.reload(answer_extractor)
        importlib.reload(answer_normalizer)
        importlib.reload(entity_extractor)

        def python_pipeline():
            dates = py_parse(LONG_TEXT, REF_DATE)
            ents = py_ents(LONG_TEXT)
            ans = py_extract(QUERY, LONG_TEXT)
            if ans:
                py_norm(ans)
            return dates, ents, ans

        # Warmup
        rust_pipeline()
        python_pipeline()

        n = 100
        t0 = time.perf_counter()
        for _ in range(n):
            python_pipeline()
        py_ms = (time.perf_counter() - t0) * 1000 / n

        t0 = time.perf_counter()
        for _ in range(n):
            rust_pipeline()
        rust_ms = (time.perf_counter() - t0) * 1000 / n

        speedup = py_ms / rust_ms if rust_ms > 0 else 0
        print(
            f"\n  FULL PIPELINE: Python={py_ms:.2f}ms Rust={rust_ms:.2f}ms Speedup={speedup:.1f}×"
        )
        # Speedup varies by workload size; on short texts PyO3 overhead can dominate
        # On 47K+ texts with many regex matches, expect consistent improvement
        assert rust_ms < py_ms * 2, (
            f"Rust should not be 2× slower than Python (py={py_ms:.2f} rust={rust_ms:.2f})"
        )

    def test_rust_python_results_identical(self):
        """Verify Rust and Python produce identical results."""
        try:
            import remanentia_temporal as rt
            import remanentia_answer_extractor as rae
        except ImportError:
            pytest.skip("Rust modules not installed")

        import sys
        import importlib

        # Get Python results
        saved_t = sys.modules.get("remanentia_temporal")
        saved_a = sys.modules.get("remanentia_answer_extractor")
        sys.modules["remanentia_temporal"] = None
        sys.modules["remanentia_answer_extractor"] = None

        import temporal_graph
        import answer_extractor

        importlib.reload(temporal_graph)
        importlib.reload(answer_extractor)

        py_dates = temporal_graph.parse_dates(SHORT_TEXT, REF_DATE)
        py_answer = answer_extractor.extract_answer(QUERY, SHORT_TEXT)

        sys.modules["remanentia_temporal"] = saved_t
        sys.modules["remanentia_answer_extractor"] = saved_a
        importlib.reload(temporal_graph)
        importlib.reload(answer_extractor)

        # Get Rust results
        rust_dates = rt.parse_dates(SHORT_TEXT, REF_DATE.isoformat())
        rust_answer = rae.extract_answer(QUERY, SHORT_TEXT)

        assert py_dates == rust_dates, f"Dates differ: py={py_dates} rust={rust_dates}"
        assert py_answer == rust_answer, f"Answers differ: py={py_answer} rust={rust_answer}"


class TestDocumentedPerformanceSummary:
    """Collect and print all performance metrics as documentation."""

    def test_print_summary(self, capsys):
        from temporal_graph import parse_dates
        from entity_extractor import _regex_entities
        from answer_extractor import extract_answer, fuzzy_match, normalize_number
        from answer_normalizer import normalize_answer, answers_match
        from knowledge_store import KnowledgeStore
        from llm_backend import resolve_backend, NullBackend

        metrics = []

        ms, _ = _timed(parse_dates, LONG_TEXT, REF_DATE)
        metrics.append(("parse_dates (47K)", ms))

        ms, _ = _timed(_regex_entities, LONG_TEXT)
        metrics.append(("regex_entities (47K)", ms))

        ms, _ = _timed(extract_answer, QUERY, LONG_TEXT)
        metrics.append(("extract_answer (47K)", ms))

        ms, _ = _timed(normalize_answer, "Likely yes, because reading")
        metrics.append(("normalize_answer", ms))

        ms, _ = _timed(answers_match, "yes", "Yes", 0.25)
        metrics.append(("answers_match", ms))

        ms, _ = _timed(fuzzy_match, "March 15", "march 15, 2026", 0.5)
        metrics.append(("fuzzy_match", ms))

        ms, _ = _timed(normalize_number, "forty-two")
        metrics.append(("normalize_number", ms))

        ms, _ = _timed(resolve_backend, "none")
        metrics.append(("resolve_backend", ms))

        b = NullBackend()
        ms, _ = _timed(b.complete, "test")
        metrics.append(("NullBackend.complete", ms))

        store = KnowledgeStore()
        ms, _ = _timed(store.add_note, "Test note for perf.", source="p.md", n=20)
        metrics.append(("KnowledgeStore.add_note", ms))

        print("\n" + "=" * 60)
        print("REMANENTIA PIPELINE PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'Component':<35} {'Avg (ms)':>10}")
        print("-" * 47)
        for name, ms_val in metrics:
            print(f"{name:<35} {ms_val:>9.3f}ms")
        print("-" * 47)
        total = sum(m for _, m in metrics)
        print(f"{'TOTAL (regex pipeline)':<35} {total:>9.3f}ms")
        print("=" * 60)

        # Rust vs Python comparison (if Rust modules installed)
        try:
            import remanentia_temporal as rt
            import remanentia_answer_extractor as rae
            import remanentia_entity_extractor as ree

            print("\n" + "=" * 70)
            print("RUST vs PYTHON ACCELERATION")
            print("=" * 70)
            print(f"{'Function':<30} {'Python':>10} {'Rust':>10} {'Speedup':>10}")
            print("-" * 62)

            rust_benches = [
                (
                    "parse_dates (47K)",
                    parse_dates,
                    lambda: rt.parse_dates(LONG_TEXT, REF_DATE.isoformat()),
                    (LONG_TEXT, REF_DATE),
                ),
                (
                    "extract_answer (47K)",
                    extract_answer,
                    lambda: rae.extract_answer(QUERY, LONG_TEXT),
                    (QUERY, LONG_TEXT),
                ),
                (
                    "regex_entities (47K)",
                    _regex_entities,
                    lambda: ree.regex_entities(LONG_TEXT),
                    (LONG_TEXT,),
                ),
            ]
            for name, py_fn, rust_fn, py_args in rust_benches:
                py_ms, _ = _timed(py_fn, *py_args)
                t0 = time.perf_counter()
                for _ in range(100):
                    rust_fn()
                rust_ms = (time.perf_counter() - t0) * 1000 / 100
                sp = py_ms / rust_ms if rust_ms > 0 else 0
                print(f"{name:<30} {py_ms:>9.3f}ms {rust_ms:>9.3f}ms {sp:>9.1f}×")

            print("=" * 70)
        except ImportError:
            print("\n(Rust modules not installed — skipping Rust comparison)")


# ── v0.4 features: end-to-end + benchmarks ──────────────────────


class TestEndToEndNewFeatures:
    """End-to-end tests exercising all 7 v0.4 features through the pipeline.

    Verifies that features are not decorative — every new component is
    wired into the infrastructure and produces measurable, correct output.
    """

    def test_e2e_extended_fact_types_through_full_pipeline(self):
        """Sessions → decompose → FactIndex → ArcaneRetriever → all 9 types survive."""
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [
                {
                    "role": "user",
                    "content": "We decided to adopt BM25 as primary retrieval engine.",
                },
                {"role": "assistant", "content": "Actually the STDP measurements were wrong."},
                {"role": "user", "content": "Always verify test coverage before merging to main."},
                {
                    "role": "assistant",
                    "content": "The deadline is April 15 and I committed to deliver.",
                },
                {
                    "role": "user",
                    "content": "To fix this run the following pytest command with -x.",
                },
                {"role": "assistant", "content": "I plan to add temporal reasoning improvements."},
                {
                    "role": "user",
                    "content": "I prefer dark mode and minimal user interface designs.",
                },
                {"role": "assistant", "content": "She started working at Google in January 2024."},
                {"role": "user", "content": "The quarterly review meeting was held last Friday."},
            ]
        ]
        ar = ArcaneRetriever(
            sessions,
            session_dates=["2024-06-01"],
            reference_date="2024-06-15",
            recency_half_life_days=30,
        )
        # All 9 types must exist in decomposed facts
        types_found = {f.fact_type for f in ar.facts}
        expected = {
            "decision",
            "correction",
            "principle",
            "commitment",
            "skill",
            "plan",
            "preference",
            "state",
            "event",
        }
        assert types_found == expected, f"Missing types: {expected - types_found}"

        # Retrieve decision — should return decision-type fact first
        results = ar.retrieve("what did we decide about retrieval", "general", top_k=5)
        assert len(results) > 0
        assert results[0].fact.fact_type == "decision"

        # Retrieve correction — should surface correction
        results = ar.retrieve("what was wrong with STDP", "general", top_k=5)
        assert any(r.fact.fact_type == "correction" for r in results[:3])

        # Build context — should produce LLM-ready string
        ctx = ar.build_context("retrieval decision", results)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_e2e_recency_decay_changes_ranking(self):
        """Recent facts must outrank old facts with identical content."""
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [{"role": "user", "content": "BM25 retrieval accuracy measured at 74.7 percent."}],
            [{"role": "user", "content": "BM25 retrieval accuracy measured at 74.7 percent."}],
        ]
        # Old session (Jan) vs recent session (Jun)
        dates = ["2024-01-01", "2024-06-01"]

        ar = ArcaneRetriever(
            sessions,
            session_dates=dates,
            reference_date="2024-06-15",
            recency_half_life_days=30,
        )
        results = ar.retrieve("BM25 accuracy", "general", top_k=10)
        session_scores = {}
        for r in results:
            session_scores.setdefault(r.fact.session_idx, []).append(r.rrf_score)

        if 0 in session_scores and 1 in session_scores:
            assert max(session_scores[1]) > max(session_scores[0])

    def test_e2e_consolidation_writes_lifecycle_and_dag(self, tmp_path):
        """consolidate() → semantic memories with validity_state + summary DAG."""
        import consolidation_engine as ce

        orig = {
            "TRACES_DIR": ce.TRACES_DIR,
            "SEMANTIC_DIR": ce.SEMANTIC_DIR,
            "CONSOLIDATION_DIR": ce.CONSOLIDATION_DIR,
            "GRAPH_DIR": ce.GRAPH_DIR,
            "ENTITIES_PATH": ce.ENTITIES_PATH,
            "RELATIONS_PATH": ce.RELATIONS_PATH,
            "CLUSTERS_PATH": ce.CLUSTERS_PATH,
            "PENDING_PATH": ce.PENDING_PATH,
            "LAST_RUN_PATH": ce.LAST_RUN_PATH,
            "SUMMARY_DAG_PATH": ce.SUMMARY_DAG_PATH,
        }
        traces = tmp_path / "traces"
        traces.mkdir()
        sem = tmp_path / "semantic"
        sem.mkdir()
        con = tmp_path / "consolidation"
        con.mkdir()
        graph = tmp_path / "graph"
        graph.mkdir()

        (traces / "2024-01-01_decision.md").write_text(
            "# Decision\n\nWe decided to remove SNN from retrieval.\n\n"
            "We found that accuracy improved to 81 percent.\n",
            encoding="utf-8",
        )
        (traces / "2024-01-02_technical.md").write_text(
            "# Technical\n\nFixed BM25 scoring bug in memory_index.\n\n"
            "Always verify test coverage before pushing.\n",
            encoding="utf-8",
        )

        ce.TRACES_DIR = traces
        ce.SEMANTIC_DIR = sem
        ce.CONSOLIDATION_DIR = con
        ce.GRAPH_DIR = graph
        ce.ENTITIES_PATH = graph / "entities.jsonl"
        ce.RELATIONS_PATH = graph / "relations.jsonl"
        ce.CLUSTERS_PATH = graph / "clusters.json"
        ce.PENDING_PATH = con / "pending.json"
        ce.LAST_RUN_PATH = con / "last.json"
        ce.SUMMARY_DAG_PATH = con / "dag.json"

        try:
            result = ce.consolidate(force=True)
            assert result["traces_processed"] >= 2
            assert result["memories_written"] >= 1

            # Verify lifecycle state in written memories
            for md in sem.rglob("*.md"):
                text = md.read_text(encoding="utf-8")
                assert "validity_state: active" in text
                assert "last_accessed:" in text

            # Verify DAG was created
            dag_path = con / "dag.json"
            assert dag_path.exists()
            import json

            dag = json.loads(dag_path.read_text(encoding="utf-8"))
            assert len(dag) >= 2  # at least 2 leaf nodes
            leaves = [n for n in dag if n["level"] == 0]
            assert len(leaves) >= 2

            # DAG should be searchable
            from consolidation_engine import search_summary_dag

            results = search_summary_dag(dag, "BM25 accuracy")
            assert len(results) >= 1
        finally:
            for k, v in orig.items():
                setattr(ce, k, v)

    def test_e2e_capacity_report_reflects_consolidation(self, tmp_path):
        """After consolidation, capacity_report should show the new memories."""
        import consolidation_engine as ce

        orig_sem = ce.SEMANTIC_DIR
        sem = tmp_path / "semantic"
        cat = sem / "decision"
        cat.mkdir(parents=True)
        ce.SEMANTIC_DIR = sem

        try:
            (cat / "d1.md").write_text(
                "---\nvalidity_state: active\n---\n" + "Decision content.\n" * 100,
                encoding="utf-8",
            )
            report = ce.capacity_report()
            assert "decision" in report
            assert report["decision"]["file_count"] == 1
            assert report["decision"]["chars"] > 0
            assert report["decision"]["state_counts"].get("active", 0) == 1
        finally:
            ce.SEMANTIC_DIR = orig_sem

    def test_e2e_heartbeat_runs_full_cycle(self, tmp_path):
        """Heartbeat should observe + consolidate + age + report capacity."""
        import consolidation_engine as ce
        from observer import ObserverState, heartbeat

        orig = {
            k: getattr(ce, k)
            for k in [
                "TRACES_DIR",
                "SEMANTIC_DIR",
                "CONSOLIDATION_DIR",
                "GRAPH_DIR",
                "ENTITIES_PATH",
                "RELATIONS_PATH",
                "CLUSTERS_PATH",
                "PENDING_PATH",
                "LAST_RUN_PATH",
                "SUMMARY_DAG_PATH",
            ]
        }
        traces = tmp_path / "traces"
        traces.mkdir()
        sem = tmp_path / "semantic"
        sem.mkdir()
        con = tmp_path / "consolidation"
        con.mkdir()
        graph = tmp_path / "graph"
        graph.mkdir()

        (traces / "2024-01-01_trace.md").write_text(
            "# Trace\n\nWe decided to benchmark BM25 retrieval accuracy.\n",
            encoding="utf-8",
        )

        ce.TRACES_DIR = traces
        ce.SEMANTIC_DIR = sem
        ce.CONSOLIDATION_DIR = con
        ce.GRAPH_DIR = graph
        ce.ENTITIES_PATH = graph / "entities.jsonl"
        ce.RELATIONS_PATH = graph / "relations.jsonl"
        ce.CLUSTERS_PATH = graph / "clusters.json"
        ce.PENDING_PATH = con / "pending.json"
        ce.LAST_RUN_PATH = con / "last.json"
        ce.SUMMARY_DAG_PATH = con / "dag.json"

        try:
            state = ObserverState()
            result = heartbeat(state, {"traces": traces})
            # All 4 sections must be present and not error
            for section in ("observe", "consolidate", "aging", "capacity"):
                assert section in result
                assert "error" not in result[section], f"{section} errored: {result[section]}"
        finally:
            for k, v in orig.items():
                setattr(ce, k, v)

    def test_e2e_incremental_index_preserves_unchanged(self, tmp_path):
        """Second build should report hash hits and keep unchanged files searchable."""
        import memory_index

        orig_sources = memory_index.SOURCES
        orig_hash = memory_index.HASH_CACHE_PATH

        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "doc.md").write_text(
            "# Document\n\nWe found that BM25 retrieval accuracy improved to 81 percent on LOCOMO.\n",
            encoding="utf-8",
        )
        memory_index.SOURCES = {"traces": traces}
        memory_index.HASH_CACHE_PATH = tmp_path / "hashes.json"

        try:
            idx = memory_index.MemoryIndex()
            s1 = idx.build(use_gpu_embeddings=False, use_gliner=False, incremental=True)
            assert s1["hash_misses"] >= 1
            assert s1["hash_hits"] == 0

            idx2 = memory_index.MemoryIndex()
            s2 = idx2.build(use_gpu_embeddings=False, use_gliner=False, incremental=True)
            assert s2["hash_hits"] >= 1
            assert s2["documents"] >= 1
            results = idx2.search("BM25 retrieval accuracy", top_k=3)
            assert any(result.name == "doc.md" for result in results)
        finally:
            memory_index.SOURCES = orig_sources
            memory_index.HASH_CACHE_PATH = orig_hash


class TestV04FeatureBenchmarks:
    """Precise benchmarks for all v0.4 features with documented budgets.

    Prints a formatted performance table at the end.
    """

    def test_classify_fact_9_types_benchmark(self):
        """Budget: classify_fact < 0.01ms per call."""
        from fact_decomposer import _classify_fact

        sentences = [
            "We decided to use BM25 for retrieval.",
            "Actually the previous approach was wrong.",
            "Always verify coverage before pushing.",
            "The deadline is March 30.",
            "To fix this, run pytest.",
            "I plan to improve temporal reasoning.",
            "I prefer dark mode.",
            "She started a new job.",
            "The meeting was productive.",
        ]
        # Warmup
        for s in sentences:
            _classify_fact(s)

        t0 = time.perf_counter()
        iterations = 5000
        for _ in range(iterations):
            for s in sentences:
                _classify_fact(s)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_call_us = elapsed_ms / (iterations * len(sentences)) * 1000
        print(f"\n  classify_fact (9 types): {per_call_us:.2f}µs/call")
        # CI has no Rust crate — Python regex fallback is ~100-200µs
        assert per_call_us < 5000, f"Too slow: {per_call_us:.2f}µs"

    def test_recency_weight_benchmark(self):
        """Budget: _recency_weight < 0.005ms per call."""
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [{"role": "user", "content": f"Fact {i} about BM25 performance."}] for i in range(10)
        ]
        dates = [f"2024-{(i % 12) + 1:02d}-15" for i in range(10)]
        ar = ArcaneRetriever(
            sessions,
            session_dates=dates,
            reference_date="2024-12-31",
            recency_half_life_days=30,
        )
        # Warmup
        for f in ar.facts:
            ar._recency_weight(f)

        t0 = time.perf_counter()
        iterations = 10000
        for _ in range(iterations):
            for f in ar.facts:
                ar._recency_weight(f)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_call_us = elapsed_ms / (iterations * len(ar.facts)) * 1000
        print(f"\n  _recency_weight: {per_call_us:.2f}µs/call")
        assert per_call_us < 5, f"Too slow: {per_call_us:.2f}µs"

    def test_content_hash_benchmark(self):
        """Budget: SHA-256 hash of 47K text < 0.1ms."""
        import hashlib

        t0 = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            hashlib.sha256(LONG_TEXT.encode("utf-8")).hexdigest()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_call_ms = elapsed_ms / iterations
        print(f"\n  SHA-256 (47K chars): {per_call_ms:.4f}ms/call")
        assert per_call_ms < 0.5, f"Too slow: {per_call_ms:.4f}ms"

    def test_capacity_report_benchmark(self, tmp_path):
        """Budget: capacity_report < 10ms for 5 categories, 50 files."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem = tmp_path / "semantic"
        ce.SEMANTIC_DIR = sem

        for cat in ["decision", "finding", "technical", "strategy", "personal"]:
            d = sem / cat
            d.mkdir(parents=True)
            for i in range(10):
                (d / f"{cat}_{i}.md").write_text(
                    f"---\nvalidity_state: active\n---\nContent {i}.\n" * 10,
                    encoding="utf-8",
                )

        try:
            # Warmup
            ce.capacity_report()
            ms, report = _timed(ce.capacity_report, n=100)
            print(f"\n  capacity_report (5 cats, 50 files): {ms:.3f}ms")
            # CI disk I/O is slower — generous budget
            assert ms < _budget(50), f"Too slow: {ms:.3f}ms"
            assert len(report) == 5
        finally:
            ce.SEMANTIC_DIR = orig

    def test_age_memories_benchmark(self, tmp_path):
        """Budget: age_memories < 20ms for 50 files."""
        import consolidation_engine as ce

        orig = ce.SEMANTIC_DIR
        sem = tmp_path / "semantic" / "decision"
        sem.mkdir(parents=True)
        ce.SEMANTIC_DIR = tmp_path / "semantic"

        for i in range(50):
            (sem / f"mem_{i}.md").write_text(
                f"---\nvalidity_state: active\nlast_accessed: 2024-0{(i % 9) + 1}-01\n---\nFact {i}.\n",
                encoding="utf-8",
            )

        try:
            ms, stats = _timed(ce.age_memories, "2024-12-01", n=10)
            print(f"\n  age_memories (50 files): {ms:.3f}ms")
            assert ms < _budget(50), f"Too slow: {ms:.3f}ms"
            assert stats["scanned"] == 50
        finally:
            ce.SEMANTIC_DIR = orig

    def test_build_summary_dag_benchmark(self):
        """Budget: build_summary_dag < 5ms for 100 traces."""
        from consolidation_engine import build_summary_dag

        data = {
            f"2024-0{(i % 9) + 1}-{(i % 28) + 1:02d}_trace_{i}.md": {
                "date": f"2024-0{(i % 9) + 1}-{(i % 28) + 1:02d}",
                "project": "remanentia",
                "entities": [f"ent_{i}", "bm25"],
                "key_lines": [f"Finding {i}: accuracy at {50 + i}%"],
                "text": f"Trace {i}: accuracy at {50 + i}% for BM25.",
            }
            for i in range(100)
        }
        # Warmup
        build_summary_dag(data)
        ms, dag = _timed(build_summary_dag, data, n=100)
        print(f"\n  build_summary_dag (100 traces): {ms:.3f}ms")
        assert ms < _budget(5), f"Too slow: {ms:.3f}ms"
        assert len(dag) > 100  # leaves + internal nodes

    def test_search_summary_dag_benchmark(self):
        """Budget: search_summary_dag < 1ms for 100-trace DAG."""
        from consolidation_engine import build_summary_dag, search_summary_dag

        data = {
            f"2024-0{(i % 9) + 1}-{(i % 28) + 1:02d}_trace_{i}.md": {
                "date": f"2024-0{(i % 9) + 1}-{(i % 28) + 1:02d}",
                "project": "remanentia",
                "entities": [f"ent_{i}", "bm25", "retrieval"],
                "key_lines": [f"Finding {i}: accuracy at {50 + i}%"],
                "text": f"Trace {i}: accuracy at {50 + i}% for BM25 retrieval.",
            }
            for i in range(100)
        }
        dag = build_summary_dag(data)
        # Warmup
        search_summary_dag(dag, "accuracy BM25 retrieval")
        ms, results = _timed(search_summary_dag, dag, "accuracy BM25 retrieval", n=1000)
        print(f"\n  search_summary_dag (100 traces): {ms:.4f}ms")
        # CI runners slower — generous budget
        assert ms < _budget(5), f"Too slow: {ms:.4f}ms"
        assert len(results) > 0

    def test_heartbeat_benchmark(self, tmp_path):
        """Budget: heartbeat (empty) < 20ms."""
        from observer import ObserverState, heartbeat

        state = ObserverState()
        # Warmup
        heartbeat(state, {"empty": tmp_path})
        ms, result = _timed(heartbeat, state, {"empty": tmp_path}, n=50)
        print(f"\n  heartbeat (empty): {ms:.3f}ms")
        assert ms < _budget(20), f"Too slow: {ms:.3f}ms"

    def test_full_v04_pipeline_benchmark(self):
        """Full v0.4 pipeline: sessions → decompose → ArcaneRetriever (with decay) → context.

        Budget: < 500ms for 5 sessions, 10 turns each.
        Uses patch to skip ML model loading (network-dependent).
        """
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [
                {
                    "role": "user",
                    "content": f"Session {s} turn {t}: We decided to use BM25 v{s}.{t} "
                    f"for retrieval. Accuracy measured at {70 + s + t} percent on LOCOMO.",
                }
                for t in range(10)
            ]
            for s in range(5)
        ]
        dates = [f"2024-0{s + 1}-15" for s in range(5)]

        # Block ML model to measure pure regex/heuristic path
        with patch.dict("sys.modules", {"fact_validity_model": None}):
            # Warmup
            ar = ArcaneRetriever(
                sessions,
                session_dates=dates,
                reference_date="2024-06-15",
                recency_half_life_days=30,
            )
            ar.retrieve("BM25 accuracy LOCOMO", "general", top_k=10)

            t0 = time.perf_counter()
            iterations = 10
            for _ in range(iterations):
                ar = ArcaneRetriever(
                    sessions,
                    session_dates=dates,
                    reference_date="2024-06-15",
                    recency_half_life_days=30,
                )
                results = ar.retrieve("BM25 accuracy LOCOMO", "general", top_k=10)
                ar.build_context("accuracy", results)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / iterations

        print(f"\n  Full v0.4 pipeline (5 sessions, 50 turns): {elapsed_ms:.1f}ms")
        assert elapsed_ms < 500, f"Too slow: {elapsed_ms:.1f}ms"
        assert len(results) > 0

    def test_documented_v04_performance_summary(self):
        """Print comprehensive performance summary for documentation."""
        from fact_decomposer import _classify_fact
        from consolidation_engine import build_summary_dag, search_summary_dag
        from arcane_retriever import ArcaneRetriever

        print("\n")
        print("=" * 70)
        print("  REMANENTIA v0.4 — FEATURE PERFORMANCE SUMMARY")
        print("=" * 70)

        # 1. classify_fact (9 types)
        sentences = [
            "We decided to use BM25.",
            "Actually it was wrong.",
            "Always verify coverage.",
            "Deadline is March 30.",
            "To fix this, run pytest.",
            "I plan to improve.",
            "I prefer dark mode.",
            "She started a job.",
            "Meeting was good.",
        ]
        t0 = time.perf_counter()
        for _ in range(5000):
            for s in sentences:
                _classify_fact(s)
        ms = (time.perf_counter() - t0) * 1000 / (5000 * 9)
        print(f"  classify_fact (9 types)          {ms * 1000:>8.2f} µs/call")

        # 2. Recency weight
        sessions = [
            [
                {
                    "role": "user",
                    "content": f"Fact number {i} about BM25 retrieval performance benchmarks.",
                }
            ]
            for i in range(10)
        ]
        dates = [f"2024-{(i % 12) + 1:02d}-15" for i in range(10)]
        ar = ArcaneRetriever(
            sessions,
            session_dates=dates,
            reference_date="2024-12-31",
            recency_half_life_days=30,
        )
        t0 = time.perf_counter()
        for _ in range(10000):
            for f in ar.facts:
                ar._recency_weight(f)
        ms = (time.perf_counter() - t0) * 1000 / (10000 * len(ar.facts))
        print(f"  _recency_weight                  {ms * 1000:>8.2f} µs/call")

        # 3. SHA-256 hash (47K)
        import hashlib

        t0 = time.perf_counter()
        for _ in range(1000):
            hashlib.sha256(LONG_TEXT.encode("utf-8")).hexdigest()
        ms = (time.perf_counter() - t0) * 1000 / 1000
        print(f"  SHA-256 (47K chars)              {ms:>8.4f} ms/call")

        # 4. build_summary_dag (100 traces)
        data = {
            f"trace_{i}.md": {
                "date": f"2024-0{(i % 9) + 1}-01",
                "project": "remanentia",
                "entities": ["bm25"],
                "key_lines": [f"Finding {i}"],
                "text": f"Trace {i}.",
            }
            for i in range(100)
        }
        t0 = time.perf_counter()
        for _ in range(100):
            build_summary_dag(data)
        ms = (time.perf_counter() - t0) * 1000 / 100
        print(f"  build_summary_dag (100 traces)   {ms:>8.3f} ms/call")

        # 5. search_summary_dag
        dag = build_summary_dag(data)
        t0 = time.perf_counter()
        for _ in range(1000):
            search_summary_dag(dag, "accuracy BM25")
        ms = (time.perf_counter() - t0) * 1000 / 1000
        print(f"  search_summary_dag               {ms:>8.4f} ms/call")

        # 6. Full pipeline (5 sessions) — block ML model for pure regex timing
        sessions_full = [
            [
                {
                    "role": "user",
                    "content": f"Session {s} turn {t}: We decided to use BM25 version {s}.{t} "
                    f"for retrieval scoring. Accuracy measured at {70 + s + t} percent on LOCOMO benchmark.",
                }
                for t in range(10)
            ]
            for s in range(5)
        ]
        dates_full = [f"2024-0{s + 1}-15" for s in range(5)]
        import sys

        saved_fvm = sys.modules.get("fact_validity_model")
        sys.modules["fact_validity_model"] = None
        try:
            t0 = time.perf_counter()
            for _ in range(10):
                a = ArcaneRetriever(
                    sessions_full,
                    session_dates=dates_full,
                    reference_date="2024-06-15",
                    recency_half_life_days=30,
                )
                r = a.retrieve("BM25 accuracy", "general", top_k=10)
                a.build_context("accuracy", r)
            ms = (time.perf_counter() - t0) * 1000 / 10
        finally:
            if saved_fvm is not None:
                sys.modules["fact_validity_model"] = saved_fvm
            else:
                sys.modules.pop("fact_validity_model", None)
        print(f"  Full v0.4 pipeline (50 turns)    {ms:>8.1f} ms/call")

        print("=" * 70)
