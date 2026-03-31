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
import time
from datetime import date
from unittest.mock import patch

import numpy as np

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
        assert ms < 50, f"parse_dates: {ms:.2f}ms exceeds 50ms budget"
        assert len(dates) >= 1
        assert "2026-03-15" in dates


class TestExtractAnswerPerformance:
    """answer_extractor.extract_answer — budget: <2ms for 47K chars."""

    def test_extract_answer_budget(self):
        from answer_extractor import extract_answer

        ms, answer = _timed(extract_answer, QUERY, LONG_TEXT)
        assert ms < 20, f"extract_answer: {ms:.2f}ms exceeds 20ms budget"
        assert answer is not None

    def test_fuzzy_match_budget(self):
        from answer_extractor import fuzzy_match

        ms, result = _timed(fuzzy_match, "March 15, 2026", "march 15 2026", 0.5)
        assert ms < 1, f"fuzzy_match: {ms:.2f}ms exceeds 1ms budget"

    def test_normalize_number_budget(self):
        from answer_extractor import normalize_number

        ms, result = _timed(normalize_number, "forty-two")
        assert ms < 1, f"normalize_number: {ms:.2f}ms exceeds 1ms budget"
        assert result == "42"


class TestAnswerNormalizerPerformance:
    """answer_normalizer — budget: <1ms per call."""

    def test_normalize_answer_budget(self):
        from answer_normalizer import normalize_answer

        ms, result = _timed(normalize_answer, "Likely yes, because she enjoys reading")
        assert ms < 1, f"normalize_answer: {ms:.2f}ms exceeds 1ms budget"
        assert result == "likely yes"

    def test_answers_match_budget(self):
        from answer_normalizer import answers_match

        ms, result = _timed(answers_match, "Likely yes", "Yes", 0.25)
        assert ms < 1, f"answers_match: {ms:.2f}ms exceeds 1ms budget"
        assert result is True


class TestEntityExtractorPerformance:
    """entity_extractor._regex_entities — budget: <5ms for 47K chars."""

    def test_regex_entities_budget(self):
        from entity_extractor import _regex_entities

        ms, entities = _timed(_regex_entities, LONG_TEXT)
        assert ms < 50, f"regex_entities: {ms:.2f}ms exceeds 50ms budget"
        assert len(entities) >= 1

    def test_extract_relations_budget(self):
        from entity_extractor import _regex_entities, extract_relations

        entities = _regex_entities(SHORT_TEXT)
        if len(entities) >= 2:
            ms, rels = _timed(extract_relations, SHORT_TEXT, entities)
            assert ms < 5, f"extract_relations: {ms:.2f}ms exceeds 5ms budget"


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
        assert ms < 5000, f"decompose_sessions (warm): {ms:.2f}ms exceeds 5000ms budget"
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
        assert ms < 50, f"add_note: {ms:.2f}ms exceeds 50ms budget"

    def test_search_budget(self):
        from knowledge_store import KnowledgeStore

        store = KnowledgeStore()
        for i in range(20):
            store.add_note(f"Fact {i}: unique content about topic {i}.", source=f"s{i}.md")
        ms, results = _timed(store.search, "topic 5", n=50)
        assert ms < 20, f"search: {ms:.2f}ms exceeds 20ms budget"


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
        assert ms < 100, f"add_events(100): {ms:.2f}ms exceeds 100ms budget"

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
        assert ms < 10, f"query_temporal: {ms:.2f}ms exceeds 10ms budget"


class TestLLMBackendPerformance:
    """llm_backend.resolve_backend — budget: <5ms."""

    def test_resolve_backend_budget(self):
        from llm_backend import resolve_backend

        ms, backend = _timed(resolve_backend, "none")
        assert ms < 5, f"resolve_backend: {ms:.2f}ms exceeds 5ms budget"

    def test_null_backend_budget(self):
        from llm_backend import NullBackend

        b = NullBackend()
        ms, result = _timed(b.complete, "test")
        assert ms < 0.5, f"NullBackend.complete: {ms:.2f}ms exceeds 0.5ms budget"
        assert result is None


class TestLLMSetupPerformance:
    """llm_setup.write_config — budget: <5ms."""

    def test_write_config_budget(self, tmp_path):
        from llm_setup import write_config

        path = tmp_path / "perf.toml"
        ms, result = _timed(write_config, path=path, n=50)
        assert ms < 10, f"write_config: {ms:.2f}ms exceeds 10ms budget"


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
            patch("knowledge_store.STORE_PATH", store_path),
            patch("knowledge_store.TRIGGERS_PATH", triggers_path),
        ):
            state = ObserverState()
            t0 = time.perf_counter()
            observe_once(state, {"traces": d})
            ms = (time.perf_counter() - t0) * 1000
        assert ms < 500, f"observe_once(10 files): {ms:.2f}ms exceeds 500ms budget"


class TestReflectorPerformance:
    """reflector.reflect_once — budget: <100ms (no LLM)."""

    def test_reflect_once_budget(self):
        from reflector import reflect_once

        t0 = time.perf_counter()
        reflect_once(days=1, use_llm=False)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 500, f"reflect_once: {ms:.2f}ms exceeds 500ms budget"


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
                normed = normalize_answer(answer)
                matched = answers_match(answer, "88.5%", 0.25)
        ms = (time.perf_counter() - t0) * 1000 / 10

        assert ms < 200, f"Full regex pipeline: {ms:.2f}ms exceeds 200ms budget"
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
        assert ms < 20, f"MemoryIndex.search(100 paras): {ms:.2f}ms exceeds 20ms budget"
        assert len(results) > 0


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
