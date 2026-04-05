# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Rust parity tests for remanentia_recall crate
"""Parity and performance tests for remanentia_recall Rust functions.

Skipped entirely when the crate is not installed (CI).
"""

from __future__ import annotations

import re
import time

import pytest

rc = pytest.importorskip("remanentia_recall")


# ── tokenize_words ──────────────────────────────────────────────


class TestTokenizeWords:
    def test_basic(self):
        result = rc.tokenize_words("Hello World 123")
        assert result == {"hello", "world", "123"}

    def test_empty(self):
        assert rc.tokenize_words("") == set()

    def test_unicode(self):
        result = rc.tokenize_words("Šotek café naïve")
        assert "otek" in result or "café" in result or len(result) >= 1

    def test_parity_with_python(self):
        text = "BM25 retrieval scoring improved accuracy on LOCOMO benchmark at 83.1%"
        py = set(re.findall(r"\w+", text.lower()))
        rust = rc.tokenize_words(text)
        assert py == rust

    def test_special_chars(self):
        result = rc.tokenize_words("a-b_c.d e")
        assert "b_c" in result or "a" in result

    def test_large_input(self):
        text = "word " * 10000
        result = rc.tokenize_words(text)
        assert "word" in result


# ── tokenize_words_min ──────────────────────────────────────────


class TestTokenizeWordsMin:
    def test_min4(self):
        result = rc.tokenize_words_min("one two three four fives", 4)
        assert "one" not in result
        assert "two" not in result
        assert "four" in result
        assert "fives" in result
        assert "three" in result

    def test_parity_with_python(self):
        text = "The LOCOMO benchmark measures conversational QA accuracy"
        py = set(re.findall(r"\w{4,}", text.lower()))
        rust = rc.tokenize_words_min(text, 4)
        assert py == rust

    def test_empty(self):
        assert rc.tokenize_words_min("", 4) == set()

    def test_all_short(self):
        assert rc.tokenize_words_min("a b c", 4) == set()


# ── token_overlap_score ─────────────────────────────────────────


class TestTokenOverlapScore:
    def test_full_overlap(self):
        tokens = {"hello", "world"}
        assert rc.token_overlap_score(tokens, tokens) == 1.0

    def test_no_overlap(self):
        assert rc.token_overlap_score({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        score = rc.token_overlap_score({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(score - 2 / 3) < 1e-9

    def test_empty_query(self):
        assert rc.token_overlap_score(set(), {"a", "b"}) == 0.0

    def test_empty_doc(self):
        assert rc.token_overlap_score({"a"}, set()) == 0.0

    def test_parity_with_python(self):
        q = set(re.findall(r"\w+", "BM25 scoring accuracy".lower()))
        d = set(re.findall(r"\w+", "BM25 retrieval accuracy improved".lower()))
        py_score = len(q & d) / max(len(q), 1)
        rust_score = rc.token_overlap_score(q, d)
        assert abs(py_score - rust_score) < 1e-9


# ── assess_novelty ──────────────────────────────────────────────


class TestAssessNovelty:
    def test_all_known(self):
        known = {"locomo", "benchmark", "accuracy"}
        score = rc.assess_novelty("LOCOMO benchmark accuracy", known)
        assert score == 0.0

    def test_all_novel(self):
        score = rc.assess_novelty("quantum entanglement decoherence", set())
        assert score == 1.0

    def test_partial(self):
        known = {"locomo"}
        score = rc.assess_novelty("LOCOMO benchmark accuracy", known)
        assert 0.0 < score < 1.0

    def test_empty_query(self):
        assert rc.assess_novelty("hi", {"locomo"}) == 0.0

    def test_parity_with_python(self):
        query = "Dimits shift convergence analysis results"
        known = {"dimits", "shift", "plasma", "tokamak"}
        # Python path
        q_tokens = set(re.findall(r"\w{4,}", query.lower()))
        unknown = q_tokens - known
        py_score = len(unknown) / len(q_tokens) if q_tokens else 0.0
        # Rust path
        rust_score = rc.assess_novelty(query, known)
        assert abs(py_score - rust_score) < 1e-9


# ── Performance ─────────────────────────────────────────────────


class TestPerformance:
    def test_tokenize_words_speed(self):
        text = "BM25 retrieval scoring " * 100
        t0 = time.perf_counter()
        for _ in range(1000):
            rc.tokenize_words(text)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 5000, f"1000 tokenize_words calls took {ms:.1f}ms"

    def test_overlap_score_speed(self):
        q = rc.tokenize_words("BM25 retrieval scoring accuracy benchmark")
        d = rc.tokenize_words("BM25 retrieval accuracy improved on LOCOMO benchmark results")
        t0 = time.perf_counter()
        for _ in range(10000):
            rc.token_overlap_score(q, d)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 1000, f"10000 overlap_score calls took {ms:.1f}ms"

    def test_assess_novelty_speed(self):
        known = {f"token_{i}" for i in range(1000)}
        query = "quantum entanglement decoherence analysis convergence results"
        t0 = time.perf_counter()
        for _ in range(1000):
            rc.assess_novelty(query, known)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 2000, f"1000 assess_novelty calls took {ms:.1f}ms"
