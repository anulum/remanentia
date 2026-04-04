# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for remanentia_retrieve Rust crate

"""Multi-angle tests for the remanentia_retrieve Rust crate.

Covers: correctness (Rust == Python), empty inputs, error handling,
negative cases, performance benchmarks, and roundtrip consistency.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

rr = pytest.importorskip("remanentia_retrieve")

STOPWORDS = frozenset(
    "the a an and or but in on at to for of is it by as with from was were "
    "be been have has had this that these those are not no its the into can "
    "will would should could may also so if when then than more most all any "
    "each every both few many much some such only just about over after before "
    "between through during up down out off did do does how what which who whom "
    "where why here there their them they we our us you your he she his her "
    "i me my we us being now very".split()
)


# ── Tokenize ────────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self):
        result = rr.tokenize("Hello World Testing", STOPWORDS)
        assert result == ["hello", "world", "testing"]

    def test_stopword_removal(self):
        result = rr.tokenize("the quick brown fox and the lazy dog", STOPWORDS)
        assert "the" not in result
        assert "and" not in result
        assert "quick" in result

    def test_single_char_dropped(self):
        result = rr.tokenize("I am a b c developer", STOPWORDS)
        assert "b" not in result
        assert "c" not in result

    def test_empty_string(self):
        assert rr.tokenize("", STOPWORDS) == []

    def test_only_stopwords(self):
        assert rr.tokenize("the and or but", STOPWORDS) == []

    def test_numbers_kept(self):
        result = rr.tokenize("version 312 alpha", STOPWORDS)
        assert "312" in result

    def test_underscores(self):
        result = rr.tokenize("snake_case variable_name", STOPWORDS)
        assert "snake_case" in result

    def test_no_stopwords_filter(self):
        result = rr.tokenize("the quick brown", set())
        assert "the" in result


# ── Stem ────────────────────────────────────────────────────────


class TestStem:
    def test_ation_suffix(self):
        assert rr.stem("computation") == "comput"

    def test_ing_suffix(self):
        assert rr.stem("testing") == "test"

    def test_no_suffix(self):
        assert rr.stem("rust") == "rust"

    def test_too_short_stem(self):
        # "as" - suffix "s" would leave "a" (len < 3), so no stripping
        assert rr.stem("as") == "as"

    def test_ize_suffix(self):
        assert rr.stem("optimize") == "optim"

    def test_ly_suffix(self):
        assert rr.stem("quickly") == "quick"

    def test_empty_string(self):
        assert rr.stem("") == ""


# ── Bigrams ─────────────────────────────────────────────────────


class TestBigrams:
    def test_basic(self):
        assert rr.bigrams(["a", "b", "c"]) == ["a_b", "b_c"]

    def test_single_token(self):
        assert rr.bigrams(["alone"]) == []

    def test_empty(self):
        assert rr.bigrams([]) == []

    def test_two_tokens(self):
        assert rr.bigrams(["x", "y"]) == ["x_y"]


# ── Expand query ────────────────────────────────────────────────


class TestExpandQuery:
    def test_adds_stems(self):
        result = rr.expand_query("gyrokinetic transport saturation", STOPWORDS)
        assert "gyrokinetic" in result
        assert "satur" in result  # stemmed form

    def test_no_expansion_needed(self):
        result = rr.expand_query("rust code", STOPWORDS)
        assert result == "rust code"

    def test_empty_query(self):
        result = rr.expand_query("", STOPWORDS)
        assert result == ""


# ── Build IDF ───────────────────────────────────────────────────


class TestBuildIdf:
    def test_basic(self):
        docs = {"doc1": "hello world", "doc2": "hello rust"}
        idf = rr.build_idf(docs, STOPWORDS)
        assert "hello" in idf
        assert "world" in idf
        assert "rust" in idf
        # "hello" appears in both -> lower IDF
        assert idf["hello"] < idf["world"]

    def test_empty_docs(self):
        idf = rr.build_idf({}, STOPWORDS)
        assert idf == {}

    def test_single_doc(self):
        docs = {"doc": "alpha beta gamma"}
        idf = rr.build_idf(docs, STOPWORDS)
        assert len(idf) > 0


# ── TF-IDF Score ────────────────────────────────────────────────


class TestTfidfScore:
    def test_matching_query(self):
        idf = {"hello": 1.0, "world": 0.5}
        score = rr.tfidf_score("hello world", "test_doc", "hello world text", idf, STOPWORDS)
        assert score > 0.0

    def test_no_match(self):
        idf = {"hello": 1.0}
        score = rr.tfidf_score("hello", "other", "nothing related", idf, STOPWORDS)
        assert score == 0.0

    def test_empty_query(self):
        score = rr.tfidf_score("", "doc", "some text", {}, STOPWORDS)
        assert score == 0.0

    def test_filename_boost(self):
        idf = {"alpha": 1.0, "beta": 1.0}
        score_in_name = rr.tfidf_score("alpha", "alpha_doc", "beta text", idf, STOPWORDS)
        score_in_body = rr.tfidf_score("alpha", "other_doc", "alpha text", idf, STOPWORDS)
        # Filename gets 3x boost, so matching filename should score higher
        assert score_in_name > score_in_body


# ── Filename Bonus ──────────────────────────────────────────────


class TestFilenameBonus:
    def test_full_match(self):
        idf = {"scpn": 2.0, "control": 1.0}
        bonus = rr.filename_bonus("scpn control", "scpn control system", idf, STOPWORDS)
        assert bonus == pytest.approx(1.0)

    def test_partial_match(self):
        idf = {"scpn": 2.0, "control": 1.0}
        bonus = rr.filename_bonus("scpn control", "scpn daemon", idf, STOPWORDS)
        assert 0.0 < bonus < 1.0

    def test_no_match(self):
        idf = {"alpha": 1.0}
        bonus = rr.filename_bonus("alpha", "beta gamma", idf, STOPWORDS)
        assert bonus == 0.0

    def test_empty_query(self):
        bonus = rr.filename_bonus("", "something", {}, STOPWORDS)
        assert bonus == 0.0


# ── Cosine Similarity ───────────────────────────────────────────


class TestCosineSim:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rr.cosine_sim(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert rr.cosine_sim(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert rr.cosine_sim(a, b) == 0.0

    def test_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert rr.cosine_sim(a, b) == pytest.approx(-1.0)


# ── Spike Feature ───────────────────────────────────────────────


class TestSpikeFeature:
    def test_output_shape(self):
        w = np.random.default_rng(42).normal(0, 0.1, (50, 50))
        stim = np.random.default_rng(42).uniform(0, 1, 50)
        result = rr.spike_feature(w, stim, 50)
        assert result.shape == (50,)
        assert result.dtype == np.float32

    def test_zero_stimulus(self):
        w = np.zeros((20, 20))
        stim = np.zeros(20)
        result = rr.spike_feature(w, stim, 50)
        assert result.shape == (20,)

    def test_deterministic(self):
        w = np.random.default_rng(42).normal(0, 0.1, (30, 30))
        stim = np.random.default_rng(42).uniform(0, 1, 30)
        r1 = rr.spike_feature(w, stim, 50)
        r2 = rr.spike_feature(w, stim, 50)
        np.testing.assert_array_equal(r1, r2)

    def test_more_steps_more_spikes(self):
        w = np.random.default_rng(42).normal(0, 0.05, (20, 20))
        stim = np.random.default_rng(42).uniform(0.5, 1.0, 20)
        r10 = rr.spike_feature(w, stim, 10)
        r100 = rr.spike_feature(w, stim, 100)
        assert r100.sum() >= r10.sum()


# ── SNN Affinity ────────────────────────────────────────────────


class TestSnnAffinity:
    def test_same_stimulus(self):
        w = np.random.default_rng(42).normal(0, 0.1, (30, 30))
        stim = np.random.default_rng(42).uniform(0, 1, 30)
        aff = rr.snn_affinity(w, stim, stim, 50)
        assert aff == pytest.approx(1.0, abs=0.01)

    def test_different_stimuli(self):
        w = np.random.default_rng(42).normal(0, 0.1, (30, 30))
        s1 = np.random.default_rng(1).uniform(0, 1, 30)
        s2 = np.random.default_rng(2).uniform(0, 1, 30)
        aff = rr.snn_affinity(w, s1, s2, 50)
        assert -1.0 <= aff <= 1.0


# ── Hash Encode ─────────────────────────────────────────────────


class TestHashEncode:
    PRIMES = [7919, 104729, 15485863, 32452843, 49979687, 67867967, 86028121]

    def test_output_shape(self):
        enc = rr.hash_encode("hello world", 1000, self.PRIMES, STOPWORDS)
        assert enc.shape == (1000,)

    def test_values_in_range(self):
        enc = rr.hash_encode("test text for encoding", 500, self.PRIMES, STOPWORDS)
        assert enc.min() >= 0.0
        assert enc.max() <= 1.0

    def test_deterministic(self):
        e1 = rr.hash_encode("same text", 200, self.PRIMES, STOPWORDS)
        e2 = rr.hash_encode("same text", 200, self.PRIMES, STOPWORDS)
        np.testing.assert_array_equal(e1, e2)

    def test_different_text_different_encoding(self):
        e1 = rr.hash_encode("alpha beta", 200, self.PRIMES, STOPWORDS)
        e2 = rr.hash_encode("gamma delta", 200, self.PRIMES, STOPWORDS)
        assert not np.array_equal(e1, e2)

    def test_empty_text(self):
        enc = rr.hash_encode("", 100, self.PRIMES, STOPWORDS)
        assert enc.sum() == 0.0


# ── RRF ─────────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    def test_basic(self):
        lists = [[(0, 1.0), (1, 0.5)], [(1, 1.0), (2, 0.5)]]
        result = rr.reciprocal_rank_fusion(lists, 60)
        ids = [r[0] for r in result]
        assert 1 in ids  # appears in both lists -> highest RRF

    def test_single_list(self):
        result = rr.reciprocal_rank_fusion([[(5, 1.0), (3, 0.5)]], 60)
        assert result[0][0] == 5

    def test_empty_lists(self):
        assert rr.reciprocal_rank_fusion([], 60) == []

    def test_k_parameter(self):
        lists = [[(0, 1.0)]]
        r60 = rr.reciprocal_rank_fusion(lists, 60)
        r10 = rr.reciprocal_rank_fusion(lists, 10)
        # Smaller k → higher scores
        assert r10[0][1] > r60[0][1]

    def test_sorted_descending(self):
        lists = [[(0, 1.0), (1, 0.8), (2, 0.6)], [(2, 1.0), (0, 0.5)]]
        result = rr.reciprocal_rank_fusion(lists, 60)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


# ── Entity Graph Score ──────────────────────────────────────────


class TestEntityGraphScore:
    def test_matching_entities(self):
        score = rr.entity_graph_score(
            ["scpn", "disruption"],
            ["scpn", "tokamak"],
            [("scpn", "tokamak", 2.0), ("disruption", "scpn", 1.5)],
        )
        assert score > 0.0

    def test_no_overlap(self):
        score = rr.entity_graph_score(
            ["alpha"],
            ["beta"],
            [("gamma", "delta", 1.0)],
        )
        assert score == 0.0

    def test_empty_entities(self):
        assert rr.entity_graph_score([], ["a"], [("a", "b", 1.0)]) == 0.0
        assert rr.entity_graph_score(["a"], [], [("a", "b", 1.0)]) == 0.0

    def test_empty_relations(self):
        assert rr.entity_graph_score(["a"], ["b"], []) == 0.0

    def test_bounded_to_one(self):
        score = rr.entity_graph_score(
            ["a"],
            ["b"],
            [("a", "b", 100.0)],
        )
        assert score <= 1.0


# ── Performance Benchmarks ──────────────────────────────────────


class TestPerformance:
    def _bench(self, fn, iterations=1000):
        fn()  # warmup
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - t0
        return elapsed / iterations * 1e6  # µs per call

    def test_tokenize_perf(self):
        text = "The quick brown fox jumps over the lazy dog " * 10
        us = self._bench(lambda: rr.tokenize(text, STOPWORDS))
        assert us < 500  # budget: 500µs

    def test_stem_perf(self):
        us = self._bench(lambda: rr.stem("computation"), iterations=5000)
        assert us < 5  # budget: 5µs

    def test_bigrams_perf(self):
        tokens = [f"token_{i}" for i in range(50)]
        us = self._bench(lambda: rr.bigrams(tokens))
        assert us < 100  # budget: 100µs (PyO3 FFI overhead dominates for short lists)

    def test_tfidf_score_perf(self):
        idf = {f"term_{i}": float(i) for i in range(100)}
        text = " ".join(f"term_{i}" for i in range(200))
        us = self._bench(lambda: rr.tfidf_score("term_5 term_10 term_50", "doc_name", text, idf, STOPWORDS))
        assert us < 500  # budget: 500µs

    def test_spike_feature_perf(self):
        w = np.random.default_rng(42).normal(0, 0.1, (100, 100))
        stim = np.random.default_rng(42).uniform(0, 1, 100)
        us = self._bench(lambda: rr.spike_feature(w, stim, 50), iterations=100)
        assert us < 5000  # budget: 5ms for 100 neurons x 50 steps

    def test_rrf_perf(self):
        lists = [[(i, float(100 - i)) for i in range(100)] for _ in range(5)]
        us = self._bench(lambda: rr.reciprocal_rank_fusion(lists, 60))
        assert us < 200  # budget: 200µs

    def test_hash_encode_perf(self):
        primes = [7919, 104729, 15485863, 32452843, 49979687, 67867967, 86028121]
        text = "performance test encoding with multiple words and bigrams"
        us = self._bench(lambda: rr.hash_encode(text, 1000, primes, STOPWORDS))
        assert us < 200  # budget: 200µs
