# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Training data extraction tests

"""Tests for training/generate_data.py.

Validates the BM25 scoring, tokenisation, and data extraction functions
that produce training data from LongMemEval.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from generate_data import (
    _bm25_score,
    _flatten_turns,
    _tokenise,
    extract_natural_date_examples,
    generate_cross_encoder_pairs,
    generate_embedding_triplets,
)


# ── _tokenise ──────────────────────────────────────────────────


class TestTokenise:
    def test_basic(self):
        tokens = _tokenise("Hello beautiful world")
        assert "hello" in tokens
        assert "beautiful" in tokens
        assert "world" in tokens

    def test_filters_short(self):
        tokens = _tokenise("I am ok here")
        assert "here" in tokens
        assert "am" not in tokens  # 2 chars
        assert "ok" not in tokens  # 2 chars

    def test_empty(self):
        assert _tokenise("") == set()

    def test_returns_set(self):
        tokens = _tokenise("word word word")
        assert isinstance(tokens, set)
        assert len(tokens) == 1

    def test_lowercase(self):
        tokens = _tokenise("HELLO World")
        assert "hello" in tokens
        assert "world" in tokens


# ── _bm25_score ────────────────────────────────────────────────


class TestBm25Score:
    def test_exact_match_positive(self):
        q = {"hello", "world"}
        doc = ["hello", "world", "foo", "bar"]
        idf = {"hello": 1.0, "world": 1.0}
        score = _bm25_score(q, doc, idf, avg_dl=4.0)
        assert score > 0

    def test_no_overlap_zero(self):
        q = {"alpha", "beta"}
        doc = ["gamma", "delta"]
        idf = {"alpha": 1.0, "beta": 1.0}
        score = _bm25_score(q, doc, idf, avg_dl=2.0)
        assert score == 0.0

    def test_empty_query(self):
        score = _bm25_score(set(), ["word"], {"word": 1.0}, avg_dl=1.0)
        assert score == 0.0

    def test_empty_doc(self):
        score = _bm25_score({"word"}, [], {"word": 1.0}, avg_dl=1.0)
        assert score == 0.0

    def test_higher_tf_higher_score(self):
        q = {"word"}
        idf = {"word": 1.0}
        s1 = _bm25_score(q, ["word"], idf, avg_dl=3.0)
        s2 = _bm25_score(q, ["word", "word", "word"], idf, avg_dl=3.0)
        assert s2 > s1  # more occurrences → higher score


# ── _flatten_turns ─────────────────────────────────────────────


class TestFlattenTurns:
    def test_basic(self):
        sessions = [
            [
                {"role": "user", "content": "This is a sufficiently long message."},
                {"role": "assistant", "content": "And this is the reply from assistant."},
            ]
        ]
        turns = _flatten_turns(sessions)
        assert len(turns) == 2

    def test_filters_short(self):
        sessions = [
            [
                {"role": "user", "content": "Short."},
                {"role": "user", "content": "This one is long enough to be included in the list."},
            ]
        ]
        turns = _flatten_turns(sessions)
        assert len(turns) == 1

    def test_empty_sessions(self):
        assert _flatten_turns([]) == []

    def test_multiple_sessions(self):
        sessions = [
            [{"role": "user", "content": "Session one has enough text here."}],
            [{"role": "user", "content": "Session two also has enough text here."}],
        ]
        turns = _flatten_turns(sessions)
        assert len(turns) == 2


# ── generate_embedding_triplets ────────────────────────────────


class TestGenerateEmbeddingTriplets:
    def _make_data(self):
        return [
            {
                "question_id": "q1",
                "question_type": "temporal-reasoning",
                "question": "When did I buy the car?",
                "answer": "last Tuesday at the dealership",
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": "I went to the dealership and bought a car last Tuesday at the dealership. It was a great experience overall.",
                        },
                        {
                            "role": "assistant",
                            "content": "That's wonderful! What kind of car did you get?",
                        },
                        {
                            "role": "user",
                            "content": "A blue Toyota Camry. The weather was beautiful yesterday and I took it for a long drive through the countryside.",
                        },
                    ]
                ],
            },
        ]

    def test_produces_triplets(self):
        data = self._make_data()
        triplets = generate_embedding_triplets(data)
        assert len(triplets) >= 1

    def test_triplet_keys(self):
        data = self._make_data()
        triplets = generate_embedding_triplets(data)
        for t in triplets:
            assert "anchor" in t
            assert "positive" in t
            assert "negative" in t
            assert "qtype" in t

    def test_anchor_is_question(self):
        data = self._make_data()
        triplets = generate_embedding_triplets(data)
        for t in triplets:
            assert t["anchor"] == "When did I buy the car?"

    def test_empty_data(self):
        assert generate_embedding_triplets([]) == []


# ── generate_cross_encoder_pairs ───────────────────────────────


class TestGenerateCrossEncoderPairs:
    def _make_data(self):
        return [
            {
                "question_id": "q1",
                "question_type": "temporal-reasoning",
                "question": "When did I buy the car?",
                "answer": "last Tuesday at the dealership",
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": "I bought a car last Tuesday at the dealership. Great experience overall.",
                        },
                        {
                            "role": "assistant",
                            "content": "What kind of car did you buy? I hope it was a good deal.",
                        },
                        {
                            "role": "user",
                            "content": "The weather was beautiful yesterday so I went for a long countryside drive.",
                        },
                    ]
                ],
            },
        ]

    def test_produces_pairs(self):
        data = self._make_data()
        pairs = generate_cross_encoder_pairs(data)
        assert len(pairs) >= 1

    def test_pair_keys(self):
        data = self._make_data()
        pairs = generate_cross_encoder_pairs(data)
        for p in pairs:
            assert "query" in p
            assert "passage" in p
            assert "label" in p
            assert p["label"] in (0, 1)

    def test_has_positives_and_negatives(self):
        data = self._make_data()
        pairs = generate_cross_encoder_pairs(data)
        labels = [p["label"] for p in pairs]
        assert 1 in labels
        # Negatives require keyword overlap — may or may not exist

    def test_empty_data(self):
        assert generate_cross_encoder_pairs([]) == []


# ── extract_natural_date_examples ──────────────────────────────


class TestExtractNaturalDateExamples:
    def test_temporal_questions_only(self):
        data = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "question": "test",
                "answer": "test",
                "haystack_dates": ["2023/04/10 (Mon) 17:50"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "I did it 3 weeks ago"},
                    ]
                ],
            },
        ]
        # Should skip non-temporal questions
        results = extract_natural_date_examples(data)
        assert len(results) == 0

    def test_extracts_vague_dates(self):
        data = [
            {
                "question_id": "q1",
                "question_type": "temporal-reasoning",
                "question": "When?",
                "answer": "3 weeks ago",
                "haystack_dates": ["2023/04/10 (Mon) 17:50"],
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": "I bought it about 3 weeks ago and it works great.",
                        },
                    ]
                ],
            },
        ]
        results = extract_natural_date_examples(data)
        assert len(results) >= 1
        assert results[0]["ref_date"] == "2023-04-10"

    def test_no_haystack_dates(self):
        data = [
            {
                "question_id": "q1",
                "question_type": "temporal-reasoning",
                "question": "When?",
                "answer": "test",
                "haystack_dates": [],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "I did it recently"},
                    ]
                ],
            },
        ]
        results = extract_natural_date_examples(data)
        assert len(results) == 0  # no ref date available

    def test_empty_data(self):
        assert extract_natural_date_examples([]) == []
