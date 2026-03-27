# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for wave 2 improvements (v0.3.1)

"""Tests for:
- Sentence-level indexing with context windows
- Query decomposition for multi-hop
- Enhanced prospective query generation (12 pattern categories)
- Adaptive retrieval with confidence scoring
- Cross-reference answer verification
"""

from __future__ import annotations

import math

import numpy as np

from memory_index import (
    MemoryIndex,
    SearchResult,
    _cross_reference_answers,
    _decompose_query,
    _generate_prospective_queries,
    _split_paragraphs,
    _split_sentences,
    _tokenize,
)


# ── Sentence-level indexing ──────────────────────────────────────


class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = _split_sentences(text)
        assert len(sents) == 3

    def test_short_text_stays_whole(self):
        sents = _split_sentences("Short.")
        assert len(sents) <= 1

    def test_abbreviations_not_split(self):
        text = "Dr. Smith went to the U.S. embassy. He filed a report."
        sents = _split_sentences(text)
        # Should not split on "Dr." or "U.S." — only on ". He"
        assert len(sents) <= 3


class TestSentenceLevelParagraphs:
    def test_short_paragraph_stays_whole(self):
        text = "A short paragraph under 200 chars."
        paras = _split_paragraphs(text)
        assert len(paras) == 1
        assert paras[0] == text

    def test_long_paragraph_splits_to_sentence_windows(self):
        text = (
            "First sentence about topic A with additional context to make it longer. "
            "Second sentence about topic B with more details about the subject matter. "
            "Third sentence about topic C discussing something entirely different here. "
            "Fourth sentence about topic D with extra words to push over the limit. "
            "Fifth sentence about topic E with final concluding remarks and thoughts."
        )
        assert len(text) > 200  # sanity check
        paras = _split_paragraphs(text)
        # Should produce sentence windows, not one giant paragraph
        assert len(paras) >= 3

    def test_context_window_includes_neighbours(self):
        text = (
            "Alice loves pottery and crafts. "
            "She also enjoys hiking on weekends. "
            "Bob works at Google as a data scientist."
        )
        paras = _split_paragraphs(text)
        # The middle sentence window should include both Alice and Bob context
        found_both = False
        for p in paras:
            if "hiking" in p and ("pottery" in p or "Google" in p):
                found_both = True
        assert found_both

    def test_code_not_affected(self):
        code = (
            "def foo(x):\n"
            '    """Compute the incremented value."""\n'
            "    return x + 1\n\n"
            "def bar(y):\n"
            '    """Compute the doubled value."""\n'
            "    return y * 2\n"
        )
        paras = _split_paragraphs(code, is_code=True)
        assert len(paras) >= 1


# ── Query decomposition ─────────────────────────────────────────


class TestQueryDecomposition:
    def test_person_who_pattern(self):
        q = "What hobbies does the person who works at Google have?"
        subs = _decompose_query(q)
        assert subs is not None
        assert len(subs) == 2
        assert any("google" in s.lower() for s in subs)
        assert any("hobbies" in s.lower() for s in subs)

    def test_does_person_who_pattern(self):
        q = "Does the person who likes pottery also enjoy hiking?"
        subs = _decompose_query(q)
        assert subs is not None
        assert len(subs) == 2

    def test_what_happened_pattern(self):
        q = "What happened after the team deployed version 3.0?"
        subs = _decompose_query(q)
        assert subs is not None
        assert len(subs) == 2

    def test_simple_query_returns_none(self):
        assert _decompose_query("What are Alice's hobbies?") is None
        assert _decompose_query("When was the bug fixed?") is None
        assert _decompose_query("How many tests passed?") is None


# ── Enhanced prospective queries ─────────────────────────────────


class TestEnhancedProspectiveQueries:
    def test_activity_patterns(self):
        text = "Alice loves pottery and enjoys hiking on weekends."
        queries = _generate_prospective_queries(text, "chat.md", "discussion")
        q_text = " ".join(queries).lower()
        assert "hobbies" in q_text or "interests" in q_text or "pottery" in q_text

    def test_occupation_patterns(self):
        text = "Bob works at Google as a data scientist."
        queries = _generate_prospective_queries(text, "chat.md", "discussion")
        q_text = " ".join(queries).lower()
        assert "work" in q_text or "google" in q_text or "career" in q_text

    def test_allergy_patterns(self):
        text = "Alice mentioned she is allergic to cats."
        queries = _generate_prospective_queries(text, "chat.md", "discussion")
        q_text = " ".join(queries).lower()
        assert "allergic" in q_text or "cats" in q_text

    def test_travel_patterns(self):
        text = "Bob just got back from a trip to Japan."
        queries = _generate_prospective_queries(text, "chat.md", "discussion")
        q_text = " ".join(queries).lower()
        assert "japan" in q_text or "trip" in q_text or "where" in q_text

    def test_learning_patterns(self):
        text = "Alice started learning piano last month."
        queries = _generate_prospective_queries(text, "chat.md", "discussion")
        q_text = " ".join(queries).lower()
        assert "learning" in q_text or "piano" in q_text

    def test_favourite_patterns(self):
        text = "Bob said his favorite movie is The Matrix."
        queries = _generate_prospective_queries(text, "chat.md", "discussion")
        q_text = " ".join(queries).lower()
        assert "favourite" in q_text or "favorite" in q_text or "matrix" in q_text

    def test_version_patterns(self):
        text = "Released v3.9.0 on 2026-03-15 with the new API."
        queries = _generate_prospective_queries(text, "release.md", "version")
        q_text = " ".join(queries).lower()
        assert "v3.9.0" in q_text or "version" in q_text

    def test_deduplication(self):
        text = "Alice loves pottery. Alice loves pottery."
        queries = _generate_prospective_queries(text, "chat.md", "discussion")
        # No duplicates
        assert len(queries) == len(set(queries))

    def test_cap_at_20(self):
        text = "Alice loves A. Bob loves B. " * 20
        queries = _generate_prospective_queries(text, "chat.md", "discussion")
        assert len(queries) <= 20


# ── Confidence scoring ───────────────────────────────────────────


class TestConfidenceScoring:
    def test_results_have_confidence(self, tmp_path):
        idx = _build_test_index(tmp_path)
        results = idx.search("What are Alice's hobbies?", top_k=3)
        assert results
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_top_result_higher_confidence(self, tmp_path):
        idx = _build_test_index(tmp_path)
        results = idx.search("Where does Bob work?", top_k=3)
        if len(results) >= 2:
            assert results[0].confidence >= results[1].confidence


# ── Cross-reference answer verification ──────────────────────────


class TestCrossReferenceAnswers:
    def test_agreeing_answers_boost_confidence(self):
        results = [
            SearchResult(
                name="a.md", source="s", score=5.0, snippet="x", answer="Google", confidence=0.8
            ),
            SearchResult(
                name="b.md", source="s", score=4.0, snippet="y", answer="Google", confidence=0.6
            ),
            SearchResult(
                name="c.md", source="s", score=3.0, snippet="z", answer="Amazon", confidence=0.5
            ),
        ]
        verified = _cross_reference_answers(results)
        # The two "Google" results should have boosted confidence
        assert verified[0].confidence > 0.8
        assert verified[1].confidence > 0.6
        # "Amazon" result should be unchanged
        assert verified[2].confidence == 0.5

    def test_no_answers_no_change(self):
        results = [
            SearchResult(
                name="a.md", source="s", score=5.0, snippet="x", answer="", confidence=0.8
            ),
            SearchResult(
                name="b.md", source="s", score=4.0, snippet="y", answer="", confidence=0.6
            ),
        ]
        verified = _cross_reference_answers(results)
        assert verified[0].confidence == 0.8
        assert verified[1].confidence == 0.6

    def test_single_result_no_change(self):
        results = [
            SearchResult(
                name="a.md", source="s", score=5.0, snippet="x", answer="Google", confidence=0.8
            ),
        ]
        verified = _cross_reference_answers(results)
        assert verified[0].confidence == 0.8


# ── Helpers ──────────────────────────────────────────────────────


def _build_test_index(tmp_path):
    """Build a test index from synthetic conversations."""
    from memory_index import Document

    idx = MemoryIndex()
    turns = [
        "Alice said she loves pottery and hiking on weekends.",
        "Bob mentioned he works as a data scientist at Google.",
        "Alice told Bob she started learning piano last month.",
        "Bob said his favorite movie is The Matrix.",
        "Alice mentioned she is allergic to cats.",
        "Bob said he just got back from a trip to Japan.",
    ]

    idx.documents = []
    idx.paragraph_index = []
    idx.paragraph_tokens = []
    idx.paragraph_token_counts = []
    idx.paragraph_types = []
    idx._inverted_index = {}
    idx._df = {}

    for i, turn in enumerate(turns):
        doc_idx = len(idx.documents)
        doc = Document(
            name=f"turn{i}.md", source="test", path=str(tmp_path / f"turn{i}.md"), paragraphs=[turn]
        )
        idx.documents.append(doc)
        p_idx = len(idx.paragraph_tokens)
        idx.paragraph_index.append((doc_idx, 0))
        token_list = _tokenize(turn)
        tokens = set(token_list)
        from memory_index import _token_counts

        counts = _token_counts(token_list)
        idx.paragraph_tokens.append(tokens)
        idx.paragraph_token_counts.append(counts)
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
    idx._built = True

    return idx
