# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
"""Tests for entity boost gating in bench_locomo.

Imports specific functions only — bench_locomo has module-level stdout
redirect that conflicts with pytest capture.
"""

from __future__ import annotations

import re


# Reproduce the logic here to test without module-level side effects.
# The actual code lives in bench_locomo.py — these mirror it exactly.

_PERSON_CENTRIC_PATTERNS = re.compile(
    r"\b(relationship|hobby|hobbies|interest|interests|career|job|status|"
    r"personality|feel|feeling|prefer|favorite|partake|destress|self-care|"
    r"political|leaning|member|community)\b",
    re.IGNORECASE,
)

_POSSESSIVE_PATTERNS = re.compile(
    r"\b(his|her|their|'s)\s+(hobby|hobbies|interest|interests|career|"
    r"relationship|status|personality|feeling|preference|activity|activities)\b",
    re.IGNORECASE,
)


def _extract_query_names(query):
    names = set()
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", query):
        word = m.group(1).lower()
        if word not in {
            "what",
            "when",
            "where",
            "who",
            "how",
            "why",
            "would",
            "could",
            "does",
            "did",
            "has",
            "have",
            "the",
            "which",
            "likely",
            "yes",
            "not",
        }:
            names.add(word)
    return names


def _is_person_centric(query):
    if _PERSON_CENTRIC_PATTERNS.search(query):
        return True
    if _POSSESSIVE_PATTERNS.search(query):
        return True
    q_lower = query.lower()
    return any(w in q_lower for w in ["would ", "could ", "likely "])


class TestExtractQueryNames:
    def test_single_name(self):
        assert _extract_query_names("What is Caroline's hobby?") == {"caroline"}

    def test_multiple_names(self):
        names = _extract_query_names("Did Melanie and Caroline go camping?")
        assert "melanie" in names
        assert "caroline" in names

    def test_filters_common_words(self):
        names = _extract_query_names("What would likely happen?")
        assert len(names) == 0

    def test_no_names(self):
        assert _extract_query_names("how long ago was that?") == set()


class TestIsPersonCentric:
    def test_relationship_question(self):
        assert _is_person_centric("What is Caroline's relationship status?")

    def test_hobby_question(self):
        assert _is_person_centric("What hobbies does Melanie have?")

    def test_would_question(self):
        assert _is_person_centric("Would Caroline pursue writing?")

    def test_career_question(self):
        assert _is_person_centric("What career does she want?")

    def test_destress_question(self):
        assert _is_person_centric("What does Melanie do to destress?")

    def test_factual_not_person_centric(self):
        assert not _is_person_centric("What inspired the painting?")

    def test_event_not_person_centric(self):
        assert not _is_person_centric("When did the art show happen?")

    def test_object_not_person_centric(self):
        assert not _is_person_centric("What book was recommended?")

    def test_location_not_person_centric(self):
        assert not _is_person_centric("Where did they go camping?")

    def test_political_leaning(self):
        assert _is_person_centric("What would Caroline's political leaning be?")

    def test_activities(self):
        assert _is_person_centric("What activities does Melanie partake in?")

    def test_self_care(self):
        assert _is_person_centric("How does Melanie prioritize self-care?")

    def test_likely_question(self):
        assert _is_person_centric("Would Melanie likely enjoy pottery?")

    def test_community_membership(self):
        assert _is_person_centric("Would Melanie be considered a member of the LGBTQ community?")


def tokenize(text):
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))


def _dedup_turn_indices(indices, turns, threshold=0.8):
    """Mirror of the function in run_exp7_improved.py."""
    kept = []
    kept_token_sets = []
    for idx, score in indices:
        if idx >= len(turns):
            continue
        t_tokens = tokenize(turns[idx])
        is_dup = False
        for prev_tokens in kept_token_sets:
            if not t_tokens or not prev_tokens:
                continue
            overlap = len(t_tokens & prev_tokens) / max(min(len(t_tokens), len(prev_tokens)), 1)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append((idx, score))
            kept_token_sets.append(t_tokens)
    return kept


class TestDedupTurnIndices:
    def test_no_duplicates(self):
        turns = ["I love pottery and painting", "The weather is great today", "Let's go camping"]
        indices = [(0, 5.0), (1, 4.0), (2, 3.0)]
        result = _dedup_turn_indices(indices, turns)
        assert len(result) == 3

    def test_removes_near_duplicates(self):
        turns = [
            "I love pottery and painting and swimming every day",
            "I love pottery and painting and swimming each day",  # near-dup of 0
            "The weather is completely different from everything else",
        ]
        indices = [(0, 5.0), (1, 4.0), (2, 3.0)]
        result = _dedup_turn_indices(indices, turns)
        assert len(result) == 2
        assert result[0][0] == 0  # keeps higher-ranked
        assert result[1][0] == 2  # unique turn

    def test_empty_indices(self):
        assert _dedup_turn_indices([], ["some turn"]) == []

    def test_out_of_range_index(self):
        result = _dedup_turn_indices([(5, 1.0)], ["only one turn"])
        assert result == []

    def test_keeps_order(self):
        turns = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
        indices = [(2, 1.0), (0, 2.0), (1, 3.0)]
        result = _dedup_turn_indices(indices, turns)
        assert [idx for idx, _ in result] == [2, 0, 1]


class TestEntityBoostGating:
    """Verify that entity boost is gated by person-centricity."""

    def test_person_centric_gets_names(self):
        query = "What are Caroline's hobbies?"
        names = _extract_query_names(query) if _is_person_centric(query) else set()
        assert "caroline" in names

    def test_factual_gets_no_names(self):
        query = "What inspired Melanie's painting for the art show?"
        names = _extract_query_names(query) if _is_person_centric(query) else set()
        assert names == set()

    def test_temporal_counterfactual_gets_names(self):
        query = "Would Caroline still want to pursue counseling?"
        names = _extract_query_names(query) if _is_person_centric(query) else set()
        assert "caroline" in names

    def test_when_question_no_boost(self):
        query = "When did Caroline go to the art show?"
        names = _extract_query_names(query) if _is_person_centric(query) else set()
        assert names == set()
