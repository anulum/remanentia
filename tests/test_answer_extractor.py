# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for answer_extractor.py

from __future__ import annotations

import pytest

from answer_extractor import (
    extract_all_candidates,
    extract_answer,
    _extract_date_answer,
    _extract_name_answer,
    _extract_number_answer,
    _extract_percentage_answer,
    _extract_version_answer,
    _extract_yes_no,
    _is_how_many_question,
    _is_version_question,
    _is_when_question,
    _is_who_question,
    _is_yes_no_question,
)


# ── Question type detection ──────────────────────────────────────


class TestQuestionTypes:
    def test_when(self):
        assert _is_when_question("when did we fix the bug")
        assert _is_when_question("what date was the release")
        assert not _is_when_question("where is the file")

    def test_how_many(self):
        assert _is_how_many_question("how many tests passed")
        assert _is_how_many_question("how much did it cost")
        assert not _is_how_many_question("how does it work")

    def test_version(self):
        assert _is_version_question("what version is deployed")
        assert _is_version_question("latest release number")
        assert not _is_version_question("what is the status")

    def test_who(self):
        assert _is_who_question("who wrote this code")
        assert not _is_who_question("what is this code")

    def test_yes_no(self):
        assert _is_yes_no_question("is the daemon running")
        assert _is_yes_no_question("did we fix the bug")
        assert _is_yes_no_question("can we deploy now")
        assert not _is_yes_no_question("what is the status")


# ── Date extraction ──────────────────────────────────────────────


class TestExtractDate:
    def test_iso_date(self):
        assert _extract_date_answer("Fixed on 2026-03-15 during the sprint.") == "2026-03-15"

    def test_english_date(self):
        result = _extract_date_answer("Released on March 15, 2026.")
        assert result is not None
        assert "March" in result

    def test_no_date(self):
        assert _extract_date_answer("No dates in this text.") is None

    def test_multiple_dates_returns_first(self):
        assert _extract_date_answer("Started 2026-03-10, finished 2026-03-15.") == "2026-03-10"


# ── Number extraction ────────────────────────────────────────────


class TestExtractNumber:
    def test_simple_number(self):
        result = _extract_number_answer("We processed 1986 questions.", "how many questions")
        assert result is not None
        assert "1986" in result

    def test_decimal_number(self):
        result = _extract_number_answer("Score was 66.4 on the benchmark.", "what score")
        assert result is not None
        assert "66.4" in result

    def test_filters_years(self):
        result = _extract_number_answer("In 2026 we processed 500 items.", "how many items")
        assert result is not None
        assert result != "2026"


# ── Version extraction ───────────────────────────────────────────


class TestExtractVersion:
    def test_semver(self):
        assert _extract_version_answer("Released v3.9.0 to PyPI.") == "v3.9.0"

    def test_two_part(self):
        assert _extract_version_answer("Updated to v0.2.") == "v0.2"

    def test_no_version(self):
        assert _extract_version_answer("No version here.") is None


# ── Percentage extraction ────────────────────────────────────────


class TestExtractPercentage:
    def test_integer_percent(self):
        assert _extract_percentage_answer("Accuracy was 100% on benchmark.") == "100%"

    def test_decimal_percent(self):
        assert _extract_percentage_answer("LOCOMO score: 66.4%.") == "66.4%"

    def test_no_percent(self):
        assert _extract_percentage_answer("No percentages here.") is None


# ── Name extraction ──────────────────────────────────────────────


class TestExtractName:
    def test_multi_word_name(self):
        result = _extract_name_answer("Written by Miroslav Sotek in 2026.", "who wrote")
        assert result is not None
        assert "Miroslav" in result

    def test_no_name(self):
        assert _extract_name_answer("no names here at all.", "who did") is None


# ── Yes/No extraction ───────────────────────────────────────────


class TestExtractYesNo:
    def test_negation(self):
        result = _extract_yes_no("The daemon is not running.", "is daemon running")
        assert result == "No"

    def test_affirmation(self):
        result = _extract_yes_no("The daemon is running.", "is daemon running")
        assert result == "Yes"


# ── Full extract_answer ──────────────────────────────────────────


class TestExtractAnswer:
    def test_when_question(self):
        result = extract_answer(
            "When did we fix the STDP bug?",
            "The STDP bug was fixed on 2026-03-15 by correcting the mask.",
        )
        assert result == "2026-03-15"

    def test_version_question(self):
        result = extract_answer(
            "What version was released?",
            "We released v3.9.0 to PyPI on March 15.",
        )
        assert result == "v3.9.0"

    def test_percentage_question(self):
        result = extract_answer(
            "What is the LOCOMO accuracy?",
            "The LOCOMO benchmark scored 66.4% on 1,986 questions.",
        )
        assert result == "66.4%"

    def test_yes_no_question(self):
        result = extract_answer(
            "Is the SNN used for retrieval?",
            "SNN retrieval is not used. Weight set to 0.00.",
        )
        assert result == "No"

    def test_generic_fallback(self):
        result = extract_answer(
            "What happened to the daemon?",
            "Daemon killed. GPU freed. v0.2.0 released. 66.4% accuracy.",
        )
        assert result is not None


# ── extract_all_candidates ───────────────────────────────────────


class TestExtractAllCandidates:
    def test_mixed_types(self):
        candidates = extract_all_candidates(
            "STDP results",
            "On 2026-03-15, accuracy reached 66.4% with v3.9.0. Processed 1,986 items.",
        )
        types = {c["type"] for c in candidates}
        assert "date" in types
        assert "percentage" in types
        assert "version" in types

    def test_proximity_boost(self):
        candidates = extract_all_candidates(
            "STDP accuracy",
            "STDP scored 66.4%. BM25 scored 48.9%.",
        )
        # 66.4% should score higher (closer to "STDP")
        pcts = [c for c in candidates if c["type"] == "percentage"]
        if len(pcts) >= 2:
            assert pcts[0]["answer"] == "66.4%"

    def test_empty(self):
        assert extract_all_candidates("query", "no extractable answers") == []
