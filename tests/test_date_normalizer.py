# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Date normaliser tests

"""Comprehensive tests for date_normalizer.py (C4 runtime wrapper).

Covers rule-based normalisation, ML fallback, month arithmetic,
weekday resolution, confidence thresholds, and edge cases.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from date_normalizer import (
    VAGUE_DATE_RE,
    DateResult,
    _month_delta,
    _rule_based_normalise,
    extract_and_normalise,
    normalize_date_expression,
)


# ── Month arithmetic ────────────────────────────────────────────


class TestMonthDelta:
    def test_forward_one_month(self):
        assert _month_delta(date(2024, 1, 15), 1) == date(2024, 2, 15)

    def test_backward_one_month(self):
        assert _month_delta(date(2024, 3, 15), -1) == date(2024, 2, 15)

    def test_year_rollover_forward(self):
        assert _month_delta(date(2024, 11, 10), 2) == date(2025, 1, 10)

    def test_year_rollover_backward(self):
        assert _month_delta(date(2024, 1, 10), -2) == date(2023, 11, 10)

    def test_day_clamping_feb(self):
        # March 31 - 1 month = Feb 29 (2024 is leap year)
        assert _month_delta(date(2024, 3, 31), -1) == date(2024, 2, 29)

    def test_day_clamping_feb_non_leap(self):
        assert _month_delta(date(2023, 3, 31), -1) == date(2023, 2, 28)

    def test_day_clamping_30_day_month(self):
        # Jan 31 + 3 months = Apr 30
        assert _month_delta(date(2024, 1, 31), 3) == date(2024, 4, 30)

    def test_large_backward(self):
        assert _month_delta(date(2024, 6, 15), -18) == date(2022, 12, 15)

    def test_large_forward(self):
        assert _month_delta(date(2024, 6, 15), 24) == date(2026, 6, 15)

    def test_zero_delta(self):
        d = date(2024, 6, 15)
        assert _month_delta(d, 0) == d


# ── Quantified patterns ("N days/weeks/months/years ago") ──────


class TestQuantifiedPatterns:
    REF = date(2023, 4, 10)

    def test_days_ago(self):
        r = _rule_based_normalise("5 days ago", self.REF)
        assert r is not None
        assert r.iso_date == "2023-04-05"
        assert r.confidence == 0.95
        assert r.method == "rule"

    def test_day_singular(self):
        r = _rule_based_normalise("1 day ago", self.REF)
        assert r is not None
        assert r.iso_date == "2023-04-09"

    def test_weeks_ago(self):
        r = _rule_based_normalise("3 weeks ago", self.REF)
        assert r is not None
        assert r.iso_date == date(2023, 3, 20).isoformat()

    def test_months_ago(self):
        r = _rule_based_normalise("2 months ago", self.REF)
        assert r is not None
        assert r.iso_date == date(2023, 2, 10).isoformat()

    def test_years_ago(self):
        r = _rule_based_normalise("1 year ago", self.REF)
        assert r is not None
        assert r.iso_date == date(2022, 4, 10).isoformat()

    def test_about_modifier(self):
        r = _rule_based_normalise("about 3 days ago", self.REF)
        assert r is not None
        assert r.iso_date == "2023-04-07"
        assert r.confidence == 0.95

    def test_roughly_modifier(self):
        r = _rule_based_normalise("roughly 2 weeks ago", self.REF)
        assert r is not None
        assert r.iso_date == date(2023, 3, 27).isoformat()

    def test_approximately_modifier(self):
        r = _rule_based_normalise("approximately 1 month ago", self.REF)
        assert r is not None


# ── "A couple of" patterns ─────────────────────────────────────


class TestCouplePatterns:
    REF = date(2023, 6, 15)

    def test_couple_days(self):
        r = _rule_based_normalise("a couple of days ago", self.REF)
        assert r is not None
        assert r.iso_date == "2023-06-13"
        assert r.confidence == 0.9

    def test_couple_weeks(self):
        r = _rule_based_normalise("a couple of weeks ago", self.REF)
        assert r is not None
        assert r.iso_date == date(2023, 6, 1).isoformat()

    def test_couple_months(self):
        r = _rule_based_normalise("a couple of months ago", self.REF)
        assert r is not None
        assert r.iso_date == date(2023, 4, 15).isoformat()


# ── "A few" patterns ───────────────────────────────────────────


class TestFewPatterns:
    REF = date(2023, 6, 15)

    def test_few_days(self):
        r = _rule_based_normalise("a few days ago", self.REF)
        assert r is not None
        assert r.iso_date == "2023-06-12"
        assert r.confidence == 0.8

    def test_few_weeks(self):
        r = _rule_based_normalise("a few weeks ago", self.REF)
        assert r is not None
        assert r.iso_date == date(2023, 5, 25).isoformat()

    def test_few_months(self):
        r = _rule_based_normalise("a few months ago", self.REF)
        assert r is not None


# ── "Several" patterns ─────────────────────────────────────────


class TestSeveralPatterns:
    REF = date(2023, 6, 15)

    def test_several_days(self):
        r = _rule_based_normalise("several days ago", self.REF)
        assert r is not None
        assert r.iso_date == "2023-06-10"
        assert r.confidence == 0.7

    def test_several_weeks(self):
        r = _rule_based_normalise("several weeks ago", self.REF)
        assert r is not None

    def test_several_months(self):
        r = _rule_based_normalise("several months ago", self.REF)
        assert r is not None


# ── Weekday patterns ───────────────────────────────────────────


class TestWeekdayPatterns:
    def test_last_monday_from_wednesday(self):
        # Wednesday 2023-06-14 → last Monday = 2023-06-12
        r = _rule_based_normalise("last Monday", date(2023, 6, 14))
        assert r is not None
        assert r.iso_date == "2023-06-12"
        assert r.confidence == 0.95

    def test_last_friday_from_wednesday(self):
        # Wednesday 2023-06-14 → last Friday = 2023-06-09
        r = _rule_based_normalise("last Friday", date(2023, 6, 14))
        assert r is not None
        assert r.iso_date == "2023-06-09"

    def test_last_same_weekday(self):
        # Wednesday → last Wednesday = 7 days back
        r = _rule_based_normalise("last Wednesday", date(2023, 6, 14))
        assert r is not None
        assert r.iso_date == "2023-06-07"

    def test_this_past_saturday(self):
        r = _rule_based_normalise("this past Saturday", date(2023, 6, 14))
        assert r is not None
        assert r.iso_date == "2023-06-10"

    def test_last_sunday(self):
        r = _rule_based_normalise("last Sunday", date(2023, 6, 14))
        assert r is not None
        assert r.iso_date == "2023-06-11"

    def test_all_seven_weekdays(self):
        ref = date(2023, 6, 14)  # Wednesday
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]:
            r = _rule_based_normalise(f"last {day}", ref)
            assert r is not None, f"Failed for last {day}"
            parsed = date.fromisoformat(r.iso_date)
            assert parsed < ref, f"last {day} should be before reference"


# ── Simple fixed patterns ──────────────────────────────────────


class TestSimplePatterns:
    REF = date(2023, 6, 15)

    def test_yesterday(self):
        r = _rule_based_normalise("yesterday", self.REF)
        assert r is not None
        assert r.iso_date == "2023-06-14"

    def test_today(self):
        r = _rule_based_normalise("today", self.REF)
        assert r is not None
        assert r.iso_date == "2023-06-15"

    def test_the_other_day(self):
        r = _rule_based_normalise("the other day", self.REF)
        assert r is not None
        parsed = date.fromisoformat(r.iso_date)
        assert (self.REF - parsed).days == 3

    def test_not_long_ago(self):
        r = _rule_based_normalise("not long ago", self.REF)
        assert r is not None

    def test_recently(self):
        r = _rule_based_normalise("recently", self.REF)
        assert r is not None

    def test_earlier_this_week(self):
        r = _rule_based_normalise("earlier this week", self.REF)
        assert r is not None
        parsed = date.fromisoformat(r.iso_date)
        assert parsed <= self.REF

    def test_earlier_this_month(self):
        r = _rule_based_normalise("earlier this month", self.REF)
        assert r is not None
        parsed = date.fromisoformat(r.iso_date)
        assert parsed.month == self.REF.month

    def test_last_week(self):
        r = _rule_based_normalise("last week", self.REF)
        assert r is not None
        assert r.iso_date == "2023-06-08"

    def test_last_month(self):
        r = _rule_based_normalise("last month", self.REF)
        assert r is not None
        parsed = date.fromisoformat(r.iso_date)
        assert parsed.month == 5

    def test_last_year(self):
        r = _rule_based_normalise("last year", self.REF)
        assert r is not None
        parsed = date.fromisoformat(r.iso_date)
        assert parsed.year == 2022

    def test_this_week(self):
        r = _rule_based_normalise("this week", self.REF)
        assert r is not None

    def test_this_month(self):
        r = _rule_based_normalise("this month", self.REF)
        assert r is not None
        assert r.iso_date == "2023-06-01"

    def test_this_year(self):
        r = _rule_based_normalise("this year", self.REF)
        assert r is not None
        assert r.iso_date == "2023-01-01"


# ── No-match and edge cases ───────────────────────────────────


class TestEdgeCases:
    REF = date(2023, 6, 15)

    def test_empty_string(self):
        assert _rule_based_normalise("", self.REF) is None

    def test_whitespace_only(self):
        assert _rule_based_normalise("   ", self.REF) is None

    def test_unrecognised_expression(self):
        assert _rule_based_normalise("next full moon", self.REF) is None

    def test_partial_match(self):
        # "ago" alone should not match
        assert _rule_based_normalise("long ago", self.REF) is None

    def test_leading_trailing_whitespace(self):
        r = _rule_based_normalise("  3 days ago  ", self.REF)
        assert r is not None

    def test_case_insensitivity(self):
        r = _rule_based_normalise("LAST MONDAY", self.REF)
        assert r is not None

    def test_january_boundary(self):
        # last month from January → December
        r = _rule_based_normalise("last month", date(2023, 1, 15))
        assert r is not None
        parsed = date.fromisoformat(r.iso_date)
        assert parsed.month == 12
        assert parsed.year == 2022

    def test_large_number_days(self):
        r = _rule_based_normalise("365 days ago", self.REF)
        assert r is not None
        parsed = date.fromisoformat(r.iso_date)
        assert (self.REF - parsed).days == 365


# ── Public API ─────────────────────────────────────────────────


class TestNormalizeDateExpression:
    REF = date(2023, 4, 10)

    def test_rule_based_hit(self):
        r = normalize_date_expression("3 weeks ago", self.REF)
        assert r is not None
        assert r.method == "rule"

    def test_no_match_returns_none_without_model(self):
        # "spring 2023" has no rule and no model loaded
        with patch("date_normalizer._load_model", return_value=False):
            r = normalize_date_expression("spring 2023", self.REF)
            assert r is None

    def test_ml_fallback_when_rule_misses(self):
        mock_result = DateResult(iso_date="2023-03-15", confidence=0.8, method="model")
        with patch("date_normalizer._rule_based_normalise", return_value=None):
            with patch("date_normalizer._model_normalise", return_value=mock_result):
                r = normalize_date_expression("some exotic expression", self.REF)
                assert r is not None
                assert r.method == "model"


class TestExtractAndNormalise:
    REF = date(2023, 4, 10)

    def test_single_expression(self):
        results = extract_and_normalise("I did it 3 weeks ago.", self.REF)
        assert len(results) >= 1
        assert results[0].method == "rule"

    def test_multiple_expressions(self):
        text = "I started 2 months ago and finished yesterday."
        results = extract_and_normalise(text, self.REF)
        assert len(results) >= 2

    def test_no_expressions(self):
        results = extract_and_normalise("The sky is blue.", self.REF)
        assert results == []

    def test_empty_text(self):
        results = extract_and_normalise("", self.REF)
        assert results == []


# ── VAGUE_DATE_RE coverage ─────────────────────────────────────


class TestVagueDateRegex:
    def test_quantified_match(self):
        assert VAGUE_DATE_RE.search("about 3 weeks ago")

    def test_couple_match(self):
        assert VAGUE_DATE_RE.search("a couple of days ago")

    def test_few_match(self):
        assert VAGUE_DATE_RE.search("a few months ago")

    def test_weekday_match(self):
        assert VAGUE_DATE_RE.search("last Tuesday")

    def test_yesterday_match(self):
        assert VAGUE_DATE_RE.search("yesterday")

    def test_recently_match(self):
        assert VAGUE_DATE_RE.search("recently")

    def test_this_week_match(self):
        assert VAGUE_DATE_RE.search("earlier this week")

    def test_some_time_ago_match(self):
        assert VAGUE_DATE_RE.search("some time ago")

    def test_no_match_plain_text(self):
        assert VAGUE_DATE_RE.search("The cat sat on the mat") is None

    def test_no_match_iso_date(self):
        assert VAGUE_DATE_RE.search("2023-04-10") is None


# ── ML model loading paths ─────────────────────────────────────


class TestModelLoading:
    def test_model_dir_missing(self):
        with patch("date_normalizer._MODEL_DIR") as mock_dir:
            mock_dir.__truediv__ = lambda self, x: MagicMock(exists=lambda: False)
            # Reset global state
            import date_normalizer
            date_normalizer._model = None
            date_normalizer._tokenizer = None
            result = date_normalizer._load_model()
            # Model dir check happens inside, but we test the fallback
            # When model is None and loading fails, normalize returns None
            r = date_normalizer._model_normalise("test", date(2023, 1, 1))
            assert r is None

    def test_already_loaded_returns_true(self):
        import date_normalizer
        date_normalizer._model = MagicMock()  # pretend loaded
        assert date_normalizer._load_model() is True
        date_normalizer._model = None  # cleanup

    def test_load_model_exception_returns_false(self):
        """_load_model returns False when model construction raises."""
        import date_normalizer
        date_normalizer._model = None
        date_normalizer._tokenizer = None
        model_pt = MagicMock(exists=lambda: True)
        mock_dir = MagicMock()
        mock_dir.__truediv__ = lambda self, x: model_pt
        with patch.object(date_normalizer, "_MODEL_DIR", mock_dir):
            with patch("date_normalizer.json.load", side_effect=OSError("corrupt")):
                result = date_normalizer._load_model()
                assert result is False

    def _mock_digit_model(self, digits: list[int], conf: float = 0.9):
        """Helper: set up mock model returning specific digit predictions."""
        import date_normalizer

        mock_model = MagicMock()
        digit_logits = []
        for d in digits:
            logit = torch.zeros(1, 10)
            logit[0, d] = 5.0
            digit_logits.append(logit)
        mock_model.return_value = (digit_logits, torch.tensor([conf]))

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.zeros(1, 64, dtype=torch.long),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }

        date_normalizer._model = mock_model
        date_normalizer._tokenizer = mock_tok

    def _cleanup_model(self):
        import date_normalizer
        date_normalizer._model = None
        date_normalizer._tokenizer = None

    def test_model_normalise_full_flow(self):
        """_model_normalise with valid digits → DateResult."""
        self._mock_digit_model([2, 0, 2, 3, 0, 4, 1, 0])  # 2023-04-10

        import date_normalizer
        r = date_normalizer._model_normalise("test expression", date(2023, 6, 1))
        assert r is not None
        assert r.iso_date == "2023-04-10"
        assert r.confidence == pytest.approx(0.9, abs=0.01)
        assert r.method == "model"
        self._cleanup_model()

    def test_model_normalise_invalid_date(self):
        """_model_normalise returns None for invalid reconstructed dates."""
        self._mock_digit_model([2, 0, 2, 3, 1, 3, 3, 2])  # 2023-13-32

        import date_normalizer
        r = date_normalizer._model_normalise("bad date", date(2023, 6, 1))
        assert r is None
        self._cleanup_model()

    def test_model_normalise_month_zero(self):
        """Month 00 → ValueError → None."""
        self._mock_digit_model([2, 0, 2, 3, 0, 0, 1, 5])  # 2023-00-15

        import date_normalizer
        r = date_normalizer._model_normalise("zero month", date(2023, 6, 1))
        assert r is None
        self._cleanup_model()


# ── Simple pattern resolver exception path ─────────────────────


class TestSimpleResolverException:
    def test_resolver_value_error_continues(self):
        """When a resolver raises ValueError, the pattern is skipped."""
        import date_normalizer

        # Inject a broken resolver for "earlier this month" that raises
        original_simple = dict(date_normalizer._SIMPLE_RE)

        def _bad_resolver(ref):
            raise ValueError("intentional")

        broken_re = {}
        for pattern, resolver in date_normalizer._SIMPLE_RE.items():
            if "earlier.*month" in pattern.pattern:
                broken_re[pattern] = _bad_resolver
            else:
                broken_re[pattern] = resolver

        date_normalizer._SIMPLE_RE = broken_re
        try:
            # "earlier this month" would normally match; now it raises
            r = _rule_based_normalise("earlier this month", date(2023, 6, 15))
            # Should either be None (if no other pattern matches)
            # or match a different pattern
            # The test verifies no exception escapes
        finally:
            date_normalizer._SIMPLE_RE = original_simple


# ── DateResult dataclass ───────────────────────────────────────


class TestDateResult:
    def test_fields(self):
        r = DateResult(iso_date="2023-04-10", confidence=0.95, method="rule")
        assert r.iso_date == "2023-04-10"
        assert r.confidence == 0.95
        assert r.method == "rule"

    def test_equality(self):
        a = DateResult(iso_date="2023-04-10", confidence=0.95, method="rule")
        b = DateResult(iso_date="2023-04-10", confidence=0.95, method="rule")
        assert a == b
