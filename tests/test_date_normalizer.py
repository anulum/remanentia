# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for date normaliser

"""Comprehensive tests for date_normalizer.py (C4 runtime wrapper).

Covers rule-based normalisation, ML fallback, month arithmetic,
weekday resolution, confidence thresholds, and edge cases.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

import date_normalizer

from date_normalizer import (
    VAGUE_DATE_RE,
    DateResult,
    _rule_based_normalise_python,
    _month_delta,
    _parse_session_date,
    _parse_session_datetime,
    _rule_based_normalise,
    extract_and_normalise,
    normalise_in_context,
    normalize_date_expression,
)


def _reset_model() -> None:
    date_normalizer._model = None
    date_normalizer._tokenizer = None


@contextmanager
def _model_at(model_dir: Path) -> Iterator[None]:
    _reset_model()
    original = date_normalizer._MODEL_DIR
    date_normalizer._MODEL_DIR = model_dir
    try:
        yield
    finally:
        date_normalizer._MODEL_DIR = original
        _reset_model()


def _write_checkpoint(root: Path, digits: list[int], confidence: float = 0.9) -> Path:
    """Write a tiny real BERT tokenizer, backbone and production date head."""
    model_dir = root / "date-normalizer-v1"
    backbone_dir = root / "backbone"
    model_dir.mkdir(parents=True)
    backbone_dir.mkdir(parents=True)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "reference",
        "expression",
        "spring",
        "date",
    ]
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True).save_pretrained(model_dir)
    BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=96,
        )
    ).save_pretrained(backbone_dir)
    (model_dir / "config.json").write_text(
        json.dumps({"model_name": str(backbone_dir), "num_digits": 8}), encoding="utf-8"
    )
    model = date_normalizer._build_model(str(backbone_dir), 8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        dynamic_model = cast(Any, model)
        for head, digit in zip(dynamic_model.digit_heads, digits, strict=True):
            head.bias[digit] = 5.0
        dynamic_model.confidence_head[-2].bias.fill_(torch.logit(torch.tensor(confidence)))
    torch.save(model.state_dict(), model_dir / "model.pt")
    return model_dir


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
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
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

    def test_last_year_from_leap_day_clamps(self):
        # 29 Feb 2024 has no counterpart in 2023 — "last year" must clamp the
        # day to 28 (as its sibling "last month" clamps via _month_delta),
        # not silently fail to resolve.
        r = _rule_based_normalise("last year", date(2024, 2, 29))
        assert r is not None
        assert r.iso_date == "2023-02-28"

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

    def test_no_match_returns_none_without_model(self, tmp_path: Path):
        # "spring 2023" has no rule and no model loaded
        with _model_at(tmp_path / "missing"):
            r = normalize_date_expression("spring 2023", self.REF)
            assert r is None

    def test_ml_fallback_when_rule_misses(self, tmp_path: Path):
        model_dir = _write_checkpoint(tmp_path, [2, 0, 2, 3, 0, 3, 1, 5], confidence=0.8)
        with _model_at(model_dir):
            r = normalize_date_expression("some exotic expression", self.REF)
        assert r is not None
        assert r.iso_date == "2023-03-15"
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

    def test_matched_but_unresolved_expression_is_omitted(self, tmp_path: Path):
        with _model_at(tmp_path / "missing"):
            results = extract_and_normalise("It happened some time ago.", self.REF)
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
    def test_model_dir_missing(self, tmp_path: Path):
        with _model_at(tmp_path / "missing"):
            assert date_normalizer._model_normalise("test", date(2023, 1, 1)) is None

    def test_loaded_real_checkpoint_is_reused(self, tmp_path: Path):
        model_dir = _write_checkpoint(tmp_path, [2, 0, 2, 3, 0, 4, 1, 0])
        with _model_at(model_dir):
            assert date_normalizer._load_model() is True
            loaded = date_normalizer._model
            assert date_normalizer._load_model() is True
            assert date_normalizer._model is loaded

    def test_model_normalise_returns_none_when_real_cache_is_incomplete(self, tmp_path: Path):
        model_dir = _write_checkpoint(tmp_path, [2, 0, 2, 3, 0, 4, 1, 0])
        with _model_at(model_dir):
            assert date_normalizer._load_model() is True
            date_normalizer._tokenizer = None
            assert date_normalizer._model_normalise("spring", date(2023, 1, 1)) is None

    def test_corrupt_config_returns_false(self, tmp_path: Path):
        model_dir = tmp_path / "corrupt"
        model_dir.mkdir()
        (model_dir / "model.pt").write_bytes(b"checkpoint")
        (model_dir / "config.json").write_text("{", encoding="utf-8")
        with _model_at(model_dir):
            assert date_normalizer._load_model() is False

    def test_model_normalise_full_flow(self, tmp_path: Path):
        model_dir = _write_checkpoint(tmp_path, [2, 0, 2, 3, 0, 4, 1, 0], confidence=0.9)
        with _model_at(model_dir):
            r = date_normalizer._model_normalise("test expression", date(2023, 6, 1))
        assert r is not None
        assert r.iso_date == "2023-04-10"
        assert r.confidence == pytest.approx(0.9, abs=0.01)
        assert r.method == "model"

    @pytest.mark.parametrize(
        "digits",
        [[2, 0, 2, 3, 1, 3, 3, 2], [2, 0, 2, 3, 0, 0, 1, 5]],
    )
    def test_model_normalise_invalid_dates(self, tmp_path: Path, digits: list[int]):
        model_dir = _write_checkpoint(tmp_path, digits)
        with _model_at(model_dir):
            r = date_normalizer._model_normalise("bad date", date(2023, 6, 1))
        assert r is None


# ── Simple pattern resolver exception path ─────────────────────


class TestSimpleResolverException:
    def test_resolver_value_error_continues(self):
        """A resolver that raises is skipped; a later pattern still resolves.

        The text "earlier this month" matches both the "earlier this month"
        pattern and the later "this month" pattern. Breaking the first and
        feeding that text proves the ``except ... continue`` swallows the
        error and falls through to "this month" (day 1), rather than
        propagating or returning None.
        """
        import date_normalizer

        original_simple = dict(date_normalizer._SIMPLE_RE)
        original_import = date_normalizer.import_module

        def _bad_resolver(ref):
            raise ValueError("intentional")

        def _force_python_fallback(name, *args, **kwargs):
            # This test exercises the Python fallback's ``except ValueError:
            # continue`` fall-through. When the compiled ``remanentia_temporal``
            # extension is installed, ``_rule_based_normalise`` resolves via Rust
            # and returns before the patched ``_SIMPLE_RE`` is ever reached, so
            # force the fallback here to test the branch deterministically
            # regardless of whether the extension is present.
            if name == "remanentia_temporal":
                raise ImportError(name)
            return original_import(name, *args, **kwargs)

        # Exact-match the compiled pattern's source so the injection actually
        # lands on the intended resolver (a substring like "earlier.*month"
        # never occurs in the real r"\bearlier\s+this\s+month\b" source).
        broken_re = {}
        for pattern, resolver in date_normalizer._SIMPLE_RE.items():
            if pattern.pattern == r"\bearlier\s+this\s+month\b":
                broken_re[pattern] = _bad_resolver
            else:
                broken_re[pattern] = resolver

        date_normalizer._SIMPLE_RE = broken_re
        date_normalizer.import_module = _force_python_fallback
        try:
            result = _rule_based_normalise("earlier this month", date(2023, 6, 15))
            assert result is not None
            # Fell through to the "this month" pattern (first of the month).
            assert result.iso_date == "2023-06-01"
        finally:
            date_normalizer._SIMPLE_RE = original_simple
            date_normalizer.import_module = original_import


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


# ── Missing patterns: pipeline, roundtrip ─────────────────────


class TestDateNormalizerPipeline:
    def test_feeds_temporal_graph(self):
        """date_normalizer results feed into temporal_graph.parse_dates."""
        from temporal_graph import parse_dates
        from datetime import date

        # Vague expression → date_normalizer → temporal_graph
        result = parse_dates("about 3 weeks ago we started", date(2026, 3, 30))
        assert len(result) >= 1
        assert any("2026-03" in d for d in result)

    def test_feeds_fact_decomposer(self):
        """date_normalizer dates appear in decomposed facts."""
        from fact_decomposer import decompose_sessions

        facts = decompose_sessions(
            [[{"role": "user", "content": "I moved to Berlin on January 15, 2024."}]]
        )
        dated = [f for f in facts if f.valid_from]
        assert len(dated) >= 1

    def test_roundtrip_rule_based(self):
        from date_normalizer import _rule_based_normalise
        from datetime import date

        ref = date(2026, 3, 30)
        for expr in ["yesterday", "last week", "3 days ago", "a few months ago"]:
            result = _rule_based_normalise(expr, ref)
            if result is not None:
                assert len(result.iso_date) == 10
                assert result.confidence > 0


# ── _parse_session_date ───────────────────────────────────────


class TestParseSessionDate:
    def test_longmemeval_format(self):
        d = _parse_session_date("2023/05/28 (Sun) 21:04")
        assert d == date(2023, 5, 28)

    def test_iso_format(self):
        d = _parse_session_date("2023-05-28")
        assert d == date(2023, 5, 28)

    def test_slash_no_time(self):
        d = _parse_session_date("2023/05/28")
        assert d == date(2023, 5, 28)

    def test_empty_string(self):
        assert _parse_session_date("") is None

    def test_whitespace_only(self):
        assert _parse_session_date("   ") is None

    def test_invalid_date(self):
        assert _parse_session_date("2023/13/45") is None

    def test_garbage(self):
        assert _parse_session_date("not a date at all") is None

    def test_leading_trailing_whitespace(self):
        d = _parse_session_date("  2023/05/28 (Sun) 21:04  ")
        assert d == date(2023, 5, 28)


# ── _parse_session_datetime (Task #32 — intraday) ──────────────


class TestParseSessionDatetime:
    def test_longmemeval_with_time(self):
        from datetime import datetime as dt

        d = _parse_session_datetime("2023/05/28 (Sun) 21:04")
        assert d == dt(2023, 5, 28, 21, 4)

    def test_longmemeval_no_dow(self):
        from datetime import datetime as dt

        d = _parse_session_datetime("2023/05/28 09:38")
        assert d == dt(2023, 5, 28, 9, 38)

    def test_iso_with_T(self):
        from datetime import datetime as dt

        d = _parse_session_datetime("2023-05-28T14:30")
        assert d == dt(2023, 5, 28, 14, 30)

    def test_date_only_defaults_midnight(self):
        from datetime import datetime as dt

        d = _parse_session_datetime("2023-05-28")
        assert d == dt(2023, 5, 28, 0, 0)

    def test_intraday_ordering_via_iso(self):
        """Same-day timestamps sort correctly via isoformat."""
        a = _parse_session_datetime("2023/05/22 (Mon) 09:38")
        b = _parse_session_datetime("2023/05/22 (Mon) 11:58")
        assert a < b
        assert a.isoformat() < b.isoformat()

    def test_empty(self):
        assert _parse_session_datetime("") is None

    def test_whitespace(self):
        assert _parse_session_datetime("   ") is None

    def test_invalid_date(self):
        assert _parse_session_datetime("2023/13/45 (Bad) 99:99") is None

    def test_garbage(self):
        assert _parse_session_datetime("not a date") is None

    def test_leading_trailing_whitespace(self):
        from datetime import datetime as dt

        d = _parse_session_datetime("  2023/05/28 (Sun) 21:04  ")
        assert d == dt(2023, 5, 28, 21, 4)


# ── normalise_in_context ─────────────────────────────────────


class TestNormaliseInContext:
    def test_yesterday_resolved(self):
        text = "I went to the gym yesterday"
        result = normalise_in_context(text, "2023/05/28 (Sun) 21:04")
        assert "2023-05-27" in result
        assert "yesterday" in result  # original kept as annotation

    def test_last_saturday(self):
        text = "I went to the gym last Saturday"
        result = normalise_in_context(text, "2023/05/28 (Sun) 21:04")
        assert "2023-05-27" in result  # last Saturday from Sunday 28th = 27th
        assert "last Saturday" in result

    def test_multiple_expressions(self):
        text = "I started yesterday and finished 3 days ago"
        result = normalise_in_context(text, "2023/06/15 (Thu) 10:00")
        assert "2023-06-14" in result  # yesterday
        assert "2023-06-12" in result  # 3 days ago

    def test_no_vague_expressions(self):
        text = "The weather is nice today."
        # "today" is a vague expression, should be resolved
        result = normalise_in_context(text, "2023/06/15 (Thu) 10:00")
        assert "2023-06-15" in result

    def test_plain_text_unchanged(self):
        text = "The cat sat on the mat."
        result = normalise_in_context(text, "2023/06/15 (Thu) 10:00")
        assert result == text

    def test_matched_but_unresolved_expression_is_unchanged(self, tmp_path: Path):
        text = "It happened some time ago."
        with _model_at(tmp_path / "missing"):
            result = normalise_in_context(text, "2023/06/15 (Thu) 10:00")
        assert result == text

    def test_empty_reference_date(self):
        text = "I went there yesterday"
        result = normalise_in_context(text, "")
        assert result == text  # unchanged

    def test_invalid_reference_date(self):
        text = "I went there yesterday"
        result = normalise_in_context(text, "not a date")
        assert result == text  # unchanged

    def test_iso_reference_date(self):
        text = "I did it last week"
        result = normalise_in_context(text, "2023-06-15")
        assert "2023-06-08" in result

    def test_weeks_ago(self):
        text = "We met about 3 weeks ago to discuss the project"
        result = normalise_in_context(text, "2023/04/10 (Mon) 17:50")
        assert "2023-03-20" in result

    def test_annotation_format(self):
        """Resolved expressions use 'on YYYY-MM-DD (original)' format."""
        text = "I visited yesterday"
        result = normalise_in_context(text, "2023-06-15")
        assert "on 2023-06-14 (yesterday)" in result

    def test_preserves_surrounding_text(self):
        text = "Before that, yesterday I went shopping, and it was great."
        result = normalise_in_context(text, "2023-06-15")
        assert result.startswith("Before that,")
        assert result.endswith("and it was great.")
        assert "2023-06-14" in result


# ── Benchmark index carries session dates ─────────────────────


class TestBenchIndexSessionDates:
    def test_documents_carry_session_date(self):
        from bench_longmemeval import _build_index_for_question

        sessions = [
            [{"role": "user", "content": "I went to the gym last Saturday and it was great."}],
            [{"role": "user", "content": "Today I bought a new car from the dealer."}],
        ]
        dates = ["2023/05/28 (Sun) 21:04", "2023/06/01 (Thu) 10:00"]

        idx = _build_index_for_question(sessions, haystack_dates=dates)
        assert len(idx.documents) == 2
        assert idx.documents[0].date == "2023/05/28 (Sun) 21:04"
        assert idx.documents[1].date == "2023/06/01 (Thu) 10:00"

    def test_documents_without_dates(self):
        from bench_longmemeval import _build_index_for_question

        sessions = [
            [{"role": "user", "content": "I went to the gym last Saturday and it was great."}],
        ]
        idx = _build_index_for_question(sessions)
        assert idx.documents[0].date == ""

    def test_vague_dates_resolved_in_content(self):
        """Content indexed with resolved dates when haystack_dates provided."""
        from bench_longmemeval import _build_index_for_question

        sessions = [
            [{"role": "user", "content": "I went to the gym yesterday and it was great fun."}],
        ]
        dates = ["2023/05/28 (Sun) 21:04"]

        idx = _build_index_for_question(sessions, haystack_dates=dates)
        # The indexed content should contain the resolved date
        content = idx.documents[0].paragraphs[0]
        assert "2023-05-27" in content

    def test_build_context_includes_dates(self):
        """_build_context includes session dates in headers."""
        from bench_longmemeval import _build_context

        sessions = [
            [{"role": "user", "content": "Hello world, this is a test message."}],
        ]
        dates = ["2023/05/28 (Sun) 21:04"]

        context = _build_context(
            "test question",
            None,
            [],
            sessions,
            "temporal-reasoning",
            haystack_dates=dates,
        )
        assert "2023/05/28 (Sun) 21:04" in context

    def test_build_context_without_dates(self):
        """_build_context works without dates."""
        from bench_longmemeval import _build_context

        sessions = [
            [{"role": "user", "content": "Hello world, this is a test message."}],
        ]
        context = _build_context(
            "test question",
            None,
            [],
            sessions,
            "temporal-reasoning",
        )
        assert "Session 1" in context

    def test_sessions_sorted_chronologically(self):
        """Sessions appear in chronological order regardless of input order."""
        from bench_longmemeval import _build_context

        sessions = [
            [{"role": "user", "content": "This is the LATER session content here."}],
            [{"role": "user", "content": "This is the EARLIER session content here."}],
        ]
        dates = ["2023/06/15 (Thu) 10:00", "2023/01/10 (Tue) 08:00"]

        context = _build_context(
            "which came first",
            None,
            [],
            sessions,
            "temporal-reasoning",
            haystack_dates=dates,
        )
        # Earlier date should appear before later date in output
        pos_earlier = context.find("2023/01/10")
        pos_later = context.find("2023/06/15")
        assert pos_earlier < pos_later, "Sessions not in chronological order"

    def test_sessions_unsorted_without_dates(self):
        """Without dates, sessions keep original order."""
        from bench_longmemeval import _build_context

        sessions = [
            [{"role": "user", "content": "First input session with enough content."}],
            [{"role": "user", "content": "Second input session with enough content."}],
        ]
        context = _build_context(
            "test",
            None,
            [],
            sessions,
            "temporal-reasoning",
        )
        pos_first = context.find("First input")
        pos_second = context.find("Second input")
        assert pos_first < pos_second


# ── temporal_graph.extract_events with reference_date ─────────


class TestExtractEventsWithReferenceDate:
    def test_vague_dates_resolved(self):
        """extract_events resolves 'yesterday' when reference_date provided."""
        from temporal_graph import TemporalGraph

        tg = TemporalGraph()
        events = tg.extract_events(
            "I went to the gym yesterday.",
            "session_0",
            reference_date="2023/05/28 (Sun) 21:04",
        )
        # Should find at least the resolved date 2023-05-27
        dates = [e.date for e in events]
        assert any("2023-05-27" in d for d in dates)

    def test_no_reference_date(self):
        """extract_events without reference_date still works for explicit dates."""
        from temporal_graph import TemporalGraph

        tg = TemporalGraph()
        events = tg.extract_events(
            "On 2023-05-20 I went to the gym.",
            "session_0",
        )
        assert any(e.date == "2023-05-20" for e in events)

    def test_empty_reference_date(self):
        """Empty reference_date string behaves like no reference."""
        from temporal_graph import TemporalGraph

        tg = TemporalGraph()
        events = tg.extract_events(
            "On 2023-05-20 I went to the gym.",
            "session_0",
            reference_date="",
        )
        assert any(e.date == "2023-05-20" for e in events)


class TestPythonRuleFallbackCoverage:
    REF = date(2023, 4, 10)

    def test_quantified_and_vague_rules_without_native_extension(self):
        expectations = {
            "2 days ago": "2023-04-08",
            "2 weeks ago": "2023-03-27",
            "2 months ago": "2023-02-10",
            "2 years ago": "2021-04-10",
            "a couple of days ago": "2023-04-08",
            "a couple of weeks ago": "2023-03-27",
            "a couple of months ago": "2023-02-10",
            "a few days ago": "2023-04-07",
            "a few weeks ago": "2023-03-20",
            "a few months ago": "2023-01-10",
            "several days ago": "2023-04-05",
            "several weeks ago": "2023-03-06",
            "several months ago": "2022-11-10",
            "last Monday": "2023-04-03",
            "last Tuesday": "2023-04-04",
            "yesterday": "2023-04-09",
        }
        for expr, expected in expectations.items():
            result = _rule_based_normalise_python(expr, self.REF)
            assert result is not None
            assert result.iso_date == expected
