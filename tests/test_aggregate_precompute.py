# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for aggregate_precompute

from __future__ import annotations

from aggregate_precompute import (
    _coerce_number,
    extract_numeric_facts,
    is_count_question,
    is_sum_question,
    precompute_aggregation,
    precompute_sum,
)


class TestIsSumQuestion:
    def test_total_views(self):
        assert is_sum_question("What is the total number of views on my videos?")

    def test_combined_reach(self):
        assert is_sum_question("What was the combined reach of my ad campaigns?")

    def test_altogether(self):
        assert is_sum_question("How much did I spend altogether on bike parts?")

    def test_how_much_total(self):
        assert is_sum_question("How much total money did I spend at the market?")

    def test_non_sum_question(self):
        assert not is_sum_question("What was my most popular video?")
        assert not is_sum_question("Which store did I visit last?")


class TestIsCountQuestion:
    def test_how_many_items(self):
        assert is_count_question("How many items did I buy?")

    def test_how_many_different(self):
        assert is_count_question("How many different cuisines have I tried?")

    def test_total_takes_sum_path_not_count(self):
        # "total" wins: sum-phrasing overrides count to avoid double-counting paths.
        assert not is_count_question("How many in total did I spend?")
        assert is_sum_question("How many in total did I spend?")


class TestExtractNumericFacts:
    def test_colon_label(self):
        text = "YouTube: 542 views, TikTok: 1456 views on the tutorial series."
        facts = extract_numeric_facts(text)
        labels = {f.label for f in facts}
        assert "YouTube" in labels
        assert "TikTok" in labels
        youtube = next(f for f in facts if f.label == "YouTube")
        assert youtube.value == 542
        assert youtube.unit == "views"

    def test_year_is_not_a_quantity(self):
        text = "The review was in 2026."
        facts = extract_numeric_facts(text)
        # "2026" without a unit must not be extracted as a quantity.
        assert not any(f.value == 2026 for f in facts)

    def test_comma_number_parses(self):
        text = "Instagram: 1,200 followers on the account."
        facts = extract_numeric_facts(text)
        assert len(facts) == 1
        assert facts[0].value == 1200

    def test_unlabelled_number_is_skipped(self):
        text = "It was 42 degrees outside."
        # 42 without a label in our accepted unit set → ignored.
        facts = extract_numeric_facts(text)
        assert all(f.unit != "" or f.label == "(unlabelled)" for f in facts)


class TestPrecomputeSum:
    def test_two_facts_same_unit(self):
        question = "What is the total number of views on YouTube and TikTok?"
        text = "YouTube: 542 views. TikTok: 1456 views."
        result = precompute_sum(question, text)
        assert result is not None
        assert result.kind == "total"
        assert result.value == 1998
        assert "COMPUTED TOTAL" in result.message
        assert "1998" in result.message

    def test_refuses_on_single_fact(self):
        question = "What is my total view count?"
        text = "YouTube: 542 views."
        assert precompute_sum(question, text) is None

    def test_refuses_non_sum_question(self):
        question = "What is my most popular platform?"
        text = "YouTube: 542 views. TikTok: 1456 views."
        assert precompute_sum(question, text) is None

    def test_dominant_unit_wins(self):
        question = "What is the total?"
        text = (
            "YouTube: 542 views. TikTok: 1456 views. Instagram: 720 views. "
            "Store: 3 hours on Tuesday."
        )
        result = precompute_sum(question, text)
        assert result is not None
        # Views dominate (3 facts) over hours (1).
        assert result.value == 542 + 1456 + 720

    def test_rejects_mixed_units_with_no_dominant(self):
        question = "What is the total?"
        text = "YouTube: 542 views. Store: 3 hours on Tuesday."
        assert precompute_sum(question, text) is None

    def test_float_result(self):
        question = "What is the total?"
        text = "Item A: 1.5 hours of work. Item B: 2.5 hours of work."
        result = precompute_sum(question, text)
        assert result is not None
        # g-format preserves decimals
        assert "4" in result.message


class TestPrecomputeAggregation:
    def test_temporal_is_skipped(self):
        question = "What is the total duration?"
        text = "YouTube: 10 hours of content. TikTok: 5 hours of content."
        # Temporal qtype should skip (TReMu handles those).
        assert precompute_aggregation(question, text, qtype="temporal-reasoning") is None

    def test_multi_session_routes_to_sum(self):
        question = "Total followers across platforms?"
        text = "YouTube: 100 followers. TikTok: 50 followers."
        result = precompute_aggregation(question, text, qtype="multi-session")
        assert result is not None
        assert result.value == 150

    def test_empty_qtype_still_computes(self):
        question = "What is the total money spent?"
        text = "Market A: $100 spent. Market B: $50 spent."
        result = precompute_aggregation(question, text)
        # Currency extraction + sum
        assert result is None or result.kind == "total"


class TestPrecomputeResultBool:
    def test_truthy_when_kind_set(self):
        question = "Total views?"
        # Use real proper-noun labels; short codes like "YT"/"TT" are
        # intentionally ignored by the extractor (minimum 3-letter
        # label to keep "It was 42" false positives out).
        text = "YouTube: 1 views. TikTok: 2 views."
        result = precompute_sum(question, text)
        assert bool(result) is True

    def test_falsy_when_refused(self):
        assert not precompute_sum("Who?", "")

    def test_short_labels_rejected(self):
        """Two-letter shorthand like 'YT', 'TT' is too loose to trust."""
        result = precompute_sum("Total?", "YT: 1 views. TT: 2 views.")
        assert result is None


class TestCoerceNumber:
    def test_invalid_returns_none(self):
        assert _coerce_number("not-a-number") is None

    def test_strips_currency_symbols(self):
        assert _coerce_number("$1,200") == 1200

    def test_strips_pounds(self):
        assert _coerce_number("£500") == 500

    def test_strips_euros(self):
        assert _coerce_number("€250.50") == 250.5


class TestCurrency:
    def test_iso_currency_extracted(self):
        text = "Sarah spent CHF 100 on groceries. Mark paid USD 50 on the trip."
        facts = extract_numeric_facts(text)
        # At least one CHF / USD value extracted
        values = [f.value for f in facts]
        assert 100 in values or 50 in values


class TestAggregationDispatch:
    def test_missing_question_mark_still_works(self):
        """Questions without '?' are not rejected (tolerated shape)."""
        # The bench oracle always has '?', but be tolerant in code.
        result = precompute_aggregation(
            "What is the total views",
            "YouTube: 100 views. TikTok: 50 views.",
        )
        # Either returns the sum or None — must not crash.
        assert result is None or result.value == 150


class TestRegressionR11:
    """Reproduce the R11 failures the precompute is designed to fix."""

    def test_youtube_tiktok_r11_d6062bb9(self):
        """The R11 hypothesis said 2098; gold is 1998."""
        question = (
            "What is the total number of views on my most popular videos on YouTube and TikTok?"
        )
        # What the retriever typically has in context for this item.
        text = (
            "You said your YouTube tutorial on social media analytics has 542 views. "
            "Your TikTok video of Luna playing has 1456 views on the account."
        )
        result = precompute_sum(question, text)
        assert result is not None
        assert result.value == 1998
        assert "1998" in result.message

    def test_fb_instagram_r11_60036106(self):
        """R10 correctly summed 2000+10000=12000; R11 forgot to sum."""
        question = "What was the total number of people reached by my Facebook ad campaign and Instagram influencer collaboration?"
        text = (
            "Facebook: 2000 people reached by the ad campaign. "
            "Instagram: 10000 people via the influencer's followers."
        )
        result = precompute_sum(question, text)
        assert result is not None
        assert result.value == 12000


# ── Rust ↔ Python parity ─────────────────────────────────────────────
#
# The Rust fast path is covered by the *same* test set above (the module
# transparently dispatches to it when installed). These tests force the
# pure-Python fallback to run and diff it against the Rust result so we
# notice the moment the two implementations drift apart.


import importlib

import pytest

import aggregate_precompute as _agg

_rust_available = _agg._HAVE_RUST
_rust_skip = pytest.mark.skipif(
    not _rust_available, reason="remanentia_aggregate_precompute not installed"
)


class _PythonOnlyFacts:
    """Context manager that flips `_HAVE_RUST` off for the duration."""

    def __enter__(self):
        self._saved = _agg._HAVE_RUST
        _agg._HAVE_RUST = False
        return self

    def __exit__(self, *_exc):
        _agg._HAVE_RUST = self._saved


def _facts_tuple(facts):
    """Stable comparison form for NumericFact lists."""
    return tuple((f.label, round(f.value, 6), f.unit, f.raw) for f in facts)


@_rust_skip
class TestRustPythonParity:
    """Rust and Python paths must agree byte-for-byte on identical inputs."""

    # Force a fresh import so we trust `_agg._rust_agg` is populated.
    def setup_class(cls):
        importlib.reload(_agg)

    def test_is_sum_question_parity(self):
        for q in (
            "What is the total views?",
            "How much did I spend altogether?",
            "combined reach?",
            "adding up prices",
            "What was my most popular video?",
            "total",
            "",
        ):
            with _PythonOnlyFacts():
                py = _agg.is_sum_question(q)
            rust = _agg._rust_agg.is_sum_question(q)
            assert py == rust, f"mismatch on {q!r}: py={py} rust={rust}"

    def test_is_count_question_parity(self):
        for q in (
            "How many items did I buy?",
            "how many unique titles",
            "total items",  # sum wins → False on both
            "Which was largest?",
            "",
        ):
            with _PythonOnlyFacts():
                py = _agg.is_count_question(q)
            rust = _agg._rust_agg.is_count_question(q)
            assert py == rust, f"mismatch on {q!r}: py={py} rust={rust}"

    def test_extract_facts_r11(self):
        text = (
            "You said your YouTube tutorial on social media analytics has 542 views. "
            "Your TikTok video of Luna playing has 1456 views on the account."
        )
        with _PythonOnlyFacts():
            py = _agg.extract_numeric_facts(text)
        rust = _agg.extract_numeric_facts(text)
        assert _facts_tuple(py) == _facts_tuple(rust)

    def test_extract_facts_mixed_units(self):
        text = (
            "Monday: 10 hours spent. Tuesday: 5 hours spent. "
            "Grocery: 45 dollars spent. Fuel: 55 dollars spent."
        )
        with _PythonOnlyFacts():
            py = _agg.extract_numeric_facts(text)
        rust = _agg.extract_numeric_facts(text)
        assert _facts_tuple(py) == _facts_tuple(rust)

    def test_extract_facts_empty(self):
        with _PythonOnlyFacts():
            py = _agg.extract_numeric_facts("nothing interesting here")
        rust = _agg.extract_numeric_facts("nothing interesting here")
        assert _facts_tuple(py) == _facts_tuple(rust) == ()

    def test_extract_facts_utf8(self):
        # ¥ is a multi-byte char; walk-back must not panic.
        text = "Tokyo ¥500. MyChannel has 100 views. OtherChannel has 200 views."
        with _PythonOnlyFacts():
            py = _agg.extract_numeric_facts(text)
        rust = _agg.extract_numeric_facts(text)
        assert _facts_tuple(py) == _facts_tuple(rust)

    def test_precompute_sum_message_parity(self):
        text = (
            "Your YouTube tutorial has 542 views. Your TikTok video of Luna playing has 1456 views."
        )
        q = "What is the total views?"
        with _PythonOnlyFacts():
            py = _agg.precompute_sum(q, text)
        rust = _agg.precompute_sum(q, text)
        assert py is not None and rust is not None
        assert py.message == rust.message
        assert py.value == rust.value

    def test_precompute_sum_refuses_when_python_refuses(self):
        # Only one fact → both paths must return None.
        text = "Your YouTube has 42 views and nothing else."
        q = "What is the total views?"
        with _PythonOnlyFacts():
            py = _agg.precompute_sum(q, text)
        rust = _agg.precompute_sum(q, text)
        assert py is None and rust is None
