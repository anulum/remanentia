# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for answer_confidence

from __future__ import annotations

from answer_confidence import confidence_suffix, normalise_score, parse_confidence


class TestConfidenceSuffix:
    def test_mentions_confidence_and_range(self) -> None:
        s = confidence_suffix()
        assert "CONFIDENCE" in s
        assert "0 and 1" in s


class TestParseConfidence:
    def test_trailing_rating_on_new_line(self) -> None:
        answer, conf = parse_confidence("The budget is $25k.\nCONFIDENCE: 0.9")
        assert answer == "The budget is $25k."
        assert conf == 0.9

    def test_equals_form_and_integer_one(self) -> None:
        answer, conf = parse_confidence("Yes, she did.\nconfidence = 1")
        assert answer == "Yes, she did."
        assert conf == 1.0

    def test_inline_trailing_without_newline(self) -> None:
        answer, conf = parse_confidence("Paris CONFIDENCE:0.5")
        assert answer == "Paris"
        assert conf == 0.5

    def test_absent_returns_none_and_stripped(self) -> None:
        answer, conf = parse_confidence("  just an answer, no rating  ")
        assert answer == "just an answer, no rating"
        assert conf is None

    def test_out_of_range_clamped(self) -> None:
        _, conf = parse_confidence("x\nCONFIDENCE: 7")
        assert conf == 1.0


class TestNormaliseScore:
    def test_zero_is_half(self) -> None:
        assert normalise_score(0.0) == 0.5

    def test_large_positive_near_one(self) -> None:
        v = normalise_score(100.0)
        assert 0.99 < v <= 1.0

    def test_large_negative_near_zero(self) -> None:
        v = normalise_score(-100.0)
        assert 0.0 <= v < 0.01

    def test_monotone_in_unit_interval(self) -> None:
        assert normalise_score(-2.0) < normalise_score(0.0) < normalise_score(2.0)
        assert all(0.0 <= normalise_score(s) <= 1.0 for s in (-50.0, -1.0, 1.0, 50.0))
