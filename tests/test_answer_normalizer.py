# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
from __future__ import annotations

import pytest
from answer_normalizer import (
    answers_match,
    extract_answer_items,
    normalize_answer,
)


class TestNormalizeAnswer:
    def test_likely_yes(self):
        assert normalize_answer("Likely yes, because she enjoys reading") == "likely yes"

    def test_likely_no(self):
        assert normalize_answer("Likely no; though she likes reading, she wants to be a counselor") == "likely no"

    def test_yes_since(self):
        assert normalize_answer("Yes, since she collects classic children's books") == "yes"

    def test_no_because(self):
        assert normalize_answer("No, because she never mentioned it") == "no"

    def test_plain_yes(self):
        assert normalize_answer("Yes") == "yes"

    def test_plain_no(self):
        assert normalize_answer("No") == "no"

    def test_probably_yes(self):
        assert normalize_answer("Probably yes, given her interests") == "probably yes"

    def test_strip_whitespace(self):
        assert normalize_answer("  Running, pottery  ") == "running, pottery"

    def test_empty(self):
        assert normalize_answer("") == ""

    def test_hedging_prefix(self):
        assert normalize_answer("I think yes") == "yes"
        assert normalize_answer("I believe no, because reasons") == "no"

    def test_non_yesno_explanation_strip(self):
        assert normalize_answer("Swimming, because it's relaxing") == "swimming"

    def test_trailing_period(self):
        assert normalize_answer("Liberal.") == "liberal"

    def test_simple_answer(self):
        assert normalize_answer("Liberal") == "liberal"

    def test_multi_word(self):
        assert normalize_answer("a stained glass window") == "a stained glass window"

    def test_unlikely(self):
        assert normalize_answer("Unlikely, since she prefers art") == "unlikely"


class TestExtractAnswerItems:
    def test_comma_list(self):
        items = extract_answer_items("pottery, camping, painting, swimming")
        assert items == {"pottery", "camping", "painting", "swimming"}

    def test_and_separator(self):
        items = extract_answer_items("running and pottery")
        assert items == {"running", "pottery"}

    def test_comma_and_mixed(self):
        items = extract_answer_items("running, swimming, and painting")
        assert items == {"running", "swimming", "painting"}

    def test_single_item(self):
        items = extract_answer_items("pottery")
        assert items == {"pottery"}

    def test_empty(self):
        assert extract_answer_items("") == set()


class TestAnswersMatch:
    def test_exact(self):
        assert answers_match("Yes", "Yes")

    def test_yes_polarity(self):
        assert answers_match("Likely yes, because she loves books", "Yes")

    def test_no_polarity(self):
        assert answers_match("No", "Likely no")

    def test_polarity_mismatch(self):
        assert not answers_match("Yes", "No")
        assert not answers_match("Likely yes", "Likely no")

    def test_containment(self):
        assert answers_match("Sweden", "She moved from Sweden")

    def test_list_overlap(self):
        assert answers_match("pottery, running", "pottery, camping, painting, swimming")

    def test_list_no_overlap(self):
        assert not answers_match("coding, gaming", "pottery, camping, painting, swimming")

    def test_normalized_containment(self):
        assert answers_match("Liberal", "liberal")

    def test_completely_different(self):
        assert not answers_match("completely unrelated answer", "The actual answer")

    def test_empty_predicted(self):
        assert not answers_match("", "Yes")

    def test_empty_ground_truth(self):
        assert not answers_match("Yes", "")

    def test_yes_with_explanation_matches_yes(self):
        assert answers_match(
            "Yes, since she collects classic children's books",
            "Yes, since she collects classic children's books"
        )

    def test_likely_no_matches_likely_no(self):
        assert answers_match(
            "Likely no, because she never said so",
            "Likely no"
        )
