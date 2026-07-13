# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for memory_guarded (Director-AI bridge)

"""Real Director-AI integration tests for the Guarded memory tier."""

from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError
from typing import Protocol, cast

import pytest

from memory_guarded import (
    DEFAULT_POLICY,
    GuardedPolicy,
    GuardedResult,
    facts_from_results,
    is_available,
    score_memory_answer,
)
from memory_index import SearchResult


_requires_director_ai = pytest.mark.skipif(
    importlib.util.find_spec("director_ai") is None,
    reason="director-ai guarded extra is not installed",
)


class _MutablePolicy(Protocol):
    approve_threshold: float


class TestAvailabilityAndPolicy:
    def test_reports_installed_real_dependency(self) -> None:
        assert is_available() is (importlib.util.find_spec("director_ai") is not None)

    def test_default_policy(self) -> None:
        assert GuardedPolicy(
            approve_threshold=0.3,
            block_below=0.15,
            use_nli=None,
            injection_detection=True,
        ) == DEFAULT_POLICY

    def test_policy_is_frozen(self) -> None:
        mutable_view = cast(_MutablePolicy, DEFAULT_POLICY)
        with pytest.raises(FrozenInstanceError):
            mutable_view.approve_threshold = 0.5


def _hit(name: str, *, snippet: str = "", answer: str = "") -> SearchResult:
    return SearchResult(
        name=name,
        source="trace",
        score=1.0,
        snippet=snippet,
        answer=answer,
    )


class TestFactsFromRealSearchResults:
    def test_prefers_answer_over_snippet(self) -> None:
        facts = facts_from_results([_hit("a", snippet="long snippet", answer="short answer")])
        assert facts == {"a": "short answer"}

    def test_falls_back_to_snippet(self) -> None:
        assert facts_from_results([_hit("a", snippet="snippet only")]) == {
            "a": "snippet only"
        }

    def test_deduplicates_first_named_result(self) -> None:
        facts = facts_from_results(
            [
                _hit("a", snippet="first"),
                _hit("a", snippet="second"),
                _hit("b", snippet="third"),
            ]
        )
        assert facts == {"a": "first", "b": "third"}

    def test_skips_empty_name_and_text(self) -> None:
        assert facts_from_results([_hit("", snippet="orphan"), _hit("a")]) == {}

    def test_truncates_evidence_to_400_characters(self) -> None:
        facts = facts_from_results([_hit("a", snippet="x" * 1000)])
        assert facts == {"a": "x" * 400}


@_requires_director_ai
class TestRealDirectorScoring:
    @staticmethod
    def _policy(
        *,
        block_below: float = 0.15,
        injection_detection: bool = False,
    ) -> GuardedPolicy:
        return GuardedPolicy(
            approve_threshold=0.3,
            block_below=block_below,
            use_nli=False,
            injection_detection=injection_detection,
        )

    def test_grounded_answer_is_approved(self) -> None:
        result = score_memory_answer(
            "Where does Alice live?",
            "Alice lives in Berlin.",
            {"alice": "Alice lives in Berlin."},
            policy=self._policy(),
        )

        assert result is not None
        assert result.approved is True
        assert result.blocked is False
        assert result.score > 0.3
        assert result.reason == ""

    def test_contradiction_is_disapproved_but_not_blocked(self) -> None:
        result = score_memory_answer(
            "Where does Alice live?",
            "Alice lives in Tokyo.",
            {"alice": "Alice lives in Berlin."},
            policy=self._policy(),
        )

        assert result is not None
        assert result.approved is False
        assert result.blocked is False
        assert 0.15 < result.score < 0.3
        assert "approve_threshold" in result.reason

    def test_stricter_policy_blocks_same_real_score(self) -> None:
        result = score_memory_answer(
            "Where does Alice live?",
            "Alice lives in Tokyo.",
            {"alice": "Alice lives in Berlin."},
            policy=self._policy(block_below=0.3),
        )

        assert result is not None
        assert result.approved is False
        assert result.blocked is True
        assert "block_below" in result.reason

    def test_real_keyword_evidence_is_normalised(self) -> None:
        result = score_memory_answer(
            "Where does Alice live?",
            "Alice lives in Berlin.",
            {"alice": "Alice lives in Berlin."},
            policy=self._policy(),
        )

        assert result is not None
        assert result.evidence == [
            {
                "text": "Alice lives in Berlin.",
                "distance": 0.0,
                "source": "keyword",
            }
        ]
        assert result.h_logical > 0.0
        assert result.h_factual >= 0.0

    def test_injection_policy_controls_real_result_shape(self) -> None:
        disabled = score_memory_answer(
            "Where does Alice live?",
            "Alice lives in Berlin.",
            {"alice": "Alice lives in Berlin."},
            policy=self._policy(injection_detection=False),
        )
        enabled = score_memory_answer(
            "Where does Alice live?",
            "Alice lives in Berlin.",
            {"alice": "Alice lives in Berlin."},
            policy=self._policy(injection_detection=True),
        )

        assert disabled is not None and disabled.injection_risk is None
        assert enabled is not None and enabled.injection_risk is not None


class TestGuardedResult:
    def test_to_dict_roundtrip(self) -> None:
        result = GuardedResult(
            score=0.5,
            approved=True,
            blocked=False,
            h_logical=0.8,
            h_factual=0.3,
            injection_risk=0.1,
            evidence=[{"text": "x", "distance": 0.0, "source": "keyword"}],
        )

        assert result.to_dict() == {
            "score": 0.5,
            "approved": True,
            "blocked": False,
            "h_logical": 0.8,
            "h_factual": 0.3,
            "injection_risk": 0.1,
            "evidence": [{"text": "x", "distance": 0.0, "source": "keyword"}],
            "reason": "",
        }
