# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for memory_guarded (Director-AI bridge)

"""Guarded-tier bridge behaviour.

The Director-AI dep is optional. Tests split into two groups:

1. Always-run: behaviour when Director-AI is NOT installed, policy
   shape, fact-extraction helper, dataclass hygiene.
2. Skipped-if-unavailable: real Director-AI scoring with mocked
   director_ai.score so the test stays CPU-cheap (no model load).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from memory_guarded import (
    DEFAULT_POLICY,
    GuardedPolicy,
    GuardedResult,
    facts_from_results,
    is_available,
    score_memory_answer,
)


# ── Optional dep probe ───────────────────────────────────────────────


class TestIsAvailable:
    def test_returns_bool(self):
        assert is_available() in (True, False)

    def test_no_director_ai_returns_false(self, monkeypatch):
        import sys

        real = sys.modules.get("director_ai")
        sys.modules["director_ai"] = None  # poisoned sentinel
        try:
            # Reimport the shim within the probe; the poisoned sentinel
            # makes `import director_ai` raise ImportError.
            assert is_available() is False
        finally:
            if real is not None:
                sys.modules["director_ai"] = real
            else:
                sys.modules.pop("director_ai", None)


# ── Policy ───────────────────────────────────────────────────────────


class TestPolicy:
    def test_default_thresholds(self):
        assert DEFAULT_POLICY.approve_threshold == 0.3
        assert DEFAULT_POLICY.block_below == 0.15
        assert DEFAULT_POLICY.use_nli is None
        assert DEFAULT_POLICY.injection_detection is True

    def test_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_POLICY.approve_threshold = 0.5  # type: ignore[misc]

    def test_custom_policy(self):
        p = GuardedPolicy(approve_threshold=0.6, block_below=0.3, use_nli=True)
        assert p.approve_threshold == 0.6
        assert p.block_below == 0.3
        assert p.use_nli is True


# ── facts_from_results ───────────────────────────────────────────────


@dataclass
class _FakeSearchHit:
    name: str
    snippet: str = ""
    answer: str = ""


class TestFactsFromResults:
    def test_prefers_answer_over_snippet(self):
        hits = [_FakeSearchHit(name="a", snippet="long snippet here", answer="short answer")]
        facts = facts_from_results(hits)
        assert facts == {"a": "short answer"}

    def test_falls_back_to_snippet(self):
        hits = [_FakeSearchHit(name="a", snippet="snippet only", answer="")]
        assert facts_from_results(hits) == {"a": "snippet only"}

    def test_dedupes_by_name(self):
        hits = [
            _FakeSearchHit(name="a", snippet="first"),
            _FakeSearchHit(name="a", snippet="second"),
            _FakeSearchHit(name="b", snippet="third"),
        ]
        facts = facts_from_results(hits)
        assert facts == {"a": "first", "b": "third"}

    def test_skips_empty_text(self):
        hits = [_FakeSearchHit(name="a", snippet="", answer="")]
        assert facts_from_results(hits) == {}

    def test_skips_missing_name(self):
        hits = [_FakeSearchHit(name="", snippet="orphan")]
        assert facts_from_results(hits) == {}

    def test_truncates_to_400_chars(self):
        hits = [_FakeSearchHit(name="a", snippet="x" * 1000)]
        facts = facts_from_results(hits)
        assert len(facts["a"]) == 400


# ── score_memory_answer ──────────────────────────────────────────────


def _director_ai_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("director_ai") is not None


_requires_director_ai = pytest.mark.skipif(
    not _director_ai_importable(),
    reason="director_ai not installed — Guarded tier is an optional dep",
)


class TestScoreMemoryAnswer:
    def test_returns_none_when_director_ai_unavailable(self):
        with patch("memory_guarded.is_available", return_value=False):
            result = score_memory_answer("q?", "a.", {"f": "evidence"})
        assert result is None

    @_requires_director_ai
    def test_approved_path(self):
        """Director-AI approves; policy block_below not triggered."""
        fake_score = _fake_coherence(score=0.6, approved=True)
        with (
            patch("memory_guarded.is_available", return_value=True),
            patch("director_ai.score", return_value=fake_score) as mock_score,
        ):
            r = score_memory_answer("q", "a", {"f": "ev"})
        assert mock_score.called
        assert r is not None
        assert r.approved is True
        assert r.blocked is False
        assert r.score == 0.6
        assert r.reason == ""

    @_requires_director_ai
    def test_blocked_path(self):
        """Director-AI disapproves AND score falls below block_below."""
        fake_score = _fake_coherence(score=0.05, approved=False)
        with (
            patch("memory_guarded.is_available", return_value=True),
            patch("director_ai.score", return_value=fake_score),
        ):
            r = score_memory_answer("q", "a", {"f": "ev"})
        assert r is not None
        assert r.blocked is True
        assert r.approved is False
        assert "block_below" in r.reason

    @_requires_director_ai
    def test_disapproved_but_not_blocked(self):
        """Score between block_below (0.15) and approve_threshold (0.3)."""
        fake_score = _fake_coherence(score=0.2, approved=False)
        with (
            patch("memory_guarded.is_available", return_value=True),
            patch("director_ai.score", return_value=fake_score),
        ):
            r = score_memory_answer("q", "a", {"f": "ev"})
        assert r is not None
        assert r.blocked is False
        assert r.approved is False
        assert "approve_threshold" in r.reason

    @_requires_director_ai
    def test_evidence_extracted(self):
        ev = _fake_evidence([("evidence text", 0.1, "keyword")])
        fake_score = _fake_coherence(score=0.7, approved=True, evidence=ev)
        with (
            patch("memory_guarded.is_available", return_value=True),
            patch("director_ai.score", return_value=fake_score),
        ):
            r = score_memory_answer("q", "a", {"f": "ev"})
        assert r is not None
        assert len(r.evidence) == 1
        assert r.evidence[0]["text"] == "evidence text"
        assert r.evidence[0]["source"] == "keyword"

    @_requires_director_ai
    def test_injection_detection_flag_flows_through(self):
        fake_score = _fake_coherence(score=0.8, approved=True, injection_risk=0.2)
        with (
            patch("memory_guarded.is_available", return_value=True),
            patch("director_ai.score", return_value=fake_score) as mock_score,
        ):
            score_memory_answer(
                "q",
                "a",
                {"f": "ev"},
                policy=GuardedPolicy(injection_detection=True),
            )
        kwargs = mock_score.call_args.kwargs
        assert kwargs["injection_detection"] is True

    @_requires_director_ai
    def test_use_nli_flag_flows_through(self):
        fake_score = _fake_coherence(score=0.6, approved=True)
        with (
            patch("memory_guarded.is_available", return_value=True),
            patch("director_ai.score", return_value=fake_score) as mock_score,
        ):
            score_memory_answer(
                "q",
                "a",
                {"f": "ev"},
                policy=GuardedPolicy(use_nli=True),
            )
        kwargs = mock_score.call_args.kwargs
        assert kwargs["use_nli"] is True


# ── GuardedResult.to_dict ────────────────────────────────────────────


class TestResultDict:
    def test_roundtrips_to_dict(self):
        r = GuardedResult(
            score=0.5,
            approved=True,
            blocked=False,
            h_logical=0.8,
            h_factual=0.3,
            injection_risk=0.1,
            evidence=[{"text": "x", "distance": 0.0, "source": "keyword"}],
            reason="",
        )
        d = r.to_dict()
        assert d["score"] == 0.5
        assert d["approved"] is True
        assert d["evidence"][0]["source"] == "keyword"


# ── Test helpers ─────────────────────────────────────────────────────


def _fake_coherence(*, score, approved, evidence=None, injection_risk=None):
    """Build a stand-in for director_ai.CoherenceScore."""

    class _FakeScore:
        def __init__(self):
            self.score = score
            self.approved = approved
            self.h_logical = 0.5
            self.h_factual = 0.5
            self.evidence = evidence if evidence is not None else _fake_evidence([])
            self.injection_risk = injection_risk

    return _FakeScore()


def _fake_evidence(chunks):
    """Build a stand-in for director_ai.ScoringEvidence."""

    class _FakeChunk:
        def __init__(self, text, distance, source):
            self.text = text
            self.distance = distance
            self.source = source

    class _FakeEvidence:
        def __init__(self):
            self.chunks = [_FakeChunk(*c) for c in chunks]

    return _FakeEvidence()
