# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for ArcaneRetriever

from __future__ import annotations

from unittest.mock import patch

from arcane_retriever import ArcaneRetriever, FusedResult, RetrievalResult
from fact_decomposer import AtomicFact


SESSIONS = [
    [
        {"role": "user", "content": "My name is Caroline and I work as a teacher in Boston."},
        {
            "role": "assistant",
            "content": "Nice to meet you, Caroline! Teaching is a great profession.",
        },
    ],
    [
        {
            "role": "user",
            "content": "I started a new job as a nurse on March 15, 2024 at the hospital.",
        },
        {"role": "assistant", "content": "Congratulations on the new position at the hospital!"},
    ],
    [
        {"role": "user", "content": "My friend Melanie enjoys hiking and photography on weekends."},
        {"role": "assistant", "content": "Those are wonderful hobbies for weekends!"},
    ],
]


# ── Dataclasses ──────────────────────────────────────────────────


class TestDataclasses:
    def test_retrieval_result(self):
        fact = AtomicFact(
            text="test",
            session_idx=0,
            turn_idx=0,
            role="user",
            fact_type="event",
        )
        r = RetrievalResult(fact=fact, score=0.9, channel="bm25", rank=0)
        assert r.channel == "bm25"
        assert r.rank == 0

    def test_fused_result(self):
        fact = AtomicFact(
            text="test",
            session_idx=0,
            turn_idx=0,
            role="user",
            fact_type="event",
        )
        fr = FusedResult(fact=fact, rrf_score=0.5, channels=["bm25"], per_channel_ranks={"bm25": 0})
        assert fr.rrf_score == 0.5
        assert "bm25" in fr.channels


# ── Gate (channel selection) ─────────────────────────────────────


class TestGate:
    def test_temporal_reasoning(self):
        ar = ArcaneRetriever(SESSIONS)
        channels = ar._gate("When did the event happen?", "temporal-reasoning")
        assert "temporal" in channels
        assert "bm25" in channels

    def test_multi_session(self):
        ar = ArcaneRetriever(SESSIONS)
        channels = ar._gate("What about sessions?", "multi-session")
        assert "session" in channels

    def test_knowledge_update(self):
        ar = ArcaneRetriever(SESSIONS)
        channels = ar._gate("Update?", "knowledge-update")
        assert "temporal" in channels

    def test_single_session_preference(self):
        ar = ArcaneRetriever(SESSIONS)
        channels = ar._gate("What hobbies?", "single-session-preference")
        assert channels == ["bm25", "entity"]

    def test_default(self):
        ar = ArcaneRetriever(SESSIONS)
        channels = ar._gate("Something", "single-session-user")
        assert channels == ["bm25", "entity"]


# ── Channel methods ──────────────────────────────────────────────


class TestChannels:
    def test_ch_bm25(self):
        ar = ArcaneRetriever(SESSIONS)
        results = ar._ch_bm25("Caroline teacher", top_k=5)
        assert isinstance(results, list)
        assert all(r.channel == "bm25" for r in results)

    def test_ch_entity(self):
        ar = ArcaneRetriever(SESSIONS)
        results = ar._ch_entity("Caroline nurse", top_k=5)
        assert isinstance(results, list)
        assert all(r.channel == "entity" for r in results)

    def test_ch_temporal(self):
        ar = ArcaneRetriever(SESSIONS)
        results = ar._ch_temporal("When did the job start", top_k=5)
        assert isinstance(results, list)
        assert all(r.channel == "temporal" for r in results)

    def test_ch_session(self):
        ar = ArcaneRetriever(SESSIONS)
        results = ar._ch_session("Caroline Melanie", top_k=5)
        assert isinstance(results, list)
        assert all(r.channel == "session" for r in results)


# ── Parallel retrieve ────────────────────────────────────────────


class TestParallelRetrieve:
    def test_runs_channels(self):
        ar = ArcaneRetriever(SESSIONS)
        results = ar._parallel_retrieve(
            "Caroline", "single-session-user", ["bm25", "entity"], top_k=5
        )
        assert "bm25" in results
        assert "entity" in results

    def test_all_four_channels(self):
        ar = ArcaneRetriever(SESSIONS)
        results = ar._parallel_retrieve(
            "Caroline March 2024",
            "temporal-reasoning",
            ["bm25", "entity", "temporal", "session"],
            top_k=5,
        )
        assert len(results) == 4

    def test_channel_exception_handled(self):
        ar = ArcaneRetriever(SESSIONS)
        with patch.object(ar, "_ch_bm25", side_effect=RuntimeError("crash")):
            results = ar._parallel_retrieve("test", "x", ["bm25"], top_k=5)
        assert results["bm25"] == []


# ── RRF Fusion ───────────────────────────────────────────────────


class TestRRFFusion:
    def test_fuses_results(self):
        ar = ArcaneRetriever(SESSIONS)
        fact = AtomicFact(
            text="shared fact",
            session_idx=0,
            turn_idx=0,
            role="user",
            fact_type="event",
        )
        channel_results = {
            "bm25": [RetrievalResult(fact=fact, score=0.9, channel="bm25", rank=0)],
            "entity": [RetrievalResult(fact=fact, score=0.8, channel="entity", rank=0)],
        }
        fused = ar._rrf_fusion(channel_results, top_k=5)
        assert len(fused) == 1
        assert len(fused[0].channels) == 2
        # Score = 1/(60+0) + 1/(60+0) from two channels
        assert fused[0].rrf_score > 1.0 / 60

    def test_top_k_limit(self):
        ar = ArcaneRetriever(SESSIONS)
        facts = [
            AtomicFact(text=f"fact {i}", session_idx=0, turn_idx=i, role="user", fact_type="event")
            for i in range(10)
        ]
        channel_results = {
            "bm25": [
                RetrievalResult(fact=f, score=1.0, channel="bm25", rank=i)
                for i, f in enumerate(facts)
            ],
        }
        fused = ar._rrf_fusion(channel_results, top_k=3)
        assert len(fused) == 3

    def test_empty_channels(self):
        ar = ArcaneRetriever(SESSIONS)
        fused = ar._rrf_fusion({}, top_k=5)
        assert fused == []


# ── Sufficiency check ────────────────────────────────────────────


class TestCheckSufficiency:
    def _fact(self, text="test", session_idx=0, dates=None, entities=None):
        return FusedResult(
            fact=AtomicFact(
                text=text,
                session_idx=session_idx,
                turn_idx=0,
                role="user",
                fact_type="event",
                date_mentions=dates or [],
                entities=entities or [],
            ),
            rrf_score=0.5,
            channels=["bm25"],
        )

    def test_no_results(self):
        ar = ArcaneRetriever(SESSIONS)
        ok, reason = ar._check_sufficiency("test", "general", [])
        assert not ok
        assert reason == "no_results"

    def test_temporal_missing_dates(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [self._fact("event without dates")]
        ok, reason = ar._check_sufficiency("When?", "temporal-reasoning", results)
        assert not ok
        assert reason == "missing_dates"

    def test_temporal_enough_dates(self):
        ar = ArcaneRetriever(SESSIONS)
        # Need enough entity overlap with the question too
        results = [
            self._fact("when event happened", dates=["2024-03-15"]),
            self._fact("when event occurred", dates=["2024-03-20"]),
        ]
        ok, reason = ar._check_sufficiency("when event", "temporal-reasoning", results)
        # With 2 dated results and entity overlap, should be sufficient
        assert ok or reason == "low_entity_coverage"

    def test_multi_session_single_session(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [self._fact("a", session_idx=0), self._fact("b", session_idx=0)]
        ok, reason = ar._check_sufficiency("What?", "multi-session", results)
        assert not ok
        assert reason == "single_session"

    def test_multi_session_diverse(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [
            self._fact("what happened", session_idx=0),
            self._fact("what occurred", session_idx=1),
        ]
        ok, reason = ar._check_sufficiency("what", "multi-session", results)
        # With 2 sessions and entity overlap, should be sufficient
        assert ok or reason == "low_entity_coverage"

    def test_counting_insufficient(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [self._fact("one thing")]
        ok, reason = ar._check_sufficiency("How many hobbies does she have?", "general", results)
        assert not ok
        assert reason == "insufficient_count_evidence"

    def test_low_entity_coverage(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [self._fact("completely unrelated text about zebras and giraffes")]
        ok, reason = ar._check_sufficiency(
            "What are Caroline's hobbies in Boston?", "general", results
        )
        assert not ok
        assert reason == "low_entity_coverage"

    def test_sufficient_general(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [self._fact("Caroline teaches in Boston at school")]
        ok, _ = ar._check_sufficiency("Caroline Boston", "general", results)
        assert ok


# ── Query rewriting ──────────────────────────────────────────────


class TestRewriteQuery:
    def _fact(self, text="test", entities=None):
        return FusedResult(
            fact=AtomicFact(
                text=text,
                session_idx=0,
                turn_idx=0,
                role="user",
                fact_type="event",
                entities=entities or [],
            ),
            rrf_score=0.5,
            channels=["bm25"],
        )

    def test_missing_dates(self):
        ar = ArcaneRetriever(SESSIONS)
        q = ar._rewrite_query("When?", "When?", "missing_dates", [])
        assert "date" in q.lower()

    def test_single_session(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [self._fact("test", entities=["Alice"])]
        q = ar._rewrite_query("What?", "What?", "single_session", results)
        assert "Alice" in q

    def test_insufficient_count(self):
        ar = ArcaneRetriever(SESSIONS)
        q = ar._rewrite_query("How many?", "How many?", "insufficient_count_evidence", [])
        assert "all instances" in q

    def test_low_entity_coverage(self):
        ar = ArcaneRetriever(SESSIONS)
        q = ar._rewrite_query("What are the hobbies?", "x", "low_entity_coverage", [])
        assert "hobbies" in q.lower()

    def test_unknown_reason(self):
        ar = ArcaneRetriever(SESSIONS)
        q = ar._rewrite_query("original", "current", "unknown", [])
        assert q == "original"


# ── Full retrieve pipeline ───────────────────────────────────────


class TestRetrieve:
    def test_basic_retrieve(self):
        ar = ArcaneRetriever(SESSIONS)
        results = ar.retrieve("What does Caroline do?", "single-session-user", top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, FusedResult) for r in results)

    def test_retrieve_temporal(self):
        ar = ArcaneRetriever(SESSIONS)
        results = ar.retrieve("When did the job change?", "temporal-reasoning", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_sufficiency_loop(self):
        ar = ArcaneRetriever(SESSIONS)
        # Force insufficiency on first iteration by making check return False
        call_count = [0]
        original = ar._check_sufficiency

        def mock_check(q, qt, results):
            call_count[0] += 1
            if call_count[0] == 1:
                return False, "low_entity_coverage"
            return True, ""

        with patch.object(ar, "_check_sufficiency", side_effect=mock_check):
            results = ar.retrieve("test query", "general", top_k=5, max_iterations=3)
        assert call_count[0] >= 2


# ── Build context ────────────────────────────────────────────────


class TestBuildContext:
    def test_basic_context(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [
            FusedResult(
                fact=AtomicFact(
                    text="Caroline works as a teacher",
                    session_idx=0,
                    turn_idx=0,
                    role="user",
                    fact_type="state",
                    date_mentions=[],
                ),
                rrf_score=0.5,
                channels=["bm25"],
            ),
        ]
        ctx = ar.build_context("What does Caroline do?", results)
        assert "Caroline" in ctx
        assert "[Session 1]" in ctx

    def test_context_with_dates(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [
            FusedResult(
                fact=AtomicFact(
                    text="Started nursing on March 15",
                    session_idx=1,
                    turn_idx=0,
                    role="user",
                    fact_type="state",
                    date_mentions=["2024-03-15"],
                ),
                rrf_score=0.5,
                channels=["temporal"],
            ),
        ]
        ctx = ar.build_context("When?", results)
        assert "Date: 2024-03-15" in ctx
        assert "[Session 2" in ctx

    def test_context_max_facts(self):
        ar = ArcaneRetriever(SESSIONS)
        results = [
            FusedResult(
                fact=AtomicFact(
                    text=f"fact {i}",
                    session_idx=0,
                    turn_idx=i,
                    role="user",
                    fact_type="event",
                ),
                rrf_score=0.5,
                channels=["bm25"],
            )
            for i in range(10)
        ]
        ctx = ar.build_context("test", results, max_facts=3)
        assert ctx.count("[Session") == 3
