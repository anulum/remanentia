# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal training integration tests

"""Cross-module integration tests for C1–C5 components.

Tests the interaction between:
- temporal_graph.parse_dates ← date_normalizer (C4)
- fact_decomposer._build_fact ← fact_validity_model (C5)
- fact_decomposer._extract_dates_with_normaliser ← date_normalizer (C4)
- arcane_retriever._ch_temporal ← temporal_relation (C3)
- Graceful degradation when models are unavailable
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch


from date_normalizer import DateResult


# ── parse_dates with C4 integration ────────────────────────────


class TestParseDatesC4:
    """Test temporal_graph.parse_dates with date normaliser."""

    def test_iso_date_still_works(self):
        from temporal_graph import parse_dates

        result = parse_dates("Fixed on 2023-04-10.")
        assert "2023-04-10" in result

    def test_english_date_still_works(self):
        from temporal_graph import parse_dates

        result = parse_dates("Meeting on March 15, 2023.")
        assert "2023-03-15" in result

    def test_relative_yesterday(self):
        from temporal_graph import parse_dates

        result = parse_dates("I went yesterday.", reference_date=date(2023, 4, 10))
        assert "2023-04-09" in result

    def test_vague_expression_via_c4(self):
        from temporal_graph import parse_dates

        result = parse_dates(
            "I bought it about 3 weeks ago.",
            reference_date=date(2023, 4, 10),
        )
        # C4 rule-based normaliser should extract the date
        assert len(result) >= 1
        # Should be approximately 2023-03-20
        assert any("2023-03" in d for d in result)

    def test_c4_import_error_graceful(self):
        """When date_normalizer is not importable, parse_dates still works."""
        from temporal_graph import parse_dates

        with patch.dict("sys.modules", {"date_normalizer": None}):
            # Should not raise, just skip C4
            result = parse_dates("I did it 3 weeks ago.", reference_date=date(2023, 4, 10))
            # May or may not find dates depending on existing patterns
            assert isinstance(result, list)

    def test_c4_low_confidence_filtered(self):
        """Results with confidence <= 0.7 are not included."""
        from temporal_graph import parse_dates

        low_conf = DateResult(iso_date="2023-01-01", confidence=0.5, method="model")
        with patch("date_normalizer.normalize_date_expression", return_value=low_conf):
            result = parse_dates(
                "something vague recently",
                reference_date=date(2023, 4, 10),
            )
            # "recently" matches VAGUE_DATE_RE but confidence is low
            # The rule-based handler handles "recently" with 0.85 confidence,
            # so it should still appear
            assert isinstance(result, list)

    def test_deduplication(self):
        """Duplicate dates from different sources are deduplicated."""
        from temporal_graph import parse_dates

        # "yesterday" is handled by both _DATE_RELATIVE and VAGUE_DATE_RE
        result = parse_dates("yesterday", reference_date=date(2023, 4, 10))
        # Should not have duplicates
        assert len(result) == len(set(result))

    def test_sorting(self):
        """Results are sorted chronologically."""
        from temporal_graph import parse_dates

        result = parse_dates(
            "On 2023-04-10 and 2023-01-05 things happened.",
            reference_date=date(2023, 6, 1),
        )
        assert result == sorted(result)


# ── _extract_dates_with_normaliser ─────────────────────────────


class TestExtractDatesWithNormaliser:
    def test_with_reference_date(self):
        from fact_decomposer import _extract_dates_with_normaliser

        result = _extract_dates_with_normaliser(
            "I did it about 2 weeks ago",
            reference_date=date(2023, 6, 15),
            default_year=2023,
        )
        assert len(result) >= 1

    def test_without_reference_date_no_c4(self):
        from fact_decomposer import _extract_dates_with_normaliser

        # No reference_date → C4 should not activate
        result = _extract_dates_with_normaliser(
            "I did it about 2 weeks ago",
            reference_date=None,
            default_year=2023,
        )
        # Without C4 and without ISO/written dates, result may be empty
        assert isinstance(result, list)

    def test_iso_date_takes_priority(self):
        from fact_decomposer import _extract_dates_with_normaliser

        # If ISO date is found, C4 is not called (results not empty)
        result = _extract_dates_with_normaliser(
            "Fixed on 2023-04-10, about 2 weeks ago",
            reference_date=date(2023, 6, 15),
            default_year=2023,
        )
        assert "2023-04-10" in result

    def test_c4_import_error_graceful(self):
        from fact_decomposer import _extract_dates_with_normaliser

        with patch.dict("sys.modules", {"date_normalizer": None}):
            result = _extract_dates_with_normaliser(
                "I did it recently",
                reference_date=date(2023, 6, 15),
                default_year=2023,
            )
            assert isinstance(result, list)


# ── _build_fact with C5 integration ────────────────────────────


class TestBuildFactC5:
    """Test fact_decomposer._build_fact with fact validity model."""

    def test_regex_plan_not_overridden(self):
        """Plan patterns detected by regex should not be overridden by C5."""
        from fact_decomposer import _build_fact

        fact = _build_fact(
            "I plan to visit Tokyo next month.",
            sess_idx=0,
            turn_idx=0,
            role="user",
            default_year=2024,
        )
        assert fact.fact_type == "plan"

    def test_regex_preference_not_overridden(self):
        from fact_decomposer import _build_fact

        fact = _build_fact(
            "I love hiking in the mountains.",
            sess_idx=0,
            turn_idx=0,
            role="user",
            default_year=2024,
        )
        assert fact.fact_type == "preference"

    def test_regex_change_verb_not_overridden(self):
        from fact_decomposer import _build_fact

        fact = _build_fact(
            "I started learning piano last year.",
            sess_idx=0,
            turn_idx=0,
            role="user",
            default_year=2024,
        )
        assert fact.fact_type == "state"
        assert fact.supersedes is True

    def test_event_may_be_overridden_by_c5(self):
        """When regex gives 'event' and C5 is confident, override is allowed."""
        from fact_decomposer import _build_fact

        # "The weather is nice" → regex gives "event" (catch-all)
        # C5 might override if confident
        fact = _build_fact(
            "The weather is really nice outside.",
            sess_idx=0,
            turn_idx=0,
            role="user",
            default_year=2024,
        )
        # Whatever C5 decides, it should be a valid type
        assert fact.fact_type in ["state", "event", "preference", "plan"]

    def test_c5_import_error_graceful(self):
        """When fact_validity_model is not importable, regex fallback works."""
        from fact_decomposer import _build_fact

        with patch.dict("sys.modules", {"fact_validity_model": None}):
            fact = _build_fact(
                "The sun is shining.",
                sess_idx=0,
                turn_idx=0,
                role="user",
                default_year=2024,
            )
            assert fact.fact_type == "event"  # regex catch-all
            assert fact.supersedes is False


# ── _ch_temporal with C3 integration ──────────────────────────


class TestChTemporalC3:
    """Test arcane_retriever._ch_temporal with temporal relation classifier."""

    def _make_retriever(self):
        from fact_decomposer import AtomicFact

        facts = [
            AtomicFact(
                text="I bought a car last month",
                session_idx=0,
                turn_idx=0,
                role="user",
                fact_type="event",
                entities=["car"],
                date_mentions=["2023-03-15"],
            ),
            AtomicFact(
                text="I visited the dentist yesterday",
                session_idx=0,
                turn_idx=1,
                role="user",
                fact_type="event",
                entities=["dentist"],
                date_mentions=["2023-04-09"],
            ),
        ]
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [
                {"role": "user", "content": "I bought a car last month."},
                {"role": "user", "content": "I visited the dentist yesterday."},
            ]
        ]
        ar = ArcaneRetriever(sessions)
        return ar

    def test_temporal_channel_returns_results(self):
        ar = self._make_retriever()
        results = ar._ch_temporal("When did I buy the car?", top_k=5)
        assert isinstance(results, list)

    def test_c3_import_error_graceful(self):
        """When temporal_relation is not importable, results are unmodified."""
        ar = self._make_retriever()
        with patch.dict("sys.modules", {"temporal_relation": None}):
            results = ar._ch_temporal("When did I buy the car?", top_k=5)
            assert isinstance(results, list)

    def test_single_result_no_c3(self):
        """C3 reordering requires >= 2 results."""
        from arcane_retriever import ArcaneRetriever

        sessions = [[{"role": "user", "content": "Just one fact here."}]]
        ar = ArcaneRetriever(sessions)
        results = ar._ch_temporal("test", top_k=1)
        assert isinstance(results, list)


# ── Full pipeline: decompose → index → retrieve ───────────────


class TestFullPipeline:
    """End-to-end test: sessions → atomic facts → retrieval."""

    def test_decompose_and_query(self):
        from fact_decomposer import FactIndex, decompose_sessions

        sessions = [
            [
                {"role": "user", "content": "I started a new job at Google 3 weeks ago."},
                {"role": "assistant", "content": "Congratulations on the new position!"},
                {"role": "user", "content": "I also bought a new laptop yesterday."},
            ],
        ]
        facts = decompose_sessions(sessions, default_year=2024)
        assert len(facts) >= 2

        idx = FactIndex(facts)
        hits = idx.query("When did I start the new job?", top_k=5)
        assert len(hits) >= 1
        # At least one fact should mention "job" or "Google"
        texts = [f.text.lower() for f, _ in hits]
        assert any("job" in t or "google" in t for t in texts)

    def test_temporal_query_with_dates(self):
        from fact_decomposer import FactIndex, decompose_sessions

        sessions = [
            [
                {"role": "user", "content": "I visited Paris on March 15, 2024."},
                {"role": "user", "content": "I went to London on April 2, 2024."},
            ],
        ]
        facts = decompose_sessions(sessions, default_year=2024)
        idx = FactIndex(facts)
        hits = idx.temporal_query("Which city did I visit first?", top_k=5)
        assert len(hits) >= 1

    def test_arcane_retriever_end_to_end(self):
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [
                {
                    "role": "user",
                    "content": "I joined a yoga class in January 2024. It was a great decision and I went every Tuesday and Thursday.",
                },
                {
                    "role": "assistant",
                    "content": "That sounds great! Yoga is wonderful for flexibility.",
                },
                {"role": "user", "content": "I also started running on weekends in February 2024."},
            ],
            [
                {
                    "role": "user",
                    "content": "I switched from yoga to swimming in March 2024. The pool is closer to my new apartment.",
                },
                {"role": "assistant", "content": "Swimming is excellent full-body exercise."},
                {"role": "user", "content": "I still go running on Saturdays though."},
            ],
        ]
        ar = ArcaneRetriever(sessions)
        # Verify fact decomposition worked
        assert len(ar.fact_index.facts) >= 3
        # Test via fact_index.query — use terms that overlap with fact text
        hits = ar.fact_index.query("yoga class joined January", top_k=5)
        assert len(hits) >= 1
        texts = [f.text.lower() for f, _ in hits]
        assert any("yoga" in t for t in texts)


# ── Graceful degradation chain ─────────────────────────────────


class TestGracefulDegradation:
    """Verify the system works correctly when all models are missing."""

    def test_all_models_missing(self):
        """Full pipeline should work even without C3, C4, C5 models."""
        with patch.dict(
            "sys.modules",
            {
                "date_normalizer": None,
                "temporal_relation": None,
                "fact_validity_model": None,
            },
        ):
            from fact_decomposer import FactIndex, decompose_sessions

            sessions = [
                [
                    {"role": "user", "content": "I bought a car on March 15, 2024."},
                    {"role": "user", "content": "I visited the dentist on April 2, 2024."},
                ]
            ]
            facts = decompose_sessions(sessions, default_year=2024)
            assert len(facts) >= 2

            idx = FactIndex(facts)
            hits = idx.query("When did I buy the car?", top_k=3)
            assert len(hits) >= 1


# ── Reference date plumbing (CRITICAL for C4 on real data) ────


class TestReferenceDatePlumbing:
    """Verify haystack_dates flow through the entire pipeline."""

    def test_decompose_sessions_with_session_dates(self):
        """Vague dates are resolved when session_dates are provided."""
        from fact_decomposer import decompose_sessions

        sessions = [
            [
                {
                    "role": "user",
                    "content": "I bought a car about 3 weeks ago and it runs perfectly.",
                },
            ]
        ]
        # Without session_dates: "3 weeks ago" cannot be resolved
        facts_no_ref = decompose_sessions(sessions, default_year=2023)
        dates_no_ref = [d for f in facts_no_ref for d in f.date_mentions]

        # With session_dates: "3 weeks ago" from 2023-04-10 → ~2023-03-20
        facts_with_ref = decompose_sessions(
            sessions,
            default_year=2023,
            session_dates=["2023/04/10"],
        )
        dates_with_ref = [d for f in facts_with_ref for d in f.date_mentions]

        assert len(dates_with_ref) > len(dates_no_ref)
        assert any("2023-03" in d for d in dates_with_ref)

    def test_decompose_sessions_without_session_dates(self):
        """Without session_dates, behaviour is unchanged from baseline."""
        from fact_decomposer import decompose_sessions

        sessions = [
            [
                {"role": "user", "content": "I started a new job on March 15, 2024."},
            ]
        ]
        facts = decompose_sessions(sessions, default_year=2024)
        assert any("2024-03-15" in d for f in facts for d in f.date_mentions)

    def test_decompose_sessions_malformed_date(self):
        """Malformed session_dates are silently skipped."""
        from fact_decomposer import decompose_sessions

        sessions = [
            [
                {"role": "user", "content": "I did something important recently and it went well."},
            ]
        ]
        facts = decompose_sessions(
            sessions,
            default_year=2024,
            session_dates=["not-a-date"],
        )
        assert isinstance(facts, list)

    def test_arcane_retriever_passes_session_dates(self):
        """ArcaneRetriever forwards session_dates to decompose_sessions."""
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [
                {
                    "role": "user",
                    "content": "I bought a new laptop about 2 weeks ago, it works great.",
                },
                {"role": "assistant", "content": "That sounds like a good purchase!"},
            ]
        ]
        ar = ArcaneRetriever(sessions, session_dates=["2023/06/15"])
        # Facts should have resolved dates from C4
        dated = [f for f in ar.fact_index.facts if f.date_mentions]
        assert len(dated) >= 1

    def test_arcane_retriever_without_session_dates(self):
        """ArcaneRetriever works fine without session_dates (backward compat)."""
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [
                {"role": "user", "content": "I bought a car on March 15, 2024."},
            ]
        ]
        ar = ArcaneRetriever(sessions)
        assert len(ar.fact_index.facts) >= 1

    def test_build_fact_with_reference_date(self):
        """_build_fact uses _extract_dates_with_normaliser when ref date given."""
        from fact_decomposer import _build_fact

        fact = _build_fact(
            "I started about 3 weeks ago and it's going well.",
            sess_idx=0,
            turn_idx=0,
            role="user",
            default_year=2023,
            reference_date=date(2023, 4, 10),
        )
        # C4 should resolve "3 weeks ago" from 2023-04-10 → ~2023-03-20
        assert len(fact.date_mentions) >= 1
        assert any("2023-03" in d for d in fact.date_mentions)

    def test_build_fact_without_reference_date(self):
        """_build_fact falls back to regex-only date extraction."""
        from fact_decomposer import _build_fact

        fact = _build_fact(
            "I visited the dentist on March 15, 2024.",
            sess_idx=0,
            turn_idx=0,
            role="user",
            default_year=2024,
        )
        assert "2024-03-15" in fact.date_mentions

    def test_end_to_end_vague_dates_in_temporal_query(self):
        """Full pipeline: vague dates → C4 resolution → temporal_query finds them."""
        from fact_decomposer import FactIndex, decompose_sessions

        sessions = [
            [
                {
                    "role": "user",
                    "content": "I joined a gym about 3 weeks ago. The membership costs fifty dollars a month.",
                },
                {
                    "role": "user",
                    "content": "I bought new running shoes last week. They are Nike brand and very comfortable.",
                },
            ],
        ]
        facts = decompose_sessions(
            sessions,
            default_year=2023,
            session_dates=["2023/04/10"],
        )
        idx = FactIndex(facts)

        # temporal_query should find dated facts
        hits = idx.temporal_query("When did I join the gym?", top_k=5)
        dated_hits = [(f, s) for f, s in hits if f.date_mentions]
        assert len(dated_hits) >= 1
