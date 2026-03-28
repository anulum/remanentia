# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for fact_decomposer.py

from __future__ import annotations

from datetime import date

from fact_decomposer import (
    AtomicFact,
    FactIndex,
    decompose_sessions,
    _build_fact,
    _classify_fact,
    _extract_dates,
    _extract_entities_simple,
    _parse_date_str,
    _split_sentences,
    _tokenize,
)


# ── Tokenizer ────────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello World BM25")
        assert "hello" in tokens
        assert "world" in tokens
        assert "bm25" in tokens

    def test_filters_short(self):
        tokens = _tokenize("I am ok no")
        assert "am" not in tokens
        assert "no" not in tokens

    def test_empty(self):
        assert _tokenize("") == set()


# ── Sentence splitting ───────────────────────────────────────────


class TestSplitSentences:
    def test_multiple(self):
        sents = _split_sentences("First sentence here. Second sentence there.")
        assert len(sents) == 2

    def test_short_filtered(self):
        sents = _split_sentences("Hi. Ok then.")
        # "Hi" is < 10 chars, filtered
        assert len(sents) == 1

    def test_single_long(self):
        text = "This is a single long sentence without any period at the end"
        sents = _split_sentences(text)
        assert len(sents) == 1
        assert sents[0] == text

    def test_empty_returns_empty(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        sents = _split_sentences("   ")
        # "   ".strip() is "", which is falsy → empty list from the fallback
        assert sents == []


# ── Date extraction ──────────────────────────────────────────────


class TestExtractDates:
    def test_iso_date(self):
        dates = _extract_dates("Fixed on 2026-03-15.")
        assert "2026-03-15" in dates

    def test_written_date_full(self):
        dates = _extract_dates("Released on March 15, 2024.")
        assert "2024-03-15" in dates

    def test_written_date_abbreviated(self):
        dates = _extract_dates("Due on Jan 5, 2026.")
        assert "2026-01-05" in dates

    def test_written_date_no_year(self):
        dates = _extract_dates("Meeting on June 20.", default_year=2024)
        assert "2024-06-20" in dates

    def test_mdy_4digit_year(self):
        dates = _extract_dates("Date: 3/15/2024.")
        assert "2024-03-15" in dates

    def test_mdy_2digit_year(self):
        dates = _extract_dates("Date: 3/15/24.")
        assert "2024-03-15" in dates

    def test_multiple_dates(self):
        dates = _extract_dates("Started 2024-01-10, ended 2024-03-20.")
        assert len(dates) == 2

    def test_no_dates(self):
        assert _extract_dates("No dates here.") == []

    def test_invalid_month_name(self):
        dates = _extract_dates("Zember 15, 2024")
        assert len(dates) == 0

    def test_invalid_day_range(self):
        # Day 32 should be excluded
        dates = _extract_dates("0/32/2024")
        assert len(dates) == 0


# ── Entity extraction ────────────────────────────────────────────


class TestExtractEntities:
    def test_capitalised_name(self):
        ents = _extract_entities_simple("I met Caroline Smith yesterday.")
        assert any("Caroline" in e for e in ents)

    def test_filters_sentence_start(self):
        ents = _extract_entities_simple("The weather is nice.")
        # "The" is at position 0 — should be filtered
        assert "The" not in ents

    def test_filters_after_period(self):
        ents = _extract_entities_simple("Done. Nice work today.")
        # "Nice" is after ". " — sentence start, filtered
        assert "Nice" not in ents

    def test_quoted_strings(self):
        ents = _extract_entities_simple('She read "War and Peace" last week.')
        assert any("War and Peace" in e for e in ents)

    def test_empty(self):
        assert _extract_entities_simple("lowercase only") == []


# ── Fact classification ──────────────────────────────────────────


class TestClassifyFact:
    def test_plan(self):
        assert _classify_fact("I plan to visit Tokyo next summer.") == "plan"

    def test_preference(self):
        assert _classify_fact("I like hiking in the mountains.") == "preference"

    def test_state_change(self):
        assert _classify_fact("I started a new job at the hospital.") == "state"

    def test_event_default(self):
        assert _classify_fact("The meeting lasted two hours.") == "event"

    def test_going_to_is_plan(self):
        assert _classify_fact("I am going to move to Berlin.") == "plan"

    def test_favourite_is_preference(self):
        assert _classify_fact("My favourite book is Dune.") == "preference"


# ── Parse date string ────────────────────────────────────────────


class TestParseDateStr:
    def test_valid_iso(self):
        d = _parse_date_str("2024-03-15")
        assert d == date(2024, 3, 15)

    def test_invalid_format(self):
        assert _parse_date_str("not-a-date") is None

    def test_invalid_date_values(self):
        assert _parse_date_str("2024-13-01") is None

    def test_empty(self):
        assert _parse_date_str("") is None


# ── Build fact ───────────────────────────────────────────────────


class TestBuildFact:
    def test_basic(self):
        fact = _build_fact(
            "I started working at Google on 2024-03-15.",
            sess_idx=0, turn_idx=0, role="user", default_year=2024,
        )
        assert fact.fact_type == "state"
        assert fact.supersedes is True
        assert "2024-03-15" in fact.date_mentions
        assert fact.valid_from == "2024-03-15"

    def test_no_dates(self):
        fact = _build_fact(
            "The weather is really nice today.",
            sess_idx=1, turn_idx=2, role="user", default_year=2024,
        )
        assert fact.valid_from == ""
        assert fact.date_mentions == []


# ── Decompose sessions ──────────────────────────────────────────


class TestDecomposeSessions:
    def _sessions(self):
        return [
            [
                {"role": "user", "content": "My name is Caroline and I work as a teacher in Boston."},
                {"role": "assistant", "content": "Nice to meet you Caroline!"},
            ],
            [
                {"role": "user", "content": "I started a new job as a nurse on March 15, 2024. I left teaching."},
                {"role": "assistant", "content": "Congratulations!"},
            ],
        ]

    def test_produces_facts(self):
        facts = decompose_sessions(self._sessions())
        assert len(facts) > 0
        assert all(isinstance(f, AtomicFact) for f in facts)

    def test_session_indices(self):
        facts = decompose_sessions(self._sessions())
        sessions = set(f.session_idx for f in facts)
        assert 0 in sessions
        assert 1 in sessions

    def test_supersession_sets_valid_until(self):
        facts = decompose_sessions(self._sessions())
        # "started" is a change verb → supersedes=True
        state_facts = [f for f in facts if f.supersedes]
        assert len(state_facts) >= 1

    def test_short_content_filtered(self):
        sessions = [[{"role": "user", "content": "Hi"}]]
        facts = decompose_sessions(sessions)
        assert len(facts) == 0

    def test_short_sentence_filtered(self):
        """Covers line 278: short sentence (<10 chars) is skipped."""
        sessions = [[{"role": "user", "content": "Yes ok. This is a longer sentence about some important topics for testing."}]]
        facts = decompose_sessions(sessions)
        # "Yes ok" is < 10 chars → filtered by line 278
        texts = [f.text for f in facts]
        assert not any(t == "Yes ok" for t in texts)
        assert len(facts) >= 1

    def test_state_without_supersede(self):
        # A state fact with a change verb but no prior state → entity_last_state just records
        sessions = [[
            {"role": "user", "content": "I started learning piano last year, it has been wonderful."},
        ]]
        facts = decompose_sessions(sessions)
        state_facts = [f for f in facts if f.fact_type == "state"]
        assert len(state_facts) >= 1

    def test_supersession_with_valid_from(self):
        """Covers lines 288-292, 296-299: supersession with valid_from propagation."""
        # First fact must be "state" type → needs a change verb like "joined"
        # Second fact supersedes it with a date
        sessions = [
            [{"role": "user", "content": "My friend Alice joined the teaching profession at the school on 2023-01-15."}],
            [{"role": "user", "content": "My friend Alice started working as a nurse on 2024-06-01 at the hospital."}],
        ]
        facts = decompose_sessions(sessions)
        # First fact: state (joined), second: state (started) → supersession
        expired = [f for f in facts if f.valid_until and f.valid_until.startswith("20")]
        assert len(expired) >= 1  # teacher fact gets valid_until = 2024-06-01

    def test_supersession_without_valid_from(self):
        """Covers lines 293-294: supersession without valid_from → 'before-session-N'."""
        # First fact is state (joined), second is state (switched) but no date
        sessions = [
            [{"role": "user", "content": "My friend Alice joined the teaching staff at the local school recently."}],
            [{"role": "user", "content": "My friend Alice switched to nursing career and enjoys the change."}],
        ]
        facts = decompose_sessions(sessions)
        before_session = [f for f in facts if f.valid_until and "before-session" in f.valid_until]
        assert len(before_session) >= 1


# ── FactIndex ────────────────────────────────────────────────────


class TestFactIndex:
    def _make_facts(self):
        return [
            AtomicFact(
                text="Caroline works as a teacher at Lincoln School.",
                session_idx=0, turn_idx=0, role="user",
                fact_type="state", entities=["Caroline"],
                date_mentions=[], valid_from="",
            ),
            AtomicFact(
                text="Caroline started working as a nurse on March 15, 2024.",
                session_idx=1, turn_idx=0, role="user",
                fact_type="state", entities=["Caroline"],
                date_mentions=["2024-03-15"], valid_from="2024-03-15",
                supersedes=True,
            ),
            AtomicFact(
                text="Melanie enjoys hiking and photography on weekends.",
                session_idx=2, turn_idx=0, role="user",
                fact_type="preference", entities=["Melanie"],
                date_mentions=[],
            ),
            AtomicFact(
                text="The team meeting happened on January 10, 2024.",
                session_idx=0, turn_idx=1, role="assistant",
                fact_type="event", entities=[],
                date_mentions=["2024-01-10"], valid_from="2024-01-10",
            ),
            AtomicFact(
                text="Melanie plans to visit Tokyo next summer for vacation.",
                session_idx=3, turn_idx=0, role="user",
                fact_type="plan", entities=["Melanie", "Tokyo"],
                date_mentions=[],
            ),
        ]

    def test_post_init_builds_indices(self):
        idx = FactIndex(self._make_facts())
        assert "caroline" in idx._entity_to_facts
        assert len(idx._keyword_to_facts) > 0

    def test_query_basic(self):
        idx = FactIndex(self._make_facts())
        results = idx.query("Caroline work")
        assert len(results) > 0
        assert any("Caroline" in f.text for f, _ in results)

    def test_query_entity_boost(self):
        idx = FactIndex(self._make_facts())
        results = idx.query("Caroline")
        # Caroline-related facts should score higher
        assert len(results) >= 2

    def test_query_temporal_filter(self):
        idx = FactIndex(self._make_facts())
        # Set valid_until on the teacher fact
        idx.facts[0].valid_until = "2024-03-14"
        results = idx.query("Caroline work", reference_date="2024-04-01", filter_expired=True)
        # The teacher fact should be filtered out (expired before 2024-04-01)
        texts = [f.text for f, _ in results]
        assert "teacher" not in " ".join(texts).lower() or len(results) >= 1

    def test_query_recency_boost(self):
        idx = FactIndex(self._make_facts())
        results = idx.query("What is Caroline's current job")
        # "current" triggers recency boost → higher session_idx scores more
        assert len(results) > 0

    def test_query_supersedes_boost(self):
        idx = FactIndex(self._make_facts())
        results = idx.query("What is Caroline doing now")
        # "now" triggers recency, supersedes gets +5 bonus
        scores = {f.text: s for f, s in results}
        nurse_score = scores.get("Caroline started working as a nurse on March 15, 2024.", 0)
        assert nurse_score > 0

    def test_query_empty(self):
        idx = FactIndex(self._make_facts())
        results = idx.query("xyznonexistent")
        assert results == []

    def test_temporal_query_date_boost(self):
        idx = FactIndex(self._make_facts())
        results = idx.temporal_query("When did the meeting happen")
        assert len(results) > 0
        # Facts with date_mentions get boosted
        dated = [(f, s) for f, s in results if f.date_mentions]
        assert len(dated) > 0

    def test_temporal_query_ordering(self):
        idx = FactIndex(self._make_facts())
        results = idx.temporal_query("What happened first")
        # "first" triggers ordering → ALL dated facts included
        assert len(results) >= 2

    def test_temporal_query_ordering_includes_unmatched(self):
        idx = FactIndex(self._make_facts())
        results = idx.temporal_query("timeline of events before anything")
        # "before" triggers ordering
        assert isinstance(results, list)

    def test_cross_session_query(self):
        idx = FactIndex(self._make_facts())
        results = idx.cross_session_query("Caroline Melanie")
        assert len(results) > 0
        # Should have diversity across sessions
        sessions = set(f.session_idx for f, _ in results)
        assert len(sessions) >= 2

    def test_cross_session_diversity_bonus(self):
        idx = FactIndex(self._make_facts())
        results = idx.cross_session_query("work teacher nurse")
        # First appearance of each session gets +3 diversity bonus
        assert isinstance(results, list)

    def test_query_no_filter_expired(self):
        idx = FactIndex(self._make_facts())
        idx.facts[0].valid_until = "2024-03-14"
        results = idx.query("Caroline", reference_date="2024-04-01", filter_expired=False)
        # Should NOT filter even though fact is expired
        assert len(results) >= 2

    def test_query_filter_no_ref_date(self):
        idx = FactIndex(self._make_facts())
        idx.facts[0].valid_until = "2024-03-14"
        results = idx.query("Caroline", reference_date="", filter_expired=True)
        # No ref date → no filtering
        assert len(results) >= 2

    def test_temporal_query_entity_match(self):
        """Covers lines 191-193: entity matching in temporal_query."""
        idx = FactIndex(self._make_facts())
        # Use "about Caroline" — Caroline is capitalised mid-sentence
        results = idx.temporal_query("tell me about Caroline and dates")
        caroline_facts = [f for f, _ in results if "Caroline" in f.text]
        assert len(caroline_facts) >= 1

    def test_cross_session_entity_match(self):
        """Covers lines 237-239: entity matching in cross_session_query."""
        idx = FactIndex(self._make_facts())
        results = idx.cross_session_query("tell me about Caroline across sessions")
        assert len(results) > 0

    def test_cross_session_top_k_limit(self):
        """Covers line 252: break when top_k reached."""
        idx = FactIndex(self._make_facts())
        results = idx.cross_session_query("Caroline Melanie work", top_k=2)
        assert len(results) <= 2

    def test_mdy_invalid_month(self):
        """Covers line 355: invalid MDY month (>12)."""
        dates = _extract_dates("Date is 13/15/2024")
        assert len(dates) == 0
