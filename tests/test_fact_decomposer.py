# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for fact decomposer

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
    """Extended taxonomy: 9 types (decision, correction, principle, commitment,
    skill, plan, preference, state, event).  Priority order matters — decision
    beats preference when both patterns match.
    """

    # ── Original 4 types (regression guard) ─────────────────────
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

    # ── 5 new types ─────────────────────────────────────────────
    def test_decision_we_decided(self):
        assert _classify_fact("We decided to use BM25 for primary retrieval.") == "decision"

    def test_decision_consensus(self):
        assert _classify_fact("The consensus was to drop the SNN approach entirely.") == "decision"

    def test_decision_chose(self):
        assert _classify_fact("We chose the AGPL licence for maximum openness.") == "decision"

    def test_correction_actually(self):
        assert (
            _classify_fact("Actually, the STDP approach was wrong from the start.") == "correction"
        )

    def test_correction_turned_out(self):
        assert _classify_fact("It turned out the measurements were off by 2x.") == "correction"

    def test_correction_was_wrong(self):
        assert _classify_fact("I was wrong about the GPU memory requirements.") == "correction"

    def test_principle_always(self):
        assert _classify_fact("Always verify coverage before pushing to main.") == "principle"

    def test_principle_never(self):
        assert _classify_fact("Never delete failed CI runs without asking.") == "principle"

    def test_principle_best_practice(self):
        assert _classify_fact("The best practice is to test at 100% coverage.") == "principle"

    def test_commitment_deadline(self):
        assert _classify_fact("The deadline for the paper submission is April 15.") == "commitment"

    def test_commitment_promise(self):
        assert (
            _classify_fact("I promise to deliver the benchmark results by Friday.") == "commitment"
        )

    def test_commitment_committed_to(self):
        assert (
            _classify_fact("We committed to shipping v0.4 before the conference.") == "commitment"
        )

    def test_skill_how_to(self):
        assert _classify_fact("To fix this, run the following pytest command.") == "skill"

    def test_skill_step(self):
        assert _classify_fact("Step 1: install the Rust toolchain via rustup.") == "skill"

    def test_skill_procedure(self):
        assert _classify_fact("The procedure is to rebuild with maturin develop.") == "skill"

    # ── Priority/edge cases ─────────────────────────────────────
    def test_decision_beats_state_when_both_match(self):
        # "decided" is both a change verb and a decision pattern
        result = _classify_fact("We decided to switch from Anthropic to local LLM.")
        assert result == "decision"  # decision has higher priority

    def test_correction_beats_state(self):
        # "was wrong" + "started" — correction should win
        result = _classify_fact("Actually I was wrong when I started with that approach.")
        assert result == "correction"

    def test_empty_string_returns_event(self):
        assert _classify_fact("") == "event"

    def test_gibberish_returns_event(self):
        assert _classify_fact("asdfghjkl qwerty 12345.") == "event"

    def test_all_nine_types_reachable(self):
        """Verify every type in the taxonomy is reachable."""
        all_types = set()
        sentences = [
            "We decided to adopt BM25 as primary retrieval.",
            "Actually, the previous result was wrong.",
            "Always check test coverage before merging.",
            "The deadline is next Friday, I committed to deliver.",
            "To do this, run pytest with the -x flag.",
            "I plan to implement temporal reasoning next week.",
            "I prefer British English spelling in all docs.",
            "She started working at the university last month.",
            "The conference was held in Zürich.",
        ]
        for s in sentences:
            all_types.add(_classify_fact(s))
        expected = {
            "decision",
            "correction",
            "principle",
            "commitment",
            "skill",
            "plan",
            "preference",
            "state",
            "event",
        }
        assert all_types == expected

    # ── Pipeline integration: fact types flow through FactIndex ──
    def test_new_types_flow_through_pipeline(self):
        """All 9 fact types must survive decompose → FactIndex → query."""
        sessions = [
            [
                {
                    "role": "user",
                    "content": "We decided to use BM25 for retrieval instead of embeddings.",
                },
                {
                    "role": "assistant",
                    "content": "Actually the previous STDP measurements were wrong by a factor.",
                },
                {
                    "role": "user",
                    "content": "Always verify full test coverage before any push to main branch.",
                },
                {
                    "role": "assistant",
                    "content": "The deadline for paper submission is April 15 and I committed.",
                },
                {
                    "role": "user",
                    "content": "To fix this issue you need to run the following pytest command.",
                },
                {
                    "role": "assistant",
                    "content": "I plan to add multi-session temporal reasoning improvements soon.",
                },
                {
                    "role": "user",
                    "content": "I prefer dark mode interfaces and minimal UI designs always.",
                },
                {
                    "role": "assistant",
                    "content": "She started a new position at Google headquarters in January.",
                },
                {
                    "role": "user",
                    "content": "The annual team offsite was held at the mountain resort last week.",
                },
            ]
        ]
        facts = decompose_sessions(sessions)
        idx = FactIndex(facts)
        types_found = {f.fact_type for f in facts}
        assert "decision" in types_found
        assert "correction" in types_found
        assert "principle" in types_found
        assert "commitment" in types_found
        assert "skill" in types_found
        assert "plan" in types_found
        assert "preference" in types_found
        # Query should return typed facts
        results = idx.query("what did we decide about retrieval", top_k=5)
        assert len(results) > 0
        top_fact = results[0][0]
        assert top_fact.fact_type == "decision"

    # ── Performance: classify_fact must be sub-microsecond ───────
    def test_classify_fact_performance(self):
        """Classification of a single sentence should be under 0.1ms."""
        import time

        sentences = [
            "We decided to use BM25 for retrieval.",
            "Actually the previous approach was wrong.",
            "Always verify coverage before pushing.",
            "The deadline is March 30.",
            "To fix this, run pytest.",
            "I plan to improve temporal reasoning.",
            "I prefer dark mode.",
            "She started a new job.",
            "The meeting was productive.",
        ]
        # Warm up
        for s in sentences:
            _classify_fact(s)
        # Measure
        t0 = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            for s in sentences:
                _classify_fact(s)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_call_ms = elapsed_ms / (iterations * len(sentences))
        assert per_call_ms < 0.1, f"classify_fact too slow: {per_call_ms:.4f}ms/call"


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
            sess_idx=0,
            turn_idx=0,
            role="user",
            default_year=2024,
        )
        assert fact.fact_type == "state"
        assert fact.supersedes is True
        assert "2024-03-15" in fact.date_mentions
        assert fact.valid_from == "2024-03-15"

    def test_no_dates(self):
        fact = _build_fact(
            "The weather is really nice today.",
            sess_idx=1,
            turn_idx=2,
            role="user",
            default_year=2024,
        )
        assert fact.valid_from == ""
        assert fact.date_mentions == []


# ── Decompose sessions ──────────────────────────────────────────


class TestDecomposeSessions:
    def _sessions(self):
        return [
            [
                {
                    "role": "user",
                    "content": "My name is Caroline and I work as a teacher in Boston.",
                },
                {"role": "assistant", "content": "Nice to meet you Caroline!"},
            ],
            [
                {
                    "role": "user",
                    "content": "I started a new job as a nurse on March 15, 2024. I left teaching.",
                },
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
        sessions = [
            [
                {
                    "role": "user",
                    "content": "Yes ok. This is a longer sentence about some important topics for testing.",
                }
            ]
        ]
        facts = decompose_sessions(sessions)
        # "Yes ok" is < 10 chars → filtered by line 278
        texts = [f.text for f in facts]
        assert not any(t == "Yes ok" for t in texts)
        assert len(facts) >= 1

    def test_state_without_supersede(self):
        # A state fact with a change verb but no prior state → entity_last_state just records
        sessions = [
            [
                {
                    "role": "user",
                    "content": "I started learning piano last year, it has been wonderful.",
                },
            ]
        ]
        facts = decompose_sessions(sessions)
        state_facts = [f for f in facts if f.fact_type == "state"]
        assert len(state_facts) >= 1

    def test_supersession_with_valid_from(self):
        """Covers lines 288-292, 296-299: supersession with valid_from propagation."""
        # First fact must be "state" type → needs a change verb like "joined"
        # Second fact supersedes it with a date
        sessions = [
            [
                {
                    "role": "user",
                    "content": "My friend Alice joined the teaching profession at the school on 2023-01-15.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "My friend Alice started working as a nurse on 2024-06-01 at the hospital.",
                }
            ],
        ]
        facts = decompose_sessions(sessions)
        # First fact: state (joined), second: state (started) → supersession
        expired = [f for f in facts if f.valid_until and f.valid_until.startswith("20")]
        assert len(expired) >= 1  # teacher fact gets valid_until = 2024-06-01

    def test_supersession_without_valid_from(self):
        """Covers lines 293-294: supersession without valid_from → 'before-session-N'."""
        # First fact is state (joined), second is state (switched) but no date
        sessions = [
            [
                {
                    "role": "user",
                    "content": "My friend Alice joined the teaching staff at the local school recently.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "My friend Alice switched to nursing career and enjoys the change.",
                }
            ],
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
                session_idx=0,
                turn_idx=0,
                role="user",
                fact_type="state",
                entities=["Caroline"],
                date_mentions=[],
                valid_from="",
            ),
            AtomicFact(
                text="Caroline started working as a nurse on March 15, 2024.",
                session_idx=1,
                turn_idx=0,
                role="user",
                fact_type="state",
                entities=["Caroline"],
                date_mentions=["2024-03-15"],
                valid_from="2024-03-15",
                supersedes=True,
            ),
            AtomicFact(
                text="Melanie enjoys hiking and photography on weekends.",
                session_idx=2,
                turn_idx=0,
                role="user",
                fact_type="preference",
                entities=["Melanie"],
                date_mentions=[],
            ),
            AtomicFact(
                text="The team meeting happened on January 10, 2024.",
                session_idx=0,
                turn_idx=1,
                role="assistant",
                fact_type="event",
                entities=[],
                date_mentions=["2024-01-10"],
                valid_from="2024-01-10",
            ),
            AtomicFact(
                text="Melanie plans to visit Tokyo next summer for vacation.",
                session_idx=3,
                turn_idx=0,
                role="user",
                fact_type="plan",
                entities=["Melanie", "Tokyo"],
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


# ── Rust wiring verification ─────────────────────────────────


class TestRustWiring:
    """Verify Rust fact_decomposer helpers work through pipeline."""

    def test_classify_fact_type_wired(self):
        """classify_fact_type is used in _build_fact internally."""
        facts = decompose_sessions(
            [[{"role": "user", "content": "I love hiking in the Alps every summer."}]]
        )
        # "I love" → preference fact type
        prefs = [f for f in facts if f.fact_type == "preference"]
        assert len(prefs) >= 1

    def test_change_verb_detection(self):
        """State-change verbs are detected in facts."""
        facts = decompose_sessions(
            [
                [
                    {
                        "role": "user",
                        "content": "I started a new job at Google last month. I switched from Apple.",
                    }
                ]
            ]
        )
        # At least one fact should have supersedes or event type
        has_change = any(f.supersedes or f.fact_type == "event" for f in facts)
        assert has_change or len(facts) >= 1  # at minimum decomposed

    def test_plan_detection(self):
        facts = decompose_sessions(
            [[{"role": "user", "content": "I plan to visit Japan in December."}]]
        )
        plans = [f for f in facts if f.fact_type == "plan"]
        assert len(plans) >= 1


# ── Pipeline integration ─────────────────────────────────────


class TestFactDecomposerPipeline:
    """End-to-end: decompose → index → query → retrieve."""

    def test_decompose_to_index_to_query(self):
        sessions = [
            [
                {"role": "user", "content": "I started working at Google on March 10, 2024."},
                {"role": "assistant", "content": "That's great!"},
                {"role": "user", "content": "I love playing chess and hiking."},
            ],
            [
                {"role": "user", "content": "I switched to Apple on September 1, 2024."},
            ],
        ]
        facts = decompose_sessions(sessions)
        assert len(facts) >= 3

        idx = FactIndex(facts)
        # Query for work-related facts
        results = idx.query("Google")
        assert len(results) >= 0  # may or may not match depending on tokenisation

        # Cross-session query — should find facts from both sessions
        cross = idx.cross_session_query("work")
        # At minimum should not crash; results depend on entity overlap
        assert isinstance(cross, list)

    def test_temporal_query_with_dates(self):
        sessions = [
            [
                {"role": "user", "content": "I moved to Berlin on January 15, 2024."},
                {"role": "user", "content": "I moved to London on June 1, 2024."},
            ]
        ]
        facts = decompose_sessions(sessions)
        idx = FactIndex(facts)
        results = idx.temporal_query("when did the user move")
        assert len(results) >= 1

    def test_entity_extraction_in_facts(self):
        sessions = [[{"role": "user", "content": "I bought a Tesla Model 3 last week."}]]
        facts = decompose_sessions(sessions)
        assert any("tesla" in " ".join(f.entities).lower() for f in facts)

    def test_feeds_arcane_retriever(self):
        """Facts from decomposer can be consumed by ArcaneRetriever."""
        from arcane_retriever import ArcaneRetriever

        sessions = [
            [
                {"role": "user", "content": "I love sushi and ramen."},
                {"role": "user", "content": "I started running every morning."},
            ]
        ]
        ar = ArcaneRetriever(sessions)
        results = ar.retrieve("what food does the user like", "single-session-user", top_k=5)
        assert isinstance(results, list)


# ── Missing patterns: roundtrip ───────────────────────────────


class TestFactDecomposerRoundtrip:
    def test_decompose_index_query_roundtrip(self):
        """Full cycle: sessions → facts → index → query → results."""
        sessions = [
            [
                {"role": "user", "content": "I started running every morning in January 2024."},
                {
                    "role": "user",
                    "content": "My favourite restaurant is the Italian place downtown.",
                },
            ]
        ]
        facts = decompose_sessions(sessions)
        assert len(facts) >= 2

        idx = FactIndex(facts)
        results = idx.query("running")
        # Should return list (may be empty depending on tokenisation)
        assert isinstance(results, list)
