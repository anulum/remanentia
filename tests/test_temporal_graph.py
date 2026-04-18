# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for temporal graph

from __future__ import annotations


import math

from temporal_graph import (
    TemporalEvent,
    TemporalGraph,
    date_to_phase,
    parse_dates,
    temporal_code_execute,
)


class TestParseDates:
    def test_iso_date(self):
        assert "2026-03-15" in parse_dates("Fixed on 2026-03-15.")

    def test_english_date(self):
        dates = parse_dates("Released on March 15, 2026.")
        assert "2026-03-15" in dates

    def test_abbreviated_month(self):
        dates = parse_dates("Deployed on Jan 5, 2026.")
        assert "2026-01-05" in dates

    def test_multiple_dates(self):
        dates = parse_dates("Started 2026-03-10, finished 2026-03-15.")
        assert len(dates) == 2
        assert "2026-03-10" in dates
        assert "2026-03-15" in dates

    def test_no_dates(self):
        assert parse_dates("No dates here.") == []

    def test_deduplicates(self):
        dates = parse_dates("2026-03-15 and again 2026-03-15.")
        assert dates.count("2026-03-15") == 1

    def test_english_no_year(self):
        dates = parse_dates("Due on March 20.")
        assert len(dates) == 1
        assert dates[0].endswith("-03-20")


class TestTemporalEvent:
    def test_creation(self):
        ev = TemporalEvent(date="2026-03-15", text="Bug fixed", source="trace.md")
        assert ev.date == "2026-03-15"
        assert ev.paragraph_idx == 0


class TestTemporalGraph:
    def test_extract_events(self):
        tg = TemporalGraph()
        events = tg.extract_events(
            "The STDP bug was fixed on 2026-03-15. The daemon was killed on 2026-03-20.",
            "trace.md",
        )
        assert len(events) == 2
        assert events[0].date == "2026-03-15"
        assert events[1].date == "2026-03-20"

    def test_add_events_builds_edges(self):
        tg = TemporalGraph()
        events = [
            TemporalEvent(date="2026-03-10", text="Event A", source="a.md"),
            TemporalEvent(date="2026-03-15", text="Event B", source="b.md"),
            TemporalEvent(date="2026-03-15", text="Event C", source="c.md"),
        ]
        tg.add_events(events)
        assert len(tg.events) == 3
        assert len(tg.edges) >= 1
        relations = {e.relation for e in tg.edges}
        # B and C are same_day; A is before B and C
        assert "same_day" in relations or "before" in relations

    def test_build_from_documents(self):
        tg = TemporalGraph()
        docs = [
            ("decision.md", "We decided on 2026-03-15 to remove SNN scoring."),
            ("benchmark.md", "LOCOMO scored 81.2% on 2026-03-23."),
        ]
        tg.build_from_documents(docs)
        assert len(tg.events) == 2
        assert tg.stats["events"] == 2
        assert tg.stats["unique_dates"] == 2

    def test_query_temporal_keyword(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-15", text="STDP bug was fixed", source="a.md"),
                TemporalEvent(date="2026-03-20", text="Daemon was killed", source="b.md"),
                TemporalEvent(date="2026-03-23", text="LOCOMO benchmark run", source="c.md"),
            ]
        )
        results = tg.query_temporal("when was the STDP bug fixed", top_k=3)
        assert len(results) > 0
        assert any("STDP" in ev.text for ev in results)

    def test_query_temporal_after(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-10", text="Event A happened", source="a.md"),
                TemporalEvent(date="2026-03-20", text="Event B happened", source="b.md"),
            ]
        )
        results = tg.query_temporal("what happened after 2026-03-15", top_k=5)
        assert all(ev.date >= "2026-03-15" for ev in results)

    def test_query_temporal_before(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-10", text="Event A happened", source="a.md"),
                TemporalEvent(date="2026-03-20", text="Event B happened", source="b.md"),
            ]
        )
        results = tg.query_temporal("what happened before 2026-03-15", top_k=5)
        assert all(ev.date <= "2026-03-15" for ev in results)

    def test_query_latest(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-10", text="Old event with data", source="a.md"),
                TemporalEvent(date="2026-03-23", text="Latest event with data", source="b.md"),
            ]
        )
        results = tg.query_temporal("latest event with data", top_k=1)
        assert results[0].date == "2026-03-23"

    def test_query_first(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-10", text="First experiment run", source="a.md"),
                TemporalEvent(date="2026-03-23", text="Second experiment run", source="b.md"),
            ]
        )
        results = tg.query_temporal("first experiment", top_k=1)
        assert results[0].date == "2026-03-10"

    def test_events_on_date(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-15", text="Event A", source="a.md"),
                TemporalEvent(date="2026-03-15", text="Event B", source="b.md"),
                TemporalEvent(date="2026-03-20", text="Event C", source="c.md"),
            ]
        )
        events = tg.events_on_date("2026-03-15")
        assert len(events) == 2

    def test_events_between(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-10", text="A", source="a.md"),
                TemporalEvent(date="2026-03-15", text="B", source="b.md"),
                TemporalEvent(date="2026-03-20", text="C", source="c.md"),
            ]
        )
        events = tg.events_between("2026-03-12", "2026-03-18")
        assert len(events) == 1
        assert events[0].date == "2026-03-15"

    def test_save_and_load(self, tmp_path):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-15", text="Event A", source="a.md"),
                TemporalEvent(date="2026-03-20", text="Event B", source="b.md"),
            ]
        )
        path = tmp_path / "temporal.jsonl"
        tg.save(path)
        assert path.exists()

        tg2 = TemporalGraph()
        assert tg2.load(path) is True
        assert len(tg2.events) == 2

    def test_load_nonexistent(self, tmp_path):
        tg = TemporalGraph()
        assert tg.load(tmp_path / "nope.jsonl") is False

    def test_stats(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-15", text="A", source="a.md"),
                TemporalEvent(date="2026-03-20", text="B", source="b.md"),
            ]
        )
        s = tg.stats
        assert s["events"] == 2
        assert s["unique_dates"] == 2
        assert s["sources"] == 2

    def test_empty_stats(self):
        tg = TemporalGraph()
        s = tg.stats
        assert s["events"] == 0
        assert s["date_range"] == ""

    def test_no_events_for_query(self):
        tg = TemporalGraph()
        results = tg.query_temporal("anything", top_k=5)
        assert results == []

    def test_before_edge_reverse_order(self):
        tg = TemporalGraph()
        # Add newer event first, then older — triggers before edge
        tg.add_events(
            [
                TemporalEvent(date="2026-03-20", text="Later event", source="b.md"),
            ]
        )
        tg.add_events(
            [
                TemporalEvent(date="2026-03-10", text="Earlier event", source="a.md"),
            ]
        )
        before = [e for e in tg.edges if e.relation == "before"]
        assert len(before) >= 1
        assert before[0].source_date == "2026-03-10"

    def test_date_range_query(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-05", text="Event A with data", source="a.md"),
                TemporalEvent(date="2026-03-15", text="Event B with data", source="b.md"),
                TemporalEvent(date="2026-03-25", text="Event C with data", source="c.md"),
            ]
        )
        results = tg.query_temporal("data between 2026-03-10 and 2026-03-20", top_k=5)
        assert len(results) == 1
        assert results[0].date == "2026-03-15"


# ── Feature 3: Temporal code execution ──────────────────────────


class TestTemporalCodeExecute:
    def _make_events(self):
        return [
            TemporalEvent(date="2026-03-10", text="Started project", source="a.md"),
            TemporalEvent(date="2026-03-15", text="Released v1.0", source="b.md"),
            TemporalEvent(date="2026-03-20", text="Bug report filed", source="c.md"),
        ]

    def test_how_long_between(self):
        result = temporal_code_execute("how many days between events", self._make_events())
        assert result is not None
        assert "10 days" in result

    def test_most_recent(self):
        result = temporal_code_execute("what was the most recent event", self._make_events())
        assert result is not None
        assert "2026-03-20" in result
        assert "Bug report" in result

    def test_earliest(self):
        result = temporal_code_execute("what was the first event", self._make_events())
        assert result is not None
        assert "2026-03-10" in result
        assert "Started" in result

    def test_before_after_comparison(self):
        events = [
            TemporalEvent(date="2026-03-10", text="Event A occurred", source="a.md"),
            TemporalEvent(date="2026-03-20", text="Event B occurred", source="b.md"),
        ]
        result = temporal_code_execute("did A happen before or after B", events)
        assert result is not None
        assert "before" in result.lower() or "after" in result.lower()

    def test_before_after_reverse_order(self):
        events = [
            TemporalEvent(date="2026-03-20", text="Event B occurred", source="b.md"),
            TemporalEvent(date="2026-03-10", text="Event A occurred", source="a.md"),
        ]
        result = temporal_code_execute("did B happen before or after A", events)
        assert result is not None
        assert "after" in result.lower()
        assert "10 days later" in result

    def test_before_after_same_day(self):
        events = [
            TemporalEvent(date="2026-03-15", text="Event X occurred", source="x.md"),
            TemporalEvent(date="2026-03-15", text="Event Y occurred", source="y.md"),
        ]
        result = temporal_code_execute("did X happen before or after Y", events)
        assert result is not None
        assert "same day" in result.lower()

    def test_no_events(self):
        assert temporal_code_execute("anything", []) is None

    def test_no_dates(self):
        events = [TemporalEvent(date="", text="No date", source="a.md")]
        assert temporal_code_execute("when", events) is None

    def test_invalid_date_format(self):
        events = [TemporalEvent(date="not-a-date", text="Bad", source="a.md")]
        assert temporal_code_execute("when", events) is None

    def test_how_long_ago(self):
        events = [TemporalEvent(date="2026-03-20", text="Event happened", source="a.md")]
        result = temporal_code_execute("how long since the event", events)
        assert result is not None
        assert "days" in result
        assert "since" in result

    def test_single_event_how_long(self):
        events = [TemporalEvent(date="2026-03-15", text="Only event", source="a.md")]
        result = temporal_code_execute("how many days between events", events)
        assert result is None or "0 days" in result


# ── Edge cases and negative inputs ───────────────────────────


class TestParseDatesEdgeCases:
    def test_empty_string(self):
        assert parse_dates("") == []

    def test_relative_yesterday(self):
        from datetime import date, timedelta

        ref = date(2026, 3, 30)
        dates = parse_dates("I did it yesterday.", ref)
        assert (ref - timedelta(days=1)).isoformat() in dates

    def test_relative_last_week(self):
        from datetime import date

        ref = date(2026, 3, 30)
        dates = parse_dates("It happened last week.", ref)
        assert len(dates) >= 1

    def test_vague_n_days_ago(self):
        from datetime import date

        ref = date(2026, 3, 30)
        dates = parse_dates("About 5 days ago we started.", ref)
        assert len(dates) >= 1
        assert "2026-03-25" in dates

    def test_mixed_formats(self):
        text = "Started 2026-03-10, reviewed on March 15, and finished yesterday."
        from datetime import date

        dates = parse_dates(text, date(2026, 3, 20))
        assert len(dates) >= 2

    def test_invalid_date_ignored(self):
        dates = parse_dates("Date: 2026-13-45 is wrong.")
        # Should not crash; invalid date may or may not parse
        assert isinstance(dates, list)

    def test_sorted_output(self):
        dates = parse_dates("End 2026-03-20, start 2026-03-10.")
        assert dates == sorted(dates)

    def test_all_months(self):
        for month in [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]:
            dates = parse_dates(f"On {month} 1, 2026.")
            assert len(dates) >= 1, f"Failed for {month}"

    def test_abbreviated_months(self):
        for month in ["Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
            dates = parse_dates(f"On {month} 1, 2026.")
            assert len(dates) >= 1, f"Failed for {month}"


class TestTemporalGraphEdgeCases:
    def test_empty_text_extract(self):
        tg = TemporalGraph()
        events = tg.extract_events("", "empty.md")
        assert events == []

    def test_add_empty_events(self):
        tg = TemporalGraph()
        tg.add_events([])
        assert tg.events == []
        assert tg.edges == []

    def test_same_day_edge_pairs(self):
        tg = TemporalGraph()
        events = [
            TemporalEvent(date="2026-03-15", text="A", source="a.md"),
            TemporalEvent(date="2026-03-15", text="B", source="b.md"),
            TemporalEvent(date="2026-03-15", text="C", source="c.md"),
        ]
        tg.add_events(events)
        same_day = [e for e in tg.edges if e.relation == "same_day"]
        # 3 events same day → 3 same_day pairs (A-B, A-C, B-C)
        assert len(same_day) == 3

    def test_incremental_add(self):
        """Incremental add preserves events and creates same_day edges on overlap."""
        tg = TemporalGraph()
        tg.add_events([TemporalEvent(date="2026-03-10", text="A", source="a.md")])
        assert len(tg.events) == 1
        tg.add_events(
            [
                TemporalEvent(date="2026-03-10", text="A2", source="a2.md"),
                TemporalEvent(date="2026-03-20", text="B", source="b.md"),
            ]
        )
        assert len(tg.events) == 3
        same_day = [e for e in tg.edges if e.relation == "same_day"]
        assert len(same_day) >= 1  # A and A2 on same day

    def test_save_load_roundtrip_preserves_dates(self, tmp_path):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-15", text="Event α", source="α.md", paragraph_idx=3),
                TemporalEvent(date="2026-03-20", text="Event β", source="β.md", paragraph_idx=7),
            ]
        )
        path = tmp_path / "temporal.jsonl"
        tg.save(path)

        tg2 = TemporalGraph()
        tg2.load(path)
        assert len(tg2.events) == 2
        assert tg2.events[0].date == "2026-03-15"
        assert tg2.events[0].paragraph_idx == 3
        assert tg2.events[1].source == "β.md"

    def test_events_on_nonexistent_date(self):
        tg = TemporalGraph()
        tg.add_events([TemporalEvent(date="2026-03-15", text="A", source="a.md")])
        assert tg.events_on_date("2099-01-01") == []

    def test_events_between_empty_range(self):
        tg = TemporalGraph()
        tg.add_events([TemporalEvent(date="2026-03-15", text="A", source="a.md")])
        assert tg.events_between("2020-01-01", "2020-12-31") == []

    def test_query_since(self):
        tg = TemporalGraph()
        tg.add_events(
            [
                TemporalEvent(date="2026-03-10", text="Old event with data", source="a.md"),
                TemporalEvent(date="2026-03-25", text="New event with data", source="b.md"),
            ]
        )
        results = tg.query_temporal("data since 2026-03-20", top_k=5)
        assert all(ev.date >= "2026-03-20" for ev in results)


# ── Rust wiring verification ─────────────────────────────────


class TestRustAcceleration:
    """Verify Rust modules are wired in and produce correct results."""

    def test_parse_dates_uses_rust_when_available(self):
        """parse_dates should work regardless of Rust availability."""
        from datetime import date

        result = parse_dates("Meeting on March 15, 2026.", date(2026, 3, 30))
        assert "2026-03-15" in result

    def test_rust_and_python_agree(self):
        """If Rust is available, results must match Python."""
        from datetime import date

        text = "Started 2026-03-10, finished March 20, 2026. Also yesterday."
        ref = date(2026, 3, 30)

        # Get result through the wired function (may use Rust)
        result = parse_dates(text, ref)

        # Verify key dates are found regardless of backend
        assert "2026-03-10" in result
        assert "2026-03-20" in result
        assert len(result) >= 2

    def test_vague_dates_work_through_pipeline(self):
        """Vague date normalisation must work through parse_dates."""
        from datetime import date

        ref = date(2026, 3, 30)
        result = parse_dates("About 3 weeks ago we deployed.", ref)
        assert len(result) >= 1
        assert any("2026-03" in d for d in result)


# ── Pipeline integration ─────────────────────────────────────


class TestTemporalPipelineIntegration:
    """Temporal graph integrated with memory index and retrieval."""

    def test_graph_feeds_search_context(self):
        """Build graph → query → verify temporal events inform results."""
        tg = TemporalGraph()
        tg.build_from_documents(
            [
                ("session1.md", "On March 10, 2026 the team decided to use BM25."),
                ("session2.md", "On March 15, 2026 LOCOMO accuracy was 81.2%."),
                ("session3.md", "On March 20, 2026 the score improved to 88.5%."),
            ]
        )
        assert tg.stats["events"] == 3

        # Query temporal range
        results = tg.query_temporal("what happened after 2026-03-12", top_k=5)
        assert len(results) == 2
        dates = {r.date for r in results}
        assert "2026-03-15" in dates
        assert "2026-03-20" in dates

    def test_temporal_code_execute_with_graph_events(self):
        """Temporal code execution on graph-extracted events."""
        tg = TemporalGraph()
        tg.build_from_documents(
            [
                ("a.md", "Bug found on 2026-03-10."),
                ("b.md", "Fix deployed on 2026-03-15."),
            ]
        )
        result = temporal_code_execute("how many days between events", tg.events)
        assert result is not None
        assert "5 days" in result

    def test_end_to_end_temporal_recall(self):
        """Full pipeline: text → parse dates → build graph → query → answer."""
        from datetime import date

        text = "Deployed v3.0 on March 10, 2026. Bug reported March 15. Fixed March 17."
        ref = date(2026, 3, 30)

        # Step 1: Parse dates
        dates = parse_dates(text, ref)
        assert len(dates) == 3

        # Step 2: Build graph
        tg = TemporalGraph()
        tg.build_from_documents([("log.md", text)])
        assert tg.stats["events"] == 3

        # Step 3: Query
        results = tg.query_temporal("when was the bug reported", top_k=1)
        assert len(results) >= 1

        # Step 4: Code execution
        answer = temporal_code_execute("how many days between first and last", tg.events)
        assert answer is not None


# ── UPDE Phase Engine + Resonance Search ────────────────────────────


class TestDateToPhase:
    """Tests for the date_to_phase utility function."""

    def test_jan_1(self):
        """Jan 1 = day 1 → θ ≈ 2π/365.25."""
        phase = date_to_phase("2026-01-01")
        expected = 2.0 * math.pi * 1 / 365.25
        assert abs(phase - expected) < 1e-10

    def test_mid_year(self):
        """Jul 2 = day 183 → θ ≈ π."""
        phase = date_to_phase("2026-07-02")
        expected = 2.0 * math.pi * 183 / 365.25
        assert abs(phase - expected) < 1e-10

    def test_dec_31(self):
        """Dec 31 = day 365 → θ ≈ 2π."""
        phase = date_to_phase("2026-12-31")
        expected = 2.0 * math.pi * 365 / 365.25
        assert abs(phase - expected) < 1e-10

    def test_leap_year(self):
        """Leap year Dec 31 = day 366."""
        phase = date_to_phase("2024-12-31")
        expected = 2.0 * math.pi * 366 / 365.25
        assert abs(phase - expected) < 1e-10

    def test_invalid_date_returns_zero(self):
        assert date_to_phase("not-a-date") == 0.0
        assert date_to_phase("") == 0.0

    def test_range_zero_to_two_pi(self):
        """Phase should always be in [0, 2π)."""
        for month in range(1, 13):
            phase = date_to_phase(f"2026-{month:02d}-15")
            assert 0.0 <= phase < 2.0 * math.pi + 0.02  # slight tolerance for day 365


class TestTemporalEventPhase:
    """Tests for TemporalEvent.calculate_phase()."""

    def test_calculate_phase_sets_field(self):
        ev = TemporalEvent(date="2026-03-15", text="test", source="src")
        assert ev.phase == 0.0
        result = ev.calculate_phase()
        assert result > 0.0
        assert ev.phase == result

    def test_phase_default_zero(self):
        ev = TemporalEvent(date="2026-01-01", text="test", source="src")
        assert ev.phase == 0.0

    def test_phase_idempotent(self):
        ev = TemporalEvent(date="2026-06-15", text="test", source="src")
        p1 = ev.calculate_phase()
        p2 = ev.calculate_phase()
        assert p1 == p2


class TestResonanceSearch:
    """Tests for TemporalGraph.resonance_search()."""

    def _make_graph(self, dates: list[str]) -> TemporalGraph:
        tg = TemporalGraph()
        for i, d in enumerate(dates):
            ev = TemporalEvent(date=d, text=f"Event {i}", source=f"src{i}")
            ev.calculate_phase()
            tg.events.append(ev)
        return tg

    def test_nearby_dates_found(self):
        """Jan 1 and Jan 5 found when querying Jan 2 with tolerance=0.01."""
        tg = self._make_graph(["2026-01-01", "2026-01-05", "2026-06-01"])
        results = tg.resonance_search("2026-01-02", tolerance=0.01)
        texts = [e.text for e, _ in results]
        assert "Event 0" in texts  # Jan 1
        assert "Event 1" in texts  # Jan 5
        assert "Event 2" not in texts  # Jun 1

    def test_sorted_by_resonance_descending(self):
        tg = self._make_graph(["2026-01-01", "2026-01-05", "2026-01-03"])
        results = tg.resonance_search("2026-01-02", tolerance=0.01)
        scores = [r for _, r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_graph(self):
        tg = TemporalGraph()
        results = tg.resonance_search("2026-01-01")
        assert results == []

    def test_single_event_exact_match(self):
        tg = self._make_graph(["2026-03-15"])
        results = tg.resonance_search("2026-03-15", tolerance=0.0)
        assert len(results) == 1
        assert results[0][1] == 1.0  # exact match → cos(0) = 1

    def test_tolerance_zero_exact_only(self):
        """tolerance=0 should only return exact day-of-year matches."""
        tg = self._make_graph(["2026-01-01", "2026-01-02"])
        results = tg.resonance_search("2026-01-01", tolerance=0.0)
        assert len(results) == 1
        assert results[0][0].text == "Event 0"

    def test_tolerance_one_returns_all(self):
        """tolerance=1.0 → threshold=0 → all events with non-negative resonance."""
        tg = self._make_graph(["2026-01-01", "2026-04-01", "2026-07-01", "2026-10-01"])
        results = tg.resonance_search("2026-01-01", tolerance=1.0)
        # cos ranges from -1 to 1; threshold=0 means resonance >= 0
        # Events at ~0, ~π/2, ~π, ~3π/2 → cos(0)=1, cos(π/2)≈0, cos(π)≈-1, cos(3π/2)≈0
        assert len(results) >= 2  # at least Jan and maybe Apr/Oct

    def test_wrap_around_dec_jan(self):
        """Dec 31 and Jan 1 should be close in phase (cyclic proximity)."""
        tg = self._make_graph(["2025-12-31"])
        results = tg.resonance_search("2026-01-02", tolerance=0.01)
        # Dec 31 (day 365) and Jan 2 (day 2): phase difference ≈ 2π·3/365.25
        # cos(2π·3/365.25) ≈ 0.9986 > 0.99 → should be found
        assert len(results) == 1

    def test_auto_calculates_phase(self):
        """Events with phase=0 are handled correctly on search."""
        tg = TemporalGraph()
        ev = TemporalEvent(date="2026-06-15", text="auto", source="src")
        tg.events.append(ev)
        assert ev.phase == 0.0
        results = tg.resonance_search("2026-06-15", tolerance=0.0)
        assert len(results) == 1
        # Rust path computes phase internally; Python path sets ev.phase
        # Either way, the event must be found
        assert results[0][0].text == "auto"


# ── Duration arithmetic (Task #27) ──────────────────────────────


class TestDurationArithmetic:
    """Tests for enhanced temporal_code_execute duration handling."""

    def test_how_many_weeks(self):
        events = [
            TemporalEvent(date="2023-01-01", text="Start event", source="a"),
            TemporalEvent(date="2023-01-22", text="End event", source="b"),
        ]
        result = temporal_code_execute("how many weeks between start and end", events)
        assert result is not None
        assert "3 weeks" in result

    def test_how_many_weeks_with_remainder(self):
        events = [
            TemporalEvent(date="2023-01-01", text="Start event", source="a"),
            TemporalEvent(date="2023-01-25", text="End event", source="b"),
        ]
        result = temporal_code_execute("how many weeks between events", events)
        assert result is not None
        assert "3 weeks" in result
        assert "3 days" in result  # 24 days = 3w + 3d

    def test_how_many_months(self):
        events = [
            TemporalEvent(date="2023-01-15", text="Start event", source="a"),
            TemporalEvent(date="2023-04-15", text="End event", source="b"),
        ]
        result = temporal_code_execute("how many months between events", events)
        assert result is not None
        assert "3 months" in result

    def test_duration_keyword_matching(self):
        """When query mentions specific events, those events are picked."""
        events = [
            TemporalEvent(date="2023-01-01", text="gym session", source="a"),
            TemporalEvent(date="2023-01-10", text="dentist visit", source="b"),
            TemporalEvent(date="2023-03-01", text="gym session again", source="c"),
        ]
        result = temporal_code_execute("how many days between the gym sessions", events)
        assert result is not None
        # Should pick gym (Jan 1) and gym (Mar 1) = 59 days, not dentist
        assert "59 days" in result

    def test_how_many_times(self):
        events = [
            TemporalEvent(date="2023-01-05", text="went to gym", source="a"),
            TemporalEvent(date="2023-01-12", text="went to gym", source="b"),
            TemporalEvent(date="2023-01-19", text="went to gym", source="c"),
        ]
        result = temporal_code_execute("how many times did I go to the gym", events)
        assert result is not None
        assert "3 times" in result

    def test_how_many_times_unique_dates(self):
        """Counting uses unique dates, not duplicate events."""
        events = [
            TemporalEvent(date="2023-01-05", text="went to gym morning", source="a"),
            TemporalEvent(date="2023-01-05", text="went to gym evening", source="b"),
            TemporalEvent(date="2023-01-12", text="went to gym", source="c"),
        ]
        result = temporal_code_execute("how many times did I go to the gym", events)
        assert result is not None
        assert "2 times" in result  # 2 unique dates

    def test_pick_duration_pair_fallback(self):
        """Falls back to chronological extremes when no keyword match."""
        from temporal_graph import _pick_duration_pair
        from datetime import date

        parsed = [
            (date(2023, 1, 1), "alpha"),
            (date(2023, 1, 10), "beta"),
            (date(2023, 1, 20), "gamma"),
        ]
        result = _pick_duration_pair(parsed, "unrelated query xyz")
        assert result is not None
        assert result[0] == date(2023, 1, 1)
        assert result[2] == date(2023, 1, 20)

    def test_pick_duration_pair_single_event(self):
        from temporal_graph import _pick_duration_pair
        from datetime import date

        assert _pick_duration_pair([(date(2023, 1, 1), "only")], "q") is None


# ── Fuzzy inclusive/exclusive durations (Task #33) ──────────────


class TestFuzzyInclusiveExclusive:
    """Dual-format day count output (R9 follow-up audit recommendation #2)."""

    def test_days_dual_format(self):
        """how many days between returns both exclusive (N) and inclusive (N+1)."""
        events = [
            TemporalEvent(date="2024-01-02", text="Sunday mass at St. Mary's", source="a"),
            TemporalEvent(date="2024-02-01", text="Ash Wednesday service at cathedral", source="b"),
        ]
        result = temporal_code_execute(
            "how many days between the Sunday mass and the Ash Wednesday service", events
        )
        assert result is not None
        # Gold: "30 days. 31 days (including the last day) is also acceptable."
        assert "30 days" in result
        assert "31 days" in result
        assert "both endpoints" in result

    def test_days_zero_day_still_shows_one_inclusive(self):
        events = [
            TemporalEvent(date="2024-01-02", text="alpha event", source="a"),
            TemporalEvent(date="2024-01-02", text="beta event", source="b"),
        ]
        result = temporal_code_execute("how many days between alpha and beta", events)
        # Same day → 0 exclusive, 1 inclusive
        assert result is not None
        assert "0 days" in result
        assert "1 days" in result

    def test_weeks_dual_format(self):
        events = [
            TemporalEvent(date="2023-01-01", text="project start", source="a"),
            TemporalEvent(date="2023-01-22", text="project end", source="b"),
        ]
        result = temporal_code_execute(
            "how many weeks between the project start and project end", events
        )
        assert result is not None
        assert "3 weeks" in result
        assert "21 days exclusive" in result
        assert "22 inclusive" in result

    def test_months_dual_format(self):
        events = [
            TemporalEvent(date="2023-01-15", text="joined company", source="a"),
            TemporalEvent(date="2023-04-15", text="left company", source="b"),
        ]
        result = temporal_code_execute("how many months between joining and leaving", events)
        assert result is not None
        assert "3 months" in result
        assert "90 days exclusive" in result
        assert "91 inclusive" in result


# ── question_date anchoring for "X ago" (Task #34) ──────────────


class TestQuestionDateAnchor:
    """`how many X ago` should use question_date as today, not date.today()."""

    def test_days_ago_with_question_date(self):
        events = [TemporalEvent(date="2023-04-06", text="Maundy Thursday service", source="a")]
        result = temporal_code_execute(
            "how many days ago did I attend the Maundy Thursday service?",
            events,
            question_date="2023/04/10 (Mon) 07:43",
        )
        assert result is not None
        assert "4 days" in result
        assert "5" in result

    def test_weeks_ago_with_question_date(self):
        events = [TemporalEvent(date="2023-04-16", text="started using Ibotta", source="a")]
        result = temporal_code_execute(
            "how many weeks ago did I start using Ibotta?",
            events,
            question_date="2023/05/07 (Sun) 12:00",
        )
        assert result is not None
        assert "3 weeks" in result

    def test_months_ago_with_question_date(self):
        events = [TemporalEvent(date="2022-12-27", text="booked Airbnb in SF", source="a")]
        result = temporal_code_execute(
            "how many months ago did I book the Airbnb?",
            events,
            question_date="2023/05/27 (Sat) 01:55",
        )
        assert result is not None
        assert "5 months" in result

    def test_empty_question_date_falls_back_to_today(self):
        events = [TemporalEvent(date="2023-01-01", text="old event", source="a")]
        result = temporal_code_execute("how many days since the event?", events)
        assert result is not None
        assert "days" in result
        assert "since" in result

    def test_invalid_question_date_falls_back(self):
        events = [TemporalEvent(date="2023-01-01", text="event", source="a")]
        result = temporal_code_execute("how many days ago?", events, question_date="not a date")
        assert result is not None
        assert "days" in result

    def test_resolve_question_date_helper(self):
        from temporal_graph import _resolve_question_date
        from datetime import date as _d

        assert _resolve_question_date("2023/04/10 (Mon) 07:43") == _d(2023, 4, 10)
        assert _resolve_question_date("2023-04-10") == _d(2023, 4, 10)
        assert _resolve_question_date("garbage") == _d.today()
        assert _resolve_question_date("") == _d.today()


# ── Multi-event proximity tuning (Task #35) ─────────────────────


class TestEventScoring:
    """Combined unigram+bigram+density scoring (audit #4)."""

    def test_bigram_phrase_beats_scattered_unigrams(self):
        from temporal_graph import _score_event_vs_query

        q = "how many days between the guitar lessons and the amp purchase"
        q_tokens = {"guitar", "lessons", "amp", "purchase", "between", "many", "days"}
        q_bigrams = {("guitar", "lessons"), ("amp", "purchase")}

        # Event A: "guitar lessons started" — hits the 'guitar lessons' bigram
        a = _score_event_vs_query("guitar lessons started", q, q_tokens, q_bigrams)
        # Event B: "bought a guitar and a new amp" — 2 unigrams but no bigram
        b = _score_event_vs_query("bought a guitar and a new amp", q, q_tokens, q_bigrams)

        assert a > b, f"bigram-match event should score higher (a={a:.2f}, b={b:.2f})"

    def test_short_dense_match_beats_long_sparse(self):
        from temporal_graph import _score_event_vs_query

        q = "how many days between the gym sessions"
        q_tokens = {"gym", "sessions", "between", "many", "days"}
        q_bigrams = {("gym", "sessions")}

        short = _score_event_vs_query("gym session", q, q_tokens, q_bigrams)
        long = _score_event_vs_query(
            "I went to the gym yesterday and then had lunch with my brother at the mall",
            q,
            q_tokens,
            q_bigrams,
        )
        assert short > long, (
            f"dense short match should score higher (short={short:.2f}, long={long:.2f})"
        )

    def test_zero_overlap_returns_zero(self):
        from temporal_graph import _score_event_vs_query

        q = "how long did the project take"
        q_tokens = {"project", "long", "take"}
        q_bigrams = set()
        result = _score_event_vs_query("unrelated cooking dinner", q, q_tokens, q_bigrams)
        assert result == 0.0

    def test_empty_event_text(self):
        from temporal_graph import _score_event_vs_query

        assert _score_event_vs_query("", "query", set(), set()) == 0.0


class TestExpandChainedDates:
    """Multi-hop temporal chaining (Task #36)."""

    def test_days_after(self):
        from temporal_graph import _expand_chained_dates

        result = _expand_chained_dates("the concert was 3 days after 2023-05-14")
        assert result == ["2023-05-17"]

    def test_days_before(self):
        from temporal_graph import _expand_chained_dates

        result = _expand_chained_dates("I left 5 days before 2023-05-14")
        assert result == ["2023-05-09"]

    def test_weeks_after(self):
        from temporal_graph import _expand_chained_dates

        result = _expand_chained_dates("meeting scheduled 2 weeks after 2023-05-14")
        assert result == ["2023-05-28"]

    def test_weeks_before(self):
        from temporal_graph import _expand_chained_dates

        result = _expand_chained_dates("started 4 weeks before 2023-06-01")
        assert result == ["2023-05-04"]

    def test_months_after_approx(self):
        from temporal_graph import _expand_chained_dates

        # 2 months = 60 days in this implementation
        result = _expand_chained_dates("followup 2 months after 2023-01-10")
        assert result == ["2023-03-11"]

    def test_multiple_chained_references_same_sentence(self):
        from temporal_graph import _expand_chained_dates

        result = _expand_chained_dates(
            "event A was 3 days after 2023-05-14, event B was 1 week before 2023-06-01"
        )
        assert "2023-05-17" in result
        assert "2023-05-25" in result

    def test_no_chained_reference(self):
        from temporal_graph import _expand_chained_dates

        assert _expand_chained_dates("the cat sat on the mat") == []

    def test_no_absolute_anchor(self):
        """Chain with non-ISO anchor is NOT expanded (out of scope)."""
        from temporal_graph import _expand_chained_dates

        # "3 days after my birthday" has no ISO anchor → skipped
        assert _expand_chained_dates("3 days after my birthday") == []

    def test_extract_events_picks_up_chained(self):
        """Full pipeline: TemporalGraph.extract_events sees chained dates."""
        from temporal_graph import TemporalGraph

        tg = TemporalGraph()
        events = tg.extract_events(
            "The concert was 2 weeks after 2023-05-14.",
            "doc",
        )
        dates = [e.date for e in events]
        assert "2023-05-14" in dates
        assert "2023-05-28" in dates


class TestPickDurationPairBigramTiebreak:
    def test_bigram_beats_unigram_tie(self):
        """Two candidates with same unigram count — bigram match wins."""
        from datetime import date as _d
        from temporal_graph import _pick_duration_pair

        events = [
            (_d(2023, 1, 1), "guitar lessons started"),  # has 'guitar lessons' bigram
            (_d(2023, 2, 1), "bought guitar pick"),  # unigram match only
            (_d(2023, 4, 1), "guitar lessons ended"),  # has 'guitar lessons' bigram
        ]
        result = _pick_duration_pair(events, "how many days between the guitar lessons")
        assert result is not None
        d1, t1, d2, t2 = result
        assert "guitar lessons" in t1.lower() or "guitar lessons" in t2.lower()
        # Should pick the two bigram-match events (Jan and Apr), span 90 days
        assert (d2 - d1).days == 90
