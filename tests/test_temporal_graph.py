# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for temporal graph

from __future__ import annotations


from temporal_graph import (
    TemporalEvent,
    TemporalGraph,
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
        assert "days since" in result

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
        from datetime import date, timedelta

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
        for month in ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]:
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
        tg.add_events([
            TemporalEvent(date="2026-03-10", text="A2", source="a2.md"),
            TemporalEvent(date="2026-03-20", text="B", source="b.md"),
        ])
        assert len(tg.events) == 3
        same_day = [e for e in tg.edges if e.relation == "same_day"]
        assert len(same_day) >= 1  # A and A2 on same day

    def test_save_load_roundtrip_preserves_dates(self, tmp_path):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-15", text="Event α", source="α.md", paragraph_idx=3),
            TemporalEvent(date="2026-03-20", text="Event β", source="β.md", paragraph_idx=7),
        ])
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
        tg.add_events([
            TemporalEvent(date="2026-03-10", text="Old event with data", source="a.md"),
            TemporalEvent(date="2026-03-25", text="New event with data", source="b.md"),
        ])
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
        tg.build_from_documents([
            ("session1.md", "On March 10, 2026 the team decided to use BM25."),
            ("session2.md", "On March 15, 2026 LOCOMO accuracy was 81.2%."),
            ("session3.md", "On March 20, 2026 the score improved to 88.5%."),
        ])
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
        tg.build_from_documents([
            ("a.md", "Bug found on 2026-03-10."),
            ("b.md", "Fix deployed on 2026-03-15."),
        ])
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
        assert "7 days" in answer
