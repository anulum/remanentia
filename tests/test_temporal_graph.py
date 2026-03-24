# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Tests for temporal_graph.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        tg.add_events([
            TemporalEvent(date="2026-03-15", text="STDP bug was fixed", source="a.md"),
            TemporalEvent(date="2026-03-20", text="Daemon was killed", source="b.md"),
            TemporalEvent(date="2026-03-23", text="LOCOMO benchmark run", source="c.md"),
        ])
        results = tg.query_temporal("when was the STDP bug fixed", top_k=3)
        assert len(results) > 0
        assert any("STDP" in ev.text for ev in results)

    def test_query_temporal_after(self):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-10", text="Event A happened", source="a.md"),
            TemporalEvent(date="2026-03-20", text="Event B happened", source="b.md"),
        ])
        results = tg.query_temporal("what happened after 2026-03-15", top_k=5)
        assert all(ev.date >= "2026-03-15" for ev in results)

    def test_query_temporal_before(self):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-10", text="Event A happened", source="a.md"),
            TemporalEvent(date="2026-03-20", text="Event B happened", source="b.md"),
        ])
        results = tg.query_temporal("what happened before 2026-03-15", top_k=5)
        assert all(ev.date <= "2026-03-15" for ev in results)

    def test_query_latest(self):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-10", text="Old event with data", source="a.md"),
            TemporalEvent(date="2026-03-23", text="Latest event with data", source="b.md"),
        ])
        results = tg.query_temporal("latest event with data", top_k=1)
        assert results[0].date == "2026-03-23"

    def test_query_first(self):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-10", text="First experiment run", source="a.md"),
            TemporalEvent(date="2026-03-23", text="Second experiment run", source="b.md"),
        ])
        results = tg.query_temporal("first experiment", top_k=1)
        assert results[0].date == "2026-03-10"

    def test_events_on_date(self):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-15", text="Event A", source="a.md"),
            TemporalEvent(date="2026-03-15", text="Event B", source="b.md"),
            TemporalEvent(date="2026-03-20", text="Event C", source="c.md"),
        ])
        events = tg.events_on_date("2026-03-15")
        assert len(events) == 2

    def test_events_between(self):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-10", text="A", source="a.md"),
            TemporalEvent(date="2026-03-15", text="B", source="b.md"),
            TemporalEvent(date="2026-03-20", text="C", source="c.md"),
        ])
        events = tg.events_between("2026-03-12", "2026-03-18")
        assert len(events) == 1
        assert events[0].date == "2026-03-15"

    def test_save_and_load(self, tmp_path):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-15", text="Event A", source="a.md"),
            TemporalEvent(date="2026-03-20", text="Event B", source="b.md"),
        ])
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
        tg.add_events([
            TemporalEvent(date="2026-03-15", text="A", source="a.md"),
            TemporalEvent(date="2026-03-20", text="B", source="b.md"),
        ])
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
        tg.add_events([
            TemporalEvent(date="2026-03-20", text="Later event", source="b.md"),
        ])
        tg.add_events([
            TemporalEvent(date="2026-03-10", text="Earlier event", source="a.md"),
        ])
        before = [e for e in tg.edges if e.relation == "before"]
        assert len(before) >= 1
        assert before[0].source_date == "2026-03-10"

    def test_date_range_query(self):
        tg = TemporalGraph()
        tg.add_events([
            TemporalEvent(date="2026-03-05", text="Event A with data", source="a.md"),
            TemporalEvent(date="2026-03-15", text="Event B with data", source="b.md"),
            TemporalEvent(date="2026-03-25", text="Event C with data", source="c.md"),
        ])
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
        # "how many days" with only 1 event — no pair to compare
        result = temporal_code_execute("how many days between events", events)
        # Should still return None or a sensible result
        # With only 1 dated event, can't compute a range
        assert result is None or "0 days" in result
