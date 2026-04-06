# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal knowledge graph

"""Parse dates/events from documents, build temporal edges, query by time.

Addresses Remanentia's weakest LOCOMO category (temporal: 47.9%).
Zep's bitemporal model and Mem0g's graph variant both improve temporal
reasoning significantly via explicit temporal edges.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path

BASE = Path(__file__).parent
TEMPORAL_PATH = BASE / "memory" / "graph" / "temporal_edges.jsonl"

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_ENGLISH = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+(\d{1,2})(?:,?\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_DATE_RELATIVE = re.compile(
    r"\b(yesterday|today|last\s+(?:week|month|year)|this\s+(?:week|month|year))\b",
    re.IGNORECASE,
)


_rust_date_to_phase = None
_rust_resonance_search = None
try:
    from remanentia_temporal import (
        date_to_phase as _rust_dtp,
        resonance_search as _rust_rs,
    )

    _rust_date_to_phase = _rust_dtp  # pragma: no cover
    _rust_resonance_search = _rust_rs  # pragma: no cover
except ImportError:
    pass


def date_to_phase(date_str: str) -> float:
    """Map an ISO date string to a cyclic phase θ ∈ [0, 2π).

    θ(date) = 2π · day_of_year(date) / 365.25

    This is the UPDE (Universal Phase-Date Encoding) used across the
    SCPN ecosystem for cyclic temporal resonance scoring.
    Uses Rust (remanentia_temporal) when available.
    """
    if _rust_date_to_phase is not None:
        return _rust_date_to_phase(date_str)  # pragma: no cover
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return 0.0
    day_of_year = d.timetuple().tm_yday
    return 2.0 * math.pi * day_of_year / 365.25


@dataclass
class TemporalEvent:
    date: str  # ISO format YYYY-MM-DD
    text: str  # event description (sentence containing the date)
    source: str  # document name
    paragraph_idx: int = 0
    phase: float = field(default=0.0, repr=False)

    def calculate_phase(self) -> float:
        """Calculate and store the UPDE cyclic phase for this event's date."""
        self.phase = date_to_phase(self.date)
        return self.phase


@dataclass
class TemporalEdge:
    source_event: str
    target_event: str
    relation: str  # "before", "after", "same_day"
    source_date: str
    target_date: str


class TemporalGraph:
    def __init__(self):
        self.events: list[TemporalEvent] = []
        self.edges: list[TemporalEdge] = []
        self._by_date: dict[str, list[int]] = defaultdict(list)

    def resonance_search(
        self, query_date: str, tolerance: float = 0.01
    ) -> list[tuple[TemporalEvent, float]]:
        """Find events whose cyclic phase resonates with the query date.

        Resonance = cos(θ_event - θ_query). Events with resonance ≥ (1 - tolerance)
        are returned, sorted by resonance descending.
        Uses Rust (remanentia_temporal) when available.

        Args:
            query_date: ISO date string (YYYY-MM-DD).
            tolerance: Maximum allowed deviation from perfect resonance (1.0).
                       Default 0.01 ≈ ±5 days of year.

        Returns:
            List of (event, resonance_score) tuples, sorted descending.
        """
        if _rust_resonance_search is not None:  # pragma: no cover
            dates = [ev.date for ev in self.events]
            rust_results = _rust_resonance_search(dates, query_date, tolerance)
            return [(self.events[idx], score) for idx, score in rust_results]
        # Python fallback
        query_phase = date_to_phase(query_date)
        threshold = 1.0 - tolerance
        results: list[tuple[TemporalEvent, float]] = []
        for event in self.events:
            if event.phase == 0.0 and event.date:
                event.calculate_phase()
            resonance = math.cos(event.phase - query_phase)
            if resonance >= threshold:
                results.append((event, resonance))
        results.sort(key=lambda x: -x[1])
        return results

    def extract_events(self, text: str, doc_name: str) -> list[TemporalEvent]:
        """Extract dated events from a document."""
        events = []
        sentences = re.split(r"(?<=[.!?\n])\s+", text)

        for i, sent in enumerate(sentences):
            dates = parse_dates(sent)
            for d in dates:
                events.append(
                    TemporalEvent(
                        date=d,
                        text=sent.strip()[:200],
                        source=doc_name,
                        paragraph_idx=i,
                    )
                )
        return events

    def add_events(self, events: list[TemporalEvent]):
        """Add events and build temporal edges.

        Uses date-bucketed approach instead of O(N²) pairwise comparison:
        same_day edges within each date bucket, before/after edges only
        between adjacent date buckets.
        """
        start_idx = len(self.events)
        new_dates: set[str] = set()
        for ev in events:
            idx = len(self.events)
            self.events.append(ev)
            self._by_date[ev.date].append(idx)
            new_dates.add(ev.date)

        try:
            from remanentia_temporal import build_temporal_edges as _rust_bte

            by_date_snap = {d: list(idxs) for d, idxs in self._by_date.items()}  # pragma: no cover
            # Remove new events from snapshot (Rust re-adds them)
            for d in new_dates:  # pragma: no cover
                by_date_snap[d] = [i for i in by_date_snap[d] if i < start_idx]  # pragma: no cover
                if not by_date_snap[d]:  # pragma: no cover
                    del by_date_snap[d]  # pragma: no cover
            new_ev = [(ev.date, ev.text[:80]) for ev in events]  # pragma: no cover
            old_texts = {i: self.events[i].text[:80] for i in range(start_idx)}  # pragma: no cover
            raw_edges = _rust_bte(by_date_snap, new_ev, start_idx, old_texts)  # pragma: no cover
            for src_t, tgt_t, rel, src_d, tgt_d in raw_edges:  # pragma: no cover
                self.edges.append(  # pragma: no cover
                    TemporalEdge(
                        source_event=src_t,
                        target_event=tgt_t,
                        relation=rel,
                        source_date=src_d,
                        target_date=tgt_d,
                    )
                )
            return  # pragma: no cover
        except ImportError:
            pass

        # same_day edges: within each bucket that received new events
        for d in new_dates:
            bucket = self._by_date[d]
            new_in_bucket = [i for i in bucket if i >= start_idx]
            old_in_bucket = [i for i in bucket if i < start_idx]
            for ni in new_in_bucket:
                for oi in old_in_bucket:
                    self.edges.append(
                        TemporalEdge(
                            source_event=self.events[oi].text[:80],
                            target_event=self.events[ni].text[:80],
                            relation="same_day",
                            source_date=d,
                            target_date=d,
                        )
                    )
                # new-new pairs within same date
                for nj in new_in_bucket:
                    if nj > ni:
                        self.edges.append(
                            TemporalEdge(
                                source_event=self.events[ni].text[:80],
                                target_event=self.events[nj].text[:80],
                                relation="same_day",
                                source_date=d,
                                target_date=d,
                            )
                        )

        # before/after edges: only between adjacent dates (sorted)
        all_dates = sorted(self._by_date.keys())
        for idx_d, d in enumerate(all_dates):
            if d not in new_dates:
                continue
            if idx_d + 1 < len(all_dates):
                next_d = all_dates[idx_d + 1]
                new_here = [i for i in self._by_date[d] if i >= start_idx]
                for ni in new_here[:3]:  # cap edges per date pair
                    for nj in self._by_date[next_d][:3]:
                        self.edges.append(
                            TemporalEdge(
                                source_event=self.events[ni].text[:80],
                                target_event=self.events[nj].text[:80],
                                relation="before",
                                source_date=d,
                                target_date=next_d,
                            )
                        )

    def build_from_documents(self, documents: list[tuple[str, str]]):
        """Build graph from list of (doc_name, text) pairs."""
        for doc_name, text in documents:
            events = self.extract_events(text, doc_name)
            self.add_events(events)

    def query_temporal(self, query: str, top_k: int = 5) -> list[TemporalEvent]:
        """Answer temporal queries using the graph.

        Handles: "when did X happen", "what happened before/after X",
        "latest/first", "between date1 and date2".
        """
        dates_in_query = parse_dates(query)

        try:
            from remanentia_temporal import score_temporal_query as _rust_stq

            ev_tuples = [
                (e.date, e.text, e.source, e.paragraph_idx) for e in self.events
            ]  # pragma: no cover
            indices = _rust_stq(ev_tuples, query, dates_in_query, top_k)  # pragma: no cover
            return [self.events[i] for i in indices]  # pragma: no cover
        except ImportError:
            pass

        q = query.lower()
        events = self.events

        if dates_in_query:
            if "after" in q or "since" in q:
                events = [e for e in events if e.date >= dates_in_query[0]]
            elif "before" in q or "until" in q:
                events = [e for e in events if e.date <= dates_in_query[0]]
            elif len(dates_in_query) >= 2:
                d1, d2 = sorted(dates_in_query[:2])
                events = [e for e in events if d1 <= e.date <= d2]

        # Sort by relevance to query terms
        q_tokens = set(re.findall(r"[a-z0-9]{3,}", q))
        scored = []
        for ev in events:
            ev_tokens = set(re.findall(r"[a-z0-9]{3,}", ev.text.lower()))
            overlap = len(q_tokens & ev_tokens)
            if overlap > 0 or dates_in_query:
                scored.append((ev, overlap))

        # Sort: newest first for "latest/recent", oldest first for "first/earliest"
        if any(w in q for w in ["latest", "recent", "last", "newest"]):
            scored.sort(key=lambda x: (-len(x[0].date), -x[1]))
            scored.sort(key=lambda x: x[0].date, reverse=True)
        elif any(w in q for w in ["first", "earliest", "oldest", "original"]):
            scored.sort(key=lambda x: (x[0].date, -x[1]))
        else:
            scored.sort(key=lambda x: -x[1])

        return [ev for ev, _ in scored[:top_k]]

    def events_on_date(self, date_str: str) -> list[TemporalEvent]:
        """Get all events on a specific date."""
        indices = self._by_date.get(date_str, [])
        return [self.events[i] for i in indices]

    def events_between(self, start: str, end: str) -> list[TemporalEvent]:
        """Get events in a date range."""
        return sorted(
            [e for e in self.events if start <= e.date <= end],
            key=lambda e: e.date,
        )

    def save(self, path: Path | None = None):
        path = path or TEMPORAL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for ev in self.events:
            lines.append(
                json.dumps(
                    {
                        "date": ev.date,
                        "text": ev.text,
                        "source": ev.source,
                        "paragraph_idx": ev.paragraph_idx,
                    }
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def load(self, path: Path | None = None) -> bool:
        path = path or TEMPORAL_PATH
        if not path.exists():
            return False
        self.events = []
        self._by_date = defaultdict(list)
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():  # pragma: no cover
                continue
            d = json.loads(line)
            idx = len(self.events)
            ev = TemporalEvent(
                date=d["date"],
                text=d["text"],
                source=d["source"],
                paragraph_idx=d.get("paragraph_idx", 0),
            )
            self.events.append(ev)
            self._by_date[ev.date].append(idx)
        return True

    @property
    def stats(self) -> dict:
        dates = sorted(set(e.date for e in self.events))
        return {
            "events": len(self.events),
            "edges": len(self.edges),
            "unique_dates": len(dates),
            "date_range": f"{dates[0]} to {dates[-1]}" if dates else "",
            "sources": len(set(e.source for e in self.events)),
        }


def temporal_code_execute(query: str, events: list[TemporalEvent]) -> str | None:
    """Execute temporal arithmetic via Python code instead of LLM guessing.

    TReMu technique (ACL 2025): +6pp on temporal questions by replacing
    natural language temporal reasoning with date arithmetic.

    Handles:
    - "How long between X and Y?"
    - "What happened N days/weeks/months before/after X?"
    - "Did X happen before or after Y?"
    - "What was the most recent X?"
    - "How many days since X?"
    """
    q = query.lower()
    event_dates = [(e.date, e.text) for e in events if e.date]
    if not event_dates:
        return None

    parsed = [
        (datetime.strptime(d, "%Y-%m-%d").date(), t)
        for d, t in event_dates
        if re.match(r"\d{4}-\d{2}-\d{2}$", d)
    ]
    if not parsed:
        return None

    # "How long between" / "how many days"
    if re.search(r"how (long|many days|many weeks|many months)", q):
        if len(parsed) >= 2:
            dates_sorted = sorted(parsed, key=lambda x: x[0])
            delta = (dates_sorted[-1][0] - dates_sorted[0][0]).days
            return f"{delta} days (from {dates_sorted[0][0].isoformat()} to {dates_sorted[-1][0].isoformat()})"

    # "before or after" comparison
    if "before" in q and "after" in q and len(parsed) >= 2:
        d1, d2 = parsed[0][0], parsed[1][0]
        if d1 < d2:
            return f"{parsed[0][1][:60]} happened before {parsed[1][1][:60]} ({(d2 - d1).days} days earlier)"
        elif d1 > d2:
            return f"{parsed[0][1][:60]} happened after {parsed[1][1][:60]} ({(d1 - d2).days} days later)"
        return f"Both happened on the same day: {d1.isoformat()}"

    # "most recent" / "latest"
    if any(w in q for w in ["most recent", "latest", "last", "newest"]):
        latest = max(parsed, key=lambda x: x[0])
        return f"{latest[1][:100]} ({latest[0].isoformat()})"

    # "first" / "earliest"
    if any(w in q for w in ["first", "earliest", "oldest"]):
        earliest = min(parsed, key=lambda x: x[0])
        return f"{earliest[1][:100]} ({earliest[0].isoformat()})"

    # "how many days since" / "how long ago"
    if re.search(r"(since|ago|how long)", q) and parsed:
        ref = max(parsed, key=lambda x: x[0])
        delta = (date.today() - ref[0]).days
        return f"{delta} days since {ref[1][:60]} ({ref[0].isoformat()})"

    return None


def parse_dates(text: str, reference_date: date | None = None) -> list[str]:
    """Extract all dates from text, return as ISO strings.

    Resolves relative expressions (yesterday, last week) against reference_date
    (defaults to today). Uses Rust engine when available (~14× faster).
    """
    ref = reference_date or date.today()

    # Try Rust engine first
    try:
        from remanentia_temporal import parse_dates as _rust_parse

        return _rust_parse(text, ref.isoformat())  # pragma: no cover
    except ImportError:
        pass

    # Python fallback
    dates = []

    for m in _DATE_ISO.finditer(text):
        dates.append(m.group(0))

    for m in _DATE_ENGLISH.finditer(text):
        month_name = m.group(1).lower()
        month = _MONTHS.get(month_name)
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else ref.year
        if month and 1 <= day <= 31:
            try:
                d = date(year, month, day)
                dates.append(d.isoformat())
            except ValueError:  # pragma: no cover
                pass

    for m in _DATE_RELATIVE.finditer(text):
        expr = m.group(1).lower().strip()
        resolved = _resolve_relative_date(expr, ref)
        if resolved:
            dates.append(resolved)

    # ML-augmented vague date normalisation (C4)
    try:
        from date_normalizer import VAGUE_DATE_RE, normalize_date_expression

        for m in VAGUE_DATE_RE.finditer(text):
            expr_text = m.group(0)
            result = normalize_date_expression(expr_text, ref)
            if result is not None and result.confidence > 0.7:
                dates.append(result.iso_date)
    except ImportError:
        pass

    return sorted(set(dates))


def _resolve_relative_date(expr: str, ref: date) -> str | None:
    """Resolve a relative date expression to ISO string."""
    if expr == "yesterday":
        return (ref - timedelta(days=1)).isoformat()
    if expr == "today":
        return ref.isoformat()
    if expr == "last week":
        return (ref - timedelta(weeks=1)).isoformat()
    if expr == "last month":
        m = ref.month - 1 or 12
        y = ref.year if ref.month > 1 else ref.year - 1
        return date(y, m, min(ref.day, 28)).isoformat()
    if expr == "last year":
        return date(ref.year - 1, ref.month, min(ref.day, 28)).isoformat()
    if expr == "this week":
        return (ref - timedelta(days=ref.weekday())).isoformat()
    if expr == "this month":
        return date(ref.year, ref.month, 1).isoformat()
    if expr == "this year":
        return date(ref.year, 1, 1).isoformat()
    return None
