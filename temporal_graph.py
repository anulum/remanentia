# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal Knowledge Graph

"""Parse dates/events from documents, build temporal edges, query by time.

Addresses Remanentia's weakest LOCOMO category (temporal: 47.9%).
Zep's bitemporal model and Mem0g's graph variant both improve temporal
reasoning significantly via explicit temporal edges.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path

BASE = Path(__file__).parent
TEMPORAL_PATH = BASE / "memory" / "graph" / "temporal_edges.jsonl"

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
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


@dataclass
class TemporalEvent:
    date: str  # ISO format YYYY-MM-DD
    text: str  # event description (sentence containing the date)
    source: str  # document name
    paragraph_idx: int = 0


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

    def extract_events(self, text: str, doc_name: str) -> list[TemporalEvent]:
        """Extract dated events from a document."""
        events = []
        sentences = re.split(r"(?<=[.!?\n])\s+", text)

        for i, sent in enumerate(sentences):
            dates = parse_dates(sent)
            for d in dates:
                events.append(TemporalEvent(
                    date=d, text=sent.strip()[:200], source=doc_name, paragraph_idx=i,
                ))
        return events

    def add_events(self, events: list[TemporalEvent]):
        """Add events and build temporal edges."""
        start_idx = len(self.events)
        for ev in events:
            idx = len(self.events)
            self.events.append(ev)
            self._by_date[ev.date].append(idx)

        # Build edges between new events and existing events
        for i in range(start_idx, len(self.events)):
            ev_i = self.events[i]
            for j in range(i):
                ev_j = self.events[j]
                if ev_i.date == ev_j.date:
                    self.edges.append(TemporalEdge(
                        source_event=ev_j.text[:80],
                        target_event=ev_i.text[:80],
                        relation="same_day",
                        source_date=ev_j.date,
                        target_date=ev_i.date,
                    ))
                elif ev_i.date < ev_j.date:
                    self.edges.append(TemporalEdge(
                        source_event=ev_i.text[:80],
                        target_event=ev_j.text[:80],
                        relation="before",
                        source_date=ev_i.date,
                        target_date=ev_j.date,
                    ))

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
        q = query.lower()
        events = self.events

        # Date range query: "between X and Y" or "after X" or "before Y"
        dates_in_query = parse_dates(query)

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
            lines.append(json.dumps({
                "date": ev.date, "text": ev.text,
                "source": ev.source, "paragraph_idx": ev.paragraph_idx,
            }))
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
                date=d["date"], text=d["text"],
                source=d["source"], paragraph_idx=d.get("paragraph_idx", 0),
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


def parse_dates(text: str) -> list[str]:
    """Extract all dates from text, return as ISO strings."""
    dates = []

    for m in _DATE_ISO.finditer(text):
        dates.append(m.group(0))

    for m in _DATE_ENGLISH.finditer(text):
        month_name = m.group(1).lower()
        month = _MONTHS.get(month_name)
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else date.today().year
        if month and 1 <= day <= 31:
            try:
                d = date(year, month, day)
                dates.append(d.isoformat())
            except ValueError:  # pragma: no cover
                pass

    return sorted(set(dates))
