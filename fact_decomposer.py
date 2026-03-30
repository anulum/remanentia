# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Atomic fact decomposer with temporal validity

"""Decompose conversation turns into atomic facts with validity windows.

Addresses the -45.9pp temporal-reasoning gap (45.1% vs Hindsight 91.0%).
Inspired by EverMemOS MemCell (Foresight signals with [t_start, t_end])
and Hindsight's bitemporal model (fact validity intervals [τs, τe]).

Each conversation turn is split into atomic facts, each annotated with:
- valid_from / valid_until: when the fact is/was true
- entities: mentioned entity names
- fact_type: state / event / preference / plan
- session_idx: which session the fact came from

Usage::

    from fact_decomposer import decompose_sessions, FactIndex
    facts = decompose_sessions(haystack_sessions)
    fidx = FactIndex(facts)
    results = fidx.query("current job", reference_date="2024-03-15")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# Date parsing patterns
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_MDY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_DATE_WRITTEN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:[,\s]+(\d{4}))?\b",
    re.IGNORECASE,
)
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

# State-change verbs that indicate a fact supersedes a prior one
_CHANGE_VERBS = re.compile(
    r"\b(started|began|switched|changed|moved|left|quit|joined|"
    r"got|bought|sold|upgraded|downgraded|replaced|updated|"
    r"decided|chose|picked|adopted|dropped|stopped|finished|"
    r"married|divorced|graduated|retired|hired|fired)\b",
    re.IGNORECASE,
)

# Preference indicators
_PREFERENCE_PATTERNS = re.compile(
    r"\b(I (?:like|love|prefer|enjoy|hate|dislike|want|need|always|never|usually|"
    r"favorite|favourite)|my (?:favorite|favourite|go-to|preferred))\b",
    re.IGNORECASE,
)

# Plan/future indicators
_PLAN_PATTERNS = re.compile(
    r"\b(I (?:plan to|will|am going to|intend to|might|may|hope to|want to|'m planning)|"
    r"going to|planning to|scheduled for|booked for|appointment|reservation)\b",
    re.IGNORECASE,
)

# Sentence splitter (handles abbreviations reasonably)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class AtomicFact:
    """A single atomic fact extracted from a conversation turn."""

    text: str
    session_idx: int
    turn_idx: int
    role: str  # "user" or "assistant"
    fact_type: str  # "state", "event", "preference", "plan"
    valid_from: str = ""  # ISO date or ""
    valid_until: str = ""  # ISO date or "" (still valid)
    entities: list[str] = field(default_factory=list)
    date_mentions: list[str] = field(default_factory=list)
    supersedes: bool = False  # True if this fact updates a prior one


@dataclass
class FactIndex:
    """Index over atomic facts with temporal filtering and keyword search."""

    facts: list[AtomicFact]
    _entity_to_facts: dict[str, list[int]] = field(default_factory=dict)
    _keyword_to_facts: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self):
        self._entity_to_facts = {}
        self._keyword_to_facts = {}
        for i, fact in enumerate(self.facts):
            for ent in fact.entities:
                key = ent.lower()
                self._entity_to_facts.setdefault(key, []).append(i)
            for word in _tokenize(fact.text):
                self._keyword_to_facts.setdefault(word, []).append(i)

    def query(
        self,
        question: str,
        reference_date: str = "",
        top_k: int = 20,
        filter_expired: bool = True,
    ) -> list[tuple[AtomicFact, float]]:
        """Retrieve facts matching the question, filtered by temporal validity.

        Returns (fact, score) pairs sorted by descending score.
        """
        q_tokens = _tokenize(question)
        q_entities = _extract_entities_simple(question)

        scores: dict[int, float] = {}

        # Keyword matching (BM25-like TF scoring)
        for token in q_tokens:
            for fact_idx in self._keyword_to_facts.get(token, []):
                scores[fact_idx] = scores.get(fact_idx, 0) + 1.0

        # Entity matching (boosted)
        for ent in q_entities:
            key = ent.lower()
            for fact_idx in self._entity_to_facts.get(key, []):
                scores[fact_idx] = scores.get(fact_idx, 0) + 3.0

        # Temporal filtering
        if filter_expired and reference_date:
            ref = _parse_date_str(reference_date)
            if ref:
                expired = set()
                for idx in scores:
                    fact = self.facts[idx]
                    if fact.valid_until:
                        until = _parse_date_str(fact.valid_until)
                        if until and until < ref:
                            expired.add(idx)
                for idx in expired:
                    del scores[idx]

        # Recency boost: later sessions score higher for knowledge-update
        is_update_query = any(
            w in question.lower()
            for w in ("current", "now", "latest", "most recent", "today", "right now")
        )
        if is_update_query:
            for idx in scores:
                fact = self.facts[idx]
                scores[idx] += fact.session_idx * 2.0
                if fact.supersedes:
                    scores[idx] += 5.0

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.facts[idx], score) for idx, score in ranked]

    def temporal_query(
        self,
        question: str,
        top_k: int = 20,
    ) -> list[tuple[AtomicFact, float]]:
        """Specialised query for temporal-reasoning questions.

        Boosts facts with date mentions. Returns all dated facts if
        the question asks about ordering or counting.
        """
        q_tokens = _tokenize(question)
        q_entities = _extract_entities_simple(question)

        scores: dict[int, float] = {}

        # Keyword matching
        for token in q_tokens:
            for fact_idx in self._keyword_to_facts.get(token, []):
                scores[fact_idx] = scores.get(fact_idx, 0) + 1.0

        # Entity matching
        for ent in q_entities:
            key = ent.lower()
            for fact_idx in self._entity_to_facts.get(key, []):
                scores[fact_idx] = scores.get(fact_idx, 0) + 3.0

        # Boost facts with date mentions
        for idx in list(scores.keys()):
            fact = self.facts[idx]
            if fact.date_mentions:
                scores[idx] += 5.0
            if fact.valid_from:
                scores[idx] += 2.0

        # For ordering questions, include ALL dated facts even without keyword match
        is_ordering = any(
            w in question.lower()
            for w in (
                "first",
                "last",
                "before",
                "after",
                "earlier",
                "later",
                "how many days",
                "how long",
                "when did",
                "what date",
                "which came first",
                "chronological",
                "order",
                "timeline",
            )
        )
        if is_ordering:
            for i, fact in enumerate(self.facts):
                if fact.date_mentions and i not in scores:
                    scores[i] = 2.0

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.facts[idx], score) for idx, score in ranked]

    def cross_session_query(
        self,
        question: str,
        top_k: int = 20,
    ) -> list[tuple[AtomicFact, float]]:
        """Specialised query for multi-session questions.

        Ensures results span multiple sessions.
        """
        q_tokens = _tokenize(question)
        q_entities = _extract_entities_simple(question)

        scores: dict[int, float] = {}

        for token in q_tokens:
            for fact_idx in self._keyword_to_facts.get(token, []):
                scores[fact_idx] = scores.get(fact_idx, 0) + 1.0

        for ent in q_entities:
            key = ent.lower()
            for fact_idx in self._entity_to_facts.get(key, []):
                scores[fact_idx] = scores.get(fact_idx, 0) + 3.0

        # Ensure session diversity in top results
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        result = []
        sessions_seen: set[int] = set()
        for idx, score in ranked:
            fact = self.facts[idx]
            # Boost facts from underrepresented sessions
            diversity_bonus = 3.0 if fact.session_idx not in sessions_seen else 0.0
            result.append((fact, score + diversity_bonus))
            sessions_seen.add(fact.session_idx)
            if len(result) >= top_k:
                break

        result.sort(key=lambda x: -x[1])
        return result


def decompose_sessions(
    sessions: list[list[dict]],
    default_year: int = 2024,
    session_dates: list[str] | None = None,
) -> list[AtomicFact]:
    """Decompose all sessions into atomic facts with validity windows.

    Args:
        sessions: List of sessions, each a list of ``{"role", "content"}`` dicts.
        default_year: Fallback year for date parsing when no year is mentioned.
        session_dates: Optional per-session ISO date strings (from
            ``haystack_dates``). When provided, vague date expressions like
            "3 weeks ago" are resolved against the session's timestamp via
            the C4 date normaliser.
    """
    all_facts: list[AtomicFact] = []
    entity_last_state: dict[str, int] = {}  # entity → index of latest state fact

    for sess_idx, session in enumerate(sessions):
        # Resolve per-session reference date for C4 date normaliser
        _ref_date = None
        if session_dates and sess_idx < len(session_dates):
            try:
                from datetime import date as _dt_date

                parts = session_dates[sess_idx].replace("/", "-").split("-")
                if len(parts) >= 3:
                    _ref_date = _dt_date(int(parts[0]), int(parts[1]), int(parts[2][:2]))
            except (ValueError, IndexError):
                pass

        for turn_idx, turn in enumerate(session):
            content = turn["content"]
            role = turn["role"]

            if len(content) < 15:
                continue

            sentences = _split_sentences(content)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10:  # pragma: no cover — _split_sentences pre-filters
                    continue

                fact = _build_fact(
                    sent,
                    sess_idx,
                    turn_idx,
                    role,
                    default_year,
                    reference_date=_ref_date,
                )
                fact_idx = len(all_facts)

                # Detect if this fact supersedes a prior state for the same entity
                if fact.fact_type == "state" and fact.supersedes:
                    for ent in fact.entities:
                        key = ent.lower()
                        if key in entity_last_state:
                            prev_idx = entity_last_state[key]
                            if prev_idx < len(all_facts):
                                prev = all_facts[prev_idx]
                                if not prev.valid_until and fact.valid_from:
                                    prev.valid_until = fact.valid_from
                                elif not prev.valid_until:
                                    prev.valid_until = f"before-session-{sess_idx}"
                        entity_last_state[key] = fact_idx
                elif (
                    fact.fact_type == "state"
                ):  # pragma: no cover — state implies change verb implies supersedes
                    for ent in fact.entities:
                        key = ent.lower()
                        entity_last_state[key] = fact_idx

                all_facts.append(fact)

    return all_facts


def _build_fact(
    sentence: str,
    sess_idx: int,
    turn_idx: int,
    role: str,
    default_year: int,
    reference_date: "date | None" = None,
) -> AtomicFact:
    """Build a single AtomicFact from a sentence.

    When *reference_date* is provided, vague expressions ("3 weeks ago")
    are resolved via the C4 date normaliser.
    """
    if reference_date is not None:
        dates = _extract_dates_with_normaliser(sentence, reference_date, default_year)
    else:
        dates = _extract_dates(sentence, default_year)
    entities = _extract_entities_simple(sentence)

    fact_type = _classify_fact(sentence)
    supersedes = bool(_CHANGE_VERBS.search(sentence))

    # C5: ML-based fact classification — only override when regex gives
    # the catch-all "event" type (no explicit pattern match)
    if fact_type == "event" and not supersedes:
        try:
            from fact_validity_model import classify_fact as _ml_classify

            prediction = _ml_classify(sentence)
            if prediction is not None and prediction.confidence > 0.7:  # pragma: no cover
                fact_type = prediction.fact_type  # pragma: no cover
                if prediction.supersedes_prob > 0.5:  # pragma: no cover
                    supersedes = True  # pragma: no cover
        except ImportError:
            pass

    valid_from = dates[0] if dates else ""

    return AtomicFact(
        text=sentence,
        session_idx=sess_idx,
        turn_idx=turn_idx,
        role=role,
        fact_type=fact_type,
        valid_from=valid_from,
        entities=entities,
        date_mentions=dates,
        supersedes=supersedes,
    )


def _classify_fact(sentence: str) -> str:
    """Classify a sentence into fact type."""
    if _PLAN_PATTERNS.search(sentence):
        return "plan"
    if _PREFERENCE_PATTERNS.search(sentence):
        return "preference"
    if _CHANGE_VERBS.search(sentence):
        return "state"  # state change = new state
    return "event"


def _extract_dates(text: str, default_year: int = 2024) -> list[str]:
    """Extract ISO dates from text."""
    results = []

    for m in _DATE_ISO.finditer(text):
        results.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    for m in _DATE_WRITTEN.finditer(text):
        month_name = m.group(1).lower().rstrip(".")
        month = _MONTHS.get(month_name)
        if not month:  # pragma: no cover — regex only matches known month names
            continue
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else default_year
        if 1 <= day <= 31:
            results.append(f"{year}-{month:02d}-{day:02d}")

    for m in _DATE_MDY.finditer(text):
        month = int(m.group(1))
        day = int(m.group(2))
        year_str = m.group(3)
        year = int(year_str)
        if year < 100:
            year += 2000
        if 1 <= month <= 12 and 1 <= day <= 31:
            results.append(f"{year}-{month:02d}-{day:02d}")

    return results


def _extract_dates_with_normaliser(
    text: str,
    reference_date: "date | None" = None,
    default_year: int = 2024,
) -> list[str]:
    """Extract dates with ML-augmented vague expression normalisation (C4).

    Only call this when a concrete reference_date is available (e.g., from
    haystack_dates in LongMemEval), not for arbitrary text.
    """
    results = _extract_dates(text, default_year)
    if not results and reference_date is not None:
        try:
            from date_normalizer import extract_and_normalise

            for r in extract_and_normalise(text, reference_date):
                if r.confidence > 0.7:
                    results.append(r.iso_date)
        except ImportError:
            pass
    return results


def _extract_entities_simple(text: str) -> list[str]:
    """Extract likely entity names via capitalisation heuristic.

    Not as good as GLiNER but zero-dependency and fast.
    """
    entities = []
    # Capitalised multi-word sequences (proper nouns)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        name = m.group(1)
        # Filter sentence starters (first word after ". " or start)
        pos = m.start()
        if pos == 0:
            continue
        if pos >= 2 and text[pos - 2 : pos] in (". ", "! ", "? "):
            continue
        if len(name) > 1:
            entities.append(name)

    # Quoted strings (often names/titles)
    for m in re.finditer(r'"([^"]{2,40})"', text):
        entities.append(m.group(1))

    return entities


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = _SENT_SPLIT.split(text)
    result = []
    for part in parts:
        part = part.strip()
        if len(part) >= 10:
            result.append(part)
    if not result and text.strip():
        result = [text.strip()]
    return result


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase words (3+ chars)."""
    return set(re.findall(r"\w{3,}", text.lower()))


def _parse_date_str(s: str) -> Optional[date]:
    """Parse ISO date string to date object."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None
