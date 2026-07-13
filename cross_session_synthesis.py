# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Cross-session entity-summary synthesis

"""Consolidate cross-session facts about a question's entities for the reader.

The deterministic aggregation pre-compute (:mod:`aggregate_precompute`) only
fires for SUM and COUNT questions. The larger multi-session failure class on
the honest full-S benchmark is *synthesis*, not arithmetic: recall@10 is high
(multi-session 88 %, knowledge-update gold usually retrieved) yet accuracy is
~42 %, because the reader now sees ~10 sessions and must combine scattered,
sometimes-superseded statements about the same entity itself
(``benchmark_2026-06-21_p11_failure_attribution.md`` §4, §5). This module is
the reflection / entity-summary lever for that class (TODO P1.3).

Given the already-retrieved atomic facts, it builds a compact, deduplicated,
chronologically-ordered ``ENTITY SUMMARY`` block for the entities that the
question actually names. Each entity's statements are grouped, deduped, ordered
oldest→newest, and the latest dated statement is marked ``(most recent)`` — the
exact signal the knowledge-update reader needs ("use the most recent answer").

The design mirrors :mod:`aggregate_precompute`: deliberately conservative,
high-precision, no extra LLM call (it operates only over facts the retriever
already surfaced, so it adds no per-question API cost and is fully
reproducible). It returns ``None`` whenever the question names no retrieved
entity or no entity has enough cross-session evidence to be worth consolidating
— over-firing would add reader-distracting noise, which is the very failure it
targets.

Pure Python only. There is no ``remanentia_cross_session_synthesis`` crate in
this repository (BACKLOG L1.3): earlier builds advertised an optional Rust
import that could never load. Tokenisation and entity matching stay in this
module so the public surface does not claim an accelerator that does not exist.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ─── Fact shape ───────────────────────────────────────────────────────


@runtime_checkable
class SynthFact(Protocol):
    """Structural shape of an atomic fact consumed by the synthesiser.

    Matches :class:`fact_decomposer.AtomicFact`; declared structurally so this
    module does not import the retriever (and cannot create an import cycle).
    """

    text: str
    session_idx: int
    role: str
    fact_type: str
    valid_from: str
    entities: list[str]
    date_mentions: list[str]
    session_date: str


# ─── Question-grounded entity selection ───────────────────────────────


_MIN_TOKEN_LEN = 4
# Generic words that should not, on their own, bind a fact entity to a
# question. Without this filter "the project budget" would match any entity
# containing "project"/"budget", over-firing on unrelated facts.
_STOP = frozenset(
    {
        "about",
        "after",
        "again",
        "another",
        "around",
        "before",
        "being",
        "between",
        "could",
        "current",
        "currently",
        "different",
        "distinct",
        "does",
        "doing",
        "during",
        "first",
        "follow",
        "have",
        "having",
        "into",
        "just",
        "last",
        "later",
        "latest",
        "many",
        "more",
        "most",
        "much",
        "number",
        "other",
        "over",
        "recent",
        "recently",
        "same",
        "some",
        "still",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "thing",
        "things",
        "this",
        "those",
        "time",
        "times",
        "total",
        "unique",
        "until",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
    }
)

_TOKEN = re.compile(r"[a-z0-9]+")
_WS = re.compile(r"\s+")


def _significant_tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens long enough to bind an entity."""
    return {
        token
        for token in _TOKEN.findall(text.lower())
        if len(token) >= _MIN_TOKEN_LEN and token not in _STOP
    }


def _dedupe_entities(entities: Iterable[str]) -> list[str]:
    """Return entities with case-insensitive duplicates removed, order kept."""
    seen: set[str] = set()
    out: list[str] = []
    for entity in entities:
        cleaned = _WS.sub(" ", entity.strip())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def focus_entities(question: str, entities: Iterable[str]) -> list[str]:
    """Return the retrieved *entities* the *question* explicitly grounds.

    An entity qualifies when at least one of its significant tokens appears in
    the question. This keeps the synthesis tied to what was asked rather than
    summarising every entity the retriever happened to surface.
    """
    deduped = _dedupe_entities(entities)
    q_tokens = _significant_tokens(question)
    if not q_tokens:
        return []
    return [entity for entity in deduped if _significant_tokens(entity) & q_tokens]


# ─── Statement extraction and ordering ────────────────────────────────


def _statement_date(fact: SynthFact) -> str:
    """Return the best-effort ISO date for *fact* (explicit mention wins)."""
    if fact.date_mentions:
        return fact.date_mentions[0]
    if fact.valid_from:
        return fact.valid_from
    return fact.session_date


def _normalise_statement(text: str) -> str:
    """Return a dedup key collapsing whitespace, case, and edge punctuation."""
    return _WS.sub(" ", text.strip().casefold()).strip(" .,:;-")


@dataclass
class StatementLine:
    """One consolidated statement about an entity, with provenance."""

    date: str
    session_idx: int
    text: str

    def _sort_key(self) -> tuple[int, str, int]:
        """Order dated statements oldest→newest, undated last by session."""
        if self.date:
            return (0, self.date, self.session_idx)
        return (1, "", self.session_idx)

    def render(self, most_recent: bool) -> str:
        """Return the prompt line for this statement."""
        when = self.date if self.date else f"session {self.session_idx + 1}"
        tag = "  (most recent)" if most_recent else ""
        return f"  - [{when}] {self.text}{tag}"


@dataclass
class EntityDigest:
    """All consolidated statements about a single focus entity."""

    entity: str
    statements: list[StatementLine] = field(default_factory=list)

    def most_recent_index(self) -> int:
        """Return the index of the latest dated statement, or ``-1``."""
        latest = -1
        for i, statement in enumerate(self.statements):
            if statement.date:
                latest = i
        return latest

    def render(self) -> str:
        """Return the entity block: heading plus one line per statement."""
        recent = self.most_recent_index()
        lines = [f"• {self.entity}:"]
        for i, statement in enumerate(self.statements):
            lines.append(statement.render(most_recent=(i == recent and len(self.statements) > 1)))
        return "\n".join(lines)


def _digest_entity(entity: str, facts: Sequence[SynthFact], max_statements: int) -> EntityDigest:
    """Collect, dedupe, and order the statements mentioning *entity*."""
    key = entity.casefold()
    seen: set[str] = set()
    collected: list[StatementLine] = []
    for fact in facts:
        if not any(e.casefold() == key for e in fact.entities):
            continue
        text = _WS.sub(" ", fact.text.strip())
        norm = _normalise_statement(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        collected.append(
            StatementLine(date=_statement_date(fact), session_idx=fact.session_idx, text=text)
        )
    collected.sort(key=lambda s: s._sort_key())
    return EntityDigest(entity=entity, statements=collected[:max_statements])


# ─── Public entry point ───────────────────────────────────────────────


@dataclass
class SynthesisResult:
    """Outcome of one cross-session synthesis attempt."""

    entities: list[EntityDigest]
    message: str  # the rendered "ENTITY SUMMARY ..." block

    def __bool__(self) -> bool:
        return bool(self.entities)


_HEADER = "ENTITY SUMMARY (consolidated cross-session facts — deduplicated, oldest→newest):"

# An entity is only worth consolidating when at least two distinct statements
# mention it; a single statement carries no cross-session synthesis value.
_MIN_STATEMENTS = 2


def synthesise(
    question: str,
    facts: Sequence[SynthFact],
    *,
    qtype: str = "",
    max_entities: int = 4,
    max_statements: int = 4,
    char_budget: int = 1400,
) -> SynthesisResult | None:
    """Return an ``ENTITY SUMMARY`` block for the question's entities, or ``None``.

    Fires only when the question names at least one retrieved entity that has
    two or more distinct statements across the retrieved facts. Temporal
    questions are skipped — they already receive a dedicated dated timeline in
    the reader context, so an entity summary would be redundant noise.
    """
    if qtype == "temporal-reasoning" or not facts:
        return None

    all_entities = focus_entities(question, (e for fact in facts for e in fact.entities))
    if not all_entities:
        return None

    digests: list[EntityDigest] = []
    for entity in all_entities:
        digest = _digest_entity(entity, facts, max_statements)
        if len(digest.statements) >= _MIN_STATEMENTS:
            digests.append(digest)

    if not digests:
        return None

    # Most-evidenced entities first; cap to keep the block compact.
    digests.sort(key=lambda d: len(d.statements), reverse=True)

    rendered: list[str] = [_HEADER]
    used = 0
    chars = len(_HEADER)
    for digest in digests:
        block = digest.render()
        if used and chars + len(block) + 1 > char_budget:
            break
        rendered.append(block)
        chars += len(block) + 1
        used += 1
        if used >= max_entities:
            break

    # ``digests`` is non-empty here and the first block is always appended
    # (the budget guard is gated on ``used``), so at least one entity renders.
    return SynthesisResult(entities=digests[:used], message="\n".join(rendered))
