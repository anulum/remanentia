# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — lean bi-temporal reader context

"""Build a lean, dated, supersession-resolved context for the reader.

The realistic-SOTA systems do not dump raw history into the reader; they feed a
*lean, dated, compressed* context and let bi-temporal ordering, not an LLM,
resolve conflicts (Engram: a ~9.6k-token bi-temporal slice beats full context by
+10.4 pts; Mastra OM: dated observations with no per-turn retrieval). REMANENTIA's
honest full-S gap is reader-synthesis over a raw dump, so this module is the
W2 lever: turn the retrieved atomic facts into a compact set of dated
observations that *replaces* the raw-session context the reader sees.

It is deterministic and uses the facts' bi-temporal fields — no extra model call.
Duplicates collapse; a fact whose validity has closed (`valid_until` set) is
marked superseded and yielded after the current ones (or dropped); observations
are ordered newest-first so the reader meets the current worldview before its
history; the set is capped to stay lean. This is the failed P1.3 entity-summary
done right: the lean context *is* the reader input, not a block bolted on top of
the raw dump.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class BiTemporalFact(Protocol):
    """Structural shape of an atomic fact with bi-temporal validity.

    Matches :class:`fact_decomposer.AtomicFact`; declared structurally so this
    module does not import the retriever.
    """

    text: str
    valid_from: str
    valid_until: str
    date_mentions: list[str]
    session_date: str
    confidence: float


_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Observation:
    """One dated, deduplicated observation for the reader."""

    date: str
    text: str
    superseded: bool

    def render(self) -> str:
        """Return the prompt line for this observation."""
        when = self.date if self.date else "undated"
        tag = " [superseded]" if self.superseded else ""
        return f"- [{when}] {self.text}{tag}"


@dataclass(frozen=True)
class LeanContext:
    """The lean observation set that replaces the raw-session reader context."""

    observations: tuple[Observation, ...]
    rendered: str

    def __bool__(self) -> bool:
        """Truthy when at least one observation is present."""
        return bool(self.observations)


def _fact_date(fact: BiTemporalFact) -> str:
    """Best-effort ISO date for a fact (explicit mention wins, then validity)."""
    if fact.date_mentions:
        return fact.date_mentions[0]
    if fact.valid_from:
        return fact.valid_from
    return fact.session_date


def _normalise(text: str) -> str:
    """Dedup key collapsing whitespace, case, and edge punctuation."""
    return _WS.sub(" ", text.strip().casefold()).strip(" .,:;-")


_HEADER = "OBSERVATIONS (lean, deduplicated, newest first; [superseded] = no longer valid):"


def build_lean_context(
    facts: Sequence[BiTemporalFact],
    *,
    max_observations: int = 40,
    char_budget: int = 8000,
    drop_superseded: bool = False,
) -> LeanContext:
    """Collapse retrieved facts into a lean, dated, supersession-resolved set.

    Duplicates (by normalised text) collapse, keeping the highest-confidence
    instance. A fact with ``valid_until`` set is superseded; it is dropped when
    *drop_superseded* is true, else kept and marked. Observations are ordered
    newest-first (current facts ahead of superseded ones at the same date) and
    capped by *max_observations* and *char_budget* to stay lean.
    """
    best: dict[str, BiTemporalFact] = {}
    for fact in facts:
        text = _WS.sub(" ", fact.text.strip())
        key = _normalise(text)
        if not key:
            continue
        kept = best.get(key)
        if kept is None or fact.confidence > kept.confidence:
            best[key] = fact

    observations: list[Observation] = []
    for fact in best.values():
        superseded = bool(fact.valid_until)
        if superseded and drop_superseded:
            continue
        observations.append(
            Observation(
                date=_fact_date(fact),
                text=_WS.sub(" ", fact.text.strip()),
                superseded=superseded,
            )
        )

    # Newest first; current (not superseded) ahead of superseded at the same date.
    observations.sort(key=lambda o: (o.date, not o.superseded), reverse=True)

    selected: list[Observation] = []
    chars = len(_HEADER)
    for obs in observations:
        line = obs.render()
        if selected and chars + len(line) + 1 > char_budget:
            break
        selected.append(obs)
        chars += len(line) + 1
        if len(selected) >= max_observations:
            break

    if not selected:
        return LeanContext(observations=(), rendered="")

    body = "\n".join(o.render() for o in selected)
    return LeanContext(observations=tuple(selected), rendered=f"{_HEADER}\n{body}")
