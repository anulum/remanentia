# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — LLM observe→reflect reader context

"""Distil retrieved sessions into a lean, dated observation set via an LLM.

The deterministic sibling :mod:`lean_context` collapses retrieved *atomic facts*
by rule (dedup, validity flag, cap). Measured on full-S that *hurt* — a rule
cannot tell which dropped clause the reader still needed (−7.6 pp vs the raw
dump). The realistic-SOTA systems do not dedup by rule: Mastra's Observer and
Engram's bi-temporal slice use an *LLM* to read the raw exchange and write dated,
supersession-aware observations, then feed those instead of the transcript. The
budget-minimal cloud ablation confirmed the LLM version lifts exactly the
categories it touches (multi-session +6.5 pp, preference +16.7 pp); this module
supplies that quality-extraction.

The observer runs over the retrieved sessions in size-bounded chunks: sessions
are packed into groups no larger than *per_call_char_budget* and each group gets
one completion. A large budget puts every session in one call (the cloud shape
the ablation validated, byte-for-byte); a small budget yields one call per session
or two, so a local model with a short context window is never handed a prompt it
cannot read (the whole retrieved dump is ~110 k chars — far past a small model's
window, which is why the single-call local smoke returned nothing). Observations
merge across chunks, deduplicated, order preserved (chunks are oldest-first; the
observer writes newest-first within each), then capped to stay lean.

Safety mirrors :mod:`lean_context`: no sessions, or every completion blank/failed/
unparseable, yields an empty context and the caller falls back to the raw dump —
the reader is never starved by a bad observation pass.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from lean_context import Observation


class Completer(Protocol):
    """LLM completion boundary: return text for *prompt*, or ``None`` on failure.

    Declared structurally so this module imports no specific backend; the bench
    injects a callable that routes to the hosted or the local model, keeping the
    sovereign no-egress path free of any cloud call.
    """

    def __call__(self, prompt: str, max_tokens: int) -> str | None:
        """Return a completion for *prompt* capped at *max_tokens*, or ``None``."""


@dataclass(frozen=True)
class ObservedContext:
    """The LLM-distilled observation set that replaces the raw reader context."""

    observations: tuple[Observation, ...]
    rendered: str

    def __bool__(self) -> bool:
        """Truthy when at least one observation was distilled."""
        return bool(self.observations)


_HEADER = (
    "OBSERVATIONS (distilled from the retrieved sessions, dated, newest first; "
    "[superseded] = a later observation overrides this):"
)

# One observation line: optional bullet, optional [date], text, optional trailing
# [superseded...] marker. The date is kept verbatim when present so the reader
# meets the same date strings the transcript used.
_LINE = re.compile(
    r"^\s*[-*]?\s*(?:\[(?P<date>[^\]]*)\]\s*)?(?P<text>.*?)\s*$",
    re.IGNORECASE,
)
_SUPERSEDED = re.compile(r"\[\s*superseded[^\]]*\]", re.IGNORECASE)
_WS = re.compile(r"\s+")


def _pack(sessions: Sequence[str], per_call_char_budget: int) -> list[str]:
    """Group session blocks into chunks no larger than *per_call_char_budget*.

    Sessions are kept whole (never split mid-transcript); a session larger than
    the budget forms its own chunk. Empty/whitespace blocks are dropped.
    """
    chunks: list[str] = []
    current: list[str] = []
    used = 0
    for block in sessions:
        block = block.strip()
        if not block:
            continue
        cost = len(block) + 2  # +2 for the blank-line join
        if current and used + cost > per_call_char_budget:
            chunks.append("\n\n".join(current))
            current = []
            used = 0
        current.append(block)
        used += cost
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _build_prompt(question: str, sessions_text: str, max_observations: int) -> str:
    """Compose the Observer instruction over one chunk of *sessions_text*."""
    return (
        "You are a memory observer. Read the dated conversation sessions below "
        "and write the observations relevant to answering the question. Rules:\n"
        "- One observation per line, newest first.\n"
        "- Begin each line with the event's date in square brackets, e.g. "
        "[2023-05-01], taken from the session it came from.\n"
        "- State only what the sessions say — never invent a fact or a date.\n"
        "- When a later session overrides an earlier fact (a changed preference, "
        "an updated plan), keep both and append [superseded] to the older one.\n"
        f"- Write at most {max_observations} lines; omit anything irrelevant to "
        "the question.\n\n"
        f"QUESTION: {question}\n\n"
        f"SESSIONS:\n{sessions_text}\n\n"
        "OBSERVATIONS:"
    )


def _parse(text: str, *, max_observations: int) -> list[Observation]:
    """Parse one Observer completion into deduplicated observations (per-chunk)."""
    observations: list[Observation] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        if len(observations) >= max_observations:
            break
        stripped = raw_line.strip()
        if not stripped:
            continue
        superseded = bool(_SUPERSEDED.search(stripped))
        cleaned = _SUPERSEDED.sub("", stripped)
        match = _LINE.match(cleaned)
        if match is None:  # pragma: no cover — the pattern matches any line
            continue
        body = _WS.sub(" ", match.group("text").strip())
        if not body:
            continue
        date = _WS.sub(" ", (match.group("date") or "").strip())
        key = body.casefold()
        if key in seen:
            continue
        seen.add(key)
        observations.append(Observation(date=date, text=body, superseded=superseded))
    return observations


def build_observed_context(
    question: str,
    sessions: Sequence[str],
    complete: Completer,
    *,
    max_observations: int = 40,
    char_budget: int = 8000,
    per_call_char_budget: int = 100_000,
    max_tokens: int = 700,
) -> ObservedContext:
    """Distil the retrieved *sessions* into a lean, dated observation set.

    Sessions are packed into chunks of at most *per_call_char_budget* characters,
    each observed in one call to *complete*; observations merge across chunks
    (deduplicated by text, arrival order preserved) and the result is capped by
    *max_observations* and *char_budget*. An empty *sessions*, or every chunk
    returning a blank/failed/unparseable completion, yields an empty context so
    the caller falls back to the raw dump rather than answering from nothing.
    """
    chunks = _pack(sessions, per_call_char_budget)
    if not chunks:
        return ObservedContext(observations=(), rendered="")

    merged: list[Observation] = []
    seen: set[str] = set()
    for chunk in chunks:
        prompt = _build_prompt(question, chunk, max_observations)
        try:
            completion = complete(prompt, max_tokens=max_tokens)
        except Exception:
            completion = None
        if not completion:
            continue
        for obs in _parse(completion, max_observations=max_observations):
            key = obs.text.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(obs)

    if not merged:
        return ObservedContext(observations=(), rendered="")

    selected: list[Observation] = []
    chars = len(_HEADER)
    for obs in merged:
        line = obs.render()
        if selected and chars + len(line) + 1 > char_budget:
            break
        selected.append(obs)
        chars += len(line) + 1
        if len(selected) >= max_observations:
            break

    body = "\n".join(o.render() for o in selected)
    return ObservedContext(observations=tuple(selected), rendered=f"{_HEADER}\n{body}")
