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
supersession-aware observations, then feed those instead of the transcript. That
quality-extraction is the lever this module supplies.

Given the raw text of the sessions retrieval already selected, an injected
completion function (cloud OR the same local model that answers — the boundary is
a plain callable, so the no-egress path stays no-egress) is asked to emit one
dated observation per line, marking any fact a later one supersedes. The lines
are parsed into :class:`lean_context.Observation` so the rendered block shares one
format with the deterministic path, then capped to stay lean.

Safety mirrors :mod:`lean_context`: an empty, unparseable, or failed completion
yields an empty context, and the caller falls back to the raw dump — the reader
is never starved by a bad observation pass.
"""

from __future__ import annotations

import re
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


def _build_prompt(question: str, sessions_text: str, max_observations: int) -> str:
    """Compose the Observer instruction over the retrieved *sessions_text*."""
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


def _parse(text: str, *, max_observations: int, char_budget: int) -> list[Observation]:
    """Parse the Observer completion into capped, deduplicated observations."""
    observations: list[Observation] = []
    seen: set[str] = set()
    chars = len(_HEADER)
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
        obs = Observation(date=date, text=body, superseded=superseded)
        line = obs.render()
        if observations and chars + len(line) + 1 > char_budget:
            break
        seen.add(key)
        observations.append(obs)
        chars += len(line) + 1
    return observations


def build_observed_context(
    question: str,
    sessions_text: str,
    complete: Completer,
    *,
    max_observations: int = 40,
    char_budget: int = 8000,
    max_tokens: int = 700,
) -> ObservedContext:
    """Distil *sessions_text* into a lean, dated observation set for the reader.

    Asks *complete* to observe the retrieved sessions and returns the parsed,
    capped observation block. A blank *sessions_text*, a ``None`` completion, or a
    completion with no parseable observation yields an empty context so the caller
    falls back to the raw dump rather than answering from nothing.
    """
    if not sessions_text.strip():
        return ObservedContext(observations=(), rendered="")

    prompt = _build_prompt(question, sessions_text, max_observations)
    try:
        completion = complete(prompt, max_tokens=max_tokens)
    except Exception:
        completion = None
    if not completion:
        return ObservedContext(observations=(), rendered="")

    observations = _parse(completion, max_observations=max_observations, char_budget=char_budget)
    if not observations:
        return ObservedContext(observations=(), rendered="")

    body = "\n".join(o.render() for o in observations)
    return ObservedContext(observations=tuple(observations), rendered=f"{_HEADER}\n{body}")
