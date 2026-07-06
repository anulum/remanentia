# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — reader context-window budget

"""Fit a priority-ordered reader context into the model's window.

A cloud reader has a ~128 k-token window, so the benchmark assembles the whole
retrieved context — retrieved facts, a precompute/synthesis header, and the raw
conversation history — and sends it in one call. A sovereign *local* reader has a
short window (a llama.cpp server pinned at 8192 tokens); handed a ~110 k-char
prompt it does not answer — llama.cpp stalls on the over-window prompt rather than
truncating, which hangs the run (Ollama silently truncates and merely answers from
a clipped tail — neither is what we want).

:func:`fit_context` closes that: given the context sections in *priority order*
(most answer-critical first — computed totals, distilled observations, retrieved
facts — with the raw history last) and a character budget, it keeps whole sections
until the budget, truncates the section that would overflow, and drops the rest.
The budget is the reader window expressed in characters (tokens × chars-per-token,
minus the instruction and the reserved answer); a non-positive budget means an
effectively unbounded cloud window and returns every section verbatim. The
answer-critical sections thus always survive and only the raw-history tail is
clipped, never the observations the reader needs.
"""

from __future__ import annotations

from collections.abc import Sequence

_TRUNCATION_MARKER = "\n[... truncated to fit the reader context window ...]"
_JOIN = "\n\n"


def fit_context(sections: Sequence[str], budget_chars: int) -> str:
    """Join priority-ordered *sections* into a context within *budget_chars*.

    Sections are ordered most-important first. With ``budget_chars <= 0`` (a cloud
    window, effectively unbounded) every non-empty section is joined verbatim.
    Otherwise sections are added whole while they fit; the first section that would
    overflow is truncated to the remaining room and marked, and every later (lower
    priority) section is dropped. Blank sections are skipped so they never consume
    a separator or the budget.
    """
    kept = [s for s in sections if s.strip()]
    if budget_chars <= 0:
        return _JOIN.join(kept)

    out: list[str] = []
    used = 0
    for section in kept:
        sep = len(_JOIN) if out else 0
        if used + sep + len(section) <= budget_chars:
            out.append(section)
            used += sep + len(section)
            continue
        # This section overflows: fit as much as the marker allows, then stop.
        room = budget_chars - used - sep - len(_TRUNCATION_MARKER)
        if room > 0:
            out.append(section[:room] + _TRUNCATION_MARKER)
        break
    return _JOIN.join(out)
