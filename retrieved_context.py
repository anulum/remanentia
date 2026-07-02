# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — retrieved-context assembly for large-haystack benchmarking

"""Assemble a budget-limited reader context from retrieval results.

The oracle LongMemEval setting hands the reader every haystack session (~8K
tokens) because the haystack *is* the gold sessions. The realistic full-S
setting has ~50 sessions per question (~123K tokens) of which only ~2 are gold,
so the full history cannot be dumped — the system must *retrieve* the right
sessions. This module turns a list of ranked :class:`FusedResult` facts into the
set of source sessions the reader should see, under an explicit session-count
and character budget, and reports which sessions were selected so retrieval
recall can be measured independently of synthesis.

Session selection is rank-first: a session enters the selected set the first
time one of its facts appears in the (already RRF/rerank-ordered) results.
Highest-ranked sessions win the budget; the survivors are rendered oldest-first
so the reader sees a chronological transcript.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

JsonDiagnosticValue = int | float | list[int] | list[str] | None


class _RankedFact(Protocol):
    """Structural type for the part of a retrieval result this module reads."""

    @property
    def session_idx(self) -> int: ...


class _Result(Protocol):
    """Structural type: a retrieval result exposing its source fact."""

    @property
    def fact(self) -> _RankedFact: ...


@dataclass
class RetrievedContext:
    """The reader context plus the provenance needed to score retrieval.

    Attributes:
        session_text: the rendered transcript of the selected sessions,
            oldest-first, ready to drop into the reader prompt.
        session_blocks: the same selected sessions as individual rendered
            transcripts (the parts joined into ``session_text``), oldest-first —
            so an observer can process one bounded session at a time rather than
            the whole concatenated dump.
        selected_session_idxs: haystack indices of the sessions included, in
            display (chronological) order — the set against which retrieval
            recall is measured.
        n_candidate_sessions: how many distinct sessions the ranked results
            spanned before the budget was applied (selection headroom).
        dropped_to_session_limit: candidate session indices that were valid but
            outside the configured ``max_sessions`` reader limit.
        dropped_to_budget: candidate session indices inside ``max_sessions``
            that were removed because the rendered context would exceed the
            character budget.
    """

    session_text: str
    selected_session_idxs: list[int]
    session_blocks: list[str] = field(default_factory=list)
    n_candidate_sessions: int = 0
    dropped_to_session_limit: list[int] = field(default_factory=list)
    dropped_to_budget: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalSelectionDiagnostics:
    """Structured evidence for a full-S reader session-selection decision.

    Attributes:
        selected_session_idxs: Session indices rendered into the reader prompt.
        candidate_session_count: Number of distinct retrieved sessions before
            reader limits were applied.
        dropped_to_session_limit: Valid candidate session indices excluded by
            the ``max_sessions`` cap.
        dropped_to_budget: Candidate session indices excluded by the character
            budget after the session cap.
        answer_session_recall: Fraction of answer-session IDs selected into the
            reader context, or ``None`` when the dataset row has no answer
            sessions.
        selected_answer_session_ids: Gold answer-session IDs included in the
            reader prompt.
        missing_answer_session_ids: Gold answer-session IDs absent from the
            reader prompt, including IDs missing from the haystack mapping.
        session_limited_answer_session_ids: Gold answer-session IDs excluded by
            the ``max_sessions`` cap.
        budget_dropped_answer_session_ids: Gold answer-session IDs excluded by
            the character budget.
    """

    selected_session_idxs: list[int]
    candidate_session_count: int
    dropped_to_session_limit: list[int]
    dropped_to_budget: list[int]
    answer_session_recall: float | None
    selected_answer_session_ids: list[str]
    missing_answer_session_ids: list[str]
    session_limited_answer_session_ids: list[str]
    budget_dropped_answer_session_ids: list[str]

    def as_json_dict(self) -> dict[str, JsonDiagnosticValue]:
        """Return a JSON-serialisable representation for benchmark artefacts."""
        return {
            "selected_session_idxs": self.selected_session_idxs,
            "candidate_session_count": self.candidate_session_count,
            "dropped_to_session_limit": self.dropped_to_session_limit,
            "dropped_to_budget": self.dropped_to_budget,
            "answer_session_recall": self.answer_session_recall,
            "selected_answer_session_ids": self.selected_answer_session_ids,
            "missing_answer_session_ids": self.missing_answer_session_ids,
            "session_limited_answer_session_ids": self.session_limited_answer_session_ids,
            "budget_dropped_answer_session_ids": self.budget_dropped_answer_session_ids,
        }


def rank_ordered_sessions(results: Sequence[_Result]) -> list[int]:
    """Distinct source-session indices in order of first appearance in *results*.

    *results* is assumed already ordered best-first (RRF + cross-encoder). The
    first fact from a session fixes that session's rank; later facts from the
    same session do not move it. This is the order sessions compete for the
    budget in.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for r in results:
        idx = r.fact.session_idx
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def _render_session(order: int, session: list[dict[str, str]], session_date: str) -> str:
    """Render one session as a dated, role-tagged transcript block."""
    header = f"=== Session {order}"
    if session_date:
        header += f" ({session_date})"
    header += " ==="
    turns = [f"[{turn['role'].upper()}]: {turn['content']}" for turn in session]
    return header + "\n" + "\n".join(turns)


def select_sessions(
    results: Sequence[_Result],
    sessions: Sequence[list[dict[str, str]]],
    session_dates: Sequence[str] | None = None,
    *,
    max_sessions: int = 10,
    char_budget: int = 80_000,
) -> RetrievedContext:
    """Pick the sessions the reader should see and render them chronologically.

    The top ``max_sessions`` distinct sessions by best-fact rank are taken, then
    trimmed lowest-rank-first until the rendered transcript fits ``char_budget``
    (a session is kept whole or dropped whole — never half-rendered). Survivors
    are displayed oldest-first when *session_dates* are available, so multi-step
    reasoning sees the conversation in order.

    Returns a :class:`RetrievedContext` carrying both the prompt text and the
    selected session indices (for retrieval-recall scoring).
    """
    dates = session_dates or []
    candidates = rank_ordered_sessions(results)
    n_candidates = len(candidates)

    # Take top-N by rank, guarding against stale indices.
    valid = [i for i in candidates if 0 <= i < len(sessions)]
    chosen = valid[:max_sessions]
    dropped_to_session_limit = valid[max_sessions:]

    # Enforce the character budget, dropping the lowest-ranked chosen sessions
    # first. Rendered length is recomputed per session (header + turns).
    kept: list[int] = []
    dropped: list[int] = []
    used = 0
    for idx in chosen:
        date = dates[idx] if idx < len(dates) else ""
        block = _render_session(0, sessions[idx], date)
        # +2 for the blank-line join between blocks.
        cost = len(block) + 2
        if kept and used + cost > char_budget:
            dropped.append(idx)
            continue
        kept.append(idx)
        used += cost

    # Display oldest-first by session date; undated sort last but stable.
    def _date_key(idx: int) -> str:
        return dates[idx] if idx < len(dates) and dates[idx] else "\xff"

    display = sorted(kept, key=_date_key)

    parts = [
        _render_session(order + 1, sessions[idx], _date_key(idx).rstrip("\xff"))
        for order, idx in enumerate(display)
    ]
    return RetrievedContext(
        session_text="\n\n".join(parts),
        selected_session_idxs=display,
        session_blocks=parts,
        n_candidate_sessions=n_candidates,
        dropped_to_session_limit=dropped_to_session_limit,
        dropped_to_budget=dropped,
    )


def gold_session_recall(selected_idxs: Sequence[int], gold_idxs: Sequence[int | None]) -> float:
    """Fraction of gold answer sessions present in the selected set.

    1.0 means every gold session was retrieved into the reader context (so any
    failure is synthesis, not retrieval); 0.0 means none were (a pure retrieval
    miss). Empty gold returns 1.0 (nothing to recall — vacuously complete).
    """
    gold = {g for g in gold_idxs if g is not None}
    if not gold:
        return 1.0
    selected = set(selected_idxs)
    return len(gold & selected) / len(gold)


def build_selection_diagnostics(
    context: RetrievedContext,
    *,
    haystack_session_ids: Sequence[str] | None = None,
    answer_session_ids: Sequence[str] | None = None,
) -> RetrievalSelectionDiagnostics:
    """Map selected session indices back to LongMemEval answer-session IDs.

    Parameters
    ----------
    context:
        The selected reader context returned by :func:`select_sessions`.
    haystack_session_ids:
        Dataset session IDs in haystack order. When omitted, answer-session
        recall cannot be tied back to external IDs.
    answer_session_ids:
        Gold answer-session IDs for the question.

    Returns
    -------
    RetrievalSelectionDiagnostics
        The selected/dropped session evidence suitable for JSONL benchmark
        artefacts.
    """
    haystack_ids = [str(sid) for sid in haystack_session_ids or ()]
    answer_ids = [str(sid) for sid in answer_session_ids or ()]
    id_to_idx = {sid: idx for idx, sid in enumerate(haystack_ids)}

    selected_idxs = set(context.selected_session_idxs)
    session_limited_idxs = set(context.dropped_to_session_limit)
    budget_dropped_idxs = set(context.dropped_to_budget)

    selected_answer_ids: list[str] = []
    missing_answer_ids: list[str] = []
    session_limited_answer_ids: list[str] = []
    budget_dropped_answer_ids: list[str] = []

    for answer_id in answer_ids:
        idx = id_to_idx.get(answer_id)
        if idx is None:
            missing_answer_ids.append(answer_id)
            continue
        if idx in selected_idxs:
            selected_answer_ids.append(answer_id)
            continue
        missing_answer_ids.append(answer_id)
        if idx in session_limited_idxs:
            session_limited_answer_ids.append(answer_id)
        if idx in budget_dropped_idxs:
            budget_dropped_answer_ids.append(answer_id)

    answer_session_recall = len(selected_answer_ids) / len(answer_ids) if answer_ids else None
    return RetrievalSelectionDiagnostics(
        selected_session_idxs=list(context.selected_session_idxs),
        candidate_session_count=context.n_candidate_sessions,
        dropped_to_session_limit=list(context.dropped_to_session_limit),
        dropped_to_budget=list(context.dropped_to_budget),
        answer_session_recall=answer_session_recall,
        selected_answer_session_ids=selected_answer_ids,
        missing_answer_session_ids=missing_answer_ids,
        session_limited_answer_session_ids=session_limited_answer_ids,
        budget_dropped_answer_session_ids=budget_dropped_answer_ids,
    )
