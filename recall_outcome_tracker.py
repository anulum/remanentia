# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — recall outcome tracker (MS.1: auto-derive the was_used label)

"""Close the recall→use loop so the ``was_used`` label populates itself.

The recall ledger has always had room for a ``was_used`` outcome, but nothing
filled it: the feedback tool existed and no one called it, so every recall sat
unlabelled — the schema-present-but-empty failure the fleet-memory design warned
about. A label that depends on an agent remembering to rate each recall does not
hold at scale; the honest fix is to *derive* usage from observable behaviour.

This tracker derives it. When a recall returns memories, their texts are
fingerprinted and held briefly. When the same agent next writes a memory whose
text substantially echoes one of those returned memories, the recall demonstrably
informed downstream work and is recorded ``was_used=True``. Recalls that age out
with no such echo are recorded ``was_used=False`` — returned but unused. No agent
goodwill, no human labels.

It is a *usage* proxy, not a correctness label. A recalled memory can be echoed
into a wrong answer (used yet incorrect) or inform an answer without being quoted
(correct yet not detected as used). ``was_used`` is therefore suitable for
cold-start calibration, a retrieval-precision monitor, and ranking — never for
the safety/abstention threshold, which calibrates on the separate ``was_correct``
label supplied by a downstream verifier.

Matching is token containment, not similarity: a returned memory counts as used
when a high fraction of *its* content tokens reappear in the new text (the new
text contains the memory), so a long new memory that merely shares a few common
words does not trip it, and a short returned memory must be almost wholly echoed.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TTL_S = 1800.0
"""How long a recall stays eligible for loop closure before it is ruled unused.
Matches the working-memory horizon: use that follows a recall follows within the
same work session, not half a day later."""

DEFAULT_MAX_PENDING = 512
"""Cap on buffered recalls; the oldest is ruled unused when the cap is hit, so a
burst of recalls cannot grow memory without bound."""

DEFAULT_MIN_CONTAINMENT = 0.6
"""Fraction of a returned memory's tokens that must reappear in the new text for
it to count as echoed. High enough that incidental word overlap does not register."""

DEFAULT_MIN_TOKENS = 4
"""Returned memories with fewer distinctive tokens are ignored for matching — too
short to echo meaningfully, and prone to false positives on common words."""

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Frequent function words carry no usage signal; dropping them stops a new memory
# from "containing" a returned one purely through shared connective tissue.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "as",
        "at",
        "by",
        "from",
        "we",
        "i",
        "you",
        "they",
        "he",
        "she",
        "not",
        "no",
        "do",
        "did",
    }
)


def tokenize(text: str) -> frozenset[str]:
    """Return the distinctive content-token set of *text* for containment tests."""
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS)


def containment(memory_tokens: frozenset[str], text_tokens: frozenset[str]) -> float:
    """Fraction of *memory_tokens* present in *text_tokens* (0.0 when empty)."""
    if not memory_tokens:
        return 0.0
    return len(memory_tokens & text_tokens) / len(memory_tokens)


@dataclass
class _Pending:
    """A buffered recall awaiting evidence that its memories were used."""

    event_id: str
    by: str
    ts: float
    memory_tokens: tuple[frozenset[str], ...]


@dataclass
class RecallOutcomeTracker:
    """Derive ``was_used`` from recall→remember loop closure.

    Parameters
    ----------
    ttl_s
        Seconds a recall stays eligible before being ruled unused.
    max_pending
        Maximum buffered recalls; the oldest is ruled unused past the cap.
    min_containment
        Token-containment threshold for a returned memory to count as echoed.
    min_tokens
        Returned memories with fewer content tokens are ignored for matching.
    """

    ttl_s: float = DEFAULT_TTL_S
    max_pending: int = DEFAULT_MAX_PENDING
    min_containment: float = DEFAULT_MIN_CONTAINMENT
    min_tokens: int = DEFAULT_MIN_TOKENS
    _pending: list[_Pending] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe_recall(
        self,
        event_id: str,
        by: str,
        returned_texts: Sequence[str],
        *,
        ledger: Any | None = None,
        now: float | None = None,
    ) -> None:
        """Buffer a recall's returned memories for later loop closure.

        Memories with too few content tokens are dropped; a recall left with no
        usable memory is not buffered (it could never be matched, and would only
        ever age out as a spurious ``was_used=False``). Buffering first expires
        the stale tail and trims any overflow, recording those outcomes.
        """
        moment = time.time() if now is None else now
        usable = tuple(
            tokens for text in returned_texts if len(tokens := tokenize(text)) >= self.min_tokens
        )
        with self._lock:
            self._expire_locked(ledger, moment)
            if usable:
                self._pending.append(
                    _Pending(event_id=event_id, by=by, ts=moment, memory_tokens=usable)
                )
                self._trim_overflow_locked(ledger)

    def note_text(
        self,
        text: str,
        by: str,
        ledger: Any,
        *,
        now: float | None = None,
    ) -> list[str]:
        """Close loops against *text* written by *by*; return used event ids.

        Expires the stale tail first, then matches *text* against every pending
        recall from the same agent: a recall whose most-echoed returned memory
        meets the containment threshold is recorded ``was_used=True`` and
        dropped. Recalls from other agents are untouched.
        """
        moment = time.time() if now is None else now
        text_tokens = tokenize(text)
        used: list[str] = []
        with self._lock:
            self._expire_locked(ledger, moment)
            survivors: list[_Pending] = []
            for pending in self._pending:
                if pending.by == by and self._is_echoed(pending, text_tokens):
                    ledger.record_outcome(pending.event_id, was_used=True)
                    used.append(pending.event_id)
                else:
                    survivors.append(pending)
            self._pending = survivors
        return used

    def expire(self, ledger: Any, *, now: float | None = None) -> list[str]:
        """Rule every recall older than the TTL unused; return their event ids."""
        moment = time.time() if now is None else now
        with self._lock:
            return self._expire_locked(ledger, moment)

    def flush(self, ledger: Any) -> list[str]:
        """Rule every still-pending recall unused (e.g. at shutdown)."""
        with self._lock:
            drained = [p.event_id for p in self._pending]
            for pending in self._pending:
                ledger.record_outcome(pending.event_id, was_used=False)
            self._pending = []
            return drained

    @property
    def pending_count(self) -> int:
        """Number of recalls currently awaiting loop closure."""
        with self._lock:
            return len(self._pending)

    def _is_echoed(self, pending: _Pending, text_tokens: frozenset[str]) -> bool:
        return any(
            containment(memory, text_tokens) >= self.min_containment
            for memory in pending.memory_tokens
        )

    def _expire_locked(self, ledger: Any | None, now: float) -> list[str]:
        cutoff = now - self.ttl_s
        expired = [p for p in self._pending if p.ts < cutoff]
        if not expired:
            return []
        self._pending = [p for p in self._pending if p.ts >= cutoff]
        if ledger is not None:
            for pending in expired:
                ledger.record_outcome(pending.event_id, was_used=False)
        return [p.event_id for p in expired]

    def _trim_overflow_locked(self, ledger: Any | None) -> None:
        while len(self._pending) > self.max_pending:
            oldest = self._pending.pop(0)
            if ledger is not None:
                ledger.record_outcome(oldest.event_id, was_used=False)
