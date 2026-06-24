# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — recall query-stream ledger (MS.1: calibration + salience source)

"""Append-only ledger of recall queries and their outcomes.

Every ``remanentia_recall`` invocation appends one ``query`` record here:
the text asked, which memories came back, and whether the system abstained.
A later, separate ``outcome`` record links a prior query to whether the
recalled memories were actually used. The two-record design keeps the file
append-only — no line is ever rewritten — which makes concurrent writers
safe and matches the memory model's "never delete, keep both" lineage rule.

This stream is the **query-weighted** distribution the fleet-memory design
needs (brainstorm 2026-06-24, MS.1): the conformal abstention gate must be
calibrated on the real questions asked, not on activity-weighted ingest, or
its coverage guarantee is void under covariate shift. It is also the
admission salience prior — what actually gets queried is what is worth
remembering well.

The ledger records facts (asked / returned / abstained / used); it does not
decide abstention. ``abstained`` is caller-supplied so that, once the
conformal gate exists, the real decision is recorded; until then a caller
may pass ``None`` and downstream code reads ``found`` (objective) instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_QUERY = "query"
_OUTCOME = "outcome"


def _now() -> float:
    """Return the current wall-clock time in epoch seconds."""
    return time.time()


def _identity() -> str:
    """Return the querying agent's identity from the environment.

    Falls back to ``"unknown"`` when ``REMANENTIA_AGENT`` is unset, so a
    record always carries a non-empty ``by`` field for provenance.
    """
    return os.environ.get("REMANENTIA_AGENT", "").strip() or "unknown"


@dataclass(frozen=True)
class RecallQuery:
    """One recorded recall request and what it returned.

    Parameters
    ----------
    event_id
        Stable identifier other records (outcomes) refer back to.
    ts
        Epoch seconds when the recall was served.
    by
        Identity of the querying agent.
    query
        The recall query text.
    top_k
        The requested result count.
    project
        Project filter applied to the recall, or ``""`` for none.
    returned_ids
        ``source:name`` identifiers of the memories returned, in rank order.
    found
        Whether any memory was returned (objective; the abstention proxy
        until the conformal gate lands).
    score
        The top retrieval score of the returned memories, or ``None`` when
        nothing was returned. The per-recall nonconformity signal the
        conformal retrieval gate calibrates on.
    abstained
        Whether the system abstained, when the caller knows; ``None`` when
        no abstention decision was made (read ``found`` instead).
    was_used
        Whether the recalled memories were used downstream, once an outcome
        has been recorded; ``None`` until then. A *usage* signal, not a
        correctness label — a recalled memory can be used and still wrong.
        Auto-derived from recall→remember loop closure; suitable for
        cold-start calibration and a retrieval-precision monitor, not for the
        safety threshold.
    was_correct
        Whether the recalled memories were *right*, once a correctness outcome
        has been recorded; ``None`` until then. The label the conformal
        abstention gate must calibrate on, supplied by a downstream verifier's
        verdict (a clean answer that used the memory ⇒ correct; a
        flagged/halted/corrected answer ⇒ not). Orthogonal to ``was_used``: a
        memory can be used and wrong, or correct but unused.
    """

    event_id: str
    ts: float
    by: str
    query: str
    top_k: int
    project: str
    returned_ids: tuple[str, ...]
    found: bool
    score: float | None = None
    abstained: bool | None = None
    was_used: bool | None = None
    was_correct: bool | None = None


@dataclass
class RecallLedger:
    """Append-only JSONL ledger of recall queries and outcomes.

    Parameters
    ----------
    path
        Destination JSONL file. Parent directories are created on first
        write; reads of a missing file yield nothing.
    """

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        query: str,
        returned_ids: list[str] | tuple[str, ...],
        *,
        top_k: int,
        project: str = "",
        by: str | None = None,
        score: float | None = None,
        abstained: bool | None = None,
    ) -> str:
        """Append a ``query`` record and return its ``event_id``.

        Parameters
        ----------
        query
            The recall query text.
        returned_ids
            ``source:name`` identifiers returned, in rank order.
        top_k
            The requested result count.
        project
            Project filter applied, or ``""``.
        by
            Querying agent identity; resolved from the environment when
            ``None``.
        score
            The top retrieval score, or ``None`` when nothing was returned;
            the conformal gate's per-recall nonconformity signal.
        abstained
            The abstention decision when known, else ``None``.

        Returns
        -------
        str
            The ``event_id`` to pass to :meth:`record_outcome`.
        """
        ts = _now()
        ids = tuple(returned_ids)
        actor = by if by is not None else _identity()
        event_id = self._event_id(query, ts, actor)
        row = {
            "kind": _QUERY,
            "event_id": event_id,
            "ts": ts,
            "by": actor,
            "query": query,
            "top_k": int(top_k),
            "project": project,
            "returned_ids": list(ids),
            "found": bool(ids),
            "score": score,
            "abstained": abstained,
        }
        self._append(row)
        return event_id

    def record_outcome(
        self,
        event_id: str,
        *,
        was_used: bool | None = None,
        was_correct: bool | None = None,
    ) -> None:
        """Append an ``outcome`` record linking a query to a realised label.

        The two labels are independent and may be recorded in separate calls
        (usage is known at loop-closure time; correctness arrives later, from a
        downstream verifier's verdict). At least one must be supplied; only the
        supplied label(s) are written, so a usage outcome never clobbers a
        correctness outcome or vice versa.

        Parameters
        ----------
        event_id
            The ``event_id`` returned by :meth:`record`.
        was_used
            Whether the recalled memories were used downstream (usage proxy).
        was_correct
            Whether the recalled memories were right (calibration label).
        """
        if was_used is None and was_correct is None:
            raise ValueError("record_outcome requires was_used and/or was_correct")
        row: dict[str, object] = {
            "kind": _OUTCOME,
            "event_id": event_id,
            "ts": _now(),
        }
        if was_used is not None:
            row["was_used"] = bool(was_used)
        if was_correct is not None:
            row["was_correct"] = bool(was_correct)
        self._append(row)

    def queries(self) -> Iterator[RecallQuery]:
        """Yield each recorded query with its outcome merged in.

        A query's ``was_used`` is filled from the latest matching outcome
        record (later outcomes supersede earlier ones). Malformed or
        non-query lines are skipped. Queries are yielded in the order they
        were recorded.

        Yields
        ------
        RecallQuery
            One per ``query`` record, with ``was_used`` resolved.
        """
        if not self.path.exists():
            return
        rows = self._read_rows()
        used: dict[str, bool] = {}
        correct: dict[str, bool] = {}
        for row in rows:
            if row.get("kind") != _OUTCOME or not isinstance(row.get("event_id"), str):
                continue
            event_id = row["event_id"]
            if "was_used" in row:
                used[event_id] = bool(row["was_used"])
            if "was_correct" in row:
                correct[event_id] = bool(row["was_correct"])
        for row in rows:
            if row.get("kind") != _QUERY:
                continue
            event_id = row.get("event_id")
            if not isinstance(event_id, str):
                continue
            yield RecallQuery(
                event_id=event_id,
                ts=float(row.get("ts", 0.0)),
                by=str(row.get("by", "unknown")),
                query=str(row.get("query", "")),
                top_k=int(row.get("top_k", 0)),
                project=str(row.get("project", "")),
                returned_ids=tuple(row.get("returned_ids", []) or ()),
                found=bool(row.get("found", False)),
                score=row.get("score"),
                abstained=row.get("abstained"),
                was_used=used.get(event_id),
                was_correct=correct.get(event_id),
            )

    def latest_for(self, query: str, by: str | None = None) -> str | None:
        """Return the ``event_id`` of the most recent matching query.

        Matches on exact ``query`` text and, when ``by`` is given, the
        querying agent. Lets a consumer attach an outcome to the recall it
        just made without tracking event ids itself.

        Parameters
        ----------
        query
            The recall query text to match.
        by
            When given, also require this querying agent.

        Returns
        -------
        str or None
            The latest matching ``event_id``, or ``None`` if none match.
        """
        event_id: str | None = None
        for q in self.queries():
            if q.query == query and (by is None or q.by == by):
                event_id = q.event_id
        return event_id

    @staticmethod
    def _event_id(query: str, ts: float, by: str) -> str:
        """Return a stable short id for a query record."""
        digest = hashlib.sha256(f"{by}\x00{ts!r}\x00{query}".encode()).hexdigest()
        return digest[:16]

    def _append(self, row: dict[str, object]) -> None:
        """Atomically append one JSON row as a line, creating parents."""
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _read_rows(self) -> list[dict[str, Any]]:
        """Read and parse all JSONL rows, skipping malformed lines."""
        rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
        return rows


def default_ledger() -> RecallLedger:
    """Return the ledger at the conventional runtime location.

    Honours ``REMANENTIA_RECALL_LEDGER`` when set; otherwise writes under
    the repository's ``.coordination/runtime/`` runtime directory alongside
    the other audit streams.
    """
    override = os.environ.get("REMANENTIA_RECALL_LEDGER", "").strip()
    if override:
        return RecallLedger(Path(override))
    base = Path(__file__).resolve().parent
    return RecallLedger(base / ".coordination" / "runtime" / "recall_ledger.jsonl")
