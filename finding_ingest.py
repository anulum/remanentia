# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — finding ingest bridge (MS.3: the persistent-memory read-side seam)

"""Consume authored findings off the SYNAPSE hub event store into memory.

SYNAPSE is the write side: agents author ``finding`` events onto a durable,
sequence-numbered hub log, each event a claim with provenance and the three
honesty axes (evidence kind, claim status, freshness). This module is the read
side — the seam that pulls those findings into Remanentia's retrievable store.
It reads everything above a persisted sequence cursor (loss-free, idempotent
resume across hub restarts), runs each finding through the admission gate, and
writes the admitted ones to a sink. Structurally invalid findings (a missing
validity window, an invariant breach) are gated out and logged, never indexed.

The core (:func:`ingest_findings`) is deliberately free of any SYNAPSE or project-workspace
coupling: the event store, the finding parser, the admission gate, and the sink
are all injected, so it is unit-testable with fakes and extractable as the
read-side core of a standalone fleet-memory package. Only :func:`ingest_from_hub`
binds the live ``synapse_channel`` event store + gate, and only the default sink
knows where Remanentia keeps its findings on disk.

The cursor advances past every scanned event — admitted, floored, or rejected —
so a structurally invalid finding is gated once and never re-examined. Admission
follows the gate's verdict: ``accept`` (above boundary) and ``floor`` (rendered
at boundary, but kept) are stored; ``reject`` is dropped with its reasons. A
refuted-but-well-formed finding is *accepted* — a negative result is first-class
memory, not an error.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# The hub's memory event kinds and the gate verdicts are stable wire values; we
# name them here so the core never imports a synapse_channel internal module.
MEMORY_KINDS = ("recall", "checkpoint", "handoff", "finding")
VERDICT_ACCEPT = "accept"
VERDICT_FLOOR = "floor"
VERDICT_REJECT = "reject"
DEFAULT_ADMITTING_VERDICTS = (VERDICT_ACCEPT, VERDICT_FLOOR)


@dataclass(frozen=True)
class IngestReport:
    """Outcome of one ingest pass."""

    scanned: int
    admitted: int
    rejected: int
    last_seq: int
    rejections: tuple[tuple[int, str], ...] = ()
    """``(seq, reason)`` for each finding the gate dropped, for an audit trail."""

    @property
    def advanced(self) -> bool:
        """Whether the cursor moved (new events were processed)."""
        return self.scanned > 0


class StoredEventLike(Protocol):
    """The minimal shape :func:`ingest_findings` needs from a hub event.

    Declared read-only so a ``NamedTuple`` hub event (the real ``StoredEvent``)
    satisfies the protocol structurally.
    """

    @property
    def seq(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def payload(self) -> Any: ...


class EventSource(Protocol):
    """The minimal read surface of a hub event store."""

    def read_since(
        self, after_seq: int, *, kinds: Iterable[str] | None = ..., limit: int | None = ...
    ) -> Sequence[StoredEventLike]: ...


class DecisionLike(Protocol):
    """The admission gate's verdict shape (read-only, so a frozen gate
    ``Decision`` satisfies it structurally)."""

    @property
    def verdict(self) -> str: ...
    @property
    def reasons(self) -> Sequence[str]: ...


class FindingSink(Protocol):
    """Where an admitted finding is persisted for retrieval."""

    def write(self, finding: Any, seq: int, verdict: str) -> None: ...


@dataclass
class SeqCursor:
    """File-backed last-sequence cursor for loss-free incremental resume.

    Stores a single integer — the highest hub sequence already ingested — so a
    restart resumes exactly above it, with no re-ingest and no gap. A missing or
    unreadable cursor file reads as ``0`` (ingest from the beginning).
    """

    path: Path

    def load(self) -> int:
        try:
            return int(json.loads(self.path.read_text(encoding="utf-8"))["last_seq"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def save(self, seq: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"last_seq": int(seq)}) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def ingest_findings(
    store: EventSource,
    sink: FindingSink,
    cursor: SeqCursor,
    *,
    parse_finding: Callable[..., Any],
    admit: Callable[..., DecisionLike],
    kinds: Iterable[str] = ("finding",),
    limit: int | None = None,
    admitting_verdicts: Sequence[str] = DEFAULT_ADMITTING_VERDICTS,
) -> IngestReport:
    """Pull findings above the cursor, gate them, write the admitted, advance.

    Every collaborator is injected so the core carries no SYNAPSE or storage
    coupling. *parse_finding* turns an event payload into a finding (a payload
    that will not parse is treated as a rejection, never a crash); *admit*
    returns the gate verdict; *sink* persists the admitted finding together with
    its verdict so the read side can honour the boundary distinction. The cursor
    is saved at the highest scanned sequence — including rejected events — so a
    malformed or invalid finding is gated exactly once.
    """
    last = cursor.load()
    events = list(store.read_since(last, kinds=kinds, limit=limit))
    admitting = set(admitting_verdicts)
    admitted = 0
    rejections: list[tuple[int, str]] = []
    high = last
    for event in events:
        if event.seq > high:
            high = event.seq
        try:
            finding = parse_finding(event.payload)
        except Exception as exc:  # noqa: BLE001 — a bad payload is a rejection, not a crash
            rejections.append((event.seq, f"unparseable payload: {exc}"))
            continue
        decision = admit(finding)
        if decision.verdict in admitting:
            sink.write(finding, event.seq, decision.verdict)
            admitted += 1
        else:
            rejections.append((event.seq, "; ".join(decision.reasons) or decision.verdict))
    if events:
        cursor.save(high)
    return IngestReport(
        scanned=len(events),
        admitted=admitted,
        rejected=len(rejections),
        last_seq=high,
        rejections=tuple(rejections),
    )


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, *, limit: int = 48) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:limit] or "finding"


def _statement_hash(statement: str) -> str:
    """Short content hash of a finding's statement, for idempotent filenames."""
    from hashlib import sha256

    return sha256(statement.encode("utf-8")).hexdigest()[:16]


@dataclass
class MarkdownFindingSink:
    """Persist admitted findings as Markdown with an honesty-axis frontmatter.

    One file per finding, named by a content hash of the statement so a
    re-ingest of the same finding overwrites in place rather than duplicating.
    The verdict is recorded so a floored (boundary) finding is not later read as
    if it were above the boundary — honesty propagation needs that distinction
    at the source, not only at synthesis.
    """

    directory: Path
    content_hash: Callable[[str], str] = _statement_hash

    def write(self, finding: Any, seq: int, verdict: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = finding.as_dict() if hasattr(finding, "as_dict") else dict(finding)
        statement = str(record.get("statement", ""))
        name = f"{_slug(str(record.get('subkind', 'finding')))}-{self.content_hash(statement)}.md"
        front = {
            "subkind": record.get("subkind"),
            "claim_status": record.get("claim_status"),
            "freshness": record.get("freshness"),
            "evidence_kind": record.get("evidence_kind"),
            "verdict": verdict,
            "hub_seq": seq,
            "provenance": record.get("provenance"),
            "lifecycle": record.get("lifecycle"),
            "entities": list(record.get("entities", ()) or ()),
            "tags": list(record.get("tags", ()) or ()),
        }
        body = "\n".join(
            [
                "---",
                json.dumps(front, ensure_ascii=False, indent=2, sort_keys=True),
                "---",
                "",
                statement,
                "",
            ]
        )
        path = self.directory / name
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)


def ingest_from_hub(
    hub_db_path: str | Path,
    sink: FindingSink,
    cursor_path: str | Path,
    *,
    kinds: Iterable[str] = ("finding",),
    limit: int | None = None,
) -> IngestReport:
    """Bind the live ``synapse_channel`` hub store + gate and run one pass.

    The only SYNAPSE-coupled entry point: it opens the hub event store, uses
    ``Finding.from_dict`` as the parser and the package ``admit`` as the gate,
    and delegates everything else to :func:`ingest_findings`. Requires the
    ``synapse-channel`` extra; the import is deferred so the module loads (and
    the generic core stays usable) without it.
    """
    import synapse_channel as sc  # deferred: optional [bus] dependency

    store = sc.EventStore(str(hub_db_path))
    try:
        return ingest_findings(
            store,
            sink,
            SeqCursor(Path(cursor_path)),
            parse_finding=sc.Finding.from_dict,
            admit=sc.admit,
            kinds=kinds,
            limit=limit,
        )
    finally:
        store.close()


def default_findings_dir(base: str | Path | None = None) -> Path:
    """Where Remanentia keeps ingested findings (env-pointed, project-workspace default)."""
    if base is None:
        base = os.environ.get("REMANENTIA_BASE")
    root = Path(base) if base is not None else Path(__file__).resolve().parent
    return root / "memory" / "semantic" / "findings"


def main() -> int:  # pragma: no cover — CLI/cron entry point
    """Run one ingest pass against the local hub; print the report."""
    import sys

    hub = os.environ.get("SYNAPSE_HUB_DB", str(Path.home() / "synapse" / "hub.db"))
    cursor = os.environ.get(
        "REMANENTIA_FINDING_CURSOR",
        str(default_findings_dir().parent / "finding_ingest_cursor.json"),
    )
    report = ingest_from_hub(hub, MarkdownFindingSink(default_findings_dir()), cursor)
    print(
        f"finding-ingest: scanned={report.scanned} admitted={report.admitted} "
        f"rejected={report.rejected} last_seq={report.last_seq}"
    )
    for seq, reason in report.rejections:
        print(f"  rejected seq={seq}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
