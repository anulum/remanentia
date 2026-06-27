# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — SYNAPSE feed ingest bridge (MS.2 fleet-fed recall)

"""Ingest explicit SYNAPSE feed findings into Remanentia memory.

The SYNAPSE hub has two durable surfaces. :mod:`finding_ingest` consumes the
structured SQLite event store. This module consumes the append-only
``feed.ndjson`` relay file used by terminal agents and keeps the same honesty
boundary: only explicit finding or decision records are promoted into
Synapse-style finding payloads, then the real ``synapse_channel`` parser and
admission gate decide what may enter memory.

The filter is intentionally precision-first. Ordinary chat such as
``"build is green"`` is not inferred into memory. A feed row must either carry
``ty == "finding"``, ``ty == "decision"``, or use an explicit text marker such
as ``"Finding: ..."`` or ``"Decision: ..."``. Producer-authored feed text is
born as ``producer_asserted`` + ``bounded_support`` + ``traceable_unchecked``:
useful for recall salience, but never above the boundary until independently
verified.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from finding_ingest import (
    DEFAULT_ADMITTING_VERDICTS,
    DecisionLike,
    FindingSink,
    MarkdownFindingSink,
    SeqCursor,
)
from store_paths import DEFAULT_FEED_CURSOR_NAME as STORE_DEFAULT_FEED_CURSOR_NAME
from store_paths import default_feed_cursor as store_default_feed_cursor
from store_paths import default_findings_dir
from feed_normalization import (
    normalise_entities,
    normalise_provenance,
    normalise_source_check,
    normalise_timestamp,
    normalise_validity,
)

DEFAULT_PROJECT = "REMANENTIA"
DEFAULT_FEED_PATH = Path.home() / "synapse" / "feed.ndjson"
DEFAULT_FEED_CURSOR_NAME = STORE_DEFAULT_FEED_CURSOR_NAME
_MARKER_RE = re.compile(
    r"^\s*(?:\[(?P<bracket>finding|key finding|decision|decision point)\]"
    r"|(?P<label>finding|key finding|decision|decision point))\s*:\s*(?P<statement>.+?)\s*$",
    re.IGNORECASE,
)


ParseFinding = Callable[[Mapping[str, object]], object]
"""Callable that turns a wire finding mapping into a parsed finding."""


@dataclass(frozen=True)
class FeedIngestReport:
    """Outcome of one ``feed.ndjson`` ingest pass.

    Parameters
    ----------
    scanned
        Physical feed rows read after the line cursor.
    candidates
        Rows promoted to finding candidates by the precision-first filter.
    admitted
        Candidates written to the sink after admission.
    skipped
        Well-formed rows ignored because they are not explicit findings or
        decisions.
    rejected
        Malformed JSON rows, unparsable finding candidates, or gate rejections.
    last_seq
        Physical line number stored in the cursor after this pass.
    rejections
        ``(line_number, reason)`` entries for the rejected rows.
    """

    scanned: int
    candidates: int
    admitted: int
    skipped: int
    rejected: int
    last_seq: int
    rejections: tuple[tuple[int, str], ...] = ()

    @property
    def advanced(self) -> bool:
        """Whether the physical feed cursor moved."""
        return self.scanned > 0


def feed_record_to_finding(
    record: Mapping[str, object],
    *,
    line_no: int,
    project: str = DEFAULT_PROJECT,
) -> dict[str, object] | None:
    """Convert one explicit feed record into a Synapse-style finding payload.

    Parameters
    ----------
    record
        Decoded JSON feed row with SYNAPSE relay fields such as ``ty``, ``s``,
        ``p``, ``t``, and ``h``.
    line_no
        Physical feed line number. Used for idempotent provenance references and
        as a stable cursor coordinate.
    project
        Project fallback used when the record does not declare one.

    Returns
    -------
    dict[str, object] or None
        Finding payload ready for ``synapse_channel.Finding.from_dict``. ``None``
        means the row is well-formed but not an explicit memory candidate.
    """
    kind = _text(record.get("ty")).replace("-", "_")
    payload = record.get("p")
    if kind in {"finding", "decision", "decision_point"}:
        return _candidate_from_payload(payload, record=record, line_no=line_no, project=project)
    if isinstance(payload, str):
        marked = _marked_text(payload)
        if marked is None:
            return None
        subkind, statement = marked
        if not statement:
            return None
        return _finding_from_text(
            statement,
            subkind=subkind,
            record=record,
            line_no=line_no,
            project=project,
        )
    return None


def ingest_feed(
    feed_path: str | Path,
    sink: FindingSink,
    cursor: SeqCursor,
    *,
    parse_finding: ParseFinding,
    admit: Callable[[object], DecisionLike],
    project: str = DEFAULT_PROJECT,
    limit: int | None = None,
    admitting_verdicts: Sequence[str] = DEFAULT_ADMITTING_VERDICTS,
) -> FeedIngestReport:
    """Read new feed rows, admit explicit candidates, persist them, advance.

    The cursor is a physical line cursor, not the feed row's ``i`` field. The
    live relay file can contain presence rows whose ``i`` value repeats, so line
    numbers are the only loss-free resume coordinate.
    """
    path = Path(feed_path)
    last = cursor.load()
    high = last
    scanned = 0
    candidates = 0
    admitted = 0
    skipped = 0
    rejections: list[tuple[int, str]] = []
    admitting = set(admitting_verdicts)
    for line_no, raw_line in _iter_new_lines(path, after_line=last, limit=limit):
        scanned += 1
        high = line_no
        try:
            decoded = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            rejections.append((line_no, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(decoded, dict):
            rejections.append((line_no, "feed row must be a JSON object"))
            continue
        candidate = feed_record_to_finding(decoded, line_no=line_no, project=project)
        if candidate is None:
            skipped += 1
            continue
        candidates += 1
        try:
            finding = parse_finding(candidate)
        except Exception as exc:  # noqa: BLE001 — bad feed candidate is rejected, not fatal
            rejections.append((line_no, f"unparsable finding: {exc}"))
            continue
        decision = admit(finding)
        if decision.verdict in admitting:
            sink.write(finding, line_no, decision.verdict)
            admitted += 1
        else:
            rejections.append((line_no, "; ".join(decision.reasons) or decision.verdict))
    if scanned:
        cursor.save(high)
    return FeedIngestReport(
        scanned=scanned,
        candidates=candidates,
        admitted=admitted,
        skipped=skipped,
        rejected=len(rejections),
        last_seq=high,
        rejections=tuple(rejections),
    )


def ingest_from_feed(
    feed_path: str | Path,
    sink: FindingSink,
    cursor_path: str | Path,
    *,
    project: str = DEFAULT_PROJECT,
    limit: int | None = None,
) -> FeedIngestReport:
    """Bind the live ``synapse_channel`` parser/gate and ingest ``feed.ndjson``.

    The import is deferred so the parser-free helpers remain usable without the
    optional ``synapse-channel`` dependency. The sink is injected to keep storage
    policy shared with :mod:`finding_ingest`.
    """
    import synapse_channel as sc

    def parse(raw: Mapping[str, object]) -> object:
        return sc.Finding.from_dict(dict(raw))

    def admit_finding(finding: object) -> DecisionLike:
        return sc.admit(cast(Any, finding))

    return ingest_feed(
        feed_path,
        sink,
        SeqCursor(Path(cursor_path)),
        parse_finding=parse,
        admit=admit_finding,
        project=project,
        limit=limit,
    )


def default_feed_cursor(base: str | Path | None = None) -> Path:
    """Return the default file cursor for ``feed.ndjson`` ingestion."""
    return store_default_feed_cursor(base)


def main() -> int:  # pragma: no cover - CLI/cron entry point
    """Run one feed-ingest pass and print an operator-readable summary."""
    import sys

    feed = Path(os.environ.get("REMANENTIA_SYNAPSE_FEED", str(DEFAULT_FEED_PATH)))
    cursor = Path(os.environ.get("REMANENTIA_FEED_CURSOR", str(default_feed_cursor())))
    report = ingest_from_feed(feed, MarkdownFindingSink(default_findings_dir()), cursor)
    print(
        f"feed-ingest: scanned={report.scanned} candidates={report.candidates} "
        f"admitted={report.admitted} skipped={report.skipped} "
        f"rejected={report.rejected} last_seq={report.last_seq}"
    )
    for line_no, reason in report.rejections:
        print(f"  rejected line={line_no}: {reason}", file=sys.stderr)
    return 0


def _iter_new_lines(
    path: Path,
    *,
    after_line: int,
    limit: int | None,
) -> list[tuple[int, str]]:
    if not path.exists():
        return []
    rows: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if line_no <= after_line:
                continue
            rows.append((line_no, raw_line.rstrip("\n")))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _candidate_from_payload(
    payload: object,
    *,
    record: Mapping[str, object],
    line_no: int,
    project: str,
) -> dict[str, object] | None:
    kind = _text(record.get("ty")).replace("-", "_")
    if isinstance(payload, Mapping):
        default_subkind = "decision" if kind in {"decision", "decision_point"} else "outcome"
        return _finding_from_mapping(
            payload,
            record=record,
            line_no=line_no,
            project=project,
            default_subkind=default_subkind,
        )
    if isinstance(payload, str):
        marked = _marked_text(payload)
        if marked is not None:
            subkind, statement = marked
        else:
            subkind = "decision" if kind in {"decision", "decision_point"} else "outcome"
            statement = payload.strip()
        if not statement:
            return None
        return _finding_from_text(
            statement,
            subkind=subkind,
            record=record,
            line_no=line_no,
            project=project,
        )
    return None


def _finding_from_mapping(
    payload: Mapping[str, object],
    *,
    record: Mapping[str, object],
    line_no: int,
    project: str,
    default_subkind: str,
) -> dict[str, object]:
    timestamp = normalise_timestamp(record.get("t"))
    statement = _text(payload.get("statement"))
    source_ref = _text(payload.get("evidence_ref")) or f"synapse-feed:{line_no}"
    result: dict[str, object] = dict(payload)
    result.setdefault("statement", statement)
    result.setdefault("subkind", default_subkind)
    result.setdefault("evidence_kind", "producer_asserted")
    result.setdefault("claim_status", "bounded_support")
    result.setdefault("freshness", "traceable_unchecked")
    result.setdefault("evidence_ref", source_ref)
    result["provenance"] = _provenance(
        result.get("provenance"),
        record=record,
        line_no=line_no,
        project=project,
    )
    result["validity"] = _validity(result.get("validity"), timestamp=timestamp)
    result.setdefault("lifecycle", "active")
    result.setdefault("supersedes", None)
    result["verified_at_source"] = _source_check(
        result.get("verified_at_source"),
        record=record,
        line_no=line_no,
        project=project,
    )
    result.setdefault("producer_confidence", None)
    result.setdefault("execution_substrate", None)
    result.setdefault("entities", _entities(record, project=project))
    result.setdefault("tags", _tags(default_subkind))
    return result


def _finding_from_text(
    statement: str,
    *,
    subkind: str,
    record: Mapping[str, object],
    line_no: int,
    project: str,
) -> dict[str, object]:
    return _finding_from_mapping(
        {"statement": statement},
        record=record,
        line_no=line_no,
        project=project,
        default_subkind=subkind,
    )


def _marked_text(text: str) -> tuple[str, str] | None:
    match = _MARKER_RE.match(text)
    if match is None:
        return None
    label = (match.group("bracket") or match.group("label") or "").lower()
    subkind = "decision" if "decision" in label else "outcome"
    return subkind, match.group("statement").strip()


def _provenance(
    value: object,
    *,
    record: Mapping[str, object],
    line_no: int,
    project: str,
) -> dict[str, object]:
    return normalise_provenance(
        value,
        record=record,
        line_no=line_no,
        default_project=project,
    )


def _validity(value: object, *, timestamp: float) -> dict[str, object]:
    return normalise_validity(value, fallback_timestamp=timestamp)


def _source_check(
    value: object,
    *,
    record: Mapping[str, object],
    line_no: int,
    project: str,
) -> dict[str, object]:
    return normalise_source_check(
        value,
        record=record,
        line_no=line_no,
        default_project=project,
    )


def _entities(record: Mapping[str, object], *, project: str) -> list[str]:
    return normalise_entities(record, default_project=project)


def _tags(subkind: str) -> list[str]:
    return ["synapse-feed", subkind]


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
