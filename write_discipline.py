# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — ecosystem write-discipline gate

"""Enforce the structural contract every memory write must satisfy.

Remanentia is the memory authority for the fleet, so it owns the contract that
governs what may become retrievable memory — and enforces it as a gate, not as
goodwill. The MS.0 audit found the write side was undisciplined: actor/event
fields 84–99 % empty, three timestamp formats, uncontrolled project vocabulary.
The normalisation layer (:mod:`feed_normalization`) silently *defaults* those
missing fields, which hides the omission rather than catching it — a record with
no producer becomes ``actor="synapse"``, a record with no time becomes ``0.0``.

This module closes that gap. It inspects a raw write *before* defaults are
applied, distinguishes a field the producer genuinely provided from one that
would only be back-filled by a sentinel, and assigns a disposition:

- ``accepted`` — satisfies the contract; safe to normalise and index.
- ``quarantined`` — has usable content but violates discipline; held out of the
  retrievable index (recoverable) and attributed to its producer.
- ``rejected`` — structurally useless (no content); never enters memory.

Every verdict names the producer (``project/actor``), so :class:`DisciplineLedger`
can report which agents write undisciplined memory and the contract can be
enforced across the ecosystem by evidence, not exhortation. The default policy is
lenient (quarantine + attribute) so a rollout can measure the violation surface
without losing data; flip ``strict`` once producers conform to reject at source.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from feed_normalization import normalise_actor, normalise_project, normalise_timestamp

Disposition = Literal["accepted", "quarantined", "rejected"]

# Violation codes — stable identifiers so reports and tests can assert on them.
MISSING_CONTENT = "missing_content"
CONTENT_TOO_SHORT = "content_too_short"
MISSING_PROJECT = "missing_project"
UNCONTROLLED_PROJECT = "uncontrolled_project"
MISSING_ACTOR = "missing_actor"
MISSING_TIMESTAMP = "missing_timestamp"
MISSING_ENTITIES = "missing_entities"


@dataclass(frozen=True)
class FieldMap:
    """Which record keys hold each contract field.

    Memory writes arrive in several shapes (SNN stimuli use ``text``/``source``;
    feed findings use ``statement``/``provenance``). The caller maps its shape
    onto the contract once; the gate stays shape-agnostic. ``project``/``actor``
    accept several candidate keys, tried in order, so one map covers near-shapes.
    """

    content: tuple[str, ...] = ("text", "statement", "content", "summary", "event")
    project: tuple[str, ...] = ("project", "repo")
    actor: tuple[str, ...] = ("actor", "source", "s", "agent")
    timestamp: tuple[str, ...] = ("timestamp", "timestamp_unix", "ts", "t")
    entities: tuple[str, ...] = ("entities",)


_DEFAULT_FIELD_MAP = FieldMap()
"""Shared immutable default field map (frozen, safe to reuse)."""


@dataclass(frozen=True)
class WriteContract:
    """The structural contract a memory write must satisfy to be indexed."""

    require_project: bool = True
    require_actor: bool = True
    require_timestamp: bool = True
    require_entities: bool = False
    min_content_chars: int = 15
    known_projects: frozenset[str] = frozenset()
    strict: bool = False
    field_map: FieldMap = field(default_factory=FieldMap)


_DEFAULT_CONTRACT = WriteContract()
"""Shared immutable default contract (frozen, safe to reuse)."""


@dataclass(frozen=True)
class DisciplineVerdict:
    """Outcome of inspecting one write against the contract."""

    disposition: Disposition
    producer: str  # "PROJECT/actor", normalised, for accountability
    violations: tuple[str, ...]

    def __bool__(self) -> bool:
        """Truthy when the write is admissible (accepted)."""
        return self.disposition == "accepted"


def _first_present(record: Mapping[str, object], keys: Sequence[str]) -> object:
    """Return the first key's value that is present and not None."""
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _nonempty_text(value: object) -> str:
    """Return stripped text for a string value, else empty."""
    return value.strip() if isinstance(value, str) else ""


def _has_real_timestamp(value: object) -> bool:
    """Whether *value* parses to a real (non-sentinel) timestamp."""
    if value is None:
        return False
    return bool(normalise_timestamp(value) > 0.0)


def _has_entities(value: object) -> bool:
    """Whether *value* carries at least one non-empty entity."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence):
        return any(_nonempty_text(item) for item in value)
    return False


def producer_label(record: Mapping[str, object], field_map: FieldMap = _DEFAULT_FIELD_MAP) -> str:
    """Return the normalised ``PROJECT/actor`` producer label for a record."""
    project = normalise_project(_first_present(record, field_map.project))
    actor = normalise_actor(_first_present(record, field_map.actor))
    return f"{project}/{actor}"


def resolve_content(record: Mapping[str, object], field_map: FieldMap = _DEFAULT_FIELD_MAP) -> str:
    """Read a memory record's content via the canonical key order.

    Resolves the canonical ``content`` key and the legacy ``text``/``statement``
    shapes alike, so readers stay correct across the write-key migration: a
    reader that calls this keeps working whether a producer has flipped to
    ``content`` or still writes ``text``.
    """
    return _nonempty_text(_first_present(record, field_map.content))


def inspect_write(
    record: Mapping[str, object],
    *,
    contract: WriteContract = _DEFAULT_CONTRACT,
) -> DisciplineVerdict:
    """Return the discipline verdict for one memory-write record.

    The check runs on the *raw* record so a missing field is caught as a
    violation rather than silently defaulted. Content failures reject the write
    (nothing usable to remember); any other violation quarantines it in lenient
    mode or rejects it in strict mode.
    """
    fm = contract.field_map
    violations: list[str] = []

    content = resolve_content(record, fm)
    content_failed = False
    if not content:
        violations.append(MISSING_CONTENT)
        content_failed = True
    elif len(content) < contract.min_content_chars:
        violations.append(CONTENT_TOO_SHORT)
        content_failed = True

    if contract.require_project:
        raw_project = _nonempty_text(_first_present(record, fm.project))
        if not raw_project:
            violations.append(MISSING_PROJECT)
        elif contract.known_projects:
            slug = normalise_project(raw_project)
            if slug not in contract.known_projects:
                violations.append(UNCONTROLLED_PROJECT)

    if contract.require_actor and not _nonempty_text(_first_present(record, fm.actor)):
        violations.append(MISSING_ACTOR)

    if contract.require_timestamp and not _has_real_timestamp(_first_present(record, fm.timestamp)):
        violations.append(MISSING_TIMESTAMP)

    if contract.require_entities and not _has_entities(_first_present(record, fm.entities)):
        violations.append(MISSING_ENTITIES)

    producer = producer_label(record, fm)
    disposition = _disposition(violations, content_failed=content_failed, strict=contract.strict)
    return DisciplineVerdict(
        disposition=disposition, producer=producer, violations=tuple(violations)
    )


def _disposition(violations: Sequence[str], *, content_failed: bool, strict: bool) -> Disposition:
    """Map a violation set to a disposition under the policy."""
    if not violations:
        return "accepted"
    if content_failed:
        return "rejected"
    return "rejected" if strict else "quarantined"


@dataclass
class ProducerRecord:
    """Per-producer discipline tally for accountability reporting."""

    producer: str
    total: int = 0
    accepted: int = 0
    quarantined: int = 0
    rejected: int = 0
    violations: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def conformance(self) -> float:
        """Fraction of this producer's writes that were accepted."""
        return self.accepted / self.total if self.total else 1.0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the tally."""
        return {
            "producer": self.producer,
            "total": self.total,
            "accepted": self.accepted,
            "quarantined": self.quarantined,
            "rejected": self.rejected,
            "conformance": round(self.conformance(), 4),
            "violations": dict(sorted(self.violations.items())),
        }


class DisciplineLedger:
    """Accumulate verdicts and attribute violations to their producers."""

    def __init__(self) -> None:
        """Start an empty ledger."""
        self._by_producer: dict[str, ProducerRecord] = {}
        self.total = 0
        self.accepted = 0
        self.quarantined = 0
        self.rejected = 0

    def record(self, verdict: DisciplineVerdict) -> None:
        """Tally one verdict into the global and per-producer counters."""
        self.total += 1
        rec = self._by_producer.setdefault(
            verdict.producer, ProducerRecord(producer=verdict.producer)
        )
        rec.total += 1
        if verdict.disposition == "accepted":
            self.accepted += 1
            rec.accepted += 1
        elif verdict.disposition == "quarantined":
            self.quarantined += 1
            rec.quarantined += 1
        else:
            self.rejected += 1
            rec.rejected += 1
        for violation in verdict.violations:
            rec.violations[violation] += 1

    def conformance(self) -> float:
        """Fraction of all recorded writes that were accepted."""
        return self.accepted / self.total if self.total else 1.0

    def worst_producers(self, limit: int = 10) -> list[ProducerRecord]:
        """Return producers with the most non-accepted writes, worst first."""
        ranked = sorted(
            self._by_producer.values(),
            key=lambda r: (r.quarantined + r.rejected, r.total),
            reverse=True,
        )
        return ranked[:limit]

    def as_report(self, *, worst_limit: int = 10) -> dict[str, object]:
        """Return a JSON-serialisable accountability report."""
        return {
            "total": self.total,
            "accepted": self.accepted,
            "quarantined": self.quarantined,
            "rejected": self.rejected,
            "conformance": round(self.conformance(), 4),
            "worst_producers": [r.as_dict() for r in self.worst_producers(worst_limit)],
        }


def audit_records(
    records: Iterable[Mapping[str, object]],
    *,
    contract: WriteContract = _DEFAULT_CONTRACT,
) -> DisciplineLedger:
    """Inspect many records and return the populated accountability ledger."""
    ledger = DisciplineLedger()
    for record in records:
        ledger.record(inspect_write(record, contract=contract))
    return ledger


def build_memory_record(
    content: str,
    project: str,
    actor: str,
    *,
    timestamp: float | None = None,
    entities: Sequence[str] | None = None,
    kind: str | None = None,
    source_ref: str | None = None,
) -> dict[str, object]:
    """Build a canonical, contract-conformant memory record.

    The writer-side complement to :func:`inspect_write`: the easiest way to emit
    a conformant write is to construct it here. The producer must supply real
    ``content``/``project``/``actor`` (a writer that cannot name what, where, and
    who has nothing worth recording) — those raise :class:`ValueError` if empty.
    ``timestamp`` defaults to the wall clock (the writer knows the time, so
    stamping it here is real provenance, not a sentinel default). ``project`` is
    normalised to an uppercase slug and ``actor`` to its controlled role.
    Optional ``entities``/``kind``/``source_ref`` are included only when given.
    """
    import time

    if not content.strip():
        raise ValueError("content is required and must be non-empty")
    if not project.strip():
        raise ValueError("project is required and must be non-empty")
    if not actor.strip():
        raise ValueError("actor is required and must be non-empty")

    record: dict[str, object] = {
        "content": content.strip(),
        "project": normalise_project(project),
        "actor": normalise_actor(actor),
        "timestamp": time.time() if timestamp is None else float(timestamp),
    }
    if entities:
        deduped = [e.strip() for e in entities if e.strip()]
        if deduped:
            record["entities"] = deduped
    if kind and kind.strip():
        record["kind"] = kind.strip()
    if source_ref and source_ref.strip():
        record["source_ref"] = source_ref.strip()
    return record


def load_stimulus_records(directory: object) -> list[Mapping[str, object]]:
    """Load every ``*.json`` stimulus mapping under *directory*.

    Malformed or non-object files are skipped, not raised, so an audit over a
    real firehose directory is robust to the occasional corrupt write.
    """
    import json
    from pathlib import Path

    root = Path(str(directory))
    records: list[Mapping[str, object]] = []
    if not root.exists():
        return records
    for path in sorted(root.glob("*.json")):
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(decoded, Mapping):
            records.append(decoded)
    return records


def main() -> int:  # pragma: no cover - CLI/audit entry point
    """Audit a stimulus directory and print the discipline accountability report."""
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        prog="remanentia-write-discipline",
        description="Audit memory writes against the ecosystem write-discipline contract.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.environ.get("REMANENTIA_STIMULI_DIR", "snn_stimuli"),
        help="stimulus directory to audit (default: $REMANENTIA_STIMULI_DIR or snn_stimuli)",
    )
    parser.add_argument("--require-entities", action="store_true", help="also require entities")
    parser.add_argument(
        "--strict", action="store_true", help="reject violations instead of quarantine"
    )
    parser.add_argument("--worst", type=int, default=15, help="how many worst producers to list")
    args = parser.parse_args()

    contract = WriteContract(require_entities=args.require_entities, strict=args.strict)
    ledger = audit_records(load_stimulus_records(args.directory), contract=contract)
    print(json.dumps(ledger.as_report(worst_limit=args.worst), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
