# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for write_discipline

from __future__ import annotations

from pathlib import Path

import pytest

from write_discipline import (
    CONTENT_TOO_SHORT,
    MISSING_ACTOR,
    MISSING_CONTENT,
    MISSING_ENTITIES,
    MISSING_PROJECT,
    MISSING_TIMESTAMP,
    UNCONTROLLED_PROJECT,
    DisciplineLedger,
    DisciplineVerdict,
    ProducerRecord,
    WriteContract,
    _first_present,
    _has_entities,
    _has_real_timestamp,
    _nonempty_text,
    audit_records,
    build_memory_record,
    inspect_write,
    load_stimulus_records,
    producer_label,
    resolve_content,
)

# A well-disciplined stimulus that satisfies the default contract.
_GOOD = {
    "text": "Hardened SCPN order-parameter accelerator boundaries. Commit 4149dc33.",
    "source": "codex",
    "project": "SCPN-PHASE-ORCHESTRATOR",
    "timestamp": 1782700000,
}


# ─── primitives ───────────────────────────────────────────────────────


class TestPrimitives:
    def test_first_present_found(self) -> None:
        assert _first_present({"b": 2, "a": 1}, ("a", "b")) == 1

    def test_first_present_skips_none(self) -> None:
        assert _first_present({"a": None, "b": 2}, ("a", "b")) == 2

    def test_first_present_absent(self) -> None:
        assert _first_present({}, ("a",)) is None

    def test_nonempty_text(self) -> None:
        assert _nonempty_text("  x ") == "x"
        assert _nonempty_text(42) == ""

    def test_has_real_timestamp(self) -> None:
        assert _has_real_timestamp(1782700000) is True
        assert _has_real_timestamp(None) is False
        assert _has_real_timestamp(0) is False  # 0.0 sentinel = no real time

    def test_has_entities(self) -> None:
        assert _has_entities("REMANENTIA") is True
        assert _has_entities("  ") is False
        assert _has_entities(["a"]) is True
        assert _has_entities([" ", ""]) is False
        assert _has_entities(None) is False


class TestProducerLabel:
    def test_label_from_stimulus(self) -> None:
        assert producer_label(_GOOD) == "SCPN-PHASE-ORCHESTRATOR/codex"

    def test_label_defaults_when_absent(self) -> None:
        assert producer_label({}) == "REMANENTIA/synapse"


class TestResolveContent:
    def test_canonical_content_key(self) -> None:
        assert resolve_content({"content": "the canonical write"}) == "the canonical write"

    def test_legacy_text_key(self) -> None:
        assert resolve_content({"text": "the legacy write"}) == "the legacy write"

    def test_legacy_statement_key(self) -> None:
        assert resolve_content({"statement": "a feed finding"}) == "a feed finding"

    def test_absent_or_blank_is_empty(self) -> None:
        assert resolve_content({}) == ""
        assert resolve_content({"text": "   "}) == ""


# ─── inspect_write ────────────────────────────────────────────────────


class TestInspectWrite:
    def test_good_accepted(self) -> None:
        v = inspect_write(_GOOD)
        assert v.disposition == "accepted"
        assert v.violations == ()
        assert bool(v) is True

    def test_missing_content_rejected(self) -> None:
        v = inspect_write({"source": "codex", "project": "REMANENTIA", "timestamp": 1782700000})
        assert v.disposition == "rejected"
        assert MISSING_CONTENT in v.violations
        assert bool(v) is False

    def test_content_too_short_rejected(self) -> None:
        v = inspect_write({**_GOOD, "text": "short"})
        assert v.disposition == "rejected"
        assert CONTENT_TOO_SHORT in v.violations

    def test_missing_timestamp_quarantined(self) -> None:
        rec = {k: val for k, val in _GOOD.items() if k != "timestamp"}
        v = inspect_write(rec)
        assert v.disposition == "quarantined"
        assert v.violations == (MISSING_TIMESTAMP,)

    def test_missing_actor_and_project_quarantined(self) -> None:
        v = inspect_write({"text": _GOOD["text"], "timestamp": 1782700000})
        assert v.disposition == "quarantined"
        assert MISSING_ACTOR in v.violations
        assert MISSING_PROJECT in v.violations

    def test_strict_rejects_discipline_violation(self) -> None:
        rec = {k: val for k, val in _GOOD.items() if k != "timestamp"}
        v = inspect_write(rec, contract=WriteContract(strict=True))
        assert v.disposition == "rejected"
        assert v.violations == (MISSING_TIMESTAMP,)

    def test_known_projects_allowlist_uncontrolled(self) -> None:
        contract = WriteContract(known_projects=frozenset({"REMANENTIA"}))
        v = inspect_write(_GOOD, contract=contract)
        assert UNCONTROLLED_PROJECT in v.violations
        assert v.disposition == "quarantined"

    def test_known_projects_allowlist_controlled(self) -> None:
        contract = WriteContract(known_projects=frozenset({"SCPN-PHASE-ORCHESTRATOR"}))
        v = inspect_write(_GOOD, contract=contract)
        assert v.disposition == "accepted"

    def test_require_entities_missing(self) -> None:
        v = inspect_write(_GOOD, contract=WriteContract(require_entities=True))
        assert v.violations == (MISSING_ENTITIES,)

    def test_require_entities_present(self) -> None:
        rec = {**_GOOD, "entities": ["SCPN"]}
        v = inspect_write(rec, contract=WriteContract(require_entities=True))
        assert v.disposition == "accepted"

    def test_relaxed_contract_accepts_minimal(self) -> None:
        contract = WriteContract(
            require_project=False, require_actor=False, require_timestamp=False
        )
        v = inspect_write({"text": "a sufficiently long statement of content"}, contract=contract)
        assert v.disposition == "accepted"


# ─── ProducerRecord ───────────────────────────────────────────────────


class TestProducerRecord:
    def test_conformance_empty(self) -> None:
        assert ProducerRecord(producer="x").conformance() == 1.0

    def test_conformance_partial(self) -> None:
        rec = ProducerRecord(producer="x", total=4, accepted=1)
        assert rec.conformance() == 0.25

    def test_as_dict(self) -> None:
        rec = ProducerRecord(producer="x", total=2, accepted=1, quarantined=1)
        rec.violations["missing_timestamp"] += 1
        out = rec.as_dict()
        assert out["producer"] == "x"
        assert out["conformance"] == 0.5
        assert out["violations"] == {"missing_timestamp": 1}


# ─── DisciplineLedger ─────────────────────────────────────────────────


class TestDisciplineLedger:
    def test_records_all_dispositions(self) -> None:
        led = DisciplineLedger()
        led.record(DisciplineVerdict("accepted", "P/a", ()))
        led.record(DisciplineVerdict("quarantined", "P/a", (MISSING_TIMESTAMP,)))
        led.record(DisciplineVerdict("rejected", "Q/b", (MISSING_CONTENT,)))
        assert (led.total, led.accepted, led.quarantined, led.rejected) == (3, 1, 1, 1)
        assert abs(led.conformance() - 1 / 3) < 1e-9

    def test_conformance_empty(self) -> None:
        assert DisciplineLedger().conformance() == 1.0

    def test_worst_producers_ordering(self) -> None:
        led = DisciplineLedger()
        for _ in range(3):
            led.record(DisciplineVerdict("quarantined", "noisy/x", (MISSING_TIMESTAMP,)))
        led.record(DisciplineVerdict("accepted", "clean/y", ()))
        worst = led.worst_producers(limit=1)
        assert len(worst) == 1
        assert worst[0].producer == "noisy/x"

    def test_as_report(self) -> None:
        led = DisciplineLedger()
        led.record(DisciplineVerdict("quarantined", "noisy/x", (MISSING_ACTOR,)))
        report = led.as_report(worst_limit=5)
        assert report["total"] == 1
        assert report["quarantined"] == 1
        assert isinstance(report["worst_producers"], list)


class TestAuditRecords:
    def test_audit_aggregates(self) -> None:
        sloppy = {k: v for k, v in _GOOD.items() if k != "timestamp"}
        led = audit_records([_GOOD, sloppy, {"text": "x"}])
        assert led.total == 3
        assert led.accepted == 1
        assert led.quarantined == 1
        assert led.rejected == 1


class TestBuildMemoryRecord:
    def test_full_record_normalised(self) -> None:
        rec = build_memory_record(
            "  a real statement of content  ",
            "scpn-quantum-control",
            "SC-NEUROCORE/codex-7f3a",
            timestamp=1782700000,
            entities=["SCPN", " ", "Kuramoto"],
            kind="finding",
            source_ref="abc123",
        )
        assert rec["content"] == "a real statement of content"
        assert rec["project"] == "SCPN-QUANTUM-CONTROL"
        assert rec["actor"] == "codex"  # role, hex suffix stripped
        assert rec["timestamp"] == 1782700000.0
        assert rec["entities"] == ["SCPN", "Kuramoto"]
        assert rec["kind"] == "finding"
        assert rec["source_ref"] == "abc123"

    def test_minimal_stamps_timestamp_and_omits_optionals(self) -> None:
        rec = build_memory_record("a real statement of content", "REMANENTIA", "claude")
        ts = rec["timestamp"]
        assert isinstance(ts, float) and ts > 0  # wall-clock stamped
        assert "entities" not in rec
        assert "kind" not in rec
        assert "source_ref" not in rec

    def test_empty_optionals_are_dropped(self) -> None:
        rec = build_memory_record(
            "a real statement of content",
            "REMANENTIA",
            "claude",
            entities=[" ", ""],
            kind="  ",
            source_ref="  ",
        )
        assert "entities" not in rec
        assert "kind" not in rec
        assert "source_ref" not in rec

    def test_built_record_passes_the_gate(self) -> None:
        rec = build_memory_record("a real statement of content", "REMANENTIA", "codex")
        assert inspect_write(rec).disposition == "accepted"

    def test_missing_content_raises(self) -> None:
        with pytest.raises(ValueError, match="content"):
            build_memory_record("   ", "REMANENTIA", "codex")

    def test_missing_project_raises(self) -> None:
        with pytest.raises(ValueError, match="project"):
            build_memory_record("a real statement of content", "  ", "codex")

    def test_missing_actor_raises(self) -> None:
        with pytest.raises(ValueError, match="actor"):
            build_memory_record("a real statement of content", "REMANENTIA", "")


class TestLoadStimulusRecords:
    def test_missing_directory(self, tmp_path: Path) -> None:
        assert load_stimulus_records(tmp_path / "nope") == []

    def test_loads_valid_skips_malformed_and_nonobject(self, tmp_path: Path) -> None:
        (tmp_path / "good.json").write_text('{"text": "x", "source": "codex"}', encoding="utf-8")
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
        records = load_stimulus_records(tmp_path)
        assert len(records) == 1
        assert records[0]["source"] == "codex"
