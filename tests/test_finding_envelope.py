# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for signed finding envelopes

"""Tests for :mod:`finding_envelope`.

The suite uses the real ``scpn_studio_platform.seal`` Ed25519 signer, keyring,
and verifier so the WS-1 boundary is tested through the production signature
contract. It does not replace the platform's seal tests; it verifies
Remanentia's finding-specific unit shape, recall-gate regrade function, and
as-of / lifecycle / void-closure gates.
"""

from __future__ import annotations

import builtins
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any

import pytest

if sys.version_info < (3, 12):
    pytest.skip(
        "scpn-studio-platform 0.10 requires Python 3.12; finding envelopes are optional",
        allow_module_level=True,
    )

from scpn_studio_platform.seal import Ed25519Signer, Keyring, Verdict

import finding_envelope as finding_envelope_module
from finding_envelope import (
    FINDING_SCHEMA,
    FindingSealPolicy,
    LineageClosureEntry,
    SealDependencyError,
    finding_unit,
    lineage_closure_digests,
    regrade_finding_unit,
    seal_finding,
    verify_finding,
)
from recall_gate import BOUNDARY, REFUTED, VALIDATED


def _finding(**overrides: Any) -> dict[str, Any]:
    """Return a realistic Synapse finding payload for envelope tests."""
    record: dict[str, Any] = {
        "statement": "vector-index reuse prevented a full embedding rebuild",
        "subkind": "outcome",
        "evidence_kind": "measured",
        "claim_status": "reference_validated",
        "freshness": "verified_at_source",
        "evidence_ref": "commit:0d92a47",
        "provenance": {
            "project": "REMANENTIA",
            "actor": "REMANENTIA",
            "session": "s1",
            "source_event_seq": 7,
            "ts": 10.0,
        },
        "validity": {"valid_from": 10.0, "valid_to": 20.0, "observed_at": 10.0},
        "lifecycle": "active",
        "supersedes": None,
        "verified_at_source": {
            "checked_this_session": True,
            "source_ref": "commit:0d92a47",
            "by": "REMANENTIA",
            "at": 10.0,
        },
        "producer_confidence": None,
        "execution_substrate": None,
        "entities": ["vector_index"],
        "tags": ["memory"],
    }
    record.update(overrides)
    return record


class _FindingObject:
    """Object form matching ``synapse_channel.Finding.as_dict``."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        self._record = dict(record)

    def as_dict(self) -> dict[str, Any]:
        """Return the finding wire record."""
        return dict(self._record)


def _signer_and_keyring() -> tuple[Ed25519Signer, Keyring]:
    """Return a real platform signer and trust keyring."""
    signer = Ed25519Signer.generate("remanentia:k1")
    keyring = Keyring()
    keyring.add(signer.key_id, signer.verifier())
    return signer, keyring


class _VerdictDouble:
    """Tiny verdict namespace for unreachable platform-verified edge states."""

    VERIFIED = object()
    FORGED = object()
    UNGRADED = object()


def _always_verified_api() -> tuple[
    Callable[..., object], Callable[..., object], type[_VerdictDouble]
]:
    """Return a seal API double whose verifier always reports ``VERIFIED``."""

    def unused_seal(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("seal is not used by verifier edge tests")

    def always_verified(
        _envelope: Mapping[str, Any] | None,
        _rendered_grade: str | None,
        **_kwargs: object,
    ) -> object:
        return _VerdictDouble.VERIFIED

    return unused_seal, always_verified, _VerdictDouble


class TestFindingUnit:
    def test_finding_unit_wraps_as_dict_with_schema_and_admission(self) -> None:
        unit = finding_unit(_FindingObject(_finding()), admission_verdict="accept")
        assert unit["schema"] == FINDING_SCHEMA
        assert unit["finding"]["statement"] == _finding()["statement"]
        assert unit["admission_verdict"] == "accept"

    def test_regrade_finding_unit_uses_recall_gate_axes(self) -> None:
        assert (
            regrade_finding_unit(finding_unit(_finding(), admission_verdict="accept")) == VALIDATED
        )
        assert (
            regrade_finding_unit(
                finding_unit(_finding(claim_status="bounded_support"), admission_verdict="accept")
            )
            == BOUNDARY
        )
        assert (
            regrade_finding_unit(
                finding_unit(_finding(evidence_kind="falsified"), admission_verdict="accept")
            )
            == REFUTED
        )

    def test_regrade_rejects_wrong_schema(self) -> None:
        with pytest.raises(ValueError, match="schema"):
            regrade_finding_unit({"schema": "wrong", "finding": _finding()})

    def test_regrade_rejects_missing_finding_mapping(self) -> None:
        with pytest.raises(ValueError, match="finding mapping"):
            regrade_finding_unit({"schema": FINDING_SCHEMA, "finding": None})


class TestSealDependencies:
    def test_policy_uses_distribution_version_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing_version(_name: str) -> str:
            raise importlib_metadata.PackageNotFoundError

        monkeypatch.setattr(importlib_metadata, "version", missing_version)

        assert FindingSealPolicy().grader()["version"] == "0.3.1"

    def test_missing_optional_seal_dependency_has_actionable_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_import = builtins.__import__

        def blocked_import(
            name: str,
            globals_: dict[str, Any] | None = None,
            locals_: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name == "scpn_studio_platform.seal":
                raise ModuleNotFoundError(name)
            return real_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        with pytest.raises(SealDependencyError, match="remanentia\\[seal\\]"):
            finding_envelope_module._seal_api()


class TestSealAndVerify:
    def test_signed_finding_verifies_with_matching_rendered_grade(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer, admission_verdict="accept")

        assert envelope.unit["schema"] == FINDING_SCHEMA
        assert envelope.grader["name"] == "remanentia.recall_gate.present"
        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=keyring, as_of=15.0)
            is Verdict.VERIFIED
        )

    def test_rendered_grade_disagreement_is_forged(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer, admission_verdict="accept")

        assert (
            verify_finding(envelope.to_dict(), BOUNDARY, keyring=keyring, as_of=15.0)
            is Verdict.FORGED
        )

    def test_missing_envelope_is_stripped_when_grade_is_rendered(self) -> None:
        _, keyring = _signer_and_keyring()
        assert verify_finding(None, VALIDATED, keyring=keyring) is Verdict.STRIPPED

    def test_unknown_key_is_forged(self) -> None:
        signer, _ = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer, admission_verdict="accept")

        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=Keyring(), as_of=15.0)
            is Verdict.FORGED
        )

    def test_stale_as_of_window_is_ungraded_not_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer, admission_verdict="accept")

        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=keyring, as_of=25.0)
            is Verdict.UNGRADED
        )

    def test_before_validity_window_is_ungraded_not_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer, admission_verdict="accept")

        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=keyring, as_of=5.0)
            is Verdict.UNGRADED
        )

    def test_iso_and_datetime_validity_windows_are_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(
            _finding(validity={"valid_from": "1970-01-01T00:00:10Z", "valid_to": None}),
            signer=signer,
        )

        assert (
            verify_finding(
                envelope.to_dict(),
                VALIDATED,
                keyring=keyring,
                as_of=datetime.fromtimestamp(15, tz=timezone.utc),
            )
            is Verdict.VERIFIED
        )

    def test_unparsable_as_of_is_ungraded_not_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer)

        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=keyring, as_of="")
            is Verdict.UNGRADED
        )

    def test_invalid_iso_as_of_is_ungraded_not_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer)

        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=keyring, as_of="not-a-time")
            is Verdict.UNGRADED
        )

    def test_missing_validity_mapping_is_ungraded_not_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(validity=None), signer=signer)

        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=keyring, as_of=15.0)
            is Verdict.UNGRADED
        )

    def test_non_scalar_validity_bounds_are_open_ended(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(
            _finding(validity={"valid_from": ["unknown"], "valid_to": None}),
            signer=signer,
        )

        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=keyring, as_of=15.0)
            is Verdict.VERIFIED
        )

    def test_retracted_lifecycle_is_ungraded_not_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(lifecycle="retracted"), signer=signer)

        assert (
            verify_finding(envelope.to_dict(), VALIDATED, keyring=keyring, as_of=15.0)
            is Verdict.UNGRADED
        )

    def test_voided_content_digest_is_ungraded_not_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer)

        assert (
            verify_finding(
                envelope.to_dict(),
                VALIDATED,
                keyring=keyring,
                as_of=15.0,
                voided_digests={envelope.content_digest},
            )
            is Verdict.UNGRADED
        )

    def test_supersession_closure_digest_is_ungraded_not_verified(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer)

        assert (
            verify_finding(
                envelope.to_dict(),
                VALIDATED,
                keyring=keyring,
                as_of=15.0,
                supersession_closure=(
                    LineageClosureEntry(
                        content_digest=envelope.content_digest,
                        reason="newer source measurement superseded this finding",
                        superseded_at=21.0,
                        successor_digest="sha256:newer",
                    ),
                ),
            )
            is Verdict.UNGRADED
        )

    def test_unrelated_supersession_closure_keeps_verified_finding(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer)

        assert (
            verify_finding(
                envelope.to_dict(),
                VALIDATED,
                keyring=keyring,
                as_of=15.0,
                supersession_closure=(
                    {
                        "content_digest": "sha256:unrelated",
                        "reason": "different finding was replaced",
                        "superseded_at": 21.0,
                        "successor_digest": "sha256:newer",
                    },
                ),
            )
            is Verdict.VERIFIED
        )

    def test_malformed_supersession_closure_fails_closed_to_ungraded(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer)

        assert (
            verify_finding(
                envelope.to_dict(),
                VALIDATED,
                keyring=keyring,
                as_of=15.0,
                supersession_closure=(
                    {
                        "content_digest": envelope.content_digest,
                        "reason": "",
                    },
                ),
            )
            is Verdict.UNGRADED
        )

    def test_lineage_closure_requires_digest_and_reason(self) -> None:
        assert lineage_closure_digests(
            (
                {
                    "content_digest": "sha256:older",
                    "reason": "newer source measurement superseded this finding",
                    "superseded_at": "1970-01-01T00:00:21Z",
                },
            )
        ) == frozenset({"sha256:older"})

        with pytest.raises(ValueError, match="reason"):
            lineage_closure_digests(({"content_digest": "sha256:older", "reason": ""},))

        with pytest.raises(ValueError, match="content_digest"):
            lineage_closure_digests(({"reason": "newer source measurement superseded it"},))

    def test_lineage_closure_allows_missing_superseded_at_but_rejects_invalid_time(
        self,
    ) -> None:
        assert lineage_closure_digests(
            (
                {
                    "content_digest": "sha256:older",
                    "reason": "newer source measurement superseded this finding",
                },
            )
        ) == frozenset({"sha256:older"})

        with pytest.raises(ValueError, match="superseded_at"):
            lineage_closure_digests(
                (
                    {
                        "content_digest": "sha256:older",
                        "reason": "newer source measurement superseded this finding",
                        "superseded_at": "not-a-time",
                    },
                )
            )

    def test_floored_admission_verifies_as_boundary(self) -> None:
        signer, keyring = _signer_and_keyring()
        envelope = seal_finding(_finding(), signer=signer, admission_verdict="floor")

        assert (
            verify_finding(envelope.to_dict(), BOUNDARY, keyring=keyring, as_of=15.0)
            is Verdict.VERIFIED
        )

    def test_custom_policy_sets_grader_version(self) -> None:
        signer, _ = _signer_and_keyring()
        policy = FindingSealPolicy(grader_version="9.9-test")
        envelope = seal_finding(_finding(), signer=signer, policy=policy)
        assert envelope.grader["version"] == "9.9-test"

    def test_platform_verified_missing_envelope_is_ungraded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(finding_envelope_module, "_seal_api", _always_verified_api)

        assert verify_finding(None, VALIDATED, keyring=Keyring()) is _VerdictDouble.UNGRADED

    def test_platform_verified_malformed_unit_is_forged(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(finding_envelope_module, "_seal_api", _always_verified_api)

        assert (
            verify_finding({"unit": None, "content_digest": "d1"}, VALIDATED, keyring=Keyring())
            is _VerdictDouble.FORGED
        )

    def test_platform_verified_missing_finding_is_ungraded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(finding_envelope_module, "_seal_api", _always_verified_api)

        assert (
            verify_finding(
                {"unit": {"finding": None}, "content_digest": "d1"},
                VALIDATED,
                keyring=Keyring(),
            )
            is _VerdictDouble.UNGRADED
        )

    def test_validity_gate_rejects_missing_finding_mapping(self) -> None:
        assert finding_envelope_module._valid_at({"finding": None}, 15.0) is False
