# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — signed finding envelopes (WS-1 read-side honesty proof)

"""Seal and verify Remanentia findings with the SCPN Studio honesty envelope.

The WS-1 contract signs the claim unit, not the rendered badge. Remanentia's
claim unit is the Synapse ``Finding.as_dict()`` record plus the admission verdict
from :mod:`finding_ingest`; the grade is recomputed by :func:`recall_gate.present`
through :func:`recall_gate.mode_for_record`. A verifier therefore checks the real
platform signature and recomputes the presentation mode from signed finding axes.

The platform seal proves authenticity. This module adds Remanentia's
finding-specific gates after a valid platform verdict: a finding must still be
active, valid at the requested ``as_of`` instant, and absent from any
content-addressed void closure. These gates never turn a forged or stripped
platform verdict into something softer; they only prevent stale or voided signed
findings from rendering as verified.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from typing import TYPE_CHECKING, Any, Protocol, cast

from recall_gate import mode_for_record

if TYPE_CHECKING:  # pragma: no cover
    from scpn_studio_platform.seal import HonestyEnvelope, Keyring, Signer, Verdict

FINDING_SCHEMA = "remanentia.finding.v1"
GRADER_NAME = "remanentia.recall_gate.present"
TimeLike = float | int | str | datetime


class SealDependencyError(RuntimeError):
    """Raised when the optional SCPN Studio seal dependency is unavailable."""


class SupportsAsDict(Protocol):
    """A finding object that exposes the Synapse ``as_dict`` wire form."""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable finding record."""


@dataclass(frozen=True, slots=True)
class FindingSealPolicy:
    """Policy metadata carried in every Remanentia finding envelope.

    Parameters
    ----------
    grader_name
        Stable identifier of the pure grading function used by verifiers.
    grader_version
        Version of the Remanentia grader contract. Defaults to the installed
        ``remanentia`` package version when available.
    exactness_class
        Exactness mode passed through to the platform seal. Finding grades are
        string-valued and recomputed exactly, so ``"bit-exact"`` is the default.
    """

    grader_name: str = GRADER_NAME
    grader_version: str | None = None
    exactness_class: str | Mapping[str, Any] = "bit-exact"

    def grader(self) -> dict[str, str]:
        """Return the platform ``grader`` mapping for this policy."""
        return {
            "name": self.grader_name,
            "version": self.grader_version or _installed_remanentia_version(),
        }


FindingInput = Mapping[str, Any] | SupportsAsDict


@dataclass(frozen=True, slots=True)
class LineageClosureEntry:
    """Audit record for a signed finding superseded by a newer finding.

    Parameters
    ----------
    content_digest
        Signed platform content digest that must no longer verify as current.
    reason
        Human-auditable reason the older finding was superseded.
    superseded_at
        Optional timestamp when the replacement became effective.
    successor_digest
        Optional platform content digest for the replacing finding.
    """

    content_digest: str
    reason: str
    superseded_at: float | None = None
    successor_digest: str | None = None

    def __post_init__(self) -> None:
        """Normalise and validate closure metadata after dataclass creation."""
        object.__setattr__(
            self,
            "content_digest",
            _required_text(self.content_digest, "content_digest"),
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(
            self,
            "successor_digest",
            _optional_text(self.successor_digest, "successor_digest"),
        )

    @classmethod
    def from_mapping(cls, entry: Mapping[str, object]) -> "LineageClosureEntry":
        """Build one closure entry from an external audit mapping.

        Parameters
        ----------
        entry
            Mapping with ``content_digest`` and ``reason`` string fields, plus
            optional ``superseded_at`` and ``successor_digest`` metadata.

        Returns
        -------
        LineageClosureEntry
            Validated lineage closure entry.

        Raises
        ------
        ValueError
            If the entry omits required audit fields or carries an invalid
            timestamp.
        """
        return cls(
            content_digest=_required_text(entry.get("content_digest"), "content_digest"),
            reason=_required_text(entry.get("reason"), "reason"),
            superseded_at=_optional_time(entry.get("superseded_at"), "superseded_at"),
            successor_digest=_optional_text(entry.get("successor_digest"), "successor_digest"),
        )


LineageClosureInput = LineageClosureEntry | Mapping[str, object]


def _installed_remanentia_version() -> str:
    try:
        return str(metadata.version("remanentia"))
    except metadata.PackageNotFoundError:
        return "0.3.1"


def _seal_api() -> tuple[Any, Any, Any]:
    try:
        from scpn_studio_platform.seal import Verdict, seal, verify
    except ModuleNotFoundError as exc:
        raise SealDependencyError(
            'finding envelopes require the optional seal extra: pip install "remanentia[seal]"'
        ) from exc
    return seal, verify, Verdict


def _record_from_finding(finding: FindingInput) -> dict[str, Any]:
    if hasattr(finding, "as_dict"):
        return dict(finding.as_dict())
    return dict(finding)


def finding_unit(finding: FindingInput, *, admission_verdict: str = "accept") -> dict[str, Any]:
    """Return the signed Remanentia finding unit.

    Parameters
    ----------
    finding
        A Synapse ``Finding`` object or an equivalent mapping.
    admission_verdict
        Verdict assigned by the admission gate. ``"accept"`` is the path to
        ``admitted``; ``"floor"`` and all other values remain boundary-limited.

    Returns
    -------
    dict[str, Any]
        JSON-compatible unit consumed by the SCPN Studio seal.
    """
    return {
        "schema": FINDING_SCHEMA,
        "finding": _record_from_finding(finding),
        "admission_verdict": admission_verdict,
    }


def regrade_finding_unit(unit: Mapping[str, Any]) -> str:
    """Recompute the presentation mode implied by a signed finding unit.

    Parameters
    ----------
    unit
        A unit produced by :func:`finding_unit`.

    Returns
    -------
    str
        ``"validated"``, ``"boundary"``, or ``"refuted"``.

    Raises
    ------
    ValueError
        If the signed unit is not a Remanentia finding unit.
    """
    if unit.get("schema") != FINDING_SCHEMA:
        raise ValueError(f"expected schema {FINDING_SCHEMA!r}")
    finding = unit.get("finding")
    if not isinstance(finding, Mapping):
        raise ValueError("finding unit must carry a finding mapping")
    record = dict(finding)
    record["verdict"] = str(unit.get("admission_verdict", "floor"))
    return str(mode_for_record(record))


def seal_finding(
    finding: FindingInput,
    *,
    signer: "Signer",
    admission_verdict: str = "accept",
    policy: FindingSealPolicy | None = None,
) -> "HonestyEnvelope":
    """Seal one finding into a verifiable honesty envelope.

    Parameters
    ----------
    finding
        A Synapse ``Finding`` object or an equivalent mapping.
    signer
        SCPN Studio Platform signer; the Ed25519 reference signer and later
        ML-DSA signers satisfy the same protocol.
    admission_verdict
        Admission result for the finding.
    policy
        Optional grader/exactness metadata.

    Returns
    -------
    HonestyEnvelope
        Platform envelope with ``verifiability_mode="recompute"``.
    """
    seal, _, _ = _seal_api()
    active_policy = policy or FindingSealPolicy()
    return cast(
        "HonestyEnvelope",
        seal(
            finding_unit(finding, admission_verdict=admission_verdict),
            signer=signer,
            grader=active_policy.grader(),
            verifiability_mode="recompute",
            exactness_class=active_policy.exactness_class,
        ),
    )


def lineage_closure_digests(entries: Iterable[LineageClosureInput]) -> frozenset[str]:
    """Return all superseded content digests from a lineage closure.

    Parameters
    ----------
    entries
        Supersession entries carrying both digest and audit reason metadata.

    Returns
    -------
    frozenset[str]
        Content digests that must be downgraded from verified to ungraded.

    Raises
    ------
    ValueError
        If any closure entry is missing a digest, reason, or parseable timestamp.
    """
    return frozenset(_lineage_entry(entry).content_digest for entry in entries)


def verify_finding(
    envelope: Mapping[str, Any] | None,
    rendered_grade: str | None,
    *,
    keyring: "Keyring",
    as_of: TimeLike | None = None,
    voided_digests: Iterable[str] = (),
    supersession_closure: Iterable[LineageClosureInput] = (),
) -> "Verdict":
    """Verify a rendered finding grade and apply Remanentia recall gates.

    Parameters
    ----------
    envelope
        Platform honesty-envelope wire mapping, or ``None``.
    rendered_grade
        Grade displayed by the caller, if any.
    keyring
        Trust anchor for platform signature verification.
    as_of
        Optional query instant. When provided, the signed finding's validity
        interval must contain it.
    voided_digests
        Content digests voided by an upstream retraction closure.
    supersession_closure
        Audit entries for findings superseded by newer signed content. Each
        entry must include the superseded digest and the reason it was replaced.

    Returns
    -------
    Verdict
        Platform verdict, further downgraded to ``UNGRADED`` when an otherwise
        verified finding is stale, retracted, superseded, or voided.
    """
    _, verify, verdict = _seal_api()
    base = cast(
        "Verdict", verify(envelope, rendered_grade, keyring=keyring, regrade=regrade_finding_unit)
    )
    if base is not cast("Verdict", verdict.VERIFIED):
        return base
    if envelope is None:
        return cast("Verdict", verdict.UNGRADED)
    unit = envelope.get("unit")
    if not isinstance(unit, Mapping):
        return cast("Verdict", verdict.FORGED)
    content_digest = str(envelope.get("content_digest", ""))
    try:
        superseded_digests = lineage_closure_digests(supersession_closure)
    except ValueError:
        return cast("Verdict", verdict.UNGRADED)
    if content_digest in set(voided_digests) or content_digest in superseded_digests:
        return cast("Verdict", verdict.UNGRADED)
    if not _lifecycle_is_active(unit):
        return cast("Verdict", verdict.UNGRADED)
    if as_of is not None and not _valid_at(unit, as_of):
        return cast("Verdict", verdict.UNGRADED)
    return base


def _lifecycle_is_active(unit: Mapping[str, Any]) -> bool:
    finding = unit.get("finding")
    if not isinstance(finding, Mapping):
        return False
    return str(finding.get("lifecycle", "")).replace("_", "-").lower() == "active"


def _lineage_entry(entry: LineageClosureInput) -> LineageClosureEntry:
    if isinstance(entry, LineageClosureEntry):
        return entry
    return LineageClosureEntry.from_mapping(entry)


def _valid_at(unit: Mapping[str, Any], as_of: TimeLike) -> bool:
    finding = unit.get("finding")
    if not isinstance(finding, Mapping):
        return False
    validity = finding.get("validity")
    if not isinstance(validity, Mapping):
        return False
    instant = _coerce_time(as_of)
    start = _coerce_time(validity.get("valid_from"))
    stop = _coerce_time(validity.get("valid_to"))
    if instant is None:
        return False
    if start is not None and instant < start:
        return False
    return not (stop is not None and instant > stop)


def _required_text(value: object, field_name: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    raise ValueError(f"lineage closure {field_name} must be a non-empty string")


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _optional_time(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = _coerce_time(value)
    if parsed is None:
        raise ValueError(f"lineage closure {field_name} must be a parseable time")
    return parsed


def _coerce_time(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return aware.timestamp()
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None
