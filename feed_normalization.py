# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — SYNAPSE feed identity and timestamp normalisation

"""Normalise SYNAPSE feed metadata before it becomes retrievable memory.

The live feed carries multiple shapes for the same concepts: sender names can be
bare projects, project-local worker identities, or presence rows; timestamps can
be seconds, milliseconds, numeric strings, or ISO-8601 strings. This module
collapses those inputs into a controlled project/actor/timestamp vocabulary so
downstream finding ingestion does not index accidental process IDs or mixed time
formats as durable memory facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import re

DEFAULT_PROJECT = "REMANENTIA"
"""Fallback project used when a feed row carries no usable project identity."""

_PROJECT_RE = re.compile(r"[^A-Z0-9]+")
_ACTOR_RE = re.compile(r"[^a-z0-9]+")
_HEX_SUFFIX_RE = re.compile(r"-[0-9a-f]{4,}$")
_PRESENCE_PREFIX = "synapse-presence@"


@dataclass(frozen=True)
class FeedIdentity:
    """Canonical identity extracted from one feed row.

    Parameters
    ----------
    project
        Controlled uppercase project identifier.
    actor
        Controlled lower-kebab actor identifier.
    session
        Stable lower-kebab session or feed-line identifier.
    source_event_seq
        Feed event sequence when valid, otherwise the physical line number.
    timestamp
        UNIX timestamp in seconds.
    """

    project: str
    actor: str
    session: str
    source_event_seq: int
    timestamp: float


def normalise_project(value: object, *, default_project: str = DEFAULT_PROJECT) -> str:
    """Return a controlled uppercase project identifier.

    Parameters
    ----------
    value
        Sender, explicit project, or presence identity from the feed.
    default_project
        Fallback project when ``value`` has no usable project segment.

    Returns
    -------
    str
        Uppercase slug such as ``"REMANENTIA"`` or ``"SCPN-FUSION-CORE"``.
    """

    text = _project_segment(_text(value))
    if not text:
        text = default_project
    slug = _PROJECT_RE.sub("-", text.upper()).strip("-")
    return slug or _PROJECT_RE.sub("-", default_project.upper()).strip("-") or DEFAULT_PROJECT


def normalise_actor(value: object) -> str:
    """Return a controlled lower-kebab actor identifier.

    Project-local random suffixes such as ``worker-7f3a`` collapse to the stable
    actor role ``worker``; presence rows collapse to ``system``.
    """

    text = _text(value)
    if not text:
        return "synapse"
    lowered = text.lower()
    if lowered.startswith(_PRESENCE_PREFIX):
        return "system"
    if text.upper() == "CEO":
        return "operator"
    if "/" in text:
        text = text.split("/", 1)[1]
    text = _HEX_SUFFIX_RE.sub("", text.lower())
    slug = _ACTOR_RE.sub("-", text).strip("-")
    return slug or "synapse"


def normalise_timestamp(value: object) -> float:
    """Return a feed timestamp as UNIX seconds.

    Parameters
    ----------
    value
        Seconds, milliseconds, numeric string, ISO-8601 string, or missing
        timestamp from a feed row.

    Returns
    -------
    float
        UNIX seconds. Invalid values return ``0.0`` so malformed feed metadata
        degrades to an explicit sentinel instead of raising during ingestion.
    """

    raw = _numeric_timestamp(value)
    if raw is None and isinstance(value, str):
        raw = _iso_timestamp(value)
    if raw is None:
        return 0.0
    return raw / 1000.0 if raw > 10_000_000_000 else raw


def normalise_feed_identity(
    record: Mapping[str, object],
    *,
    line_no: int,
    default_project: str = DEFAULT_PROJECT,
) -> FeedIdentity:
    """Return canonical identity fields for one decoded feed row."""

    project_source = record.get("project") or record.get("repo") or record.get("s")
    return FeedIdentity(
        project=normalise_project(project_source, default_project=default_project),
        actor=normalise_actor(record.get("actor") or record.get("s")),
        session=normalise_session(record.get("h"), line_no=line_no),
        source_event_seq=normalise_sequence(record.get("i"), default=line_no),
        timestamp=normalise_timestamp(record.get("t")),
    )


def normalise_session(value: object, *, line_no: int) -> str:
    """Return a controlled session identifier for provenance."""

    text = _text(value) or f"feed-line-{line_no}"
    slug = _ACTOR_RE.sub("-", text.lower()).strip("-")
    return slug or f"feed-line-{line_no}"


def normalise_sequence(value: object, *, default: int) -> int:
    """Return a feed sequence integer, rejecting booleans and invalid strings."""

    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def normalise_provenance(
    value: object,
    *,
    record: Mapping[str, object],
    line_no: int,
    default_project: str = DEFAULT_PROJECT,
) -> dict[str, object]:
    """Return provenance with controlled project, actor, sequence, and time."""

    identity = normalise_feed_identity(record, line_no=line_no, default_project=default_project)
    base = dict(value) if isinstance(value, Mapping) else {}
    base.update(
        {
            "project": normalise_project(base.get("project") or identity.project),
            "actor": normalise_actor(base.get("actor") or identity.actor),
            "session": normalise_session(base.get("session") or identity.session, line_no=line_no),
            "source_event_seq": normalise_sequence(
                base.get("source_event_seq"),
                default=identity.source_event_seq,
            ),
            "ts": _timestamp_or_default(base.get("ts"), identity.timestamp),
        }
    )
    return base


def normalise_validity(
    value: object,
    *,
    fallback_timestamp: float,
) -> dict[str, object]:
    """Return a validity window with numeric timestamp fields.

    ``valid_from`` and ``observed_at`` are lower coordinates: a missing one falls
    back (to ``fallback_timestamp``, then to ``valid_from``) and an unreadable one
    degrades to the ``0.0`` sentinel — both read downstream as "valid since the
    epoch", a harmless open *lower* bound. ``valid_to`` is an *upper* bound with
    the opposite polarity: there ``0.0`` reads as "expired at the epoch" and would
    silently drop the finding from every as-of-scoped recall. So a missing *or*
    unreadable ``valid_to`` is treated as no upper bound at all (``None``, open):
    an expiry we could not parse must never manufacture an immediate one.
    """

    base = dict(value) if isinstance(value, Mapping) else {}
    valid_from = _timestamp_or_default(base.get("valid_from"), fallback_timestamp)
    observed_at = _timestamp_or_default(base.get("observed_at"), valid_from)
    base.update(
        {
            "valid_from": valid_from,
            "valid_to": _upper_bound(base.get("valid_to")),
            "observed_at": observed_at,
        }
    )
    return base


def normalise_source_check(
    value: object,
    *,
    record: Mapping[str, object],
    line_no: int,
    default_project: str = DEFAULT_PROJECT,
) -> dict[str, object]:
    """Return source-check metadata with controlled verifier identity."""

    identity = normalise_feed_identity(record, line_no=line_no, default_project=default_project)
    base = dict(value) if isinstance(value, Mapping) else {}
    checked = base.get("checked_this_session")
    base.update(
        {
            "checked_this_session": checked if isinstance(checked, bool) else False,
            "source_ref": _text(base.get("source_ref")) or f"synapse-feed:{line_no}",
            "by": normalise_project(base.get("by") or identity.project),
            "at": _timestamp_or_default(base.get("at"), identity.timestamp),
        }
    )
    return base


def normalise_entities(
    record: Mapping[str, object],
    *,
    default_project: str = DEFAULT_PROJECT,
) -> list[str]:
    """Return unique controlled project hints from source and target fields."""

    entities: list[str] = []
    for value in (record.get("s"), record.get("to")):
        text = _text(value)
        if not text or text.lower() == "all":
            continue
        project = normalise_project(text, default_project=default_project)
        if project not in entities:
            entities.append(project)
    return entities


def _project_segment(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith(_PRESENCE_PREFIX):
        return text.split("@", 1)[1]
    if text.upper() == "CEO":
        return ""
    if "/" in text:
        return text.split("/", 1)[0]
    return text


def _numeric_timestamp(value: object) -> float | None:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _iso_timestamp(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _timestamp_or_default(value: object, default: float) -> float:
    if value is None:
        return default
    return normalise_timestamp(value)


def _upper_bound(value: object) -> float | None:
    """Return a validity upper bound, or ``None`` for an absent/unreadable one.

    A real (positive) timestamp is kept — including a past one, which legitimately
    closes a superseded window. A missing bound (``None``) or an unparseable one
    (the ``normalise_timestamp`` sentinel ``<= 0.0``, e.g. ``"next week"``, ``""``,
    or a bool) both mean "no trustworthy upper bound", so both stay open (``None``)
    rather than collapsing to ``0.0``, which downstream reads as already expired.
    """
    if value is None:
        return None
    parsed = normalise_timestamp(value)
    return parsed if parsed > 0.0 else None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
