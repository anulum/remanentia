# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for SYNAPSE feed normalisation

"""Real-schema tests for SYNAPSE feed identity and timestamp normalisation."""

from __future__ import annotations

import pytest

from feed_normalization import (
    FeedIdentity,
    normalise_actor,
    normalise_entities,
    normalise_feed_identity,
    normalise_project,
    normalise_provenance,
    normalise_source_check,
    normalise_timestamp,
    normalise_validity,
)


def test_normalise_project_uses_sender_project_before_identity_suffix() -> None:
    """Project names should be canonical uppercase identifiers."""

    assert normalise_project("scpn-fusion-core/worker-7f3a") == "SCPN-FUSION-CORE"
    assert normalise_project("synapse-presence@director-ai") == "DIRECTOR-AI"
    assert normalise_project("CEO", default_project="REMANENTIA") == "REMANENTIA"


def test_normalise_actor_returns_controlled_stable_actor_names() -> None:
    """Actor names should not retain per-process random suffixes."""

    assert normalise_actor("scpn-fusion-core/worker-7f3a") == "worker"
    assert normalise_actor("Arcane Sapience") == "arcane-sapience"
    assert normalise_actor("synapse-presence@remanentia") == "system"
    assert normalise_actor("CEO") == "operator"
    assert normalise_actor("") == "synapse"


def test_normalise_timestamp_accepts_seconds_millis_and_iso() -> None:
    """Feed timestamps arrive in multiple real SYNAPSE shapes."""

    assert normalise_timestamp(1_782_486_245) == pytest.approx(1_782_486_245.0)
    assert normalise_timestamp(1_782_486_245_000) == pytest.approx(1_782_486_245.0)
    assert normalise_timestamp("2026-06-26T15:04:05Z") == pytest.approx(1_782_486_245.0)
    assert normalise_timestamp("2026-06-26T15:04:05") == pytest.approx(1_782_486_245.0)
    assert normalise_timestamp("   ") == 0.0
    assert normalise_timestamp("bad") == 0.0


def test_normalise_feed_identity_combines_record_fields() -> None:
    """The record identity should prefer explicit project then source sender."""

    identity = normalise_feed_identity(
        {
            "project": " director-ai ",
            "s": "scpn-fusion-core/worker-7f3a",
            "h": "session 17",
            "i": "42",
            "t": "2026-06-26T15:04:05Z",
        },
        line_no=9,
        default_project="REMANENTIA",
    )

    assert identity == FeedIdentity(
        project="DIRECTOR-AI",
        actor="worker",
        session="session-17",
        source_event_seq=42,
        timestamp=1_782_486_245.0,
    )


def test_normalise_provenance_and_validity_preserve_extra_fields() -> None:
    """Existing mapping payloads should be normalised without losing metadata."""

    record = {"s": "remanentia/worker-7f3a", "i": True, "t": 1_782_486_245_000}

    provenance = normalise_provenance(
        {"project": " scpn-mif-core ", "actor": "Arcane Sapience", "extra": "kept"},
        record=record,
        line_no=7,
        default_project="REMANENTIA",
    )
    validity = normalise_validity(
        {"valid_from": "2026-06-26T15:04:05Z", "valid_to": None, "extra": "kept"},
        fallback_timestamp=0.0,
    )

    assert provenance["project"] == "SCPN-MIF-CORE"
    assert provenance["actor"] == "arcane-sapience"
    assert provenance["source_event_seq"] == 7
    assert provenance["extra"] == "kept"
    assert validity["valid_from"] == pytest.approx(1_782_486_245.0)
    assert validity["observed_at"] == pytest.approx(1_782_486_245.0)
    assert validity["extra"] == "kept"


def test_normalise_source_check_and_entities_use_controlled_project_names() -> None:
    """Source checks and entity hints should share the same project vocabulary."""

    record = {
        "s": "scpn-fusion-core/worker-7f3a",
        "to": "remanentia",
        "t": "2026-06-26T15:04:05Z",
    }

    source_check = normalise_source_check(
        {"checked_this_session": True, "by": " director-ai "},
        record=record,
        line_no=3,
        default_project="REMANENTIA",
    )

    assert source_check["by"] == "DIRECTOR-AI"
    assert source_check["at"] == pytest.approx(1_782_486_245.0)
    assert normalise_entities(record, default_project="REMANENTIA") == [
        "SCPN-FUSION-CORE",
        "REMANENTIA",
    ]
