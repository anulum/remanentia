# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Fuzz target: entity extraction
"""Property-based fuzz tests for entity_extractor.regex_entities."""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from entity_extractor import _regex_entities


@given(st.text(max_size=5000))
@settings(max_examples=5000, suppress_health_check=[HealthCheck.too_slow])
def test_regex_entities_no_crash(text: str) -> None:
    result = _regex_entities(text)
    assert isinstance(result, list)
