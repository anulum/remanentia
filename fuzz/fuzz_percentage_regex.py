# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Fuzz target: percentage regex in knowledge_store
"""Property-based fuzz tests for knowledge_store._extract_entities."""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from knowledge_store import _extract_entities


@given(st.text(max_size=5000))
@settings(max_examples=5000, suppress_health_check=[HealthCheck.too_slow])
def test_extract_entities_no_crash(text: str) -> None:
    result = _extract_entities(text)
    assert isinstance(result, set)


@given(st.from_regex(r"(\d+\.\d+%\s*){1,100}", fullmatch=True))
@settings(max_examples=2000, suppress_health_check=[HealthCheck.too_slow])
def test_percentage_heavy_no_hang(text: str) -> None:
    result = _extract_entities(text)
    assert isinstance(result, set)
