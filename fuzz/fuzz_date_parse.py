# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Fuzz target: date parsing
"""Property-based fuzz tests for temporal_graph.parse_dates."""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from temporal_graph import parse_dates


@given(st.text(max_size=5000))
@settings(max_examples=5000, suppress_health_check=[HealthCheck.too_slow])
def test_parse_dates_no_crash(text: str) -> None:
    result = parse_dates(text)
    assert isinstance(result, list)
    for d in result:
        assert isinstance(d, str)
