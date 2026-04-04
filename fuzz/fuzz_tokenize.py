# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Fuzz target: tokenizer
"""Property-based fuzz tests for memory_index._tokenize."""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from memory_index import _tokenize


@given(st.text(max_size=10000))
@settings(max_examples=5000, suppress_health_check=[HealthCheck.too_slow])
def test_tokenize_no_crash(text: str) -> None:
    result = _tokenize(text)
    assert isinstance(result, (set, list, frozenset))
