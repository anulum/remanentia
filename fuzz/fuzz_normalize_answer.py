# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Fuzz target: answer normaliser
"""Property-based fuzz tests for answer_normalizer."""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from answer_normalizer import normalize_answer, answers_match


@given(st.text(max_size=2000))
@settings(max_examples=5000, suppress_health_check=[HealthCheck.too_slow])
def test_normalize_no_crash(text: str) -> None:
    result = normalize_answer(text)
    assert isinstance(result, str)


@given(st.text(max_size=500), st.text(max_size=500), st.floats(0.0, 1.0))
@settings(max_examples=2000, suppress_health_check=[HealthCheck.too_slow])
def test_answers_match_no_crash(a: str, b: str, threshold: float) -> None:
    result = answers_match(a, b, threshold)
    assert isinstance(result, bool)
