# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for LongMemEval prompt aggregation contracts

"""Tests for benchmark reader prompt aggregation contracts."""

from __future__ import annotations

from bench_longmemeval import _type_prompt


def test_multi_session_prompt_trusts_computed_count() -> None:
    """The full-S reader should consume deterministic distinct-count lines."""
    prompt = _type_prompt(
        "How many different doctors did I visit?",
        "multi-session",
        "COMPUTED COUNT: Dr. Smith, Dr. Patel, Dr. Lee = 3 distinct doctors",
    )

    assert "COMPUTED COUNT:" in prompt
    assert "aggregation has been double-checked by deterministic code" in prompt


def test_knowledge_update_prompt_trusts_computed_count() -> None:
    """Knowledge-update answers should share the same count-precompute contract."""
    prompt = _type_prompt(
        "How many different services have I used?",
        "knowledge-update",
        "COMPUTED COUNT: Instacart, DoorDash = 2 distinct services",
    )

    assert "COMPUTED COUNT:" in prompt
    assert "trust it unless it clearly contradicts the question" in prompt
