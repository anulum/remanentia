# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for hierarchical context builder (Phase 3)

from fact_decomposer import AtomicFact
from context_builder import build_hierarchical_context, HierarchicalContext


def test_hierarchical_context_layering():
    ref_date = "2024-03-10 12:00"

    # 1. Top-of-Mind (Recent 24h)
    f1 = AtomicFact("Recent fact", 0, 0, "user", "state", valid_from="2024-03-10", confidence=0.5)

    # 2. Work Context (Last 7 days)
    f2 = AtomicFact(
        "Last week fact", 0, 0, "user", "state", valid_from="2024-03-05", confidence=0.5
    )

    # 3. Stable Fact (Old but high confidence)
    f3 = AtomicFact("Stable fact", 0, 0, "user", "state", valid_from="2023-01-01", confidence=0.9)

    # 4. Background (Old, low confidence)
    f4 = AtomicFact(
        "Old background", 0, 0, "user", "state", valid_from="2023-01-01", confidence=0.5
    )

    ctx = build_hierarchical_context([f1, f2, f3, f4], reference_date=ref_date)

    assert f1 in ctx.top_of_mind
    assert f2 in ctx.work_context
    assert f3 in ctx.stable_facts
    assert f4 in ctx.background


def test_to_prompt_string_all_layers():
    """All 4 layers render in prompt output."""
    f1 = AtomicFact("Recent", 0, 0, "user", "state", confidence=0.5)
    f2 = AtomicFact("Work", 1, 0, "user", "state", confidence=0.5)
    f3 = AtomicFact("Stable", 2, 0, "user", "state", confidence=0.9)
    f4 = AtomicFact("Background", 3, 0, "user", "state", confidence=0.3)
    ctx = HierarchicalContext(
        top_of_mind=[f1], work_context=[f2], stable_facts=[f3], background=[f4]
    )
    prompt = ctx.to_prompt_string()
    assert "### TOP-OF-MIND" in prompt
    assert "### WORK CONTEXT" in prompt
    assert "### STABLE FACTS" in prompt
    assert "### BACKGROUND" in prompt
    assert "Recent" in prompt
    assert "Work" in prompt
    assert "Stable" in prompt
    assert "Background" in prompt


def test_to_prompt_string_with_dates():
    """Facts with date_mentions render date in prompt."""
    f = AtomicFact(
        "Dated fact", 0, 0, "user", "state", confidence=0.5, date_mentions=["2024-03-10"]
    )
    ctx = HierarchicalContext(top_of_mind=[f])
    prompt = ctx.to_prompt_string()
    assert "Date: 2024-03-10" in prompt


def test_context_budgeting():
    facts = [AtomicFact(f"Fact {i}", 0, 0, "user", "state", confidence=0.9) for i in range(50)]
    ctx = HierarchicalContext(stable_facts=facts)
    prompt = ctx.to_prompt_string()
    # 25% of 20 = 5 facts
    assert prompt.count("- [Session 1] Fact") == 5


def test_no_reference_date():
    """Without reference_date, uses datetime.now()."""
    f = AtomicFact("Recent", 0, 0, "user", "state", confidence=0.5)
    ctx = build_hierarchical_context([f])
    # Without valid_from and no session_dates, falls to background/stable
    assert len(ctx.background) == 1


def test_invalid_reference_date():
    """Invalid reference_date falls back to now()."""
    f = AtomicFact("Test", 0, 0, "user", "state", valid_from="2024-03-10", confidence=0.5)
    ctx = build_hierarchical_context([f], reference_date="not-a-date")
    assert (
        len(ctx.top_of_mind) + len(ctx.work_context) + len(ctx.stable_facts) + len(ctx.background)
        == 1
    )


def test_invalid_valid_from():
    """Invalid valid_from on fact doesn't crash, falls to default bucket."""
    f = AtomicFact("Bad date", 0, 0, "user", "state", valid_from="garbage", confidence=0.5)
    ctx = build_hierarchical_context([f], reference_date="2024-03-10")
    assert len(ctx.background) == 1


def test_session_dates_fallback():
    """Facts without valid_from use session_dates for age calculation."""
    f = AtomicFact("Session fact", 0, 0, "user", "state", confidence=0.5)
    ctx = build_hierarchical_context([f], reference_date="2024-03-10", session_dates=["2024-03-09"])
    assert f in ctx.top_of_mind


def test_session_dates_invalid():
    """Invalid session date doesn't crash."""
    f = AtomicFact("Bad session", 0, 0, "user", "state", confidence=0.9)
    ctx = build_hierarchical_context([f], reference_date="2024-03-10", session_dates=["not-valid"])
    # Falls to stable (confidence > 0.8, no date)
    assert f in ctx.stable_facts


def test_empty_context():
    ctx = HierarchicalContext()
    assert ctx.to_prompt_string() == ""
