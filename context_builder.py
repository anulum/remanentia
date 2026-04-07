# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Hierarchical context builder for LLM prompts

"""Hierarchical context builder for LLM prompts.

Organises retrieved facts into 4 layers:
1. Top-of-Mind (Recent 24h)
2. Work Context (Last 7 days)
3. Stable Facts (High confidence > 0.8)
4. Background (Validated historical facts)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from fact_decomposer import AtomicFact


@dataclass
class HierarchicalContext:
    """Retrieved facts organised by temporal and confidence layers."""

    top_of_mind: list[AtomicFact] = field(default_factory=list)
    work_context: list[AtomicFact] = field(default_factory=list)
    stable_facts: list[AtomicFact] = field(default_factory=list)
    background: list[AtomicFact] = field(default_factory=list)

    def to_prompt_string(self, budget_ratios: dict[str, float] = None) -> str:
        """Convert layers into a formatted prompt section with budget constraints."""
        if budget_ratios is None:
            budget_ratios = {
                "top_of_mind": 0.30,
                "work_context": 0.30,
                "stable_facts": 0.25,
                "background": 0.15,
            }

        sections = []

        if self.top_of_mind:
            count = max(1, int(20 * budget_ratios["top_of_mind"]))
            facts = [
                "- [Session "
                + str(f.session_idx + 1)
                + (", Date: " + ", ".join(f.date_mentions) if f.date_mentions else "")
                + "] "
                + f.text
                for f in self.top_of_mind[:count]
            ]
            sections.append("### TOP-OF-MIND (RECENT 24H)\n" + "\n".join(facts))

        if self.work_context:
            count = max(1, int(20 * budget_ratios["work_context"]))
            facts = [
                "- [Session "
                + str(f.session_idx + 1)
                + (", Date: " + ", ".join(f.date_mentions) if f.date_mentions else "")
                + "] "
                + f.text
                for f in self.work_context[:count]
            ]
            sections.append("### WORK CONTEXT (LAST 7 DAYS)\n" + "\n".join(facts))

        if self.stable_facts:
            count = max(1, int(20 * budget_ratios["stable_facts"]))
            facts = [
                "- [Session "
                + str(f.session_idx + 1)
                + (", Date: " + ", ".join(f.date_mentions) if f.date_mentions else "")
                + "] "
                + f.text
                for f in self.stable_facts[:count]
            ]
            sections.append("### STABLE FACTS\n" + "\n".join(facts))

        if self.background:
            count = max(1, int(20 * budget_ratios["background"]))
            facts = [
                "- [Session "
                + str(f.session_idx + 1)
                + (", Date: " + ", ".join(f.date_mentions) if f.date_mentions else "")
                + "] "
                + f.text
                for f in self.background[:count]
            ]
            sections.append("### BACKGROUND KNOWLEDGE\n" + "\n".join(facts))

        return "\n\n".join(sections)


def build_hierarchical_context(
    facts: list[AtomicFact],
    reference_date: Optional[str] = None,
    session_dates: list[str] = None,
) -> HierarchicalContext:
    """Organise facts into hierarchy based on age and confidence."""
    if reference_date:
        try:
            ref = datetime.fromisoformat(reference_date.split(" ")[0].replace("/", "-"))
        except ValueError:
            ref = datetime.now()
    else:
        ref = datetime.now()

    ctx = HierarchicalContext()

    for fact in facts:
        # Determine fact date
        fact_date = None
        if fact.valid_from:
            try:
                fact_date = datetime.fromisoformat(fact.valid_from)
            except ValueError:
                pass

        if not fact_date and session_dates and fact.session_idx < len(session_dates):
            try:
                # Handle YYYY/MM/DD (Day) HH:MM format
                ds = session_dates[fact.session_idx].split(" ")[0].replace("/", "-")
                fact_date = datetime.fromisoformat(ds)
            except ValueError:
                pass

        if not fact_date:
            # Fallback for facts without dates
            if fact.confidence > 0.8:
                ctx.stable_facts.append(fact)
            else:
                ctx.background.append(fact)
            continue

        age = ref - fact_date

        if age <= timedelta(days=1):
            ctx.top_of_mind.append(fact)
        elif age <= timedelta(days=7):
            ctx.work_context.append(fact)
        elif fact.confidence > 0.8:
            ctx.stable_facts.append(fact)
        else:
            ctx.background.append(fact)

    return ctx
