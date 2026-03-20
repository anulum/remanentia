# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Active retrieval — automatic memory consultation during reasoning.

Instead of waiting for explicit queries, this module monitors
the agent's reasoning text and proactively surfaces relevant
past decisions, patterns, and skills.

Architecture::

    agent reasoning text (ongoing)
        │
        ▼
    extract_decision_points()
        │
        ▼
    for each decision point:
        ├─► retrieve() past related decisions
        ├─► query_skills() matching patterns
        └─► related_traces() for context
        │
        ▼
    format as advisory context block

Usage in a session hook::

    from active_retrieval import consult_memory
    advisory = consult_memory(
        "I'm about to change the STDP time constant from 20ms to 15ms"
    )
    # Returns relevant past decisions, related traces, applicable skills

Usage as decision guard::

    from active_retrieval import decision_guard
    warnings = decision_guard(
        action="delete sparse W save code",
        context="cleaning up snn_backend.py"
    )
    # Returns warnings if past reasoning contradicts this action
"""
from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Patterns that signal a decision point
_DECISION_PATTERNS = [
    r"(?:going to|will|plan to|about to)\s+(?:change|modify|delete|remove|add|replace|refactor)",
    r"(?:choosing|chose|decision|decided)\s+(?:to|between|against)",
    r"(?:trade.?off|alternative|instead of|rather than)",
    r"(?:should we|should i|do we|question is)",
]


def extract_decision_points(text: str) -> list[str]:
    """Identify sentences that contain decision points."""
    sentences = re.split(r"[.!?\n]", text)
    points = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped or len(stripped) < 15:
            continue
        for pattern in _DECISION_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                points.append(stripped)
                break
    return points


def consult_memory(reasoning_text: str, top_k: int = 3) -> str:
    """Proactively consult memory for relevant past reasoning.

    Takes the agent's current reasoning text, identifies decision
    points, and retrieves related past decisions + applicable skills.

    Returns a formatted advisory block (or empty string if nothing relevant).
    """
    from retrieve import retrieve
    from skill_extractor import query_skills

    decision_points = extract_decision_points(reasoning_text)

    if not decision_points:
        # Fall back to using the full text as a query
        results = retrieve(reasoning_text[:200], top_k=2)
        if not results or results[0]["score"] < 0.3:
            return ""
        lines = ["## Memory Advisory (automatic)"]
        for r in results:
            if r["score"] >= 0.3:
                lines.append(f"- **{r['trace']}** (relevance: {r['score']:.3f}, tier: {r.get('tier', '?')})")
        return "\n".join(lines) if len(lines) > 1 else ""

    advisories = []
    seen_traces = set()

    for point in decision_points[:3]:
        # Retrieve related past reasoning
        results = retrieve(point, top_k=top_k)
        relevant = [r for r in results if r["score"] >= 0.2 and r["trace"] not in seen_traces]

        # Query applicable skills
        skills = query_skills(point, top_k=2)

        if not relevant and not skills:
            continue

        advisory = [f"**Decision:** {point[:100]}"]

        if relevant:
            advisory.append("  Past reasoning:")
            for r in relevant:
                advisory.append(f"  - {r['trace']} (score={r['score']:.3f})")
                seen_traces.add(r["trace"])

        if skills:
            advisory.append("  Applicable skills:")
            for s in skills:
                advisory.append(f"  - {s['name']}: {s['description'][:60]}")

        advisories.append("\n".join(advisory))

    if not advisories:
        return ""

    return "## Memory Advisory (automatic)\n\n" + "\n\n".join(advisories)


def decision_guard(action: str, context: str = "") -> list[str]:
    """Check if a proposed action contradicts past reasoning.

    Returns a list of warnings (empty = no conflicts found).
    This is a safety net, not a blocker — the agent can proceed
    but should acknowledge the contradiction.
    """
    from retrieve import retrieve

    query = f"{action} {context}".strip()
    results = retrieve(query, top_k=5, include_content=True)

    warnings = []
    action_lower = action.lower()

    # Check for contradiction signals
    contradiction_words = {"don't", "never", "avoid", "wrong", "mistake", "failed", "broke"}

    for r in results:
        if r["score"] < 0.3:
            continue
        content = r.get("content", "").lower()
        # Look for sentences that contain both the action topic and contradiction words
        for sentence in content.split("."):
            if any(w in sentence for w in contradiction_words):
                # Check if the sentence is about the same topic
                action_words = set(action_lower.split())
                sentence_words = set(sentence.split())
                if len(action_words & sentence_words) >= 2:
                    warnings.append(
                        f"Trace '{r['trace']}' (score={r['score']:.3f}): "
                        f"{sentence.strip()[:120]}"
                    )

    return warnings
