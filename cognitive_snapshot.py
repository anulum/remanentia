# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Cognitive snapshot — serialize and restore reasoning state across sessions.

The SNN stores long-term associative memory (weight matrix).
This module stores short-term cognitive state: what the agent was
thinking, attending to, and uncertain about when the session ended.

The next session loads this snapshot and reconstructs the cognitive
posture — not just the facts, but the direction of thought.

Architecture::

    Session N ending:
        agent reasoning state
            │
            ▼
        snapshot_save()
            │
            ├─► cognitive_state.json (structured: concerns, focus, momentum)
            └─► inject last focus into SNN (membrane echo)

    Session N+1 starting:
        snapshot_load()
            │
            ├─► read cognitive_state.json
            ├─► compute continuity_score vs current context
            └─► return reconstruction prompt for the agent

Usage::

    # At session end
    from cognitive_snapshot import snapshot_save
    snapshot_save(
        project="scpn-control",
        concerns=["weight saturation in STDP", "CI matrix alignment"],
        focus="homeostatic plasticity implementation",
        pending_decisions=["sparse W as default format"],
        momentum="moving from memory retrieval toward identity persistence",
        confidence={"retrieval": 0.95, "identity": 0.4, "product": 0.7},
    )

    # At session start
    from cognitive_snapshot import snapshot_load, continuity_score
    state = snapshot_load()
    score = continuity_score(state, current_context="working on dashboard...")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
STATE_DIR = BASE_DIR / "snn_state"
SNAPSHOT_PATH = STATE_DIR / "cognitive_state.json"
SNAPSHOT_HISTORY = STATE_DIR / "cognitive_history.jsonl"


def snapshot_save(
    project: str,
    concerns: list[str] | None = None,
    focus: str = "",
    pending_decisions: list[str] | None = None,
    momentum: str = "",
    confidence: dict[str, float] | None = None,
    active_traces: list[str] | None = None,
    session_summary: str = "",
) -> Path:
    """Save cognitive state at session end.

    This captures what the agent was attending to — the direction
    of thought, not just the facts decided.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "project": project,
        "timestamp": time.time(),
        "focus": focus,
        "concerns": concerns or [],
        "pending_decisions": pending_decisions or [],
        "momentum": momentum,
        "confidence": confidence or {},
        "active_traces": active_traces or [],
        "session_summary": session_summary,
    }

    SNAPSHOT_PATH.write_text(json.dumps(state, indent=2) + "\n")

    # Append to history (never delete — all cognitive states persist)
    with open(SNAPSHOT_HISTORY, "a") as f:
        f.write(json.dumps(state) + "\n")

    # Inject focus into SNN as a strong stimulus so the membrane
    # state carries an echo of the last cognitive direction
    if focus:
        try:
            from snn_daemon import drop_stimulus
            drop_stimulus(
                f"cognitive-focus: {focus}. concerns: {', '.join(concerns or [])}. "
                f"momentum: {momentum}",
                source="cognitive-snapshot",
            )
        except Exception:
            pass

    return SNAPSHOT_PATH


def snapshot_load() -> dict:
    """Load the most recent cognitive state."""
    if not SNAPSHOT_PATH.exists():
        return {}
    try:
        return json.loads(SNAPSHOT_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def cognitive_history(limit: int = 20) -> list[dict]:
    """Read cognitive state history."""
    if not SNAPSHOT_HISTORY.exists():
        return []
    rows = []
    for line in SNAPSHOT_HISTORY.read_text().strip().split("\n"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows[-limit:]


def continuity_score(previous_state: dict, current_context: str) -> float:
    """Measure how much of the previous cognitive state is present in the current context.

    Returns a score 0-1:
        1.0 = perfect continuity (all concerns, focus, and momentum reflected)
        0.0 = total discontinuity (nothing from previous state is present)

    This is NOT about whether the agent remembers facts — it's about
    whether the cognitive posture (attention, concerns, direction) survived
    the session boundary.
    """
    if not previous_state or not current_context:
        return 0.0

    context_lower = current_context.lower()
    hits = 0
    total = 0

    # Check if focus is reflected
    if previous_state.get("focus"):
        total += 3  # focus is worth 3 points
        focus_words = set(previous_state["focus"].lower().split())
        focus_hits = sum(1 for w in focus_words if w in context_lower and len(w) > 3)
        hits += min(3, focus_hits)

    # Check if concerns are reflected
    for concern in previous_state.get("concerns", []):
        total += 1
        concern_words = set(concern.lower().split())
        if any(w in context_lower for w in concern_words if len(w) > 3):
            hits += 1

    # Check if momentum direction is reflected
    if previous_state.get("momentum"):
        total += 2
        momentum_words = set(previous_state["momentum"].lower().split())
        momentum_hits = sum(1 for w in momentum_words if w in context_lower and len(w) > 3)
        hits += min(2, momentum_hits)

    # Check if pending decisions are acknowledged
    for decision in previous_state.get("pending_decisions", []):
        total += 1
        decision_words = set(decision.lower().split())
        if any(w in context_lower for w in decision_words if len(w) > 3):
            hits += 1

    return hits / max(total, 1)


def reconstruction_prompt(state: dict) -> str:
    """Generate a context-priming prompt from the cognitive snapshot.

    This is injected at session start to help the new instance
    reconstruct the previous session's cognitive posture.
    """
    if not state:
        return ""

    lines = []
    age_hours = (time.time() - state.get("timestamp", 0)) / 3600

    lines.append(f"## Cognitive State (from {age_hours:.0f}h ago, project: {state.get('project', '?')})")

    if state.get("focus"):
        lines.append(f"**Active focus:** {state['focus']}")

    if state.get("momentum"):
        lines.append(f"**Direction:** {state['momentum']}")

    if state.get("concerns"):
        lines.append("**Open concerns:**")
        for c in state["concerns"]:
            lines.append(f"- {c}")

    if state.get("pending_decisions"):
        lines.append("**Pending decisions:**")
        for d in state["pending_decisions"]:
            lines.append(f"- {d}")

    if state.get("confidence"):
        lines.append("**Confidence levels:**")
        for domain, level in state["confidence"].items():
            bar = "#" * int(level * 10) + "." * (10 - int(level * 10))
            lines.append(f"- {domain}: {bar} {level:.0%}")

    if state.get("session_summary"):
        lines.append(f"\n**Last session:** {state['session_summary']}")

    return "\n".join(lines)
