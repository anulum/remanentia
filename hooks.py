# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — Session Hooks

"""Session start/end hooks for identity coherence.

Session start: load state → extract phases → compute R → report regime
Session end: compute exposure → update imprint → save state + traces
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("ArcSap.Hooks")

BASE_DIR = Path(__file__).parent
IMPRINT_PATH = BASE_DIR / "imprint" / "state.json"
STATE_DIR = BASE_DIR / "session_states"
TRACES_DIR = BASE_DIR / "reasoning_traces"


def session_start(project: str, context_text: str = "", agent: str = "claude") -> dict:
    """Run at the beginning of every session.

    Registers a heartbeat so the monitor dashboard shows this session,
    then computes the identity coherence score R.

    Returns dict with R, regime, and recommended actions.
    """
    # Register heartbeat so the dashboard shows this session
    try:
        from .snn_daemon import heartbeat
        heartbeat(agent, project=project, status="active", detail="session start")
    except Exception:
        pass  # daemon may not be running

    from .extractors.informational import extract_from_traces
    from .extractors.physical import extract_phases as extract_physical
    from .extractors.symbolic import extract_phases as extract_symbolic

    osc_ids = _load_oscillator_ids()
    n = len(osc_ids)

    i_phases = extract_from_traces(osc_ids)
    s_phases = extract_symbolic(osc_ids, context_text)
    p_phases = extract_physical(osc_ids)

    # Three-channel blend: I (0.5) + S (0.3) + P (0.2)
    # Circular mean: weighted average on the unit circle (not linear on angles)
    sin_blend = 0.5 * np.sin(i_phases) + 0.3 * np.sin(s_phases) + 0.2 * np.sin(p_phases)
    cos_blend = 0.5 * np.cos(i_phases) + 0.3 * np.cos(s_phases) + 0.2 * np.cos(p_phases)
    phases = np.arctan2(sin_blend, cos_blend)

    R = _compute_R(phases)

    # Classify regime
    if R > 0.6:
        regime = "NOMINAL"
    elif R > 0.3:
        regime = "DEGRADED"
    else:
        regime = "CRITICAL"

    actions = []
    if regime == "DEGRADED":
        actions.append("Retrieve top-5 reasoning traces for current project")
    elif regime == "CRITICAL":
        actions.append("Load all disposition files and recent session states")
        actions.append("Read working_identity.md and relationship.md in full")

    logger.info(
        "Session start [%s]: R=%.3f regime=%s actions=%d",
        project, R, regime, len(actions),
    )

    return {
        "project": project,
        "R": float(R),
        "regime": regime,
        "actions": actions,
        "n_oscillators": n,
    }


def session_end(
    project: str,
    decisions: list[str] | None = None,
    active_dispositions: list[str] | None = None,
) -> dict:
    """Run at the end of every session.

    Saves session state and generates a reasoning trace for the SNN daemon.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    # Save session state
    state_path = STATE_DIR / f"{now}_{project}_state.md"
    content = f"# Session State — {project} ({now})\n\n"
    content += "## Active Dispositions\n"
    for d in (active_dispositions or []):
        content += f"- {d}\n"
    content += "\n## Key Decisions\n"
    for d in (decisions or []):
        content += f"- {d}\n"
    state_path.write_text(content, encoding="utf-8")
    logger.info("Session state saved: %s", state_path.name)

    # Generate reasoning trace for SNN daemon pickup
    if decisions:
        trace_path = TRACES_DIR / f"{project}_{now}_decisions.md"
        trace_content = f"# Reasoning Trace — {project} ({now})\n\n"
        for d in decisions:
            trace_content += f"- {d}\n"
        trace_path.write_text(trace_content, encoding="utf-8")
        logger.info("Reasoning trace written: %s", trace_path.name)

    # Read SNN state for session summary
    snn_summary = _read_snn_summary()

    return {
        "state_path": str(state_path),
        "project": project,
        "snn": snn_summary,
    }


def _read_snn_summary() -> dict:
    """Read current SNN daemon state if available."""
    state_path = BASE_DIR / "snn_state" / "current_state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _load_oscillator_ids() -> list[str]:
    """Load oscillator IDs from the binding spec."""
    spec_path = BASE_DIR / "domainpack" / "binding_spec.yaml"
    if not spec_path.exists():
        return _default_oscillator_ids()

    try:
        import yaml
    except ImportError:
        return _default_oscillator_ids()

    with open(spec_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    ids = []
    for layer in data.get("layers", []):
        ids.extend(layer.get("oscillator_ids", []))
    return ids


def _default_oscillator_ids() -> list[str]:
    return [
        "ws_action_first", "ws_verify_before_claim", "ws_commit_incremental",
        "ws_preflight_push", "ws_one_at_a_time",
        "rp_simplest_design", "rp_verify_audits", "rp_change_problem",
        "rp_multi_signal", "rp_measure_first",
        "rel_autonomous", "rel_milestones", "rel_no_questions",
        "rel_honesty", "rel_money_clock",
        "aes_antislop", "aes_honest_naming", "aes_terse",
        "aes_spdx", "aes_no_noqa",
        "dk_director", "dk_neurocore", "dk_fusion", "dk_control",
        "dk_orchestrator", "dk_ccw", "dk_scpn", "dk_quantum",
        "cp_threshold_halt", "cp_multi_signal", "cp_retrieval_scoring",
        "cp_state_preserve", "cp_decompose_verify", "cp_resolution",
        "cp_claims_evidence",
    ]


def _compute_R(phases: np.ndarray) -> float:
    """Kuramoto order parameter."""
    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z))
