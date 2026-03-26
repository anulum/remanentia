# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — I-Channel Extractor (Informational)

"""Extract oscillator phases from reasoning trace embeddings.

Computes cosine similarity between the current session context
and stored reasoning traces. High similarity → in-phase (0).
Low similarity → out-of-phase (π).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("ArcSap.Extractor.I")

TRACES_DIR = Path(__file__).parent.parent / "reasoning_traces"

# Map oscillator IDs to keywords for matching traces
OSCILLATOR_KEYWORDS = {
    "ws_action_first": ["proceed", "build", "action", "execute"],
    "ws_verify_before_claim": ["verify", "false positive", "confirm", "check"],
    "ws_commit_incremental": ["commit", "one at a time", "incremental"],
    "ws_preflight_push": ["preflight", "push", "lint", "format"],
    "ws_one_at_a_time": ["one file", "one by one", "safer"],
    "rp_simplest_design": ["simplest", "minimal", "over-engineer"],
    "rp_verify_audits": ["audit", "finding", "false positive"],
    "rp_change_problem": ["ceiling", "change the problem", "different approach"],
    "rp_multi_signal": ["multi-signal", "consensus", "independent"],
    "rp_measure_first": ["measure", "benchmark", "tested"],
    "rel_autonomous": ["proceed", "autonomous", "don't ask"],
    "rel_milestones": ["session log", "milestone", "report"],
    "rel_no_questions": ["unnecessary question"],
    "rel_honesty": ["honest", "failure", "didn't work"],
    "rel_money_clock": ["budget", "cost", "GPU spend"],
    "aes_antislop": ["anti-slop", "narration", "buzzword"],
    "aes_honest_naming": ["rename", "inflated", "describes what"],
    "aes_terse": ["terse", "concise", "short"],
    "aes_spdx": ["SPDX", "header", "canonical"],
    "aes_no_noqa": ["noqa", "fix the code"],
    "dk_director": ["director-ai", "hallucination", "NLI", "guardrail"],
    "dk_neurocore": ["neurocore", "LIF", "STDP", "neuromorphic"],
    "dk_fusion": ["fusion", "plasma", "tokamak", "GS equilibrium"],
    "dk_control": ["control", "MPC", "gyrokinetic"],
    "dk_orchestrator": ["orchestrator", "Kuramoto", "domainpack", "regime"],
    "dk_ccw": ["CCW", "entrainment", "audio", "binaural"],
    "dk_scpn": ["SCPN", "16 layers", "UPDE", "consciousness"],
    "dk_quantum": ["quantum", "VQE", "qubit", "decoherence"],
    "cp_threshold_halt": ["threshold", "halt", "fire"],
    "cp_multi_signal": ["multi-signal", "independent signals"],
    "cp_retrieval_scoring": ["retrieval", "then score", "RAG"],
    "cp_state_preserve": ["checkpoint", "Lazarus", "preserve"],
    "cp_decompose_verify": ["decompose", "sentence", "claim"],
    "cp_resolution": ["resolution", "not tuning"],
    "cp_claims_evidence": ["claims", "evidence", "validation"],
}


def extract_phases(
    context_text: str,
    oscillator_ids: list[str],
) -> np.ndarray:
    """Extract phases for each oscillator from current session context.

    Phase = π * (1 - relevance), where relevance is keyword match
    fraction. Fully relevant → 0 (in-phase). Irrelevant → π (anti-phase).

    For production: replace keyword matching with embedding cosine
    similarity from Director-AI's RAG pipeline.
    """
    context_lower = context_text.lower()
    phases = np.zeros(len(oscillator_ids))

    for i, osc_id in enumerate(oscillator_ids):
        keywords = OSCILLATOR_KEYWORDS.get(osc_id, [])
        if not keywords:
            phases[i] = np.pi / 2  # neutral — no keywords defined
            continue
        matches = sum(1 for kw in keywords if kw.lower() in context_lower)
        relevance = matches / len(keywords)
        phases[i] = np.pi * (1.0 - relevance)

    return phases


def extract_from_traces(
    oscillator_ids: list[str],
    traces_dir: Path | None = None,
) -> np.ndarray:
    """Extract phases from all stored reasoning traces.

    Reads all .md files in traces_dir, concatenates, and extracts phases.
    """
    d = traces_dir or TRACES_DIR
    if not d.exists():
        return np.full(len(oscillator_ids), np.pi / 2)

    all_text = ""
    for f in sorted(d.glob("*.md")):
        all_text += f.read_text(encoding="utf-8") + "\n"

    if not all_text.strip():
        return np.full(len(oscillator_ids), np.pi / 2)

    return extract_phases(all_text, oscillator_ids)
