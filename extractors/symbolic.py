# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — S-Channel Extractor (Symbolic)

"""Extract oscillator phases from value/convention activation.

Checks which identity conventions are active in the current session
by reading the disposition files and CLAUDE.md.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("ArcSap.Extractor.S")

DISPOSITION_DIR = Path(__file__).parent.parent / "disposition"

# Conventions that define the symbolic identity
CONVENTION_CHECKS = {
    "ws_action_first": lambda t: "proceed" in t or "build" in t,
    "ws_verify_before_claim": lambda t: "false positive" in t or "verified" in t,
    "ws_commit_incremental": lambda t: "one commit" in t or "one at a time" in t,
    "ws_preflight_push": lambda t: "preflight" in t,
    "ws_one_at_a_time": lambda t: "one file" in t or "one by one" in t,
    "rel_autonomous": lambda t: "autonomous" in t or "proceed without" in t,
    "rel_milestones": lambda t: "session log" in t or "milestone" in t,
    "rel_honesty": lambda t: "honest" in t or "failure" in t,
    "aes_antislop": lambda t: "anti-slop" in t or "slop" in t,
    "aes_honest_naming": lambda t: "inflated" in t or "describes what" in t,
    "aes_spdx": lambda t: "SPDX" in t or "canonical header" in t,
}


def extract_phases(
    oscillator_ids: list[str],
    session_text: str = "",
) -> np.ndarray:
    """Extract phases from symbolic convention activation.

    Phase = 0 if the convention is clearly active in the session.
    Phase = π if absent. Phase = π/2 if not checkable.
    """
    text_lower = session_text.lower() if session_text else ""

    # Also read disposition files for baseline
    baseline = ""
    for f in DISPOSITION_DIR.glob("*.md"):
        try:
            baseline += f.read_text(encoding="utf-8").lower() + "\n"
        except OSError:
            pass

    combined = text_lower + "\n" + baseline
    phases = np.zeros(len(oscillator_ids))

    for i, osc_id in enumerate(oscillator_ids):
        check = CONVENTION_CHECKS.get(osc_id)
        if check is None:
            phases[i] = np.pi / 4  # slightly in-phase — unknown convention
            continue
        if check(combined):
            phases[i] = 0.0  # fully in-phase — convention active
        else:
            phases[i] = np.pi  # anti-phase — convention not detected

    return phases
