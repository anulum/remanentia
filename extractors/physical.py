# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — P-Channel Extractor (Physical)

"""Extract oscillator phases from SNN membrane state.

Reads sc-neurocore persistent network state (if available)
and converts membrane potentials to phases via Hilbert transform.

Falls back to uniform random phases if no SNN state exists.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("ArcSap.Extractor.P")

SNN_STATE_PATH = Path(__file__).parent.parent / "snn_state" / "identity_net.pkl"
ARCANE_STATE_PATH = Path(__file__).parent.parent / "snn_state" / "arcane_ensemble.json"


def extract_phases(
    oscillator_ids: list[str],
    snn_state_path: Path | None = None,
) -> np.ndarray:
    """Extract phases from SNN state.

    Priority: ArcaneNeuron ensemble (identity signal) > LIF membrane
    potentials > random fallback.
    """
    import json

    n = len(oscillator_ids)

    # Try ArcaneNeuron ensemble first (richer identity signal)
    if ARCANE_STATE_PATH.exists():
        try:
            ensemble = json.loads(ARCANE_STATE_PATH.read_text())
            return _extract_from_arcane(ensemble, n)
        except Exception as exc:
            logger.warning("Failed to read ArcaneNeuron state: %s", exc)

    # Fall back to LIF membrane potentials
    state_path = snn_state_path or SNN_STATE_PATH
    if state_path.exists():
        try:
            return _extract_from_snn(state_path, n)
        except Exception as exc:
            logger.warning("Failed to read SNN state: %s — using random phases", exc)

    rng = np.random.default_rng(n)
    return rng.uniform(0, 2 * np.pi, n)


def _extract_from_arcane(ensemble: list[dict], n: int) -> np.ndarray:
    """Convert ArcaneNeuron states to oscillator phases.

    v_deep carries identity (slow timescale).
    v_work carries session context (medium timescale).
    Combine: phase = atan2(v_work, v_deep + epsilon).
    """
    n_arcane = len(ensemble)
    if n_arcane == 0:
        raise ValueError("Empty ArcaneNeuron ensemble")

    raw_phases = np.zeros(n_arcane)
    for i, state in enumerate(ensemble):
        v_deep = state.get("v_deep", 0.0)
        v_work = state.get("v_work", 0.0)
        raw_phases[i] = np.arctan2(v_work, v_deep + 1e-12)

    # Subsample or tile to match oscillator count
    if n_arcane >= n:
        indices = np.linspace(0, n_arcane - 1, n, dtype=int)
        return raw_phases[indices]
    return np.tile(raw_phases, (n // n_arcane) + 1)[:n]


def _extract_from_snn(state_path: Path, n: int) -> np.ndarray:
    """Load SNN state and extract phases."""
    import pickle

    with open(state_path, "rb") as f:
        state = pickle.load(f)

    # Expect state to contain membrane potentials
    if isinstance(state, dict):
        v = state.get("membrane_potentials", state.get("v", None))
    elif hasattr(state, "v"):
        v = state.v
    else:
        v = np.array(state)

    if v is None:
        raise ValueError("No membrane potentials found in SNN state")

    v = np.asarray(v, dtype=np.float64).ravel()

    # Subsample or pad to match oscillator count
    if len(v) >= n:
        indices = np.linspace(0, len(v) - 1, n, dtype=int)
        v_sub = v[indices]
    else:
        v_sub = np.pad(v, (0, n - len(v)), mode="wrap")

    # Hilbert transform: v → analytic signal → instantaneous phase
    try:
        from scipy.signal import hilbert

        analytic = hilbert(v_sub)
        phases = np.angle(analytic)
    except ImportError:
        # Fallback: normalize v to [0, 2π]
        v_min, v_max = v_sub.min(), v_sub.max()
        if v_max - v_min < 1e-12:
            phases = np.zeros(n)
        else:
            phases = 2 * np.pi * (v_sub - v_min) / (v_max - v_min)

    return phases
