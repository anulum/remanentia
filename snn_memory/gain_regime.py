# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Recurrent gain-regime diagnostic for the temporal SNN memory

"""Why a trained recurrent LIF network is (or is not) silent after cue removal.

The held-out G1 shakeout found the 512-neuron development network completely silent once the
external cue drive is removed (completion signal ~0 across seeds), so recall is zero in every
condition — an honest negative. This module explains *why* mechanistically and answers whether
scaling the neuron count could change it, without paying for a full training run.

A neuron at rest (``v_rest``) must climb the ``v_threshold - v_rest`` gap to fire. With the cue
gone the only drive is the recurrent input ``I = spikes @ W`` (millivolts added to the membrane),
so self-sustained completion needs the excitatory recurrent depolarisation to reach that gap. The
excitatory drive a neuron receives scales with ``n_excitatory * connectivity`` (∝ N), so the same
question at 512 vs 2048 neurons is a scaling statement, not a guess. ``weight_max``-saturated
E→E weights bound the most that online STDP could ever build.

The raw spectral radius of the signed weight matrix is reported for completeness but is *not* the
operative criterion: Dale's 80/20 split gives the matrix a net-excitatory mean whose outlier
eigenvalue exceeds one even when the threshold-gated, sparsely-active dynamics cannot self-sustain.
The depolarisation-versus-threshold-gap comparison is the physical test.
"""

from __future__ import annotations

import numpy as np

from .contracts import ModelConfig
from .state import initialise_weights

FloatArray = np.ndarray


def threshold_gap_mv(config: ModelConfig) -> float:
    """Millivolts a resting neuron must climb to fire: ``v_threshold - v_rest``."""
    return config.v_threshold_mv - config.v_rest_mv


def saturate_excitatory(weights: FloatArray, n_excitatory: int, weight_max: float) -> FloatArray:
    """Return a copy with every existing E→E synapse raised to ``weight_max``.

    This is the upper bound on the excitatory recurrence online STDP could construct — plasticity
    is confined to the E→E block and cannot exceed ``weight_max``.
    """
    saturated = weights.copy()
    block = saturated[:n_excitatory, :n_excitatory]
    block[block > 0.0] = weight_max
    saturated[:n_excitatory, :n_excitatory] = block
    return saturated


def spectral_radius(weights: FloatArray) -> float:
    """Largest eigenvalue magnitude of the recurrent weight matrix."""
    return float(np.max(np.abs(np.linalg.eigvals(weights))))


def _mean_excitatory_column_sum(weights: FloatArray, n_excitatory: int) -> float:
    """Mean over postsynaptic neurons of the total excitatory weight each one receives."""
    return float(weights[:n_excitatory].sum(axis=0).mean())


def recurrent_depolarisation_mv(
    weights: FloatArray, n_excitatory: int, active_fraction: float
) -> float:
    """Expected excitatory recurrent depolarisation when ``active_fraction`` of the E population fires."""
    if not 0.0 <= active_fraction <= 1.0:
        raise ValueError("active_fraction must be in [0, 1]")
    return active_fraction * _mean_excitatory_column_sum(weights, n_excitatory)


def active_fraction_for_threshold(
    weights: FloatArray, n_excitatory: int, threshold_gap: float
) -> float:
    """Excitatory active fraction whose expected recurrent depolarisation reaches the threshold gap.

    ``inf`` when the excitatory recurrence carries no positive weight (the gap is unreachable).
    """
    column_sum = _mean_excitatory_column_sum(weights, n_excitatory)
    if column_sum <= 0.0:
        return float("inf")
    return threshold_gap / column_sum


def gain_regime_report(
    config: ModelConfig, seed: int, active_fraction: float
) -> dict[str, float | bool]:
    """Diagnose one network's recurrent gain regime at the operative active fraction.

    ``crosses_threshold_at_saturation`` is the decision-bearing field: if even the ``weight_max``-
    saturated E→E network cannot depolarise a neuron to threshold at ``active_fraction``, no amount
    of STDP under this configuration produces self-sustained completion at this scale.
    """
    weights, _topology = initialise_weights(config, seed)
    saturated = saturate_excitatory(weights, config.n_excitatory, config.weight_max)
    gap = threshold_gap_mv(config)
    depolarisation_saturated = recurrent_depolarisation_mv(
        saturated, config.n_excitatory, active_fraction
    )
    return {
        "n_neurons": config.n_neurons,
        "n_excitatory": config.n_excitatory,
        "threshold_gap_mv": gap,
        "active_fraction": active_fraction,
        "spectral_radius_initial": spectral_radius(weights),
        "spectral_radius_saturated": spectral_radius(saturated),
        "depolarisation_initial_mv": recurrent_depolarisation_mv(
            weights, config.n_excitatory, active_fraction
        ),
        "depolarisation_saturated_mv": depolarisation_saturated,
        "active_fraction_for_threshold_saturated": active_fraction_for_threshold(
            saturated, config.n_excitatory, gap
        ),
        "crosses_threshold_at_saturation": depolarisation_saturated >= gap,
    }


def reachable_gain_report(
    config: ModelConfig, trained_weights: FloatArray, active_fraction: float
) -> dict[str, float | bool]:
    """Score the TRAINED (reachable) E→E depolarisation — the sufficient-side go/no-go.

    ``gain_regime_report`` scores the ``weight_max``-saturated ceiling: a NECESSARY condition (the most
    online STDP could ever build must clear the firing gap). This scores what plasticity ACTUALLY built —
    the trained ``trained_weights`` block — so ``crosses_threshold_reachable`` is the sufficient-side
    predictor. A supra-threshold ceiling with a sub-threshold reachable depolarisation is exactly the
    ADR-0007 dev-probe outcome: the ceiling crosses but online STDP does not reach it.
    """
    gap = threshold_gap_mv(config)
    depolarisation = recurrent_depolarisation_mv(
        trained_weights, config.n_excitatory, active_fraction
    )
    return {
        "n_neurons": config.n_neurons,
        "threshold_gap_mv": gap,
        "active_fraction": active_fraction,
        "depolarisation_reachable_mv": depolarisation,
        "active_fraction_for_threshold_reachable": active_fraction_for_threshold(
            trained_weights, config.n_excitatory, gap
        ),
        "crosses_threshold_reachable": depolarisation >= gap,
    }
