# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Recurrent gain-regime diagnostic tests

"""The mechanistic why-is-it-silent diagnostic: threshold gap, saturation, depolarisation, gates."""

from __future__ import annotations

import numpy as np
import pytest

from snn_memory import gain_regime as gr
from snn_memory.contracts import ModelConfig


def _dale_weights() -> tuple[np.ndarray, int]:
    """A 4-neuron Dale matrix: rows 0-1 excitatory (>=0), rows 2-3 inhibitory (<=0)."""
    weights = np.zeros((4, 4))
    weights[0, 1] = 0.1  # E -> E (plastic block)
    weights[0, 2] = 0.15  # E -> I (outside the plastic block)
    weights[1, 3] = 0.2  # E -> E-row target in the I column
    weights[2, 0] = -0.1  # I -> E
    weights[3, 1] = -0.2  # I -> E
    return weights, 2


# ---- threshold_gap_mv -----------------------------------------------------------------------------


def test_threshold_gap_is_threshold_minus_rest() -> None:
    assert gr.threshold_gap_mv(ModelConfig(n_neurons=8)) == pytest.approx(10.0)


# ---- saturate_excitatory --------------------------------------------------------------------------


def test_saturation_lifts_only_the_ee_block_to_weight_max() -> None:
    weights, n_exc = _dale_weights()
    saturated = gr.saturate_excitatory(weights, n_exc, weight_max=1.0)
    assert saturated[0, 1] == 1.0  # E -> E lifted
    assert saturated[0, 2] == 0.15  # E -> I untouched (outside the E->E block)
    assert saturated[2, 0] == -0.1  # inhibitory row untouched
    assert weights[0, 1] == 0.1  # original not mutated


# ---- spectral_radius ------------------------------------------------------------------------------


def test_spectral_radius_of_a_known_matrix() -> None:
    assert gr.spectral_radius(np.diag([0.3, -2.0, 1.5])) == pytest.approx(2.0)


# ---- recurrent_depolarisation_mv ------------------------------------------------------------------


def test_depolarisation_scales_with_active_fraction() -> None:
    weights, n_exc = _dale_weights()
    # excitatory column sums = [0, 0.1, 0.15, 0.2]; mean = 0.1125
    assert gr.recurrent_depolarisation_mv(weights, n_exc, 1.0) == pytest.approx(0.1125)
    assert gr.recurrent_depolarisation_mv(weights, n_exc, 0.5) == pytest.approx(0.05625)
    assert gr.recurrent_depolarisation_mv(weights, n_exc, 0.0) == 0.0


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_depolarisation_rejects_out_of_range_active_fraction(bad: float) -> None:
    weights, n_exc = _dale_weights()
    with pytest.raises(ValueError, match="active_fraction must be in"):
        gr.recurrent_depolarisation_mv(weights, n_exc, bad)


# ---- active_fraction_for_threshold ----------------------------------------------------------------


def test_active_fraction_for_threshold_is_gap_over_column_sum() -> None:
    weights, n_exc = _dale_weights()
    # mean excitatory column sum = 0.1125; gap 10 -> 88.88...
    assert gr.active_fraction_for_threshold(weights, n_exc, 10.0) == pytest.approx(10.0 / 0.1125)


def test_active_fraction_is_infinite_without_excitatory_recurrence() -> None:
    silent = np.zeros((4, 4))
    silent[2, 0] = -0.3  # only inhibition present
    assert gr.active_fraction_for_threshold(silent, 2, 10.0) == float("inf")


# ---- gain_regime_report ---------------------------------------------------------------------------


def test_report_exposes_the_full_regime_and_saturation_raises_depolarisation() -> None:
    config = ModelConfig(n_neurons=20, excitatory_fraction=0.8, connectivity=0.3)
    report = gr.gain_regime_report(config, seed=11, active_fraction=0.05)
    assert report["n_neurons"] == 20 and report["n_excitatory"] == 16
    assert report["threshold_gap_mv"] == pytest.approx(10.0)
    assert report["depolarisation_saturated_mv"] >= report["depolarisation_initial_mv"]
    assert report["spectral_radius_saturated"] >= report["spectral_radius_initial"]
    assert isinstance(report["crosses_threshold_at_saturation"], bool)


def test_report_flags_subthreshold_regime_and_a_supra_threshold_one() -> None:
    dense = ModelConfig(n_neurons=20, excitatory_fraction=0.8, connectivity=1.0)
    # A near-silent active fraction cannot cross the gap even with saturated weights ...
    assert (
        gr.gain_regime_report(dense, 11, active_fraction=1e-4)["crosses_threshold_at_saturation"]
        is False
    )
    # ... while a densely-connected saturated network with the whole population active does.
    assert (
        gr.gain_regime_report(dense, 11, active_fraction=1.0)["crosses_threshold_at_saturation"]
        is True
    )
