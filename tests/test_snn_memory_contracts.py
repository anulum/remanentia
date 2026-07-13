# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN contract tests

"""Immutable-configuration validation and derived-quantity tests."""

from __future__ import annotations

import pytest

from snn_memory.contracts import EncoderConfig, ModelConfig, ProbeConfig, TrainConfig


def test_model_config_derived_quantities() -> None:
    model = ModelConfig(n_neurons=100, excitatory_fraction=0.8, refractory_ms=2.0, dt_ms=1.0)
    assert model.n_excitatory == 80
    assert model.refractory_steps == 2
    assert model.to_dict()["n_neurons"] == 100


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_neurons", 1, "at least 2"),
        ("excitatory_fraction", 0.0, "between zero and one"),
        ("excitatory_fraction", 1.0, "between zero and one"),
        ("dt_ms", 0.0, "time constants must be positive"),
        ("tau_m_ms", 0.0, "time constants must be positive"),
        ("refractory_ms", -1.0, "cannot be negative"),
        ("connectivity", 0.0, r"connectivity must be in"),
        ("connectivity", 1.5, r"connectivity must be in"),
        ("weight_min", -0.1, "invalid excitatory weight bounds"),
        ("weight_max", 0.0, "invalid excitatory weight bounds"),
    ],
)
def test_model_config_rejects_invalid_parameters(field: str, value: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ModelConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("feature_dim", 4, "dimensions or timing"),
        ("packet_ms", 1, "dimensions or timing"),
        ("silent_ms", -1, "dimensions or timing"),
        ("active_fraction", 0.0, r"active_fraction must be in"),
        ("active_fraction", 1.5, r"active_fraction must be in"),
    ],
)
def test_encoder_config_rejects_invalid_parameters(field: str, value: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EncoderConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("epochs", 0), ("input_current", 0.0)],
)
def test_train_config_rejects_invalid_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="epochs >= 1 and positive current"):
        TrainConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("cue_fraction", 0.0, r"cue_fraction must be in"),
        ("cue_fraction", 1.5, r"cue_fraction must be in"),
        ("completion_steps", 0, "completion window and positive current"),
        ("input_current", 0.0, "completion window and positive current"),
    ],
)
def test_probe_config_rejects_invalid_parameters(field: str, value: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ProbeConfig(**{field: value})
