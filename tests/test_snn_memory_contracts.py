# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN contract tests

"""Immutable-configuration validation and derived-quantity tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from snn_memory.contracts import EncoderConfig, ModelConfig, ProbeConfig, TrainConfig


def test_model_config_derived_quantities() -> None:
    model = ModelConfig(n_neurons=100, excitatory_fraction=0.8, refractory_ms=2.0, dt_ms=1.0)
    assert model.n_excitatory == 80
    assert model.refractory_steps == 2
    assert model.to_dict()["n_neurons"] == 100


def test_model_config_rejects_fraction_that_floors_to_empty_excitatory_population() -> None:
    with pytest.raises(ValueError, match="both excitatory and inhibitory"):
        ModelConfig(n_neurons=2, excitatory_fraction=0.1)


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


@pytest.mark.parametrize(
    ("refractory_ms", "dt_ms", "expected"),
    [(0.0, 1.0, 0), (2.0, 1.0, 2), (2.5, 1.0, 3), (0.4, 1.0, 1), (1.0, 0.5, 2)],
)
def test_refractory_steps_round_positive_durations_up(
    refractory_ms: float, dt_ms: float, expected: int
) -> None:
    assert ModelConfig(refractory_ms=refractory_ms, dt_ms=dt_ms).refractory_steps == expected


def test_refractory_steps_share_the_exact_u32_cross_language_domain() -> None:
    boundary = ModelConfig(refractory_ms=float(2**32 - 1), dt_ms=1.0)
    assert boundary.refractory_steps == 2**32 - 1
    with pytest.raises(ValueError, match="u32 timestep domain"):
        ModelConfig(refractory_ms=float(2**32), dt_ms=1.0)
    with pytest.raises(ValueError, match="u32 timestep domain"):
        ModelConfig(refractory_ms=1.0e308, dt_ms=1.0e-308)


def test_model_config_rejects_nonfinite_and_cross_field_invalidity() -> None:
    invalid: tuple[Callable[[], ModelConfig], ...] = (
        lambda: ModelConfig(dt_ms=float("nan")),
        lambda: ModelConfig(tau_m_ms=float("inf")),
        lambda: ModelConfig(tau_plus_ms=0.0),
        lambda: ModelConfig(tau_minus_ms=-1.0),
        lambda: ModelConfig(a_plus=-0.1),
        lambda: ModelConfig(a_minus=float("nan")),
        lambda: ModelConfig(v_rest_mv=-50.0),
        lambda: ModelConfig(v_reset_mv=-50.0),
        lambda: ModelConfig(connectivity=float("nan")),
        lambda: ModelConfig(weight_max=float("inf")),
        lambda: ModelConfig(weight_min=0.1),
    )
    for constructor in invalid:
        with pytest.raises(ValueError):
            constructor()


def test_train_and_probe_reject_nonfinite_input_current() -> None:
    with pytest.raises(ValueError, match="positive current"):
        TrainConfig(input_current=float("nan"))
    with pytest.raises(ValueError, match="positive current"):
        ProbeConfig(input_current=float("nan"))


def test_public_configs_reject_json_boolean_and_seed_range_drift() -> None:
    invalid: tuple[Callable[[], object], ...] = (
        lambda: ModelConfig(dt_ms=True),
        lambda: ModelConfig(a_plus=True),
        lambda: ModelConfig(weight_max=True),
        lambda: ModelConfig(connectivity=True),
        lambda: EncoderConfig(feature_dim=True),
        lambda: EncoderConfig(packet_ms=True),
        lambda: EncoderConfig(silent_ms=True),
        lambda: EncoderConfig(active_fraction=True),
        lambda: EncoderConfig(projection_seed=-1),
        lambda: EncoderConfig(projection_seed=4_294_967_296),
        lambda: TrainConfig(seed=-1),
        lambda: TrainConfig(seed=4_294_967_296),
        lambda: TrainConfig(epochs=True),
        lambda: ProbeConfig(seed=-1),
        lambda: ProbeConfig(seed=4_294_967_296),
        lambda: ProbeConfig(completion_steps=True),
        lambda: ProbeConfig(cue_fraction=True),
    )
    for constructor in invalid:
        with pytest.raises(ValueError):
            constructor()
