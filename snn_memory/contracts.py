# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN memory configuration contracts

"""Immutable contracts for temporal SNN memory experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, TypeGuard

_U32_MAX = 4_294_967_295


def _is_finite_number(value: object) -> bool:
    """Return whether ``value`` is a finite JSON number, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: object) -> TypeGuard[int]:
    """Return whether ``value`` is an exact JSON integer, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_u32(value: object) -> bool:
    """Return whether ``value`` is an unsigned 32-bit seed."""
    return _is_int(value) and 0 <= value <= _U32_MAX


@dataclass(frozen=True)
class ModelConfig:
    """Parameters for one recurrent leaky-integrate-and-fire network."""

    n_neurons: int = 2048
    excitatory_fraction: float = 0.8
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    v_rest_mv: float = -65.0
    v_reset_mv: float = -70.0
    v_threshold_mv: float = -55.0
    refractory_ms: float = 2.0
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0
    a_plus: float = 0.005
    a_minus: float = 0.006
    weight_min: float = 0.0
    weight_max: float = 1.0
    connectivity: float = 0.1

    def __post_init__(self) -> None:
        """Reject parameter sets that cannot define the declared model."""
        if not _is_int(self.n_neurons) or self.n_neurons < 2:
            raise ValueError("n_neurons must be at least 2")
        numeric = (
            self.excitatory_fraction,
            self.dt_ms,
            self.tau_m_ms,
            self.v_rest_mv,
            self.v_reset_mv,
            self.v_threshold_mv,
            self.refractory_ms,
            self.tau_plus_ms,
            self.tau_minus_ms,
            self.a_plus,
            self.a_minus,
            self.weight_min,
            self.weight_max,
            self.connectivity,
        )
        if not all(_is_finite_number(value) for value in numeric):
            raise ValueError("model parameters must be finite")
        if not 0.0 < self.excitatory_fraction < 1.0:
            raise ValueError("excitatory_fraction must be between zero and one")
        if not 0 < self.n_excitatory < self.n_neurons:
            raise ValueError("population must contain both excitatory and inhibitory neurons")
        if (
            self.dt_ms <= 0.0
            or self.tau_m_ms <= 0.0
            or self.tau_plus_ms <= 0.0
            or self.tau_minus_ms <= 0.0
        ):
            raise ValueError("time constants must be positive")
        if self.refractory_ms < 0.0:
            raise ValueError("refractory_ms cannot be negative")
        refractory_ratio = self.refractory_ms / self.dt_ms
        if not math.isfinite(refractory_ratio) or math.ceil(refractory_ratio) > _U32_MAX:
            raise ValueError("refractory duration exceeds the u32 timestep domain")
        if self.a_plus < 0.0 or self.a_minus < 0.0:
            raise ValueError("learning amplitudes cannot be negative")
        if self.v_reset_mv >= self.v_threshold_mv or self.v_rest_mv >= self.v_threshold_mv:
            raise ValueError("reset and resting potentials must be below threshold")
        if not 0.0 < self.connectivity <= 1.0:
            raise ValueError("connectivity must be in (0, 1]")
        if self.weight_min != 0.0 or self.weight_max <= self.weight_min:
            raise ValueError(
                "invalid excitatory weight bounds: weight_min must be zero and weight_max positive"
            )

    @property
    def n_excitatory(self) -> int:
        """Return the number of excitatory presynaptic rows."""
        return int(self.n_neurons * self.excitatory_fraction)

    @property
    def refractory_steps(self) -> int:
        """Return the refractory duration in whole timesteps.

        Declared language-independent discretisation rule: a zero duration stays
        zero; any positive duration rounds UP (ceil) to a whole timestep, so a
        sub-step refractory period never silently quantises to zero and there are
        no rounding-tie ambiguities between the Python and Rust backends.
        """
        if self.refractory_ms <= 0.0:
            return 0
        return math.ceil(self.refractory_ms / self.dt_ms)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class EncoderConfig:
    """Parameters for ordered text-to-spike encoding."""

    feature_dim: int = 512
    packet_ms: int = 20
    silent_ms: int = 5
    active_fraction: float = 0.05
    projection_seed: int = 1729

    def __post_init__(self) -> None:
        """Reject invalid packet and projection settings."""
        integers = (self.feature_dim, self.packet_ms, self.silent_ms)
        if not all(_is_int(value) for value in integers) or not _is_u32(self.projection_seed):
            raise ValueError("invalid encoder dimensions, timing, or projection seed")
        if self.feature_dim < 8 or self.packet_ms < 2 or self.silent_ms < 0:
            raise ValueError("invalid encoder dimensions or timing")
        if not _is_finite_number(self.active_fraction):
            raise ValueError("active_fraction must be a finite number in (0, 1]")
        if not 0.0 < self.active_fraction <= 1.0:
            raise ValueError("active_fraction must be in (0, 1]")


@dataclass(frozen=True)
class TrainConfig:
    """Parameters controlling deterministic replay and training."""

    seed: int = 11
    epochs: int = 3
    input_current: float = 18.0

    def __post_init__(self) -> None:
        """Reject empty schedules and non-positive input current."""
        if not _is_u32(self.seed) or not _is_int(self.epochs):
            raise ValueError("training seed must be u32 and epochs an integer")
        if (
            not _is_finite_number(self.input_current)
            or self.epochs < 1
            or self.input_current <= 0.0
        ):
            raise ValueError("training requires epochs >= 1 and positive current")


@dataclass(frozen=True)
class ProbeConfig:
    """Parameters for frozen cue-only recall."""

    seed: int = 11
    cue_fraction: float = 0.5
    completion_steps: int = 40
    input_current: float = 18.0

    def __post_init__(self) -> None:
        """Reject invalid cue and completion windows."""
        if not _is_u32(self.seed) or not _is_int(self.completion_steps):
            raise ValueError("probe seed must be u32 and completion_steps an integer")
        if not _is_finite_number(self.cue_fraction):
            raise ValueError("cue_fraction must be a finite number in (0, 1]")
        if not 0.0 < self.cue_fraction <= 1.0:
            raise ValueError("cue_fraction must be in (0, 1]")
        if (
            not _is_finite_number(self.input_current)
            or self.completion_steps < 1
            or self.input_current <= 0.0
        ):
            raise ValueError("probe requires a completion window and positive current")
