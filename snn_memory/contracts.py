# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN memory configuration contracts

"""Immutable contracts for temporal SNN memory experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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
        if self.n_neurons < 2:
            raise ValueError("n_neurons must be at least 2")
        if not 0.0 < self.excitatory_fraction < 1.0:
            raise ValueError("excitatory_fraction must be between zero and one")
        if self.dt_ms <= 0.0 or self.tau_m_ms <= 0.0:
            raise ValueError("time constants must be positive")
        if self.refractory_ms < 0.0:
            raise ValueError("refractory_ms cannot be negative")
        if not 0.0 < self.connectivity <= 1.0:
            raise ValueError("connectivity must be in (0, 1]")
        if self.weight_min < 0.0 or self.weight_max <= self.weight_min:
            raise ValueError("invalid excitatory weight bounds")

    @property
    def n_excitatory(self) -> int:
        """Return the number of excitatory presynaptic rows."""
        return int(self.n_neurons * self.excitatory_fraction)

    @property
    def refractory_steps(self) -> int:
        """Return the refractory duration in discrete timesteps."""
        return int(round(self.refractory_ms / self.dt_ms))

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
        if self.feature_dim < 8 or self.packet_ms < 2 or self.silent_ms < 0:
            raise ValueError("invalid encoder dimensions or timing")
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
        if self.epochs < 1 or self.input_current <= 0.0:
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
        if not 0.0 < self.cue_fraction <= 1.0:
            raise ValueError("cue_fraction must be in (0, 1]")
        if self.completion_steps < 1 or self.input_current <= 0.0:
            raise ValueError("probe requires a completion window and positive current")
