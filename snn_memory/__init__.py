# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN memory research package

"""Isolated temporal spiking-memory experiment surfaces."""

from snn_memory.contracts import EncoderConfig, ModelConfig, ProbeConfig, TrainConfig
from snn_memory.reference import EpisodeResult, step_network
from snn_memory.state import NetworkState, initialise_state, initialise_weights

__all__ = [
    "EncoderConfig",
    "EpisodeResult",
    "ModelConfig",
    "NetworkState",
    "ProbeConfig",
    "TrainConfig",
    "initialise_state",
    "initialise_weights",
    "step_network",
]
