# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Matched SNN memory controls

"""Trained, shuffled, random and zero controls scoped to the learned pathway."""

from __future__ import annotations

from typing import Literal

import numpy as np

from snn_memory.contracts import ModelConfig
from snn_memory.state import BoolArray, FloatArray, validate_weights

ControlName = Literal["trained", "shuffled", "random", "zero"]


def make_control(
    weights: FloatArray,
    topology: BoolArray,
    config: ModelConfig,
    condition: ControlName,
    seed: int,
) -> FloatArray:
    """Return one matched recurrent control that perturbs only learned E→E weights.

    The excitatory-to-excitatory block is the sole plastic pathway. The shuffled and
    random conditions therefore alter only its connected weights; every
    excitatory-to-inhibitory, inhibitory-to-excitatory and inhibitory-to-inhibitory
    weight stays byte-identical to the trained matrix so the ablation isolates learned
    structure rather than the fixed E/I dynamics. The zero condition removes all
    recurrent weight.
    """
    validate_weights(weights, topology, config)
    if condition == "trained":
        return weights.copy()
    if condition == "zero":
        return np.zeros_like(weights)
    excitatory = config.n_excitatory
    result = weights.copy()
    block_mask = topology[:excitatory, :excitatory]
    source = weights[:excitatory, :excitatory][block_mask]
    if source.size == 0:
        return result
    rng = np.random.default_rng(seed)
    if condition == "shuffled":
        values = rng.permutation(source)
    else:
        values = np.abs(rng.normal(source.mean(), source.std(), size=source.size))
        values = np.clip(values, config.weight_min, config.weight_max)
    block = result[:excitatory, :excitatory]
    block[block_mask] = values
    result[:excitatory, :excitatory] = block
    validate_weights(result, topology, config)
    return result
