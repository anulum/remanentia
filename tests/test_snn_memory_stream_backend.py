# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Streamed backend result-validation regression tests

"""Regressions on validate_stream_result exercised through the real installed Rust backend."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from snn_memory.contracts import ModelConfig
from snn_memory.state import initialise_state, initialise_weights
from snn_memory.stream_backend import BackendIdentity, StreamInputs, load_stream_backend


def _stream_backend():
    module = pytest.importorskip("rust_snn_memory.rust_snn_memory")
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return load_stream_backend(
        BackendIdentity(module.STREAMED_API_VERSION, module.CRATE_VERSION, digest)
    )


def _inputs(model: ModelConfig, packets: np.ndarray) -> StreamInputs:
    state = initialise_state(model)
    weights, topology = initialise_weights(model, 11)
    return StreamInputs(
        voltage_mv=np.ascontiguousarray(state.voltage, dtype=np.float64),
        refractory_steps=np.ascontiguousarray(state.refractory, dtype=np.uint32),
        spikes=np.ascontiguousarray(state.spikes, dtype=np.bool_),
        pre_trace=np.ascontiguousarray(state.pre_trace, dtype=np.float64),
        post_trace=np.ascontiguousarray(state.post_trace, dtype=np.float64),
        weights=np.ascontiguousarray(weights, dtype=np.float64),
        topology=np.ascontiguousarray(topology, dtype=np.bool_),
        packets=np.ascontiguousarray(packets, dtype=np.float64),
    )


def test_validate_accepts_a_silent_first_timestep() -> None:
    """A spikeless first timestep (spike_offsets[1] == 0) must not fall into stop-1 negative indexing.

    An all-silent episode produces an empty first spike row; the earlier row-ascending check indexed
    ``spike_indices[start : stop - 1]`` which, at stop == 0, became ``[0:-1]`` (the whole array bar the
    last element) and broadcast-crashed. Completion probes routinely open with a silent step, so this
    is on the real recall path, not a corner case.
    """
    model = ModelConfig(n_neurons=8, excitatory_fraction=0.75, connectivity=0.5)
    result = _stream_backend().run(_inputs(model, np.zeros((4, model.n_neurons))), 2, False, model)
    assert int(result.spike_offsets[1]) == 0  # first timestep is silent
    assert int(result.spike_offsets[-1]) == result.spike_indices.size
