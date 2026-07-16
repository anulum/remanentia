# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Installed-wheel streamed SNN Stage-1 gate

"""Exercise exact streamed parity and invariants through clean installed wheels."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from snn_memory.contracts import ModelConfig
from snn_memory.reference import step_network
from snn_memory.state import NetworkState, initialise_weights
from snn_memory.stream_backend import (
    BackendIdentity,
    StreamBackend,
    StreamInputs,
    load_stream_backend,
    state_digest,
    topology_digest,
    validate_stream_result,
)


@dataclass(frozen=True)
class OracleEvidence:
    spikes: tuple[tuple[int, ...], ...]
    phases: npt.NDArray[np.uint8]
    values: npt.NDArray[np.float64]
    counts: npt.NDArray[np.uint64]
    digests: tuple[str, ...]
    state: NetworkState
    weights: npt.NDArray[np.float64]


def _summary(vector: npt.NDArray[np.float64]) -> tuple[float, float, float, int]:
    signed = 0.0
    l1 = 0.0
    l2 = 0.0
    count = 0
    for value in vector:
        scalar = float(value)
        signed += scalar
        l1 += abs(scalar)
        high = max(l2, abs(scalar))
        low = min(l2, abs(scalar))
        if high != 0.0:
            ratio = low / high
            l2 = high * math.sqrt(1.0 + ratio * ratio)
        count += int(scalar != 0.0)
    return signed, l1, l2, count


def _oracle(
    inputs: StreamInputs,
    cue_steps: int,
    plasticity: bool,
    config: ModelConfig,
) -> OracleEvidence:
    state = NetworkState(
        inputs.voltage_mv.copy(),
        inputs.refractory_steps.astype(np.int64),
        inputs.pre_trace.copy(),
        inputs.post_trace.copy(),
        inputs.spikes.copy(),
    )
    weights = inputs.weights.copy()
    rows: list[tuple[int, ...]] = []
    values: list[list[float]] = []
    counts: list[list[int]] = []
    digests: list[str] = []
    for packet in inputs.packets:
        excitatory = np.zeros(config.n_neurons, dtype=np.float64)
        inhibitory = np.zeros(config.n_neurons, dtype=np.float64)
        net = np.zeros(config.n_neurons, dtype=np.float64)
        for pre in range(config.n_neurons):
            if not state.spikes[pre]:
                continue
            for post in range(config.n_neurons):
                value = weights[pre, post]
                net[post] += value
                if pre < config.n_excitatory:
                    excitatory[post] += value
                else:
                    inhibitory[post] += value
        timestep_values: list[float] = []
        timestep_counts: list[int] = []
        for vector in (packet, excitatory, inhibitory, net):
            signed, l1, l2, count = _summary(vector)
            timestep_values.extend((signed, l1, l2))
            timestep_counts.append(count)
        values.append(timestep_values)
        counts.append(timestep_counts)
        state, weights, _ = step_network(
            state,
            weights,
            inputs.topology,
            packet,
            config,
            plasticity_enabled=plasticity,
        )
        rows.append(tuple(int(index) for index in np.flatnonzero(state.spikes)))
        digests.append(
            state_digest(
                state.voltage,
                state.refractory.astype(np.uint32),
                state.spikes,
                state.pre_trace,
                state.post_trace,
            )
        )
    phases = np.ones(inputs.packets.shape[0], dtype=np.uint8)
    phases[:cue_steps] = 0
    return OracleEvidence(
        tuple(rows),
        phases,
        np.asarray(values, dtype=np.float64),
        np.asarray(counts, dtype=np.uint64),
        tuple(digests),
        state,
        weights,
    )


def _rows(
    offsets: npt.NDArray[np.uint64], indices: npt.NDArray[np.uint64]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in indices[int(offsets[row]) : int(offsets[row + 1])])
        for row in range(offsets.size - 1)
    )


def _readonly(array: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Return a read-only C-contiguous copy for malformed-result cases."""
    copied = np.array(array, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _expect_error(
    error: type[BaseException] | tuple[type[BaseException], ...], operation: object
) -> None:
    """Require one real operation callable to fail with the declared error."""
    assert callable(operation)
    try:
        operation()
    except error:
        return
    raise AssertionError("expected declared error")


def _assert_exact_parity(
    backend: StreamBackend,
    inputs: StreamInputs,
    cue_steps: int,
    plasticity: bool,
    config: ModelConfig,
) -> None:
    before = tuple(array.tobytes() for array in inputs.__dict__.values())
    expected = _oracle(inputs, cue_steps, plasticity, config)
    actual = backend.run(inputs, cue_steps, plasticity, config)
    assert before == tuple(array.tobytes() for array in inputs.__dict__.values())
    assert _rows(actual.spike_offsets, actual.spike_indices) == expected.spikes
    assert np.array_equal(actual.phases, expected.phases)
    if not np.array_equal(actual.current_values, expected.values):
        mismatch = np.argwhere(actual.current_values != expected.values)[0]
        row, column = (int(value) for value in mismatch)
        raise AssertionError(
            f"current summary mismatch at {(row, column)}: "
            f"Rust={actual.current_values[row, column]!r}, "
            f"Python={expected.values[row, column]!r}"
        )
    assert np.array_equal(actual.current_nonzero_counts, expected.counts)
    assert actual.state_digests == expected.digests
    assert np.array_equal(actual.final_voltage_mv, expected.state.voltage)
    assert np.array_equal(
        actual.final_refractory_steps, expected.state.refractory.astype(np.uint32)
    )
    assert np.array_equal(actual.final_spikes, expected.state.spikes)
    assert np.array_equal(actual.final_pre_trace, expected.state.pre_trace)
    assert np.array_equal(actual.final_post_trace, expected.state.post_trace)
    assert np.array_equal(actual.final_weights, expected.weights)
    assert np.array_equal(actual.topology, inputs.topology)


def _fixture(config: ModelConfig, seed: int, timesteps: int) -> StreamInputs:
    weights, topology = initialise_weights(config, seed)
    rng = np.random.default_rng(seed + 1000)
    packets = np.zeros((timesteps, config.n_neurons), dtype=np.float64)
    for timestep in range(timesteps):
        if timestep % 4 != 2:
            indices = rng.choice(
                config.n_neurons, size=max(2, config.n_neurons // 32), replace=False
            )
            packets[timestep, indices] = rng.choice((0.25, 3.0, 18.0), size=indices.size)
    spikes = np.zeros(config.n_neurons, dtype=np.bool_)
    spikes[[0, config.n_excitatory]] = True
    return StreamInputs(
        np.full(config.n_neurons, config.v_rest_mv, dtype=np.float64),
        np.zeros(config.n_neurons, dtype=np.uint32),
        spikes,
        rng.random(config.n_neurons, dtype=np.float64) * 0.2,
        rng.random(config.n_neurons, dtype=np.float64) * 0.2,
        weights,
        topology,
        packets,
    )


def _small_adversarial(backend: StreamBackend) -> None:
    config = ModelConfig(
        n_neurons=4,
        excitatory_fraction=0.5,
        connectivity=1.0,
        refractory_ms=1.0,
    )
    topology = np.array(
        [
            [False, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [True, True, True, False],
        ],
        dtype=np.bool_,
    )
    weights = np.array(
        [
            [0.0, 0.0, 0.5, 1.0],
            [0.2, 0.0, 0.4, 0.1],
            [-0.3, -0.2, 0.0, -0.5],
            [-0.7, -0.1, -0.4, 0.0],
        ],
        dtype=np.float64,
    )
    inputs = StreamInputs(
        np.array([-55.00000000000001, -65.0, -65.0, -65.0], dtype=np.float64),
        np.array([0, 1, 0, 0], dtype=np.uint32),
        np.array([True, False, True, False], dtype=np.bool_),
        np.array([0.2, 0.0, 0.4, 0.0], dtype=np.float64),
        np.array([0.0, 0.3, 0.0, 0.1], dtype=np.float64),
        weights,
        topology,
        np.array(
            [[0.0, 0.0, 0.0, 0.0], [18.0, 0.0, 18.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float64
        ),
    )
    _assert_exact_parity(backend, inputs, 1, False, config)
    _assert_exact_parity(backend, inputs, 2, True, config)

    association_config = ModelConfig(
        n_neurons=2,
        excitatory_fraction=0.5,
        dt_ms=1.0,
        tau_m_ms=1.0e-16,
        v_rest_mv=0.0,
        v_reset_mv=-2.0,
        v_threshold_mv=10.0,
        refractory_ms=0.0,
        connectivity=0.5,
    )
    association = StreamInputs(
        np.array([-1.0, 0.0], dtype=np.float64),
        np.zeros(2, dtype=np.uint32),
        np.array([False, True], dtype=np.bool_),
        np.zeros(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.array([[0.0, 0.0], [-1.0, 0.0]], dtype=np.float64),
        np.array([[False, False], [True, False]], dtype=np.bool_),
        np.array([[-1.0e16, 0.0]], dtype=np.float64),
    )
    _assert_exact_parity(backend, association, 1, False, association_config)


def _negative_boundaries(backend: StreamBackend, identity: BackendIdentity) -> None:
    _expect_error(
        RuntimeError,
        lambda: load_stream_backend(
            BackendIdentity(
                identity.api_version + 1, identity.crate_version, identity.extension_sha256
            )
        ),
    )
    _expect_error(
        ValueError,
        lambda: load_stream_backend(
            BackendIdentity(identity.api_version, identity.crate_version, "g" * 64)
        ),
    )
    _expect_error(
        RuntimeError,
        lambda: load_stream_backend(
            BackendIdentity(identity.api_version, "wrong-crate", identity.extension_sha256)
        ),
    )
    _expect_error(
        RuntimeError,
        lambda: load_stream_backend(
            BackendIdentity(identity.api_version, identity.crate_version, "0" * 64)
        ),
    )
    _expect_error(
        RuntimeError,
        lambda: load_stream_backend(identity, "definitely_missing_snn_extension"),
    )
    _expect_error(
        RuntimeError,
        lambda: load_stream_backend(identity, "snn_memory.stream_backend"),
    )
    module = importlib.import_module("rust_snn_memory.rust_snn_memory")
    rust_config = module.ModelConfig(0)
    zero_f = np.zeros(0, dtype=np.float64)
    zero_u = np.zeros(0, dtype=np.uint32)
    zero_b = np.zeros(0, dtype=np.bool_)
    _expect_error(
        ValueError,
        lambda: module.run_streamed_episode_v2(
            zero_f,
            zero_u,
            zero_b,
            zero_f,
            zero_f,
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.bool_),
            np.zeros((1, 0), dtype=np.float64),
            0,
            False,
            rust_config,
        ),
    )
    config = ModelConfig(n_neurons=4, excitatory_fraction=0.5, connectivity=0.5)
    inputs = _fixture(config, 8, 2)
    bad = inputs.packets.astype(">f8")
    _expect_error(TypeError, lambda: backend.run(replace(inputs, packets=bad), 1, False, config))
    _expect_error(
        TypeError,
        lambda: backend.run(
            replace(inputs, voltage_mv=[0.0] * config.n_neurons),  # type: ignore[arg-type]
            1,
            False,
            config,
        ),
    )
    _expect_error(
        ValueError,
        lambda: backend.run(
            replace(inputs, packets=np.asfortranarray(inputs.packets)), 1, False, config
        ),
    )
    nonfinite = inputs.packets.copy()
    nonfinite[0, 0] = np.nan
    _expect_error(
        ValueError, lambda: backend.run(replace(inputs, packets=nonfinite), 1, False, config)
    )
    _expect_error(TypeError, lambda: backend.run(inputs, -1, False, config))
    _expect_error(TypeError, lambda: backend.run(inputs, 1, 0, config))  # type: ignore[arg-type]
    _expect_error(
        ValueError,
        lambda: backend.run(replace(inputs, voltage_mv=inputs.voltage_mv[:-1]), 1, False, config),
    )
    _expect_error(
        ValueError,
        lambda: backend.run(replace(inputs, weights=inputs.weights[:-1]), 1, False, config),
    )
    _expect_error(
        ValueError,
        lambda: backend.run(replace(inputs, packets=inputs.packets[:0]), 0, False, config),
    )
    _expect_error(ValueError, lambda: backend.run(inputs, 3, False, config))

    transactional = _fixture(config, 18, 2)
    packets = transactional.packets.copy()
    packets[0] = 0.0
    packets[1] = np.finfo(np.float64).max
    transactional = replace(transactional, packets=packets)
    before = tuple(array.tobytes() for array in transactional.__dict__.values())
    _expect_error(ValueError, lambda: backend.run(transactional, 1, False, config))
    assert before == tuple(array.tobytes() for array in transactional.__dict__.values())


def _result_rejections(backend: StreamBackend) -> None:
    """Drive public result rejections from a genuine compiled-extension result."""
    config = ModelConfig(n_neurons=8, excitatory_fraction=0.75, connectivity=0.4)
    inputs = _fixture(config, 41, 4)
    result = backend.run(inputs, 2, True, config)

    def rejected(**changes: object) -> None:
        candidate = replace(result, **changes)  # type: ignore[arg-type]
        _expect_error(
            (TypeError, ValueError, RuntimeError),
            lambda: validate_stream_result(candidate, inputs, 2, config),
        )

    rejected(spike_offsets=list(result.spike_offsets))
    rejected(spike_offsets=_readonly(result.spike_offsets.astype(np.int64)))
    rejected(spike_offsets=_readonly(result.spike_offsets.reshape(1, -1)))
    fortran_values = np.asfortranarray(result.current_values)
    fortran_values.setflags(write=False)
    rejected(current_values=fortran_values)
    nonfinite = result.current_values.copy()
    nonfinite[0, 0] = np.nan
    rejected(current_values=_readonly(nonfinite))
    rejected(spike_offsets=result.spike_offsets.copy())

    offsets = result.spike_offsets.copy()
    offsets[0] = 1
    rejected(spike_offsets=_readonly(offsets))
    rejected(spike_offsets=_readonly(result.spike_offsets[:-1]))
    offsets = result.spike_offsets.copy()
    offsets[-1] -= 1
    rejected(spike_offsets=_readonly(offsets))
    offsets = result.spike_offsets.copy()
    offsets[1] = offsets[2] + 1
    rejected(spike_offsets=_readonly(offsets))
    indices = result.spike_indices.copy()
    indices[0] = config.n_neurons
    rejected(spike_indices=_readonly(indices))
    rejected(
        spike_offsets=_readonly(np.array([0, 2, 2, 2, 2], dtype=np.uint64)),
        spike_indices=_readonly(np.array([1, 1], dtype=np.uint64)),
    )
    phases = result.phases.copy()
    phases[0] = 1
    rejected(phases=_readonly(phases))
    rejected(current_values=_readonly(result.current_values[:-1]))
    rejected(current_nonzero_counts=_readonly(result.current_nonzero_counts[:-1]))
    rejected(state_digests=result.state_digests[:-1])
    rejected(state_digests=("G" * 64,) + result.state_digests[1:])
    rejected(final_voltage_mv=_readonly(result.final_voltage_mv[:-1]))
    rejected(final_weights=_readonly(result.final_weights[:-1]))
    topology = result.topology.copy()
    topology[0, 1] = not topology[0, 1]
    rejected(topology=_readonly(topology))
    rejected(topology_digest="G" * 64)
    rejected(topology_digest="0" * 64)
    rejected(state_digests=result.state_digests[:-1] + ("0" * 64,))
    counts = result.current_nonzero_counts.copy()
    counts[0, 0] = config.n_neurons + 1
    rejected(current_nonzero_counts=_readonly(counts))

    for prefix in ("outgoing", "incoming"):
        offsets_name = f"{prefix}_offsets"
        indices_name = f"{prefix}_indices"
        base_offsets = getattr(result, offsets_name)
        base_indices = getattr(result, indices_name)
        rejected(**{offsets_name: _readonly(base_offsets[:-1])})
        offsets = base_offsets.copy()
        offsets[0] = 1
        rejected(**{offsets_name: _readonly(offsets)})
        offsets = base_offsets.copy()
        offsets[-1] -= 1
        rejected(**{offsets_name: _readonly(offsets)})
        offsets = base_offsets.copy()
        offsets[1] = offsets[2] + 1
        rejected(**{offsets_name: _readonly(offsets)})
        indices = base_indices.copy()
        indices[0] = config.n_neurons
        rejected(**{indices_name: _readonly(indices)})
        duplicate = np.array([1, 1], dtype=np.uint64)
        duplicate_offsets = np.zeros(config.n_neurons + 1, dtype=np.uint64)
        duplicate_offsets[1:] = 2
        rejected(
            **{
                offsets_name: _readonly(duplicate_offsets),
                indices_name: _readonly(duplicate),
            }
        )
        removed = base_indices.copy()
        edge_position = int(base_offsets[1])
        if edge_position == int(base_offsets[0]):
            edge_position = 0
        removed = np.delete(removed, edge_position)
        removed_offsets = base_offsets.copy()
        owner = int(np.searchsorted(base_offsets, edge_position, side="right") - 1)
        removed_offsets[owner + 1 :] -= 1
        rejected(
            **{
                offsets_name: _readonly(removed_offsets),
                indices_name: _readonly(removed),
            }
        )

    disconnected = np.argwhere(~inputs.topology & ~np.eye(config.n_neurons, dtype=np.bool_))[0]
    excitatory = np.argwhere(inputs.topology[: config.n_excitatory])[0]
    inhibitory = np.argwhere(inputs.topology[config.n_excitatory :])[0]
    ex_pre, ex_post = int(excitatory[0]), int(excitatory[1])
    in_pre = int(inhibitory[0]) + config.n_excitatory
    in_post = int(inhibitory[1])
    weight_cases = (
        (0, 0, 0.1),
        (int(disconnected[0]), int(disconnected[1]), 0.1),
        (ex_pre, ex_post, -0.1),
        (ex_pre, ex_post, config.weight_max + 0.1),
        (in_pre, in_post, 0.1),
        (in_pre, in_post, -config.weight_max - 0.1),
    )
    for pre, post, value in weight_cases:
        weights = result.final_weights.copy()
        weights[pre, post] = value
        rejected(final_weights=_readonly(weights))


def _immutable_outputs(backend: StreamBackend) -> None:
    config = ModelConfig(n_neurons=8, excitatory_fraction=0.75, connectivity=0.25)
    inputs = _fixture(config, 12, 3)
    for array in inputs.__dict__.values():
        array.setflags(write=False)
    result = backend.run(inputs, 1, True, config)
    arrays = [value for value in result.__dict__.values() if type(value) is np.ndarray]
    for array in arrays:
        assert not array.flags.writeable
        try:
            array.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("output WRITEABLE flag could be re-enabled")
        base = array.base
        while isinstance(base, np.ndarray):
            assert not base.flags.writeable
            try:
                base.setflags(write=True)
            except ValueError:
                pass
            else:
                raise AssertionError("writable output base escape")
            base = base.base
        assert isinstance(base, bytes)


def _shadow_rejection(target: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory) / "rust_snn_memory"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "rust_snn_memory.py").write_text("STREAMED_API_VERSION = 2\n", encoding="utf-8")
        code = (
            "from snn_memory.stream_backend import BackendIdentity, load_stream_backend;"
            "load_stream_backend(BackendIdentity(2, '0.1.0', '0'*64))"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = f"{directory}:{target}"
        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode != 0
        assert "compiled extension file" in process.stderr


def _large_replay(backend: StreamBackend) -> dict[str, float | int]:
    config = ModelConfig(n_neurons=2048, excitatory_fraction=0.8, connectivity=0.01)
    inputs = _fixture(config, 1314, 16)
    for array in inputs.__dict__.values():
        array.setflags(write=False)
    started = time.perf_counter()
    first = backend.run(inputs, 4, True, config)
    second = backend.run(inputs, 4, True, config)
    elapsed = time.perf_counter() - started
    for field in first.__dict__:
        left = getattr(first, field)
        right = getattr(second, field)
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        else:
            assert left == right
    assert first.topology_digest == topology_digest(inputs.topology, config.n_excitatory)
    return {
        "neurons": config.n_neurons,
        "timesteps": inputs.packets.shape[0],
        "edges": int(inputs.topology.sum()),
        "two_replays_wall_seconds": elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--install-target", type=Path, required=True)
    arguments = parser.parse_args()
    identity = BackendIdentity(2, "0.1.0", arguments.extension_sha256)
    backend = load_stream_backend(identity)
    adapter_file = sys.modules["snn_memory.stream_backend"].__file__
    assert adapter_file is not None
    adapter_origin = Path(adapter_file).resolve()
    repo = arguments.repo_root.resolve()
    assert not backend.origin.is_relative_to(repo)
    assert not adapter_origin.is_relative_to(repo)
    topology = np.array(
        [[False, True, False], [True, False, False], [True, True, False]], dtype=np.bool_
    )
    assert (
        topology_digest(topology, 2)
        == "78726f211c1f9110c3f074155b5853159eda87dd3732e9aa5852e4b0741fd325"
    )
    _small_adversarial(backend)
    for plasticity in (False, True):
        config = ModelConfig(n_neurons=256, excitatory_fraction=0.8, connectivity=0.03)
        _assert_exact_parity(
            backend, _fixture(config, 256 + int(plasticity), 12), 5, plasticity, config
        )
    _negative_boundaries(backend, identity)
    _result_rejections(backend)
    _immutable_outputs(backend)
    _shadow_rejection(arguments.install_target)
    large = _large_replay(backend)
    print(
        json.dumps(
            {
                "status": "pass",
                "adapter_origin": str(adapter_origin),
                "extension_origin": str(backend.origin),
                "extension_sha256": arguments.extension_sha256,
                "large_replay": large,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
