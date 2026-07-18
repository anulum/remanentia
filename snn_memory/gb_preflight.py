# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Deterministic G-B completion descriptor over the streamed backend

"""Deterministic G-B descriptor evaluation over the installed streamed Rust surface.

The evaluator consumes a verified :class:`~snn_memory.stream_backend.StreamBackend`
and exact :class:`~snn_memory.stream_backend.StreamInputs`, runs the real
plasticity-disabled streamed episode itself, and derives — from the exported
per-timestep phase vector, current summary matrix, and sparse spike CSR — the
frozen completion descriptor of Amendment 2 (``bd4ec3b4…``), its U-2 binary64
addendum (``ae715600…``), the design-input contract pin (``69452969…``), and the
digest-pin correction (``07ef76b74a29d55c84c258e12a66ed8124b8953867a6830cb32419b9c752437b``).

It never accepts a caller-fabricated current matrix, a Python-reference
substitute, state digests as numeric distances, or a fallback backend. The
installed PyO3 result layout is binding: ``current_values`` columns 3..5 hold the
recurrent excitatory ``signed_sum``/``l1``/``l2`` scalars, columns 6..8 the
inhibitory scalars, and columns 9..11 the recurrent-net scalars; column 1 is the
external L1 that must be exactly zero at every completion row; and column 11 —
the exported recurrent-net ``l2_norm`` scaled-hypot scalar — is the only
admissible ``r_t`` for the U-2 completion energy.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from snn_memory.contracts import ModelConfig
from snn_memory.state import initialise_weights
from snn_memory.stream_backend import (
    BackendIdentity,
    StreamBackend,
    StreamInputs,
    StreamResult,
    load_stream_backend,
)

BINS = 8
TAIL_BINS = (4, 5, 6, 7)
CANDIDATE_LAGS = (1, 2, 3, 4)
EPSILON = 2.220446049250313e-16
EPSILON_BITS = "3cb0000000000000"
_EXTERNAL_L1_COLUMN = 1
_RECURRENT_COLUMNS = (3, 4, 5, 6, 7, 8, 9, 10, 11)
_RECURRENT_NET_L2_COLUMN = 11
_RECURRENT_NET_L1_COLUMN = 10
_TRAJECTORY_CLASSES = frozenset(
    {"invalid", "silent_decay", "settled_fixed", "settled_periodic", "wandering_active"}
)


class GbPreflightError(ValueError):
    """A fail-closed G-B descriptor contract violation."""


@dataclass(frozen=True)
class GbDescriptor:
    """Immutable deterministic completion descriptor for one streamed episode."""

    completion_steps: int
    completion_rows: tuple[
        tuple[int, float, float, float, float, float, float, float, float, float], ...
    ]
    completion_spike_raster: tuple[tuple[int, ...], ...]
    dt_ms: float
    n_neurons: int
    spike_drift_ceiling: float
    current_drift_ceiling: float
    numerical_zero_floor: float
    recurrent_energy: float
    recurrent_energy_bits: str
    half_life_steps: int | None
    trajectory_class: str
    settled: bool
    bin_spike_hamming: tuple[int, ...]
    topology_digest: str
    episode_input_digest: str
    model_config_digest: str
    backend_extension_sha256: str


def _energy_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _require_completion_steps(completion_steps: int) -> int:
    if type(completion_steps) is not int:
        raise GbPreflightError("completion_steps must be an exact int")
    if completion_steps < 32 or completion_steps % BINS != 0:
        raise GbPreflightError("completion_steps must be a multiple of eight and at least 32")
    return completion_steps


def _require_positive_grid_value(value: float, label: str) -> float:
    if not isinstance(value, float) or not np.isfinite(value) or value < 0.0:
        raise GbPreflightError(f"{label} must be a finite non-negative float")
    return value


def _completion_rows(
    result: StreamResult, cue_steps: int, completion_steps: int
) -> np.ndarray[Any, Any]:
    phases = result.phases
    completion_mask = phases == 1
    completion_indices = np.flatnonzero(completion_mask)
    if completion_indices.size != completion_steps:
        raise GbPreflightError("completion phase length differs from the frozen completion_steps")
    if int(completion_indices[0]) != cue_steps or not np.array_equal(
        completion_indices, np.arange(cue_steps, cue_steps + completion_steps)
    ):
        raise GbPreflightError(  # pragma: no cover - real backend tail is always contiguous
            "completion phase is not the contiguous ascending episode tail"
        )
    rows = np.asarray(result.current_values[completion_indices], dtype=np.float64)
    if not bool(np.isfinite(rows).all()):
        raise GbPreflightError(  # pragma: no cover - real backend summaries are finite
            "completion current summary is non-finite"
        )
    if bool(np.any(rows[:, _EXTERNAL_L1_COLUMN] != 0.0)):
        raise GbPreflightError("external L1 must be exactly zero at every completion row")
    return rows


def _recurrent_energy(rows: np.ndarray[Any, Any], dt_ms: float) -> tuple[float, str]:
    # Binding U-2 arithmetic: scalar left fold over ascending completion timesteps,
    # squaring the exported recurrent-net l2_norm scalar (never re-derived), with a
    # single trailing dt_ms multiply. No pairwise/tree/vectorised/fma reduction.
    s = 0.0
    for value in rows[:, _RECURRENT_NET_L2_COLUMN]:
        r_t = float(value)
        if not np.isfinite(r_t) or r_t < 0.0:
            raise GbPreflightError(  # pragma: no cover - real l2_norm is finite non-negative
                "recurrent-net l2_norm must be finite and non-negative"
            )
        q_t = r_t * r_t
        s = s + q_t
    energy = float(dt_ms) * s
    if not np.isfinite(energy) or energy < 0.0:
        raise GbPreflightError(  # pragma: no cover - real energy is finite non-negative
            "recurrent energy must be finite and non-negative"
        )
    return energy, _energy_bits(energy)


def _completion_spike_bitsets(
    result: StreamResult, cue_steps: int, completion_steps: int, n_neurons: int
) -> list[frozenset[int]]:
    offsets = result.spike_offsets
    indices = result.spike_indices
    bitsets: list[frozenset[int]] = []
    for step in range(completion_steps):
        global_step = cue_steps + step
        start = int(offsets[global_step])
        stop = int(offsets[global_step + 1])
        neurons = frozenset(int(neuron) for neuron in indices[start:stop])
        if any(neuron >= n_neurons for neuron in neurons):
            raise GbPreflightError(  # pragma: no cover - real backend indices are in range
                "spike index exceeds the population"
            )
        bitsets.append(neurons)
    return bitsets


def _bin_hamming(bitsets: list[frozenset[int]], bin_high: int, bin_low: int, width: int) -> int:
    total = 0
    for offset in range(width):
        high = bitsets[bin_high * width + offset]
        low = bitsets[bin_low * width + offset]
        total += len(high.symmetric_difference(low))
    return total


def _bin_current(rows: np.ndarray[Any, Any], bin_index: int, width: int) -> np.ndarray[Any, Any]:
    block = rows[bin_index * width : (bin_index + 1) * width]
    return np.asarray(block[:, _RECURRENT_COLUMNS], dtype=np.float64).reshape(-1)


def _l2(vector: np.ndarray[Any, Any]) -> float:
    total = 0.0
    for value in vector:
        scalar = float(value)
        total = total + scalar * scalar
    return float(np.sqrt(total))


def _current_drift(high: np.ndarray[Any, Any], low: np.ndarray[Any, Any]) -> float:
    difference = _l2(high - low)
    denominator = max(_l2(high), _l2(low), EPSILON)
    return difference / denominator


def _half_life(rows: np.ndarray[Any, Any], completion_steps: int, width: int) -> int | None:
    net_l2 = rows[:, _RECURRENT_NET_L2_COLUMN]
    means: list[float] = []
    for bin_index in range(BINS):
        block = net_l2[bin_index * width : (bin_index + 1) * width]
        running = 0.0
        for value in block:
            scalar = float(value)
            running = running + scalar * scalar
        means.append(running / float(width))
    if means[0] == 0.0:
        return 0
    threshold = 0.5 * means[0]
    for bin_index in range(BINS - 1):
        if means[bin_index] <= threshold and means[bin_index + 1] <= threshold:
            return bin_index * width
    return None


def _tail_is_silent(
    rows: np.ndarray[Any, Any],
    bitsets: list[frozenset[int]],
    width: int,
    numerical_zero_floor: float,
) -> bool:
    tail_start = TAIL_BINS[0] * width
    for step in range(tail_start, BINS * width):
        if bitsets[step]:
            return False
        # No preregistered witness episode leaves residual tail current above the zero floor.
        if float(rows[step, _RECURRENT_NET_L1_COLUMN]) > numerical_zero_floor:  # pragma: no cover
            return False
    return True


def _lag_passes(
    rows: np.ndarray[Any, Any],
    bitsets: list[frozenset[int]],
    width: int,
    n_neurons: int,
    lag: int,
    spike_drift_ceiling: float,
    current_drift_ceiling: float,
) -> bool:
    denominator = float(width * n_neurons)
    for tail_bin in TAIL_BINS:
        spike_drift = _bin_hamming(bitsets, tail_bin, tail_bin - lag, width) / denominator
        current_drift = _current_drift(
            _bin_current(rows, tail_bin, width), _bin_current(rows, tail_bin - lag, width)
        )
        if spike_drift > spike_drift_ceiling or current_drift > current_drift_ceiling:
            return False
    return True


def _tail_hamming(bitsets: list[frozenset[int]], width: int) -> tuple[int, ...]:
    return tuple(
        _bin_hamming(bitsets, tail_bin, tail_bin - lag, width)
        for tail_bin in TAIL_BINS
        for lag in CANDIDATE_LAGS
    )


def _classify(
    rows: np.ndarray[Any, Any],
    bitsets: list[frozenset[int]],
    width: int,
    n_neurons: int,
    spike_drift_ceiling: float,
    current_drift_ceiling: float,
    numerical_zero_floor: float,
) -> str:
    if _tail_is_silent(rows, bitsets, width, numerical_zero_floor):
        return "silent_decay"
    if _lag_passes(rows, bitsets, width, n_neurons, 1, spike_drift_ceiling, current_drift_ceiling):
        return "settled_fixed"
    for lag in (2, 3, 4):
        if _lag_passes(  # pragma: no cover - no preregistered periodic witness
            rows, bitsets, width, n_neurons, lag, spike_drift_ceiling, current_drift_ceiling
        ):
            return "settled_periodic"
    return "wandering_active"


def evaluate_gb_descriptor(
    backend: StreamBackend,
    inputs: StreamInputs,
    cue_steps: int,
    config: ModelConfig,
    *,
    completion_steps: int,
    spike_drift_ceiling: float,
    current_drift_ceiling: float,
    numerical_zero_floor: float,
) -> GbDescriptor:
    """Run the real streamed episode and return the frozen completion descriptor.

    Parameters
    ----------
    backend
        A verified installed streamed Rust backend; ``backend.run`` is invoked
        here with plasticity disabled.
    inputs
        Exact conversion-free episode inputs whose ``packets`` supply
        ``cue_steps`` cue rows followed by exactly ``completion_steps`` autonomous
        completion rows.
    cue_steps
        Number of external-input timesteps before the autonomous completion phase.
    config
        The validated model configuration whose ``dt_ms`` scales the U-2 energy.
    completion_steps
        Frozen completion horizon; a positive multiple of eight and at least 32.
    spike_drift_ceiling, current_drift_ceiling, numerical_zero_floor
        Frozen calibration thresholds applied to the tail-bin drift measures and
        the silent-decay recurrent-L1 test.

    Returns
    -------
    GbDescriptor
        The deterministic completion descriptor, U-2 energy bits, half-life,
        trajectory class, and tail-bin spike Hamming distances.

    Raises
    ------
    GbPreflightError
        If the completion telemetry is malformed, the external L1 is non-zero on a
        completion row, the energy is non-finite/negative, or a threshold is invalid.
    """
    if type(backend) is not StreamBackend:
        raise GbPreflightError("a verified StreamBackend is required")
    completion_steps = _require_completion_steps(completion_steps)
    spike_drift_ceiling = _require_positive_grid_value(spike_drift_ceiling, "spike_drift_ceiling")
    current_drift_ceiling = _require_positive_grid_value(
        current_drift_ceiling, "current_drift_ceiling"
    )
    numerical_zero_floor = _require_positive_grid_value(
        numerical_zero_floor, "numerical_zero_floor"
    )
    width = completion_steps // BINS
    result = backend.run(inputs, cue_steps, False, config)
    rows = _completion_rows(result, cue_steps, completion_steps)
    energy, energy_bits = _recurrent_energy(rows, config.dt_ms)
    bitsets = _completion_spike_bitsets(result, cue_steps, completion_steps, config.n_neurons)
    half_life = _half_life(rows, completion_steps, width)
    trajectory = _classify(
        rows,
        bitsets,
        width,
        config.n_neurons,
        spike_drift_ceiling,
        current_drift_ceiling,
        numerical_zero_floor,
    )
    settled = trajectory in ("settled_fixed", "settled_periodic")
    completion_rows = tuple(
        (
            cue_steps + step,
            float(rows[step, 3]),
            float(rows[step, 4]),
            float(rows[step, 5]),
            float(rows[step, 6]),
            float(rows[step, 7]),
            float(rows[step, 8]),
            float(rows[step, 9]),
            float(rows[step, 10]),
            float(rows[step, 11]),
        )
        for step in range(completion_steps)
    )
    completion_spike_raster = tuple(tuple(sorted(bitset)) for bitset in bitsets)
    return GbDescriptor(
        completion_steps=completion_steps,
        completion_rows=completion_rows,
        completion_spike_raster=completion_spike_raster,
        dt_ms=float(config.dt_ms),
        n_neurons=config.n_neurons,
        spike_drift_ceiling=spike_drift_ceiling,
        current_drift_ceiling=current_drift_ceiling,
        numerical_zero_floor=numerical_zero_floor,
        recurrent_energy=energy,
        recurrent_energy_bits=energy_bits,
        half_life_steps=half_life,
        trajectory_class=trajectory,
        settled=settled,
        bin_spike_hamming=_tail_hamming(bitsets, width),
        topology_digest=result.topology_digest,
        episode_input_digest=episode_input_digest(inputs, cue_steps),
        model_config_digest=model_config_digest(config),
        backend_extension_sha256=backend.identity.extension_sha256,
    )


def _framed_digest(domain: bytes, chunks: tuple[bytes, ...]) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    digest.update(len(chunks).to_bytes(8, "big"))
    for chunk in chunks:
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest()


def episode_input_digest(inputs: StreamInputs, cue_steps: int) -> str:
    """Return a domain-separated digest binding the exact streamed episode inputs."""
    chunks = (
        cue_steps.to_bytes(8, "big"),
        inputs.voltage_mv.astype(">f8", copy=False).tobytes(),
        inputs.refractory_steps.astype(">u4", copy=False).tobytes(),
        inputs.spikes.astype(np.uint8, copy=False).tobytes(),
        inputs.pre_trace.astype(">f8", copy=False).tobytes(),
        inputs.post_trace.astype(">f8", copy=False).tobytes(),
        inputs.weights.astype(">f8", copy=False).tobytes(),
        inputs.topology.astype(np.uint8, copy=False).tobytes(),
        inputs.packets.astype(">f8", copy=False).tobytes(),
    )
    return _framed_digest(b"remanentia:snn-v2-gb-episode:v1\0", chunks)


def model_config_digest(config: ModelConfig) -> str:
    """Return a domain-separated digest over the canonical model configuration."""
    import json

    raw = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _framed_digest(b"remanentia:snn-v2-gb-model-config:v1\0", (raw,))


def build_episode(
    config: ModelConfig, seed: int, cue_steps: int, completion_steps: int
) -> StreamInputs:
    """Build one deterministic real streamed episode from immutable seeds.

    The weights/topology come from :func:`snn_memory.state.initialise_weights` and
    the cue-phase packets from a seed-domain-separated generator, so two fresh
    processes with identical parameters produce byte-identical inputs, descriptors,
    and sealed evidence artifacts.
    """
    weights, topology = initialise_weights(config, seed)
    rng = np.random.default_rng(seed + 900_000)
    timesteps = cue_steps + completion_steps
    packets = np.zeros((timesteps, config.n_neurons), dtype=np.float64)
    for step in range(cue_steps):
        indices = rng.choice(config.n_neurons, size=max(2, config.n_neurons // 8), replace=False)
        packets[step, indices] = 18.0
    spikes = np.zeros(config.n_neurons, dtype=np.bool_)
    spikes[[0, config.n_excitatory]] = True
    return StreamInputs(
        np.full(config.n_neurons, config.v_rest_mv, dtype=np.float64),
        np.zeros(config.n_neurons, dtype=np.uint32),
        spikes,
        rng.random(config.n_neurons, dtype=np.float64) * 0.1,
        rng.random(config.n_neurons, dtype=np.float64) * 0.1,
        weights,
        topology,
        packets,
    )


def descriptor_payload(
    descriptor: GbDescriptor,
    *,
    lane_role: str,
    state: str,
    calibration_spec_digest: str,
    scoring_target_digest: str,
) -> dict[str, Any]:
    """Build the schema-conformant gb-preflight-evidence payload without the self digest."""
    return {
        "schema_version": 2,
        "artifact_type": "snn-memory-gb-preflight-evidence-v2",
        "state": state,
        "lane_role": lane_role,
        "epsilon_bits": EPSILON_BITS,
        "completion_steps": descriptor.completion_steps,
        "bins": BINS,
        "dt_ms": descriptor.dt_ms,
        "n_neurons": descriptor.n_neurons,
        "spike_drift_ceiling": descriptor.spike_drift_ceiling,
        "current_drift_ceiling": descriptor.current_drift_ceiling,
        "numerical_zero_floor": descriptor.numerical_zero_floor,
        "completion_rows": [
            {
                "timestep": row[0],
                "e_signed_sum": row[1],
                "e_l1": row[2],
                "e_l2": row[3],
                "i_signed_sum": row[4],
                "i_l1": row[5],
                "i_l2": row[6],
                "net_signed_sum": row[7],
                "net_l1": row[8],
                "net_l2": row[9],
            }
            for row in descriptor.completion_rows
        ],
        "completion_spike_raster": [list(step) for step in descriptor.completion_spike_raster],
        "recurrent_energy_bits": descriptor.recurrent_energy_bits,
        "recurrent_energy": descriptor.recurrent_energy,
        "half_life_steps": descriptor.half_life_steps,
        "trajectory_class": descriptor.trajectory_class,
        "settled": descriptor.settled,
        "bin_spike_hamming": list(descriptor.bin_spike_hamming),
        "calibration_spec_digest": calibration_spec_digest,
        "scoring_target_digest": scoring_target_digest,
        "backend_extension_sha256": descriptor.backend_extension_sha256,
        "model_config_digest": descriptor.model_config_digest,
        "topology_digest": descriptor.topology_digest,
        "episode_input_digest": descriptor.episode_input_digest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one deterministic real G-B episode and seal the evidence artifact.

    The evidence artifact carries no process identity, so two fresh processes with
    identical parameters produce byte-identical sealed evidence; the caller records
    the distinct process IDs in its own envelope.
    """
    from snn_memory.experiment_lock import ExperimentLockError, write_artifact

    parser = argparse.ArgumentParser(prog="python -m snn_memory.gb_preflight")
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--crate-version", default="0.1.0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cue-steps", type=int, required=True)
    parser.add_argument("--completion-steps", type=int, required=True)
    parser.add_argument("--n-neurons", type=int, required=True)
    parser.add_argument("--excitatory-fraction", type=float, required=True)
    parser.add_argument("--connectivity", type=float, required=True)
    parser.add_argument("--dt-ms", type=float, required=True)
    parser.add_argument("--spike-drift-ceiling", type=float, required=True)
    parser.add_argument("--current-drift-ceiling", type=float, required=True)
    parser.add_argument("--numerical-zero-floor", type=float, required=True)
    parser.add_argument("--lane-role", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--calibration-spec-digest", required=True)
    parser.add_argument("--scoring-target-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        backend = load_stream_backend(
            BackendIdentity(2, arguments.crate_version, arguments.extension_sha256)
        )
        config = ModelConfig(
            n_neurons=arguments.n_neurons,
            excitatory_fraction=arguments.excitatory_fraction,
            connectivity=arguments.connectivity,
            dt_ms=arguments.dt_ms,
        )
        inputs = build_episode(
            config, arguments.seed, arguments.cue_steps, arguments.completion_steps
        )
        descriptor = evaluate_gb_descriptor(
            backend,
            inputs,
            arguments.cue_steps,
            config,
            completion_steps=arguments.completion_steps,
            spike_drift_ceiling=arguments.spike_drift_ceiling,
            current_drift_ceiling=arguments.current_drift_ceiling,
            numerical_zero_floor=arguments.numerical_zero_floor,
        )
        payload = descriptor_payload(
            descriptor,
            lane_role=arguments.lane_role,
            state=arguments.state,
            calibration_spec_digest=arguments.calibration_spec_digest,
            scoring_target_digest=arguments.scoring_target_digest,
        )
        artifact = write_artifact(payload, arguments.output)
    except (OSError, GbPreflightError, ExperimentLockError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "file_sha256": artifact.file_sha256,
                "payload_self_sha256": artifact.payload_self_sha256,
                "recurrent_energy_bits": descriptor.recurrent_energy_bits,
                "trajectory_class": descriptor.trajectory_class,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
