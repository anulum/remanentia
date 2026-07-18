# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Locked installed-extension adapter for streamed SNN episodes

"""Fail-closed NumPy adapter for the installed streamed Rust backend."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from snn_memory.contracts import ModelConfig

FloatArray: TypeAlias = npt.NDArray[np.float64]
U32Array: TypeAlias = npt.NDArray[np.uint32]
U64Array: TypeAlias = npt.NDArray[np.uint64]
U8Array: TypeAlias = npt.NDArray[np.uint8]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

STREAMED_API_VERSION = 2
TOPOLOGY_DIGEST_FRAMING_VERSION = 2
STATE_DIGEST_FRAMING_VERSION = 2
_TOPOLOGY_DOMAIN = b"remanentia:snn-memory:topology:v2"
_STATE_DOMAIN = b"remanentia:snn-memory:state:v2"


@dataclass(frozen=True)
class BackendIdentity:
    """Independently pinned identity of one installed Rust extension."""

    api_version: int
    crate_version: str
    extension_sha256: str


@dataclass(frozen=True)
class StreamInputs:
    """Exact, conversion-free arrays supplied to one streamed episode."""

    voltage_mv: FloatArray
    refractory_steps: U32Array
    spikes: BoolArray
    pre_trace: FloatArray
    post_trace: FloatArray
    weights: FloatArray
    topology: BoolArray
    packets: FloatArray


@dataclass(frozen=True)
class StreamResult:
    """Immutable compact evidence returned by the streamed backend."""

    spike_offsets: U64Array
    spike_indices: U64Array
    phases: U8Array
    current_values: FloatArray
    current_nonzero_counts: U64Array
    state_digests: tuple[str, ...]
    final_voltage_mv: FloatArray
    final_refractory_steps: U32Array
    final_spikes: BoolArray
    final_pre_trace: FloatArray
    final_post_trace: FloatArray
    final_weights: FloatArray
    topology: BoolArray
    outgoing_offsets: U64Array
    outgoing_indices: U64Array
    incoming_offsets: U64Array
    incoming_indices: U64Array
    topology_digest: str


def _extension_origin(module_name: str) -> Path:
    """Resolve a real extension binary, rejecting source and namespace shadows."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None or spec.loader is None:
        raise RuntimeError(f"{module_name} installed extension is unavailable")
    if not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        raise RuntimeError(f"{module_name} must resolve to a compiled extension file")
    return Path(spec.origin).resolve(strict=True)


def _digest_file(path: Path) -> str:
    """Return the SHA-256 of the binary actually loaded by Python."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_array(
    value: object,
    dtype: np.dtype[np.generic],
    ndim: int,
    label: str,
) -> npt.NDArray[np.generic]:
    """Reject every implicit array, dtype, byte-order, or layout conversion."""
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact numpy.ndarray")
    array = value
    if array.dtype != dtype or not array.dtype.isnative:
        raise TypeError(f"{label} must have native-endian dtype {dtype}")
    if array.ndim != ndim or not array.flags.c_contiguous:
        raise ValueError(f"{label} must be {ndim}D and C-contiguous")
    if np.issubdtype(dtype, np.floating) and not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} must contain only finite values")
    return array


def topology_digest(topology: BoolArray, excitatory_neurons: int) -> str:
    """Reproduce the frozen v2 Rust topology digest framing."""
    neurons = topology.shape[0]
    framed = bytearray()
    framed.extend(len(_TOPOLOGY_DOMAIN).to_bytes(4, "big"))
    framed.extend(_TOPOLOGY_DOMAIN)
    framed.extend(neurons.to_bytes(8, "big"))
    framed.extend(excitatory_neurons.to_bytes(8, "big"))
    framed.extend(topology.size.to_bytes(8, "big"))
    framed.extend(topology.astype(np.uint8, copy=False).tobytes(order="C"))
    return hashlib.sha256(framed).hexdigest()


def state_digest(
    voltage_mv: FloatArray,
    refractory_steps: U32Array,
    spikes: BoolArray,
    pre_trace: FloatArray,
    post_trace: FloatArray,
) -> str:
    """Reproduce the frozen v2 Rust post-timestep state digest framing."""
    framed = bytearray()
    framed.extend(len(_STATE_DOMAIN).to_bytes(4, "big"))
    framed.extend(_STATE_DOMAIN)
    framed.extend(voltage_mv.size.to_bytes(8, "big"))
    components = (
        voltage_mv.astype(">f8", copy=False).tobytes(),
        refractory_steps.astype(">u4", copy=False).tobytes(),
        spikes.astype(np.uint8, copy=False).tobytes(),
        pre_trace.astype(">f8", copy=False).tobytes(),
        post_trace.astype(">f8", copy=False).tobytes(),
    )
    for component in components:
        count = voltage_mv.size
        framed.extend(count.to_bytes(8, "big"))
        framed.extend(component)
    return hashlib.sha256(framed).hexdigest()


class StreamBackend:
    """A verified extension handle with a conversion-free execution surface."""

    def __init__(self, module: ModuleType, origin: Path, identity: BackendIdentity) -> None:
        self._module = module
        self.origin = origin
        self.identity = identity

    def run(
        self,
        inputs: StreamInputs,
        cue_steps: int,
        plasticity_enabled: bool,
        config: ModelConfig,
    ) -> StreamResult:
        """Validate exact inputs, execute transactionally, and validate outputs."""
        if type(cue_steps) is not int or cue_steps < 0:
            raise TypeError("cue_steps must be a non-negative exact int")
        if type(plasticity_enabled) is not bool:
            raise TypeError("plasticity_enabled must be an exact bool")
        n = config.n_neurons
        voltage = _exact_array(inputs.voltage_mv, np.dtype(np.float64), 1, "voltage_mv")
        refractory = _exact_array(
            inputs.refractory_steps, np.dtype(np.uint32), 1, "refractory_steps"
        )
        spikes = _exact_array(inputs.spikes, np.dtype(np.bool_), 1, "spikes")
        pre_trace = _exact_array(inputs.pre_trace, np.dtype(np.float64), 1, "pre_trace")
        post_trace = _exact_array(inputs.post_trace, np.dtype(np.float64), 1, "post_trace")
        weights = _exact_array(inputs.weights, np.dtype(np.float64), 2, "weights")
        topology = _exact_array(inputs.topology, np.dtype(np.bool_), 2, "topology")
        packets = _exact_array(inputs.packets, np.dtype(np.float64), 2, "packets")
        if any(
            array.shape != (n,) for array in (voltage, refractory, spikes, pre_trace, post_trace)
        ):
            raise ValueError("all state arrays must have shape (n_neurons,)")
        if weights.shape != (n, n) or topology.shape != (n, n) or packets.shape[1:] != (n,):
            raise ValueError("weights/topology/packets must match n_neurons")
        if packets.shape[0] == 0 or cue_steps > packets.shape[0]:
            raise ValueError("episode must be nonempty and cue_steps within it")

        rust_config = self._module.ModelConfig(
            config.n_excitatory,
            dt_ms=config.dt_ms,
            tau_m_ms=config.tau_m_ms,
            v_rest_mv=config.v_rest_mv,
            v_reset_mv=config.v_reset_mv,
            v_threshold_mv=config.v_threshold_mv,
            refractory_ms=config.refractory_ms,
            tau_plus_ms=config.tau_plus_ms,
            tau_minus_ms=config.tau_minus_ms,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            excitatory_weight_max=config.weight_max,
        )
        raw = self._module.run_streamed_episode_v2(
            voltage,
            refractory,
            spikes,
            pre_trace,
            post_trace,
            weights,
            topology,
            packets,
            cue_steps,
            plasticity_enabled,
            rust_config,
        )
        result = StreamResult(
            spike_offsets=cast(U64Array, raw.spike_offsets),
            spike_indices=cast(U64Array, raw.spike_indices),
            phases=cast(U8Array, raw.phases),
            current_values=cast(FloatArray, raw.current_values),
            current_nonzero_counts=cast(U64Array, raw.current_nonzero_counts),
            state_digests=tuple(raw.state_digests),
            final_voltage_mv=cast(FloatArray, raw.final_voltage_mv),
            final_refractory_steps=cast(U32Array, raw.final_refractory_steps),
            final_spikes=cast(BoolArray, raw.final_spikes),
            final_pre_trace=cast(FloatArray, raw.final_pre_trace),
            final_post_trace=cast(FloatArray, raw.final_post_trace),
            final_weights=cast(FloatArray, raw.final_weights),
            topology=cast(BoolArray, raw.topology),
            outgoing_offsets=cast(U64Array, raw.outgoing_offsets),
            outgoing_indices=cast(U64Array, raw.outgoing_indices),
            incoming_offsets=cast(U64Array, raw.incoming_offsets),
            incoming_indices=cast(U64Array, raw.incoming_indices),
            topology_digest=str(raw.topology_digest),
        )
        validate_stream_result(result, inputs, cue_steps, config)
        return result


def _exact_output(
    value: object, dtype: np.dtype[np.generic], ndim: int, label: str
) -> npt.NDArray[np.generic]:
    """Require a native, contiguous, NumPy-owned read-only extension result."""
    array = _exact_array(value, dtype, ndim, label)
    if array.flags.writeable:
        raise RuntimeError(f"{label} must be read-only")
    return array


def validate_stream_result(
    result: StreamResult,
    inputs: StreamInputs,
    cue_steps: int,
    config: ModelConfig,
) -> None:
    """Validate one real-shaped streamed result against its episode contract."""
    _exact_output(result.spike_offsets, np.dtype(np.uint64), 1, "spike_offsets")
    _exact_output(result.spike_indices, np.dtype(np.uint64), 1, "spike_indices")
    _exact_output(result.phases, np.dtype(np.uint8), 1, "phases")
    _exact_output(result.current_values, np.dtype(np.float64), 2, "current_values")
    _exact_output(
        result.current_nonzero_counts,
        np.dtype(np.uint64),
        2,
        "current_nonzero_counts",
    )
    _exact_output(result.final_voltage_mv, np.dtype(np.float64), 1, "final_voltage_mv")
    _exact_output(
        result.final_refractory_steps,
        np.dtype(np.uint32),
        1,
        "final_refractory_steps",
    )
    _exact_output(result.final_spikes, np.dtype(np.bool_), 1, "final_spikes")
    _exact_output(result.final_pre_trace, np.dtype(np.float64), 1, "final_pre_trace")
    _exact_output(result.final_post_trace, np.dtype(np.float64), 1, "final_post_trace")
    _exact_output(result.final_weights, np.dtype(np.float64), 2, "final_weights")
    _exact_output(result.topology, np.dtype(np.bool_), 2, "topology")
    _exact_output(result.outgoing_offsets, np.dtype(np.uint64), 1, "outgoing_offsets")
    _exact_output(result.outgoing_indices, np.dtype(np.uint64), 1, "outgoing_indices")
    _exact_output(result.incoming_offsets, np.dtype(np.uint64), 1, "incoming_offsets")
    _exact_output(result.incoming_indices, np.dtype(np.uint64), 1, "incoming_indices")
    timesteps = inputs.packets.shape[0]
    n = config.n_neurons
    if result.spike_offsets.shape != (timesteps + 1,) or result.spike_offsets[0] != 0:
        raise RuntimeError("invalid spike CSR offsets")
    if result.spike_offsets[-1] != result.spike_indices.size:
        raise RuntimeError("invalid spike CSR terminal offset")
    if np.any(result.spike_offsets[1:] < result.spike_offsets[:-1]) or np.any(
        result.spike_indices >= n
    ):
        raise RuntimeError("invalid sparse spike indices")
    for timestep in range(timesteps):
        start = int(result.spike_offsets[timestep])
        stop = int(result.spike_offsets[timestep + 1])
        # Slice the row explicitly: a silent timestep (stop == start, e.g. an empty first row where
        # stop == 0) must not fall into ``stop - 1`` negative indexing.
        row = result.spike_indices[start:stop]
        if np.any(row[1:] <= row[:-1]):
            raise RuntimeError("spike rows must be strictly ascending and unique")
    expected_phases = np.ones(timesteps, dtype=np.uint8)
    expected_phases[:cue_steps] = 0
    if not np.array_equal(result.phases, expected_phases):
        raise RuntimeError("phase evidence mismatch")
    if result.current_values.shape != (timesteps, 12) or result.current_nonzero_counts.shape != (
        timesteps,
        4,
    ):
        raise RuntimeError("current summary shape mismatch")
    if len(result.state_digests) != timesteps or any(
        not _is_lower_sha256(digest) for digest in result.state_digests
    ):
        raise RuntimeError("invalid state digest evidence")
    state_shapes = (
        result.final_voltage_mv,
        result.final_refractory_steps,
        result.final_spikes,
        result.final_pre_trace,
        result.final_post_trace,
    )
    if any(array.shape != (n,) for array in state_shapes) or result.final_weights.shape != (n, n):
        raise RuntimeError("invalid final state shape")
    if not np.array_equal(result.topology, inputs.topology):
        raise RuntimeError("topology changed during episode")
    expected_digest = topology_digest(inputs.topology, config.n_excitatory)
    if not _is_lower_sha256(result.topology_digest) or result.topology_digest != expected_digest:
        raise RuntimeError("topology digest mismatch")
    expected_state_digest = state_digest(
        result.final_voltage_mv,
        result.final_refractory_steps,
        result.final_spikes,
        result.final_pre_trace,
        result.final_post_trace,
    )
    if result.state_digests[-1] != expected_state_digest:
        raise RuntimeError("final state digest mismatch")
    if np.any(result.current_nonzero_counts > n):
        raise RuntimeError("current nonzero count exceeds population")
    _validate_csr(
        result.outgoing_offsets,
        result.outgoing_indices,
        inputs.topology,
        outgoing=True,
    )
    _validate_csr(
        result.incoming_offsets,
        result.incoming_indices,
        inputs.topology,
        outgoing=False,
    )
    weights = result.final_weights
    if (
        np.any(np.diag(weights) != 0.0)
        or np.any(weights[~inputs.topology] != 0.0)
        or np.any(weights[: config.n_excitatory] < 0.0)
        or np.any(weights[: config.n_excitatory] > config.weight_max)
        or np.any(weights[config.n_excitatory :] > 0.0)
        or np.any(weights[config.n_excitatory :] < -config.weight_max)
    ):
        raise RuntimeError("final weights violate topology, diagonal, Dale sign, or bounds")


def _is_lower_sha256(value: str) -> bool:
    """Return whether a value is exactly 64 lowercase hexadecimal characters."""
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_csr(
    offsets: U64Array,
    indices: U64Array,
    topology: BoolArray,
    *,
    outgoing: bool,
) -> None:
    """Require sorted unique CSR to exactly encode one topology orientation."""
    n = topology.shape[0]
    if offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != indices.size:
        raise RuntimeError("invalid adjacency CSR offsets")
    if np.any(offsets[1:] < offsets[:-1]) or np.any(indices >= n):
        raise RuntimeError("invalid adjacency CSR bounds")
    for node in range(n):
        start = int(offsets[node])
        stop = int(offsets[node + 1])
        row = indices[start:stop]
        if np.any(row[1:] <= row[:-1]):
            raise RuntimeError("adjacency CSR rows must be strictly ascending")
        expected = np.flatnonzero(topology[node] if outgoing else topology[:, node]).astype(
            np.uint64,
            copy=False,
        )
        if not np.array_equal(row, expected):
            raise RuntimeError("adjacency CSR and topology disagree")


def load_stream_backend(
    identity: BackendIdentity,
    module_name: str = "rust_snn_memory.rust_snn_memory",
) -> StreamBackend:
    """Load only an exact installed binary with the independently pinned identity."""
    if identity.api_version != STREAMED_API_VERSION:
        raise RuntimeError("requested streamed API version is unsupported")
    if not _is_lower_sha256(identity.extension_sha256):
        raise ValueError("extension_sha256 must be a lowercase SHA-256")
    _extension_origin(module_name)
    module = importlib.import_module(module_name)
    loaded_origin = Path(cast(str, module.__file__)).resolve(strict=True)
    required = {
        "STREAMED_API_VERSION": identity.api_version,
        "CRATE_VERSION": identity.crate_version,
        "TOPOLOGY_DIGEST_FRAMING_VERSION": TOPOLOGY_DIGEST_FRAMING_VERSION,
        "STATE_DIGEST_FRAMING_VERSION": STATE_DIGEST_FRAMING_VERSION,
    }
    for attribute, expected in required.items():
        if getattr(module, attribute, None) != expected:
            raise RuntimeError(f"installed extension has wrong {attribute}")
    _ = (module.run_streamed_episode_v2, module.StreamedEpisodeV2)
    if _digest_file(loaded_origin) != identity.extension_sha256:
        raise RuntimeError("installed extension binary SHA-256 mismatch")
    return StreamBackend(module, loaded_origin, identity)
