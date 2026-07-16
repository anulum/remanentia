# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN schema-v2 checkpoint-bundle contract

"""Authenticated checkpoint-bundle writer and single-read byte validator.

The bundle is the accepted ``snn_memory_checkpoint_v2`` manifest plus a canonical
checkpoint-bundle descriptor. The descriptor binds the manifest and the complete,
exact, unique numeric/JSON component inventory. Every component is written once,
digested by a versioned role domain over its canonical little-endian C-contiguous
bytes (or canonical JSON), and re-derived on read from the captured bytes through a
symlink-refusing descriptor-relative walk. The writer is transactional and
no-clobber; a partial failure leaves no accepted target.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from snn_memory.stream_backend import state_digest, topology_digest

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
U32Array = npt.NDArray[np.uint32]
U64Array = npt.NDArray[np.uint64]

_BUNDLE_SCHEMA = "snn_memory_checkpoint_bundle_v2.schema.json"
_CHECKPOINT_SCHEMA = "snn_memory_checkpoint_v2.schema.json"

_COMPONENT_DOMAIN = b"remanentia:snn-memory:checkpoint-component:v2\0"
_CSR_DOMAIN = b"remanentia:snn-memory:adjacency-csr:v2\0"
_REPLAY_DOMAIN = b"remanentia:snn-memory:replay-schedule:v2\0"
_RECORD_IDS_DOMAIN = b"remanentia:snn-memory:ordered-record-ids:v2"
_DESCRIPTOR_DOMAIN = b"remanentia:snn-memory:checkpoint-bundle-descriptor:v2\0"

_COMPONENT_FRAMING = 2
_STATE_BYTES_PER_NEURON = 8 + 4 + 1 + 8 + 8

_COMPONENT_ORDER = (
    "topology", "adjacency_outgoing", "adjacency_incoming", "weights",
    "training_final_state", "probe_initial_state", "signatures", "record_ids",
    "replay_schedule",
)
_COMPONENT_PATHS = {
    "topology": "topology.bin",
    "adjacency_outgoing": "adjacency_outgoing.bin",
    "adjacency_incoming": "adjacency_incoming.bin",
    "weights": "weights.bin",
    "training_final_state": "training_final_state.bin",
    "probe_initial_state": "probe_initial_state.bin",
    "signatures": "signatures.bin",
    "record_ids": "record_ids.json",
    "replay_schedule": "replay_schedule.json",
}
_MANIFEST_PATH = "manifest.json"
_DESCRIPTOR_PATH = "descriptor.json"


class CheckpointBundleError(ValueError):
    """Raised when a checkpoint bundle fails a contract, byte, or digest check."""


@dataclass(frozen=True)
class StateArrays:
    """One immutable membrane/refractory/spike/trace network state snapshot."""

    voltage_mv: FloatArray
    refractory_steps: U32Array
    spikes: BoolArray
    pre_trace: FloatArray
    post_trace: FloatArray


@dataclass(frozen=True)
class CsrArrays:
    """One immutable CSR adjacency orientation (offsets and sorted indices)."""

    offsets: U64Array
    indices: U64Array


@dataclass(frozen=True)
class CheckpointBundleInputs:
    """Every array and schedule a checkpoint bundle seals for one (seed, condition)."""

    topology: BoolArray
    outgoing: CsrArrays
    incoming: CsrArrays
    weights: FloatArray
    training_final_state: StateArrays
    probe_initial_state: StateArrays
    signatures: FloatArray
    ordered_record_ids: tuple[str, ...]
    replay_schedule: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ValidatedCheckpointBundle:
    """An authenticated checkpoint bundle with read-only descriptor and manifest.

    ``inputs`` carries the independently owned, read-only arrays, state, and replay
    schedule decoded once from the captured bytes so a caller consumes them without
    reopening any component file.
    """

    descriptor: Mapping[str, Any]
    manifest: Mapping[str, Any]
    descriptor_self_sha256: str
    manifest_file_sha256: str
    manifest_payload_self_sha256: str
    inputs: CheckpointBundleInputs


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _framed(domain: bytes, chunks: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    digest.update(len(chunks).to_bytes(8, "big"))
    for chunk in chunks:
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest()


def _self_digest(payload: Mapping[str, Any], domain: bytes) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "self_sha256"}
    canonical = _canonical_bytes(unsigned)
    return hashlib.sha256(domain + len(canonical).to_bytes(8, "big") + canonical).hexdigest()


def ordered_record_ids_digest(record_ids: Sequence[str]) -> str:
    """Reproduce the accepted v2 ordered-record-id identity of a candidate set."""
    return _framed(_RECORD_IDS_DOMAIN, [record_id.encode("utf-8") for record_id in record_ids])


def replay_schedule_digest(schedule: Sequence[Mapping[str, Any]]) -> str:
    """Hash the canonical replay schedule binding epoch, record, position, and steps."""
    canonical = [
        {
            "epoch": int(entry["epoch"]),
            "record_id": str(entry["record_id"]),
            "replay_position": int(entry["replay_position"]),
            "timesteps": int(entry["timesteps"]),
        }
        for entry in schedule
    ]
    return _sha(_REPLAY_DOMAIN + _canonical_bytes(canonical))


def _le(array: npt.NDArray[Any], dtype: str) -> npt.NDArray[Any]:
    return np.ascontiguousarray(array, dtype=np.dtype(dtype))


def checkpoint_component_digest(role: str, dtype: str, array: npt.NDArray[Any]) -> str:
    """Digest a numeric component: role domain, dtype, shape, canonical LE C bytes."""
    canonical = _le(array, dtype)
    shape = b"".join(int(dim).to_bytes(8, "big") for dim in canonical.shape)
    return _framed(
        _COMPONENT_DOMAIN,
        [role.encode("ascii"), dtype.encode("ascii"), shape, canonical.tobytes(order="C")],
    )


def _csr_digest(role: str, csr: CsrArrays) -> str:
    offsets = _le(csr.offsets, "<u8")
    indices = _le(csr.indices, "<u8")
    return _framed(_CSR_DOMAIN, [role.encode("ascii"), offsets.tobytes(), indices.tobytes()])


def _csr_bytes(csr: CsrArrays) -> bytes:
    return _le(csr.offsets, "<u8").tobytes() + _le(csr.indices, "<u8").tobytes()


def _state_bytes(state: StateArrays) -> bytes:
    return (
        _le(state.voltage_mv, "<f8").tobytes()
        + _le(state.refractory_steps, "<u4").tobytes()
        + _le(state.spikes, "|b1").tobytes()
        + _le(state.pre_trace, "<f8").tobytes()
        + _le(state.post_trace, "<f8").tobytes()
    )


def _state_digest(state: StateArrays) -> str:
    return state_digest(
        _le(state.voltage_mv, "<f8"),
        _le(state.refractory_steps, "<u4"),
        _le(state.spikes, "|b1"),
        _le(state.pre_trace, "<f8"),
        _le(state.post_trace, "<f8"),
    )


def _schema_bytes(name: str) -> bytes:
    return files("snn_memory").joinpath("schema", name).read_bytes()


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise CheckpointBundleError(f"{label} contains a non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, item in pairs:
            if key in seen:
                raise CheckpointBundleError(f"{label} contains a duplicate object key {key!r}")
            seen[key] = item
        return seen

    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointBundleError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CheckpointBundleError(f"{label} root must be an object")
    return cast(dict[str, Any], value)


def _validate_schema(value: Mapping[str, Any], schema_name: str, label: str) -> None:
    schema = json.loads(_schema_bytes(schema_name).decode("utf-8"))
    validate = cast(Callable[[object, object], None], import_module("jsonschema").validate)
    try:
        validate(value, schema)
    except Exception as error:  # noqa: BLE001 - jsonschema raises its own hierarchy
        raise CheckpointBundleError(f"{label} failed schema {schema_name}: {error}") from error


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _open_dir_from_root(path: Path, label: str) -> int:
    """Open a directory by walking every component from the filesystem root, refusing symlinks.

    Each level is opened ``O_DIRECTORY|O_NOFOLLOW`` relative to the previous descriptor, so no
    intermediate component can be a symlink or be raced. The caller owns and closes the result.
    """
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise CheckpointBundleError(f"{label} filesystem root cannot be opened safely") from error
    try:
        for part in absolute.parts[1:]:
            if part in (os.curdir, os.pardir):
                raise CheckpointBundleError(f"{label} path carries a non-normal '{part}' component")
            try:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            except OSError as error:
                raise CheckpointBundleError(f"{label} cannot be opened safely") from error
            os.close(current)
            current = nxt
    except BaseException:
        os.close(current)
        raise
    return current


_RENAME_NOREPLACE = 1


def _renameat2_noreplace(old_dir_fd: int, old_name: str, new_dir_fd: int, new_name: str) -> None:
    """Atomically install ``old_name`` as ``new_name`` and fail closed if the target exists.

    Uses ``renameat2(RENAME_NOREPLACE)`` relative to the given directory descriptors, so a
    competing creator cannot be clobbered and no intermediate component is re-resolved; unlike
    ``os.replace`` this never overwrites an existing (even empty) target directory.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    result = libc.renameat2(
        old_dir_fd, os.fsencode(old_name), new_dir_fd, os.fsencode(new_name), _RENAME_NOREPLACE
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise CheckpointBundleError("checkpoint bundle target already exists")
        raise CheckpointBundleError(f"atomic no-replace install failed: {os.strerror(code)}")


def _write_all_at(dir_fd: int, name: str, data: bytes) -> None:
    """Write ``data`` in full to a fresh no-follow, no-clobber file relative to ``dir_fd``."""
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd
    )
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_staging_at(parent_fd: int, name: str) -> None:
    try:
        staging_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError:
        return
    try:
        for entry in os.listdir(staging_fd):
            try:
                os.unlink(entry, dir_fd=staging_fd)
            except OSError:
                pass
    finally:
        os.close(staging_fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except OSError:
        pass


def _install_bundle_directory(target: Path, files: Mapping[str, bytes]) -> None:
    """Stage every file and atomically install the completed directory with no clobber.

    The parent is opened by a root-anchored symlink-refusing walk; staging creation, writes,
    the ``renameat2(RENAME_NOREPLACE)`` install, and both fsyncs are all performed relative to
    that verified descriptor, so no intermediate component can be a symlink or be raced.
    """
    absolute = target if target.is_absolute() else Path.cwd() / target
    final = absolute.name
    if final in (os.curdir, os.pardir, ""):
        raise CheckpointBundleError("checkpoint bundle target has no valid final component")
    parent_fd = _open_dir_from_root(absolute.parent, "checkpoint bundle parent directory")
    try:
        staging_name = f".{final}.staging-{os.urandom(8).hex()}"
        os.mkdir(staging_name, 0o755, dir_fd=parent_fd)
        try:
            staging_fd = os.open(
                staging_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            try:
                for name, data in files.items():
                    _write_all_at(staging_fd, name, data)
                os.fsync(staging_fd)
            finally:
                os.close(staging_fd)
            _renameat2_noreplace(parent_fd, staging_name, parent_fd, final)
        except BaseException:
            _remove_staging_at(parent_fd, staging_name)
            raise
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _read_regular_bytes(directory_fd: int, name: str, label: str) -> bytes:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as error:
        raise CheckpointBundleError(f"{label} cannot be opened safely") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CheckpointBundleError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _component_records(inputs: CheckpointBundleInputs, n_excitatory: int) -> dict[str, dict[str, Any]]:
    """Build the exact numeric/JSON component inventory and its semantic digests."""
    topology_bytes = _le(inputs.topology, "|b1").tobytes()
    weights_bytes = _le(inputs.weights, "<f8").tobytes()
    signatures_bytes = _le(inputs.signatures, "<f8").tobytes()
    record_ids_bytes = _canonical_bytes(list(inputs.ordered_record_ids))
    schedule_bytes = _canonical_bytes(
        [
            {
                "epoch": int(entry["epoch"]),
                "record_id": str(entry["record_id"]),
                "replay_position": int(entry["replay_position"]),
                "timesteps": int(entry["timesteps"]),
            }
            for entry in inputs.replay_schedule
        ]
    )
    payloads: dict[str, dict[str, Any]] = {
        "topology": {
            "dtype": "|b1", "shape": list(inputs.topology.shape), "bytes": topology_bytes,
            "semantic": topology_digest(_le(inputs.topology, "|b1"), n_excitatory),
        },
        "adjacency_outgoing": {
            "dtype": "<u8",
            "shape": [int(inputs.outgoing.offsets.size + inputs.outgoing.indices.size)],
            "bytes": _csr_bytes(inputs.outgoing), "semantic": _csr_digest("outgoing", inputs.outgoing),
        },
        "adjacency_incoming": {
            "dtype": "<u8",
            "shape": [int(inputs.incoming.offsets.size + inputs.incoming.indices.size)],
            "bytes": _csr_bytes(inputs.incoming), "semantic": _csr_digest("incoming", inputs.incoming),
        },
        "weights": {
            "dtype": "<f8", "shape": list(inputs.weights.shape), "bytes": weights_bytes,
            "semantic": checkpoint_component_digest("weights", "<f8", inputs.weights),
        },
        "training_final_state": {
            "dtype": "state", "shape": [int(inputs.training_final_state.voltage_mv.size)],
            "bytes": _state_bytes(inputs.training_final_state),
            "semantic": _state_digest(inputs.training_final_state),
        },
        "probe_initial_state": {
            "dtype": "state", "shape": [int(inputs.probe_initial_state.voltage_mv.size)],
            "bytes": _state_bytes(inputs.probe_initial_state),
            "semantic": _state_digest(inputs.probe_initial_state),
        },
        "signatures": {
            "dtype": "<f8", "shape": list(inputs.signatures.shape), "bytes": signatures_bytes,
            "semantic": checkpoint_component_digest("signatures", "<f8", inputs.signatures),
        },
        "record_ids": {
            "dtype": "json", "shape": [len(inputs.ordered_record_ids)], "bytes": record_ids_bytes,
            "semantic": ordered_record_ids_digest(inputs.ordered_record_ids),
        },
        "replay_schedule": {
            "dtype": "json", "shape": [len(inputs.replay_schedule)], "bytes": schedule_bytes,
            "semantic": replay_schedule_digest(inputs.replay_schedule),
        },
    }
    records: dict[str, dict[str, Any]] = {}
    for role in _COMPONENT_ORDER:
        payload = payloads[role]
        body: bytes = payload["bytes"]
        records[role] = {
            "path": _COMPONENT_PATHS[role],
            "role": role,
            "file_sha256": _sha(body),
            "semantic_digest": str(payload["semantic"]),
            "dtype": str(payload["dtype"]),
            "shape": list(payload["shape"]),
            "byte_length": len(body),
            "framing_version": _COMPONENT_FRAMING,
            "bytes": body,
        }
    return records


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointBundleError(message)


def _check_manifest_bindings(manifest: Mapping[str, Any], records: Mapping[str, dict[str, Any]],
                             ordered_record_ids: Sequence[str]) -> None:
    adjacency = manifest["adjacency"]
    arrays = manifest["arrays"]
    topology_semantic = records["topology"]["semantic_digest"]
    _require(manifest["topology_digest"] == topology_semantic, "manifest topology digest mismatch")
    _require(adjacency["topology_digest"] == topology_semantic, "manifest adjacency topology mismatch")
    _require(adjacency["outgoing_digest"] == records["adjacency_outgoing"]["semantic_digest"],
             "manifest outgoing adjacency digest mismatch")
    _require(adjacency["incoming_digest"] == records["adjacency_incoming"]["semantic_digest"],
             "manifest incoming adjacency digest mismatch")
    _require(arrays["weights_digest"] == records["weights"]["semantic_digest"],
             "manifest weights digest mismatch")
    _require(arrays["training_final_state_digest"] == records["training_final_state"]["semantic_digest"],
             "manifest training-final state digest mismatch")
    _require(arrays["probe_initial_state_digest"] == records["probe_initial_state"]["semantic_digest"],
             "manifest probe initial state digest mismatch")
    _require(arrays["signatures_digest"] == records["signatures"]["semantic_digest"],
             "manifest signatures digest mismatch")
    record_ids_semantic = records["record_ids"]["semantic_digest"]
    _require(arrays["record_ids_digest"] == record_ids_semantic, "manifest record-ids digest mismatch")
    _require(manifest["candidate_set_digest"] == record_ids_semantic,
             "manifest candidate-set digest does not bind the ordered records")
    _require(manifest["replay_order_digest"] == records["replay_schedule"]["semantic_digest"],
             "manifest replay-order digest mismatch")
    _require(tuple(manifest["ordered_record_ids"]) == tuple(ordered_record_ids),
             "manifest record order differs from the sealed record IDs")


def _read_only(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    owned = np.array(array, copy=True)
    owned.setflags(write=False)
    return owned


def _validate_topology(topology: BoolArray, n_neurons: int) -> None:
    if topology.dtype != np.bool_:
        raise CheckpointBundleError("topology is not a boolean array")
    if topology.shape != (n_neurons, n_neurons):
        raise CheckpointBundleError("topology is not a square population matrix")
    if bool(np.any(np.diagonal(topology))):
        raise CheckpointBundleError("topology carries a self-connection on the diagonal")


def _validate_csr(csr: CsrArrays, topology: BoolArray, *, outgoing: bool) -> None:
    n = topology.shape[0]
    offsets, indices = csr.offsets, csr.indices
    orientation = "outgoing" if outgoing else "incoming"
    if offsets.dtype != np.uint64 or indices.dtype != np.uint64:
        raise CheckpointBundleError(f"{orientation} adjacency arrays are not <u8")
    if offsets.shape != (n + 1,):
        raise CheckpointBundleError(f"{orientation} adjacency offsets length mismatch")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(indices.size):
        raise CheckpointBundleError(f"{orientation} adjacency offsets do not bracket the indices")
    if bool(np.any(np.diff(offsets.astype(np.int64)) < 0)):
        raise CheckpointBundleError(f"{orientation} adjacency offsets are not monotonic")
    for node in range(n):
        start, stop = int(offsets[node]), int(offsets[node + 1])
        row = indices[start:stop]
        if bool(np.any(row[1:] <= row[:-1])):
            raise CheckpointBundleError(f"{orientation} adjacency row {node} is not strictly ascending")
        expected = np.flatnonzero(topology[node] if outgoing else topology[:, node]).astype(np.uint64)
        if row.size != expected.size or bool(np.any(row != expected)):
            raise CheckpointBundleError(f"{orientation} adjacency row {node} disagrees with the topology")


def _validate_weights(weights: FloatArray, topology: BoolArray, n_excitatory: int,
                      weight_max: float) -> None:
    n = topology.shape[0]
    if weights.dtype != np.float64 or weights.shape != (n, n):
        raise CheckpointBundleError("weights are not an <f8 square population matrix")
    if not bool(np.isfinite(weights).all()):
        raise CheckpointBundleError("weights carry a non-finite value")
    if bool(np.any(np.diagonal(weights) != 0.0)):
        raise CheckpointBundleError("weights carry a self-connection on the diagonal")
    if bool(np.any(weights[~topology] != 0.0)):
        raise CheckpointBundleError("weights are non-zero off the sealed topology")
    if bool(np.any(weights[:n_excitatory] < 0.0)):
        raise CheckpointBundleError("an excitatory row carries a negative weight")
    if bool(np.any(weights[n_excitatory:] > 0.0)):
        raise CheckpointBundleError("an inhibitory row carries a positive weight")
    if bool(np.any(np.abs(weights) > weight_max)):
        raise CheckpointBundleError("a weight magnitude exceeds the sealed weight bound")


def _validate_state(state: StateArrays, n_neurons: int, label: str) -> None:
    if state.voltage_mv.shape != (n_neurons,) or state.pre_trace.shape != (n_neurons,):
        raise CheckpointBundleError(f"{label} float fields have the wrong length")
    if state.post_trace.shape != (n_neurons,) or state.refractory_steps.shape != (n_neurons,):
        raise CheckpointBundleError(f"{label} refractory or trace fields have the wrong length")
    if state.spikes.shape != (n_neurons,):
        raise CheckpointBundleError(f"{label} spike field has the wrong length")
    if state.refractory_steps.dtype != np.uint32 or state.spikes.dtype != np.bool_:
        raise CheckpointBundleError(f"{label} refractory or spike dtype mismatch")
    for field, name in ((state.voltage_mv, "voltage"), (state.pre_trace, "pre-trace"),
                        (state.post_trace, "post-trace")):
        if field.dtype != np.float64 or not bool(np.isfinite(field).all()):
            raise CheckpointBundleError(f"{label} {name} is not finite <f8")


def _validate_replay_schedule(schedule: Sequence[Mapping[str, Any]], ordered_record_ids: Sequence[str],
                              seed: int, epochs_completed: int) -> None:
    if epochs_completed == 0:
        if schedule:
            raise CheckpointBundleError("an untrained checkpoint must carry no replay entries")
        return
    n = len(ordered_record_ids)
    by_epoch: dict[int, list[Mapping[str, Any]]] = {}
    for entry in schedule:
        strict = _strict_replay_entry(dict(entry))
        by_epoch.setdefault(int(strict["epoch"]), []).append(strict)
    if sorted(by_epoch) != list(range(epochs_completed)):
        raise CheckpointBundleError("replay schedule does not cover every completed epoch exactly once")
    record_timesteps: dict[str, int] = {}
    for epoch in range(epochs_completed):
        entries = by_epoch[epoch]
        if sorted(int(entry["replay_position"]) for entry in entries) != list(range(n)):
            raise CheckpointBundleError(f"epoch {epoch} replay positions are not a full permutation")
        permutation = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(n)
        ordered = sorted(entries, key=lambda entry: int(entry["replay_position"]))
        for position, entry in enumerate(ordered):
            expected = ordered_record_ids[int(permutation[position])]
            if str(entry["record_id"]) != expected:
                raise CheckpointBundleError(f"epoch {epoch} replay order breaks the seeded permutation")
            timesteps = int(entry["timesteps"])
            if timesteps <= 0:
                raise CheckpointBundleError("replay entry carries a non-positive timestep count")
            prior = record_timesteps.setdefault(str(entry["record_id"]), timesteps)
            if prior != timesteps:
                raise CheckpointBundleError("a record replays with inconsistent timestep counts")


def _reconstruct_model(identities: Mapping[str, Any]) -> Any:
    """Rebuild the sealed model configuration and cross-check its authenticated digests."""
    from snn_memory.contracts import ModelConfig
    from snn_memory.experiment_lock import canonical_config_digest

    sealed = dict(identities["model_config"])
    try:
        model = ModelConfig(**sealed)
    except (TypeError, ValueError) as error:
        raise CheckpointBundleError(f"descriptor model configuration is invalid: {error}") from error
    if sealed != model.to_dict():
        raise CheckpointBundleError("descriptor model config is not the full canonical configuration")
    if canonical_config_digest(model.to_dict()) != identities["model_config_digest"]:
        raise CheckpointBundleError("descriptor model config does not match its model-config digest")
    if model.n_neurons != int(identities["n_neurons"]):
        raise CheckpointBundleError("descriptor neuron count disagrees with the model configuration")
    if model.n_excitatory != int(identities["n_excitatory"]):
        raise CheckpointBundleError("descriptor excitatory count disagrees with the model configuration")
    if float(model.weight_max) != float(identities["weight_max"]):
        raise CheckpointBundleError("descriptor weight bound disagrees with the model configuration")
    return model


def _seed_topology(model: Any, seed: int) -> BoolArray:
    from snn_memory.state import initialise_weights

    _, topology = initialise_weights(model, seed)
    return topology


def _require_fresh_reset(state: StateArrays, model: Any, label: str) -> None:
    if not bool(np.all(state.voltage_mv == np.float64(model.v_rest_mv))):
        raise CheckpointBundleError(f"{label} voltage is not the fresh resting potential")
    if bool(np.any(state.refractory_steps != 0)) or bool(np.any(state.spikes)):
        raise CheckpointBundleError(f"{label} refractory or spike field is not a fresh reset")
    if bool(np.any(state.pre_trace != 0.0)) or bool(np.any(state.post_trace != 0.0)):
        raise CheckpointBundleError(f"{label} synaptic trace field is not a fresh reset")


def _validate_semantics(inputs: CheckpointBundleInputs, model: Any, expected_topology: BoolArray,
                        condition: str, seed: int, epochs_completed: int) -> None:
    n_neurons = model.n_neurons
    _validate_topology(inputs.topology, n_neurons)
    if inputs.topology.shape != expected_topology.shape or not bool(
        np.array_equal(inputs.topology, expected_topology)
    ):
        raise CheckpointBundleError("sealed topology differs from the seed-derived connectivity")
    _validate_csr(inputs.outgoing, inputs.topology, outgoing=True)
    _validate_csr(inputs.incoming, inputs.topology, outgoing=False)
    _validate_weights(inputs.weights, inputs.topology, model.n_excitatory, float(model.weight_max))
    _validate_state(inputs.training_final_state, n_neurons, "training-final state")
    _validate_state(inputs.probe_initial_state, n_neurons, "probe-initial state")
    _require_fresh_reset(inputs.probe_initial_state, model, "probe-initial state")
    rows = len(inputs.ordered_record_ids)
    if inputs.signatures.dtype != np.float64 or inputs.signatures.shape != (rows, 8 * n_neurons):
        raise CheckpointBundleError("signatures are not an <f8 (records, eight-bins-per-neuron) block")
    if not bool(np.isfinite(inputs.signatures).all()):
        raise CheckpointBundleError("signatures carry a non-finite value")
    if len(set(inputs.ordered_record_ids)) != rows:
        raise CheckpointBundleError("ordered record identifiers are not unique")
    if condition == "untrained":
        if epochs_completed != 0:
            raise CheckpointBundleError("an untrained checkpoint must complete zero epochs")
        _require_fresh_reset(inputs.training_final_state, model, "untrained training-final state")
    elif epochs_completed < 1:
        raise CheckpointBundleError("a trained or control checkpoint must complete at least one epoch")
    _validate_replay_schedule(inputs.replay_schedule, inputs.ordered_record_ids, seed, epochs_completed)


def _check_identity_agreement(identities: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    build = manifest["build"]
    encoder = manifest["encoder"]
    pairs: tuple[tuple[Any, Any, str], ...] = (
        (identities["experiment_digest"], manifest["experiment_digest"], "experiment digest"),
        (identities["dataset_digest"], manifest["dataset_digest"], "dataset digest"),
        (identities["d2_file_sha256"], manifest["dataset_digest"], "D2 dataset digest"),
        (identities["task_set_digest"], manifest["task_set_digest"], "task-set digest"),
        (identities["candidate_set_digest"], manifest["candidate_set_digest"], "candidate-set digest"),
        (int(identities["seed"]), int(manifest["seed"]), "seed"),
        (identities["condition"], manifest["condition"], "condition"),
        (identities["model_config_digest"], manifest["model_config_digest"], "model-config digest"),
        (float(identities["input_current"]), float(manifest["input_current"]), "input current"),
        (identities["encoder_directory_digest"], encoder["directory_digest"], "encoder directory digest"),
        (identities["encoder_config_digest"], encoder["config_digest"], "encoder config digest"),
        (identities["repository_head"], build["repository_head"], "repository head"),
        (identities["patch_digest"], build["patch_digest"], "patch digest"),
        (identities["python_wheel_sha256"], build["python_wheel_digest"], "python wheel digest"),
        (identities["rust_wheel_sha256"], build["rust_wheel_digest"], "rust wheel digest"),
        (identities["backend_build_digest"], build["backend_build_digest"], "backend build digest"),
    )
    for descriptor_value, manifest_value, name in pairs:
        if descriptor_value != manifest_value:
            raise CheckpointBundleError(f"descriptor {name} disagrees with the manifest")


def _manifest_payload_digest(canonical: bytes) -> str:
    return _sha(b"remanentia:snn-memory:checkpoint-manifest:v2\0" + canonical)


def write_checkpoint_bundle_v2(
    target: Path,
    *,
    state: str,
    manifest: Mapping[str, Any],
    identities: Mapping[str, Any],
    inputs: CheckpointBundleInputs,
) -> ValidatedCheckpointBundle:
    """Transactionally seal one no-clobber checkpoint bundle and re-read it.

    Parameters
    ----------
    target
        The previously absent bundle directory to install atomically.
    state
        ``fixture_only`` or ``pre_result``.
    manifest
        A ``snn_memory_checkpoint_v2`` manifest whose declared digests must equal
        the recomputed component digests.
    identities
        The descriptor identity block, including ``n_neurons``/``n_excitatory``.
    inputs
        Every array and schedule to seal.

    Returns
    -------
    ValidatedCheckpointBundle
        The bundle re-read and authenticated from the freshly written bytes.
    """
    _validate_schema(manifest, _CHECKPOINT_SCHEMA, "checkpoint manifest")
    model = _reconstruct_model(identities)
    n_excitatory = model.n_excitatory
    n_neurons = model.n_neurons
    seed = int(identities["seed"])
    _require(inputs.topology.shape == (n_neurons, n_neurons), "topology shape differs from n_neurons")
    _require(inputs.weights.shape == (n_neurons, n_neurons), "weights shape differs from n_neurons")
    _require(0 < n_excitatory < n_neurons, "excitatory count must be within the population")
    _check_identity_agreement(identities, manifest)
    _validate_semantics(
        inputs, model, _seed_topology(model, seed), str(identities["condition"]),
        seed, int(manifest["epochs_completed"]),
    )
    records = _component_records(inputs, n_excitatory)
    _check_manifest_bindings(manifest, records, inputs.ordered_record_ids)

    manifest_canonical = _canonical_bytes(manifest)
    manifest_file_sha = _sha(manifest_canonical)
    manifest_self = _manifest_payload_digest(manifest_canonical)
    descriptor: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "snn-memory-checkpoint-bundle-v2",
        "state": state,
        "lane_role": "lane_p",
        "checkpoint_manifest": {
            "path": _MANIFEST_PATH,
            "file_sha256": manifest_file_sha,
            "payload_self_sha256": manifest_self,
        },
        "identities": dict(identities),
        "components": [
            {key: records[role][key] for key in
             ("path", "role", "file_sha256", "semantic_digest", "dtype", "shape",
              "byte_length", "framing_version")}
            for role in _COMPONENT_ORDER
        ],
        "replay_schedule_digest": records["replay_schedule"]["semantic_digest"],
    }
    descriptor["self_sha256"] = _self_digest(descriptor, _DESCRIPTOR_DOMAIN)
    _validate_schema(descriptor, _BUNDLE_SCHEMA, "checkpoint-bundle descriptor")

    files: dict[str, bytes] = {
        _MANIFEST_PATH: manifest_canonical,
        _DESCRIPTOR_PATH: _canonical_bytes(descriptor),
    }
    for role in _COMPONENT_ORDER:
        files[_COMPONENT_PATHS[role]] = records[role]["bytes"]
    _install_bundle_directory(target, files)
    return read_checkpoint_bundle_v2(target)


def read_checkpoint_bundle_v2(target: Path) -> ValidatedCheckpointBundle:
    """Authenticate one checkpoint bundle through a single symlink-refusing read.

    Every entry is opened once with ``O_NOFOLLOW`` relative to the bundle directory,
    validated against the descriptor and manifest schemas, and re-derived from the
    captured bytes. Missing, extra, duplicate, non-regular, traversing, or symlinked
    entries fail closed.
    """
    bundle_fd = _open_dir_from_root(target, "checkpoint bundle directory")
    try:
        entries = set(os.listdir(bundle_fd))
        expected = {_MANIFEST_PATH, _DESCRIPTOR_PATH, *(_COMPONENT_PATHS[role] for role in _COMPONENT_ORDER)}
        if entries != expected:
            raise CheckpointBundleError("checkpoint bundle directory inventory does not match the descriptor")
        descriptor_bytes = _read_regular_bytes(bundle_fd, _DESCRIPTOR_PATH, "checkpoint-bundle descriptor")
        descriptor = _strict_json(descriptor_bytes, "checkpoint-bundle descriptor")
        _validate_schema(descriptor, _BUNDLE_SCHEMA, "checkpoint-bundle descriptor")
        stored_self = descriptor["self_sha256"]
        if stored_self != _self_digest(descriptor, _DESCRIPTOR_DOMAIN):
            raise CheckpointBundleError("checkpoint-bundle descriptor self digest mismatch")

        manifest_bytes = _read_regular_bytes(bundle_fd, _MANIFEST_PATH, "checkpoint manifest")
        manifest = _strict_json(manifest_bytes, "checkpoint manifest")
        _validate_schema(manifest, _CHECKPOINT_SCHEMA, "checkpoint manifest")
        binding = descriptor["checkpoint_manifest"]
        if _sha(manifest_bytes) != binding["file_sha256"]:
            raise CheckpointBundleError("checkpoint manifest file digest mismatch")
        if _manifest_payload_digest(_canonical_bytes(manifest)) != binding["payload_self_sha256"]:
            raise CheckpointBundleError("checkpoint manifest payload digest mismatch")
        if manifest_bytes != _canonical_bytes(manifest):
            raise CheckpointBundleError("checkpoint manifest is not canonical")

        declared = descriptor["components"]
        if [entry["role"] for entry in declared] != list(_COMPONENT_ORDER):
            raise CheckpointBundleError("checkpoint-bundle component inventory is out of order")
        identities = descriptor["identities"]
        _check_identity_agreement(identities, manifest)
        model = _reconstruct_model(identities)
        n_neurons = model.n_neurons
        n_excitatory = model.n_excitatory
        semantics: dict[str, dict[str, Any]] = {}
        decoded: dict[str, Any] = {}
        for entry in declared:
            role = entry["role"]
            if entry["path"] != _COMPONENT_PATHS[role]:
                raise CheckpointBundleError(f"checkpoint component {role} path mismatch")
            body = _read_regular_bytes(bundle_fd, entry["path"], f"checkpoint component {role}")
            if _sha(body) != entry["file_sha256"] or len(body) != entry["byte_length"]:
                raise CheckpointBundleError(f"checkpoint component {role} file digest or length mismatch")
            semantic, value = _decode_component(role, entry, body, n_neurons, n_excitatory)
            if semantic != entry["semantic_digest"]:
                raise CheckpointBundleError(f"checkpoint component {role} semantic digest mismatch")
            semantics[role] = {"semantic_digest": semantic}
            decoded[role] = value

        _check_manifest_bindings(manifest, semantics, tuple(manifest["ordered_record_ids"]))
        if descriptor["replay_schedule_digest"] != semantics["replay_schedule"]["semantic_digest"]:
            raise CheckpointBundleError("descriptor replay-schedule digest mismatch")
        inputs = CheckpointBundleInputs(
            topology=decoded["topology"],
            outgoing=decoded["adjacency_outgoing"],
            incoming=decoded["adjacency_incoming"],
            weights=decoded["weights"],
            training_final_state=decoded["training_final_state"],
            probe_initial_state=decoded["probe_initial_state"],
            signatures=decoded["signatures"],
            ordered_record_ids=decoded["record_ids"],
            replay_schedule=decoded["replay_schedule"],
        )
        seed = int(identities["seed"])
        _validate_semantics(
            inputs, model, _seed_topology(model, seed), str(identities["condition"]),
            seed, int(manifest["epochs_completed"]),
        )
    finally:
        os.close(bundle_fd)
    return ValidatedCheckpointBundle(
        descriptor=cast(Mapping[str, Any], _freeze(descriptor)),
        manifest=cast(Mapping[str, Any], _freeze(manifest)),
        descriptor_self_sha256=stored_self,
        manifest_file_sha256=binding["file_sha256"],
        manifest_payload_self_sha256=binding["payload_self_sha256"],
        inputs=inputs,
    )


_ROLE_DTYPE = {
    "topology": "|b1", "adjacency_outgoing": "<u8", "adjacency_incoming": "<u8",
    "weights": "<f8", "training_final_state": "state", "probe_initial_state": "state",
    "signatures": "<f8", "record_ids": "json", "replay_schedule": "json",
}


def _decode_component(role: str, entry: Mapping[str, Any], body: bytes,
                      n_neurons: int, n_excitatory: int) -> tuple[str, Any]:
    """Re-derive a component's semantic digest and its decoded, independently owned value."""
    if entry["dtype"] != _ROLE_DTYPE[role]:
        raise CheckpointBundleError(f"{role} component declares an unexpected dtype")
    if int(entry["framing_version"]) != _COMPONENT_FRAMING:
        raise CheckpointBundleError(f"{role} component declares an unexpected framing version")
    shape = tuple(int(dim) for dim in entry["shape"])
    if role == "topology":
        if shape != (n_neurons, n_neurons) or len(body) != n_neurons * n_neurons:
            raise CheckpointBundleError("topology component shape mismatch")
        _require_boolean_bytes(body, "topology")
        topology = np.frombuffer(body, dtype="|b1").reshape(n_neurons, n_neurons)
        return topology_digest(topology, n_excitatory), _read_only(topology)
    if role in ("adjacency_outgoing", "adjacency_incoming"):
        if len(body) % 8 != 0 or shape != (len(body) // 8,):
            raise CheckpointBundleError(f"{role} component shape mismatch")
        values = np.frombuffer(body, dtype="<u8")
        if values.size < n_neurons + 1:
            raise CheckpointBundleError(f"{role} component is shorter than the offset row")
        offsets = values[: n_neurons + 1]
        indices = values[n_neurons + 1 :]
        orientation = "outgoing" if role == "adjacency_outgoing" else "incoming"
        csr = CsrArrays(offsets=_read_only(offsets), indices=_read_only(indices))
        return _csr_digest(orientation, csr), csr
    if role == "weights":
        if shape != (n_neurons, n_neurons) or len(body) != n_neurons * n_neurons * 8:
            raise CheckpointBundleError("weights component shape mismatch")
        weights = np.frombuffer(body, dtype="<f8").reshape(n_neurons, n_neurons)
        return checkpoint_component_digest("weights", "<f8", weights), _read_only(weights)
    if role in ("training_final_state", "probe_initial_state"):
        if shape != (n_neurons,):
            raise CheckpointBundleError(f"{role} component shape mismatch")
        state = _decode_state(body, n_neurons)
        return _state_digest(state), state
    if role == "signatures":
        if len(shape) != 2 or shape[1] % 8 != 0 or len(body) != shape[0] * shape[1] * 8:
            raise CheckpointBundleError("signatures component shape mismatch")
        signatures = np.frombuffer(body, dtype="<f8").reshape(shape)
        return checkpoint_component_digest("signatures", "<f8", signatures), _read_only(signatures)
    if role == "record_ids":
        record_ids = tuple(_strict_str(value, "record id") for value in _strict_json_array(body, "record_ids"))
        return ordered_record_ids_digest(record_ids), record_ids
    schedule = tuple(_strict_replay_entry(item) for item in _strict_json_array(body, "replay_schedule"))
    return replay_schedule_digest(schedule), schedule


def _decode_state(body: bytes, n_neurons: int) -> StateArrays:
    if len(body) != n_neurons * _STATE_BYTES_PER_NEURON:
        raise CheckpointBundleError("state component byte length mismatch")
    offset = 0
    voltage = np.frombuffer(body, dtype="<f8", count=n_neurons, offset=offset)
    offset += n_neurons * 8
    refractory = np.frombuffer(body, dtype="<u4", count=n_neurons, offset=offset)
    offset += n_neurons * 4
    _require_boolean_bytes(body[offset:offset + n_neurons], "state spikes")
    spikes = np.frombuffer(body, dtype="|b1", count=n_neurons, offset=offset)
    offset += n_neurons
    pre_trace = np.frombuffer(body, dtype="<f8", count=n_neurons, offset=offset)
    offset += n_neurons * 8
    post_trace = np.frombuffer(body, dtype="<f8", count=n_neurons, offset=offset)
    return StateArrays(
        _read_only(voltage), _read_only(refractory), _read_only(spikes),
        _read_only(pre_trace), _read_only(post_trace),
    )


def _strict_json_array(body: bytes, label: str) -> list[Any]:
    def reject_constant(value: str) -> None:
        raise CheckpointBundleError(f"{label} contains a non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, item in pairs:
            if key in seen:
                raise CheckpointBundleError(f"{label} contains a duplicate object key {key!r}")
            seen[key] = item
        return seen

    try:
        value = json.loads(
            body.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointBundleError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, list):
        raise CheckpointBundleError(f"{label} must be a JSON array")
    if body != _canonical_bytes(value):
        raise CheckpointBundleError(f"{label} is not canonical JSON")
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointBundleError(f"{label} must be a JSON integer")
    return int(value)


def _strict_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CheckpointBundleError(f"{label} must be a JSON string")
    return value


def _strict_replay_entry(item: Any) -> Mapping[str, Any]:
    if not isinstance(item, dict):
        raise CheckpointBundleError("replay entry must be a JSON object")
    if set(item) != {"epoch", "record_id", "replay_position", "timesteps"}:
        raise CheckpointBundleError("replay entry does not carry the exact schema keys")
    return MappingProxyType({
        "epoch": _strict_int(item["epoch"], "replay epoch"),
        "record_id": _strict_str(item["record_id"], "replay record id"),
        "replay_position": _strict_int(item["replay_position"], "replay position"),
        "timesteps": _strict_int(item["timesteps"], "replay timesteps"),
    })


def _require_boolean_bytes(raw: bytes, label: str) -> None:
    values = np.frombuffer(raw, dtype=np.uint8)
    if values.size and bool(np.any(values > 1)):
        raise CheckpointBundleError(f"{label} boolean bytes are not exactly zero or one")
