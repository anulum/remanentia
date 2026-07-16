# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN schema-v2 streamed-Rust checkpoint/bank materializer

"""Streamed-Rust primary checkpoint and candidate-signature-bank materialization.

D4-A orchestration only: train one authenticated primary checkpoint through the
installed compiled Rust backend, derive matched ``shuffled``/``random``/``zero``
controls and an independent ``untrained`` arm, calibrate one completion-only
neuron-major eight-bin signature per ordered candidate in a fresh child process
with plasticity disabled, and seal an authenticated checkpoint bundle plus one
candidate bank per ``(seed, condition)``. No ``reference.py``, ``trainer.py``, or
v1 checkpoint writer is imported; no D2 evaluation cue or expected answer is read.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from snn_memory import candidate_bank_v2 as bank
from snn_memory import checkpoint_bundle_v2 as bundle
from snn_memory.checkpoint_bundle_v2 import (
    CheckpointBundleInputs,
    CsrArrays,
    StateArrays,
    read_checkpoint_bundle_v2,
    write_checkpoint_bundle_v2,
)
from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.controls import make_control
from snn_memory.cue_materializer import _block_text, validate_cue_set_bytes
from snn_memory.encoder import embeddings_to_currents, split_events
from snn_memory.experiment_lock import (
    candidate_bank_digest,
    canonical_config_digest,
    scorer_identity_digest,
    validate_artifact_bytes,
    write_artifact,
)
from snn_memory.source_universe import _event_order, validate_source_universe_bytes
from snn_memory.stream_backend import (
    BackendIdentity,
    StreamBackend,
    StreamInputs,
    StreamResult,
    load_stream_backend,
    state_digest,
    topology_digest,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_ALLOWED_CONDITIONS = ("trained", "shuffled", "random", "zero", "untrained")
_CONTROL_CONDITIONS = ("shuffled", "random", "zero")
_FORBIDDEN_CONDITIONS = (
    "encoder-only", "temporal-order-permuted", "leave-one-record-out", "G1-BG", "G1-STATE",
)


class MaterializeError(ValueError):
    """Raised when a D4-A materialization violates a contract or fails closed."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _packaged_schema_sha256(name: str) -> str:
    from importlib.resources import files

    return _sha(files("snn_memory").joinpath("schema", name).read_bytes())


def _validate_live_identities(config: Mapping[str, Any]) -> None:
    """Prove the cheap live module and packaged-schema identities before any expensive work."""
    if _sha(Path(__file__).read_bytes()) != str(config["materializer_module_sha256"]):
        raise MaterializeError("live materializer module hash differs from the configured digest")
    for key, schema_name in (
        ("checkpoint_schema_sha256", "snn_memory_checkpoint_v2.schema.json"),
        ("bank_schema_sha256", "snn_memory_candidate_bank_v2.schema.json"),
    ):
        if _packaged_schema_sha256(schema_name) != str(config[key]):
            raise MaterializeError(f"live {schema_name} hash differs from the configured digest")


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise MaterializeError(f"{label} carries a non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, item in pairs:
            if key in seen:
                raise MaterializeError(f"{label} contains a duplicate key {key!r}")
            seen[key] = item
        return seen

    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializeError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise MaterializeError(f"{label} must be a JSON object")
    if raw != _canonical_json_bytes(value):
        raise MaterializeError(f"{label} is not canonical JSON")
    return cast(dict[str, Any], value)


def _model_from_config(model_config: Mapping[str, Any]) -> ModelConfig:
    try:
        return ModelConfig(**dict(model_config))
    except (TypeError, ValueError) as error:
        raise MaterializeError(f"invalid model configuration: {error}") from error


def _fresh_state(model: ModelConfig) -> StateArrays:
    n = model.n_neurons
    return StateArrays(
        voltage_mv=np.full(n, model.v_rest_mv, dtype=np.float64),
        refractory_steps=np.zeros(n, dtype=np.uint32),
        spikes=np.zeros(n, dtype=np.bool_),
        pre_trace=np.zeros(n, dtype=np.float64),
        post_trace=np.zeros(n, dtype=np.float64),
    )


def _stream_inputs(model: ModelConfig, weights: FloatArray, topology: BoolArray,
                   packets: FloatArray, state: StateArrays) -> StreamInputs:
    return StreamInputs(
        voltage_mv=np.ascontiguousarray(state.voltage_mv, dtype=np.float64),
        refractory_steps=np.ascontiguousarray(state.refractory_steps, dtype=np.uint32),
        spikes=np.ascontiguousarray(state.spikes, dtype=np.bool_),
        pre_trace=np.ascontiguousarray(state.pre_trace, dtype=np.float64),
        post_trace=np.ascontiguousarray(state.post_trace, dtype=np.float64),
        weights=np.ascontiguousarray(weights, dtype=np.float64),
        topology=np.ascontiguousarray(topology, dtype=np.bool_),
        packets=np.ascontiguousarray(packets, dtype=np.float64),
    )


def _state_signature(state: StateArrays) -> str:
    return state_digest(
        np.ascontiguousarray(state.voltage_mv, dtype=np.float64),
        np.ascontiguousarray(state.refractory_steps, dtype=np.uint32),
        np.ascontiguousarray(state.spikes, dtype=np.bool_),
        np.ascontiguousarray(state.pre_trace, dtype=np.float64),
        np.ascontiguousarray(state.post_trace, dtype=np.float64),
    )


def _result_state(result: StreamResult) -> StateArrays:
    return StateArrays(
        voltage_mv=np.ascontiguousarray(result.final_voltage_mv, dtype=np.float64),
        refractory_steps=np.ascontiguousarray(result.final_refractory_steps, dtype=np.uint32),
        spikes=np.ascontiguousarray(result.final_spikes, dtype=np.bool_),
        pre_trace=np.ascontiguousarray(result.final_pre_trace, dtype=np.float64),
        post_trace=np.ascontiguousarray(result.final_post_trace, dtype=np.float64),
    )


def _csr_from_topology(topology: BoolArray, *, outgoing: bool) -> CsrArrays:
    n = topology.shape[0]
    offsets = [0]
    indices: list[int] = []
    for node in range(n):
        row = np.flatnonzero(topology[node] if outgoing else topology[:, node])
        indices.extend(int(value) for value in row)
        offsets.append(len(indices))
    return CsrArrays(
        offsets=np.asarray(offsets, dtype=np.uint64),
        indices=np.asarray(indices, dtype=np.uint64),
    )


def _initial_weights(model: ModelConfig, seed: int) -> tuple[FloatArray, BoolArray]:
    from snn_memory.state import initialise_weights

    return initialise_weights(model, seed)


def _encode_events(encoder_module: Any, model: ModelConfig, encoder_config: EncoderConfig,
                   text: str, input_current: float) -> FloatArray:
    events = split_events(text)
    if not events:
        raise MaterializeError("training or calibration text yields no events")
    embeddings = encoder_module.encode(events)
    current = embeddings_to_currents(
        np.asarray(embeddings, dtype=np.float64), model, encoder_config, input_current=input_current
    )
    if not bool(np.isfinite(current).all()):
        raise MaterializeError("encoded training or calibration current carries a non-finite value")
    return current


def _completion_raster(result: StreamResult, cue_steps: int, completion_steps: int,
                       n_neurons: int) -> BoolArray:
    raster = np.zeros((completion_steps, n_neurons), dtype=np.bool_)
    for step in range(completion_steps):
        timestep = cue_steps + step
        start = int(result.spike_offsets[timestep])
        stop = int(result.spike_offsets[timestep + 1])
        raster[step, result.spike_indices[start:stop].astype(np.int64)] = True
    return raster


def _thread_capped_env() -> dict[str, str]:
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


def _open_dir_from_root(path: Path, label: str) -> int:
    """Open a directory by walking every component from the filesystem root, refusing symlinks.

    Each component is opened with ``O_DIRECTORY|O_NOFOLLOW`` relative to the previous directory
    descriptor, so no parent component can be a symlink or be raced. The caller owns and closes
    the returned directory descriptor.
    """
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise MaterializeError(f"{label} filesystem root cannot be opened safely") from error
    try:
        for part in absolute.parts[1:]:
            if part in (os.curdir, os.pardir):
                raise MaterializeError(f"{label} path carries a non-normal '{part}' component")
            try:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            except OSError as error:
                raise MaterializeError(f"{label} parent {part!r} cannot be opened safely") from error
            os.close(current)
            current = nxt
    except BaseException:
        os.close(current)
        raise
    return current


def _read_at(dir_fd: int, name: str, label: str) -> bytes:
    """Read a regular file by name relative to a captured directory descriptor, refusing symlinks."""
    import stat

    try:
        leaf = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    except OSError as error:
        raise MaterializeError(f"{label} cannot be opened safely") from error
    try:
        if not stat.S_ISREG(os.fstat(leaf).st_mode):
            raise MaterializeError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(leaf, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(leaf)
    return b"".join(chunks)


def _read_relative_at(dir_fd: int, relative: str, label: str) -> bytes:
    """Read a directory-relative file through an openat walk from a captured directory descriptor."""
    parts = Path(relative).parts
    if not parts or relative.startswith("/") or ".." in parts:
        raise MaterializeError(f"{label} path {relative!r} is not a safe relative path")
    intermediate: list[int] = []
    current = dir_fd
    try:
        for part in parts[:-1]:
            try:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            except OSError as error:
                raise MaterializeError(f"{label} path {relative!r} traverses unsafely") from error
            intermediate.append(nxt)
            current = nxt
        return _read_at(current, parts[-1], label)
    finally:
        for descriptor in reversed(intermediate):
            os.close(descriptor)


def _read_regular_file(path: Path, label: str) -> bytes:
    """Read one absolute regular file through a root-anchored symlink-refusing descriptor walk."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    dir_fd = _open_dir_from_root(absolute.parent, label)
    try:
        return _read_at(dir_fd, absolute.name, label)
    finally:
        os.close(dir_fd)


def _git(repo: Path, arguments: list[str]) -> bytes:
    # gc.auto=0 keeps git deterministic and forbids a detached background gc process that would
    # otherwise inflate the supervised process-group concurrency count.
    process = subprocess.run(
        ["git", "-C", str(repo), "-c", "gc.auto=0", *arguments],
        capture_output=True, check=False, env=_thread_capped_env(),
    )
    if process.returncode != 0:
        raise MaterializeError(f"git {' '.join(arguments)} failed: {process.stderr.decode('utf-8')}")
    return process.stdout


def _git_blob(repo: Path, head: str, relative: str, blob_oid: str) -> bytes:
    """Capture the exact Git blob bytes at the authenticated HEAD/path/OID identity."""
    resolved = _git(repo, ["rev-parse", f"{head}:{relative}"]).decode("ascii").strip()
    if resolved != blob_oid:
        raise MaterializeError(f"D1 record path {relative!r} resolves to a different blob at HEAD")
    return _git(repo, ["cat-file", "blob", blob_oid])


def _backend_identity(config: Mapping[str, Any]) -> BackendIdentity:
    return BackendIdentity(
        api_version=2,
        crate_version=str(config["crate_version"]),
        extension_sha256=str(config["extension_sha256"]),
    )


def _le_bytes(array: npt.NDArray[Any], dtype: str) -> bytes:
    return np.ascontiguousarray(array, dtype=np.dtype(dtype)).tobytes(order="C")


def _reject_evaluation_roles(config: Mapping[str, Any]) -> None:
    forbidden = ("evaluation_cues", "evaluation_base_cues", "expected_answers",
                 "expected_record_id", "expected_record_ids", "answer_key")
    for key in forbidden:
        if key in config:
            raise MaterializeError(f"D4-A rejects the evaluation/expected-answer field {key!r}")


_CONFIG_KEYS = frozenset({
    "abstention_threshold", "backend_build_digest", "backend_version", "bank_schema_sha256",
    "calibration_spec_digest", "checkpoint_schema_sha256", "completion_steps", "condition",
    "crate_version", "cue_set", "d1", "d1_file_sha256", "d2_file_sha256", "development_artifact_digest",
    "dirty_tree_digest", "encoder_checkpoint", "encoder_config", "encoder_config_digest",
    "encoder_digest", "encoder_locator", "epochs", "experiment_digest", "experiment_lock_module_sha256",
    "experiment_lock_schema_sha256", "extension_sha256", "gb_preflight_module_sha256", "input_current",
    "materializer_module_sha256", "metric", "model_config_digest", "output", "patch_digest",
    "python_wheel_sha256", "repo", "repository_head", "rust_wheel_sha256", "seed", "task_set_digest",
    "trained_bundle",
})


def _strict_config(path: Path) -> dict[str, Any]:
    """Parse the materialize config with duplicate-key and non-finite rejection, fail-closed."""
    raw = _read_regular_file(path, "materialize config")

    def reject_constant(value: str) -> None:
        raise MaterializeError(f"config contains a non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, item in pairs:
            if key in seen:
                raise MaterializeError(f"config contains a duplicate key {key!r}")
            seen[key] = item
        return seen

    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant,
                           object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializeError("config is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise MaterializeError("config root must be a JSON object")
    if raw != _canonical_json_bytes(value):
        raise MaterializeError("config is not canonical JSON")
    return cast(dict[str, Any], value)


def _reject_unknown_config(config: Mapping[str, Any]) -> None:
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise MaterializeError(f"config carries unknown field(s): {sorted(unknown)}")


@dataclass(frozen=True)
class PreparedInputs:
    """Authenticated frozen-order D1 training and D2 calibration inputs for one materialization."""

    ids: tuple[str, ...]
    training_currents: tuple[FloatArray, ...]
    calibration_currents: tuple[FloatArray, ...]
    model: ModelConfig
    encoder_config: EncoderConfig
    input_current: float
    training_source_digests: tuple[dict[str, str], ...]
    calibration_cue_digests: tuple[dict[str, str], ...]
    calibration_evidence: tuple[dict[str, Any], ...]
    d1_file_sha256: str
    d1_payload_self_sha256: str
    repository_head: str


def _prepare_inputs(config: Mapping[str, Any], encoder_module: Any,
                    need_training_currents: bool) -> PreparedInputs:
    """Authenticate D1 once, cross-bind D2, and build disjoint training/calibration currents.

    Plastic-training currents come from the complete authenticated D1 Git-record events in the
    frozen ``selected`` order; calibration currents come from the disjoint D2 calibration block,
    re-derived from the same captured D1 events and cross-checked against the single referenced
    calibration-cue file. The D2 manifest is validated with ``validate_cue_set_bytes`` (never
    ``read_cue_set``); no evaluation cue or bundle is dereferenced.
    """
    repo = Path(config["repo"])
    d1_raw = _read_regular_file(Path(config["d1"]), "D1 source universe")
    d1 = validate_source_universe_bytes(d1_raw, expected_file_sha256=str(config["d1_file_sha256"]))
    d1_file_sha = _sha(d1_raw)
    d1_self = str(d1.payload["self_sha256"])
    head = str(d1.payload["repository"]["head"])
    if head != str(config["repository_head"]):
        raise MaterializeError("D1 repository HEAD differs from the declared repository head")

    cue_path = Path(config["cue_set"])
    cue_absolute = cue_path if cue_path.is_absolute() else Path.cwd() / cue_path
    d2_dir_fd = _open_dir_from_root(cue_absolute.parent, "D2 cue set")
    try:
        return _prepare_from_captured(config, encoder_module, need_training_currents, d1, d1_raw,
                                      d1_file_sha, d1_self, head, repo, d2_dir_fd, cue_absolute.name)
    finally:
        os.close(d2_dir_fd)


def _prepare_from_captured(config: Mapping[str, Any], encoder_module: Any, need_training_currents: bool,
                          d1: Any, d1_raw: bytes, d1_file_sha: str, d1_self: str, head: str,
                          repo: Path, d2_dir_fd: int, cue_name: str) -> PreparedInputs:
    from snn_memory.v2_contracts import normalize_cue_text

    cue_raw = _read_at(d2_dir_fd, cue_name, "D2 cue set")
    cue = validate_cue_set_bytes(cue_raw, expected_file_sha256=str(config["d2_file_sha256"]))
    source = cue.payload["source_universe"]
    if (str(source["file_sha256"]) != d1_file_sha
            or str(source["payload_self_sha256"]) != d1_self
            or str(source["repository_head"]) != head
            or list(source["selected_record_ids"]) != list(d1.payload["selected_record_ids"])):
        raise MaterializeError("cue set does not cross-bind the exact D1 artifact and HEAD")
    model = _model_from_config(cue.payload["model"]["config"])
    encoder_config = EncoderConfig(**dict(cue.payload["encoder"]["config"]))
    input_current = float(cue.payload["model"]["input_current"])
    if canonical_config_digest(model.to_dict()) != str(config["model_config_digest"]):
        raise MaterializeError("declared model-config digest differs from the D2 model configuration")
    if canonical_config_digest(dict(cue.payload["encoder"]["config"])) != str(config["encoder_config_digest"]):
        raise MaterializeError("declared encoder-config digest differs from the D2 encoder configuration")
    if canonical_config_digest(dict(config["encoder_config"])) != str(config["encoder_config_digest"]):
        raise MaterializeError("declared encoder config differs from its declared digest")
    if float(config["input_current"]) != input_current:
        raise MaterializeError("declared input current differs from the D2 model input current")
    live_head = _git(repo, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if live_head != head:
        raise MaterializeError("live repository HEAD differs from the authenticated D1 HEAD")
    if str(cue.payload["encoder"]["directory_sha256"]) != str(config["encoder_digest"]):
        raise MaterializeError("D2 encoder directory digest differs from the pinned encoder digest")
    if int(cue.payload["model"]["n_neurons"]) != model.n_neurons:
        raise MaterializeError("D2 model neuron count differs from its configuration")

    cue_by_id = {str(record["record_id"]): record["calibration_cue"] for record in cue.payload["records"]}
    encoder_identity = str(cue.payload["encoder"]["identity"])
    encoder_directory_digest = str(cue.payload["encoder"]["directory_sha256"])
    # Lock-family convention (canonical_config_digest), proven equal to the D2-carried
    # configs above; the D2 payload's own canonical_json_digest fields stay validated
    # inside the cue reader and must not leak into D4-A identity fields.
    encoder_config_digest = str(config["encoder_config_digest"])
    model_config_digest = str(config["model_config_digest"])
    ids: list[str] = []
    training_currents: list[FloatArray] = []
    calibration_currents: list[FloatArray] = []
    training_digests: list[dict[str, str]] = []
    calibration_digests: list[dict[str, str]] = []
    calibration_evidence: list[dict[str, Any]] = []
    for record in d1.payload["selected"]:
        rid = str(record["record_id"])
        raw = _git_blob(repo, head, str(record["path"]), str(record["blob_oid"]))
        if _sha(raw) != str(record["content_sha256"]) or len(raw) != int(record["byte_count"]):
            raise MaterializeError(f"D1 blob for {rid} differs from its content binding")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MaterializeError(f"D1 record {rid} is not strict UTF-8") from error
        events = split_events(text)
        if len(events) != int(record["event_count"]):
            raise MaterializeError(f"D1 record {rid} event count differs from its binding")
        order_digest, hashes = _event_order(events)
        if order_digest != str(record["event_order_digest"]) or hashes != list(record["event_sha256"]):
            raise MaterializeError(f"D1 record {rid} event hashes differ from its binding")
        if need_training_currents:
            training_currents.append(
                _encode_events(encoder_module, model, encoder_config, text, input_current))
        training_digests.append({"record_id": rid, "digest": str(record["content_sha256"])})

        calibration = cue_by_id[rid]
        indices = [int(index) for index in calibration["event_indices"]]
        block_text = _block_text([events[index] for index in indices])
        if _sha(block_text.encode("utf-8")) != str(calibration["sha256"]):
            raise MaterializeError(f"re-derived calibration block for {rid} differs from the cue digest")
        cue_bytes = _read_relative_at(d2_dir_fd, str(calibration["path"]), f"calibration cue for {rid}")
        if _sha(cue_bytes) != str(calibration["sha256"]) or cue_bytes.decode("utf-8") != block_text:
            raise MaterializeError(f"calibration cue file for {rid} differs from the re-derived block")
        if len(block_text.split("\n")) != len(indices):
            raise MaterializeError(f"calibration cue line count for {rid} differs from the event indices")
        if _sha(normalize_cue_text(block_text).encode("utf-8")) != str(calibration["normalized_text_sha256"]):
            raise MaterializeError(f"calibration cue normalized digest for {rid} differs from its binding")
        current = np.ascontiguousarray(
            _encode_events(encoder_module, model, encoder_config, block_text, input_current), dtype="<f8")
        calibration_currents.append(current)
        calibration_digests.append({"record_id": rid, "digest": str(calibration["sha256"])})
        calibration_evidence.append({
            "record_id": rid,
            "source_cue_sha256": str(calibration["sha256"]),
            "current_semantic_digest": bundle.checkpoint_component_digest("calibration_current", "<f8", current),
            "dtype": "<f8",
            "shape": [int(current.shape[0]), int(current.shape[1])],
            "input_current": input_current,
            "encoder_identity": encoder_identity,
            "encoder_directory_digest": encoder_directory_digest,
            "encoder_config_digest": encoder_config_digest,
            "model_config_digest": model_config_digest,
        })
        ids.append(rid)

    return PreparedInputs(
        ids=tuple(ids), training_currents=tuple(training_currents),
        calibration_currents=tuple(calibration_currents), model=model, encoder_config=encoder_config,
        input_current=input_current, training_source_digests=tuple(training_digests),
        calibration_cue_digests=tuple(calibration_digests),
        calibration_evidence=tuple(calibration_evidence),
        d1_file_sha256=d1_file_sha, d1_payload_self_sha256=d1_self, repository_head=head,
    )


def _train_checkpoint(backend: StreamBackend, model: ModelConfig, seed: int,
                      currents: Sequence[FloatArray], epochs: int, ids: Sequence[str]
                      ) -> tuple[FloatArray, BoolArray, StateArrays, list[dict[str, Any]]]:
    from snn_memory.state import validate_weights

    weights, topology = _initial_weights(model, seed)
    schedule: list[dict[str, Any]] = []
    final_state = _fresh_state(model)
    for epoch in range(epochs):
        permutation = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(currents))
        for position, index in enumerate(permutation):
            record = int(index)
            timesteps = int(currents[record].shape[0])
            result = backend.run(
                _stream_inputs(model, weights, topology, currents[record], _fresh_state(model)),
                timesteps, True, model,
            )
            weights = np.ascontiguousarray(result.final_weights, dtype=np.float64)
            validate_weights(weights, topology, model)
            final_state = _result_state(result)
            schedule.append({"epoch": epoch, "record_id": str(ids[record]),
                             "replay_position": int(position), "timesteps": timesteps})
    return weights, topology, final_state, schedule


def _control_identity_expectations(config: Mapping[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    return {
        "experiment_digest": str(config["experiment_digest"]),
        "dataset_digest": str(config["d2_file_sha256"]),
        "d2_file_sha256": str(config["d2_file_sha256"]),
        "d1_file_sha256": str(config["d1_file_sha256"]),
        "task_set_digest": str(config["task_set_digest"]),
        "candidate_set_digest": bundle.ordered_record_ids_digest(ids),
        "model_config_digest": str(config["model_config_digest"]),
        "input_current": float(config["input_current"]),
        "encoder_directory_digest": str(config["encoder_digest"]),
        "encoder_config_digest": canonical_config_digest(config["encoder_config"]),
        "repository_head": str(config["repository_head"]),
        "patch_digest": str(config["patch_digest"]),
        "python_wheel_sha256": str(config["python_wheel_sha256"]),
        "rust_wheel_sha256": str(config["rust_wheel_sha256"]),
        "backend_build_digest": str(config["backend_build_digest"]),
        "materializer_module_sha256": str(config["materializer_module_sha256"]),
    }


def _control_inputs(trained_dir: Path, model: ModelConfig, seed: int, condition: str,
                    config: Mapping[str, Any], ids: Sequence[str]
                    ) -> tuple[FloatArray, BoolArray, StateArrays, list[dict[str, Any]], int]:
    from snn_memory.state import validate_weights

    trained = read_checkpoint_bundle_v2(trained_dir)
    if trained.manifest["condition"] != "trained":
        raise MaterializeError("a control must derive from a trained checkpoint")
    if int(trained.manifest["seed"]) != seed:
        raise MaterializeError("control seed differs from the trained checkpoint seed")
    identities = trained.descriptor["identities"]
    for key, value in _control_identity_expectations(config, ids).items():
        if identities.get(key) != value:
            raise MaterializeError(f"trained bundle {key} differs from the control configuration")
    if tuple(trained.manifest["ordered_record_ids"]) != tuple(ids):
        raise MaterializeError("trained bundle record order differs from the control candidate order")
    weights = np.ascontiguousarray(trained.inputs.weights, dtype=np.float64)
    topology = np.ascontiguousarray(trained.inputs.topology, dtype=np.bool_)
    control = np.ascontiguousarray(
        make_control(weights, topology, model, cast(Any, condition), seed), dtype=np.float64
    )
    validate_weights(control, topology, model)
    schedule = [dict(entry) for entry in trained.inputs.replay_schedule]
    return (control, topology, trained.inputs.training_final_state, schedule,
            int(trained.manifest["epochs_completed"]))


def _calibrate_candidates(config: Mapping[str, Any], worker_dir: Path, model: ModelConfig,
                          weights: FloatArray, topology: BoolArray, cue_currents: Sequence[FloatArray],
                          ids: Sequence[str], completion_steps: int
                          ) -> tuple[FloatArray, list[int], list[dict[str, Any]]]:
    worker_dir.mkdir(parents=True, exist_ok=True)
    weights_bytes = _le_bytes(weights, "<f8")
    topology_bytes = _le_bytes(topology, "|b1")
    model_bytes = _canonical_json_bytes(model.to_dict())
    (worker_dir / "weights.bin").write_bytes(weights_bytes)
    (worker_dir / "topology.bin").write_bytes(topology_bytes)
    (worker_dir / "model_config.json").write_bytes(model_bytes)
    weights_digest, topology_digest_hex, model_digest = _sha(weights_bytes), _sha(topology_bytes), _sha(model_bytes)
    n = model.n_neurons
    signatures = np.empty((len(ids), 8 * n), dtype="<f8")
    fresh_pre_digest = _state_signature(_fresh_state(model))
    pids: set[int] = set()
    ordered_pids: list[int] = []
    nonces: set[str] = set()
    evidence: list[dict[str, Any]] = []
    environment = _thread_capped_env()
    for index in range(len(ids)):
        cue_bytes = _le_bytes(cue_currents[index], "<f8")
        cue_path = worker_dir / f"cue_{index:03d}.bin"
        cue_path.write_bytes(cue_bytes)
        cue_digest = _sha(cue_bytes)
        nonce = os.urandom(16).hex()
        argv = [
            sys.executable, "-m", "snn_memory.checkpoint_materialize_v2", "_calibrate",
            "--weights", str(worker_dir / "weights.bin"), "--weights-digest", weights_digest,
            "--topology", str(worker_dir / "topology.bin"), "--topology-digest", topology_digest_hex,
            "--cue", str(cue_path), "--cue-digest", cue_digest,
            "--model-config", str(worker_dir / "model_config.json"), "--model-config-digest", model_digest,
            "--record-id", str(ids[index]), "--nonce", nonce,
            "--completion-steps", str(completion_steps),
            "--extension-sha256", str(config["extension_sha256"]),
            "--crate-version", str(config["crate_version"]),
            "--backend-build-digest", str(config["backend_build_digest"]),
        ]
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise MaterializeError(f"calibration worker failed: {stderr.decode('utf-8')}")
        if stderr != b"":
            raise MaterializeError(f"calibration worker emitted stray stderr: {stderr.decode('utf-8')[:500]!r}")
        report = _parse_worker_report(stdout)
        try:
            raster_raw = base64.b64decode(report["raster_b64"], validate=True)
        except (ValueError, TypeError) as error:
            raise MaterializeError("calibration worker raster is not valid base64") from error
        if len(raster_raw) != completion_steps * n:
            raise MaterializeError("calibration worker raster byte length differs from the completion window")
        if bool(np.any(np.frombuffer(raster_raw, dtype=np.uint8) > 1)):
            raise MaterializeError("calibration worker raster carries a byte outside {0, 1}")
        raster = np.frombuffer(raster_raw, dtype="|b1").reshape(completion_steps, n)
        expected = {
            "pid": process.pid, "record_id": str(ids[index]), "cue_digest": cue_digest, "nonce": nonce,
            "weights_digest": weights_digest, "final_weights_digest": weights_digest,
            "topology_digest": topology_digest_hex,
            "model_config_digest": model_digest, "completion_steps": completion_steps,
            "extension_sha256": str(config["extension_sha256"]), "crate_version": str(config["crate_version"]),
            "backend_build_digest": str(config["backend_build_digest"]),
            "pre_state_digest": fresh_pre_digest, "shape": [completion_steps, n],
            "raster_digest": _sha(_le_bytes(raster, "|b1")),
        }
        for key, value in expected.items():
            if report.get(key) != value:
                raise MaterializeError(f"calibration worker report field {key!r} is not bound to its input")
        if not _is_sha256(report["post_state_digest"]):
            raise MaterializeError("calibration worker post-state digest is not a sha256 digest")
        if process.pid in pids or nonce in nonces:
            raise MaterializeError("calibration worker reuses a PID or nonce")
        pids.add(process.pid)
        nonces.add(nonce)
        ordered_pids.append(process.pid)
        evidence.append({
            "record_id": str(ids[index]), "worker_pid": int(process.pid), "nonce": nonce,
            "pre_state_digest": fresh_pre_digest, "post_state_digest": str(report["post_state_digest"]),
            "input_weights_digest": weights_digest, "final_weights_digest": weights_digest,
            "topology_digest": topology_digest_hex, "cue_digest": cue_digest,
            "backend_build_digest": str(config["backend_build_digest"]),
            "raster_digest": str(report["raster_digest"]),
        })
        signatures[index] = bank.temporal_signature_v2(raster, completion_steps, n)
    return signatures, ordered_pids, evidence


_WORKER_REPORT_KEYS = frozenset({
    "pid", "record_id", "cue_digest", "nonce", "weights_digest", "final_weights_digest",
    "topology_digest", "model_config_digest", "completion_steps", "extension_sha256", "crate_version",
    "backend_build_digest", "pre_state_digest", "post_state_digest", "shape",
    "raster_digest", "raster_b64",
})


def _parse_worker_report(stdout: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise MaterializeError(f"calibration worker report carries a non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, item in pairs:
            if key in seen:
                raise MaterializeError(f"worker report contains a duplicate key {key!r}")
            seen[key] = item
        return seen

    try:
        report = json.loads(
            stdout.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializeError("calibration worker report is not strict JSON") from error
    if not isinstance(report, dict):
        raise MaterializeError("calibration worker report must be a JSON object")
    if set(report) != _WORKER_REPORT_KEYS:
        raise MaterializeError("calibration worker report does not carry the exact expected fields")
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    if stdout != canonical:
        raise MaterializeError("calibration worker report is not exactly canonical JSON on stdout")
    return cast(dict[str, Any], report)


def _authenticated_worker_bytes(path: Path, expected_digest: str, label: str) -> bytes:
    body = _read_regular_file(path, label)
    if _sha(body) != expected_digest:
        raise MaterializeError(f"{label} worker input digest mismatch")
    return body


def _cmd_calibrate(arguments: argparse.Namespace) -> dict[str, Any]:
    from snn_memory.state import validate_weights

    completion = int(arguments.completion_steps)
    if completion < 32 or completion % 8 != 0:
        raise MaterializeError("worker completion window must be at least 32 steps and divisible by eight")
    model_bytes = _authenticated_worker_bytes(
        Path(arguments.model_config), arguments.model_config_digest, "model-config")
    parsed_model = _strict_json_object(model_bytes, "worker model-config")
    model = _model_from_config(parsed_model)
    if parsed_model != model.to_dict():
        raise MaterializeError("worker model config is not the full canonical configuration")
    n = model.n_neurons
    weights_bytes = _authenticated_worker_bytes(Path(arguments.weights), arguments.weights_digest, "weights")
    topology_bytes = _authenticated_worker_bytes(
        Path(arguments.topology), arguments.topology_digest, "topology")
    cue_bytes = _authenticated_worker_bytes(Path(arguments.cue), arguments.cue_digest, "calibration cue")
    if bool(np.any(np.frombuffer(topology_bytes, dtype=np.uint8) > 1)):
        raise MaterializeError("worker topology carries a byte outside {0, 1}")
    weights = np.ascontiguousarray(np.frombuffer(weights_bytes, dtype="<f8").reshape(n, n), dtype=np.float64)
    topology = np.ascontiguousarray(np.frombuffer(topology_bytes, dtype="|b1").reshape(n, n), dtype=np.bool_)
    validate_weights(weights, topology, model)
    cue_flat = np.frombuffer(cue_bytes, dtype="<f8")
    if not bool(np.isfinite(cue_flat).all()):
        raise MaterializeError("calibration cue current carries a non-finite value")
    cue_steps = int(cue_flat.size // n)
    if cue_steps < 1 or cue_flat.size != cue_steps * n:
        raise MaterializeError("calibration cue current is not a whole (timesteps, n_neurons) grid")
    cue = np.ascontiguousarray(cue_flat.reshape(cue_steps, n), dtype=np.float64)
    packets = np.concatenate([cue, np.zeros((completion, n), dtype=np.float64)], axis=0)
    backend = load_stream_backend(BackendIdentity(2, arguments.crate_version, arguments.extension_sha256))
    if str(arguments.backend_build_digest) != backend.identity.extension_sha256:
        raise MaterializeError("worker backend-build digest is not the loaded extension identity")
    pre_state = _fresh_state(model)
    result = backend.run(_stream_inputs(model, weights, topology, packets, pre_state),
                         cue_steps, False, model)
    final_weights_digest = _sha(_le_bytes(np.ascontiguousarray(result.final_weights, dtype=np.float64), "<f8"))
    if final_weights_digest != arguments.weights_digest:
        raise MaterializeError("calibration with plasticity disabled must not change the weights")
    raster = _completion_raster(result, cue_steps, completion, n)
    raster_bytes = _le_bytes(raster, "|b1")
    return {
        "pid": os.getpid(), "record_id": arguments.record_id, "cue_digest": arguments.cue_digest,
        "nonce": arguments.nonce, "weights_digest": arguments.weights_digest,
        "final_weights_digest": final_weights_digest, "topology_digest": arguments.topology_digest,
        "model_config_digest": arguments.model_config_digest,
        "completion_steps": completion, "extension_sha256": backend.identity.extension_sha256,
        "crate_version": backend.identity.crate_version, "backend_build_digest": arguments.backend_build_digest,
        "pre_state_digest": _state_signature(pre_state), "post_state_digest": _state_signature(_result_state(result)),
        "shape": [completion, n], "raster_digest": _sha(raster_bytes),
        "raster_b64": base64.b64encode(raster_bytes).decode("ascii"),
    }


def _build_manifest(config: Mapping[str, Any], model: ModelConfig, seed: int, condition: str,
                    ids: Sequence[str], topology: BoolArray, outgoing: CsrArrays, incoming: CsrArrays,
                    weights: FloatArray, tf_state: StateArrays, probe_state: StateArrays,
                    signatures: FloatArray, schedule: Sequence[Mapping[str, Any]], epochs_completed: int,
                    training_digests: Sequence[Mapping[str, str]],
                    calibration_digests: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    topo_digest = topology_digest(np.ascontiguousarray(topology, dtype=np.bool_), model.n_excitatory)
    record_ids_digest = bundle.ordered_record_ids_digest(ids)
    return {
        "schema_version": 2,
        "experiment_digest": str(config["experiment_digest"]),
        "dataset_digest": str(config["d2_file_sha256"]),
        "candidate_set_digest": record_ids_digest,
        "task_set_digest": str(config["task_set_digest"]),
        "seed": seed,
        "condition": condition,
        "ordered_record_ids": list(ids),
        "topology_digest": topo_digest,
        "encoder": {
            "checkpoint_locator": str(config["encoder_locator"]),
            "directory_digest": str(config["encoder_digest"]),
            "config_digest": canonical_config_digest(config["encoder_config"]),
        },
        "model_config_digest": canonical_config_digest(model.to_dict()),
        "input_current": float(config["input_current"]),
        "training_source_digests": [dict(entry) for entry in training_digests],
        "calibration_cue_digests": [dict(entry) for entry in calibration_digests],
        "adjacency": {
            "orientation": "row-pre-column-post",
            "outgoing_digest": bundle._csr_digest("outgoing", outgoing),
            "incoming_digest": bundle._csr_digest("incoming", incoming),
            "topology_digest": topo_digest,
        },
        "arrays": {
            "weights_digest": bundle.checkpoint_component_digest("weights", "<f8", weights),
            "training_final_state_digest": bundle._state_digest(tf_state),
            "probe_initial_state_digest": bundle._state_digest(probe_state),
            "signatures_digest": bundle.checkpoint_component_digest("signatures", "<f8", signatures),
            "record_ids_digest": record_ids_digest,
        },
        "epochs_completed": epochs_completed,
        "replay_order_digest": bundle.replay_schedule_digest(schedule),
        "build": {
            "python_wheel_digest": str(config["python_wheel_sha256"]),
            "rust_wheel_digest": str(config["rust_wheel_sha256"]),
            "backend_version": str(config["backend_version"]),
            "backend_build_digest": str(config["backend_build_digest"]),
            "repository_head": str(config["repository_head"]),
            "dirty_tree_digest": str(config["dirty_tree_digest"]),
            "patch_digest": str(config["patch_digest"]),
        },
        "scoring_calibration": {
            "abstention_threshold": float(config["abstention_threshold"]),
            "metric": str(config["metric"]),
            "development_artifact_digest": str(config["development_artifact_digest"]),
        },
    }


def _build_identities(config: Mapping[str, Any], model: ModelConfig, seed: int, condition: str,
                      ids: Sequence[str]) -> dict[str, Any]:
    return {
        "experiment_digest": str(config["experiment_digest"]),
        "dataset_digest": str(config["d2_file_sha256"]),
        "task_set_digest": str(config["task_set_digest"]),
        "candidate_set_digest": bundle.ordered_record_ids_digest(ids),
        "d1_file_sha256": str(config["d1_file_sha256"]),
        "d2_file_sha256": str(config["d2_file_sha256"]),
        "seed": seed,
        "condition": condition,
        "n_neurons": model.n_neurons,
        "n_excitatory": model.n_excitatory,
        "weight_max": float(model.weight_max),
        "model_config": model.to_dict(),
        "encoder_directory_digest": str(config["encoder_digest"]),
        "encoder_config_digest": canonical_config_digest(config["encoder_config"]),
        "model_config_digest": canonical_config_digest(model.to_dict()),
        "input_current": float(config["input_current"]),
        "repository_head": str(config["repository_head"]),
        "patch_digest": str(config["patch_digest"]),
        "python_wheel_sha256": str(config["python_wheel_sha256"]),
        "rust_wheel_sha256": str(config["rust_wheel_sha256"]),
        "backend_build_digest": str(config["backend_build_digest"]),
        "materializer_module_sha256": str(config["materializer_module_sha256"]),
    }


def _lexical_candidates(ids: Sequence[str], row_digests: Sequence[str]) -> tuple[list[str], list[str]]:
    """Reorder (record id, signature digest) pairs into the lexical record-id order scoring needs."""
    order = sorted(range(len(ids)), key=lambda index: ids[index])
    return [ids[index] for index in order], [row_digests[index] for index in order]


def _build_scoring_target(config: Mapping[str, Any], ids: Sequence[str], row_digests: Sequence[str],
                          output: Path, completion_steps: int) -> Any:
    # Checkpoint/bank keep the frozen D1 candidate order; the scoring target requires the lexical
    # immutable-record-id order, so reorder the paired signature digests to match.
    lex_ids, lex_digests = _lexical_candidates(ids, row_digests)
    identities = {
        "schema_sha256": str(config["experiment_lock_schema_sha256"]),
        "python_wheel_sha256": str(config["python_wheel_sha256"]),
        "rust_wheel_sha256": str(config["rust_wheel_sha256"]),
        "backend_extension_sha256": str(config["extension_sha256"]),
        "experiment_lock_module_sha256": str(config["experiment_lock_module_sha256"]),
        "gb_preflight_module_sha256": str(config["gb_preflight_module_sha256"]),
        "candidate_bank_digest": candidate_bank_digest(lex_ids, lex_digests),
    }
    payload: dict[str, Any] = {
        "schema_version": 2, "artifact_type": "snn-memory-scoring-target-v2",
        "state": "fixture_only", "lane_role": "lane_p", "completion_steps": completion_steps, "bins": 8,
        "signature_dtype": "<f8", "similarity": "cosine",
        "zero_norm_rule": "zero-score-when-either-norm-is-zero", "score_order": "descending",
        "tie_rule": "lexical-record-id",
        "abstention": {"threshold": float(config["abstention_threshold"]), "rule": "strict-greater-than"},
        "correctness_rule": "exact-record-id", "top_k": 5, "max_payload_utf8_bytes": 20000,
        "candidate_order": lex_ids,
        "candidate_signature_digests": lex_digests,
        "identities": identities,
    }
    payload["scorer_digest"] = scorer_identity_digest(payload)
    return write_artifact(payload, output)


def _merge_calibration_evidence(calibration: Sequence[Mapping[str, Any]],
                                worker_evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(calibration) != len(worker_evidence):
        raise MaterializeError("calibration and worker evidence lengths differ")
    merged: list[dict[str, Any]] = []
    for encoder_entry, worker_entry in zip(calibration, worker_evidence):
        if str(encoder_entry["record_id"]) != str(worker_entry["record_id"]):
            raise MaterializeError("calibration and worker evidence record order differs")
        deterministic_worker = {key: value for key, value in dict(worker_entry).items()
                                if key not in ("worker_pid", "nonce")}
        merged.append({**dict(encoder_entry), **deterministic_worker})
    return merged


def _build_bank_manifest(config: Mapping[str, Any], model: ModelConfig, seed: int, condition: str,
                         ids: Sequence[str], checkpoint: Any, scoring: Any, completion_steps: int,
                         row_digests: Sequence[str], calibration: Sequence[Mapping[str, Any]],
                         worker_evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = model.n_neurons
    return {
        "schema_version": 2, "artifact_type": "snn-memory-candidate-bank-v2",
        "state": "fixture_only", "lane_role": "lane_p",
        "checkpoint": {
            "descriptor_self_sha256": checkpoint.descriptor_self_sha256,
            "manifest_file_sha256": checkpoint.manifest_file_sha256,
            "manifest_payload_self_sha256": checkpoint.manifest_payload_self_sha256,
        },
        "identities": {
            "experiment_digest": str(config["experiment_digest"]),
            "dataset_digest": str(config["d2_file_sha256"]),
            "task_set_digest": str(config["task_set_digest"]),
            "candidate_set_digest": bundle.ordered_record_ids_digest(ids),
            "d1_file_sha256": str(config["d1_file_sha256"]),
            "d2_file_sha256": str(config["d2_file_sha256"]),
            "lane_role": "lane_p", "seed": seed, "condition": condition,
            "ordered_record_ids": list(ids),
        },
        "signature_layout": {
            "semantic_version": "snn-temporal-signature-v2", "bins": 8, "layout": "neuron-major",
            "n_neurons": n, "signature_width": 8 * n, "completion_steps": completion_steps, "dtype": "<f8",
        },
        "calibration": _merge_calibration_evidence(calibration, worker_evidence),
        "scoring": {
            "scoring_target_self_sha256": scoring.payload_self_sha256,
            "candidate_bank_digest": candidate_bank_digest(*_lexical_candidates(ids, row_digests)),
            "abstention_rule": "strict_greater_than",
            "calibration_spec_digest": str(config["calibration_spec_digest"]),
            "development_artifact_digest": str(config["development_artifact_digest"]),
            "completion_steps": completion_steps,
        },
        "build": {
            "encoder_directory_digest": str(config["encoder_digest"]),
            "encoder_config_digest": canonical_config_digest(config["encoder_config"]),
            "materializer_module_sha256": str(config["materializer_module_sha256"]),
            "backend_build_digest": str(config["backend_build_digest"]),
            "python_wheel_sha256": str(config["python_wheel_sha256"]),
            "rust_wheel_sha256": str(config["rust_wheel_sha256"]),
            "repository_head": str(config["repository_head"]),
            "patch_digest": str(config["patch_digest"]),
            "checkpoint_schema_sha256": str(config["checkpoint_schema_sha256"]),
            "bank_schema_sha256": str(config["bank_schema_sha256"]),
        },
    }


def _install_output_atomically(parent_fd: int, staging_name: str, final_name: str) -> None:
    """Atomically install a staged output with no clobber relative to a verified parent fd."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    result = libc.renameat2(
        parent_fd, os.fsencode(staging_name), parent_fd, os.fsencode(final_name), 1
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise MaterializeError("materialization output already exists")
        raise MaterializeError(f"atomic output install failed: {os.strerror(code)}")
    os.fsync(parent_fd)


def _cmd_materialize(arguments: argparse.Namespace) -> dict[str, Any]:
    from snn_memory.sentence_encoder import LocalSentenceEncoder

    config = _strict_config(Path(arguments.config))
    _reject_evaluation_roles(config)
    _reject_unknown_config(config)
    condition = str(config["condition"])
    if condition in _FORBIDDEN_CONDITIONS:
        raise MaterializeError(f"condition {condition!r} is forbidden in D4-A")
    if condition not in _ALLOWED_CONDITIONS:
        raise MaterializeError(f"condition {condition!r} is not an accepted D4-A condition")
    seed = int(config["seed"])
    completion_steps = int(config["completion_steps"])
    _validate_live_identities(config)
    encoder_module = LocalSentenceEncoder(config["encoder_checkpoint"])
    if encoder_module.digest != str(config["encoder_digest"]):
        raise MaterializeError("sentence-encoder directory digest differs from the pinned digest")

    prepared = _prepare_inputs(config, encoder_module, need_training_currents=(condition == "trained"))
    model = prepared.model
    ids = list(prepared.ids)
    training_digests = list(prepared.training_source_digests)
    calibration_digests = list(prepared.calibration_cue_digests)
    backend = load_stream_backend(_backend_identity(config))
    if str(config["backend_build_digest"]) != backend.identity.extension_sha256:
        raise MaterializeError("declared backend-build digest is not the loaded extension identity")
    if condition == "trained":
        weights, topology, tf_state, schedule = _train_checkpoint(
            backend, model, seed, prepared.training_currents, int(config["epochs"]), ids
        )
        epochs_completed = int(config["epochs"])
    elif condition == "untrained":
        weights, topology = _initial_weights(model, seed)
        tf_state, schedule, epochs_completed = _fresh_state(model), [], 0
    else:
        weights, topology, tf_state, schedule, epochs_completed = _control_inputs(
            Path(config["trained_bundle"]), model, seed, condition, config, ids
        )

    final_output = Path(config["output"])
    absolute_final = final_output if final_output.is_absolute() else Path.cwd() / final_output
    if absolute_final.name in (os.curdir, os.pardir, ""):
        raise MaterializeError("materialization output has no valid final component")
    parent_fd = bundle._open_dir_from_root(absolute_final.parent, "materialization output parent directory")
    try:
        try:
            os.stat(absolute_final.name, dir_fd=parent_fd, follow_symlinks=False)
            raise MaterializeError("materialization output already exists")
        except FileNotFoundError:
            pass
        staging_name = f".{absolute_final.name}.staging-{os.urandom(8).hex()}"
        os.mkdir(staging_name, 0o755, dir_fd=parent_fd)
        staging = absolute_final.parent / staging_name
        try:
            report = _produce_output(
                config, staging, model, seed, condition, ids, weights, topology, tf_state, schedule,
                epochs_completed, prepared, training_digests, calibration_digests, completion_steps,
                final_output,
            )
            _install_output_atomically(parent_fd, staging_name, absolute_final.name)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    finally:
        os.close(parent_fd)
    return report


def _produce_output(config: Mapping[str, Any], staging: Path, model: ModelConfig, seed: int,
                    condition: str, ids: Sequence[str], weights: FloatArray, topology: BoolArray,
                    tf_state: StateArrays, schedule: Sequence[Mapping[str, Any]], epochs_completed: int,
                    prepared: PreparedInputs, training_digests: Sequence[Mapping[str, str]],
                    calibration_digests: Sequence[Mapping[str, str]], completion_steps: int,
                    final_output: Path) -> dict[str, Any]:
    probe_state = _fresh_state(model)
    signatures, worker_pids, worker_evidence = _calibrate_candidates(
        config, staging / "workers", model, weights, topology, prepared.calibration_currents, list(ids),
        completion_steps,
    )
    row_digests = [bank.signature_row_digest(record_id, signatures[index])
                   for index, record_id in enumerate(ids)]
    outgoing = _csr_from_topology(topology, outgoing=True)
    incoming = _csr_from_topology(topology, outgoing=False)
    inputs = CheckpointBundleInputs(
        topology=np.ascontiguousarray(topology, dtype=np.bool_),
        outgoing=outgoing, incoming=incoming,
        weights=np.ascontiguousarray(weights, dtype=np.float64),
        training_final_state=tf_state, probe_initial_state=probe_state,
        signatures=np.ascontiguousarray(signatures, dtype="<f8"),
        ordered_record_ids=tuple(ids), replay_schedule=tuple(schedule),
    )
    manifest = _build_manifest(config, model, seed, condition, list(ids), topology, outgoing, incoming,
                               weights, tf_state, probe_state, signatures, schedule, epochs_completed,
                               training_digests, calibration_digests)
    identities = _build_identities(config, model, seed, condition, list(ids))
    checkpoint = write_checkpoint_bundle_v2(
        staging / "checkpoint", state="fixture_only", manifest=manifest, identities=identities,
        inputs=inputs,
    )
    scoring = _build_scoring_target(config, list(ids), row_digests, staging / "scoring_target.json",
                                    completion_steps)
    bank_manifest = _build_bank_manifest(config, model, seed, condition, list(ids), checkpoint, scoring,
                                         completion_steps, row_digests,
                                         list(prepared.calibration_evidence), worker_evidence)
    validated_bank = bank.write_candidate_bank_v2(
        staging / "bank", manifest=bank_manifest,
        signatures=np.ascontiguousarray(signatures, dtype="<f8"),
    )
    return {
        "seed": seed, "condition": condition, "n_neurons": model.n_neurons,
        "epochs_completed": epochs_completed, "ordered_record_ids": list(ids),
        "checkpoint_dir": str(final_output / "checkpoint"), "bank_dir": str(final_output / "bank"),
        "checkpoint_descriptor_self_sha256": checkpoint.descriptor_self_sha256,
        "checkpoint_manifest_file_sha256": checkpoint.manifest_file_sha256,
        "bank_self_sha256": validated_bank.self_sha256,
        "scoring_target_self_sha256": scoring.payload_self_sha256,
        "candidate_bank_digest": candidate_bank_digest(list(ids), list(row_digests)),
        "worker_pids": worker_pids, "materialize_pid": os.getpid(),
        "worker_evidence": [dict(entry) for entry in worker_evidence],
    }


def _cmd_read_checkpoint(arguments: argparse.Namespace) -> dict[str, Any]:
    validated = read_checkpoint_bundle_v2(Path(arguments.bundle))
    return {"descriptor_self_sha256": validated.descriptor_self_sha256,
            "manifest_file_sha256": validated.manifest_file_sha256,
            "seed": validated.manifest["seed"], "condition": validated.manifest["condition"]}


def _cmd_read_bank(arguments: argparse.Namespace) -> dict[str, Any]:
    validated = bank.read_candidate_bank_v2(Path(arguments.bank))
    return {"self_sha256": validated.self_sha256,
            "row_digests": list(validated.row_digests),
            "candidate_bank_digest": validated.manifest["scoring"]["candidate_bank_digest"]}


def _reconcile_bank_checkpoint(validated_bank: Any, checkpoint: Any) -> None:
    """Reconcile every bank identity, build, signature, and calibration binding to the checkpoint."""
    identities = checkpoint.descriptor["identities"]
    manifest = checkpoint.manifest
    bank_identities = validated_bank.manifest["identities"]
    bank_build = validated_bank.manifest["build"]
    ordered = tuple(bank_identities["ordered_record_ids"])
    identity_pairs: tuple[tuple[Any, Any, str], ...] = (
        (bank_identities["experiment_digest"], identities["experiment_digest"], "experiment digest"),
        (bank_identities["dataset_digest"], identities["dataset_digest"], "dataset digest"),
        (bank_identities["task_set_digest"], identities["task_set_digest"], "task-set digest"),
        (bank_identities["candidate_set_digest"], identities["candidate_set_digest"], "candidate-set digest"),
        (bank_identities["d1_file_sha256"], identities["d1_file_sha256"], "D1 digest"),
        (bank_identities["d2_file_sha256"], identities["d2_file_sha256"], "D2 digest"),
        (bank_identities["lane_role"], checkpoint.descriptor["lane_role"], "lane role"),
        (int(bank_identities["seed"]), int(identities["seed"]), "seed"),
        (bank_identities["condition"], identities["condition"], "condition"),
        (bank_build["encoder_directory_digest"], identities["encoder_directory_digest"], "encoder directory"),
        (bank_build["encoder_config_digest"], identities["encoder_config_digest"], "encoder config"),
        (bank_build["materializer_module_sha256"], identities["materializer_module_sha256"], "materializer"),
        (bank_build["backend_build_digest"], identities["backend_build_digest"], "backend build"),
        (bank_build["python_wheel_sha256"], identities["python_wheel_sha256"], "python wheel"),
        (bank_build["rust_wheel_sha256"], identities["rust_wheel_sha256"], "rust wheel"),
        (bank_build["repository_head"], identities["repository_head"], "repository head"),
        (bank_build["patch_digest"], identities["patch_digest"], "patch digest"),
    )
    for bank_value, checkpoint_value, name in identity_pairs:
        if bank_value != checkpoint_value:
            raise MaterializeError(f"candidate bank {name} differs from the checkpoint")
    if validated_bank.manifest["state"] != checkpoint.descriptor["state"]:
        raise MaterializeError("candidate bank state differs from the checkpoint descriptor state")
    if bundle.ordered_record_ids_digest(ordered) != bank_identities["candidate_set_digest"]:
        raise MaterializeError("candidate bank candidate-set digest does not recompute from the ordered records")
    if _le_bytes(validated_bank.signatures, "<f8") != _le_bytes(checkpoint.inputs.signatures, "<f8"):
        raise MaterializeError("candidate bank signatures differ from the checkpoint signatures")
    completion = int(validated_bank.manifest["signature_layout"]["completion_steps"])
    if completion != int(validated_bank.manifest["scoring"]["completion_steps"]):
        raise MaterializeError("candidate bank layout and scoring completion windows disagree")
    calibration = validated_bank.manifest["calibration"]
    cue_by_id = {str(entry["record_id"]): str(entry["digest"])
                 for entry in manifest["calibration_cue_digests"]}
    if tuple(str(entry["record_id"]) for entry in calibration) != ordered:
        raise MaterializeError("candidate bank calibration order differs from the ordered records")
    for entry in calibration:
        rid = str(entry["record_id"])
        if cue_by_id.get(rid) != str(entry["source_cue_sha256"]):
            raise MaterializeError(f"calibration source cue digest for {rid} differs from the checkpoint")
        if entry["encoder_directory_digest"] != identities["encoder_directory_digest"]:
            raise MaterializeError(f"calibration encoder directory for {rid} differs from the checkpoint")
        if entry["encoder_config_digest"] != identities["encoder_config_digest"]:
            raise MaterializeError(f"calibration encoder config for {rid} differs from the checkpoint")
        if entry["model_config_digest"] != identities["model_config_digest"]:
            raise MaterializeError(f"calibration model config for {rid} differs from the checkpoint")
        if float(entry["input_current"]) != float(identities["input_current"]):
            raise MaterializeError(f"calibration input current for {rid} differs from the checkpoint")


def _cmd_bind(arguments: argparse.Namespace) -> dict[str, Any]:
    checkpoint = read_checkpoint_bundle_v2(Path(arguments.checkpoint))
    validated_bank = bank.read_candidate_bank_v2(Path(arguments.bank))
    binding = validated_bank.manifest["checkpoint"]
    if (binding["descriptor_self_sha256"] != checkpoint.descriptor_self_sha256
            or binding["manifest_file_sha256"] != checkpoint.manifest_file_sha256
            or binding["manifest_payload_self_sha256"] != checkpoint.manifest_payload_self_sha256):
        raise MaterializeError("candidate bank does not bind the presented checkpoint bundle")
    if validated_bank.manifest["identities"]["ordered_record_ids"] != checkpoint.manifest["ordered_record_ids"]:
        raise MaterializeError("candidate bank record order differs from the checkpoint")
    _reconcile_bank_checkpoint(validated_bank, checkpoint)
    scoring_bytes = _read_regular_file(Path(arguments.scoring_target), "scoring target")
    scoring = validate_artifact_bytes(scoring_bytes, expected_type="snn-memory-scoring-target-v2")
    scoring_block = validated_bank.manifest["scoring"]
    if scoring.payload_self_sha256 != scoring_block["scoring_target_self_sha256"]:
        raise MaterializeError("candidate bank scoring-target digest differs from the presented target")
    bank_ids = [str(value) for value in validated_bank.manifest["identities"]["ordered_record_ids"]]
    lexical_bank_ids, lexical_bank_digests = _lexical_candidates(bank_ids, list(validated_bank.row_digests))
    if list(scoring.payload["candidate_order"]) != lexical_bank_ids:
        raise MaterializeError("scoring-target candidate order differs from the lexical bank order")
    if list(scoring.payload["candidate_signature_digests"]) != lexical_bank_digests:
        raise MaterializeError("scoring-target signature digests differ from the bank rows")
    if scoring.payload["identities"]["candidate_bank_digest"] != scoring_block["candidate_bank_digest"]:
        raise MaterializeError("scoring-target candidate-bank digest differs from the bank")
    layout = validated_bank.manifest["signature_layout"]
    if int(scoring.payload["completion_steps"]) != int(layout["completion_steps"]):
        raise MaterializeError("scoring-target completion window differs from the bank layout")
    if int(scoring.payload["completion_steps"]) != int(scoring_block["completion_steps"]):
        raise MaterializeError("scoring-target completion window differs from the bank scoring block")
    if int(scoring.payload["bins"]) != int(layout["bins"]):
        raise MaterializeError("scoring-target bin count differs from the bank layout")
    if str(scoring.payload["signature_dtype"]) != str(layout["dtype"]):
        raise MaterializeError("scoring-target signature dtype differs from the bank layout")
    if str(scoring.payload["abstention"]["rule"]) != "strict-greater-than":
        raise MaterializeError("scoring-target abstention rule is not strict-greater-than")
    if scoring_block["abstention_rule"] != "strict_greater_than":
        raise MaterializeError("bank abstention rule is not strict_greater_than")
    return {"bound": True, "checkpoint_descriptor_self_sha256": checkpoint.descriptor_self_sha256,
            "bank_self_sha256": validated_bank.self_sha256,
            "scoring_target_self_sha256": scoring.payload_self_sha256}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the D4-A materializer CLI and return a process exit code (0 ok, 2 fail-closed)."""
    parser = argparse.ArgumentParser(prog="python -m snn_memory.checkpoint_materialize_v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--config", type=Path, required=True)
    materialize.set_defaults(handler=_cmd_materialize)
    read_checkpoint = subparsers.add_parser("read-checkpoint")
    read_checkpoint.add_argument("--bundle", type=Path, required=True)
    read_checkpoint.set_defaults(handler=_cmd_read_checkpoint)
    read_bank = subparsers.add_parser("read-bank")
    read_bank.add_argument("--bank", type=Path, required=True)
    read_bank.set_defaults(handler=_cmd_read_bank)
    bind = subparsers.add_parser("bind")
    bind.add_argument("--checkpoint", type=Path, required=True)
    bind.add_argument("--bank", type=Path, required=True)
    bind.add_argument("--scoring-target", type=Path, required=True)
    bind.set_defaults(handler=_cmd_bind)
    calibrate = subparsers.add_parser("_calibrate")
    calibrate.add_argument("--weights", type=Path, required=True)
    calibrate.add_argument("--weights-digest", required=True)
    calibrate.add_argument("--topology", type=Path, required=True)
    calibrate.add_argument("--topology-digest", required=True)
    calibrate.add_argument("--cue", type=Path, required=True)
    calibrate.add_argument("--cue-digest", required=True)
    calibrate.add_argument("--model-config", type=Path, required=True)
    calibrate.add_argument("--model-config-digest", required=True)
    calibrate.add_argument("--record-id", required=True)
    calibrate.add_argument("--nonce", required=True)
    calibrate.add_argument("--completion-steps", type=int, required=True)
    calibrate.add_argument("--extension-sha256", required=True)
    calibrate.add_argument("--crate-version", required=True)
    calibrate.add_argument("--backend-build-digest", required=True)
    calibrate.set_defaults(handler=_cmd_calibrate)
    arguments = parser.parse_args(argv)
    try:
        report = arguments.handler(arguments)
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
