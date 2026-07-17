# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN schema-v2 experiment-lock artifact I/O and semantic validation

"""Canonical D3 experiment-lock artifact I/O and fail-closed semantic validation.

This module owns only the five canonical experiment-lock artifact types
(``snn-memory-scoring-target-v2``, ``snn-memory-gb-calibration-spec-v2``,
``snn-memory-experiment-lock-v2``, ``snn-memory-task-completeness-v2``, and
``snn-memory-gb-preflight-evidence-v2``) and their semantic lock/completeness
rules. Every artifact is canonical UTF-8 JSON that rejects duplicate keys and
non-finite numbers, carries ``schema_version`` 2, an exact ``artifact_type``, a
``state``/``lane_role``, and a domain-separated ``self_sha256``. Readers
authenticate raw bytes before decoding and retain the validated byte snapshot;
unknown role, state, field, digest, task, condition, or abort reason fails closed.

The lock surface consumes D1 only through its authenticated
``snn_memory.source_universe`` public artifact and D2 only through the
``snn_memory.cue_materializer`` ``read_cue_set``/``read_cue_bundle`` public
surfaces; it never imports D1/D2 private helpers or rewrites either artifact. It
binds the four mandatory Amendment-2 digests
(``bd4ec3b4b0e77f88b4824323ea3b1f04a3037aaf6cac615bb93b7b0f6af0445a``,
``ae715600b5e5cba8603ae67bda8733a1b1fe3a181731a4442d9c77e6c1c0299d``,
``07ef76b74a29d55c84c258e12a66ed8124b8953867a6830cb32419b9c752437b``, and
``69452969cd5de3da37a79c81f7ed050893f8e6694ae0184cff8c58800b980abd``) alongside
the plan, Amendment 1, schema, wheel, extension, encoder, HEAD, and D1/D2/
scoring-target/calibration/candidate-order/lane-role digests.

D3 artifacts may declare only the ``pre_result`` or ``fixture_only`` states
(Section 5.1); no D3 artifact claims a G1/G2/G3, utility, efficacy, scientific,
product, continuity, consciousness, SOTA, or AOT pass. The writer may implement
the lock capability but this D3 run exercises it only against temporary
``fixture_only`` artifacts under a temporary runtime root; it never performs
the live dataset-lock ceremony or writes a scientific artifact into the repository.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, cast

_SCHEMA_NAME = "snn_memory_experiment_lock_v2.schema.json"
SCHEMA_VERSION = 2
SEEDS: tuple[int, ...] = (11, 29, 47, 71, 101, 131, 167, 211, 257, 307)

AMENDMENT2_FINAL_SHA256 = "bd4ec3b4b0e77f88b4824323ea3b1f04a3037aaf6cac615bb93b7b0f6af0445a"
AMENDMENT2_U2_SHA256 = "ae715600b5e5cba8603ae67bda8733a1b1fe3a181731a4442d9c77e6c1c0299d"
AMENDMENT2_DIGEST_PIN_SHA256 = "07ef76b74a29d55c84c258e12a66ed8124b8953867a6830cb32419b9c752437b"
GB_DESIGN_INPUT_SHA256 = "69452969cd5de3da37a79c81f7ed050893f8e6694ae0184cff8c58800b980abd"

_ARTIFACT_DOMAINS: Mapping[str, bytes] = MappingProxyType(
    {
        "snn-memory-scoring-target-v2": b"remanentia:snn-v2-scoring-target:v1\0",
        "snn-memory-gb-calibration-spec-v2": b"remanentia:snn-v2-gb-calibration-spec:v1\0",
        "snn-memory-experiment-lock-v2": b"remanentia:snn-v2-experiment-lock:v1\0",
        "snn-memory-task-completeness-v2": b"remanentia:snn-v2-task-completeness:v1\0",
        "snn-memory-gb-preflight-evidence-v2": b"remanentia:snn-v2-gb-preflight-evidence:v1\0",
    }
)
ABORT_REASONS = frozenset(
    {
        "digest_mismatch",
        "cross_role_contamination",
        "development_locked_overlap",
        "expected_task_incompleteness",
        "duplicate_or_unexpected_task",
        "per_seed_imbalance",
        "process_reuse",
        "task_process_mismatch",
        "nonzero_completion_input",
        "telemetry_invalidity",
        "no_admissible_calibration",
        "post_lock_mutation",
        "representation_or_dynamics_limited",
    }
)

FAMILY_CODES: Mapping[str, str] = MappingProxyType(
    {"shuffled": "sh", "zero_recurrence": "zr", "no_input": "ni", "no_match": "nm"}
)
_SYNTHETIC_POSITIVES_PER_SEED = 16
_SYNTHETIC_SELECTION_PER_SEED = 10

# Toolchain identity fields that a lock, its scoring target, and its calibration spec
# must all agree on (schema, both wheels, backend extension, both installed modules).
_SHARED_IDENTITY_KEYS: tuple[str, ...] = (
    "schema_sha256",
    "python_wheel_sha256",
    "rust_wheel_sha256",
    "backend_extension_sha256",
    "experiment_lock_module_sha256",
    "gb_preflight_module_sha256",
)

# Independent pure-Python mirror of the frozen G-B completion descriptor arithmetic
# (Amendment 2 ``bd4ec3b4…`` and its U-2 addendum ``ae715600…``). It recomputes every
# derived evidence field from the stored recurrent rows, ordered spike raster, dt_ms,
# and calibration thresholds so a resealed gb-preflight artifact cannot carry a forged
# energy, half-life, trajectory class, or Hamming vector. It shares no code with
# ``snn_memory.gb_preflight`` and must stay bit-for-bit equal to it.
_GB_BINS = 8
_GB_TAIL_BINS = (4, 5, 6, 7)
_GB_CANDIDATE_LAGS = (1, 2, 3, 4)
_GB_EPSILON = 2.220446049250313e-16
_GB_RECURRENT_KEYS = (
    "e_signed_sum",
    "e_l1",
    "e_l2",
    "i_signed_sum",
    "i_l1",
    "i_l2",
    "net_signed_sum",
    "net_l1",
    "net_l2",
)
_GB_NET_L1_KEY = "net_l1"
_GB_NET_L2_KEY = "net_l2"


class ExperimentLockError(ValueError):
    """A fail-closed experiment-lock contract violation."""


class LaneRole(str, Enum):
    """Closed set of disjoint causal lanes."""

    LANE_P = "lane_p"
    LANE_C = "lane_c"
    LANE_H = "lane_h"


@dataclass(frozen=True)
class ValidatedArtifact:
    """An authenticated experiment-lock artifact with a read-only payload view."""

    artifact_type: str
    payload: Mapping[str, Any]
    canonical_bytes: bytes
    file_sha256: str
    payload_self_sha256: str


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _self_digest(payload: Mapping[str, Any], domain: bytes) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "self_sha256"}
    canonical = _canonical(unsigned)
    framed = domain + len(canonical).to_bytes(8, "big") + canonical
    return hashlib.sha256(framed).hexdigest()


def _strict_json(raw: bytes, context: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ExperimentLockError(f"{context} contains a duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(constant: str) -> None:
        raise ExperimentLockError(f"{context} contains non-finite JSON constant {constant}")

    def parse_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ExperimentLockError(f"{context} contains a non-finite JSON number")
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ExperimentLockError):
            raise
        raise ExperimentLockError(f"{context} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ExperimentLockError(f"{context} root must be an object")
    return value


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _schema_bytes() -> bytes:
    return files("snn_memory").joinpath("schema", _SCHEMA_NAME).read_bytes()


def _validate_schema(payload: Mapping[str, Any], context: str) -> None:
    schema = _strict_json(_schema_bytes(), "experiment-lock schema")
    validator_class = cast(Any, import_module("jsonschema")).Draft202012Validator
    try:
        validator_class(schema).validate(dict(payload))
    except Exception as error:
        raise ExperimentLockError(f"{context} schema validation failed") from error


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _read_regular_bytes(path: Path, context: str) -> bytes:
    absolute = path.absolute()
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise ExperimentLockError(f"{context} path is not an absolute POSIX path")
    if any(part in {"", ".", ".."} for part in parts[1:]):
        raise ExperimentLockError(f"{context} path contains a traversal component")
    # Race-safe component-wise walk: every parent is opened O_DIRECTORY|O_NOFOLLOW
    # relative to the previously held directory descriptor (openat), so a concurrent
    # symlink/rename of any component cannot redirect the read; the final regular file
    # is opened O_NOFOLLOW relative to the last held directory descriptor.
    try:
        directory = os.open(os.sep, _DIR_FLAGS)
    except OSError as error:
        raise ExperimentLockError(f"{context} root cannot be opened safely") from error
    descriptor = -1
    try:
        for component in parts[1:-1]:
            try:
                nested = os.open(component, _DIR_FLAGS, dir_fd=directory)
            except OSError as error:
                raise ExperimentLockError(f"{context} path traverses a symlink") from error
            os.close(directory)
            directory = nested
        try:
            descriptor = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
        except OSError as error:
            raise ExperimentLockError(f"{context} cannot be opened safely") from error
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ExperimentLockError(f"{context} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        os.close(directory)
    return b"".join(chunks)


def validate_artifact_bytes(
    raw: bytes,
    *,
    expected_type: str | None = None,
    expected_file_sha256: str | None = None,
) -> ValidatedArtifact:
    """Validate canonical artifact bytes, schema, self digest, and closed enums."""
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise ExperimentLockError("experiment-lock artifact SHA-256 mismatch")
    payload = _strict_json(raw, "experiment-lock artifact")
    artifact_type = payload.get("artifact_type")
    if not isinstance(artifact_type, str) or artifact_type not in _ARTIFACT_DOMAINS:
        raise ExperimentLockError("unknown or missing artifact_type")
    if expected_type is not None and artifact_type != expected_type:
        raise ExperimentLockError("artifact_type differs from the expected type")
    _validate_schema(payload, "experiment-lock artifact")
    if _canonical(payload) != raw:
        raise ExperimentLockError("experiment-lock JSON is not canonical")
    if payload["self_sha256"] != _self_digest(payload, _ARTIFACT_DOMAINS[artifact_type]):
        raise ExperimentLockError("experiment-lock self digest mismatch")
    _SEMANTIC_VALIDATORS[artifact_type](payload)
    return ValidatedArtifact(
        artifact_type=artifact_type,
        payload=_frozen(payload),
        canonical_bytes=raw,
        file_sha256=file_sha256,
        payload_self_sha256=cast(str, payload["self_sha256"]),
    )


def seal_payload(payload: dict[str, Any]) -> bytes:
    """Return canonical sealed bytes for a payload lacking its self digest."""
    artifact_type = payload.get("artifact_type")
    if not isinstance(artifact_type, str) or artifact_type not in _ARTIFACT_DOMAINS:
        raise ExperimentLockError("cannot seal an unknown artifact_type")
    body = {key: value for key, value in payload.items() if key != "self_sha256"}
    body["self_sha256"] = _self_digest(body, _ARTIFACT_DOMAINS[artifact_type])
    return _canonical(body)


def write_artifact(payload: dict[str, Any], output: Path) -> ValidatedArtifact:
    """Seal, validate, and atomically write one artifact without clobbering."""
    raw = seal_payload(payload)
    artifact = validate_artifact_bytes(raw)
    output = output.absolute()
    parent = output.parent
    if parent != parent.resolve(strict=True):
        raise ExperimentLockError("output parent must be an absolute canonical path")
    descriptor, temporary_name = _mkstemp(parent, output.name)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, output)
        except OSError as error:
            raise ExperimentLockError("atomic no-clobber artifact write failed") from error
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return artifact


def _mkstemp(parent: Path, name: str) -> tuple[int, str]:
    import tempfile

    return tempfile.mkstemp(prefix=f".{name}.tmp.", dir=parent)


def read_artifact(
    path: Path,
    *,
    expected_type: str | None = None,
    expected_file_sha256: str | None = None,
) -> ValidatedArtifact:
    """Read one artifact through a symlink-refusing single read and validate it."""
    raw = _read_regular_bytes(path.absolute(), "experiment-lock file")
    return validate_artifact_bytes(
        raw, expected_type=expected_type, expected_file_sha256=expected_file_sha256
    )


def _require_finite_non_negative_grid(payload: Mapping[str, Any], key: str) -> None:
    for value in payload[key]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ExperimentLockError(f"{key} must contain only numbers")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ExperimentLockError(f"{key} must contain only finite non-negative values")


def _validate_scoring_target_semantics(payload: Mapping[str, Any]) -> None:
    candidate_order = list(payload["candidate_order"])
    if candidate_order != sorted(candidate_order):
        raise ExperimentLockError(
            "scoring candidate order must be lexical immutable-record-ID order"
        )
    signatures = list(payload["candidate_signature_digests"])
    if len(signatures) != len(candidate_order):
        raise ExperimentLockError("one candidate signature digest is required per candidate")
    if payload["identities"]["candidate_bank_digest"] != candidate_bank_digest(
        candidate_order, signatures
    ):
        raise ExperimentLockError("scoring candidate-bank digest does not bind the candidate bank")
    if payload["scorer_digest"] != scorer_identity_digest(payload):
        raise ExperimentLockError("scoring scorer digest does not bind the scorer configuration")


def _validate_calibration_spec_semantics(payload: Mapping[str, Any]) -> None:
    for key in (
        "numerical_zero_floor_grid",
        "spike_drift_ceiling_grid",
        "current_drift_ceiling_grid",
        "representation_margin_floor_grid",
        "normalized_effective_rank_floor_grid",
        "settled_fraction_floor_grid",
        "wandering_ceiling_grid",
        "collapse_ceiling_grid",
        "abstention_grid",
    ):
        _require_finite_non_negative_grid(payload, key)
    generator = payload["synthetic_generator"]
    positives = list(generator["positive_ids"])
    negatives = list(generator["negative_ids"])
    if positives != synthetic_positive_ids():
        raise ExperimentLockError("synthetic positive IDs are not the canonical per-seed corpus")
    if negatives != synthetic_negative_ids(positives):
        raise ExperimentLockError("synthetic negative IDs are not the matched four-family corpus")
    if generator["config_digest"] != synthetic_config_digest():
        raise ExperimentLockError(
            "synthetic config digest does not bind the generator configuration"
        )
    selection_digest, validation_digest = synthetic_partition_digests(positives)
    if generator["selection_partition_digest"] != selection_digest:
        raise ExperimentLockError("synthetic selection partition digest is not bound")
    if generator["validation_partition_digest"] != validation_digest:
        raise ExperimentLockError("synthetic validation partition digest is not bound")
    if payload["task_set_digest"] != synthetic_task_set_digest(positives, negatives):
        raise ExperimentLockError(
            "calibration task-set digest does not bind the synthetic inventory"
        )


def _validate_experiment_lock_semantics(payload: Mapping[str, Any]) -> None:
    d1 = payload["d1"]
    d2 = payload["d2"]
    if _thaw(d2["source_universe"]) != _thaw(d1):
        raise ExperimentLockError("D2 source-universe binding differs from the D1 binding")
    if payload["identities"]["repository_head"] != d1["repository_head"]:
        raise ExperimentLockError("lock repository HEAD differs from the D1 binding")
    candidate_order = list(payload["candidate_order"])
    if tuple(candidate_order) != tuple(d1["selected_record_ids"]):
        raise ExperimentLockError("candidate order differs from the D1 selected record IDs")
    lane = payload["lane_role"]
    if payload["candidate_order_digest"] != candidate_order_digest(candidate_order):
        raise ExperimentLockError("lock candidate-order digest is not bound")
    if payload["lane_role_digest"] != lane_role_digest(lane):
        raise ExperimentLockError("lock lane-role digest is not bound")
    # The lane-identity block must be internally consistent with the lock's own fields;
    # threshold and candidate_bank are bound to the real calibration/scoring by the binder.
    identity = payload["lane_identity"]
    if identity["provenance"] != payload["candidate_order_digest"]:
        raise ExperimentLockError("lane provenance identity is not bound to the candidate order")
    if identity["schema"] != payload["identities"]["schema_sha256"]:
        raise ExperimentLockError("lane schema identity is not bound to the lock schema")
    if identity["task_set"] != payload["expected_task_set_digest"]:
        raise ExperimentLockError("lane task-set identity is not bound to the expected task set")
    if identity["scorer"] != payload["scoring_target_digest"]:
        raise ExperimentLockError("lane scorer identity is not bound to the scoring target")
    if identity["root"] != payload["output_root_digest"]:
        raise ExperimentLockError("lane root identity is not bound to the output root")
    if payload["lane_domain_digest"] != lane_domain_digest(lane, lane_identity_fields(payload)):
        raise ExperimentLockError("lock lane-domain digest is not bound")
    # The foreign-lane evidence is embedded in and bound to the lock (its self digests are part
    # of the lock's own self digest). Authenticate it, derive the foreign identity set only from
    # that bound evidence, and require it disjoint from the lock's complete Lane-P inventory —
    # every identity class the lock declares, not only the five distinguishing fields.
    foreign_lanes = payload["foreign_lanes"]
    _validate_foreign_lane_c(foreign_lanes["lane_c"])
    _validate_foreign_lane_h(foreign_lanes["lane_h"])
    if foreign_evidence_digests(foreign_lanes) & lock_identity_inventory(payload):
        raise ExperimentLockError("a foreign-lane identity contaminates a Lane-P identity")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _gb_energy(rows: Sequence[Mapping[str, Any]], dt_ms: float) -> float:
    fold = 0.0
    for row in rows:
        r_t = float(row[_GB_NET_L2_KEY])
        fold = fold + r_t * r_t
    return float(dt_ms) * fold


def _gb_half_life(rows: Sequence[Mapping[str, Any]], width: int) -> int | None:
    means: list[float] = []
    for bin_index in range(_GB_BINS):
        running = 0.0
        for row in rows[bin_index * width : (bin_index + 1) * width]:
            value = float(row[_GB_NET_L2_KEY])
            running = running + value * value
        means.append(running / float(width))
    if means[0] == 0.0:
        return 0
    threshold = 0.5 * means[0]
    for bin_index in range(_GB_BINS - 1):
        if means[bin_index] <= threshold and means[bin_index + 1] <= threshold:
            return bin_index * width
    return None


def _gb_bin_hamming(
    raster: Sequence[frozenset[int]], bin_high: int, bin_low: int, width: int
) -> int:
    total = 0
    for offset in range(width):
        high = raster[bin_high * width + offset]
        low = raster[bin_low * width + offset]
        total += len(high.symmetric_difference(low))
    return total


def _gb_tail_hamming(raster: Sequence[frozenset[int]], width: int) -> tuple[int, ...]:
    return tuple(
        _gb_bin_hamming(raster, tail_bin, tail_bin - lag, width)
        for tail_bin in _GB_TAIL_BINS
        for lag in _GB_CANDIDATE_LAGS
    )


def _gb_bin_current(rows: Sequence[Mapping[str, Any]], bin_index: int, width: int) -> list[float]:
    flat: list[float] = []
    for row in rows[bin_index * width : (bin_index + 1) * width]:
        for key in _GB_RECURRENT_KEYS:
            flat.append(float(row[key]))
    return flat


def _gb_l2(vector: Sequence[float]) -> float:
    total = 0.0
    for value in vector:
        total = total + value * value
    return math.sqrt(total)


def _gb_current_drift(high: Sequence[float], low: Sequence[float]) -> float:
    difference = _gb_l2([h - l for h, l in zip(high, low, strict=True)])
    denominator = max(_gb_l2(high), _gb_l2(low), _GB_EPSILON)
    return difference / denominator


def _gb_tail_silent(
    rows: Sequence[Mapping[str, Any]],
    raster: Sequence[frozenset[int]],
    width: int,
    numerical_zero_floor: float,
) -> bool:
    for step in range(_GB_TAIL_BINS[0] * width, _GB_BINS * width):
        if raster[step]:
            return False
        if float(rows[step][_GB_NET_L1_KEY]) > numerical_zero_floor:
            return False
    return True


def _gb_lag_passes(
    rows: Sequence[Mapping[str, Any]],
    raster: Sequence[frozenset[int]],
    width: int,
    n_neurons: int,
    lag: int,
    spike_drift_ceiling: float,
    current_drift_ceiling: float,
) -> bool:
    denominator = float(width * n_neurons)
    for tail_bin in _GB_TAIL_BINS:
        spike_drift = _gb_bin_hamming(raster, tail_bin, tail_bin - lag, width) / denominator
        current_drift = _gb_current_drift(
            _gb_bin_current(rows, tail_bin, width), _gb_bin_current(rows, tail_bin - lag, width)
        )
        if spike_drift > spike_drift_ceiling or current_drift > current_drift_ceiling:
            return False
    return True


def _gb_classify(
    rows: Sequence[Mapping[str, Any]],
    raster: Sequence[frozenset[int]],
    width: int,
    n_neurons: int,
    spike_drift_ceiling: float,
    current_drift_ceiling: float,
    numerical_zero_floor: float,
) -> str:
    if _gb_tail_silent(rows, raster, width, numerical_zero_floor):
        return "silent_decay"
    if _gb_lag_passes(
        rows, raster, width, n_neurons, 1, spike_drift_ceiling, current_drift_ceiling
    ):
        return "settled_fixed"
    for lag in (2, 3, 4):
        if _gb_lag_passes(
            rows, raster, width, n_neurons, lag, spike_drift_ceiling, current_drift_ceiling
        ):
            return "settled_periodic"
    return "wandering_active"


def _validate_gb_evidence_semantics(payload: Mapping[str, Any]) -> None:
    completion_steps = payload["completion_steps"]
    rows = payload["completion_rows"]
    raster_lists = payload["completion_spike_raster"]
    if len(rows) != completion_steps:
        raise ExperimentLockError("gb evidence completion rows differ from completion_steps")
    if len(raster_lists) != completion_steps:
        raise ExperimentLockError("gb evidence spike raster differs from completion_steps")
    for index, row in enumerate(rows):
        if index > 0 and row["timestep"] != rows[index - 1]["timestep"] + 1:
            raise ExperimentLockError(
                "gb evidence completion timesteps are not contiguous ascending"
            )
    n_neurons = payload["n_neurons"]
    raster: list[frozenset[int]] = []
    for neurons in raster_lists:
        if list(neurons) != sorted(neurons):
            raise ExperimentLockError("gb evidence spike raster is not ascending")
        if any(int(neuron) >= n_neurons for neuron in neurons):
            raise ExperimentLockError("gb evidence spike index exceeds the population")
        raster.append(frozenset(int(neuron) for neuron in neurons))
    for row in rows:
        value = float(row[_GB_NET_L2_KEY])
        if not math.isfinite(value) or value < 0.0:
            raise ExperimentLockError("gb evidence net-L2 must be finite and non-negative")
    dt_ms = float(payload["dt_ms"])
    if not math.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ExperimentLockError("gb evidence dt_ms must be a positive finite value")
    width = completion_steps // _GB_BINS
    spike_drift_ceiling = float(payload["spike_drift_ceiling"])
    current_drift_ceiling = float(payload["current_drift_ceiling"])
    numerical_zero_floor = float(payload["numerical_zero_floor"])
    energy = _gb_energy(rows, dt_ms)
    if not math.isfinite(energy) or energy < 0.0:
        raise ExperimentLockError("gb evidence recurrent energy is not finite non-negative")
    if float(payload["recurrent_energy"]) != energy:
        raise ExperimentLockError("gb evidence recurrent energy differs from the recomputed fold")
    if struct.pack(">d", energy).hex() != payload["recurrent_energy_bits"]:
        raise ExperimentLockError("gb evidence energy bits differ from the recomputed energy")
    if _gb_half_life(rows, width) != payload["half_life_steps"]:
        raise ExperimentLockError("gb evidence half-life differs from the recomputed decay")
    if list(_gb_tail_hamming(raster, width)) != list(payload["bin_spike_hamming"]):
        raise ExperimentLockError("gb evidence tail Hamming differs from the recomputed raster")
    trajectory = _gb_classify(
        rows,
        raster,
        width,
        n_neurons,
        spike_drift_ceiling,
        current_drift_ceiling,
        numerical_zero_floor,
    )
    if trajectory != payload["trajectory_class"]:
        raise ExperimentLockError(
            "gb evidence trajectory class differs from the recomputed dynamics"
        )
    settled = payload["settled"]
    if settled != (payload["trajectory_class"] in ("settled_fixed", "settled_periodic")):
        raise ExperimentLockError("gb evidence settled flag disagrees with the trajectory class")


def task_identity_digest(seed: int, lane_role: str, condition: str, cue_digest: str) -> str:
    """Bind a task ID to its full identity fields."""
    material = f"{seed}\0{lane_role}\0{condition}\0{cue_digest}"
    return _framed_digest(b"remanentia:snn-v2-task-id:v1\0", [material])


def _validate_completeness_semantics(payload: Mapping[str, Any]) -> None:
    expected = payload["expected_tasks"]
    expected_ids = [task["task_id"] for task in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ExperimentLockError("expected task IDs are not unique")
    lane_role = payload["lane_role"]
    if any(task["lane_role"] != lane_role for task in expected):
        raise ExperimentLockError("expected task lane role differs from the artifact lane")
    for task in expected:
        recomputed = task_identity_digest(
            task["seed"], task["lane_role"], task["condition"], task["cue_digest"]
        )
        if task["task_id"] != recomputed:
            raise ExperimentLockError("task ID does not bind its full task identity")
    if payload["task_set_digest"] != task_set_digest(expected_ids):
        raise ExperimentLockError("task-set digest does not bind the expected tasks")
    seeds = {task["seed"] for task in expected}
    if seeds - set(SEEDS):
        raise ExperimentLockError("expected task seed is outside the frozen seed set")
    completed = set(payload["completed_task_ids"])
    missing = set(payload["missing_task_ids"])
    unexpected = set(payload["unexpected_task_ids"])
    digest_failures = set(payload["digest_failure_task_ids"])
    expected_set = set(expected_ids)
    if missing != expected_set - completed:
        raise ExperimentLockError("missing task set is inconsistent with expected and completed")
    if unexpected & expected_set:
        raise ExperimentLockError("unexpected tasks overlap the expected set")
    if not digest_failures <= expected_set:
        raise ExperimentLockError("digest-failure tasks are not expected tasks")
    balanced = _per_seed_multiset_balanced(expected)
    process_records = payload["process_records"]
    record_task_ids = [record["task_id"] for record in process_records]
    process_instances = [record["process_instance"] for record in process_records]
    process_ok = (
        set(record_task_ids) == completed
        and len(record_task_ids) == len(completed)
        and len(set(process_instances)) == len(process_instances)
    )
    clean = (
        completed == expected_set
        and not missing
        and not unexpected
        and not digest_failures
        and balanced
        and process_ok
    )
    if bool(payload["completeness"]) != clean:
        raise ExperimentLockError("completeness flag disagrees with the task census")
    abort_reason = payload["abort_reason"]
    if clean:
        if abort_reason is not None:
            raise ExperimentLockError("a complete task census must not declare an abort reason")
    else:
        if abort_reason not in ABORT_REASONS:
            raise ExperimentLockError("an incomplete task census requires a known abort reason")


def _per_seed_multiset_balanced(expected: Sequence[Mapping[str, Any]]) -> bool:
    per_seed: dict[int, list[tuple[str, str, str]]] = {}
    for task in expected:
        key = (task["lane_role"], task["condition"], task["cue_digest"])
        per_seed.setdefault(task["seed"], []).append(key)
    if set(per_seed) != set(SEEDS):
        return False
    reference = sorted(per_seed[SEEDS[0]])
    return all(sorted(per_seed[seed]) == reference for seed in SEEDS)


_SEMANTIC_VALIDATORS: Mapping[str, Callable[[Mapping[str, Any]], None]] = MappingProxyType(
    {
        "snn-memory-scoring-target-v2": _validate_scoring_target_semantics,
        "snn-memory-gb-calibration-spec-v2": _validate_calibration_spec_semantics,
        "snn-memory-experiment-lock-v2": _validate_experiment_lock_semantics,
        "snn-memory-task-completeness-v2": _validate_completeness_semantics,
        "snn-memory-gb-preflight-evidence-v2": _validate_gb_evidence_semantics,
    }
)


def _framed_digest(domain: bytes, values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def task_set_digest(task_ids: Sequence[str]) -> str:
    """Hash the ordered expected-task identity set."""
    return _framed_digest(b"remanentia:snn-v2-task-set:v1\0", list(task_ids))


def canonical_config_digest(value: Mapping[str, Any]) -> str:
    """Hash the canonical-JSON representation of a configuration mapping."""
    raw = json.dumps(
        _thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(b"remanentia:snn-v2-config:v1\0" + raw).hexdigest()


def candidate_order_digest(candidate_order: Sequence[str]) -> str:
    """Hash the ordered candidate-record identity list."""
    return _framed_digest(b"remanentia:snn-v2-candidate-order:v1\0", list(candidate_order))


def candidate_bank_digest(candidate_order: Sequence[str], signature_digests: Sequence[str]) -> str:
    """Hash the ordered candidate bank as interleaved record identity/signature pairs."""
    interleaved: list[str] = []
    for record_id, signature in zip(candidate_order, signature_digests, strict=True):
        interleaved.append(record_id)
        interleaved.append(signature)
    return _framed_digest(b"remanentia:snn-v2-candidate-bank:v1\0", interleaved)


def lane_role_digest(lane_role: str) -> str:
    """Hash a single lane-role token."""
    return _framed_digest(b"remanentia:snn-v2-lane-role:v1\0", [lane_role])


def lane_identity_fields(payload: Mapping[str, Any]) -> list[str]:
    """Return the full ordered lane-identity block bound into a lock's lane domain.

    The block spans every load-bearing lane identity — root, task set, threshold,
    candidate bank, scorer, schema, and provenance. A change to any one of them
    changes the lane-domain digest, so a lock cannot silently adopt a foreign
    lane's output root, task set, calibration thresholds, candidate bank, or scorer.
    """
    identity = payload["lane_identity"]
    return [
        identity["root"],
        identity["task_set"],
        identity["threshold"],
        identity["candidate_bank"],
        identity["scorer"],
        identity["schema"],
        identity["provenance"],
    ]


def lane_domain_digest(lane_role: str, lane_fields: Sequence[str]) -> str:
    """Hash a lane role bound to its full lane-identity block."""
    return _framed_digest(b"remanentia:snn-v2-lane-domain:v1\0", [lane_role, *lane_fields])


def calibration_threshold_digest(payload: Mapping[str, Any]) -> str:
    """Hash the frozen calibration threshold grids, tail bins, and candidate lags."""
    grids: dict[str, Any] = {
        key: list(payload[key])
        for key in (
            "numerical_zero_floor_grid",
            "spike_drift_ceiling_grid",
            "current_drift_ceiling_grid",
            "representation_margin_floor_grid",
            "normalized_effective_rank_floor_grid",
            "settled_fraction_floor_grid",
            "wandering_ceiling_grid",
            "collapse_ceiling_grid",
            "abstention_grid",
        )
    }
    grids["tail_bins"] = list(payload["tail_bins"])
    grids["candidate_lags"] = list(payload["candidate_lags"])
    return canonical_config_digest(grids)


# Foreign-lane (confound Lane-C, heldout Lane-H) evidence embedded and bound inside the Lane-P
# experiment lock before sealing. Every identity-bearing manifest component is a concrete measured
# digest — a schema-file digest, output-root/task-set/threshold/candidate-bank/scorer/provenance
# digest, or (Lane H) a Git-object ID and historical embedding/configuration/environment/ranking/
# root/provenance digest — never a free-form label. The eventual foreign-lane run must reproduce
# those exact digests, and both the derived per-field identity AND the raw component digest carried
# by the manifest are banned from Lane P, so an actual Lane-C/Lane-H artefact digest cannot slip in.
_LANE_C_MANIFEST_DOMAIN = b"remanentia:snn-v2-lane-c-diagnostic:v1\0"
_LANE_H_MANIFEST_DOMAIN = b"remanentia:snn-v2-lane-h-historical:v1\0"
_FOREIGN_LANE_C_DOMAIN = b"remanentia:snn-v2-foreign-lane-c:v1\0"
_FOREIGN_LANE_H_DOMAIN = b"remanentia:snn-v2-foreign-lane-h:v1\0"
_LANE_H_MISSING_STATUS = "inadmissible_missing_historical_evidence"
_LANE_H_ADMISSIBLE_STATUS = "admissible_authenticated_historical_evidence"
_LANE_C_COMPONENT_KEYS: tuple[str, ...] = (
    "diagnostic_schema_sha256",
    "experiment_config_sha256",
    "output_root_sha256",
    "task_set_sha256",
    "threshold_sha256",
    "candidate_bank_sha256",
    "scorer_sha256",
    "provenance_sha256",
)
_LANE_H_COMPONENT_KEYS: tuple[str, ...] = (
    "historical_git_object",
    "historical_embedding_sha256",
    "historical_configuration_sha256",
    "historical_dependency_environment_sha256",
    "historical_ranking_output_sha256",
    "historical_output_root_sha256",
    "historical_provenance_sha256",
)


def lane_c_manifest_identity(manifest: Mapping[str, str]) -> dict[str, str]:
    """Derive the seven-field Lane-C identity from its measured diagnostic manifest digests."""

    def field(*values: str) -> str:
        return _framed_digest(_LANE_C_MANIFEST_DOMAIN, list(values))

    return {
        "root": field("output_root", manifest["output_root_sha256"]),
        "task_set": field("task_set", manifest["task_set_sha256"]),
        "threshold": field("threshold", manifest["threshold_sha256"]),
        "candidate_bank": field("candidate_bank", manifest["candidate_bank_sha256"]),
        "scorer": field("scorer", manifest["scorer_sha256"]),
        "schema": field("diagnostic_schema", manifest["diagnostic_schema_sha256"]),
        "provenance": field(
            "provenance",
            manifest["provenance_sha256"],
            "experiment_config",
            manifest["experiment_config_sha256"],
        ),
    }


def lane_h_manifest_identity(manifest: Mapping[str, str]) -> dict[str, str]:
    """Derive the seven-field Lane-H identity from an authenticated historical manifest's digests."""

    def field(*values: str) -> str:
        return _framed_digest(_LANE_H_MANIFEST_DOMAIN, list(values))

    return {
        "root": field("historical_output_root", manifest["historical_output_root_sha256"]),
        "task_set": field(
            "historical_ranking_output", manifest["historical_ranking_output_sha256"]
        ),
        "threshold": field("historical_configuration", manifest["historical_configuration_sha256"]),
        "candidate_bank": field("historical_embedding", manifest["historical_embedding_sha256"]),
        "scorer": field("historical_git_object", manifest["historical_git_object"]),
        "schema": field(
            "historical_dependency_environment",
            manifest["historical_dependency_environment_sha256"],
        ),
        "provenance": field("historical_provenance", manifest["historical_provenance_sha256"]),
    }


def lane_c_raw_components(manifest: Mapping[str, str]) -> set[str]:
    """Return the raw measured Lane-C component digests carried by the manifest."""
    return {str(manifest[key]) for key in _LANE_C_COMPONENT_KEYS}


def lane_h_raw_components(manifest: Mapping[str, str]) -> set[str]:
    """Return the raw measured Lane-H component digests carried by the manifest."""
    return {str(manifest[key]) for key in _LANE_H_COMPONENT_KEYS}


def build_foreign_lane_c(manifest: Mapping[str, str]) -> dict[str, Any]:
    """Build a bound Lane-C foreign-evidence block from a concrete diagnostic manifest."""
    block: dict[str, Any] = {
        "lane_role": "lane_c",
        "manifest": dict(manifest),
        "identity": lane_c_manifest_identity(manifest),
    }
    block["self_sha256"] = _self_digest(block, _FOREIGN_LANE_C_DOMAIN)
    return block


def build_foreign_lane_h_inadmissible(
    reason: str, missing_inventory: Sequence[str]
) -> dict[str, Any]:
    """Build a bound inadmissible Lane-H record — an explicit absence, no fabricated identities."""
    block: dict[str, Any] = {
        "lane_role": "lane_h",
        "status": _LANE_H_MISSING_STATUS,
        "reason": reason,
        "missing_inventory": list(missing_inventory),
    }
    block["self_sha256"] = _self_digest(block, _FOREIGN_LANE_H_DOMAIN)
    return block


def build_foreign_lane_h_admissible(manifest: Mapping[str, str]) -> dict[str, Any]:
    """Build a bound admissible Lane-H record from an authenticated historical manifest."""
    block: dict[str, Any] = {
        "lane_role": "lane_h",
        "status": _LANE_H_ADMISSIBLE_STATUS,
        "manifest": dict(manifest),
        "identity": lane_h_manifest_identity(manifest),
    }
    block["self_sha256"] = _self_digest(block, _FOREIGN_LANE_H_DOMAIN)
    return block


def _validate_foreign_lane_c(block: Mapping[str, Any]) -> None:
    if block["lane_role"] != "lane_c":
        raise ExperimentLockError("foreign Lane-C evidence has the wrong lane role")
    if _thaw(block["identity"]) != lane_c_manifest_identity(block["manifest"]):
        raise ExperimentLockError("foreign Lane-C identity is not derived from its manifest")
    if block["self_sha256"] != _self_digest(block, _FOREIGN_LANE_C_DOMAIN):
        raise ExperimentLockError("foreign Lane-C evidence self digest mismatch")


def _validate_foreign_lane_h(block: Mapping[str, Any]) -> None:
    if block["lane_role"] != "lane_h":
        raise ExperimentLockError("foreign Lane-H evidence has the wrong lane role")
    if block["status"] == _LANE_H_ADMISSIBLE_STATUS:
        if _thaw(block["identity"]) != lane_h_manifest_identity(block["manifest"]):
            raise ExperimentLockError("admissible Lane-H identity is not derived from its manifest")
    if block["self_sha256"] != _self_digest(block, _FOREIGN_LANE_H_DOMAIN):
        raise ExperimentLockError("foreign Lane-H evidence self digest mismatch")


def foreign_evidence_digests(foreign_lanes: Mapping[str, Any]) -> frozenset[str]:
    """Derive the foreign identity set from the bound Lane-C and Lane-H evidence.

    The set carries both the derived per-field identities and the raw measured component
    digests actually held by each manifest, so a real Lane-C/Lane-H artefact digest is
    banned from Lane P even before it is domain-folded. An inadmissible Lane-H record
    contributes no identity — it is an authenticated absence.
    """
    lane_c = foreign_lanes["lane_c"]
    values: set[str] = set(_thaw(lane_c["identity"]).values())
    values.update(lane_c_raw_components(lane_c["manifest"]))
    lane_h = foreign_lanes["lane_h"]
    if lane_h["status"] == _LANE_H_ADMISSIBLE_STATUS:
        values.update(_thaw(lane_h["identity"]).values())
        values.update(lane_h_raw_components(lane_h["manifest"]))
    return frozenset(values)


def lock_identity_inventory(payload: Mapping[str, Any]) -> frozenset[str]:
    """Return the complete explicit Lane-P identity inventory carried by the lock payload."""
    inventory: set[str] = set(lane_identity_fields(payload))
    inventory.update(
        [
            payload["candidate_order_digest"],
            payload["expected_task_set_digest"],
            payload["scoring_target_digest"],
            payload["calibration_spec_digest"],
            payload["output_root_digest"],
            payload["lane_role_digest"],
            payload["lane_domain_digest"],
        ]
    )
    inventory.update(_thaw(payload["identities"]).values())
    d1 = payload["d1"]
    inventory.update([d1["file_sha256"], d1["payload_self_sha256"], d1["repository_head"]])
    inventory.update(_thaw(d1["selected_record_ids"]))
    d2 = payload["d2"]
    inventory.update(
        [
            d2["cue_set_file_sha256"],
            d2["cue_set_payload_self_sha256"],
            d2["bundle_inventory_digest"],
        ]
    )
    return frozenset(inventory)


def scorer_identity_digest(payload: Mapping[str, Any]) -> str:
    """Hash the frozen scorer-defining fields of a scoring target."""
    scorer = {
        "similarity": payload["similarity"],
        "zero_norm_rule": payload["zero_norm_rule"],
        "score_order": payload["score_order"],
        "tie_rule": payload["tie_rule"],
        "abstention": _thaw(payload["abstention"]),
        "correctness_rule": payload["correctness_rule"],
        "top_k": payload["top_k"],
        "signature_dtype": payload["signature_dtype"],
        "bins": payload["bins"],
        "completion_steps": payload["completion_steps"],
    }
    return canonical_config_digest(scorer)


def synthetic_positive_ids() -> list[str]:
    """Return the canonical synthetic-g1-v2 positive development memory IDs (16 per seed)."""
    return [
        f"dev:s{seed}-p{index:02d}"
        for seed in SEEDS
        for index in range(_SYNTHETIC_POSITIVES_PER_SEED)
    ]


def synthetic_negative_ids(positive_ids: Sequence[str]) -> list[str]:
    """Return the matched four-family negative IDs for every positive memory."""
    return [f"{positive}-{code}" for positive in positive_ids for code in FAMILY_CODES.values()]


def synthetic_config_digest() -> str:
    """Hash the frozen synthetic-generator configuration."""
    return canonical_config_digest(
        {
            "identity": "synthetic-g1-v2",
            "namespace": "dev:",
            "per_seed_positive_count": _SYNTHETIC_POSITIVES_PER_SEED,
            "negative_families": list(FAMILY_CODES.keys()),
            "seeds": list(SEEDS),
        }
    )


def synthetic_partition_digests(positive_ids: Sequence[str]) -> tuple[str, str]:
    """Split each seed's positives into a frozen selection/validation partition and hash both."""
    per_seed: dict[str, list[str]] = {}
    for identity in positive_ids:
        per_seed.setdefault(identity.split("-", 1)[0], []).append(identity)
    selection: list[str] = []
    validation: list[str] = []
    for seed_key in sorted(per_seed):
        ordered = sorted(per_seed[seed_key])
        selection.extend(ordered[:_SYNTHETIC_SELECTION_PER_SEED])
        validation.extend(ordered[_SYNTHETIC_SELECTION_PER_SEED:])
    return (
        _framed_digest(b"remanentia:snn-v2-selection-partition:v1\0", selection),
        _framed_digest(b"remanentia:snn-v2-validation-partition:v1\0", validation),
    )


def synthetic_task_set_digest(positive_ids: Sequence[str], negative_ids: Sequence[str]) -> str:
    """Hash the full ordered synthetic development task inventory."""
    inventory = sorted(positive_ids) + sorted(negative_ids)
    return _framed_digest(b"remanentia:snn-v2-synthetic-task-set:v1\0", inventory)


def cue_bundle_inventory(cue_set_payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Enumerate every declared (path, sha256) cue bundle from the authenticated cue set."""
    inventory: list[tuple[str, str]] = []
    for record in cue_set_payload["records"]:
        for key, bases in record.items():
            if not (isinstance(key, str) and key.endswith("_base_cues")):
                continue
            for base in bases:
                for variant in base["variants"]:
                    bundle = variant["bundle"]
                    inventory.append((str(bundle["path"]), str(bundle["sha256"])))
    inventory.sort()
    return inventory


def cue_bundle_inventory_digest(cue_set_payload: Mapping[str, Any]) -> str:
    """Hash the ordered cue-bundle inventory referenced by the authenticated cue set."""
    flat: list[str] = []
    for path, digest in cue_bundle_inventory(cue_set_payload):
        flat.append(path)
        flat.append(digest)
    return _framed_digest(b"remanentia:snn-v2-cue-bundle-inventory:v1\0", flat)


def _bundle_identity_digests(bundle: Any) -> set[str]:
    """Derive every identity class carried by one authenticated cue bundle.

    Spans the bundle file/self digests, its cue ID, raw/normalised text digests, an explicit
    domain-separated digest of the decoded float64 embedding payload, the encoder/model
    identities, and the three implementation/provenance logical paths and digests.
    """
    payload = bundle.payload
    raw_embedding = base64.b64decode(payload["embedding"]["data_base64"])
    embedding_digest = hashlib.sha256(
        b"remanentia:snn-v2-embedding:v1\0" + raw_embedding
    ).hexdigest()
    digests: set[str] = {
        bundle.file_sha256,
        str(bundle.payload_self_sha256),
        str(payload["cue_id"]),
        str(payload["text_sha256"]),
        str(payload["normalized_text_sha256"]),
        str(payload["self_sha256"]),
        embedding_digest,
        str(payload["encoder"]["identity"]),
        str(payload["encoder"]["directory_sha256"]),
        str(payload["encoder"]["config_digest"]),
        str(payload["model"]["config_digest"]),
    }
    for role in ("cue_materializer", "split_events", "sentence_encoder"):
        implementation = payload["implementations"][role]
        digests.add(str(implementation["logical_path"]))
        digests.add(str(implementation["sha256"]))
    return digests


def bind_lane_isolation(
    lock: ValidatedArtifact,
    scoring_target: ValidatedArtifact,
    calibration_spec: ValidatedArtifact,
    d1_artifact: Any,
    cue_set: Any,
    bundles: Sequence[Any],
) -> None:
    """Require a Lane-P lock, its scoring target and calibration to share one lane.

    The authenticated D1 source universe, D2 cue set, and the **complete** authenticated cue
    bundle sequence are mandatory — there is no caller-omissible trust surface. The bundle set
    is verified exactly against the cue set's declared inventory (no omission, duplicate,
    unexpected, or substituted bundle), and every bundle identity is derived internally here.
    The foreign Lane-C/Lane-H identity set (derived only from the evidence bound inside the
    Lane-P lock) is then required disjoint from every explicit Lane-P identity: the lock's own
    inventory plus the scoring-target, calibration-spec, D1, D2, and every bundle identity.

    Raises
    ------
    ExperimentLockError
        If a foreign-lane identity appears in any bound Lane-P identity, the bound
        scoring/calibration digests or identities disagree with the lock, or the bundle set
        does not exactly reconstruct the authenticated cue-set inventory.
    """
    if (
        lock.artifact_type != "snn-memory-experiment-lock-v2"
        or scoring_target.artifact_type != "snn-memory-scoring-target-v2"
        or calibration_spec.artifact_type != "snn-memory-gb-calibration-spec-v2"
    ):
        raise ExperimentLockError(
            "lane isolation requires the exact lock/scoring/calibration types"
        )
    lane = lock.payload["lane_role"]
    if lane != LaneRole.LANE_P.value:
        raise ExperimentLockError("only a Lane-P lock is admissible for evaluation")
    if scoring_target.payload["lane_role"] != lane:
        raise ExperimentLockError("scoring-target lane role contaminates the Lane-P lock")
    if calibration_spec.payload["lane_role"] != lane:
        raise ExperimentLockError("calibration-spec lane role contaminates the Lane-P lock")
    if lock.payload["scoring_target_digest"] != scoring_target.payload_self_sha256:
        raise ExperimentLockError("lock scoring-target digest does not bind the scoring target")
    if lock.payload["calibration_spec_digest"] != calibration_spec.payload_self_sha256:
        raise ExperimentLockError("lock calibration digest does not bind the calibration spec")
    # The scoring target and calibration spec must share the lock's exact toolchain
    # identity; a mutually inconsistent installed/schema/wheel/backend set is rejected.
    lock_identities = lock.payload["identities"]
    for key in _SHARED_IDENTITY_KEYS:
        if scoring_target.payload["identities"][key] != lock_identities[key]:
            raise ExperimentLockError(
                "scoring-target identity differs from the lock identity block"
            )
        if calibration_spec.payload["identities"][key] != lock_identities[key]:
            raise ExperimentLockError("calibration identity differs from the lock identity block")
    # The lock's granular lane candidate-bank and threshold identities must bind the exact
    # scoring candidate bank and calibration threshold grids.
    lane_identity = lock.payload["lane_identity"]
    if (
        lane_identity["candidate_bank"]
        != scoring_target.payload["identities"]["candidate_bank_digest"]
    ):
        raise ExperimentLockError(
            "lock candidate-bank identity does not bind the scoring candidate bank"
        )
    if lane_identity["threshold"] != calibration_threshold_digest(calibration_spec.payload):
        raise ExperimentLockError(
            "lock threshold identity does not bind the calibration thresholds"
        )
    # The candidate bank order is D1/D2 provenance; scoring requires lexical order.
    if tuple(scoring_target.payload["candidate_order"]) != tuple(
        sorted(lock.payload["candidate_order"])
    ):
        raise ExperimentLockError(
            "scoring candidate order must be the lexical Lane-P candidate set"
        )
    # The presented bundle set must exactly reconstruct the cue set's declared bundle inventory:
    # no omission, duplicate, unexpected, or substituted bundle, and no caller-selected subset.
    declared = sorted(set(cue_bundle_inventory(cue_set.payload)))
    if not declared:
        raise ExperimentLockError("the authenticated cue set declares no cue bundles")
    if sorted(bundle.file_sha256 for bundle in bundles) != sorted(sha for _, sha in declared):
        raise ExperimentLockError(
            "bound bundles do not reconstruct the authenticated cue-set inventory"
        )
    # The bound foreign-lane identity set must be disjoint from every presented Lane-P artifact
    # identity class (scoring/calibration toolchain identities, D1/D2 files, every bundle identity).
    foreign = foreign_evidence_digests(lock.payload["foreign_lanes"])
    artifact_inventory: set[str] = set(_thaw(scoring_target.payload["identities"]).values())
    artifact_inventory.update(_thaw(calibration_spec.payload["identities"]).values())
    artifact_inventory.update([d1_artifact.file_sha256, d1_artifact.payload_self_sha256])
    artifact_inventory.update(_thaw(d1_artifact.payload["selected_record_ids"]))
    artifact_inventory.update([cue_set.file_sha256, cue_set.payload_self_sha256])
    for bundle in bundles:
        artifact_inventory.update(_bundle_identity_digests(bundle))
    if foreign & artifact_inventory:
        raise ExperimentLockError(
            "a foreign-lane identity contaminates a bound Lane-P artifact identity"
        )


def bind_task_completeness(lock: ValidatedArtifact, completeness: ValidatedArtifact) -> None:
    """Require an authenticated completeness artifact to belong to an authenticated lock.

    Raises
    ------
    ExperimentLockError
        If the artifact types are wrong, or the completeness lane, seed set, or
        expected task-set digest disagrees with the lock. The task-to-process 1:1
        association and global process non-reuse are enforced by the completeness
        semantic validator that already ran when the artifact was authenticated.
    """
    if (
        lock.artifact_type != "snn-memory-experiment-lock-v2"
        or completeness.artifact_type != "snn-memory-task-completeness-v2"
    ):
        raise ExperimentLockError("completeness binding requires the exact lock/completeness types")
    if completeness.payload["lane_role"] != lock.payload["lane_role"]:
        raise ExperimentLockError("completeness lane role differs from the lock lane")
    if tuple(completeness.payload["seeds"]) != tuple(lock.payload["seeds"]):
        raise ExperimentLockError("completeness seed set differs from the lock seed set")
    if completeness.payload["task_set_digest"] != lock.payload["expected_task_set_digest"]:
        raise ExperimentLockError(
            "completeness task-set digest differs from the lock expected task set"
        )


def bind_d1_d2(
    lock: ValidatedArtifact,
    d1_artifact: Any,
    cue_set: Any,
) -> None:
    """Require the lock's D1/D2 bindings to match the authenticated D1/D2 artifacts.

    Parameters
    ----------
    lock
        The experiment-lock artifact carrying the ``d1``/``d2`` bindings.
    d1_artifact
        The mandatory authenticated D1 source-universe artifact; its file and self
        digests, HEAD, and ordered record IDs must equal the lock's ``d1``.
    cue_set
        The mandatory authenticated D2 cue-set artifact; its file and self digests
        and nested source-universe binding must equal the lock's ``d2``.

    Raises
    ------
    ExperimentLockError
        On any disagreement between the lock's bindings and the authenticated
        artifacts, or between the lock's ``d2`` source-universe and its ``d1``.
    """
    d1 = lock.payload["d1"]
    d2 = lock.payload["d2"]
    if _thaw(d2["source_universe"]) != _thaw(d1):
        raise ExperimentLockError("D2 source-universe binding does not bind the supplied D1")
    if lock.payload["identities"]["repository_head"] != d1["repository_head"]:
        raise ExperimentLockError("lock repository HEAD differs from the D1 binding")
    if tuple(lock.payload["candidate_order"]) != tuple(d1["selected_record_ids"]):
        raise ExperimentLockError("candidate order differs from the D1 selected record IDs")
    payload = d1_artifact.payload
    if (
        d1_artifact.file_sha256 != d1["file_sha256"]
        or d1_artifact.payload_self_sha256 != d1["payload_self_sha256"]
        or payload["repository"]["head"] != d1["repository_head"]
        or tuple(payload["selected_record_ids"]) != tuple(d1["selected_record_ids"])
    ):
        raise ExperimentLockError("lock D1 binding differs from the authenticated D1 artifact")
    payload = cue_set.payload
    source = payload["source_universe"]
    if (
        cue_set.file_sha256 != d2["cue_set_file_sha256"]
        or cue_set.payload_self_sha256 != d2["cue_set_payload_self_sha256"]
        or d2["bundle_inventory_digest"] != cue_bundle_inventory_digest(payload)
        or source["file_sha256"] != d1["file_sha256"]
        or source["payload_self_sha256"] != d1["payload_self_sha256"]
        or source["repository_head"] != d1["repository_head"]
        or tuple(source["selected_record_ids"]) != tuple(d1["selected_record_ids"])
    ):
        raise ExperimentLockError("lock D2 binding differs from the authenticated cue set")


def require_dev_namespace_disjoint(
    dev_ids: Sequence[str],
    record_ids: Sequence[str],
    locked_identities: Sequence[str] = (),
) -> None:
    """Reject any synthetic development ID that collides with a locked D2 identity.

    Parameters
    ----------
    dev_ids
        Synthetic development identities; each must use the ``dev:`` namespace.
    record_ids
        Immutable ``sha256:`` evaluation record identities.
    locked_identities
        Additional locked D2 identity strings — cue-byte/bundle digests, source
        paths, and evaluation identities — whose bodies must not alias a dev ID.

    Raises
    ------
    ExperimentLockError
        If a development ID leaves the ``dev:`` namespace or its identity or body
        collides with an immutable record or any locked D2 identity.
    """
    for dev_id in dev_ids:
        if not dev_id.startswith("dev:"):
            raise ExperimentLockError("development IDs must use the dev: namespace")
    record_bodies = {record_id.removeprefix("sha256:") for record_id in record_ids}
    dev_full = set(dev_ids)
    dev_bodies = {dev_id.removeprefix("dev:") for dev_id in dev_ids}
    if dev_full & set(record_ids) or dev_bodies & record_bodies:
        raise ExperimentLockError("development and immutable record identities overlap")
    locked_full = set(locked_identities)
    locked_bodies = {identity.split(":", 1)[-1] for identity in locked_identities}
    if (dev_full & locked_full) or (dev_bodies & locked_full) or (dev_bodies & locked_bodies):
        raise ExperimentLockError("development identity overlaps a locked D2 identity")


def _load_json_file(path: Path, context: str) -> dict[str, Any]:
    return _strict_json(_read_regular_bytes(path.absolute(), context), context)


def _cmd_write(arguments: argparse.Namespace) -> dict[str, Any]:
    payload = _load_json_file(arguments.payload, "payload")
    artifact = write_artifact(payload, arguments.output)
    return {
        "artifact_type": artifact.artifact_type,
        "file_sha256": artifact.file_sha256,
        "payload_self_sha256": artifact.payload_self_sha256,
    }


def _cmd_read(arguments: argparse.Namespace) -> dict[str, Any]:
    artifact = read_artifact(
        arguments.artifact,
        expected_type=arguments.expected_type,
        expected_file_sha256=arguments.expected_sha256,
    )
    return {
        "artifact_type": artifact.artifact_type,
        "file_sha256": artifact.file_sha256,
        "payload_self_sha256": artifact.payload_self_sha256,
    }


def _cmd_completeness(arguments: argparse.Namespace) -> dict[str, Any]:
    artifact = read_artifact(arguments.artifact, expected_type="snn-memory-task-completeness-v2")
    return {
        "completeness": bool(artifact.payload["completeness"]),
        "abort_reason": artifact.payload["abort_reason"],
        "file_sha256": artifact.file_sha256,
    }


def _load_bind_inputs(
    lock: ValidatedArtifact, d1_path: Path, cue_set_path: Path, bundle_root: Path
) -> tuple[Any, Any, list[Any]]:
    """Authenticate the mandatory D1, D2 cue set, and every declared bundle for a CLI bind.

    Any D1/D2/bundle authentication failure — a missing, symlinked, or hash-mismatched artefact —
    is surfaced as an :class:`ExperimentLockError`, so a bind that cannot authenticate every
    required input fails rather than reporting ``bound: true``.
    """
    from snn_memory.cue_materializer import CueMaterializerError, read_cue_bundle, read_cue_set
    from snn_memory.source_universe import SourceUniverseError, validate_source_universe_bytes

    try:
        d1_raw = _read_regular_bytes(d1_path, "D1 source-universe file")
        d1_artifact = validate_source_universe_bytes(
            d1_raw, expected_file_sha256=lock.payload["d1"]["file_sha256"]
        )
        cue_set = read_cue_set(cue_set_path, lock.payload["d2"]["cue_set_file_sha256"])
        bundles = [
            read_cue_bundle(bundle_root / path, sha256)
            for path, sha256 in sorted(set(cue_bundle_inventory(cue_set.payload)))
        ]
    except (SourceUniverseError, CueMaterializerError) as error:
        raise ExperimentLockError(
            f"bind could not authenticate a required D1/D2/bundle input: {error}"
        ) from error
    return d1_artifact, cue_set, bundles


def _cmd_bind(arguments: argparse.Namespace) -> dict[str, Any]:
    lock = read_artifact(arguments.lock, expected_type="snn-memory-experiment-lock-v2")
    scoring = read_artifact(arguments.scoring_target, expected_type="snn-memory-scoring-target-v2")
    calibration = read_artifact(
        arguments.calibration_spec, expected_type="snn-memory-gb-calibration-spec-v2"
    )
    d1_artifact, cue_set, bundles = _load_bind_inputs(
        lock, arguments.d1, arguments.cue_set, arguments.bundle_root
    )
    bind_lane_isolation(lock, scoring, calibration, d1_artifact, cue_set, bundles)
    bind_d1_d2(lock, d1_artifact, cue_set)
    return {"bound": True, "lock_self_sha256": lock.payload_self_sha256}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment-lock CLI and return a process exit code."""
    parser = argparse.ArgumentParser(prog="python -m snn_memory.experiment_lock")
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--payload", type=Path, required=True)
    write.add_argument("--output", type=Path, required=True)
    write.set_defaults(handler=_cmd_write)
    read = subparsers.add_parser("read")
    read.add_argument("--artifact", type=Path, required=True)
    read.add_argument("--expected-type")
    read.add_argument("--expected-sha256")
    read.set_defaults(handler=_cmd_read)
    completeness = subparsers.add_parser("completeness")
    completeness.add_argument("--artifact", type=Path, required=True)
    completeness.set_defaults(handler=_cmd_completeness)
    bind = subparsers.add_parser("bind")
    bind.add_argument("--lock", type=Path, required=True)
    bind.add_argument("--scoring-target", type=Path, required=True)
    bind.add_argument("--calibration-spec", type=Path, required=True)
    bind.add_argument("--d1", type=Path, required=True)
    bind.add_argument("--cue-set", type=Path, required=True)
    bind.add_argument("--bundle-root", type=Path, required=True)
    bind.set_defaults(handler=_cmd_bind)
    arguments = parser.parse_args(argv)
    try:
        report = arguments.handler(arguments)
    except (OSError, ExperimentLockError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
