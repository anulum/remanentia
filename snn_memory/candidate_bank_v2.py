# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN schema-v2 candidate-signature bank contract

"""Completion-only neuron-major temporal-signature bank writer and reader.

Each record signature is the frozen ``snn-temporal-signature-v2``: the exact
completion-only binary spike raster, shape ``(completion_steps, n_neurons)`` with
``completion_steps >= 32`` and divisible by eight, folded into eight equal
contiguous time bins whose per-neuron per-bin value is the ordered binary64
left-fold sum of that neuron's 0/1 spikes, flattened neuron-major to width
``8 * n_neurons`` and encoded canonical little-endian ``<f8``. The bank binds its
checkpoint, the D3 scoring target, calibration, and build, and re-derives every
digest on read from the captured bytes.
"""

from __future__ import annotations

import ctypes
import errno
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

from snn_memory.checkpoint_bundle_v2 import (
    _canonical_bytes,
    _framed,
    _le,
    _self_digest,
    _sha,
    ordered_record_ids_digest,
)
from snn_memory.experiment_lock import candidate_bank_digest

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_BANK_SCHEMA = "snn_memory_candidate_bank_v2.schema.json"
_SIGNATURE_ROW_DOMAIN = b"remanentia:snn-temporal-signature-v2:row\0"
_DECODED_DOMAIN = b"remanentia:snn-temporal-signature-v2:decoded\0"
_BANK_DOMAIN = b"remanentia:snn-memory:candidate-bank:v2\0"
_SIGNATURES_PATH = "signatures.bin"
_BANK_PATH = "bank.json"
_SEMANTIC_VERSION = "snn-temporal-signature-v2"
_BINS = 8


class CandidateBankError(ValueError):
    """Raised when a candidate bank fails a contract, byte, or digest check."""


@dataclass(frozen=True)
class ValidatedCandidateBank:
    """An authenticated candidate bank with a read-only manifest and signatures."""

    manifest: Mapping[str, Any]
    self_sha256: str
    signatures: FloatArray
    row_digests: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateBankError(message)


def _lexical_candidates(
    ids: Sequence[str], row_digests: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Reorder (record id, signature digest) pairs into the lexical record-id order scoring binds."""
    order = sorted(range(len(ids)), key=lambda index: ids[index])
    return [ids[index] for index in order], [row_digests[index] for index in order]


def _require_canonical_signatures(
    signatures: FloatArray, rows: int, width: int, source: str
) -> None:
    """Reject a signature block that is not a canonical finite float64 array of the exact shape."""
    _require_canonical(signatures, "<f8", f"{source} signatures")
    if signatures.ndim != 2 or signatures.shape != (rows, width):
        raise CandidateBankError(f"{source} signatures shape differs from the ordered record bank")


def _require_canonical(array: FloatArray | BoolArray, dtype: str, label: str) -> None:
    if array.dtype != np.dtype(dtype):
        raise CandidateBankError(f"{label} is not a canonical {dtype} array")
    if not array.flags["C_CONTIGUOUS"]:
        raise CandidateBankError(f"{label} is not C-contiguous")
    if not bool(np.isfinite(array).all()):
        raise CandidateBankError(f"{label} carries a non-finite value")


def temporal_signature_v2(raster: BoolArray, completion_steps: int, n_neurons: int) -> FloatArray:
    """Fold one completion raster into the neuron-major eight-bin float64 signature."""
    _require(
        completion_steps >= 32 and completion_steps % _BINS == 0,
        "completion window must be at least 32 steps and divisible by eight",
    )
    _require_canonical(raster, "|b1", "completion raster")
    binary = _le(raster, "|b1")
    _require(binary.shape == (completion_steps, n_neurons), "completion raster shape mismatch")
    per_bin = completion_steps // _BINS
    row = np.empty(_BINS * n_neurons, dtype="<f8")
    for neuron in range(n_neurons):
        column = binary[:, neuron]
        for bin_index in range(_BINS):
            accumulator = 0.0
            for step in range(bin_index * per_bin, (bin_index + 1) * per_bin):
                accumulator += 1.0 if bool(column[step]) else 0.0
            row[neuron * _BINS + bin_index] = accumulator
    return row


def signature_row_digest(record_id: str, row: FloatArray) -> str:
    """Domain-separated digest of one ordered candidate's signature row."""
    if not isinstance(record_id, str):
        raise CandidateBankError("signature row record identifier is not a string")
    _require_canonical(row, "<f8", "signature row")
    return _framed(_SIGNATURE_ROW_DOMAIN, [record_id.encode("utf-8"), _le(row, "<f8").tobytes()])


def decoded_float64_digest(signatures: FloatArray) -> str:
    """Domain-separated digest of the decoded row-major float64 signature payload."""
    _require_canonical(signatures, "<f8", "signature payload")
    canonical = _le(signatures, "<f8")
    shape = b"".join(int(dim).to_bytes(8, "big") for dim in canonical.shape)
    return _sha(_DECODED_DOMAIN + shape + canonical.tobytes(order="C"))


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise CandidateBankError(f"{label} contains a non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, item in pairs:
            if key in seen:
                raise CandidateBankError(f"{label} contains a duplicate object key {key!r}")
            seen[key] = item
        return seen

    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateBankError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CandidateBankError(f"{label} root must be an object")
    return cast(dict[str, Any], value)


def _validate_schema(value: Mapping[str, Any], label: str) -> None:
    schema = json.loads(
        files("snn_memory").joinpath("schema", _BANK_SCHEMA).read_bytes().decode("utf-8")
    )
    validate = cast(Callable[[object, object], None], import_module("jsonschema").validate)
    try:
        validate(value, schema)
    except Exception as error:  # noqa: BLE001 - jsonschema raises its own hierarchy
        raise CandidateBankError(f"{label} failed schema {_BANK_SCHEMA}: {error}") from error


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _read_regular_bytes(directory_fd: int, name: str, label: str) -> bytes:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as error:
        raise CandidateBankError(f"{label} cannot be opened safely") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CandidateBankError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _signatures_block(signatures: FloatArray, ordered_record_ids: Sequence[str]) -> dict[str, Any]:
    canonical = _le(signatures, "<f8")
    body = canonical.tobytes(order="C")
    row_digests = [
        signature_row_digest(record_id, canonical[index])
        for index, record_id in enumerate(ordered_record_ids)
    ]
    return {
        "path": _SIGNATURES_PATH,
        "file_sha256": _sha(body),
        "raw_bytes_sha256": _sha(body),
        "decoded_float64_digest": decoded_float64_digest(canonical),
        "shape": list(canonical.shape),
        "byte_length": len(body),
        "row_digests": row_digests,
    }


def write_candidate_bank_v2(
    target: Path,
    *,
    manifest: Mapping[str, Any],
    signatures: FloatArray,
) -> ValidatedCandidateBank:
    """Transactionally seal one no-clobber candidate bank and re-read it.

    ``manifest`` carries every binding except the ``signatures`` byte digests and the
    ``self_sha256``; these are recomputed from ``signatures`` and must reconcile with the
    scoring ``candidate_bank_digest`` and the declared ordered record identities.
    """
    layout = manifest["signature_layout"]
    ordered = list(manifest["identities"]["ordered_record_ids"])
    n_neurons = int(layout["n_neurons"])
    _require(int(layout["bins"]) == _BINS, "signature bank must declare eight bins")
    _require(
        int(layout["signature_width"]) == _BINS * n_neurons,
        "signature width must be eight times the population",
    )
    _require_canonical_signatures(signatures, len(ordered), _BINS * n_neurons, "written")
    block = _signatures_block(signatures, ordered)
    row_digests = [str(value) for value in block["row_digests"]]
    expected_bank_digest = candidate_bank_digest(*_lexical_candidates(ordered, row_digests))
    _require(
        manifest["scoring"]["candidate_bank_digest"] == expected_bank_digest,
        "scoring candidate-bank digest does not recompute from the ordered rows",
    )

    sealed: dict[str, Any] = {key: value for key, value in manifest.items() if key != "self_sha256"}
    sealed["signatures"] = block
    sealed["self_sha256"] = _self_digest(sealed, _BANK_DOMAIN)
    _validate_schema(sealed, "candidate bank")
    _validate_bank_internal(sealed, ordered, n_neurons)

    signatures_body = _le(signatures, "<f8").tobytes()
    _install_bank_directory(
        target, {_BANK_PATH: _canonical_bytes(sealed), _SIGNATURES_PATH: signatures_body}
    )
    return read_candidate_bank_v2(target)


_RENAME_NOREPLACE = 1


def _open_dir_from_root(path: Path, label: str) -> int:
    """Open a directory by walking every component from the root, refusing symlinks and traversal."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise CandidateBankError(f"{label} filesystem root cannot be opened safely") from error
    try:
        for part in absolute.parts[1:]:
            if part in (os.curdir, os.pardir):
                raise CandidateBankError(f"{label} path carries a non-normal '{part}' component")
            try:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            except OSError as error:
                raise CandidateBankError(f"{label} cannot be opened safely") from error
            os.close(current)
            current = nxt
    except BaseException:
        os.close(current)
        raise
    return current


def _renameat2_noreplace(old_dir_fd: int, old_name: str, new_dir_fd: int, new_name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    result = libc.renameat2(
        old_dir_fd, os.fsencode(old_name), new_dir_fd, os.fsencode(new_name), _RENAME_NOREPLACE
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise CandidateBankError("candidate bank target already exists")
        raise CandidateBankError(f"atomic no-replace install failed: {os.strerror(code)}")


def _write_all_at(dir_fd: int, name: str, data: bytes) -> None:
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


def _install_bank_directory(target: Path, files: Mapping[str, bytes]) -> None:
    """Stage every file and atomically install the bank directory with no clobber, root-safely."""
    absolute = target if target.is_absolute() else Path.cwd() / target
    final = absolute.name
    if final in (os.curdir, os.pardir, ""):
        raise CandidateBankError("candidate bank target has no valid final component")
    parent_fd = _open_dir_from_root(absolute.parent, "candidate bank parent directory")
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


def _packaged_schema_sha256(name: str) -> str:
    return _sha(files("snn_memory").joinpath("schema", name).read_bytes())


def _validate_bank_internal(
    manifest: Mapping[str, Any], ordered: Sequence[str], n_neurons: int
) -> None:
    """Bank-internal invariants shared by writer (pre-install) and reader."""
    build = manifest["build"]
    for key, schema_name in (
        ("bank_schema_sha256", _BANK_SCHEMA),
        ("checkpoint_schema_sha256", "snn_memory_checkpoint_v2.schema.json"),
    ):
        if _packaged_schema_sha256(schema_name) != build[key]:
            raise CandidateBankError(f"bank build {key} differs from the packaged schema hash")
    calibration = manifest["calibration"]
    if [str(entry["record_id"]) for entry in calibration] != [str(rid) for rid in ordered]:
        raise CandidateBankError("bank calibration order or count differs from the ordered records")
    for field in (
        "encoder_identity",
        "encoder_directory_digest",
        "encoder_config_digest",
        "model_config_digest",
        "input_current",
        "topology_digest",
        "backend_build_digest",
    ):
        if len({entry[field] for entry in calibration}) != 1:
            raise CandidateBankError(f"bank calibration entries disagree on {field}")
    first = calibration[0]
    if first["encoder_directory_digest"] != build["encoder_directory_digest"]:
        raise CandidateBankError(
            "bank calibration encoder directory differs from the build identity"
        )
    if first["encoder_config_digest"] != build["encoder_config_digest"]:
        raise CandidateBankError("bank calibration encoder config differs from the build identity")
    if first["backend_build_digest"] != build["backend_build_digest"]:
        raise CandidateBankError(
            "bank calibration backend-build digest differs from the build identity"
        )
    # Ephemeral process identity (worker PID, nonce) lives in the CLI-report envelope,
    # never in this deterministic artifact; uniqueness is enforced at collection time.
    for entry in calibration:
        if entry["dtype"] != "<f8":
            raise CandidateBankError("bank calibration current dtype is not <f8")
        shape = [int(dim) for dim in entry["shape"]]
        if len(shape) != 2 or shape[0] < 1 or shape[1] != n_neurons:
            raise CandidateBankError("bank calibration current shape is not (timesteps, n_neurons)")
        if entry["input_weights_digest"] != entry["final_weights_digest"]:
            raise CandidateBankError("bank calibration weights changed under disabled plasticity")


def read_candidate_bank_v2(target: Path) -> ValidatedCandidateBank:
    """Authenticate one candidate bank through a single symlink-refusing read."""
    bank_fd = _open_dir_from_root(target, "candidate bank directory")
    try:
        if set(os.listdir(bank_fd)) != {_BANK_PATH, _SIGNATURES_PATH}:
            raise CandidateBankError(
                "candidate bank directory inventory is not exactly bank.json and signatures.bin"
            )
        manifest_bytes = _read_regular_bytes(bank_fd, _BANK_PATH, "candidate bank manifest")
        manifest = _strict_json(manifest_bytes, "candidate bank manifest")
        _validate_schema(manifest, "candidate bank")
        stored_self = manifest["self_sha256"]
        if stored_self != _self_digest(manifest, _BANK_DOMAIN):
            raise CandidateBankError("candidate bank self digest mismatch")
        if manifest_bytes != _canonical_bytes(manifest):
            raise CandidateBankError("candidate bank manifest is not canonical")

        layout = manifest["signature_layout"]
        block = manifest["signatures"]
        ordered = list(manifest["identities"]["ordered_record_ids"])
        n_neurons = int(layout["n_neurons"])
        width = _BINS * n_neurons
        _require(int(layout["signature_width"]) == width, "signature width mismatch")
        _require(
            layout["semantic_version"] == _SEMANTIC_VERSION, "unexpected signature semantic version"
        )
        _require(tuple(block["shape"]) == (len(ordered), width), "signature bank shape mismatch")

        body = _read_regular_bytes(bank_fd, _SIGNATURES_PATH, "candidate bank signatures")
        if _sha(body) != block["file_sha256"] or len(body) != block["byte_length"]:
            raise CandidateBankError("signature file digest or length mismatch")
        if len(body) != len(ordered) * width * 8:
            raise CandidateBankError("signature byte length differs from the declared bank shape")
        signatures = np.frombuffer(body, dtype="<f8").reshape(len(ordered), width)
        _require_canonical_signatures(signatures, len(ordered), width, "sealed")
        recomputed = _signatures_block(signatures, ordered)
        if recomputed["raw_bytes_sha256"] != block["raw_bytes_sha256"]:
            raise CandidateBankError("signature raw-byte digest mismatch")
        if recomputed["decoded_float64_digest"] != block["decoded_float64_digest"]:
            raise CandidateBankError("signature decoded-float64 digest mismatch")
        if list(recomputed["row_digests"]) != list(block["row_digests"]):
            raise CandidateBankError("signature row digests mismatch")
        expected_bank_digest = candidate_bank_digest(
            *_lexical_candidates(ordered, list(block["row_digests"]))
        )
        if manifest["scoring"]["candidate_bank_digest"] != expected_bank_digest:
            raise CandidateBankError(
                "scoring candidate-bank digest does not recompute from the rows"
            )
        if (
            ordered_record_ids_digest(tuple(str(value) for value in ordered))
            != manifest["identities"]["candidate_set_digest"]
        ):
            raise CandidateBankError(
                "candidate-set digest does not recompute from the ordered records"
            )
        if int(layout["completion_steps"]) != int(manifest["scoring"]["completion_steps"]):
            raise CandidateBankError("signature layout and scoring completion windows disagree")
        _validate_bank_internal(manifest, ordered, n_neurons)
    finally:
        os.close(bank_fd)
    owned = np.array(signatures, dtype=np.float64, copy=True)
    owned.setflags(write=False)
    return ValidatedCandidateBank(
        manifest=cast(Mapping[str, Any], _freeze(manifest)),
        self_sha256=stored_self,
        signatures=owned,
        row_digests=tuple(str(value) for value in block["row_digests"]),
    )
