# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Deterministic schema-v2 disjoint cue and bundle materialiser

"""Materialise the schema-v2 disjoint cue set and opaque probe cue bundles.

The materialiser consumes one externally pinned D1 source-universe artifact,
reauthenticates every selected source blob against the committed repository
state, and deterministically derives, per record: five non-overlapping and
non-empty contiguous event-index blocks (one digest-keyed calibration block,
four evaluation blocks), a digest-keyed permutation assigning the four frozen
corruption families to the evaluation blocks, and sixteen corruption variants
at exactly 0, 10, 25 and 40 percent. Corruption positions, noise-token choices,
calibration assignment, family permutation, and every opaque identifier derive
only from immutable content digests under distinct domain separators; no
runtime randomness or interpreter hash state is consulted. Probe cue bundles
embed the authenticated cue bytes, the pinned encoder identity/configuration,
the exact float64 embedding rows, and the precise sparse time-resolved
input-current representation; they carry no record identity, candidate,
provenance, label, or answer data. Artifacts are written only into an output
root that this call freshly creates beneath an absolute canonical non-symlink
parent — an existing or symlinked root, a symlinked parent chain, and any
pre-existing ``cues``/``bundles`` entry are rejected before encoder work. Every
file lands as an anonymous ``O_TMPFILE`` inode fsynced and hard-linked into the
directory fd captured at creation, so a concurrent rename of the root pathname
and symlink substitution cannot redirect a write outside the real inode and a
collision fails as a raw ``OSError`` instead of replacing bytes. Final
validation reads the manifest and every referenced artifact through the captured
root fd rather than the declared pathname, and the run reasserts that the
declared pathname still resolves to the captured inode after validation and
before returning, so a hostile rename — even to an otherwise-valid prepopulated
tree — fails closed rather than validating or returning the substitute. On
failure the owned inode is emptied through the captured fds and its empty
directory is removed when its pathname still binds that inode; a hostile rename
leaves the emptied owned directory as a fail-closed residual at an unknown
pathname instead of deleting an unowned replacement (see ``_OutputTransaction``
for the exact residual-inode policy).

``validate_cue_set_bytes`` and ``validate_cue_bundle_bytes`` re-derive every
binding an artifact can prove against itself and fail closed on disagreement.
``read_cue_set`` additionally authenticates every referenced text, lexicon and
bundle file through symlink-refusing single reads and re-derives every variant
text from its zero-percent base. Offline validation cannot re-run Git or the
pinned encoder, so the following remain selection-time claims sealed only by
the self digest and the caller-supplied expected file hash: the source-universe
binding (``file_sha256``, ``payload_self_sha256``, ``repository_head``,
``selected_record_ids``, each record's ``record_id``/``event_count`` and block
event indices as properties of the real repository), the encoder
``directory_sha256`` and ``identity``, the model ``config_digest`` beyond its
recomputable mapping, the claim that embedding/current arrays were produced by
the pinned encoder, and every ``implementations`` entry beyond its internal
byte/digest consistency. ``verify_cue_set_against_sources`` closes the
repository claims against a live repository plus the validated D1 artifact;
``verify_cue_bundle_with_encoder`` closes the array claims by re-encoding the
authenticated cue text with the real pinned local encoder and requiring
bit-exact equality.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import math
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.machinery import SourceFileLoader
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Mapping, Sequence, cast

import numpy as np

from snn_memory import encoder as encoder_module
from snn_memory import sentence_encoder as sentence_encoder_module
from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.encoder import directory_digest, embeddings_to_currents, split_events
from snn_memory.sentence_encoder import LocalSentenceEncoder
from snn_memory.source_universe import (
    EVENT_DOMAIN,
    SourceUniverseArtifact,
    SourceUniverseError,
    validate_source_universe_bytes,
)
from snn_memory.state import FloatArray, IntArray
from snn_memory.v2_contracts import canonical_json_digest, normalize_cue_text, token_jaccard

SCHEMA_VERSION = 2
CUE_SET_VERSION = 1
BUNDLE_VERSION = 1
BLOCK_COUNT = 5
REQUESTED_PERCENTS = (0, 10, 25, 40)
FAMILIES = ("truncation", "deletion", "masking", "sparse_noise")
NEAR_DUPLICATE_MAXIMUM = 0.9
MASK_TOKEN = "[MASK]"
TOKENIZER_IDENTITY = "remanentia-snn-v2-cue-tokenizer-v1"
TOKENIZER_RULE = (
    "canonical-event=single-space-join-of-str-split;"
    "cue-text=lf-join-of-canonical-events;"
    "tokens=split-on-single-space-per-line"
)
NOISE_LEXICON_IDENTITY = "remanentia-snn-v2-noise-lexicon-v1"
NOISE_LEXICON = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "agate", "amber", "basalt", "beryl", "flint", "garnet", "gneiss", "jasper",
    "marble", "obsidian", "onyx", "opal", "pumice", "quartz", "schist", "slate",
    "alder", "aspen", "birch", "cedar", "elm", "hazel", "larch", "linden",
    "maple", "rowan", "spruce", "willow", "brook", "cliff", "dune", "fjord",
    "glacier", "lagoon", "mesa", "tundra", "café", "naïve", "señal", "žula",
)
CUE_SET_NAME = "cue_set.json"
NOISE_LEXICON_NAME = "noise_lexicon.txt"
MATERIALIZER_LOGICAL_PATH = "snn_memory/cue_materializer.py"
SPLIT_EVENTS_LOGICAL_PATH = "snn_memory/encoder.py"
SENTENCE_ENCODER_LOGICAL_PATH = "snn_memory/sentence_encoder.py"
_CUE_SET_SELF_DOMAIN = b"remanentia:snn-v2-cue-set:v1\0"
_BUNDLE_SELF_DOMAIN = b"remanentia:snn-v2-cue-bundle:v1\0"
_CUE_ID_DOMAIN = b"remanentia:snn-v2-cue-id:v1\0"
_VARIANT_ID_DOMAIN = b"remanentia:snn-v2-variant-id:v1\0"
_CALIBRATION_DOMAIN = b"remanentia:snn-v2-calibration-block:v1\0"
_PERMUTATION_DOMAIN = b"remanentia:snn-v2-family-permutation:v1\0"
_POSITION_DOMAIN = b"remanentia:snn-v2-corruption-positions:v1\0"
_NOISE_TOKEN_DOMAIN = b"remanentia:snn-v2-noise-token:v1\0"
_TOKENIZER_DOMAIN = b"remanentia:snn-v2-tokenizer:v1\0"
_CUE_SET_SCHEMA_NAME = "snn_memory_cue_set_v2.schema.json"
_BUNDLE_SCHEMA_NAME = "snn_memory_cue_bundle_v2.schema.json"
_FAMILY_PERMUTATIONS = tuple(itertools.permutations(FAMILIES))
_HEAD_RE = re.compile(r"[0-9a-f]{40}")


def _tokenizer_digest() -> str:
    material = (
        _TOKENIZER_DOMAIN
        + TOKENIZER_IDENTITY.encode("ascii")
        + b"\0"
        + TOKENIZER_RULE.encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()


TOKENIZER_SHA256 = _tokenizer_digest()
NOISE_LEXICON_BYTES = ("\n".join(NOISE_LEXICON) + "\n").encode("utf-8")
NOISE_LEXICON_SHA256 = hashlib.sha256(NOISE_LEXICON_BYTES).hexdigest()


class CueMaterializerError(ValueError):
    """A fail-closed cue-materialisation contract violation."""


@dataclass(frozen=True)
class CueSetArtifact:
    """Validated canonical cue-set manifest with a read-only payload view."""

    payload: Mapping[str, Any]
    canonical_bytes: bytes
    file_sha256: str
    payload_self_sha256: str


@dataclass(frozen=True)
class CueBundleArtifact:
    """Validated canonical cue bundle with decoded read-only arrays."""

    payload: Mapping[str, Any]
    canonical_bytes: bytes
    file_sha256: str
    payload_self_sha256: str
    text: str
    embedding: FloatArray
    current_rows: IntArray
    current_columns: IntArray
    current_values: FloatArray
    currents_shape: tuple[int, int]


@dataclass(frozen=True)
class CueSetWriteResult:
    """Authenticated result of one atomic cue-set materialisation."""

    output_directory: Path
    cue_set_path: Path
    file_sha256: str
    payload_self_sha256: str


def _run_git(root: Path, arguments: list[str]) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", "replace").strip()
        raise CueMaterializerError(f"git {' '.join(arguments[:2])} failed: {detail}")
    return process.stdout


def _strict_json(raw: bytes, context: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CueMaterializerError(f"{context} contains a duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(constant: str) -> None:
        raise CueMaterializerError(f"{context} contains non-finite JSON constant {constant}")

    def parse_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise CueMaterializerError(f"{context} contains a non-finite JSON number")
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, CueMaterializerError):
            raise
        raise CueMaterializerError(f"{context} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CueMaterializerError(f"{context} root must be an object")
    return value


def _canonical(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _self_digest(payload: dict[str, Any], domain: bytes) -> str:
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    canonical = _canonical(unsigned)
    framed = domain + len(canonical).to_bytes(8, "big") + canonical
    return hashlib.sha256(framed).hexdigest()


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _schema_bytes(schema_name: str) -> bytes:
    return files("snn_memory").joinpath("schema", schema_name).read_bytes()


def _validate_schema(payload: dict[str, Any], schema_name: str, context: str) -> None:
    schema = _strict_json(_schema_bytes(schema_name), f"{context} schema")
    validator_class = cast(Any, import_module("jsonschema")).Draft202012Validator
    try:
        validator_class(schema).validate(payload)
    except Exception as error:
        raise CueMaterializerError(f"{context} schema validation failed") from error


def _digest_int(domain: bytes, material: str) -> int:
    return int(hashlib.sha256(domain + material.encode("ascii")).hexdigest(), 16)


def _require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CueMaterializerError(f"{label} must be a JSON integer")
    return value


def _opaque_id(domain: bytes, material: str, prefix: str) -> str:
    return prefix + hashlib.sha256(domain + material.encode("utf-8")).hexdigest()[:32]


def _cue_id(text_sha256: str) -> str:
    return _opaque_id(_CUE_ID_DOMAIN, text_sha256, "cue-")


def _variant_id(base_text_sha256: str, family: str, requested_percent: int) -> str:
    material = f"{base_text_sha256}\0{family}\0{requested_percent}"
    return _opaque_id(_VARIANT_ID_DOMAIN, material, "variant-")


def _calibration_block_index(record_sha256: str) -> int:
    return _digest_int(_CALIBRATION_DOMAIN, record_sha256) % BLOCK_COUNT


def _family_permutation_index(record_sha256: str) -> int:
    return _digest_int(_PERMUTATION_DOMAIN, record_sha256) % len(_FAMILY_PERMUTATIONS)


def _block_boundaries(event_count: int) -> list[int]:
    return [index * event_count // BLOCK_COUNT for index in range(BLOCK_COUNT + 1)]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize_cue_text(text).encode("utf-8")).hexdigest()


def _affected_count(token_count: int, requested_percent: int) -> int:
    return (token_count * requested_percent + 50) // 100


def _ranked_positions(family: str, base_text_sha256: str, universe: int, count: int) -> list[int]:
    def rank(position: int) -> str:
        material = (
            _POSITION_DOMAIN
            + family.encode("ascii")
            + b"\0"
            + base_text_sha256.encode("ascii")
            + b"\0"
            + position.to_bytes(8, "big")
        )
        return hashlib.sha256(material).hexdigest()

    ordered = sorted(range(universe), key=rank)
    return sorted(ordered[:count])


def _selected_positions(
    family: str, base_text_sha256: str, token_count: int, requested_percent: int
) -> list[int]:
    affected = _affected_count(token_count, requested_percent)
    if family == "truncation":
        return list(range(token_count - affected, token_count))
    if family == "sparse_noise":
        return _ranked_positions(family, base_text_sha256, token_count + 1, affected)
    return _ranked_positions(family, base_text_sha256, token_count, affected)


def _noise_token(base_text_sha256: str, gap: int, lexicon: Sequence[str]) -> str:
    material = (
        _NOISE_TOKEN_DOMAIN
        + base_text_sha256.encode("ascii")
        + b"\0"
        + gap.to_bytes(8, "big")
    )
    return lexicon[int(hashlib.sha256(material).hexdigest(), 16) % len(lexicon)]


def _canonical_event(event: str) -> str:
    return " ".join(event.split())


def _block_text(events: Sequence[str]) -> str:
    return "\n".join(_canonical_event(event) for event in events)


def _line_tokens(text: str) -> list[list[str]]:
    return [line.split(" ") for line in text.split("\n")]


def _require_canonical_cue_text(text: str, context: str) -> None:
    if not text:
        raise CueMaterializerError(f"{context} cue text is empty")
    for line in text.split("\n"):
        if not line or line != " ".join(line.split()):
            raise CueMaterializerError(f"{context} cue text is not canonical")


def _require_line_count(text: str, event_indices: Sequence[int], context: str) -> None:
    if len(text.split("\n")) != len(event_indices):
        raise CueMaterializerError(f"{context} cue lines differ from its event indices")


def _apply_transform(
    family: str,
    lines: list[list[str]],
    selected: Sequence[int],
    base_text_sha256: str,
    lexicon: Sequence[str],
) -> str:
    flat = [(index, token) for index, line in enumerate(lines) for token in line]
    rebuilt: list[list[str]] = [[] for _ in lines]
    if family == "sparse_noise":
        by_gap = {gap: _noise_token(base_text_sha256, gap, lexicon) for gap in selected}
        for position, (line_index, token) in enumerate(flat):
            if position in by_gap:
                rebuilt[line_index].append(by_gap[position])
            rebuilt[line_index].append(token)
        if len(flat) in by_gap:
            rebuilt[flat[-1][0]].append(by_gap[len(flat)])
    else:
        chosen = frozenset(selected)
        for position, (line_index, token) in enumerate(flat):
            if position in chosen:
                if family == "masking":
                    rebuilt[line_index].append(MASK_TOKEN)
                continue
            rebuilt[line_index].append(token)
    return "\n".join(" ".join(line) for line in rebuilt if line)


def _implementation(module: ModuleType, logical_path: str) -> dict[str, Any]:
    specification = getattr(module, "__spec__", None)
    origin_text = getattr(module, "__file__", None)
    if (
        specification is None
        or not isinstance(specification.loader, SourceFileLoader)
        or origin_text is None
        or specification.origin is None
        or Path(origin_text).absolute() != Path(specification.origin).absolute()
    ):
        raise CueMaterializerError("implementation module is not a real non-shadowed source file")
    current = Path(origin_text).absolute()
    for candidate in (current, *current.parents):
        if candidate.is_symlink():
            raise CueMaterializerError("implementation source traverses a symlink")
    try:
        raw = current.resolve(strict=True).read_bytes()
    except OSError as error:
        raise CueMaterializerError("implementation source cannot be read") from error
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CueMaterializerError("implementation source is not strict UTF-8") from error
    return {
        "logical_path": logical_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
    }


def _event_order_digest(events: Sequence[str]) -> tuple[str, list[str]]:
    framed = bytearray(EVENT_DOMAIN)
    framed.extend(len(events).to_bytes(8, "big"))
    hashes: list[str] = []
    for event in events:
        raw = event.encode("utf-8")
        framed.extend(len(raw).to_bytes(8, "big"))
        framed.extend(raw)
        hashes.append(hashlib.sha256(raw).hexdigest())
    return hashlib.sha256(bytes(framed)).hexdigest(), hashes


def _read_regular_bytes(path: Path, context: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise CueMaterializerError(f"{context} cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CueMaterializerError(f"{context} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _secure_artifact_path(root: Path, relative: str, context: str) -> Path:
    candidate = root
    for part in Path(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise CueMaterializerError(f"{context} path traverses a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise CueMaterializerError(f"{context} path escapes or is missing") from error
    return resolved


def _read_referenced_bytes(root: Path, relative: str, expected_sha256: str, context: str) -> bytes:
    resolved = _secure_artifact_path(root, relative, context)
    raw = _read_regular_bytes(resolved, context)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise CueMaterializerError(f"{context} digest mismatch")
    return raw


def _decode_utf8(raw: bytes, context: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CueMaterializerError(f"{context} is not strict UTF-8") from error


def _encode_array(array: FloatArray | IntArray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _decode_array(
    encoded: str, dtype: str, count: int, context: str
) -> FloatArray | IntArray:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise CueMaterializerError(f"{context} base64 payload is invalid") from error
    if len(raw) != count * 8:
        raise CueMaterializerError(f"{context} byte length differs from its declared shape")
    array = np.frombuffer(raw, dtype=np.dtype(dtype))
    if array.dtype == np.float64 and not bool(np.all(np.isfinite(array))):
        raise CueMaterializerError(f"{context} contains non-finite values")
    return cast("FloatArray | IntArray", array)


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class _OutputTransaction:
    """Directory-fd-anchored writer for one freshly created output root.

    Every artifact is created as an unnamed ``O_TMPFILE`` inode inside the
    directory fd captured at creation and hard-linked into place, so a
    concurrent rename of the root pathname and symlink substitution cannot
    redirect a write outside the real created inode. Validation reads through
    ``fd_root`` (a ``/proc/self/fd`` path that always resolves to the captured
    inode), and ``assert_bound`` then reconfirms that the declared pathname still
    resolves to that inode before the result is returned: a hostile rename — even
    to an otherwise-valid prepopulated tree — fails closed instead of validating
    or returning a substituted destination.

    Failure-cleanup follows an explicit residual-inode policy. ``purge`` always
    empties the owned inode through the captured fds, then removes the empty root
    directory only when its original pathname still binds that inode (a no-follow
    ``st_dev``/``st_ino`` identity match against the captured fd). When a hostile
    rename has moved the owned inode to an attacker-chosen pathname this call
    cannot name, ``purge`` refuses to delete by the original name (which could
    remove an unowned replacement) and sets ``residual_retained`` — no artifacts
    remain and no substituted pathname is followed, but the empty owned directory
    itself survives at an unknown pathname.
    """

    def __init__(
        self, path: Path, name: str, parent_fd: int, root_fd: int, cues_fd: int, bundles_fd: int
    ) -> None:
        self.path = path
        self.residual_retained = False
        self._name = name
        self._parent_fd = parent_fd
        self._root_fd = root_fd
        self._subdirs = {"cues": cues_fd, "bundles": bundles_fd}

    def fd_root(self) -> Path:
        """Return the ``/proc/self/fd`` path that always resolves to the captured inode."""
        return Path(f"/proc/self/fd/{self._root_fd}")

    def _target(self, relative: str) -> tuple[int, str]:
        parts = relative.split("/")
        if len(parts) == 1:
            return self._root_fd, parts[0]
        subdir, leaf = parts
        return self._subdirs[subdir], leaf

    def _binds_root(self, name: str | Path, dir_fd: int | None) -> bool:
        owned = os.fstat(self._root_fd)
        try:
            entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            return False
        return (
            stat.S_ISDIR(entry.st_mode)
            and entry.st_ino == owned.st_ino
            and entry.st_dev == owned.st_dev
        )

    def write(self, relative: str, raw: bytes) -> None:
        directory_fd, leaf = self._target(relative)
        descriptor = os.open(".", os.O_WRONLY | os.O_TMPFILE, 0o644, dir_fd=directory_fd)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(descriptor)
            # linkat of the anonymous inode into the captured directory fd keeps
            # the hard no-clobber property: a collision inside the fresh root
            # surfaces as a raw OSError instead of replacing bytes.
            os.link(f"/proc/self/fd/{descriptor}", leaf, dst_dir_fd=directory_fd,
                    follow_symlinks=True)
            os.fsync(directory_fd)
        finally:
            os.close(descriptor)

    def assert_bound(self) -> None:
        if not self._binds_root(self.path, None):
            raise CueMaterializerError("output root pathname was replaced during materialisation")

    def purge(self) -> None:
        for subdir, directory_fd in self._subdirs.items():
            for entry in os.listdir(directory_fd):
                os.unlink(entry, dir_fd=directory_fd)
            os.rmdir(subdir, dir_fd=self._root_fd)
        for entry in os.listdir(self._root_fd):
            os.unlink(entry, dir_fd=self._root_fd)
        if self._binds_root(self._name, self._parent_fd):
            os.rmdir(self._name, dir_fd=self._parent_fd)
        else:
            # A hostile rename moved the owned inode to an attacker-chosen pathname
            # this call cannot name; deleting the original pathname could remove an
            # unowned replacement, so the emptied owned directory is left as a
            # fail-closed residual at an unknown pathname.
            self.residual_retained = True

    def close(self) -> None:
        os.close(self._root_fd)
        os.close(self._subdirs["cues"])
        os.close(self._subdirs["bundles"])
        os.close(self._parent_fd)


def _fresh_output_root(output_directory: Path) -> _OutputTransaction:
    output_directory = output_directory.absolute()
    parent = output_directory.parent
    name = output_directory.name
    try:
        canonical_parent = parent.resolve(strict=True)
    except OSError as error:
        raise CueMaterializerError("output parent cannot be resolved") from error
    if parent != canonical_parent:
        raise CueMaterializerError("output parent must be an absolute canonical path")
    parent_fd = os.open(parent, _DIR_FLAGS)
    try:
        os.mkdir(name, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise CueMaterializerError("output root must be freshly created") from error
    root_fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    os.mkdir("cues", dir_fd=root_fd)
    os.mkdir("bundles", dir_fd=root_fd)
    cues_fd = os.open("cues", _DIR_FLAGS, dir_fd=root_fd)
    bundles_fd = os.open("bundles", _DIR_FLAGS, dir_fd=root_fd)
    return _OutputTransaction(output_directory, name, parent_fd, root_fd, cues_fd, bundles_fd)


def _config_mapping(config: ModelConfig | EncoderConfig) -> dict[str, Any]:
    return asdict(config)


def _config_from_mapping(
    mapping: Mapping[str, Any], config_type: type[ModelConfig] | type[EncoderConfig], context: str
) -> ModelConfig | EncoderConfig:
    try:
        return config_type(**mapping)
    except (TypeError, ValueError) as error:
        raise CueMaterializerError(f"{context} configuration is invalid") from error


def _keep_count(n_neurons: int, active_fraction: float) -> int:
    return max(1, int(n_neurons * active_fraction))


def _bundle_arrays(
    adapter: LocalSentenceEncoder,
    model: ModelConfig,
    encoder_config: EncoderConfig,
    input_current: float,
    text: str,
) -> tuple[FloatArray, IntArray, IntArray, FloatArray, tuple[int, int]]:
    events = split_events(text)
    embedding = adapter.encode(events)
    currents = embeddings_to_currents(
        embedding, model, encoder_config, input_current=input_current
    )
    rows, columns = np.nonzero(currents)
    values = currents[rows, columns]
    return (
        embedding,
        rows.astype(np.int64),
        columns.astype(np.int64),
        values.astype(np.float64),
        (int(currents.shape[0]), int(currents.shape[1])),
    )


def _bundle_payload(
    variant_id: str,
    text: str,
    encoder_block: dict[str, Any],
    model_block: dict[str, Any],
    implementations: dict[str, Any],
    arrays: tuple[FloatArray, IntArray, IntArray, FloatArray, tuple[int, int]],
) -> dict[str, Any]:
    embedding, rows, columns, values, shape = arrays
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "snn-memory-cue-bundle",
        "bundle_version": BUNDLE_VERSION,
        "cue_id": variant_id,
        "text_sha256": _sha256_text(text),
        "normalized_text_sha256": _normalized_sha256(text),
        "text_utf8_base64": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "encoder": encoder_block,
        "model": model_block,
        "embedding": {
            "dtype": "<f8",
            "shape": [int(embedding.shape[0]), int(embedding.shape[1])],
            "data_base64": _encode_array(embedding),
        },
        "currents": {
            "encoding": "coo-sorted-v1",
            "dtype": "<f8",
            "index_dtype": "<i8",
            "shape": [shape[0], shape[1]],
            "row_base64": _encode_array(rows),
            "column_base64": _encode_array(columns),
            "value_base64": _encode_array(values),
        },
        "implementations": implementations,
    }
    payload["self_sha256"] = _self_digest(payload, _BUNDLE_SELF_DOMAIN)
    return payload


def validate_cue_bundle_bytes(
    raw: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> CueBundleArtifact:
    """Validate canonical bundle bytes and every offline-provable binding."""
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise CueMaterializerError("cue-bundle file SHA-256 mismatch")
    payload = _strict_json(raw, "cue-bundle artifact")
    _validate_schema(payload, _BUNDLE_SCHEMA_NAME, "cue-bundle")
    if _canonical(payload) != raw:
        raise CueMaterializerError("cue-bundle JSON is not canonical")
    if payload["self_sha256"] != _self_digest(payload, _BUNDLE_SELF_DOMAIN):
        raise CueMaterializerError("cue-bundle self digest mismatch")
    try:
        text_raw = base64.b64decode(payload["text_utf8_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise CueMaterializerError("cue-bundle text base64 is invalid") from error
    text = _decode_utf8(text_raw, "cue-bundle text")
    if hashlib.sha256(text_raw).hexdigest() != payload["text_sha256"]:
        raise CueMaterializerError("cue-bundle text digest mismatch")
    if _normalized_sha256(text) != payload["normalized_text_sha256"]:
        raise CueMaterializerError("cue-bundle normalized-text digest mismatch")
    _require_canonical_cue_text(text, "cue-bundle")
    encoder_block = payload["encoder"]
    if encoder_block["config_digest"] != canonical_json_digest(encoder_block["config"]):
        raise CueMaterializerError("cue-bundle encoder configuration digest mismatch")
    encoder_config = cast(
        EncoderConfig,
        _config_from_mapping(encoder_block["config"], EncoderConfig, "cue-bundle encoder"),
    )
    model_block = payload["model"]
    n_neurons = _require_int(model_block["n_neurons"], "cue-bundle neuron count")
    input_current = model_block["input_current"]
    events = split_events(text)
    embedding_spec = payload["embedding"]
    for value in (*embedding_spec["shape"], *payload["currents"]["shape"]):
        _require_int(value, "cue-bundle array shape")
    if embedding_spec["shape"][0] != len(events):
        raise CueMaterializerError("cue-bundle embedding rows differ from cue events")
    embedding_count = embedding_spec["shape"][0] * embedding_spec["shape"][1]
    embedding = cast(
        FloatArray,
        _decode_array(embedding_spec["data_base64"], "<f8", embedding_count, "cue-bundle embedding"),
    ).reshape(embedding_spec["shape"][0], embedding_spec["shape"][1])
    currents_spec = payload["currents"]
    timesteps = len(events) * (encoder_config.packet_ms + encoder_config.silent_ms)
    if currents_spec["shape"] != [timesteps, n_neurons]:
        raise CueMaterializerError("cue-bundle current shape differs from its cue and model")
    entry_count = len(events) * _keep_count(n_neurons, encoder_config.active_fraction)
    rows = cast(
        IntArray, _decode_array(currents_spec["row_base64"], "<i8", entry_count, "cue-bundle rows")
    )
    columns = cast(
        IntArray,
        _decode_array(currents_spec["column_base64"], "<i8", entry_count, "cue-bundle columns"),
    )
    values = cast(
        FloatArray,
        _decode_array(currents_spec["value_base64"], "<f8", entry_count, "cue-bundle values"),
    )
    _validate_current_entries(
        rows, columns, values, events, encoder_config, n_neurons, input_current
    )
    return CueBundleArtifact(
        payload=_frozen(payload),
        canonical_bytes=raw,
        file_sha256=file_sha256,
        payload_self_sha256=cast(str, payload["self_sha256"]),
        text=text,
        embedding=embedding,
        current_rows=rows,
        current_columns=columns,
        current_values=values,
        currents_shape=(timesteps, n_neurons),
    )


def _validate_current_entries(
    rows: IntArray,
    columns: IntArray,
    values: FloatArray,
    events: Sequence[str],
    encoder_config: EncoderConfig,
    n_neurons: int,
    input_current: float,
) -> None:
    period = encoder_config.packet_ms + encoder_config.silent_ms
    keep = _keep_count(n_neurons, encoder_config.active_fraction)
    if not bool(np.all(values == input_current)):
        raise CueMaterializerError("cue-bundle current values differ from the input current")
    if bool(np.any(rows < 0)) or bool(np.any(rows >= len(events) * period)):
        raise CueMaterializerError("cue-bundle current rows are out of range")
    if bool(np.any(columns < 0)) or bool(np.any(columns >= n_neurons)):
        raise CueMaterializerError("cue-bundle current columns are out of range")
    order = rows.astype(np.int64) * n_neurons + columns.astype(np.int64)
    if not bool(np.all(np.diff(order) > 0)):
        raise CueMaterializerError("cue-bundle current entries are not strictly sorted")
    if bool(np.any(rows % period >= encoder_config.packet_ms)):
        raise CueMaterializerError("cue-bundle current entries fall in a silent window")
    per_event = np.bincount(rows // period, minlength=len(events))
    if not bool(np.all(per_event == keep)):
        raise CueMaterializerError("cue-bundle per-event active counts differ from the contract")


def read_cue_bundle(
    bundle_path: Path,
    expected_file_sha256: str | None = None,
) -> CueBundleArtifact:
    """Read one bundle through a symlink-refusing single read and validate it."""
    raw = _read_regular_bytes(bundle_path, "cue-bundle file")
    return validate_cue_bundle_bytes(raw, expected_file_sha256=expected_file_sha256)


def verify_cue_bundle_with_encoder(
    bundle: CueBundleArtifact,
    encoder_checkpoint: Path,
    expected_encoder_digest: str,
) -> None:
    """Prove the bundle arrays bit-exactly through the real pinned encoder."""
    payload = bundle.payload
    if not encoder_checkpoint.is_dir():
        raise CueMaterializerError("pinned encoder checkpoint directory is missing")
    live_digest = directory_digest(encoder_checkpoint)
    if live_digest != expected_encoder_digest:
        raise CueMaterializerError("pinned encoder directory digest mismatch")
    if payload["encoder"]["directory_sha256"] != live_digest:
        raise CueMaterializerError("cue-bundle encoder directory digest differs from the pin")
    encoder_config = cast(
        EncoderConfig,
        _config_from_mapping(payload["encoder"]["config"], EncoderConfig, "cue-bundle encoder"),
    )
    # embeddings_to_currents reads only n_neurons from the model configuration,
    # so a synthetic ModelConfig reproduces the exact current derivation.
    model = cast(
        ModelConfig,
        _config_from_mapping(
            {"n_neurons": payload["model"]["n_neurons"]}, ModelConfig, "cue-bundle model"
        ),
    )
    adapter = LocalSentenceEncoder(encoder_checkpoint)
    arrays = _bundle_arrays(
        adapter,
        model,
        encoder_config,
        cast(float, payload["model"]["input_current"]),
        bundle.text,
    )
    embedding, rows, columns, values, _shape = arrays
    if embedding.tobytes() != bundle.embedding.tobytes():
        raise CueMaterializerError("cue-bundle embedding differs from the pinned encoder output")
    if (
        rows.tobytes() != bundle.current_rows.tobytes()
        or columns.tobytes() != bundle.current_columns.tobytes()
        or values.tobytes() != bundle.current_values.tobytes()
    ):
        raise CueMaterializerError("cue-bundle currents differ from the pinned derivation")


def _repository_head(repo_root: Path) -> str:
    if repo_root != repo_root.resolve(strict=True):
        raise CueMaterializerError("repo_root must be an absolute canonical path")
    top = Path(_run_git(repo_root, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip())
    if top.resolve(strict=True) != repo_root:
        raise CueMaterializerError("repo_root is not the Git top level")
    head = _run_git(repo_root, ["rev-parse", "--verify", "HEAD^{commit}"]).decode("ascii").strip()
    if _HEAD_RE.fullmatch(head) is None:
        raise CueMaterializerError("repository HEAD is not a SHA-1 commit")
    return head


def _reauthenticated_events(repo_root: Path, item: Mapping[str, Any]) -> list[str]:
    oid = _run_git(repo_root, ["rev-parse", f"HEAD:{item['path']}"]).decode("ascii").strip()
    if oid != item["blob_oid"]:
        raise CueMaterializerError("selected blob identity differs from the artifact")
    raw = _run_git(repo_root, ["cat-file", "blob", oid])
    if hashlib.sha256(raw).hexdigest() != item["content_sha256"]:
        raise CueMaterializerError("selected source bytes differ from the artifact")
    text = _decode_utf8(raw, "selected source")
    events = split_events(text)
    if len(events) != item["event_count"]:
        raise CueMaterializerError("selected source event count differs from the artifact")
    order_digest, event_hashes = _event_order_digest(events)
    if order_digest != item["event_order_digest"] or event_hashes != list(item["event_sha256"]):
        raise CueMaterializerError("selected source event order differs from the artifact")
    return events


def _record_entry(
    item: Mapping[str, Any], events: Sequence[str], lexicon: Sequence[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    record_sha = cast(str, item["content_sha256"])
    event_count = cast(int, item["event_count"])
    boundaries = _block_boundaries(event_count)
    blocks = [
        list(range(boundaries[index], boundaries[index + 1])) for index in range(BLOCK_COUNT)
    ]
    calibration_index = _calibration_block_index(record_sha)
    permutation = _FAMILY_PERMUTATIONS[_family_permutation_index(record_sha)]
    texts: dict[str, str] = {}
    calibration_indices = blocks[calibration_index]
    calibration_text = _block_text([events[index] for index in calibration_indices])
    calibration_sha = _sha256_text(calibration_text)
    calibration_id = _cue_id(calibration_sha)
    calibration_path = f"cues/{calibration_id}.txt"
    texts[calibration_path] = calibration_text
    base_cues: list[dict[str, Any]] = []
    evaluation_blocks = [
        blocks[index] for index in range(BLOCK_COUNT) if index != calibration_index
    ]
    for family, indices in zip(permutation, evaluation_blocks, strict=True):
        base_text = _block_text([events[index] for index in indices])
        base_sha = _sha256_text(base_text)
        lines = _line_tokens(base_text)
        token_count = sum(len(line) for line in lines)
        variants: list[dict[str, Any]] = []
        for percent in REQUESTED_PERCENTS:
            selected = _selected_positions(family, base_sha, token_count, percent)
            text = _apply_transform(family, lines, selected, base_sha, lexicon)
            variant_id = _variant_id(base_sha, family, percent)
            path = f"cues/{variant_id}.txt"
            texts[path] = text
            variants.append(
                {
                    "variant_id": variant_id,
                    "path": path,
                    "sha256": _sha256_text(text),
                    "normalized_text_sha256": _normalized_sha256(text),
                    "requested_percent": percent,
                    "affected_count": len(selected),
                    "realized_fraction": len(selected) / token_count,
                    "selected_positions": selected,
                    "tokenizer_digest": TOKENIZER_SHA256,
                }
            )
        base_cues.append(
            {
                "cue_id": _cue_id(base_sha),
                "task": "record_recall",
                "transform_family": family,
                "event_indices": indices,
                "token_count": token_count,
                "variants": variants,
            }
        )
    entry = {
        "record_id": item["record_id"],
        "event_count": event_count,
        "calibration_block_index": calibration_index,
        "family_permutation_index": _family_permutation_index(record_sha),
        "calibration_cue": {
            "cue_id": calibration_id,
            "path": calibration_path,
            "sha256": calibration_sha,
            "normalized_text_sha256": _normalized_sha256(calibration_text),
            "event_indices": calibration_indices,
        },
        "evaluation_base_cues": base_cues,
    }
    return entry, texts


def _collect_texts(
    entries: list[dict[str, Any]], per_record_texts: list[dict[str, str]]
) -> dict[str, str]:
    all_texts: dict[str, str] = {}
    text_sha_seen: set[str] = set()
    normalized_seen: set[str] = set()
    for texts in per_record_texts:
        for path, text in texts.items():
            if _sha256_text(text) in text_sha_seen:
                raise CueMaterializerError("duplicate cue text across the materialised set")
            if _normalized_sha256(text) in normalized_seen:
                raise CueMaterializerError("duplicate normalized cue text across the set")
            text_sha_seen.add(_sha256_text(text))
            normalized_seen.add(_normalized_sha256(text))
            all_texts[path] = text
    anchors: list[tuple[str, str]] = []
    for entry, texts in zip(entries, per_record_texts, strict=True):
        calibration = entry["calibration_cue"]
        anchors.append((cast(str, calibration["cue_id"]), texts[calibration["path"]]))
        for base_cue in entry["evaluation_base_cues"]:
            zero_variant = base_cue["variants"][0]
            anchors.append((cast(str, base_cue["cue_id"]), texts[zero_variant["path"]]))
    for index, (left_id, left_text) in enumerate(anchors):
        for right_id, right_text in anchors[index + 1 :]:
            if token_jaccard(left_text, right_text) > NEAR_DUPLICATE_MAXIMUM:
                raise CueMaterializerError(f"near-duplicate cues: {left_id} and {right_id}")
    return all_texts


def _derivation_block() -> dict[str, Any]:
    def escaped(domain: bytes) -> str:
        return domain.decode("ascii").replace("\0", "\\0")

    return {
        "block_count": BLOCK_COUNT,
        "requested_percents": list(REQUESTED_PERCENTS),
        "rounding": "half-up-integer-v1",
        "families": list(FAMILIES),
        "calibration_domain_ascii": escaped(_CALIBRATION_DOMAIN),
        "family_permutation_domain_ascii": escaped(_PERMUTATION_DOMAIN),
        "position_domain_ascii": escaped(_POSITION_DOMAIN),
        "noise_token_domain_ascii": escaped(_NOISE_TOKEN_DOMAIN),
        "cue_id_domain_ascii": escaped(_CUE_ID_DOMAIN),
        "variant_id_domain_ascii": escaped(_VARIANT_ID_DOMAIN),
    }


def materialize_cue_set(
    repo_root: Path,
    source_universe_path: Path,
    expected_source_universe_sha256: str,
    encoder_checkpoint: Path,
    expected_encoder_digest: str,
    output_directory: Path,
    *,
    model: ModelConfig | None = None,
    encoder_config: EncoderConfig | None = None,
    input_current: float = 18.0,
) -> CueSetWriteResult:
    """Materialise the deterministic cue set, bundles and lexicon atomically."""
    if not math.isfinite(input_current) or input_current <= 0.0:
        raise CueMaterializerError("input current must be a finite positive number")
    if model is None:
        model = ModelConfig()
    if encoder_config is None:
        encoder_config = EncoderConfig()
    source_raw = _read_regular_bytes(source_universe_path, "source-universe file")
    source_universe = validate_source_universe_bytes(
        source_raw, expected_file_sha256=expected_source_universe_sha256
    )
    repo_root = repo_root.absolute()
    head = _repository_head(repo_root)
    if head != source_universe.payload["repository"]["head"]:
        raise CueMaterializerError("repository HEAD differs from the source-universe artifact")
    entries: list[dict[str, Any]] = []
    per_record_texts: list[dict[str, str]] = []
    for item in source_universe.payload["selected"]:
        events = _reauthenticated_events(repo_root, item)
        entry, texts = _record_entry(item, events, NOISE_LEXICON)
        entries.append(entry)
        per_record_texts.append(texts)
    all_texts = _collect_texts(entries, per_record_texts)
    implementations = {
        "cue_materializer": _implementation(sys.modules[__name__], MATERIALIZER_LOGICAL_PATH),
        "split_events": _implementation(encoder_module, SPLIT_EVENTS_LOGICAL_PATH),
        "sentence_encoder": _implementation(
            sentence_encoder_module, SENTENCE_ENCODER_LOGICAL_PATH
        ),
    }
    bundle_implementations = {
        name: {"logical_path": block["logical_path"], "sha256": block["sha256"]}
        for name, block in implementations.items()
    }
    transaction = _fresh_output_root(output_directory)
    try:
        return _encode_and_write(
            transaction,
            entries,
            all_texts,
            implementations,
            bundle_implementations,
            source_universe,
            head,
            repo_root,
            encoder_checkpoint,
            expected_encoder_digest,
            model,
            encoder_config,
            input_current,
        )
    except BaseException:
        transaction.purge()
        raise
    finally:
        transaction.close()


def _encode_and_write(
    transaction: _OutputTransaction,
    entries: list[dict[str, Any]],
    all_texts: dict[str, str],
    implementations: dict[str, Any],
    bundle_implementations: dict[str, Any],
    source_universe: SourceUniverseArtifact,
    head: str,
    repo_root: Path,
    encoder_checkpoint: Path,
    expected_encoder_digest: str,
    model: ModelConfig,
    encoder_config: EncoderConfig,
    input_current: float,
) -> CueSetWriteResult:
    if not encoder_checkpoint.is_dir():
        raise CueMaterializerError("pinned encoder checkpoint directory is missing")
    live_digest = directory_digest(encoder_checkpoint)
    if live_digest != expected_encoder_digest:
        raise CueMaterializerError("pinned encoder directory digest mismatch")
    adapter = LocalSentenceEncoder(encoder_checkpoint)
    encoder_block = {
        "identity": encoder_checkpoint.name,
        "directory_sha256": live_digest,
        "config_digest": canonical_json_digest(_config_mapping(encoder_config)),
        "config": _config_mapping(encoder_config),
    }
    model_block = {
        "config_digest": canonical_json_digest(_config_mapping(model)),
        "n_neurons": model.n_neurons,
        "input_current": float(input_current),
        "config": _config_mapping(model),
    }
    bundle_model_block = {
        "config_digest": model_block["config_digest"],
        "n_neurons": model.n_neurons,
        "input_current": float(input_current),
    }
    bundle_files: dict[str, bytes] = {}
    for entry in entries:
        for base_cue in entry["evaluation_base_cues"]:
            for variant in base_cue["variants"]:
                text = all_texts[variant["path"]]
                arrays = _bundle_arrays(adapter, model, encoder_config, input_current, text)
                payload = _bundle_payload(
                    cast(str, variant["variant_id"]),
                    text,
                    encoder_block,
                    bundle_model_block,
                    bundle_implementations,
                    arrays,
                )
                raw = _canonical(payload)
                bundle_path = f"bundles/{variant['variant_id']}.json"
                bundle_files[bundle_path] = raw
                variant["bundle"] = {
                    "path": bundle_path,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "snn-memory-cue-set",
        "cue_set_version": CUE_SET_VERSION,
        "source_universe": {
            "file_sha256": source_universe.file_sha256,
            "payload_self_sha256": source_universe.payload_self_sha256,
            "repository_head": head,
            "selected_record_ids": list(source_universe.payload["selected_record_ids"]),
        },
        "derivation": _derivation_block(),
        "tokenizer": {
            "identity": TOKENIZER_IDENTITY,
            "rule": TOKENIZER_RULE,
            "sha256": TOKENIZER_SHA256,
        },
        "cue_contract": {
            "normalizer": "unicode-nfkc-whitespace-v1",
            "near_duplicate_metric": "token-jaccard-v1",
            "near_duplicate_maximum": NEAR_DUPLICATE_MAXIMUM,
        },
        "noise_lexicon": {
            "identity": NOISE_LEXICON_IDENTITY,
            "path": NOISE_LEXICON_NAME,
            "sha256": NOISE_LEXICON_SHA256,
            "token_count": len(NOISE_LEXICON),
        },
        "encoder": encoder_block,
        "model": model_block,
        "implementations": implementations,
        "records": entries,
    }
    manifest["self_sha256"] = _self_digest(manifest, _CUE_SET_SELF_DOMAIN)
    manifest_raw = _canonical(manifest)
    transaction.write(NOISE_LEXICON_NAME, NOISE_LEXICON_BYTES)
    for relative, text in all_texts.items():
        transaction.write(relative, text.encode("utf-8"))
    for relative, raw in bundle_files.items():
        transaction.write(relative, raw)
    transaction.write(CUE_SET_NAME, manifest_raw)
    # Validate the manifest and every referenced artifact through the captured
    # root fd, not the declared pathname, so a concurrent rename/substitution
    # cannot feed a foreign tree into validation. Then reassert that the declared
    # pathname still binds the captured inode before returning it, so success is
    # never claimed for a hijacked or prepopulated substitute.
    file_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    artifact = read_cue_set(transaction.fd_root() / CUE_SET_NAME, file_sha256)
    verify_cue_set_against_sources(artifact, source_universe, repo_root)
    transaction.assert_bound()
    return CueSetWriteResult(
        transaction.path,
        transaction.path / CUE_SET_NAME,
        artifact.file_sha256,
        artifact.payload_self_sha256,
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _strip_bundle_refs(record: dict[str, Any]) -> dict[str, Any]:
    for base_cue in record["evaluation_base_cues"]:
        for variant in base_cue["variants"]:
            variant.pop("bundle")
    return record


def _validate_record(record: dict[str, Any], registry: dict[str, set[str]]) -> None:
    record_hash = cast(str, record["record_id"]).removeprefix("sha256:")
    event_count = _require_int(record["event_count"], "record event count")
    boundaries = _block_boundaries(event_count)
    blocks = [
        list(range(boundaries[index], boundaries[index + 1])) for index in range(BLOCK_COUNT)
    ]
    calibration_index = _calibration_block_index(record_hash)
    if record["calibration_block_index"] != calibration_index:
        raise CueMaterializerError("calibration block index differs from its record digest")
    permutation_index = _family_permutation_index(record_hash)
    if record["family_permutation_index"] != permutation_index:
        raise CueMaterializerError("family permutation index differs from its record digest")
    calibration = record["calibration_cue"]
    for leak_value in (
        calibration["cue_id"],
        calibration["path"],
        *(base["cue_id"] for base in record["evaluation_base_cues"]),
    ):
        if record_hash[:32] in cast(str, leak_value):
            raise CueMaterializerError("cue identity leaks its record ID")
    if calibration["event_indices"] != blocks[calibration_index]:
        raise CueMaterializerError("calibration event indices differ from the frozen partition")
    if calibration["cue_id"] != _cue_id(cast(str, calibration["sha256"])):
        raise CueMaterializerError("calibration cue ID differs from its derivation")
    if calibration["path"] != f"cues/{calibration['cue_id']}.txt":
        raise CueMaterializerError("calibration path differs from its opaque cue ID")
    _register(registry, "cue_ids", cast(str, calibration["cue_id"]), "cue ID")
    _register(registry, "text_sha", cast(str, calibration["sha256"]), "cue text digest")
    _register(
        registry,
        "normalized_sha",
        cast(str, calibration["normalized_text_sha256"]),
        "normalized cue digest",
    )
    evaluation_blocks = [
        blocks[index] for index in range(BLOCK_COUNT) if index != calibration_index
    ]
    families = _FAMILY_PERMUTATIONS[permutation_index]
    for base_cue, family, indices in zip(
        record["evaluation_base_cues"], families, evaluation_blocks, strict=True
    ):
        if base_cue["transform_family"] != family:
            raise CueMaterializerError("transform family differs from the frozen permutation")
        if base_cue["event_indices"] != indices:
            raise CueMaterializerError("evaluation event indices differ from the partition")
        base_sha = cast(str, base_cue["variants"][0]["sha256"])
        if base_cue["cue_id"] != _cue_id(base_sha):
            raise CueMaterializerError("base cue ID differs from its derivation")
        _register(registry, "cue_ids", cast(str, base_cue["cue_id"]), "cue ID")
        token_count = _require_int(base_cue["token_count"], "base cue token count")
        for variant in base_cue["variants"]:
            _validate_variant(variant, family, base_sha, token_count, record_hash, registry)


def _validate_variant(
    variant: dict[str, Any],
    family: str,
    base_sha: str,
    token_count: int,
    record_hash: str,
    registry: dict[str, set[str]],
) -> None:
    percent = cast(int, variant["requested_percent"])
    variant_id = cast(str, variant["variant_id"])
    if record_hash[:32] in variant_id or record_hash[:32] in cast(str, variant["path"]):
        raise CueMaterializerError("variant identity leaks its record ID")
    if variant_id != _variant_id(base_sha, family, percent):
        raise CueMaterializerError("variant ID differs from its derivation")
    if variant["path"] != f"cues/{variant_id}.txt":
        raise CueMaterializerError("variant path differs from its opaque variant ID")
    if variant["bundle"]["path"] != f"bundles/{variant_id}.json":
        raise CueMaterializerError("bundle path differs from its opaque variant ID")
    selected = _selected_positions(family, base_sha, token_count, percent)
    if variant["selected_positions"] != selected:
        raise CueMaterializerError("selected positions differ from the frozen derivation")
    if variant["affected_count"] != len(selected):
        raise CueMaterializerError("affected count differs from the frozen rounding rule")
    if variant["realized_fraction"] != len(selected) / token_count:
        raise CueMaterializerError("realized fraction differs from its counts")
    _register(registry, "variant_ids", variant_id, "variant ID")
    _register(registry, "text_sha", cast(str, variant["sha256"]), "cue text digest")
    _register(
        registry,
        "normalized_sha",
        cast(str, variant["normalized_text_sha256"]),
        "normalized cue digest",
    )
    _register(registry, "bundle_sha", cast(str, variant["bundle"]["sha256"]), "bundle digest")


def _register(registry: dict[str, set[str]], group: str, value: str, label: str) -> None:
    if value in registry[group]:
        raise CueMaterializerError(f"duplicate {label} across the cue set")
    registry[group].add(value)


def validate_cue_set_bytes(
    raw: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> CueSetArtifact:
    """Validate canonical cue-set bytes and every manifest-provable binding."""
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise CueMaterializerError("cue-set file SHA-256 mismatch")
    payload = _strict_json(raw, "cue-set artifact")
    _validate_schema(payload, _CUE_SET_SCHEMA_NAME, "cue-set")
    if _canonical(payload) != raw:
        raise CueMaterializerError("cue-set JSON is not canonical")
    if payload["self_sha256"] != _self_digest(payload, _CUE_SET_SELF_DOMAIN):
        raise CueMaterializerError("cue-set self digest mismatch")
    selected_ids = payload["source_universe"]["selected_record_ids"]
    if [record["record_id"] for record in payload["records"]] != selected_ids:
        raise CueMaterializerError("record order differs from the selected record IDs")
    encoder_block = payload["encoder"]
    if encoder_block["config_digest"] != canonical_json_digest(encoder_block["config"]):
        raise CueMaterializerError("cue-set encoder configuration digest mismatch")
    _config_from_mapping(encoder_block["config"], EncoderConfig, "cue-set encoder")
    model_block = payload["model"]
    if model_block["config_digest"] != canonical_json_digest(model_block["config"]):
        raise CueMaterializerError("cue-set model configuration digest mismatch")
    model_config = cast(
        ModelConfig, _config_from_mapping(model_block["config"], ModelConfig, "cue-set model")
    )
    if _require_int(model_block["n_neurons"], "cue-set neuron count") != model_config.n_neurons:
        raise CueMaterializerError("cue-set model neuron count differs from its configuration")
    _validate_implementations(payload)
    registry: dict[str, set[str]] = {
        "cue_ids": set(),
        "variant_ids": set(),
        "text_sha": set(),
        "normalized_sha": set(),
        "bundle_sha": set(),
    }
    for record in payload["records"]:
        _validate_record(record, registry)
    return CueSetArtifact(
        _frozen(payload), raw, file_sha256, cast(str, payload["self_sha256"])
    )


def _validate_implementations(payload: dict[str, Any]) -> None:
    implementations = payload["implementations"]
    for name in ("cue_materializer", "split_events", "sentence_encoder"):
        implementation = implementations[name]
        try:
            source = base64.b64decode(implementation["bytes_base64"], validate=True)
        except (ValueError, TypeError) as error:
            raise CueMaterializerError("implementation source base64 is invalid") from error
        if (
            len(source) != implementation["byte_count"]
            or hashlib.sha256(source).hexdigest() != implementation["sha256"]
        ):
            raise CueMaterializerError("implementation source byte binding differs")


def _referenced_cue_text(
    root: Path, relative: str, expected_sha256: str, expected_normalized: str, context: str
) -> str:
    raw = _read_referenced_bytes(root, relative, expected_sha256, context)
    text = _decode_utf8(raw, context)
    if _normalized_sha256(text) != expected_normalized:
        raise CueMaterializerError(f"{context} normalized-text digest mismatch")
    return text


def read_cue_set(cue_set_path: Path, expected_file_sha256: str) -> CueSetArtifact:
    """Read and authenticate the manifest plus every referenced artifact file."""
    cue_set_path = cue_set_path.absolute()
    raw = _read_regular_bytes(cue_set_path, "cue-set file")
    artifact = validate_cue_set_bytes(raw, expected_file_sha256=expected_file_sha256)
    root = cue_set_path.parent
    payload = artifact.payload
    _read_referenced_bytes(
        root,
        cast(str, payload["noise_lexicon"]["path"]),
        cast(str, payload["noise_lexicon"]["sha256"]),
        "noise lexicon",
    )
    anchors: list[tuple[str, str]] = []
    manifest_encoder = _thaw(payload["encoder"])
    manifest_model = {
        "config_digest": payload["model"]["config_digest"],
        "n_neurons": payload["model"]["n_neurons"],
        "input_current": payload["model"]["input_current"],
    }
    manifest_implementations = {
        name: {
            "logical_path": block["logical_path"],
            "sha256": block["sha256"],
        }
        for name, block in payload["implementations"].items()
    }
    for record in payload["records"]:
        calibration = record["calibration_cue"]
        calibration_text = _referenced_cue_text(
            root,
            cast(str, calibration["path"]),
            cast(str, calibration["sha256"]),
            cast(str, calibration["normalized_text_sha256"]),
            "calibration cue",
        )
        _require_canonical_cue_text(calibration_text, "calibration")
        _require_line_count(
            calibration_text, cast("Sequence[int]", calibration["event_indices"]), "calibration"
        )
        anchors.append((cast(str, calibration["cue_id"]), calibration_text))
        for base_cue in record["evaluation_base_cues"]:
            family = cast(str, base_cue["transform_family"])
            zero_variant = base_cue["variants"][0]
            base_text = _referenced_cue_text(
                root,
                cast(str, zero_variant["path"]),
                cast(str, zero_variant["sha256"]),
                cast(str, zero_variant["normalized_text_sha256"]),
                "base cue",
            )
            _require_canonical_cue_text(base_text, "base")
            _require_line_count(
                base_text, cast("Sequence[int]", base_cue["event_indices"]), "base"
            )
            lines = _line_tokens(base_text)
            if sum(len(line) for line in lines) != base_cue["token_count"]:
                raise CueMaterializerError("base cue token count differs from its text")
            anchors.append((cast(str, base_cue["cue_id"]), base_text))
            base_sha = cast(str, zero_variant["sha256"])
            for variant in base_cue["variants"][1:]:
                derived = _apply_transform(
                    family,
                    lines,
                    cast(Sequence[int], variant["selected_positions"]),
                    base_sha,
                    NOISE_LEXICON,
                )
                stored = _referenced_cue_text(
                    root,
                    cast(str, variant["path"]),
                    cast(str, variant["sha256"]),
                    cast(str, variant["normalized_text_sha256"]),
                    "variant cue",
                )
                if stored != derived:
                    raise CueMaterializerError("variant cue text differs from its derivation")
    for index, (left_id, left_text) in enumerate(anchors):
        for right_id, right_text in anchors[index + 1 :]:
            if token_jaccard(left_text, right_text) > NEAR_DUPLICATE_MAXIMUM:
                raise CueMaterializerError(f"near-duplicate cues: {left_id} and {right_id}")
    for record in payload["records"]:
        for base_cue in record["evaluation_base_cues"]:
            for variant in base_cue["variants"]:
                bundle_raw = _read_referenced_bytes(
                    root,
                    cast(str, variant["bundle"]["path"]),
                    cast(str, variant["bundle"]["sha256"]),
                    "cue bundle",
                )
                bundle = validate_cue_bundle_bytes(bundle_raw)
                if bundle.payload["cue_id"] != variant["variant_id"]:
                    raise CueMaterializerError("bundle cue ID differs from its variant")
                if bundle.payload["text_sha256"] != variant["sha256"]:
                    raise CueMaterializerError("bundle text differs from its variant cue")
                if _thaw(bundle.payload["encoder"]) != manifest_encoder:
                    raise CueMaterializerError("bundle encoder differs from the cue set")
                if _thaw(bundle.payload["model"]) != manifest_model:
                    raise CueMaterializerError("bundle model differs from the cue set")
                if _thaw(bundle.payload["implementations"]) != manifest_implementations:
                    raise CueMaterializerError("bundle implementations differ from the cue set")
    return artifact


def verify_cue_set_against_sources(
    cue_set: CueSetArtifact,
    source_universe: SourceUniverseArtifact,
    repo_root: Path,
) -> None:
    """Re-derive every cue text from the live repository and the D1 artifact."""
    binding = cue_set.payload["source_universe"]
    if (
        binding["file_sha256"] != source_universe.file_sha256
        or binding["payload_self_sha256"] != source_universe.payload_self_sha256
    ):
        raise CueMaterializerError("cue set is bound to a different source-universe artifact")
    if binding["repository_head"] != source_universe.payload["repository"]["head"]:
        raise CueMaterializerError("cue-set repository head differs from the artifact")
    head = _repository_head(repo_root.absolute())
    if head != binding["repository_head"]:
        raise CueMaterializerError("repository HEAD differs from the cue-set binding")
    for record, item in zip(
        cue_set.payload["records"], source_universe.payload["selected"], strict=True
    ):
        events = _reauthenticated_events(repo_root.absolute(), item)
        expected_entry, _texts = _record_entry(item, events, NOISE_LEXICON)
        actual_entry = _strip_bundle_refs(cast(dict[str, Any], _thaw(record)))
        if actual_entry != expected_entry:
            raise CueMaterializerError("record cue derivation differs from its sources")


def _load_config_file(
    path: Path | None, config_type: type[ModelConfig] | type[EncoderConfig], context: str
) -> ModelConfig | EncoderConfig:
    if path is None:
        return config_type()
    mapping = _strict_json(_read_regular_bytes(path, context), context)
    return _config_from_mapping(mapping, config_type, context)


def _run_materialize(arguments: argparse.Namespace) -> dict[str, Any]:
    model = cast(ModelConfig, _load_config_file(arguments.model_config, ModelConfig, "model"))
    encoder_config = cast(
        EncoderConfig, _load_config_file(arguments.encoder_config, EncoderConfig, "encoder")
    )
    result = materialize_cue_set(
        arguments.repo_root,
        arguments.source_universe,
        arguments.source_universe_sha256,
        arguments.encoder_checkpoint,
        arguments.encoder_digest,
        arguments.output_dir,
        model=model,
        encoder_config=encoder_config,
        input_current=arguments.input_current,
    )
    return {
        "cue_set_path": str(result.cue_set_path),
        "file_sha256": result.file_sha256,
        "payload_self_sha256": result.payload_self_sha256,
    }


def _run_validate_set(arguments: argparse.Namespace) -> dict[str, Any]:
    source_arguments = (
        arguments.repo_root,
        arguments.source_universe,
        arguments.source_universe_sha256,
    )
    provided = [value for value in source_arguments if value is not None]
    if provided and len(provided) != len(source_arguments):
        raise CueMaterializerError(
            "source verification needs repo root, artifact path and artifact digest together"
        )
    artifact = read_cue_set(arguments.cue_set, arguments.cue_set_sha256)
    source_verified = False
    if provided:
        source_raw = _read_regular_bytes(arguments.source_universe, "source-universe file")
        source_universe = validate_source_universe_bytes(
            source_raw, expected_file_sha256=arguments.source_universe_sha256
        )
        verify_cue_set_against_sources(artifact, source_universe, arguments.repo_root)
        source_verified = True
    return {
        "file_sha256": artifact.file_sha256,
        "payload_self_sha256": artifact.payload_self_sha256,
        "source_verified": source_verified,
        "status": "pass",
    }


def _run_validate_bundle(arguments: argparse.Namespace) -> dict[str, Any]:
    encoder_arguments = (arguments.encoder_checkpoint, arguments.encoder_digest)
    provided = [value for value in encoder_arguments if value is not None]
    if len(provided) == 1:
        raise CueMaterializerError(
            "encoder verification needs the checkpoint path and digest together"
        )
    bundle = read_cue_bundle(arguments.bundle, arguments.bundle_sha256)
    encoder_verified = False
    if provided:
        verify_cue_bundle_with_encoder(
            bundle, arguments.encoder_checkpoint, arguments.encoder_digest
        )
        encoder_verified = True
    return {
        "cue_id": cast(str, bundle.payload["cue_id"]),
        "encoder_verified": encoder_verified,
        "file_sha256": bundle.file_sha256,
        "payload_self_sha256": bundle.payload_self_sha256,
        "status": "pass",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cue-materialiser CLI and return a process exit code."""
    parser = argparse.ArgumentParser(prog="remanentia-snn-cue-materializer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--repo-root", type=Path, required=True)
    materialize.add_argument("--source-universe", type=Path, required=True)
    materialize.add_argument("--source-universe-sha256", required=True)
    materialize.add_argument("--encoder-checkpoint", type=Path, required=True)
    materialize.add_argument("--encoder-digest", required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--model-config", type=Path)
    materialize.add_argument("--encoder-config", type=Path)
    materialize.add_argument("--input-current", type=float, default=18.0)
    materialize.set_defaults(handler=_run_materialize)
    validate_set = subparsers.add_parser("validate-set")
    validate_set.add_argument("--cue-set", type=Path, required=True)
    validate_set.add_argument("--cue-set-sha256", required=True)
    validate_set.add_argument("--repo-root", type=Path)
    validate_set.add_argument("--source-universe", type=Path)
    validate_set.add_argument("--source-universe-sha256")
    validate_set.set_defaults(handler=_run_validate_set)
    validate_bundle = subparsers.add_parser("validate-bundle")
    validate_bundle.add_argument("--bundle", type=Path, required=True)
    validate_bundle.add_argument("--bundle-sha256")
    validate_bundle.add_argument("--encoder-checkpoint", type=Path)
    validate_bundle.add_argument("--encoder-digest")
    validate_bundle.set_defaults(handler=_run_validate_bundle)
    arguments = parser.parse_args(argv)
    try:
        report = arguments.handler(arguments)
    except (OSError, SourceUniverseError, CueMaterializerError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
