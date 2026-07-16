# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Deterministic schema-v2 source-universe selector

"""Select the schema-v2 Markdown source universe from a real Git repository.

The selector consumes only committed state: every considered Markdown path must
have identical HEAD, index, and worktree identity, read through NUL-safe and
option-safe Git plumbing. Regular blobs are compared byte-for-byte through
symlink-refusing descriptors; a tracked symlink must carry exactly its
committed target bytes and is never dereferenced; a tracked gitlink must sit
exactly on the committed commit when its working directory is a repository of
its own, an absent working directory is a divergence (matching ``git
status``), and a present directory without its own repository is the
uninitialised-submodule state Git itself treats as clean. Selection is
deterministic from path and content digests alone.

``validate_source_universe_bytes`` re-derives every binding an artifact can
prove against itself and fails closed on any disagreement: canonical bytes,
the domain-separated self digest, strict schema, path normalisation, exact
audit field sets per exclusion class, reason recomputation, manifest label and
declared-path uniqueness with declared-to-resolved re-resolution, manifest
source digests bound to the audit, selection keys, record IDs, ordering, and
the eligible/selected lists. Offline validation cannot re-read repository
content, so these fields remain selection-time claims sealed only by the self
digest and the caller-supplied ``expected_file_sha256``: ``repository.head``,
every ``blob_oid``, content bindings (``content_sha256``, ``byte_count``,
``event_count``, ``event_order_digest``, ``event_sha256``), commit provenance
(``latest_path_commit``, ``timestamp_ns``), the manifest file ``sha256`` and
``blob_oid``, and a ``generated`` reason claimed for a first-line marker.
Authenticity of those claims is anchored at selection time against the live
repository and must be pinned externally through the artifact file hash.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from importlib import import_module
from importlib.machinery import SourceFileLoader
from importlib.resources import files
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Mapping, Sequence, cast

from snn_memory import encoder

SCHEMA_VERSION = 2
SELECTOR_VERSION = 1
MIN_BYTES = 1_000
MAX_BYTES = 20_000
MIN_EVENTS = 50
MAX_EVENTS = 256
SELECTED_COUNT = 16
SELECTION_DOMAIN = b"remanentia-snn-v2-lock\0"
SELF_DOMAIN = b"remanentia:snn-v2-source-universe:v1\0"
EVENT_DOMAIN = b"remanentia:snn-v2-event-order:v1\0"
MANIFEST_PATHS = (
    "experiments/snn_memory/development_corpus.json",
    "experiments/snn_memory/locked_evaluation_corpus.json",
)
SELECTOR_LOGICAL_PATH = "snn_memory/source_universe.py"
SPLIT_EVENTS_LOGICAL_PATH = "snn_memory/encoder.py"
_SCHEMA_NAME = "snn_memory_source_universe_v2.schema.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPOSITORY_IDENTITY = re.compile(rb"sha1\n([0-9a-f]{40})\n")
_LATEST_COMMIT_RECORD = re.compile(rb"([0-9a-f]{40})\x00([0-9]+)\x00\n")
_GITLINK_HEAD = re.compile(rb"([0-9a-f]{40})\n")
_REGULAR_MODES = frozenset({"100644", "100755"})
_FORBIDDEN_SEGMENTS = frozenset({"", ".", ".."})
_VENDOR_COMPONENTS = frozenset({"vendor", "vendors", "third_party", "third-party", "node_modules"})
_GENERATED_COMPONENTS = frozenset({"generated", "_generated", "build", "dist", "site"})
_GENERATED_MARKERS = frozenset(
    {
        "<!-- GENERATED FILE: DO NOT EDIT -->",
        "<!-- generated: do not edit -->",
    }
)
_LICENSE_NAMES = frozenset(
    {
        "license.md",
        "licence.md",
        "copying.md",
        "notice.md",
        "copyright.md",
        "authors.md",
    }
)
_INDEX_NAMES = frozenset({"index.md", "_index.md", "summary.md"})
_PATH_ONLY_REASONS = frozenset(
    {"internal", "coordination", "schema_v1_manifest", "vendor", "generated"}
)
_BASE_AUDIT_FIELDS = frozenset(
    {"path", "normalized_path", "status", "reason", "mode", "kind", "blob_oid"}
)
_CONTENT_AUDIT_FIELDS = _BASE_AUDIT_FIELDS | {
    "byte_count",
    "event_count",
    "content_sha256",
    "content_commit",
    "latest_path_commit",
    "timestamp_ns",
    "timestamp_source",
    "timestamp_precision",
    "event_order_digest",
}
_ELIGIBLE_AUDIT_FIELDS = _CONTENT_AUDIT_FIELDS | {
    "selection_key",
    "record_id",
    "event_sha256",
}


class SourceUniverseError(ValueError):
    """A fail-closed source-universe contract violation."""


@dataclass(frozen=True)
class SourceUniverseArtifact:
    """Validated canonical artifact with a recursively read-only payload view."""

    payload: Mapping[str, Any]
    canonical_bytes: bytes
    file_sha256: str
    payload_self_sha256: str


@dataclass(frozen=True)
class SourceUniverseWriteResult:
    """Authenticated result of one atomic no-clobber artifact write."""

    output_path: Path
    file_sha256: str
    payload_self_sha256: str


@dataclass(frozen=True)
class _GitEntry:
    mode: str
    kind: str
    oid: str
    path_bytes: bytes
    path: str
    normalized_path: str


def _run_git(root: Path, arguments: list[str], *, input_bytes: bytes | None = None) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", "replace").strip()
        raise SourceUniverseError(f"git {' '.join(arguments[:2])} failed: {detail}")
    return process.stdout


def _expect_format(raw: bytes, pattern: re.Pattern[bytes], context: str) -> tuple[bytes, ...]:
    match = pattern.fullmatch(raw)
    if match is None:
        raise SourceUniverseError(f"{context} output is not in the expected plumbing format")
    return match.groups()


def _normalize_text(decoded: str, context: str) -> str:
    if (
        not decoded
        or decoded.startswith("/")
        or "\\" in decoded
        or "\x00" in decoded
        or frozenset(decoded.split("/")) & _FORBIDDEN_SEGMENTS
    ):
        raise SourceUniverseError(f"{context} is not canonical relative POSIX syntax")
    return "/".join(unicodedata.normalize("NFC", part) for part in decoded.split("/"))


def _normalize_path(raw: bytes) -> tuple[str, str]:
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SourceUniverseError("tracked path is not strict UTF-8") from error
    return decoded, _normalize_text(decoded, "tracked path")


def _parse_tree(raw: bytes) -> dict[bytes, tuple[str, str, str]]:
    entries: dict[bytes, tuple[str, str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3 or not path or path in entries:
            raise SourceUniverseError("malformed or duplicate Git tree record")
        mode, kind, oid = (field.decode("ascii", "strict") for field in fields)
        entries[path] = (mode, kind, oid)
    return entries


def _parse_index(raw: bytes) -> dict[bytes, tuple[str, str]]:
    entries: dict[bytes, tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3 or fields[2] != b"0" or not path or path in entries:
            raise SourceUniverseError("malformed, conflicted, or duplicate Git index record")
        entries[path] = (fields[0].decode("ascii", "strict"), fields[1].decode("ascii", "strict"))
    return entries


def _repository(root: Path) -> tuple[str, dict[bytes, _GitEntry]]:
    if not root.is_absolute() or root != root.resolve(strict=True):
        raise SourceUniverseError("repo_root must be an absolute canonical path")
    top = Path(_run_git(root, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip())
    if top.resolve(strict=True) != root:
        raise SourceUniverseError("repo_root is not the Git top level")
    identity = _run_git(root, ["rev-parse", "--show-object-format"]) + _run_git(
        root, ["rev-parse", "--verify", "HEAD^{commit}"]
    )
    (head_bytes,) = _expect_format(identity, _REPOSITORY_IDENTITY, "SHA-1 repository identity")
    head = head_bytes.decode("ascii")
    tree = _parse_tree(_run_git(root, ["ls-tree", "-rz", "--full-tree", "HEAD"]))
    index = _parse_index(_run_git(root, ["ls-files", "-s", "-z"]))
    if set(tree) != set(index):
        raise SourceUniverseError("HEAD and index tracked path sets differ")
    normalized_seen: dict[str, bytes] = {}
    entries: dict[bytes, _GitEntry] = {}
    for path_bytes in sorted(tree):
        mode, kind, oid = tree[path_bytes]
        if (mode, oid) != index[path_bytes]:
            raise SourceUniverseError("HEAD and index mode/blob identity differ")
        path, normalized = _normalize_path(path_bytes)
        previous = normalized_seen.get(normalized)
        if previous is not None and previous != path_bytes:
            raise SourceUniverseError("normalized tracked paths collide")
        normalized_seen[normalized] = path_bytes
        entries[path_bytes] = _GitEntry(mode, kind, oid, path_bytes, path, normalized)
    return head, entries


def _parent_walk(root: Path, entry: _GitEntry, kind_label: str) -> None:
    current = root
    for component in entry.path.split("/")[:-1]:
        current = current / component
        if stat.S_ISLNK(current.lstat().st_mode):
            raise SourceUniverseError(f"{kind_label} tracked path traverses a symlink")


def _regular_bytes(root: Path, entry: _GitEntry) -> bytes:
    if entry.mode not in _REGULAR_MODES or entry.kind != "blob":
        raise SourceUniverseError("requested Git entry is not a regular blob")
    try:
        _parent_walk(root, entry, "regular")
        descriptor = os.open(root / entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise SourceUniverseError("tracked regular file cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SourceUniverseError("tracked regular file changed type in worktree")
        if bool(metadata.st_mode & 0o111) != (entry.mode == "100755"):
            raise SourceUniverseError("tracked regular file mode differs in worktree")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
    finally:
        os.close(descriptor)
    worktree_oid = (
        _run_git(root, ["hash-object", "--stdin"], input_bytes=content).decode("ascii").strip()
    )
    if worktree_oid != entry.oid:
        raise SourceUniverseError("HEAD, index, and worktree bytes differ")
    return content


def _gitlink_worktree(root: Path, entry: _GitEntry) -> None:
    worktree = root / entry.path
    try:
        _parent_walk(root, entry, "non-regular")
        metadata = os.lstat(worktree)
    except FileNotFoundError as error:
        raise SourceUniverseError("HEAD, index, and worktree gitlink identity differ") from error
    except OSError as error:
        raise SourceUniverseError("tracked gitlink cannot be inspected safely") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise SourceUniverseError("tracked gitlink worktree entry is not a directory")
    toplevel = Path(_run_git(worktree, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip())
    if toplevel.resolve() != worktree:
        return
    raw = _run_git(worktree, ["rev-parse", "--verify", "HEAD^{commit}"])
    (head_bytes,) = _expect_format(raw, _GITLINK_HEAD, "gitlink worktree HEAD")
    if head_bytes.decode("ascii") != entry.oid:
        raise SourceUniverseError("HEAD, index, and worktree gitlink identity differ")


def _non_regular_worktree(root: Path, entry: _GitEntry) -> None:
    if entry.kind == "commit":
        _gitlink_worktree(root, entry)
        return
    try:
        _parent_walk(root, entry, "non-regular")
        target = os.readlink(root / entry.path)
    except OSError as error:
        raise SourceUniverseError("tracked symlink cannot be read safely") from error
    worktree_oid = (
        _run_git(root, ["hash-object", "--stdin"], input_bytes=os.fsencode(target))
        .decode("ascii")
        .strip()
    )
    if worktree_oid != entry.oid:
        raise SourceUniverseError("HEAD, index, and worktree symlink identity differ")


def _latest_commit(root: Path, entry: _GitEntry, head: str) -> tuple[str, int]:
    raw = _run_git(
        root,
        ["log", "-1", "--format=%H%x00%ct%x00", head, "--", f":(literal){entry.path}"],
    )
    commit_bytes, seconds_bytes = _expect_format(raw, _LATEST_COMMIT_RECORD, "latest path commit")
    commit = commit_bytes.decode("ascii")
    _run_git(root, ["merge-base", "--is-ancestor", commit, head])
    return commit, int(seconds_bytes)


def _strict_json(raw: bytes, context: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SourceUniverseError(f"{context} contains a duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(constant: str) -> None:
        raise SourceUniverseError(f"{context} contains non-finite JSON constant {constant}")

    def parse_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise SourceUniverseError(f"{context} contains a non-finite JSON number")
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, SourceUniverseError):
            raise
        raise SourceUniverseError(f"{context} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SourceUniverseError(f"{context} root must be an object")
    return value


def _resolve_manifest_source(manifest_path: str, relative: str) -> str:
    if relative.startswith("/") or "\\" in relative:
        raise SourceUniverseError("schema-v1 corpus path is not relative POSIX syntax")
    parts = list(PurePosixPath(manifest_path).parent.parts)
    for part in relative.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise SourceUniverseError("schema-v1 corpus path escapes the repository")
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _manifest_sources(
    root: Path,
    entries: dict[bytes, _GitEntry],
) -> tuple[set[str], list[dict[str, Any]]]:
    excluded: set[str] = set()
    manifests: list[dict[str, Any]] = []
    for index, manifest_path in enumerate(MANIFEST_PATHS):
        entry = entries.get(manifest_path.encode("utf-8"))
        if entry is None:
            raise SourceUniverseError("schema-v1 corpus manifest must be a committed regular file")
        raw = _regular_bytes(root, entry)
        value = _strict_json(raw, manifest_path)
        expected_keys = {
            "schema_version",
            "split",
            "encoder_checkpoint",
            "encoder_digest",
            "entries",
        }
        if index == 1:
            expected_keys.add("locked")
        expected_split = "development" if index == 0 else "locked-evaluation"
        if set(value) != expected_keys or value.get("schema_version") != 1:
            raise SourceUniverseError("schema-v1 corpus manifest has unexpected fields or version")
        if value.get("split") != expected_split or (index == 1 and value.get("locked") is not True):
            raise SourceUniverseError("schema-v1 corpus manifest split/lock contract differs")
        if not isinstance(value.get("encoder_checkpoint"), str) or not isinstance(
            value.get("encoder_digest"), str
        ):
            raise SourceUniverseError("schema-v1 corpus manifest encoder fields are invalid")
        if not _SHA256.fullmatch(value["encoder_digest"]):
            raise SourceUniverseError("schema-v1 encoder digest is invalid")
        source_entries = value.get("entries")
        if not isinstance(source_entries, list):
            raise SourceUniverseError("schema-v1 corpus entries must be a list")
        seen_labels: set[str] = set()
        seen_paths: set[str] = set()
        resolved_sources: list[dict[str, str]] = []
        for source in source_entries:
            if not isinstance(source, dict) or set(source) != {"label", "path", "sha256"}:
                raise SourceUniverseError("schema-v1 corpus entry schema differs")
            label, relative, digest = source["label"], source["path"], source["sha256"]
            if (
                not isinstance(label, str)
                or not label
                or label != label.strip()
                or label in seen_labels
                or not isinstance(relative, str)
                or relative in seen_paths
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                raise SourceUniverseError("schema-v1 corpus entry identity is invalid or duplicate")
            seen_labels.add(label)
            seen_paths.add(relative)
            repo_path = _resolve_manifest_source(manifest_path, relative)
            if repo_path in excluded:
                raise SourceUniverseError("schema-v1 corpus sources resolve to a duplicate path")
            source_entry = entries.get(repo_path.encode("utf-8"))
            if source_entry is None or source_entry.normalized_path != repo_path:
                raise SourceUniverseError("schema-v1 corpus source is not a tracked canonical path")
            if not repo_path.endswith(".md"):
                raise SourceUniverseError("schema-v1 corpus source is not Markdown")
            source_raw = _regular_bytes(root, source_entry)
            if hashlib.sha256(source_raw).hexdigest() != digest:
                raise SourceUniverseError("schema-v1 corpus source digest mismatch")
            excluded.add(repo_path)
            resolved_sources.append(
                {
                    "label": label,
                    "declared_path": relative,
                    "resolved_path": repo_path,
                    "sha256": digest,
                }
            )
        manifests.append(
            {
                "path": manifest_path,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "blob_oid": entry.oid,
                "sources": resolved_sources,
            }
        )
    return excluded, manifests


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
        raise SourceUniverseError("implementation module is not a real non-shadowed source file")
    current = Path(origin_text).absolute()
    for candidate in (current, *current.parents):
        if candidate.is_symlink():
            raise SourceUniverseError("implementation source traverses a symlink")
    try:
        raw = current.resolve(strict=True).read_bytes()
    except OSError as error:
        raise SourceUniverseError("implementation source cannot be read") from error
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SourceUniverseError("implementation source is not strict UTF-8") from error
    return {
        "logical_path": logical_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "bytes_base64": base64.b64encode(raw).decode("ascii"),
    }


def _event_order(events: list[str]) -> tuple[str, list[str]]:
    framed = bytearray(EVENT_DOMAIN)
    framed.extend(len(events).to_bytes(8, "big"))
    hashes: list[str] = []
    for event in events:
        raw = event.encode("utf-8")
        framed.extend(len(raw).to_bytes(8, "big"))
        framed.extend(raw)
        hashes.append(hashlib.sha256(raw).hexdigest())
    return hashlib.sha256(bytes(framed)).hexdigest(), hashes


def _path_reason(path: str, v1_paths: set[str]) -> str | None:
    folded = frozenset(component.casefold() for component in path.split("/"))
    basename = path.rsplit("/", 1)[-1].casefold()
    if path.startswith("docs/internal/"):
        return "internal"
    if path.startswith(".coordination/"):
        return "coordination"
    if path in v1_paths:
        return "schema_v1_manifest"
    if _VENDOR_COMPONENTS & folded:
        return "vendor"
    if (_GENERATED_COMPONENTS & folded) or basename.endswith(".generated.md"):
        return "generated"
    if basename in _LICENSE_NAMES:
        return "license_only"
    if basename in _INDEX_NAMES:
        return "index_only"
    return None


def _bounds_reason(byte_count: int, event_count: int) -> str:
    if byte_count == 0:
        return "empty"
    if byte_count < MIN_BYTES:
        return "bytes_below_min"
    if byte_count > MAX_BYTES:
        return "bytes_above_max"
    if event_count < MIN_EVENTS:
        return "events_below_min"
    if event_count > MAX_EVENTS:
        return "events_above_max"
    return "eligible"


def _reason(path: str, raw: bytes, text: str, event_count: int, v1_paths: set[str]) -> str:
    path_based = _path_reason(path, v1_paths)
    if path_based in _PATH_ONLY_REASONS:
        return path_based
    lines = text.splitlines()
    if (lines[0] if lines else "") in _GENERATED_MARKERS:
        return "generated"
    if path_based is not None:
        return path_based
    return _bounds_reason(len(raw), event_count)


def _canonical(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _self_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    canonical = _canonical(unsigned)
    framed = SELF_DOMAIN + len(canonical).to_bytes(8, "big") + canonical
    return hashlib.sha256(framed).hexdigest()


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _schema_bytes() -> bytes:
    return files("snn_memory").joinpath("schema", _SCHEMA_NAME).read_bytes()


def _selection_key(normalized_path: str, content_sha256: str) -> str:
    material = (
        SELECTION_DOMAIN + normalized_path.encode("utf-8") + b"\0" + content_sha256.encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()


def _validate_implementations(payload: dict[str, Any]) -> None:
    implementations = payload["implementations"]
    for implementation in (implementations["selector"], implementations["split_events"]):
        try:
            source = base64.b64decode(implementation["bytes_base64"], validate=True)
        except (ValueError, TypeError) as error:
            raise SourceUniverseError("implementation source base64 is invalid") from error
        if (
            len(source) != implementation["byte_count"]
            or hashlib.sha256(source).hexdigest() != implementation["sha256"]
        ):
            raise SourceUniverseError("implementation source byte binding differs")


def _validate_manifest_bindings(
    payload: dict[str, Any], considered_by_path: dict[str, dict[str, Any]]
) -> set[str]:
    manifests = payload["schema_v1_manifests"]
    if [item["path"] for item in manifests] != list(MANIFEST_PATHS):
        raise SourceUniverseError("schema-v1 manifest order/path differs")
    v1_paths: set[str] = set()
    for manifest in manifests:
        seen_labels: set[str] = set()
        seen_declared: set[str] = set()
        for source in manifest["sources"]:
            label, declared, resolved = (
                source["label"],
                source["declared_path"],
                source["resolved_path"],
            )
            if label != label.strip() or label in seen_labels:
                raise SourceUniverseError("schema-v1 manifest source label is invalid or duplicate")
            if declared in seen_declared:
                raise SourceUniverseError("schema-v1 manifest declared paths are duplicated")
            seen_labels.add(label)
            seen_declared.add(declared)
            if _resolve_manifest_source(manifest["path"], declared) != resolved:
                raise SourceUniverseError(
                    "schema-v1 declared path does not resolve to the resolved path"
                )
            if resolved in v1_paths:
                raise SourceUniverseError("schema-v1 resolved source paths are duplicated")
            v1_paths.add(resolved)
            audit = considered_by_path.get(resolved)
            if audit is None or audit.get("content_sha256") != source["sha256"]:
                raise SourceUniverseError("schema-v1 resolved source is not bound to the audit")
    return v1_paths


def _validate_audit_reason(item: dict[str, Any], v1_paths: set[str]) -> None:
    reason = item["reason"]
    path_based = _path_reason(item["normalized_path"], v1_paths)
    if path_based in _PATH_ONLY_REASONS:
        if reason != path_based:
            raise SourceUniverseError("considered reason differs from path-derived reason")
        return
    if reason == "generated":
        return
    expected = path_based or _bounds_reason(item["byte_count"], item["event_count"])
    if reason != expected:
        raise SourceUniverseError("considered reason differs from recomputed reason")


def _validate_audit_item(item: dict[str, Any], head: str, v1_paths: set[str]) -> None:
    if _normalize_text(item["path"], "considered path") != item["normalized_path"]:
        raise SourceUniverseError("considered path does not normalise to its normalized_path")
    is_regular = item["mode"] in _REGULAR_MODES and item["kind"] == "blob"
    if item["reason"] == "non_regular":
        if set(item) != _BASE_AUDIT_FIELDS or is_regular:
            raise SourceUniverseError("non-regular audit carries unexpected bindings")
        if item["status"] != "excluded":
            raise SourceUniverseError("non-regular audit status must be excluded")
        return
    expected_fields = (
        _ELIGIBLE_AUDIT_FIELDS if item["status"] == "eligible" else _CONTENT_AUDIT_FIELDS
    )
    if set(item) != expected_fields or not is_regular:
        raise SourceUniverseError("content audit fields differ from the exact contract")
    if item["content_commit"] != head:
        raise SourceUniverseError("content audit commit differs from repository HEAD")
    if item["timestamp_ns"] % 1_000_000_000 != 0:
        raise SourceUniverseError("seconds-precision timestamp is not whole seconds")
    if (item["status"] == "eligible") != (item["reason"] == "eligible"):
        raise SourceUniverseError("considered status and reason disagree")
    _validate_audit_reason(item, v1_paths)
    if item["status"] == "eligible":
        if item["record_id"] != f"sha256:{item['content_sha256']}":
            raise SourceUniverseError("eligible record ID differs from content digest")
        if item["selection_key"] != _selection_key(item["normalized_path"], item["content_sha256"]):
            raise SourceUniverseError("eligible selection key differs")
        if item["event_count"] != len(item["event_sha256"]):
            raise SourceUniverseError("eligible event count and event hashes differ")


def _semantic_validate(payload: dict[str, Any]) -> None:
    _validate_implementations(payload)
    head = payload["repository"]["head"]
    considered = payload["considered"]
    normalized_paths = [item["normalized_path"] for item in considered]
    if normalized_paths != sorted(normalized_paths) or len(set(normalized_paths)) != len(
        normalized_paths
    ):
        raise SourceUniverseError("considered paths are not sorted and unique")
    considered_by_path = {item["normalized_path"]: item for item in considered}
    v1_paths = _validate_manifest_bindings(payload, considered_by_path)
    eligible: list[dict[str, Any]] = []
    for item in considered:
        _validate_audit_item(item, head, v1_paths)
        if item["status"] == "eligible":
            eligible.append(item)
    eligible.sort(key=lambda item: cast(str, item["selection_key"]))
    if payload["eligible_record_ids"] != [item["record_id"] for item in eligible]:
        raise SourceUniverseError("eligible record list differs from considered audit")
    selected = payload["selected"]
    expected = eligible[:SELECTED_COUNT]
    if [item["rank"] for item in selected] != list(range(SELECTED_COUNT)):
        raise SourceUniverseError("selected ranks are not stable zero-based order")
    if payload["selected_record_ids"] != [item["record_id"] for item in expected]:
        raise SourceUniverseError("selected record ID list differs")
    if payload["selected_paths"] != [item["normalized_path"] for item in expected]:
        raise SourceUniverseError("selected path list differs")
    for actual, expected_item in zip(selected, expected, strict=True):
        without_rank = dict(actual)
        without_rank.pop("rank")
        if without_rank != expected_item:
            raise SourceUniverseError("selected record object differs from eligible audit")


def validate_source_universe_bytes(
    raw: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> SourceUniverseArtifact:
    """Validate canonical bytes, strict schema, self digest, and every offline-provable binding."""
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise SourceUniverseError("source-universe file SHA-256 mismatch")
    payload = _strict_json(raw, "source-universe artifact")
    schema = _strict_json(_schema_bytes(), "source-universe schema")
    validator_class = cast(Any, import_module("jsonschema")).Draft202012Validator
    try:
        validator_class(schema).validate(payload)
    except Exception as error:
        raise SourceUniverseError("source-universe schema validation failed") from error
    try:
        canonical = _canonical(payload)
    except ValueError as error:
        raise SourceUniverseError("source-universe serialization failed") from error
    if canonical != raw:
        raise SourceUniverseError("source-universe JSON is not canonical")
    if payload["self_sha256"] != _self_digest(payload):
        raise SourceUniverseError("source-universe self digest mismatch")
    _semantic_validate(payload)
    return SourceUniverseArtifact(
        _frozen(payload), raw, file_sha256, cast(str, payload["self_sha256"])
    )


def select_source_universe(repo_root: Path) -> SourceUniverseArtifact:
    """Select sixteen deterministic records from committed lower-case ``.md`` sources."""
    head, all_entries = _repository(repo_root)
    v1_paths, manifests = _manifest_sources(repo_root, all_entries)
    considered: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    markdown = [entry for entry in all_entries.values() if entry.path.endswith(".md")]
    for entry in sorted(markdown, key=lambda item: item.normalized_path):
        if entry.mode not in _REGULAR_MODES or entry.kind != "blob":
            _non_regular_worktree(repo_root, entry)
            considered.append(
                {
                    "path": entry.path,
                    "normalized_path": entry.normalized_path,
                    "status": "excluded",
                    "reason": "non_regular",
                    "mode": entry.mode,
                    "kind": entry.kind,
                    "blob_oid": entry.oid,
                }
            )
            continue
        raw = _regular_bytes(repo_root, entry)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SourceUniverseError("tracked Markdown content is not strict UTF-8") from error
        events = encoder.split_events(text)
        order_digest, event_hashes = _event_order(events)
        content_sha = hashlib.sha256(raw).hexdigest()
        latest_commit, seconds = _latest_commit(repo_root, entry, head)
        reason = _reason(entry.normalized_path, raw, text, len(events), v1_paths)
        audit: dict[str, Any] = {
            "path": entry.path,
            "normalized_path": entry.normalized_path,
            "status": "eligible" if reason == "eligible" else "excluded",
            "reason": reason,
            "mode": entry.mode,
            "kind": entry.kind,
            "byte_count": len(raw),
            "event_count": len(events),
            "content_sha256": content_sha,
            "blob_oid": entry.oid,
            "content_commit": head,
            "latest_path_commit": latest_commit,
            "timestamp_ns": seconds * 1_000_000_000,
            "timestamp_source": "git-commit",
            "timestamp_precision": "seconds",
            "event_order_digest": order_digest,
        }
        if reason == "eligible":
            audit["selection_key"] = _selection_key(entry.normalized_path, content_sha)
            audit["record_id"] = f"sha256:{content_sha}"
            audit["event_sha256"] = event_hashes
            eligible.append(audit)
        considered.append(audit)
    eligible.sort(key=lambda item: cast(str, item["selection_key"]))
    if len(eligible) < SELECTED_COUNT:
        raise SourceUniverseError("fewer than sixteen Markdown records are eligible")
    selected = eligible[:SELECTED_COUNT]
    record_ids = [str(item["record_id"]) for item in selected]
    if len(set(record_ids)) != SELECTED_COUNT:
        raise SourceUniverseError("selected content-addressed record IDs are not unique")
    selected_records = [{"rank": rank, **item} for rank, item in enumerate(selected)]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "snn-memory-source-universe",
        "selector_version": SELECTOR_VERSION,
        "repository": {"head": head, "object_format": "sha1"},
        "constants": {
            "selection_domain_ascii": "remanentia-snn-v2-lock\\0",
            "min_bytes": MIN_BYTES,
            "max_bytes": MAX_BYTES,
            "min_events": MIN_EVENTS,
            "max_events": MAX_EVENTS,
            "selected_count": SELECTED_COUNT,
            "path_normalization": "posix-segments-unicode-nfc-v1",
            "exclusion_rules": "remanentia-snn-v2-source-exclusions-v1",
        },
        "implementations": {
            "selector": _implementation(sys.modules[__name__], SELECTOR_LOGICAL_PATH),
            "split_events": _implementation(encoder, SPLIT_EVENTS_LOGICAL_PATH),
        },
        "schema_v1_manifests": manifests,
        "considered": considered,
        "eligible_record_ids": [str(item["record_id"]) for item in eligible],
        "selected_record_ids": record_ids,
        "selected_paths": [str(item["normalized_path"]) for item in selected],
        "selected": selected_records,
    }
    payload["self_sha256"] = _self_digest(payload)
    return validate_source_universe_bytes(_canonical(payload))


def write_source_universe(repo_root: Path, output: Path) -> SourceUniverseWriteResult:
    """Write one canonical artifact atomically without replacing any destination."""
    artifact = select_source_universe(repo_root)
    output = output.absolute()
    output.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(artifact.canonical_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, output)
        except OSError as error:
            raise SourceUniverseError("atomic no-clobber output link failed") from error
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    final_artifact = validate_source_universe_bytes(
        output.read_bytes(),
        expected_file_sha256=artifact.file_sha256,
    )
    return SourceUniverseWriteResult(
        output,
        final_artifact.file_sha256,
        final_artifact.payload_self_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic selector CLI and return a process exit code."""
    parser = argparse.ArgumentParser(prog="remanentia-snn-source-universe")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = write_source_universe(arguments.repo_root, arguments.output)
    except (OSError, SourceUniverseError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "file_sha256": result.file_sha256,
                "output_path": str(result.output_path),
                "payload_self_sha256": result.payload_self_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
