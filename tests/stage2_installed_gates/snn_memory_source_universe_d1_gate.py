# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Installed-wheel source-universe D1 gate

"""Exercise D1 only through real Git repositories and installed public surfaces.

Every fixture is a real temporary repository driven by the ``git`` CLI: crafted
trees use real plumbing (``hash-object``/``mktree``/``commit-tree``), worktree
tampering mutates real files, and artifact forgeries are recomputed with an
independent local reimplementation of the canonical serialization and the
domain-separated self digest. No mock, fake inference, or in-memory substitute
stands in for a production surface.
"""

from __future__ import annotations

import argparse
import base64
import compileall
import contextlib
import hashlib
import importlib
import io
import json
import operator
import os
import runpy
import stat
import subprocess
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Callable

from snn_memory import encoder as installed_encoder
from snn_memory import source_universe
from snn_memory.source_universe import (
    SourceUniverseError,
    select_source_universe,
    validate_source_universe_bytes,
    write_source_universe,
)

SELECTION_DOMAIN = b"remanentia-snn-v2-lock\0"
SELF_DOMAIN = b"remanentia:snn-v2-source-universe:v1\0"
MARKER_ONE = b"<!-- GENERATED FILE: DO NOT EDIT -->\n"
MARKER_TWO = b"<!-- generated: do not edit -->\n"
BASE_TIMESTAMP = 1_700_000_000
RETOUCH_OFFSET = 300
NFC_NAME = "docs/public/\u00e9vidence.md"
NFC_COLLIDER = "docs/\u00e9.md"
NFD_COLLIDER = "docs/e\u0301.md"
NFD_SOURCE = "docs/public/e\u0301-nfd.md"


def _canonical(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _self_digest(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "self_sha256"}
    canonical = _canonical(unsigned)
    framed = SELF_DOMAIN + len(canonical).to_bytes(8, "big") + canonical
    return hashlib.sha256(framed).hexdigest()


def _craft(base: bytes, mutate: Callable[[dict[str, Any]], None]) -> bytes:
    payload = json.loads(base.decode("utf-8"))
    mutate(payload)
    payload["self_sha256"] = _self_digest(payload)
    return _canonical(payload)


def _expect(substring: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except SourceUniverseError as error:
        if substring not in str(error):
            raise AssertionError(f"unexpected error {error!r}, wanted {substring!r}") from error
        return
    raise AssertionError(f"expected SourceUniverseError containing {substring!r}")


def _run(
    arguments: list[str],
    cwd: Path,
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        arguments, cwd=cwd, capture_output=True, check=check, env=env, input=input_bytes
    )


def _git(repo: Path, *arguments: str) -> bytes:
    return _run(["git", *arguments], repo).stdout


def _commit_env(timestamp: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment["GIT_AUTHOR_DATE"] = f"@{timestamp} +0000"
    environment["GIT_COMMITTER_DATE"] = f"@{timestamp} +0000"
    return environment


def _commit(repo: Path, message: str, timestamp: int, *, add_all: bool = True) -> None:
    if add_all:
        _git(repo, "add", "-A")
    _run(["git", "commit", "-q", "-m", message], repo, env=_commit_env(timestamp))


def _init(parent: Path, name: str, *extra: str) -> Path:
    repo = parent / name
    repo.mkdir(parents=True)
    _run(["git", "init", "-q", *extra], repo)
    _git(repo, "config", "user.name", "D1 Gate")
    _git(repo, "config", "user.email", "d1@example.invalid")
    return repo


def _text(tag: str, events: int = 60, padding: int = 8) -> bytes:
    return "\n".join(
        f"{tag} event {index:03d} carries deterministic public memory evidence {'x' * padding}."
        for index in range(events)
    ).encode("utf-8")


def _write(repo: Path, relative: str, raw: bytes) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _manifest_payload(repo: Path, split: str, sources: list[tuple[str, str]]) -> dict[str, Any]:
    entries = [
        {
            "label": f"v1-{split}-{index}",
            "path": declared,
            "sha256": hashlib.sha256((repo / tracked).read_bytes()).hexdigest(),
        }
        for index, (tracked, declared) in enumerate(sources)
    ]
    return {
        "schema_version": 1,
        "split": split,
        "encoder_checkpoint": "../../.snn_models/model",
        "encoder_digest": "1" * 64,
        "entries": entries,
    }


def _write_manifests(
    repo: Path,
    *,
    development: list[tuple[str, str]],
    locked: list[tuple[str, str]],
    development_mutate: Callable[[dict[str, Any]], None] | None = None,
    locked_mutate: Callable[[dict[str, Any]], None] | None = None,
    development_raw: bytes | None = None,
) -> None:
    development_payload = _manifest_payload(repo, "development", development)
    if development_mutate is not None:
        development_mutate(development_payload)
    raw = development_raw or json.dumps(development_payload, sort_keys=True).encode("utf-8")
    _write(repo, "experiments/snn_memory/development_corpus.json", raw)
    locked_payload = _manifest_payload(repo, "locked-evaluation", locked)
    locked_payload["locked"] = True
    if locked_mutate is not None:
        locked_mutate(locked_payload)
    _write(
        repo,
        "experiments/snn_memory/locked_evaluation_corpus.json",
        json.dumps(locked_payload, sort_keys=True).encode("utf-8"),
    )


def _small_repo(
    parent: Path,
    name: str,
    *,
    records: int = 3,
    manifests: bool = True,
    development_mutate: Callable[[dict[str, Any]], None] | None = None,
    locked_mutate: Callable[[dict[str, Any]], None] | None = None,
    development_raw: bytes | None = None,
) -> Path:
    repo = _init(parent, name)
    for index in range(records):
        _write(repo, f"docs/public/record-{index:02d}.md", _text(f"{name}-{index:02d}"))
    if manifests:
        _write_manifests(
            repo,
            development=[("docs/public/record-00.md", "../../docs/public/record-00.md")],
            locked=[],
            development_mutate=development_mutate,
            locked_mutate=locked_mutate,
            development_raw=development_raw,
        )
    _commit(repo, "fixture", BASE_TIMESTAMP)
    return repo


def _base_repo(parent: Path, name: str, *, base_timestamp: int = BASE_TIMESTAMP) -> Path:
    repo = _init(parent, name)
    _write(repo, "top.md", _text("top"))
    for index in range(20):
        _write(repo, f"docs/public/record-{index:02d}.md", _text(f"record-{index:02d}"))
    _write(repo, NFC_NAME, _text("evidence-nfc"))
    executable = _write(repo, "docs/public/exec.md", _text("exec"))
    executable.chmod(0o755)
    _write(repo, "docs/public/deep/nested.md", _text("nested"))
    _write(repo, "docs/internal/private.md", _text("internal"))
    _write(repo, ".coordination/session.md", _text("coordination"))
    _write(repo, "vendor/package.md", _text("vendor"))
    _write(repo, "THIRD_PARTY/upper.md", _text("vendor-upper"))
    _write(repo, "generated/output.md", _text("generated"))
    _write(repo, "docs/public/tool.generated.md", _text("generated-suffix"))
    _write(repo, "docs/public/marker.md", MARKER_ONE + _text("marker"))
    _write(repo, "docs/public/marker2.md", MARKER_TWO + _text("marker-two"))
    _write(repo, "docs/public/LICENSE.md", _text("license"))
    _write(repo, "docs/public/index.md", _text("index"))
    _write(repo, "docs/public/empty.md", b"")
    _write(repo, "docs/public/small.md", b"Small document.\n")
    _write(repo, "docs/public/large.md", _text("large", 200, 100))
    _write(repo, "docs/public/few.md", _text("few", 10, 100))
    _write(repo, "docs/public/many.md", _text("many", 257, 1))
    _write_manifests(
        repo,
        development=[
            ("docs/public/record-00.md", "..//../docs/public/./record-00.md"),
            ("docs/public/record-01.md", "../../docs/public/record-01.md"),
        ],
        locked=[],
    )
    (repo / "docs/public/link-note.md").symlink_to("record-02.md")
    _commit(repo, "fixture", base_timestamp)
    _write(repo, "docs/public/record-03.md", _text("record-03-v2"))
    _commit(repo, "retouch record-03", base_timestamp + RETOUCH_OFFSET)
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{'1' * 40},docs/public/submodule.md",
    )
    _commit(repo, "attach gitlink", base_timestamp + 2 * RETOUCH_OFFSET, add_all=False)
    (repo / "docs/public/submodule.md").mkdir()
    return repo


EXPECTED_REASONS = Counter(
    {
        "eligible": 22,
        "non_regular": 2,
        "internal": 1,
        "coordination": 1,
        "schema_v1_manifest": 2,
        "vendor": 2,
        "generated": 4,
        "license_only": 1,
        "index_only": 1,
        "empty": 1,
        "bytes_below_min": 1,
        "bytes_above_max": 1,
        "events_below_min": 1,
        "events_above_max": 1,
    }
)


def _check_payload_immutability(artifact: Any) -> None:
    payload: Any = artifact.payload
    mutable: Any = artifact.payload
    mutations: list[Callable[[], object]] = [
        lambda: operator.setitem(mutable, "schema_version", 3),
        lambda: operator.delitem(mutable, "self_sha256"),
        lambda: operator.setitem(mutable["considered"][0], "reason", "vendor"),
        lambda: operator.setitem(mutable["considered"], 0, {}),
        lambda: operator.setitem(mutable["selected"][0]["event_sha256"], 0, "0" * 64),
    ]
    assert isinstance(payload, Mapping) and not isinstance(payload, dict)
    assert isinstance(payload["considered"], tuple)
    assert isinstance(payload["repository"], Mapping)
    assert isinstance(payload["selected"][0]["event_sha256"], tuple)
    assert payload["schema_version"] == 2
    for mutation in mutations:
        try:
            mutation()
        except TypeError:
            continue
        raise AssertionError("public artifact payload accepted a mutation")
    try:
        artifact.payload = {}
    except AttributeError:
        pass
    else:
        raise AssertionError("frozen artifact accepted attribute assignment")
    assert validate_source_universe_bytes(artifact.canonical_bytes).payload == payload


def _check_census(payload: dict[str, Any]) -> None:
    considered = payload["considered"]
    assert Counter(item["reason"] for item in considered) == EXPECTED_REASONS
    assert len(considered) == sum(EXPECTED_REASONS.values())
    by_reason = {item["normalized_path"]: item["reason"] for item in considered}
    assert by_reason["docs/internal/private.md"] == "internal"
    assert by_reason[".coordination/session.md"] == "coordination"
    assert by_reason["docs/public/record-00.md"] == "schema_v1_manifest"
    assert by_reason["docs/public/record-01.md"] == "schema_v1_manifest"
    assert by_reason["THIRD_PARTY/upper.md"] == "vendor"
    assert by_reason["docs/public/marker.md"] == "generated"
    assert by_reason["docs/public/marker2.md"] == "generated"
    assert by_reason["docs/public/tool.generated.md"] == "generated"
    assert by_reason["docs/public/LICENSE.md"] == "license_only"
    assert by_reason["docs/public/index.md"] == "index_only"
    non_regular = {
        item["normalized_path"]: (item["mode"], item["kind"])
        for item in considered
        if item["reason"] == "non_regular"
    }
    assert non_regular["docs/public/submodule.md"] == ("160000", "commit")
    assert non_regular["docs/public/link-note.md"] == ("120000", "blob")
    exec_item = next(
        item for item in considered if item["normalized_path"] == "docs/public/exec.md"
    )
    assert exec_item["mode"] == "100755" and exec_item["reason"] == "eligible"


def _check_independent_selection(repo: Path, payload: dict[str, Any]) -> None:
    head = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    assert payload["repository"] == {"head": head, "object_format": "sha1"}
    recomputed: list[tuple[str, str]] = []
    for item in payload["considered"]:
        if item["status"] != "eligible":
            continue
        raw = _run(["git", "cat-file", "blob", f"HEAD:{item['path']}"], repo).stdout
        content_sha = hashlib.sha256(raw).hexdigest()
        assert item["content_sha256"] == content_sha
        assert item["byte_count"] == len(raw)
        normalized = "/".join(
            unicodedata.normalize("NFC", part) for part in item["path"].split("/")
        )
        assert item["normalized_path"] == normalized
        key = hashlib.sha256(
            SELECTION_DOMAIN + normalized.encode("utf-8") + b"\0" + content_sha.encode("ascii")
        ).hexdigest()
        assert item["selection_key"] == key
        assert item["record_id"] == f"sha256:{content_sha}"
        blob_oid = _git(repo, "rev-parse", f"HEAD:{item['path']}").decode("ascii").strip()
        assert item["blob_oid"] == blob_oid
        recomputed.append((key, item["record_id"]))
    recomputed.sort()
    assert payload["eligible_record_ids"] == [record_id for _, record_id in recomputed]
    assert payload["selected_record_ids"] == [record_id for _, record_id in recomputed[:16]]
    assert [item["rank"] for item in payload["selected"]] == list(range(16))


def _check_timestamps(payload: dict[str, Any]) -> None:
    for item in payload["considered"]:
        if item["reason"] == "non_regular":
            continue
        expected = (
            BASE_TIMESTAMP + RETOUCH_OFFSET
            if item["normalized_path"] == "docs/public/record-03.md"
            else BASE_TIMESTAMP
        )
        assert item["timestamp_ns"] == expected * 1_000_000_000
        assert item["timestamp_source"] == "git-commit"
        assert item["timestamp_precision"] == "seconds"
        assert item["content_commit"] == payload["repository"]["head"]


def _check_implementations(payload: dict[str, Any]) -> None:
    module_bytes = Path(str(source_universe.__file__)).read_bytes()
    encoder_bytes = Path(str(installed_encoder.__file__)).read_bytes()
    selector = payload["implementations"]["selector"]
    split_events = payload["implementations"]["split_events"]
    assert selector["logical_path"] == "snn_memory/source_universe.py"
    assert split_events["logical_path"] == "snn_memory/encoder.py"
    for implementation, expected in ((selector, module_bytes), (split_events, encoder_bytes)):
        decoded = base64.b64decode(implementation["bytes_base64"], validate=True)
        assert decoded == expected
        assert implementation["byte_count"] == len(expected)
        assert implementation["sha256"] == hashlib.sha256(expected).hexdigest()


def _check_manifest_provenance(repo: Path, payload: dict[str, Any]) -> None:
    manifests = payload["schema_v1_manifests"]
    assert [manifest["path"] for manifest in manifests] == [
        "experiments/snn_memory/development_corpus.json",
        "experiments/snn_memory/locked_evaluation_corpus.json",
    ]
    for manifest in manifests:
        raw = _run(["git", "cat-file", "blob", f"HEAD:{manifest['path']}"], repo).stdout
        assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
        oid = _git(repo, "rev-parse", f"HEAD:{manifest['path']}").decode("ascii").strip()
        assert manifest["blob_oid"] == oid
    development = manifests[0]["sources"]
    assert [source["label"] for source in development] == ["v1-development-0", "v1-development-1"]
    assert development[0]["declared_path"] == "..//../docs/public/./record-00.md"
    assert development[0]["resolved_path"] == "docs/public/record-00.md"
    assert development[1]["resolved_path"] == "docs/public/record-01.md"
    assert manifests[1]["sources"] == []


def _check_write_and_atomicity(repo: Path, workspace: Path, base_bytes: bytes) -> None:
    output = workspace / "universe.json"
    result = write_source_universe(repo, output)
    assert output.read_bytes() == base_bytes
    assert result.file_sha256 == hashlib.sha256(base_bytes).hexdigest()
    assert result.payload_self_sha256 == _self_digest(json.loads(base_bytes.decode("utf-8")))
    assert result.file_sha256 != result.payload_self_sha256
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert not list(workspace.glob(".universe.json.tmp.*"))
    _expect("atomic no-clobber output link failed", lambda: write_source_universe(repo, output))
    link = workspace / "link.json"
    link.symlink_to(output)
    _expect("atomic no-clobber output link failed", lambda: write_source_universe(repo, link))
    nested = workspace / "fresh-parent" / "universe.json"
    assert write_source_universe(repo, nested).file_sha256 == result.file_sha256


def _module_cli(arguments: list[str], expected_code: int) -> tuple[str, str]:
    argv_backup = sys.argv[:]
    sys.argv = ["remanentia-snn-source-universe", *arguments]
    stdout, stderr = io.StringIO(), io.StringIO()
    code = -1
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_module("snn_memory.source_universe", run_name="__main__", alter_sys=True)
            except SystemExit as stop:
                code = int(stop.code or 0)
    finally:
        sys.argv = argv_backup
    assert code == expected_code, (code, expected_code, stderr.getvalue())
    return stdout.getvalue(), stderr.getvalue()


def _check_cli(repo: Path, workspace: Path, install_target: Path, base_bytes: bytes) -> None:
    cli_output = workspace / "cli.json"
    executable = install_target / "bin/remanentia-snn-source-universe"
    process = _run(
        [str(executable), "--repo-root", str(repo), "--output", str(cli_output)], workspace
    )
    cli_result = json.loads(process.stdout)
    assert cli_output.read_bytes() == base_bytes
    assert cli_result["file_sha256"] == hashlib.sha256(base_bytes).hexdigest()
    assert cli_result["output_path"] == str(cli_output)
    failed = _run(
        [str(executable), "--repo-root", str(repo), "--output", str(cli_output)],
        workspace,
        check=False,
    )
    assert failed.returncode == 2 and b"atomic no-clobber" in failed.stderr
    module_output = workspace / "module-cli.json"
    stdout, _ = _module_cli(["--repo-root", str(repo), "--output", str(module_output)], 0)
    module_result = json.loads(stdout)
    assert module_result["file_sha256"] == cli_result["file_sha256"]
    assert module_output.read_bytes() == base_bytes
    _, stderr = _module_cli(["--repo-root", str(repo), "--output", str(module_output)], 2)
    assert "atomic no-clobber" in stderr
    _, stderr = _module_cli(
        ["--repo-root", str(workspace / "ghost-root"), "--output", str(workspace / "g.json")], 2
    )
    assert stderr.strip()
    direct_output = workspace / "direct-cli.json"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = source_universe.main(["--repo-root", str(repo), "--output", str(direct_output)])
    assert code == 0
    assert json.loads(buffer.getvalue())["file_sha256"] == cli_result["file_sha256"]


def _check_repository_failures(workspace: Path) -> None:
    area = workspace / "repo-failures"
    area.mkdir()
    nonrepo = area / "nonrepo"
    nonrepo.mkdir()
    _expect("not a git repository", lambda: select_source_universe(nonrepo))
    empty = _init(area, "empty")
    _expect("git rev-parse --verify failed", lambda: select_source_universe(empty))
    plain = _small_repo(area, "plain")
    relative = Path(os.path.relpath(plain, Path.cwd()))
    _expect("absolute canonical path", lambda: select_source_universe(relative))
    linked_root = area / "linked-root"
    linked_root.symlink_to(plain)
    _expect("absolute canonical path", lambda: select_source_universe(linked_root))
    _expect("not the Git top level", lambda: select_source_universe(plain / "docs"))
    try:
        select_source_universe(area / "ghost")
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError for a nonexistent repository root")
    sha256_repo = _init(area, "sha256", "--object-format=sha256")
    _write(sha256_repo, "note.md", _text("sha256"))
    _commit(sha256_repo, "fixture", BASE_TIMESTAMP)
    _expect(
        "SHA-1 repository identity output is not in the expected plumbing format",
        lambda: select_source_universe(sha256_repo),
    )
    conflict = _init(area, "conflict")
    _write(conflict, "c.md", b"base\n")
    _commit(conflict, "base", BASE_TIMESTAMP)
    _git(conflict, "checkout", "-q", "-b", "side")
    _write(conflict, "c.md", b"side\n")
    _commit(conflict, "side", BASE_TIMESTAMP + 1)
    _git(conflict, "checkout", "-q", "-")
    _write(conflict, "c.md", b"main\n")
    _commit(conflict, "main", BASE_TIMESTAMP + 2)
    _run(["git", "merge", "-q", "side"], conflict, check=False)
    _expect(
        "malformed, conflicted, or duplicate Git index record",
        lambda: select_source_universe(conflict),
    )
    crafted = _init(area, "crafted-dup")
    _write(crafted, "seed.md", _text("seed"))
    _commit(crafted, "seed", BASE_TIMESTAMP)
    blob = _git(crafted, "hash-object", "-w", "seed.md").decode("ascii").strip()
    listing = f"100644 blob {blob}\tdup.md\n100644 blob {blob}\tdup.md\n".encode("ascii")
    tree = _run(["git", "mktree"], crafted, input_bytes=listing).stdout.decode("ascii").strip()
    commit = (
        _run(
            ["git", "commit-tree", "-m", "crafted", tree],
            crafted,
            env=_commit_env(BASE_TIMESTAMP),
        )
        .stdout.decode("ascii")
        .strip()
    )
    _git(crafted, "update-ref", "HEAD", commit)
    _expect("malformed or duplicate Git tree record", lambda: select_source_universe(crafted))
    collision = _init(area, "collision")
    _write(collision, NFC_COLLIDER, _text("nfc-form"))
    _write(collision, NFD_COLLIDER, _text("nfd-form"))
    _commit(collision, "collision", BASE_TIMESTAMP)
    _expect("normalized tracked paths collide", lambda: select_source_universe(collision))
    backslash = _init(area, "backslash")
    _write(backslash, "back\\slash.md", _text("backslash"))
    _commit(backslash, "backslash", BASE_TIMESTAMP)
    _expect("not canonical relative POSIX syntax", lambda: select_source_universe(backslash))
    binary_name = _init(area, "binary-name")
    _write(binary_name, os.fsdecode(b"bad\xff.md"), _text("binary-name"))
    _commit(binary_name, "binary name", BASE_TIMESTAMP)
    _expect("tracked path is not strict UTF-8", lambda: select_source_universe(binary_name))
    staged_new = _small_repo(area, "staged-new")
    _write(staged_new, "docs/public/new.md", _text("staged-new"))
    _git(staged_new, "add", "docs/public/new.md")
    _expect("tracked path sets differ", lambda: select_source_universe(staged_new))
    staged_modified = _small_repo(area, "staged-modified")
    _write(staged_modified, "docs/public/record-01.md", _text("staged-modified"))
    _git(staged_modified, "add", "docs/public/record-01.md")
    _expect("mode/blob identity differ", lambda: select_source_universe(staged_modified))
    insufficient = _small_repo(area, "insufficient", records=10)
    _expect("fewer than sixteen", lambda: select_source_universe(insufficient))
    duplicate = _init(area, "duplicate-content")
    for index in range(15):
        _write(duplicate, f"docs/public/unique-{index:02d}.md", _text(f"unique-{index:02d}"))
    _write(duplicate, "docs/public/twin-a.md", _text("twin"))
    _write(duplicate, "docs/public/twin-b.md", _text("twin"))
    _write_manifests(
        duplicate,
        development=[("docs/public/unique-00.md", "../../docs/public/unique-00.md")],
        locked=[],
    )
    _commit(duplicate, "duplicate content", BASE_TIMESTAMP)
    _expect("record IDs are not unique", lambda: select_source_universe(duplicate))
    invalid_content = _small_repo(area, "invalid-content")
    _write(invalid_content, "docs/public/record-01.md", b"\xff" * 1_100)
    _commit(invalid_content, "invalid utf8", BASE_TIMESTAMP + 1)
    _expect(
        "tracked Markdown content is not strict UTF-8",
        lambda: select_source_universe(invalid_content),
    )


def _check_worktree_tampering(workspace: Path) -> None:
    area = workspace / "tampering"
    area.mkdir()
    dirty = _small_repo(area, "dirty")
    _write(dirty, "docs/public/record-01.md", _text("tampered"))
    _expect("HEAD, index, and worktree bytes differ", lambda: select_source_universe(dirty))
    chmod_flip = _small_repo(area, "chmod-flip")
    (chmod_flip / "docs/public/record-01.md").chmod(0o755)
    _expect("file mode differs in worktree", lambda: select_source_universe(chmod_flip))
    dir_swap = _small_repo(area, "dir-swap")
    target = dir_swap / "docs/public/record-01.md"
    target.unlink()
    target.mkdir()
    _expect("changed type in worktree", lambda: select_source_universe(dir_swap))
    fifo_swap = _small_repo(area, "fifo-swap")
    target = fifo_swap / "docs/public/record-01.md"
    target.unlink()
    os.mkfifo(target)
    _expect("changed type in worktree", lambda: select_source_universe(fifo_swap))
    symlink_swap = _small_repo(area, "symlink-swap")
    target = symlink_swap / "docs/public/record-01.md"
    target.unlink()
    target.symlink_to("record-02.md")
    _expect("cannot be opened safely", lambda: select_source_universe(symlink_swap))
    deleted = _small_repo(area, "deleted")
    (deleted / "docs/public/record-01.md").unlink()
    _expect("cannot be opened safely", lambda: select_source_universe(deleted))
    parent_symlink = _small_repo(area, "parent-symlink")
    (parent_symlink / "docs/public").rename(parent_symlink / "docs/real")
    (parent_symlink / "docs/public").symlink_to("real")
    _expect(
        "regular tracked path traverses a symlink",
        lambda: select_source_universe(parent_symlink),
    )
    lost_parent = _small_repo(area, "lost-parent")
    subprocess.run(["rm", "-rf", str(lost_parent / "docs/public")], check=True)
    _expect("cannot be opened safely", lambda: select_source_universe(lost_parent))


def _check_non_regular_worktree(workspace: Path, base_repo: Path) -> None:
    area = workspace / "non-regular"
    area.mkdir()
    retarget = area / "clone-retarget"
    _run(["git", "clone", "-q", str(base_repo), str(retarget)], workspace)
    link = retarget / "docs/public/link-note.md"
    link.unlink()
    link.symlink_to("record-03.md")
    assert b" M docs/public/link-note.md" in _git(retarget, "status", "--porcelain")
    _expect(
        "HEAD, index, and worktree symlink identity differ",
        lambda: select_source_universe(retarget),
    )
    deleted_link = _small_repo(area, "symlink-deleted")
    (deleted_link / "docs/public/extra-link.md").symlink_to("record-00.md")
    _commit(deleted_link, "attach symlink", BASE_TIMESTAMP + 1)
    (deleted_link / "docs/public/extra-link.md").unlink()
    _expect("tracked symlink cannot be read safely", lambda: select_source_universe(deleted_link))
    file_swap = _small_repo(area, "symlink-file-swap")
    (file_swap / "docs/public/extra-link.md").symlink_to("record-00.md")
    _commit(file_swap, "attach symlink", BASE_TIMESTAMP + 1)
    (file_swap / "docs/public/extra-link.md").unlink()
    _write(file_swap, "docs/public/extra-link.md", b"regular now\n")
    _expect("tracked symlink cannot be read safely", lambda: select_source_universe(file_swap))
    walk = _small_repo(area, "symlink-parent-swap")
    (walk / "linkdir").mkdir()
    (walk / "linkdir/only.md").symlink_to("../docs/public/record-00.md")
    _commit(walk, "attach linkdir", BASE_TIMESTAMP + 1)
    (walk / "linkdir").rename(walk / "realdir")
    (walk / "linkdir").symlink_to("realdir")
    _expect("non-regular tracked path traverses a symlink", lambda: select_source_universe(walk))
    gitlink_missing = _small_repo(area, "gitlink-missing")
    _git(gitlink_missing, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},subgone.md")
    _commit(gitlink_missing, "attach gitlink", BASE_TIMESTAMP + 1, add_all=False)
    assert b" D subgone.md" in _git(gitlink_missing, "status", "--porcelain")
    _expect(
        "HEAD, index, and worktree gitlink identity differ",
        lambda: select_source_universe(gitlink_missing),
    )
    gitlink_denied = _small_repo(area, "gitlink-denied")
    _git(gitlink_denied, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},denied/sub.md")
    _commit(gitlink_denied, "attach gitlink", BASE_TIMESTAMP + 1, add_all=False)
    (gitlink_denied / "denied").mkdir()
    (gitlink_denied / "denied").chmod(0o000)
    try:
        _expect(
            "tracked gitlink cannot be inspected safely",
            lambda: select_source_universe(gitlink_denied),
        )
    finally:
        (gitlink_denied / "denied").chmod(0o755)
    gitlink_file = _small_repo(area, "gitlink-file-swap")
    _git(gitlink_file, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},subfile.md")
    _commit(gitlink_file, "attach gitlink", BASE_TIMESTAMP + 1, add_all=False)
    _write(gitlink_file, "subfile.md", b"not a directory\n")
    _expect(
        "tracked gitlink worktree entry is not a directory",
        lambda: select_source_universe(gitlink_file),
    )
    gitlink_symlink = _small_repo(area, "gitlink-symlink-swap")
    _git(gitlink_symlink, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},sublink.md")
    _commit(gitlink_symlink, "attach gitlink", BASE_TIMESTAMP + 1, add_all=False)
    (gitlink_symlink / "sublink.md").symlink_to("docs")
    _expect(
        "tracked gitlink worktree entry is not a directory",
        lambda: select_source_universe(gitlink_symlink),
    )
    gitlink_init = _small_repo(area, "gitlink-own-init")
    _git(gitlink_init, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},subinit.md")
    _commit(gitlink_init, "attach gitlink", BASE_TIMESTAMP + 1, add_all=False)
    (gitlink_init / "subinit.md").mkdir()
    _run(["git", "init", "-q"], gitlink_init / "subinit.md")
    _expect("git rev-parse --verify failed", lambda: select_source_universe(gitlink_init))
    plain_dir = _small_repo(area, "gitlink-plain-dir", records=18)
    _git(plain_dir, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},subplain.md")
    _commit(plain_dir, "attach gitlink", BASE_TIMESTAMP + 1, add_all=False)
    (plain_dir / "subplain.md").mkdir()
    _write(plain_dir, "subplain.md/loose.txt", b"foreign content\n")
    assert _git(plain_dir, "status", "--porcelain") == b""
    plain_payload = json.loads(select_source_universe(plain_dir).canonical_bytes.decode("utf-8"))
    plain_reasons = {
        item["normalized_path"]: item["reason"] for item in plain_payload["considered"]
    }
    assert plain_reasons["subplain.md"] == "non_regular"
    sub_origin = _init(area, "sub-origin")
    _write(sub_origin, "state.txt", b"origin state one\n")
    _commit(sub_origin, "origin one", BASE_TIMESTAMP)
    origin_head = _git(sub_origin, "rev-parse", "HEAD").decode("ascii").strip()
    host = _small_repo(area, "sub-host", records=18)
    _run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            "../sub-origin",
            "docs/public/subrepo.md",
        ],
        host,
    )
    _commit(host, "attach submodule", BASE_TIMESTAMP + 1)
    host_payload = json.loads(select_source_universe(host).canonical_bytes.decode("utf-8"))
    submodule_audit = next(
        item
        for item in host_payload["considered"]
        if item["normalized_path"] == "docs/public/subrepo.md"
    )
    assert submodule_audit["reason"] == "non_regular"
    assert (submodule_audit["mode"], submodule_audit["kind"]) == ("160000", "commit")
    assert submodule_audit["blob_oid"] == origin_head
    uninitialised = area / "sub-host-uninitialised"
    _run(["git", "clone", "-q", str(host), str(uninitialised)], workspace)
    assert select_source_universe(uninitialised).payload["schema_version"] == 2
    sub_path = host / "docs/public/subrepo.md"
    _git(sub_path, "config", "user.name", "D1 Gate")
    _git(sub_path, "config", "user.email", "d1@example.invalid")
    _write(host, "docs/public/subrepo.md/advance.txt", b"advanced submodule state\n")
    _run(["git", "add", "advance.txt"], sub_path)
    _run(["git", "commit", "-q", "-m", "advance"], sub_path, env=_commit_env(BASE_TIMESTAMP + 2))
    assert b" M docs/public/subrepo.md" in _git(host, "status", "--porcelain")
    _expect(
        "HEAD, index, and worktree gitlink identity differ",
        lambda: select_source_universe(host),
    )


def _check_manifest_failures(workspace: Path) -> None:
    area = workspace / "manifest-failures"
    area.mkdir()

    def expect_repo(name: str, substring: str, **kwargs: Any) -> None:
        repo = _small_repo(area, name, **kwargs)
        _expect(substring, lambda: select_source_universe(repo))

    expect_repo("missing", "must be a committed regular file", manifests=False)
    symlinked = _small_repo(area, "symlinked", manifests=False)
    _write_manifests(
        symlinked,
        development=[("docs/public/record-00.md", "../../docs/public/record-00.md")],
        locked=[],
    )
    manifest_path = symlinked / "experiments/snn_memory/development_corpus.json"
    manifest_path.rename(symlinked / "experiments/snn_memory/real.json")
    manifest_path.symlink_to("real.json")
    _commit(symlinked, "symlinked manifest", BASE_TIMESTAMP)
    _expect("not a regular blob", lambda: select_source_universe(symlinked))
    expect_repo("bad-json", "is not strict UTF-8 JSON", development_raw=b"{")
    expect_repo("array-root", "root must be an object", development_raw=b"[]")
    expect_repo("dup-key", "duplicate JSON key", development_raw=b'{"a": 1, "a": 2}')
    expect_repo("nan", "non-finite JSON constant", development_raw=b'{"a": NaN}')
    expect_repo("overflow", "non-finite JSON number", development_raw=b'{"a": 1e999}')

    def set_field(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        def mutate(payload: dict[str, Any]) -> None:
            payload[key] = value

        return mutate

    expect_repo(
        "unknown-field", "unexpected fields or version", development_mutate=set_field("extra", 1)
    )
    expect_repo(
        "float-version",
        "unexpected fields or version",
        development_mutate=set_field("schema_version", 1.5),
    )
    expect_repo(
        "wrong-split",
        "split/lock contract differs",
        development_mutate=set_field("split", "locked-evaluation"),
    )
    expect_repo("unlocked", "split/lock contract differs", locked_mutate=set_field("locked", False))
    expect_repo(
        "encoder-type",
        "encoder fields are invalid",
        development_mutate=set_field("encoder_checkpoint", 7),
    )
    expect_repo(
        "encoder-digest",
        "encoder digest is invalid",
        development_mutate=set_field("encoder_digest", "z" * 64),
    )
    expect_repo(
        "entries-object", "entries must be a list", development_mutate=set_field("entries", {})
    )
    expect_repo(
        "entry-scalar", "entry schema differs", development_mutate=set_field("entries", [7])
    )

    def mutate_entry(
        mutate: Callable[[dict[str, Any]], None],
    ) -> Callable[[dict[str, Any]], None]:
        def apply(payload: dict[str, Any]) -> None:
            mutate(payload["entries"][0])

        return apply

    expect_repo(
        "entry-extra",
        "entry schema differs",
        development_mutate=mutate_entry(lambda entry: entry.update(extra=1)),
    )

    def duplicate_label(payload: dict[str, Any]) -> None:
        clone = dict(payload["entries"][0])
        clone["path"] = "../../docs/public/record-01.md"
        payload["entries"].append(clone)

    expect_repo("dup-label", "identity is invalid or duplicate", development_mutate=duplicate_label)
    expect_repo(
        "absolute-path",
        "not relative POSIX syntax",
        development_mutate=mutate_entry(lambda entry: entry.update(path="/absolute.md")),
    )
    expect_repo(
        "backslash-path",
        "not relative POSIX syntax",
        development_mutate=mutate_entry(lambda entry: entry.update(path="a\\b.md")),
    )
    expect_repo(
        "escape",
        "escapes the repository",
        development_mutate=mutate_entry(lambda entry: entry.update(path="../../../escape.md")),
    )

    def duplicate_resolved(payload: dict[str, Any]) -> None:
        clone = dict(payload["entries"][0])
        clone["label"] = "v1-development-clone"
        clone["path"] = ".//..//..//docs/public/record-00.md"
        payload["entries"].append(clone)

    expect_repo(
        "dup-resolved", "resolve to a duplicate path", development_mutate=duplicate_resolved
    )
    expect_repo(
        "untracked",
        "not a tracked canonical path",
        development_mutate=mutate_entry(
            lambda entry: entry.update(path="../../docs/public/ghost.md")
        ),
    )
    nfd_repo = _init(area, "nfd-source")
    _write(nfd_repo, NFD_SOURCE, _text("nfd-source"))
    _write_manifests(nfd_repo, development=[(NFD_SOURCE, f"../../{NFD_SOURCE}")], locked=[])
    _commit(nfd_repo, "nfd source", BASE_TIMESTAMP)
    _expect("not a tracked canonical path", lambda: select_source_universe(nfd_repo))
    json_source = _small_repo(area, "json-source", manifests=False)
    _write(json_source, "experiments/snn_memory/data.json", b"{}\n")
    _write_manifests(
        json_source,
        development=[("experiments/snn_memory/data.json", "./data.json")],
        locked=[],
    )
    _commit(json_source, "json source", BASE_TIMESTAMP)
    _expect("not Markdown", lambda: select_source_universe(json_source))
    expect_repo(
        "digest-mismatch",
        "source digest mismatch",
        development_mutate=mutate_entry(lambda entry: entry.update(sha256="0" * 64)),
    )


def _check_crafted_artifacts(base: bytes) -> None:
    valid = json.loads(base.decode("utf-8"))
    _expect(
        "file SHA-256 mismatch",
        lambda: validate_source_universe_bytes(base, expected_file_sha256="0" * 64),
    )
    _expect("is not strict UTF-8 JSON", lambda: validate_source_universe_bytes(b"\xff"))
    _expect("is not strict UTF-8 JSON", lambda: validate_source_universe_bytes(b"{"))
    _expect("root must be an object", lambda: validate_source_universe_bytes(b"[]\n"))
    _expect("duplicate JSON key", lambda: validate_source_universe_bytes(b'{"a": 1, "a": 2}'))
    _expect("non-finite JSON constant", lambda: validate_source_universe_bytes(b'{"a": NaN}'))
    _expect("non-finite JSON number", lambda: validate_source_universe_bytes(b'{"a": 1e999}'))

    def craft(mutate: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        return lambda: validate_source_universe_bytes(_craft(base, mutate))

    _expect("schema validation failed", craft(lambda payload: payload.update(unknown=1)))
    pretty = json.dumps(valid, sort_keys=True, indent=2).encode("utf-8")
    _expect("is not canonical", lambda: validate_source_universe_bytes(pretty))
    escaped = (
        json.dumps(valid, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")
    assert escaped != base
    _expect("is not canonical", lambda: validate_source_universe_bytes(escaped))
    _expect("is not canonical", lambda: validate_source_universe_bytes(base[:-1]))
    tampered_self = base.replace(valid["self_sha256"].encode("ascii"), b"0" * 64)
    _expect("self digest mismatch", lambda: validate_source_universe_bytes(tampered_self))
    surrogate = base.replace(b'"docs/public/small.md"', b'"docs/public/small\\ud800.md"', 1)
    _expect("serialization failed", lambda: validate_source_universe_bytes(surrogate))

    def selector_b64(payload: dict[str, Any]) -> None:
        payload["implementations"]["selector"]["bytes_base64"] = "!!"

    _expect("base64 is invalid", craft(selector_b64))

    def selector_bytes(payload: dict[str, Any]) -> None:
        payload["implementations"]["selector"]["bytes_base64"] = base64.b64encode(b"x").decode()

    _expect("byte binding differs", craft(selector_bytes))

    def swap_manifests(payload: dict[str, Any]) -> None:
        payload["schema_v1_manifests"].reverse()

    _expect("manifest order/path differs", craft(swap_manifests))

    def duplicate_resolved(payload: dict[str, Any]) -> None:
        first = payload["schema_v1_manifests"][0]["sources"][0]
        payload["schema_v1_manifests"][1]["sources"] = [dict(first)]

    _expect("resolved source paths are duplicated", craft(duplicate_resolved))

    def ghost_resolved(payload: dict[str, Any]) -> None:
        source = payload["schema_v1_manifests"][0]["sources"][0]
        source["declared_path"] = "../../docs/public/ghost.md"
        source["resolved_path"] = "docs/public/ghost.md"

    _expect("not bound to the audit", craft(ghost_resolved))

    def label_whitespace(payload: dict[str, Any]) -> None:
        payload["schema_v1_manifests"][0]["sources"][0]["label"] = " v1-development-0"

    _expect("label is invalid or duplicate", craft(label_whitespace))

    def label_duplicate(payload: dict[str, Any]) -> None:
        sources = payload["schema_v1_manifests"][0]["sources"]
        sources[1]["label"] = sources[0]["label"]

    _expect("label is invalid or duplicate", craft(label_duplicate))

    def declared_duplicate(payload: dict[str, Any]) -> None:
        sources = payload["schema_v1_manifests"][0]["sources"]
        sources[1]["declared_path"] = sources[0]["declared_path"]

    _expect("declared paths are duplicated", craft(declared_duplicate))

    def declared_tampered(payload: dict[str, Any]) -> None:
        source = payload["schema_v1_manifests"][0]["sources"][0]
        source["declared_path"] = "../../docs/public/record-05.md"

    _expect("does not resolve to the resolved path", craft(declared_tampered))

    def declared_absolute(payload: dict[str, Any]) -> None:
        payload["schema_v1_manifests"][0]["sources"][0]["declared_path"] = "/x.md"

    _expect("not relative POSIX syntax", craft(declared_absolute))

    def declared_escape(payload: dict[str, Any]) -> None:
        source = payload["schema_v1_manifests"][0]["sources"][0]
        source["declared_path"] = "../../../../x.md"

    _expect("escapes the repository", craft(declared_escape))

    def resolved_sha(payload: dict[str, Any]) -> None:
        payload["schema_v1_manifests"][0]["sources"][0]["sha256"] = "0" * 64

    _expect("not bound to the audit", craft(resolved_sha))

    def unsorted(payload: dict[str, Any]) -> None:
        payload["considered"][0], payload["considered"][1] = (
            payload["considered"][1],
            payload["considered"][0],
        )

    _expect("not sorted and unique", craft(unsorted))

    def duplicated(payload: dict[str, Any]) -> None:
        payload["considered"][1] = json.loads(json.dumps(payload["considered"][0]))

    _expect("not sorted and unique", craft(duplicated))

    def find(payload: dict[str, Any], reason: str) -> dict[str, Any]:
        return next(item for item in payload["considered"] if item["reason"] == reason)

    def find_path(payload: dict[str, Any], normalized: str) -> dict[str, Any]:
        return next(item for item in payload["considered"] if item["normalized_path"] == normalized)

    _expect(
        "non-regular audit carries unexpected bindings",
        craft(lambda payload: find(payload, "non_regular").update(byte_count=0)),
    )
    _expect(
        "non-regular audit status must be excluded",
        craft(lambda payload: find(payload, "non_regular").update(status="eligible")),
    )
    _expect(
        "does not normalise to its normalized_path",
        craft(lambda payload: find_path(payload, "top.md").update(path="renamed.md")),
    )
    _expect(
        "considered path is not canonical relative POSIX syntax",
        craft(lambda payload: find_path(payload, "top.md").update(path="/top.md")),
    )

    nfd_variant = "docs/public/e\u0301vidence.md"
    assert nfd_variant != NFC_NAME
    assert unicodedata.normalize("NFC", nfd_variant) == NFC_NAME

    def nfd_path_everywhere(payload: dict[str, Any]) -> None:
        for item in [*payload["considered"], *payload["selected"]]:
            if item["normalized_path"] == NFC_NAME:
                item["path"] = nfd_variant

    nfd_reseal = _craft(base, nfd_path_everywhere)
    assert nfd_reseal != base
    assert validate_source_universe_bytes(nfd_reseal).payload["schema_version"] == 2

    def regular_marked_non_regular(payload: dict[str, Any]) -> None:
        item = find(payload, "non_regular")
        item["mode"], item["kind"] = "100644", "blob"

    _expect("non-regular audit carries unexpected bindings", craft(regular_marked_non_regular))

    def drop_field(payload: dict[str, Any]) -> None:
        del find(payload, "empty")["event_order_digest"]

    _expect("content audit fields differ", craft(drop_field))
    _expect(
        "content audit fields differ",
        craft(lambda payload: find(payload, "empty").update(rank=0)),
    )
    _expect(
        "commit differs from repository HEAD",
        craft(lambda payload: find(payload, "empty").update(content_commit="0" * 40)),
    )

    def fractional_timestamp(payload: dict[str, Any]) -> None:
        item = find(payload, "empty")
        item["timestamp_ns"] = item["timestamp_ns"] + 1

    _expect("not whole seconds", craft(fractional_timestamp))

    def status_disagrees(payload: dict[str, Any]) -> None:
        find_path(payload, "top.md")["reason"] = "bytes_below_min"

    _expect("status and reason disagree", craft(status_disagrees))

    def internal_claims_vendor(payload: dict[str, Any]) -> None:
        find_path(payload, "docs/internal/private.md")["reason"] = "vendor"

    _expect("differs from path-derived reason", craft(internal_claims_vendor))

    def small_claims_empty(payload: dict[str, Any]) -> None:
        find_path(payload, "docs/public/small.md")["reason"] = "empty"

    _expect("differs from recomputed reason", craft(small_claims_empty))
    generated_claim = _craft(
        base,
        lambda payload: find_path(payload, "docs/public/small.md").update(reason="generated"),
    )
    assert validate_source_universe_bytes(generated_claim).payload["schema_version"] == 2

    def eligible(payload: dict[str, Any]) -> dict[str, Any]:
        return find_path(payload, "top.md")

    _expect(
        "record ID differs from content digest",
        craft(lambda payload: eligible(payload).update(record_id="sha256:" + "0" * 64)),
    )
    _expect(
        "selection key differs",
        craft(lambda payload: eligible(payload).update(selection_key="0" * 64)),
    )

    def extra_event_hash(payload: dict[str, Any]) -> None:
        eligible(payload)["event_sha256"].append("0" * 64)

    _expect("event count and event hashes differ", craft(extra_event_hash))

    def reversed_eligible(payload: dict[str, Any]) -> None:
        payload["eligible_record_ids"].reverse()

    _expect("eligible record list differs", craft(reversed_eligible))

    def broken_rank(payload: dict[str, Any]) -> None:
        payload["selected"][0]["rank"] = 1

    _expect("ranks are not stable zero-based order", craft(broken_rank))

    def swapped_selected_ids(payload: dict[str, Any]) -> None:
        ids = payload["selected_record_ids"]
        ids[0], ids[1] = ids[1], ids[0]

    _expect("selected record ID list differs", craft(swapped_selected_ids))

    def swapped_selected_paths(payload: dict[str, Any]) -> None:
        paths = payload["selected_paths"]
        paths[0], paths[1] = paths[1], paths[0]

    _expect("selected path list differs", craft(swapped_selected_paths))

    def altered_selected(payload: dict[str, Any]) -> None:
        payload["selected"][0]["byte_count"] += 1

    _expect("selected record object differs", craft(altered_selected))


def _check_determinism(workspace: Path, repo: Path, base_bytes: bytes) -> None:
    assert select_source_universe(repo).canonical_bytes == base_bytes
    clone = workspace / "clone"
    _run(["git", "clone", "-q", str(repo), str(clone)], workspace)
    assert select_source_universe(clone).canonical_bytes == base_bytes
    shifted = _base_repo(
        workspace / "shifted-parent", "shifted", base_timestamp=BASE_TIMESTAMP + 7_200
    )
    shifted_artifact = select_source_universe(shifted)
    assert shifted_artifact.canonical_bytes != base_bytes
    payload = json.loads(base_bytes.decode("utf-8"))
    assert list(shifted_artifact.payload["selected_record_ids"]) == payload["selected_record_ids"]
    assert list(shifted_artifact.payload["eligible_record_ids"]) == payload["eligible_record_ids"]


def _check_module_identity(repo: Path, base_bytes: bytes) -> None:
    module_path = Path(str(source_universe.__file__))
    encoder_path = Path(str(installed_encoder.__file__))
    module_bytes = module_path.read_bytes()
    encoder_bytes = encoder_path.read_bytes()
    try:
        encoder_path.write_bytes(b"\xff\xfe")
        _expect("implementation source is not strict UTF-8", lambda: select_source_universe(repo))
    finally:
        encoder_path.write_bytes(encoder_bytes)
    try:
        module_path.unlink()
        _expect("implementation source cannot be read", lambda: select_source_universe(repo))
    finally:
        module_path.write_bytes(module_bytes)
    bytecode = encoder_path.with_suffix(".pyc")
    try:
        assert compileall.compile_file(str(encoder_path), quiet=1, legacy=True)
        encoder_path.unlink()
        importlib.invalidate_caches()
        importlib.reload(installed_encoder)
        _expect("not a real non-shadowed source file", lambda: select_source_universe(repo))
    finally:
        bytecode.unlink(missing_ok=True)
        encoder_path.write_bytes(encoder_bytes)
        importlib.invalidate_caches()
        importlib.reload(installed_encoder)
    specification = installed_encoder.__spec__
    assert specification is not None and isinstance(specification.loader, SourceFileLoader)
    package_dir = module_path.parent
    relocated = package_dir.with_name("snn_memory_relocated")
    try:
        package_dir.rename(relocated)
        package_dir.symlink_to(relocated.name)
        _expect("implementation source traverses a symlink", lambda: select_source_universe(repo))
    finally:
        if package_dir.is_symlink():
            package_dir.unlink()
        relocated.rename(package_dir)
    assert select_source_universe(repo).canonical_bytes == base_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--install-target", type=Path, required=True)
    parser.add_argument("--public-schema", type=Path, required=True)
    parser.add_argument("--public-license", type=Path, required=True)
    arguments = parser.parse_args()
    module_origin = Path(str(source_universe.__file__)).resolve()
    assert module_origin.is_relative_to(arguments.install_target.resolve())
    package_schema = module_origin.parent / "schema/snn_memory_source_universe_v2.schema.json"
    assert arguments.public_schema.read_bytes() == package_schema.read_bytes()
    package_license = package_schema.with_name(package_schema.name + ".license")
    assert arguments.public_license.read_bytes() == package_license.read_bytes()
    workspace = arguments.workspace
    repo = _base_repo(workspace, "base")
    artifact = select_source_universe(repo)
    base_bytes = artifact.canonical_bytes
    plain_payload = json.loads(base_bytes.decode("utf-8"))
    assert artifact.file_sha256 == hashlib.sha256(base_bytes).hexdigest()
    assert artifact.payload_self_sha256 == _self_digest(plain_payload)
    assert artifact.file_sha256 != artifact.payload_self_sha256
    _check_payload_immutability(artifact)
    _check_census(plain_payload)
    _check_independent_selection(repo, plain_payload)
    _check_timestamps(plain_payload)
    _check_implementations(plain_payload)
    _check_manifest_provenance(repo, plain_payload)
    _check_write_and_atomicity(repo, workspace, base_bytes)
    _check_cli(repo, workspace, arguments.install_target, base_bytes)
    _check_repository_failures(workspace)
    _check_worktree_tampering(workspace)
    _check_non_regular_worktree(workspace, repo)
    _check_manifest_failures(workspace)
    _check_crafted_artifacts(base_bytes)
    _check_determinism(workspace, repo, base_bytes)
    _check_module_identity(repo, base_bytes)
    print(
        json.dumps(
            {
                "module_origin": str(module_origin),
                "module_sha256": hashlib.sha256(module_origin.read_bytes()).hexdigest(),
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
