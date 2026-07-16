# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Clean installed-wheel D4-A checkpoint/bank materialization verifier

"""Build fresh Python and Rust wheels, install them, and cover D4-A to >=95%."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("REMANENTIA_STAGE4_RUNTIME_ROOT",
                              "[workspace]/_runtime/REMANENTIA/D4A"))
GATE = ROOT / "tests/stage4_installed_gates/snn_memory_d4a_gate.py"
EXPECTED_GATES = {"snn_memory_d4a_gate.py"}
MODEL = ROOT / ".snn_models" / "all-MiniLM-L6-v2"
MANIFESTS = (
    ROOT / "experiments/snn_memory/development_corpus.json",
    ROOT / "experiments/snn_memory/locked_evaluation_corpus.json",
)
_MODULES = ("checkpoint_bundle_v2", "candidate_bank_v2", "checkpoint_materialize_v2")
_RSS_LIMIT_KB = 1_572_864          # 1.5 GiB aggregate process-tree RSS
_DISK_LIMIT_BYTES = 2 * 1024 ** 3  # 2 GiB workspace
_WALL_LIMIT_SECONDS = 1800         # 30 minutes (D4-A amendment 2026-07-16T1819; external watchdog >=2100s)
_MIN_COVERAGE = 95.0
# workers=1: at most one materializer child plus its single calibration worker alive under the gate
# root at any instant; a larger concurrent non-root count means parallelism escaped serialization.
_MAX_CONCURRENT_NON_ROOT = 2
_CANDIDATE_COUNT = 16
_MATERIALIZER_RUNS = 11  # ten (2 seeds x 5 conditions) plus one byte-determinism replay
_N_NEURONS = 24


def _descendant_pids(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(pid)
        except (OSError, IndexError, ValueError):
            continue
    tree, queue = [root_pid], [root_pid]
    while queue:
        for child in children.get(queue.pop(), []):
            tree.append(child)
            queue.append(child)
    return tree


def _process_group_pids(pgid: int) -> list[int]:
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                members.append(pid)
        except (OSError, IndexError, ValueError):
            continue
    return members


def _live_non_root(pids: set[int], root_pid: int) -> set[int]:
    """Non-root pids excluding the root and zombies (exited-but-unreaped are not concurrent work)."""
    live = set()
    for pid in pids:
        if pid == root_pid:
            continue
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                state = handle.read().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        if state != "Z":
            live.add(pid)
    return live


def _tree_rss_kb(pids: list[int]) -> int:
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except OSError:
            continue
    return total


def _tree_size_bytes(path: Path) -> int:
    total = 0
    for current, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(current, name)).st_size
            except OSError:
                continue
    return total


def _process_cmdlines(pids: list[int]) -> list[str]:
    lines = []
    for pid in sorted(pids):
        try:
            with open(f"/proc/{pid}/cmdline", encoding="utf-8", errors="replace") as handle:
                cmd = handle.read().replace("\x00", " ").strip()
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                fields = handle.read().rsplit(")", 1)
            state = fields[1].split()[0] if len(fields) == 2 else "?"
            lines.append(f"{pid}[{state}]:{cmd[:90]}")
        except (OSError, IndexError):
            lines.append(f"{pid}[gone]")
    return lines


def _supervise(argv: list[str], cwd: Path, environment: dict[str, str], workspace: Path) -> dict[str, Any]:
    """Run the gate as a supervised process group; monitor by ancestry AND process group.

    Fails closed and kills the whole group on any RSS/disk/wall breach, captures the gate's final
    canonical JSON report, records the exact observed PID set and peak workspace bytes, and
    requires the process group to be empty after a successful exit.
    """
    # Redirect the gate's stdout/stderr to files, not pipes: an OS pipe buffer (~64 KiB) that the
    # monitor loop only drains after the loop would deadlock a child that writes more than the buffer
    # before exiting. Files have no such limit, so the supervisor never has to drain a pipe and the
    # child can never block on output. (D4-A hardening after the concurrent-run wedge, 2026-07-16.)
    stdout_path = workspace / "_supervise_stdout.bin"
    stderr_path = workspace / "_supervise_stderr.bin"
    stdout_file = open(stdout_path, "wb")
    stderr_file = open(stderr_path, "wb")
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, start_new_session=True,
                                   stdout=stdout_file, stderr=stderr_file)
    finally:
        stdout_file.close()  # the child holds its own dup; the parent needs no copy
        stderr_file.close()
    pgid = os.getpgid(process.pid)
    peak_rss_kb, peak_disk, observed, max_descendants = 0, 0, set(), 0
    start = time.monotonic()
    try:
        while True:
            returncode = process.poll()
            tree = set(_descendant_pids(process.pid)) | set(_process_group_pids(pgid))
            observed |= tree
            live_non_root = _live_non_root(tree, process.pid)
            max_descendants = max(max_descendants, len(live_non_root))
            peak_rss_kb = max(peak_rss_kb, _tree_rss_kb(sorted(tree)))
            peak_disk = max(peak_disk, _tree_size_bytes(workspace))
            elapsed = time.monotonic() - start
            breach = ("RSS" if peak_rss_kb > _RSS_LIMIT_KB else
                      "workspace-disk" if peak_disk > _DISK_LIMIT_BYTES else
                      "wall-clock" if elapsed > _WALL_LIMIT_SECONDS else
                      "concurrency" if max_descendants > _MAX_CONCURRENT_NON_ROOT else None)
            if breach is not None:
                offenders = _process_cmdlines(sorted(live_non_root))
                os.killpg(pgid, signal.SIGKILL)
                process.wait()
                raise RuntimeError(
                    f"gate process group breached the {breach} envelope "
                    f"(peak_rss={peak_rss_kb} KiB, disk={peak_disk} B, elapsed={elapsed:.0f}s, "
                    f"live_non_root={len(live_non_root)}, procs={offenders})")
            if returncode is not None:
                break
            time.sleep(0.4)
        process.wait()
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
        leftover = [pid for pid in _process_group_pids(pgid) if pid != process.pid]
        if leftover:
            for pid in leftover:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    continue
            raise RuntimeError(f"gate left orphan process-group members after exit: {sorted(leftover)}")
        if returncode != 0:
            raise RuntimeError(f"gate exited {returncode}: {stderr.decode()[:2000]}")
        gate_stderr = stderr.decode()
        gate_stdout = stdout.decode()
        canonical = json.dumps(json.loads(gate_stdout), sort_keys=True,
                               separators=(",", ":")).encode("utf-8") + b"\n"
        if stdout != canonical:
            raise RuntimeError("gate stdout is not exactly one canonical JSON line")
        if gate_stderr != "":
            raise RuntimeError(f"gate emitted uncontrolled stderr: {gate_stderr[:2000]!r}")
        return {"root_pid": process.pid, "peak_rss_kb": peak_rss_kb, "peak_workspace_bytes": peak_disk,
                "observed_pids": sorted(observed), "unique_pid_count": len(observed),
                "max_concurrent_non_root": max_descendants, "wall_seconds": round(elapsed, 1),
                "gate_report": json.loads(gate_stdout)}
    except BaseException:
        if process.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
            process.wait()
        raise


def _run(arguments: list[str], cwd: Path, environment: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, env=environment, check=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {matches}")
    return matches[0].resolve(strict=True)


def _pinned_encoder_digest() -> str:
    if not MODEL.is_dir():
        raise RuntimeError(f"pinned local encoder not provisioned: {MODEL}; report the blocker")
    declared = tuple(
        json.loads(path.read_text(encoding="utf-8"))["encoder_digest"] for path in MANIFESTS
    )
    if len(set(declared)) != 1:
        raise RuntimeError(f"tracked corpus manifests disagree on encoder digest: {declared}")
    digest = hashlib.sha256()
    digest.update(b"remanentia.encoder-directory.v1\x00")
    entries = sorted(item for item in MODEL.rglob("*") if item.is_file())
    if not entries:
        raise RuntimeError(f"pinned local encoder is empty: {MODEL}")
    for item in entries:
        relative = item.relative_to(MODEL).as_posix().encode("utf-8")
        content = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    live = digest.hexdigest()
    if live != declared[0]:
        raise RuntimeError(f"pinned local encoder digest drift: tracked={declared[0]} live={live}")
    return live


def _schema_pair_parity(name: str) -> None:
    public = (ROOT / "docs/schema" / name).read_bytes()
    packaged = (ROOT / "snn_memory/schema" / name).read_bytes()
    if public != packaged:
        raise RuntimeError(f"schema/licence pair is not byte-identical: {name}")


def _write_coverage_harness(workspace: Path, data_file: Path, includes: str) -> tuple[Path, Path]:
    rcfile = workspace / "coveragerc"
    rcfile.write_text(
        "[run]\nbranch = True\nparallel = True\nsigterm = True\n"
        f"data_file = {data_file}\n[report]\ninclude = {includes}\n",
        encoding="utf-8",
    )
    site_dir = workspace / "covsite"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(
        "import coverage\ncoverage.process_startup()\n", encoding="utf-8")
    return rcfile, site_dir


# Preregistered A/B/C classification for every accepted coverage residual (canonical T1819 meanings;
# any residual not listed here fails the verifier as unclassified). Each entry records the exact arc/line
# and the exact invariant or upstream validator:
#   A - structurally unreachable through any authorised installed real surface because of an
#       authorisation, schema, OS, or backend invariant.
#   B - a retained defense-in-depth guard made unreachable by a named, authenticated upstream validator.
#   C - reachable on a real installed surface in principle but not yet witnessed by the preregistered
#       finite search (stays in the reachable denominator; materialize must clear to zero C).
_A = "A"  # provably unreachable through any authorised installed real surface
_SCHEMA_CONST = "field pinned by a schema const, rejected at schema validation before this guard"
_POST_SHAPE = "body length is pinned by the already-validated (rows, width) shape and byte_length"
_POST_RAW = "identical bytes that pass the raw-byte digest necessarily pass the decoded/row digests"
_DECODE_PINNED = "array shape and dtype are pinned by _decode_component before this validator runs"
_SCHEMA_MIN_EPOCHS = "the manifest schema conditionally pins epochs_completed (trained >= 1, untrained == 0), so this guard cannot be reached"
_WRITER_SHAPE_MASK = "write_checkpoint_bundle_v2 rejects a non-(n,n) topology at its own shape guard before this square check"


def _classA(reason: str, *keys: str) -> dict[str, dict[str, str]]:
    return {key: {"class": "A", "reason": reason} for key in keys}


def _classB(reason: str, *keys: str) -> dict[str, dict[str, str]]:
    return {key: {"class": "B", "reason": reason} for key in keys}


def _classC(reason: str, *keys: str) -> dict[str, dict[str, str]]:
    return {key: {"class": "C", "reason": reason} for key in keys}


_FIXTURE_FULL_CONN = "fixture connectivity=1.0 leaves no off-topology edge / no valid disagreeing adjacency row"
_FIXTURE_ONE_EPOCH = "single-epoch fixture cannot present one record across two epochs"
# _reconcile_bank_checkpoint is reached only through the bind CLI, which always reads the bank via
# candidate_bank_v2.read_candidate_bank_v2 first; guards the reader already enforces are unreachable.
_BANK_READER_CANDSET = "candidate_bank_v2 reader recomputes candidate_set_digest from ordered_record_ids before bind's _reconcile, its only caller"
_BANK_READER_SIG = "candidate_bank_v2 reader pins signature width and row digests before bind's _reconcile signature-byte check"
_BANK_READER_COMPLETION = "candidate_bank_v2 reader pins signature_layout.completion_steps == scoring.completion_steps before bind's _reconcile"
_BANK_READER_CALORDER = "candidate_bank_v2 reader pins calibration order/count to ordered_record_ids before bind's _reconcile"
_BANK_READER_CALBUILD = "candidate_bank_v2 reader binds calibration encoder directory/config to the build identity, which the identity-pair guard checks against the checkpoint"
_BANK_READER_ROWORDER = "candidate_bank_v2 reader binds ordered_record_ids to the signature row digests before bind's record-order guard"
_CALIBRATE_PLASTICITY_OFF = "calibration runs the backend with plasticity disabled (learn=False), so final weights equal the input weights by construction; the changed-weights guard is a retained fail-closed invariant"
# Worker-report re-verification is retained defense-in-depth (Class B): the calibration worker is the
# digest-authenticated in-tree _cmd_calibrate (spawned as sys.executable -m ..._calibrate; its bytes are
# pinned by _authenticated_worker_bytes and it IS the module under coverage). Each arc is masked by an
# exact construction invariant of that authenticated upstream, not a blanket claim.
_WORKER_RC_STDERR = "the digest-authenticated in-tree calibration worker returns exit 0 and writes nothing to stderr on the success path the parent drives, so the non-zero-return and stray-stderr guards are unreachable"
_WORKER_RASTER = "_completion_raster builds a (completion_steps, n) numpy bool array; its base64 always decodes, is exactly completion_steps*n bytes, and carries only 0/1 dtype bytes, so the invalid-base64/length/byte guards are unreachable"
_WORKER_FIELD_BIND = "the worker fills every report field from the same inputs the parent's expected dict is derived from, and post_state_digest is _state_signature (a 64-hex sha256), so the field-mismatch and non-sha256 guards are unreachable"
_WORKER_PID_NONCE = "each worker spawn receives a fresh OS pid and a fresh os.urandom(16) nonce, so a pid or nonce reuse within one calibration loop is impossible"
_WORKER_CANONICAL = "the worker serialises its report via json.dumps(sort_keys, compact separators)+newline over exactly _WORKER_REPORT_KEYS, so the parent strict-JSON re-parse cannot reach the non-finite-constant, duplicate-key, non-dict, wrong-fields, or non-canonical guards"
_D2_READER_NEURONS = "the D2 cue-set reader binds model.n_neurons to model.config (rejects a mismatch as 'cue-set model neuron count differs from its configuration'), so a reader-valid cue set always has n_neurons == the configured model's n_neurons"
_ENCODE_MIN_EVENTS = "composite: the offline D1 reader recomputes eligibility from the declared event_count (an eligible record requires event_count >= MIN_EVENTS = 50), materialize line 515 verifies the actual split_events count equals that declared event_count BEFORE _encode_events, and the calibration index subset is nonempty by schema — so _encode_events always sees >= 50 (training) or a nonempty subset (calibration) events and the no-events guard is unreachable"
_ENCODE_FINITE = "the pinned sentence-encoder yields finite embeddings and embeddings_to_currents is finite-preserving on them, so the encoded training/calibration current is always finite"
_CUE_SCHEMA_SAFE_PATH = "the cue-set schema pins each calibration cue path to '^cues/cue-<32 hex>.txt$', which forbids a '..'/absolute/empty relative form, so materialize's unsafe-relative-path guard (the '..'/absolute/empty check) is unreachable — the intermediate-traversal guard is a two-component path and IS witnessed"
_CUE_SCHEMA_INDICES = "the cue-set schema binds the calibration event_indices shape to the block, so the re-derived line-count guard is unreachable"
_D1_READER_UTF8 = "the D1 source-universe reader validates every tracked record as strict UTF-8 (rejects otherwise as 'tracked Markdown content is not strict UTF-8'), so a reader-valid D1 record always decodes and materialize's non-UTF-8 re-check is unreachable"
_STAGING_CLEANUP = "the staging rmtree+re-raise fires only if _produce_output or the atomic install raises after staging creation; on authenticated prepared inputs both succeed, so the cleanup backstop is unreachable"
# Class C = reachable on a real installed surface but not yet witnessed by the preregistered search.
_C_GITPATH = "reachable via a consistent D1 reseal renaming a selected record path to a HEAD-absent path (with considered/eligible re-sort); witness owed"
_C_CALBLOCK = "reachable via a D2 calibration reseal desyncing the block digest from the D1-derived calibration block; witness owed"
_C_CONTROL_ID = "reachable via a trained-checkpoint reseal mutating a descriptor identity so it differs from the control configuration; witness owed"
_C_CONTROL_ORDER = "reachable via a trained-checkpoint reseal mutating manifest ordered_record_ids; witness owed"
_C_SCORING = "reachable via a scoring-target + candidate-bank two-artifact reseal desyncing a scoring field from the bank; witness owed"
_SCORING_VALIDATOR_BANKDIG = "the scoring-target semantic validator binds identities.candidate_bank_digest to candidate_order+signatures, and the order/signature guards catch any bank divergence first, so the standalone candidate-bank-digest bind check is unreachable"
_SCORING_LAYOUT_COMPLETION = "the bank reader binds signature_layout.completion_steps == the scoring block's completion_steps and the scoring-target-vs-layout check fires first, so the scoring-target-vs-scoring-block completion check is unreachable"
_SCORING_SCHEMA_CONST = "the scoring-target schema pins bins=8, signature_dtype='<f8' and abstention.rule='strict-greater-than' to constants, so a schema-valid scoring target always matches the bank and these bind re-checks are unreachable"
_MERGE_SAME_LOOP = "calibration evidence and worker evidence are produced per-record over the same ordered record ids, so their lengths and record order agree by construction"
_C_RENAME_ERRNO = "reachable via a concurrent actor stripping write permission from the caller-supplied output parent between staging creation and renameat2 (EACCES); the structurally-identical candidate_bank_v2 rename helper IS witnessed on the installed public surface, this sibling is not separately witnessed"
_C_CLEANUP_RACE = "best-effort staging-cleanup OSError handler reachable under concurrent output-parent/staging tampering (the same race that reaches the rename leaves the parent unwritable, so rmdir cannot remove the staging entry); witness owed"


_RESIDUAL_CLASSIFICATION: dict[str, dict[str, dict[str, dict[str, str]]]] = {
    "checkpoint_bundle_v2": {
        "lines": {
            **_classC(_C_CLEANUP_RACE, "336", "337", "342", "343", "348", "349"),
            **_classC(_C_RENAME_ERRNO, "315"),
            **_classA(_WRITER_SHAPE_MASK, "527"),
            **_classA(_SCHEMA_MIN_EPOCHS, "685", "682"),
            **_classC(_FIXTURE_FULL_CONN, "564", "551"),
            **_classC(_FIXTURE_ONE_EPOCH, "617"),
        },
        "arcs": {
            **_classC(_C_RENAME_ERRNO, "313->315"),
            **_classA(_WRITER_SHAPE_MASK, "526->527"),
            **_classA(_SCHEMA_MIN_EPOCHS, "684->685", "681->682"),
            **_classC(_FIXTURE_FULL_CONN, "563->564", "550->551"),
            **_classC(_FIXTURE_ONE_EPOCH, "616->617"),
        },
    },
    "candidate_bank_v2": {
        "lines": {
            **_classC(_C_CLEANUP_RACE, "310", "311", "316", "317"),
            "384": {"class": _A, "reason": _SCHEMA_CONST},
            "427": {"class": _A, "reason": _POST_RAW}, "429": {"class": _A, "reason": _POST_RAW},
        },
        "arcs": {
            "383->384": {"class": _A, "reason": _SCHEMA_CONST},
            "426->427": {"class": _A, "reason": _POST_RAW},
            "428->429": {"class": _A, "reason": _POST_RAW},
        },
    },
    "checkpoint_materialize_v2": {
        "lines": {
            **_classA(_BANK_READER_CANDSET, "1194"),
            **_classA(_BANK_READER_SIG, "1196"),
            **_classA(_BANK_READER_COMPLETION, "1199"),
            **_classA(_BANK_READER_CALORDER, "1204"),
            **_classA(_BANK_READER_CALBUILD, "1210", "1212"),
            **_classA(_BANK_READER_ROWORDER, "1228"),
            **_classA(_SCHEMA_CONST, "1255"),
            **_classA(_CALIBRATE_PLASTICITY_OFF, "805"),
            **_classB(_WORKER_RC_STDERR, "679", "681"),
            **_classB(_WORKER_RASTER, "685", "686", "688", "690"),
            **_classB(_WORKER_FIELD_BIND, "704", "706"),
            **_classB(_WORKER_PID_NONCE, "708"),
            **_classB(_WORKER_CANONICAL, "734", "740", "748", "749", "751", "753", "756"),
            **_classB(_D2_READER_NEURONS, "489"),
            **_classB(_ENCODE_MIN_EVENTS, "215"),
            **_classB(_ENCODE_FINITE, "221"),
            **_classB(_CUE_SCHEMA_SAFE_PATH, "299"),
            **_classB(_CUE_SCHEMA_INDICES, "534"),
            **_classC(_C_GITPATH, "334"),
            **_classC(_C_CALBLOCK, "529"),
            **_classC(_C_CONTROL_ID, "624"),
            **_classC(_C_CONTROL_ORDER, "626"),
            **_classC(_C_SCORING, "1238", "1240", "1245"),
            **_classB(_SCORING_VALIDATOR_BANKDIG, "1242"),
            **_classB(_SCORING_LAYOUT_COMPLETION, "1247"),
            **_classB(_SCORING_SCHEMA_CONST, "1249", "1251", "1253"),
            **_classA(_MERGE_SAME_LOOP, "947", "951"),
        },
        "arcs": {
            **_classA(_BANK_READER_CANDSET, "1193->1194"),
            **_classA(_BANK_READER_SIG, "1195->1196"),
            **_classA(_BANK_READER_COMPLETION, "1198->1199"),
            **_classA(_BANK_READER_CALORDER, "1203->1204"),
            **_classA(_BANK_READER_CALBUILD, "1209->1210", "1211->1212"),
            **_classA(_BANK_READER_ROWORDER, "1227->1228"),
            **_classA(_SCHEMA_CONST, "1254->1255"),
            **_classA(_CALIBRATE_PLASTICITY_OFF, "804->805"),
            **_classB(_WORKER_RC_STDERR, "678->679", "680->681"),
            **_classB(_WORKER_RASTER, "687->688", "689->690"),
            **_classB(_WORKER_FIELD_BIND, "703->704", "705->706"),
            **_classB(_WORKER_PID_NONCE, "707->708"),
            **_classB(_WORKER_CANONICAL, "739->740", "750->751", "752->753", "755->756"),
            **_classB(_D2_READER_NEURONS, "488->489"),
            **_classB(_ENCODE_MIN_EVENTS, "214->215"),
            **_classB(_ENCODE_FINITE, "220->221"),
            **_classB(_CUE_SCHEMA_SAFE_PATH, "298->299"),
            **_classB(_CUE_SCHEMA_INDICES, "533->534"),
            **_classC(_C_GITPATH, "333->334"),
            **_classC(_C_CALBLOCK, "528->529"),
            **_classC(_C_CONTROL_ID, "623->624"),
            **_classC(_C_CONTROL_ORDER, "625->626"),
            **_classC(_C_SCORING, "1237->1238", "1239->1240", "1244->1245"),
            **_classB(_SCORING_VALIDATOR_BANKDIG, "1241->1242"),
            **_classB(_SCORING_LAYOUT_COMPLETION, "1246->1247"),
            **_classB(_SCORING_SCHEMA_CONST, "1248->1249", "1250->1251", "1252->1253"),
            **_classA(_MERGE_SAME_LOOP, "946->947", "950->951"),
        },
    },
}


def _classify_residuals(name: str, missing_lines: list[int],
                        missing_arcs: list[list[int]]) -> tuple[list[dict[str, object]], list[str]]:
    table = _RESIDUAL_CLASSIFICATION.get(name, {})
    lines_table, arcs_table = table.get("lines", {}), table.get("arcs", {})
    records: list[dict[str, object]] = []
    unclassified: list[str] = []
    for line in missing_lines:
        entry = lines_table.get(str(line))
        if entry is None:
            unclassified.append(f"{name}:line:{line}")
        else:
            records.append({"kind": "line", "location": line, **entry})
    for arc in missing_arcs:
        key = f"{arc[0]}->{arc[1]}"
        entry = arcs_table.get(key)
        if entry is None:
            unclassified.append(f"{name}:arc:{key}")
        else:
            records.append({"kind": "arc", "location": key, **entry})
    return records, unclassified


def _per_module_coverage(rcfile: Path, data_file: Path, modules: dict[str, Path],
                         workspace: Path, environment: dict[str, str]) -> dict[str, dict[str, object]]:
    json_path = workspace / "coverage.json"
    _run([sys.executable, "-m", "coverage", "json", f"--rcfile={rcfile}",
          f"--data-file={data_file}", "-o", str(json_path)], workspace, environment)
    files = json.loads(json_path.read_text(encoding="utf-8"))["files"]
    per_module: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    unclassified: list[str] = []
    for name, path in modules.items():
        # coverage json emits paths relative to its cwd (the workspace); accept both forms.
        entry = files.get(str(path))
        if entry is None:
            entry = files.get(os.path.relpath(path, workspace))
        if entry is None:
            raise RuntimeError(f"coverage recorded no data for {name} ({path})")
        summary = entry["summary"]
        statements, covered = summary["num_statements"], summary["covered_lines"]
        branches, covered_branches = summary["num_branches"], summary["covered_branches"]
        statement_pct = 100.0 * covered / statements if statements else 100.0
        branch_pct = 100.0 * covered_branches / branches if branches else 100.0
        missing_lines = entry.get("missing_lines", [])
        missing_arcs = entry.get("missing_branches", [])
        classification, module_unclassified = _classify_residuals(name, missing_lines, missing_arcs)
        unclassified.extend(module_unclassified)
        # A/B/C classification is disjoint by construction (one dict entry per line/arc). The reachable
        # denominators exclude only proven A/B; Class C stays in them (T1819 §5-6 branch, T1905 §3-4 stmt).
        line_records = [record for record in classification if record["kind"] == "line"]
        arc_records = [record for record in classification if record["kind"] == "arc"]
        a_lines = sorted(str(record["location"]) for record in line_records if record["class"] == "A")
        b_lines = sorted(str(record["location"]) for record in line_records if record["class"] == "B")
        c_lines = sorted(str(record["location"]) for record in line_records if record["class"] == "C")
        a_arcs = sorted(str(record["location"]) for record in arc_records if record["class"] == "A")
        b_arcs = sorted(str(record["location"]) for record in arc_records if record["class"] == "B")
        c_arcs = sorted(str(record["location"]) for record in arc_records if record["class"] == "C")
        adjusted_statement_denominator = statements - len(a_lines) - len(b_lines)
        adjusted_branch_denominator = branches - len(a_arcs) - len(b_arcs)
        reachable_statement_pct = (100.0 * covered / adjusted_statement_denominator
                                   if adjusted_statement_denominator else 100.0)
        reachable_branch_pct = (100.0 * covered_branches / adjusted_branch_denominator
                                if adjusted_branch_denominator else 100.0)
        structural_statement_ceiling = (100.0 * adjusted_statement_denominator / statements
                                        if statements else 100.0)
        structural_branch_ceiling = 100.0 * adjusted_branch_denominator / branches if branches else 100.0
        per_module[name] = {
            "num_statements": statements, "covered_statements": covered,
            "missing_statements": len(missing_lines), "statement_pct": round(statement_pct, 2),
            "reachable_statement_pct": round(reachable_statement_pct, 2),
            "structural_raw_statement_ceiling_pct": round(structural_statement_ceiling, 2),
            "adjusted_statement_denominator": adjusted_statement_denominator,
            "statement_A_count": len(a_lines), "statement_B_count": len(b_lines),
            "statement_C_count": len(c_lines),
            "statement_A_lines": a_lines, "statement_B_lines": b_lines, "statement_C_lines": c_lines,
            "num_branches": branches, "covered_branches": covered_branches,
            "missing_branches": len(missing_arcs), "branch_pct": round(branch_pct, 2),
            "reachable_branch_pct": round(reachable_branch_pct, 2),
            "structural_raw_branch_ceiling_pct": round(structural_branch_ceiling, 2),
            "adjusted_branch_denominator": adjusted_branch_denominator,
            "branch_A_count": len(a_arcs), "branch_B_count": len(b_arcs), "branch_C_count": len(c_arcs),
            "branch_A_arcs": a_arcs, "branch_B_arcs": b_arcs, "branch_C_arcs": c_arcs,
            "residual_classification": classification,
        }
        # A module whose proven raw structural ceiling stays >=95% keeps the raw floor; a module below it
        # must reach 100% on the reachable denominator with zero Class C (T1819 §5-8, T1905 §2-6).
        if structural_statement_ceiling >= _MIN_COVERAGE:
            if statement_pct < _MIN_COVERAGE:
                failures.append(f"{name}:raw-statement {statement_pct:.2f}% < {_MIN_COVERAGE}%")
        elif covered != adjusted_statement_denominator or c_lines:
            failures.append(
                f"{name}:reachable-statement {reachable_statement_pct:.2f}% "
                f"(covered {covered}/{adjusted_statement_denominator}, Class-C={len(c_lines)}) is not 100% zero-C")
        if structural_branch_ceiling >= _MIN_COVERAGE:
            if branch_pct < _MIN_COVERAGE:
                failures.append(f"{name}:raw-branch {branch_pct:.2f}% < {_MIN_COVERAGE}%")
        elif covered_branches != adjusted_branch_denominator or c_arcs:
            failures.append(
                f"{name}:reachable-branch {reachable_branch_pct:.2f}% "
                f"(covered {covered_branches}/{adjusted_branch_denominator}, Class-C={len(c_arcs)}) is not 100% zero-C")
    if unclassified:
        raise RuntimeError(f"unclassified coverage residuals (need A/B/C): {unclassified}")
    if failures:
        raise RuntimeError(f"per-module coverage ruling not met: {failures}: {per_module}")
    return per_module


def main() -> int:
    actual = {path.name for path in (ROOT / "tests/stage4_installed_gates").glob("*_gate.py")}
    if actual != EXPECTED_GATES:
        raise RuntimeError(f"D4-A gate inventory drift: expected={sorted(EXPECTED_GATES)} actual={sorted(actual)}")
    for base in ("snn_memory_checkpoint_bundle_v2.schema.json", "snn_memory_candidate_bank_v2.schema.json"):
        _schema_pair_parity(base)
        _schema_pair_parity(base + ".license")
    encoder_digest = _pinned_encoder_digest()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="remanentia-d4a-", dir=RUNTIME))
    rust_wheels = workspace / "rust-wheels"
    python_wheels = workspace / "python-wheels"
    install_target = workspace / "installed"
    fixtures = workspace / "fixtures"
    for directory in (rust_wheels, python_wheels, install_target, fixtures):
        directory.mkdir()
    _run(
        ["maturin", "build", "--release", "--locked", "--offline", "--out", str(rust_wheels)],
        ROOT / "rust_snn_memory",
    )
    _run(
        [
            sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
            "--no-build-isolation", "--wheel-dir", str(python_wheels), str(ROOT),
        ],
        workspace,
    )
    rust_wheel = _one(rust_wheels, "rust_snn_memory-*.whl")
    python_wheel = _one(python_wheels, "remanentia-*.whl")
    _run(
        [
            sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
            "--target", str(install_target), str(python_wheel), str(rust_wheel),
        ],
        workspace,
    )
    extension = _one(install_target / "rust_snn_memory", "rust_snn_memory*.so")
    modules = {name: (install_target / "snn_memory" / f"{name}.py").resolve(strict=True) for name in _MODULES}
    include = ",".join(str(path) for path in modules.values())
    extension_sha256 = _sha(extension)
    coverage_data = workspace / ".coverage"
    rcfile, site_dir = _write_coverage_harness(workspace, coverage_data, include)
    gate_tmp = workspace / "tmp"
    gate_tmp.mkdir()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(site_dir), str(install_target)])
    environment["COVERAGE_PROCESS_START"] = str(rcfile)
    # TMPDIR is job-local inside the monitored workspace so temp usage counts against the disk cap.
    environment["TMPDIR"] = str(gate_tmp)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    # Silence transformers/safetensors progress + advisory logging so the gate's own D2 encoder
    # load and every child leave stderr empty for the controlled-output checks.
    environment["TQDM_DISABLE"] = "1"
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    environment["TRANSFORMERS_VERBOSITY"] = "error"
    environment["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    # The gate parent and every CLI/calibration child auto-start coverage via sitecustomize +
    # COVERAGE_PROCESS_START; the verifier supervises the whole process group and fails closed.
    supervision = _supervise(
        [
            sys.executable, str(GATE),
            "--workspace", str(fixtures),
            "--install-target", str(install_target),
            "--repo-root", str(ROOT),
            "--extension-sha256", extension_sha256,
            "--encoder-checkpoint", str(MODEL),
            "--encoder-digest", encoder_digest,
            "--python-wheel-sha256", _sha(python_wheel),
            "--rust-wheel-sha256", _sha(rust_wheel),
        ],
        workspace, environment, workspace,
    )
    _run([sys.executable, "-m", "coverage", "combine", f"--rcfile={rcfile}",
          f"--data-file={coverage_data}"], workspace, environment)
    per_module = _per_module_coverage(rcfile, coverage_data, modules, workspace, environment)
    gate_report = supervision["gate_report"]
    observed = set(supervision["observed_pids"])
    root_pid = int(supervision["root_pid"])
    if gate_report.get("status") != "pass":
        raise RuntimeError(f"gate did not report pass: {gate_report.get('status')}")
    if int(gate_report["gate_pid"]) != root_pid:
        raise RuntimeError("gate report gate_pid differs from the supervised root PID")
    if gate_report["seeds"] != [11, 29]:
        raise RuntimeError(f"gate report seeds are not [11, 29]: {gate_report['seeds']}")
    if gate_report["conditions"] != ["trained", "shuffled", "random", "zero", "untrained"]:
        raise RuntimeError(f"gate report conditions differ: {gate_report['conditions']}")
    if gate_report["candidate_count"] != _CANDIDATE_COUNT:
        raise RuntimeError(f"gate candidate count is not {_CANDIDATE_COUNT}: {gate_report['candidate_count']}")
    if gate_report["n_neurons"] != _N_NEURONS:
        raise RuntimeError(f"gate n_neurons is not {_N_NEURONS}: {gate_report['n_neurons']}")
    if gate_report["materializer_run_count"] != _MATERIALIZER_RUNS:
        raise RuntimeError(f"materializer run count is not {_MATERIALIZER_RUNS}: "
                           f"{gate_report['materializer_run_count']}")
    expected_workers = _MATERIALIZER_RUNS * _CANDIDATE_COUNT + 1
    if gate_report["worker_child_count"] != expected_workers:
        raise RuntimeError(f"worker child count is not {expected_workers}: "
                           f"{gate_report['worker_child_count']}")
    materializer_pids = gate_report["materializer_pids"]
    worker_pids = gate_report["worker_pids"]
    if len(materializer_pids) != gate_report["materializer_run_count"]:
        raise RuntimeError("materializer PIDs are not unique across runs")
    if len(worker_pids) != gate_report["worker_child_count"]:
        raise RuntimeError("worker PIDs are not unique across children")
    if not set(materializer_pids).isdisjoint(worker_pids):
        raise RuntimeError("materializer and worker PID sets overlap")
    reported = set(materializer_pids) | set(worker_pids)
    unaccounted = reported - observed
    if unaccounted:
        raise RuntimeError(f"gate-reported PIDs not observed by the supervisor: {sorted(unaccounted)}")
    if root_pid not in observed:
        raise RuntimeError("supervised root PID was not observed in the process group")
    report = {
        "status": "pass",
        "encoder_digest": encoder_digest,
        "extension_origin": str(extension),
        "extension_sha256": extension_sha256,
        "install_prefix": str(install_target),
        "python_wheel_sha256": _sha(python_wheel),
        "rust_wheel_sha256": _sha(rust_wheel),
        "supervision": supervision,
        "coverage": per_module,
    }
    for name, path in modules.items():
        report[f"{name}_module_sha256"] = _sha(path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
