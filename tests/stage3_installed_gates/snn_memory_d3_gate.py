# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Installed-wheel D3 experiment-lock and G-B preflight gate

"""Exercise D3 only through installed wheels, real Git/D1/D2, and the real Rust engine.

Every positive surface is driven through the installed public API and CLI: a real
temporary Git repository feeds the installed D1 selector and D2 materialiser with
the real pinned local encoder; the resulting canonical D1/D2 artifacts are
authenticated through their public readers and bound into the lock; the installed
experiment-lock writer/reader/completeness/bind surfaces run in real child
processes with distinct PIDs; and two fresh installed-wheel child processes each
run the identical real Rust G-B fixture and produce byte-identical sealed evidence
artifacts. No mock, monkeypatch, fabricated current matrix, source-tree import, or
Python fallback stands in for a production surface.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from snn_memory import experiment_lock as el
from snn_memory import gb_preflight as gb
from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.cue_materializer import materialize_cue_set, read_cue_bundle, read_cue_set
from snn_memory.experiment_lock import (
    ExperimentLockError,
    bind_d1_d2,
    bind_lane_isolation,
    bind_task_completeness,
    canonical_config_digest,
    read_artifact,
    require_dev_namespace_disjoint,
    task_identity_digest,
    task_set_digest,
    validate_artifact_bytes,
    write_artifact,
)
from snn_memory.gb_preflight import GbPreflightError, evaluate_gb_descriptor, model_config_digest
from snn_memory.source_universe import validate_source_universe_bytes, write_source_universe
from snn_memory.stream_backend import BackendIdentity, load_stream_backend

BASE_TIMESTAMP = 1_700_000_000
SHA = "a" * 64
WORD_A = (
    "ledger",
    "orchard",
    "harbour",
    "granite",
    "willow",
    "meadow",
    "signal",
    "quarry",
    "lantern",
    "compass",
    "thicket",
    "estuary",
    "paddock",
    "furnace",
    "gable",
    "cistern",
)
WORD_B = ("morning", "evening", "quiet", "sudden", "steady", "distant", "narrow")
WORD_C = (
    "ridge",
    "valley",
    "coast",
    "plain",
    "summit",
    "delta",
    "basin",
    "moor",
    "heath",
    "fen",
    "loch",
    "strand",
    "combe",
    "tor",
    "glen",
    "wold",
)
GB_CONFIG = ModelConfig(n_neurons=32, excitatory_fraction=0.75, connectivity=0.25, dt_ms=1.0)
GB_SEED = 7
GB_CUE_STEPS = 8
GB_COMPLETION_STEPS = 32
GB_SPIKE_CEILING = 0.05
GB_CURRENT_CEILING = 0.05
GB_ZERO_FLOOR = 1e-09


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expect(substring: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except (ExperimentLockError, GbPreflightError) as error:
        if substring not in str(error):
            raise AssertionError(f"unexpected error {error!r}, wanted {substring!r}") from error
        return
    raise AssertionError(f"expected a fail-closed error containing {substring!r}")


def _git(repo: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, check=True
    ).stdout


def _doc(record: int) -> bytes:
    lines = [
        f"Record {record} event {index:03d} explores {WORD_A[record]} {WORD_B[index % 7]} "
        f"phenomena beside {WORD_C[record]} landmark mark-{record}-{index}."
        for index in range(52)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_repo(repo: Path, salt: bytes) -> None:
    docs = repo / "docs/public"
    docs.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.name", "D3 Gate")
    _git(repo, "config", "user.email", "d3@example.invalid")
    for record in range(16):
        (docs / f"record-{record:02d}.md").write_bytes(_doc(record) + salt)
    extra = docs / "manifest-source.md"
    extra.write_bytes(_doc(0).replace(b"Record 0", b"Source X") + salt)
    mdir = repo / "experiments/snn_memory"
    mdir.mkdir(parents=True)
    development = {
        "schema_version": 1,
        "split": "development",
        "encoder_checkpoint": "../../.snn_models/model",
        "encoder_digest": "1" * 64,
        "entries": [
            {
                "label": "v1-dev-0",
                "path": "../../docs/public/manifest-source.md",
                "sha256": _sha(extra.read_bytes()),
            }
        ],
    }
    (mdir / "development_corpus.json").write_bytes(json.dumps(development, sort_keys=True).encode())
    locked = {
        "schema_version": 1,
        "split": "locked-evaluation",
        "locked": True,
        "encoder_checkpoint": "../../.snn_models/model",
        "encoder_digest": "1" * 64,
        "entries": [],
    }
    (mdir / "locked_evaluation_corpus.json").write_bytes(
        json.dumps(locked, sort_keys=True).encode()
    )
    _git(repo, "add", "-A")
    env = dict(
        os.environ,
        GIT_AUTHOR_DATE=f"@{BASE_TIMESTAMP} +0000",
        GIT_COMMITTER_DATE=f"@{BASE_TIMESTAMP} +0000",
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], env=env, check=True)


def _fixture_repo(parent: Path) -> Path:
    repo = parent / "d3-repo"
    _build_repo(repo, b"")
    return repo


# ---------------------------------------------------------------------------
# Real D1/D2 acquisition and lock construction
# ---------------------------------------------------------------------------


def _acquire_d1_d2(
    workspace: Path, repo: Path, encoder_checkpoint: Path, encoder_digest: str
) -> tuple[Any, Any, str, list[Any]]:
    universe_path = workspace / "universe.json"
    d1_result = write_source_universe(repo, universe_path)
    d1_art = validate_source_universe_bytes(
        universe_path.read_bytes(), expected_file_sha256=d1_result.file_sha256
    )
    output = workspace / "cue-set"
    d2_result = materialize_cue_set(
        repo,
        universe_path,
        d1_result.file_sha256,
        encoder_checkpoint,
        encoder_digest,
        output,
        model=ModelConfig(n_neurons=64),
        encoder_config=EncoderConfig(feature_dim=16, packet_ms=5, silent_ms=2, active_fraction=0.1),
        input_current=18.0,
    )
    d2_art = read_cue_set(output / "cue_set.json", d2_result.file_sha256)
    # Authenticate every required cue bundle exactly once through the public reader and
    # retain the validated bundle payloads for the full locked-D2 identity inventory.
    inventory = el.cue_bundle_inventory(d2_art.payload)
    assert inventory, "the authenticated cue set declares no cue bundles"
    seen: set[str] = set()
    bundle_arts: list[Any] = []
    for path, sha256 in inventory:
        if path in seen:
            continue
        seen.add(path)
        bundle_arts.append(read_cue_bundle(output / path, sha256))
    bundle_inventory_digest = el.cue_bundle_inventory_digest(d2_art.payload)
    return d1_art, d2_art, bundle_inventory_digest, bundle_arts


def _foreign_d1(workspace: Path) -> Any:
    """Build a second independent fixture repository and return only its D1 artifact.

    Used to prove foreign-root rejection without a second encoder pass: the salted
    corpus yields disjoint record digests and HEAD, so a lock bound to the primary
    source universe must reject this D1 source root.
    """
    repo = workspace / "d3-foreign"
    _build_repo(repo, b"\n<!-- foreign source root -->\n")
    universe_path = workspace / "foreign_universe.json"
    d1_result = write_source_universe(repo, universe_path)
    return validate_source_universe_bytes(
        universe_path.read_bytes(), expected_file_sha256=d1_result.file_sha256
    )


def _foreign_bundle(workspace: Path, encoder_checkpoint: Path, encoder_digest: str) -> Any:
    """Materialise a second independent cue set and return one authenticated bundle from it.

    The salted foreign corpus yields cue bundles whose file digests are disjoint from the
    primary cue-set inventory, so a bundle from here is an unexpected/substituted bundle the
    binder must reject.
    """
    repo = workspace / "d3-foreign-cues"
    _build_repo(repo, b"\n<!-- foreign cue source root -->\n")
    universe_path = workspace / "foreign_cue_universe.json"
    d1_result = write_source_universe(repo, universe_path)
    output = workspace / "foreign-cue-set"
    d2_result = materialize_cue_set(
        repo,
        universe_path,
        d1_result.file_sha256,
        encoder_checkpoint,
        encoder_digest,
        output,
        model=ModelConfig(n_neurons=64),
        encoder_config=EncoderConfig(feature_dim=16, packet_ms=5, silent_ms=2, active_fraction=0.1),
        input_current=18.0,
    )
    d2_art = read_cue_set(output / "cue_set.json", d2_result.file_sha256)
    path, sha256 = el.cue_bundle_inventory(d2_art.payload)[0]
    return read_cue_bundle(output / path, sha256)


def _check_bundle_completeness_negatives(
    lock: Any,
    scoring: Any,
    calibration: Any,
    d1_art: Any,
    d2_art: Any,
    bundle_arts: list[Any],
    foreign_bundle: Any,
) -> None:
    """The binder must reject any bundle set that does not exactly reconstruct the cue inventory."""
    assert len(bundle_arts) >= 2, "the cue set must declare at least two bundles"
    message = "bound bundles do not reconstruct the authenticated cue-set inventory"

    def bind(bundles: list[Any]) -> None:
        bind_lane_isolation(lock, scoring, calibration, d1_art, d2_art, bundles)

    # An omitted, duplicated, unexpected (foreign), or substituted (foreign) bundle is rejected.
    _expect(message, lambda: bind(bundle_arts[:-1]))
    _expect(message, lambda: bind(bundle_arts + [bundle_arts[0]]))
    _expect(message, lambda: bind(bundle_arts + [foreign_bundle]))
    _expect(message, lambda: bind(bundle_arts[:-1] + [foreign_bundle]))


def _d1_binding(d1_art: Any) -> dict[str, Any]:
    return {
        "file_sha256": d1_art.file_sha256,
        "payload_self_sha256": d1_art.payload_self_sha256,
        "repository_head": d1_art.payload["repository"]["head"],
        "selected_record_ids": list(d1_art.payload["selected_record_ids"]),
    }


def _d2_binding(
    d1_binding: dict[str, Any], d2_art: Any, bundle_inventory_digest: str
) -> dict[str, Any]:
    return {
        "cue_set_file_sha256": d2_art.file_sha256,
        "cue_set_payload_self_sha256": d2_art.payload_self_sha256,
        "bundle_inventory_digest": bundle_inventory_digest,
        "source_universe": dict(d1_binding),
    }


def _module_identities(bindings: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_sha256": bindings["schema_sha256"],
        "python_wheel_sha256": bindings["python_wheel"],
        "rust_wheel_sha256": bindings["rust_wheel"],
        "backend_extension_sha256": bindings["extension"],
        "experiment_lock_module_sha256": bindings["lock_module"],
        "gb_preflight_module_sha256": bindings["gb_module"],
    }


def _scoring_target(record_ids: list[str], bindings: dict[str, str]) -> dict[str, Any]:
    ordered = sorted(record_ids)
    signatures = [_sha(rid.encode()) for rid in ordered]
    identities = _module_identities(bindings)
    identities["candidate_bank_digest"] = el.candidate_bank_digest(ordered, signatures)
    base = {
        "schema_version": 2,
        "artifact_type": "snn-memory-scoring-target-v2",
        "state": "fixture_only",
        "lane_role": "lane_p",
        "completion_steps": 32,
        "bins": 8,
        "signature_dtype": "<f8",
        "similarity": "cosine",
        "zero_norm_rule": "zero-score-when-either-norm-is-zero",
        "score_order": "descending",
        "tie_rule": "lexical-record-id",
        "abstention": {"threshold": 0.1, "rule": "strict-greater-than"},
        "correctness_rule": "exact-record-id",
        "top_k": 5,
        "max_payload_utf8_bytes": 20000,
        "candidate_order": ordered,
        "candidate_signature_digests": signatures,
        "identities": identities,
    }
    base["scorer_digest"] = el.scorer_identity_digest(base)
    return base


def _calibration_spec(bindings: dict[str, str]) -> dict[str, Any]:
    positives = el.synthetic_positive_ids()
    negatives = el.synthetic_negative_ids(positives)
    selection_digest, validation_digest = el.synthetic_partition_digests(positives)
    identities = _module_identities(bindings)
    identities["rule_version"] = "gb-descriptor-v2"
    return {
        "schema_version": 2,
        "artifact_type": "snn-memory-gb-calibration-spec-v2",
        "state": "fixture_only",
        "lane_role": "lane_p",
        "epsilon": 2.220446049250313e-16,
        "epsilon_bits": "3cb0000000000000",
        "numerical_zero_floor_grid": [1e-09],
        "completion_steps": 32,
        "bins": 8,
        "tail_bins": [5, 6, 7, 8],
        "candidate_lags": [1, 2, 3, 4],
        "spike_drift_ceiling_grid": [0.05],
        "current_drift_ceiling_grid": [0.05],
        "representation_margin_floor_grid": [0.05],
        "normalized_effective_rank_floor_grid": [0.1],
        "settled_fraction_floor_grid": [0.5],
        "wandering_ceiling_grid": [0.3],
        "collapse_ceiling_grid": [0.3],
        "abstention_grid": [0.0],
        "selection_objective": "macro-youden-j",
        "tie_rule": "ascending-canonical-json",
        "validation_floors": {
            "sensitivity_min": 0.8,
            "specificity_min": 0.95,
            "false_recall_max": 0.05,
        },
        "synthetic_generator": {
            "identity": "synthetic-g1-v2",
            "namespace": "dev:",
            "config_digest": el.synthetic_config_digest(),
            "per_seed_positive_count": 16,
            "selection_partition_digest": selection_digest,
            "validation_partition_digest": validation_digest,
            "positive_ids": positives,
            "negative_ids": negatives,
            "negative_families": ["shuffled", "zero_recurrence", "no_input", "no_match"],
        },
        "identities": identities,
        "task_set_digest": el.synthetic_task_set_digest(positives, negatives),
    }


def _identities(bindings: dict[str, str], head: str) -> dict[str, Any]:
    return {
        "main_plan_sha256": "24c41750848b3fd94ee25ee638458b0654365cead025b18f2488840a182241c6",
        "amendment1_sha256": "e1b860b89ae68832abc192dc8a8a39b69dc62e3e7edbc494022a76e5c32fa5a5",
        "amendment2_final_sha256": el.AMENDMENT2_FINAL_SHA256,
        "amendment2_u2_sha256": el.AMENDMENT2_U2_SHA256,
        "amendment2_digest_pin_sha256": el.AMENDMENT2_DIGEST_PIN_SHA256,
        "gb_design_input_sha256": el.GB_DESIGN_INPUT_SHA256,
        "schema_sha256": bindings["schema_sha256"],
        "python_wheel_sha256": bindings["python_wheel"],
        "rust_wheel_sha256": bindings["rust_wheel"],
        "backend_extension_sha256": bindings["extension"],
        "encoder_directory_sha256": bindings["encoder_digest"],
        "repository_head": head,
        "dirty_tree_digest": bindings["dirty_tree_digest"],
        "experiment_lock_module_sha256": bindings["lock_module"],
        "gb_preflight_module_sha256": bindings["gb_module"],
    }


_LANE_DISTINGUISHING = ("root", "task_set", "threshold", "candidate_bank", "scorer")

_LANE_H_MISSING_INVENTORY = [
    "historical_git_object",
    "historical_embedding",
    "historical_configuration",
    "historical_dependency_environment",
    "historical_ranking_output",
    "historical_output_root",
    "historical_provenance",
]
_LANE_H_MISSING_REASON = (
    "no authenticated exact-historical Git-object forensic package is available for Lane H in D3"
)


def _framed(label: str) -> str:
    return _sha(("remanentia:snn-v2-d3-gate:" + label).encode())


def _measure_fixture(root: Path, name: str, content: bytes) -> str:
    """Write genuinely distinct temporary fixture bytes and return their measured SHA-256."""
    path = root / name
    path.write_bytes(content)
    return _sha(path.read_bytes())


def _build_lane_c_diagnostic_manifest(workspace: Path) -> dict[str, Any]:
    """Measure a genuinely disjoint temporary Lane-C confound diagnostic manifest.

    Every identity-bearing component is the measured canonical digest of distinct temporary
    diagnostic fixture bytes under the runtime workspace — not a free-form label — so an actual
    Lane-C diagnostic schema/root/etc. digest is a real value banned from Lane P.
    """
    root = workspace / "lane-c-diagnostic"
    root.mkdir(parents=True, exist_ok=True)

    def digest(component: str) -> str:
        return _measure_fixture(
            root,
            f"{component}.bin",
            f"remanentia lane-c confound diagnostic :: {component}\n".encode(),
        )

    return {
        "lane_role": "lane_c",
        "diagnostic_schema_sha256": digest("diagnostic_schema"),
        "experiment_config_sha256": digest("experiment_config"),
        "output_root_sha256": digest("output_root"),
        "task_set_sha256": digest("task_set"),
        "threshold_sha256": digest("threshold"),
        "candidate_bank_sha256": digest("candidate_bank"),
        "scorer_sha256": digest("scorer"),
        "provenance_sha256": digest("provenance"),
    }


def _build_lane_h_historical_manifest(workspace: Path) -> dict[str, Any]:
    """Measure a fixture-only admissible Lane-H historical manifest from distinct temporary bytes.

    Used only by the admissible-H positive/negatives; the normal D3 Lane-H state stays
    inadmissible. A measured SHA-256 is a valid Git SHA-256 object id, and every other component
    is a measured digest of distinct fixture bytes — never a label pretending to be real history.
    """
    root = workspace / "lane-h-historical"
    root.mkdir(parents=True, exist_ok=True)

    def digest(component: str) -> str:
        return _measure_fixture(
            root,
            f"{component}.bin",
            f"remanentia lane-h historical fixture :: {component}\n".encode(),
        )

    return {
        "lane_role": "lane_h",
        "historical_git_object": digest("git_object"),
        "historical_embedding_sha256": digest("embedding"),
        "historical_configuration_sha256": digest("configuration"),
        "historical_dependency_environment_sha256": digest("dependency_environment"),
        "historical_ranking_output_sha256": digest("ranking_output"),
        "historical_output_root_sha256": digest("output_root"),
        "historical_provenance_sha256": digest("provenance"),
    }


def _foreign_lanes(lane_c_manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the bound foreign-lane evidence: a measured Lane-C manifest and an inadmissible H."""
    return {
        "lane_c": el.build_foreign_lane_c(lane_c_manifest),
        "lane_h": el.build_foreign_lane_h_inadmissible(
            _LANE_H_MISSING_REASON, _LANE_H_MISSING_INVENTORY
        ),
    }


def _lock_payload(
    d1_binding: dict[str, Any],
    d2_binding: dict[str, Any],
    identities: dict[str, Any],
    bindings: dict[str, str],
    real: dict[str, str],
    foreign_lanes: dict[str, Any],
    lane: str = "lane_p",
) -> dict[str, Any]:
    order = list(d1_binding["selected_record_ids"])
    order_digest = el.candidate_order_digest(order)
    schema = bindings["schema_sha256"]
    lane_identity = {
        "root": real["root"],
        "task_set": real["task_set"],
        "threshold": real["threshold"],
        "candidate_bank": real["candidate_bank"],
        "scorer": real["scorer"],
        "schema": schema,
        "provenance": order_digest,
    }
    lane_block = [
        lane_identity[field]
        for field in (
            "root",
            "task_set",
            "threshold",
            "candidate_bank",
            "scorer",
            "schema",
            "provenance",
        )
    ]
    return {
        "schema_version": 2,
        "artifact_type": "snn-memory-experiment-lock-v2",
        "state": "fixture_only",
        "lane_role": lane,
        "seeds": list(el.SEEDS),
        "identities": identities,
        "d1": dict(d1_binding),
        "d2": dict(d2_binding),
        "scoring_target_digest": real["scorer"],
        "calibration_spec_digest": real["calibration_self"],
        "candidate_order": order,
        "candidate_order_digest": order_digest,
        "expected_task_set_digest": real["task_set"],
        "lane_role_digest": el.lane_role_digest(lane),
        "lane_identity": lane_identity,
        "lane_domain_digest": el.lane_domain_digest(lane, lane_block),
        "foreign_lanes": foreign_lanes,
        "output_root_digest": real["root"],
    }


def _lane_real(
    scoring: Any, calibration: Any, bindings: dict[str, str], expected_task_set_digest: str
) -> dict[str, str]:
    """Build the real Lane-P identity values measured from the authenticated artifacts."""
    return {
        "root": bindings["output_root_digest"],
        "task_set": expected_task_set_digest,
        "threshold": el.calibration_threshold_digest(calibration.payload),
        "candidate_bank": scoring.payload["identities"]["candidate_bank_digest"],
        "scorer": scoring.payload_self_sha256,
        "calibration_self": calibration.payload_self_sha256,
    }


def _tasks(cues: tuple[str, str], lane: str = "lane_p") -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for seed in el.SEEDS:
        for condition, cue in (("trained", cues[0]), ("shuffled", cues[1])):
            tasks.append(
                {
                    "task_id": task_identity_digest(seed, lane, condition, cue),
                    "seed": seed,
                    "lane_role": lane,
                    "condition": condition,
                    "cue_digest": cue,
                }
            )
    return tasks


def _process_instance(task_id: str, index: int) -> str:
    return _sha(f"remanentia:snn-v2-process:{task_id}:{index}".encode())


def _process_records(task_ids: list[str]) -> list[dict[str, str]]:
    return [
        {"task_id": task_id, "process_instance": _process_instance(task_id, index)}
        for index, task_id in enumerate(task_ids)
    ]


def _expected_task_set_digest(cues: tuple[str, str]) -> str:
    return task_set_digest([task["task_id"] for task in _tasks(cues)])


def _completeness_payload(
    complete: bool, cues: tuple[str, str], lane: str = "lane_p"
) -> dict[str, Any]:
    tasks = _tasks(cues, lane)
    ids = [task["task_id"] for task in tasks]
    payload = {
        "schema_version": 2,
        "artifact_type": "snn-memory-task-completeness-v2",
        "state": "fixture_only",
        "lane_role": lane,
        "seeds": list(el.SEEDS),
        "task_set_digest": task_set_digest(ids),
        "expected_tasks": tasks,
        "completed_task_ids": list(ids),
        "missing_task_ids": [],
        "unexpected_task_ids": [],
        "digest_failure_task_ids": [],
        "process_records": _process_records(ids),
        "completeness": True,
        "abort_reason": None,
    }
    if not complete:
        payload["completed_task_ids"] = ids[:-1]
        payload["missing_task_ids"] = [ids[-1]]
        payload["process_records"] = _process_records(ids)[:-1]
        payload["completeness"] = False
        payload["abort_reason"] = "expected_task_incompleteness"
    return payload


def _craft(payload: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> bytes:
    body = json.loads(json.dumps(payload))
    mutate(body)
    return el.seal_payload(body)


# ---------------------------------------------------------------------------
# CLI: real child processes and in-process runpy coverage
# ---------------------------------------------------------------------------


def _cli(
    install_target: Path, arguments: list[str], expected_code: int
) -> tuple[dict[str, Any], int]:
    environment = dict(os.environ, PYTHONPATH=str(install_target))
    process = subprocess.Popen(
        [sys.executable, "-m", "snn_memory.experiment_lock", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == expected_code, stderr.decode()
    payload = json.loads(stdout) if process.returncode == 0 and stdout.strip() else {}
    return payload, process.pid


def _module_cli(arguments: list[str], expected_code: int) -> str:
    import contextlib
    import io
    import runpy

    stdout = io.StringIO()
    stderr = io.StringIO()
    previous = sys.argv
    sys.argv = ["python -m snn_memory.experiment_lock", *arguments]
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            runpy.run_module("snn_memory.experiment_lock", run_name="__main__", alter_sys=False)
        raise AssertionError("experiment_lock __main__ did not exit")
    except SystemExit as exit_info:
        assert exit_info.code == expected_code, f"CLI exit {exit_info.code}: {stderr.getvalue()}"
    finally:
        sys.argv = previous
    return stdout.getvalue()


def _check_cli(
    workspace: Path,
    install_target: Path,
    record_ids: list[str],
    lock: Any,
    bindings: dict[str, str],
) -> list[int]:
    pids: list[int] = []
    # Mirror the authenticated D1/D2/bundle inputs written by `_acquire_d1_d2`; the installed
    # bind CLI must authenticate all three through the public D1/D2 readers.
    cue_set_dir = workspace / "cue-set"
    bind_inputs = [
        "--d1",
        str(workspace / "universe.json"),
        "--cue-set",
        str(cue_set_dir / "cue_set.json"),
        "--bundle-root",
        str(cue_set_dir),
    ]
    payload_file = workspace / "scoring_payload.json"
    payload_file.write_bytes(json.dumps(_scoring_target(record_ids, bindings)).encode())
    report, pid = _cli(
        install_target,
        ["write", "--payload", str(payload_file), "--output", str(workspace / "cli_scoring.json")],
        0,
    )
    pids.append(pid)
    assert report["artifact_type"] == "snn-memory-scoring-target-v2"
    report, pid = _cli(
        install_target,
        [
            "read",
            "--artifact",
            str(workspace / "lock.json"),
            "--expected-type",
            "snn-memory-experiment-lock-v2",
            "--expected-sha256",
            lock.file_sha256,
        ],
        0,
    )
    pids.append(pid)
    report, pid = _cli(
        install_target, ["completeness", "--artifact", str(workspace / "complete.json")], 0
    )
    pids.append(pid)
    assert report["completeness"] is True
    report, pid = _cli(
        install_target,
        [
            "bind",
            "--lock",
            str(workspace / "lock.json"),
            "--scoring-target",
            str(workspace / "scoring.json"),
            "--calibration-spec",
            str(workspace / "calibration.json"),
            *bind_inputs,
        ],
        0,
    )
    pids.append(pid)
    assert report["bound"] is True
    # A CLI bind whose bundle root omits the authenticated bundles must fail, not report bound.
    _, pid = _cli(
        install_target,
        [
            "bind",
            "--lock",
            str(workspace / "lock.json"),
            "--scoring-target",
            str(workspace / "scoring.json"),
            "--calibration-spec",
            str(workspace / "calibration.json"),
            "--d1",
            str(workspace / "universe.json"),
            "--cue-set",
            str(cue_set_dir / "cue_set.json"),
            "--bundle-root",
            str(workspace / "no-bundles"),
        ],
        2,
    )
    pids.append(pid)
    _, pid = _cli(
        install_target,
        ["read", "--artifact", str(workspace / "lock.json"), "--expected-sha256", "0" * 64],
        2,
    )
    pids.append(pid)

    # In-process runpy for installed-module coverage of main() + handlers + error exit.
    out = _module_cli(
        [
            "write",
            "--payload",
            str(payload_file),
            "--output",
            str(workspace / "inproc_scoring.json"),
        ],
        0,
    )
    assert json.loads(out)["artifact_type"] == "snn-memory-scoring-target-v2"
    out = _module_cli(["read", "--artifact", str(workspace / "lock.json")], 0)
    assert json.loads(out)["file_sha256"] == lock.file_sha256
    out = _module_cli(
        [
            "read",
            "--artifact",
            str(workspace / "lock.json"),
            "--expected-type",
            "snn-memory-experiment-lock-v2",
            "--expected-sha256",
            lock.file_sha256,
        ],
        0,
    )
    assert json.loads(out)["file_sha256"] == lock.file_sha256
    out = _module_cli(["completeness", "--artifact", str(workspace / "complete.json")], 0)
    assert json.loads(out)["completeness"] is True
    out = _module_cli(
        [
            "bind",
            "--lock",
            str(workspace / "lock.json"),
            "--scoring-target",
            str(workspace / "scoring.json"),
            "--calibration-spec",
            str(workspace / "calibration.json"),
            *bind_inputs,
        ],
        0,
    )
    assert json.loads(out)["bound"] is True
    # In-process bind whose bundle root omits the bundles: the D2/bundle authentication failure
    # is surfaced as a bind error (exit 2), not a successful bind.
    _module_cli(
        [
            "bind",
            "--lock",
            str(workspace / "lock.json"),
            "--scoring-target",
            str(workspace / "scoring.json"),
            "--calibration-spec",
            str(workspace / "calibration.json"),
            "--d1",
            str(workspace / "universe.json"),
            "--cue-set",
            str(cue_set_dir / "cue_set.json"),
            "--bundle-root",
            str(workspace / "no-bundles"),
        ],
        2,
    )
    _module_cli(["read", "--artifact", str(workspace / "cli-missing.json")], 2)
    return pids


# ---------------------------------------------------------------------------
# Two-child deterministic G-B evidence + independent U-2 recompute + witness
# ---------------------------------------------------------------------------


def _gb_child_argv(
    extension_sha256: str, output: Path, calibration_digest: str, scoring_digest: str
) -> list[str]:
    return [
        "--extension-sha256",
        extension_sha256,
        "--seed",
        str(GB_SEED),
        "--cue-steps",
        str(GB_CUE_STEPS),
        "--completion-steps",
        str(GB_COMPLETION_STEPS),
        "--n-neurons",
        str(GB_CONFIG.n_neurons),
        "--excitatory-fraction",
        str(GB_CONFIG.excitatory_fraction),
        "--connectivity",
        str(GB_CONFIG.connectivity),
        "--dt-ms",
        str(GB_CONFIG.dt_ms),
        "--spike-drift-ceiling",
        str(GB_SPIKE_CEILING),
        "--current-drift-ceiling",
        str(GB_CURRENT_CEILING),
        "--numerical-zero-floor",
        str(GB_ZERO_FLOOR),
        "--lane-role",
        "lane_p",
        "--state",
        "fixture_only",
        "--calibration-spec-digest",
        calibration_digest,
        "--scoring-target-digest",
        scoring_digest,
        "--output",
        str(output),
    ]


def _gb_child(
    install_target: Path,
    output: Path,
    extension_sha256: str,
    calibration_digest: str,
    scoring_digest: str,
) -> tuple[bytes, str, int]:
    environment = dict(os.environ, PYTHONPATH=str(install_target))
    argv = [
        sys.executable,
        "-m",
        "snn_memory.gb_preflight",
        *_gb_child_argv(extension_sha256, output, calibration_digest, scoring_digest),
    ]
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0, stderr.decode()
    report = json.loads(stdout)
    return output.read_bytes(), report["recurrent_energy_bits"], process.pid


# Independent gate reimplementation of the G-B descriptor equations. It shares no code
# with snn_memory.gb_preflight or the snn_memory.experiment_lock reader mirror; it derives
# every value straight from the real installed StreamResult so the sealed artifact is
# cross-checked by a genuinely third implementation (acceptance criterion 9 / R2-01).
_GATE_TAIL_BINS = (4, 5, 6, 7)
_GATE_LAGS = (1, 2, 3, 4)
_GATE_EPS = 2.220446049250313e-16
_GATE_RECURRENT_COLUMNS = (3, 4, 5, 6, 7, 8, 9, 10, 11)


def _gate_rows(result: Any, cue_steps: int, completion_steps: int) -> Any:
    index = np.arange(cue_steps, cue_steps + completion_steps)
    return np.asarray(result.current_values[index], dtype=np.float64)


def _gate_raster(result: Any, cue_steps: int, completion_steps: int) -> list[frozenset[int]]:
    raster: list[frozenset[int]] = []
    for step in range(completion_steps):
        global_step = cue_steps + step
        start = int(result.spike_offsets[global_step])
        stop = int(result.spike_offsets[global_step + 1])
        raster.append(frozenset(int(n) for n in result.spike_indices[start:stop]))
    return raster


def _gate_energy_bits(rows: Any, dt_ms: float) -> str:
    fold = 0.0
    for value in rows[:, 11]:
        scalar = float(value)
        fold = fold + scalar * scalar
    return struct.pack(">d", float(dt_ms) * fold).hex()


def _gate_half_life(rows: Any, width: int) -> int | None:
    means: list[float] = []
    for bin_index in range(8):
        running = 0.0
        for value in rows[bin_index * width : (bin_index + 1) * width, 11]:
            scalar = float(value)
            running = running + scalar * scalar
        means.append(running / float(width))
    if means[0] == 0.0:
        return 0
    threshold = 0.5 * means[0]
    for bin_index in range(7):
        if means[bin_index] <= threshold and means[bin_index + 1] <= threshold:
            return bin_index * width
    return None


def _gate_symmetric(raster: list[frozenset[int]], high: int, low: int, width: int) -> int:
    total = 0
    for offset in range(width):
        total += len(raster[high * width + offset] ^ raster[low * width + offset])
    return total


def _gate_hamming(raster: list[frozenset[int]], width: int) -> list[int]:
    return [
        _gate_symmetric(raster, tail_bin, tail_bin - lag, width)
        for tail_bin in _GATE_TAIL_BINS
        for lag in _GATE_LAGS
    ]


def _gate_bin_vector(rows: Any, bin_index: int, width: int) -> list[float]:
    vector: list[float] = []
    for row in range(bin_index * width, (bin_index + 1) * width):
        for column in _GATE_RECURRENT_COLUMNS:
            vector.append(float(rows[row, column]))
    return vector


def _gate_l2(vector: list[float]) -> float:
    total = 0.0
    for value in vector:
        total = total + value * value
    return math.sqrt(total)


def _gate_drift(high: list[float], low: list[float]) -> float:
    difference = _gate_l2([a - b for a, b in zip(high, low)])
    denominator = max(_gate_l2(high), _gate_l2(low), _GATE_EPS)
    return difference / denominator


def _gate_lag(
    rows: Any,
    raster: list[frozenset[int]],
    width: int,
    n_neurons: int,
    lag: int,
    spike_ceiling: float,
    current_ceiling: float,
) -> bool:
    denominator = float(width * n_neurons)
    for tail_bin in _GATE_TAIL_BINS:
        if _gate_symmetric(raster, tail_bin, tail_bin - lag, width) / denominator > spike_ceiling:
            return False
        drift = _gate_drift(
            _gate_bin_vector(rows, tail_bin, width), _gate_bin_vector(rows, tail_bin - lag, width)
        )
        if drift > current_ceiling:
            return False
    return True


def _gate_class(
    rows: Any,
    raster: list[frozenset[int]],
    width: int,
    n_neurons: int,
    spike_ceiling: float,
    current_ceiling: float,
    zero_floor: float,
) -> str:
    silent = True
    for step in range(_GATE_TAIL_BINS[0] * width, 8 * width):
        if raster[step] or float(rows[step, 10]) > zero_floor:
            silent = False
            break
    if silent:
        return "silent_decay"
    if _gate_lag(rows, raster, width, n_neurons, 1, spike_ceiling, current_ceiling):
        return "settled_fixed"
    for lag in (2, 3, 4):
        if _gate_lag(rows, raster, width, n_neurons, lag, spike_ceiling, current_ceiling):
            return "settled_periodic"
    return "wandering_active"


def _check_gb_two_child(
    workspace: Path,
    install_target: Path,
    extension_sha256: str,
    calibration_digest: str,
    scoring_digest: str,
) -> dict[str, Any]:
    first_out = workspace / "gb_evidence_a.json"
    second_out = workspace / "gb_evidence_b.json"
    first_bytes, first_bits, first_pid = _gb_child(
        install_target, first_out, extension_sha256, calibration_digest, scoring_digest
    )
    second_bytes, second_bits, second_pid = _gb_child(
        install_target, second_out, extension_sha256, calibration_digest, scoring_digest
    )
    assert first_pid != second_pid, "G-B evidence must come from two distinct child PIDs"
    assert first_bytes == second_bytes, "sealed G-B evidence bytes differ across processes"
    assert _sha(first_bytes) == _sha(second_bytes)
    assert first_bits == second_bits, "recurrent-energy bits differ across processes"
    # read_artifact independently recomputes energy, half-life, trajectory class, and the
    # 16 Hamming values from the stored rows + raster (pure-Python mirror in experiment_lock);
    # a successful read is that reader-side independent recomputation passing.
    evidence = read_artifact(first_out, expected_type="snn-memory-gb-preflight-evidence-v2")
    assert evidence.file_sha256 == _sha(first_bytes)
    # Independent U-2 recompute from the real installed telemetry.
    backend = load_stream_backend(BackendIdentity(2, "0.1.0", extension_sha256))
    inputs = gb.build_episode(GB_CONFIG, GB_SEED, GB_CUE_STEPS, GB_COMPLETION_STEPS)
    result = backend.run(inputs, GB_CUE_STEPS, False, GB_CONFIG)
    completion = np.flatnonzero(result.phases == 1)
    fold = 0.0
    for timestep in completion:
        r_t = float(result.current_values[timestep, 11])
        fold = fold + r_t * r_t
    energy = GB_CONFIG.dt_ms * fold
    assert struct.pack(">d", energy).hex() == first_bits, (
        "independent U-2 fold differs from evidence"
    )
    assert all(float(result.current_values[t, 1]) == 0.0 for t in completion)
    # Independent gate recompute of the full descriptor from the same real telemetry, using
    # a third implementation that calls neither production helper.
    rows = _gate_rows(result, GB_CUE_STEPS, GB_COMPLETION_STEPS)
    raster = _gate_raster(result, GB_CUE_STEPS, GB_COMPLETION_STEPS)
    width = GB_COMPLETION_STEPS // 8
    assert _gate_energy_bits(rows, GB_CONFIG.dt_ms) == evidence.payload["recurrent_energy_bits"]
    assert _gate_half_life(rows, width) == evidence.payload["half_life_steps"]
    assert _gate_hamming(raster, width) == list(evidence.payload["bin_spike_hamming"])
    assert (
        _gate_class(
            rows,
            raster,
            width,
            GB_CONFIG.n_neurons,
            GB_SPIKE_CEILING,
            GB_CURRENT_CEILING,
            GB_ZERO_FLOOR,
        )
        == evidence.payload["trajectory_class"]
    )
    raster_lists = [sorted(step) for step in raster]
    assert raster_lists == [list(step) for step in evidence.payload["completion_spike_raster"]]
    return {
        "evidence_sha256": evidence.file_sha256,
        "evidence_self_sha256": evidence.payload_self_sha256,
        "recurrent_energy_bits": first_bits,
        "child_pids": [first_pid, second_pid],
        "trajectory_class": evidence.payload["trajectory_class"],
    }


def _u2_witness_search(extension_sha256: str) -> dict[str, Any]:
    """Attempt a real-surface U-2 component-vs-hypot regression witness.

    The U-2 addendum asks for a real episode whose component-wise sum-of-squares of
    the ordered per-neuron recurrent-net currents rounds differently from the square
    of the exported scaled-hypot l2_norm. The installed PyO3 surface exposes only the
    twelve reduced ``current_values`` summary columns (per triplet signed_sum/l1/l2)
    and the sparse spike CSR; it does not expose the ordered per-neuron recurrent-net
    current components from which the scaled-hypot norm was formed. No authorized real
    surface therefore admits the component-wise comparison, so no witness can be
    materialised without a forbidden fabricated current matrix. This is recorded as
    ``witness_found: false`` with the exact bounded episode space actually run and the
    precise observability blocker — never a summary-column substitute.
    """
    backend = load_stream_backend(BackendIdentity(2, "0.1.0", extension_sha256))
    tried: list[dict[str, int]] = []
    for seed in (11, 29, 47, 71):
        config = ModelConfig(n_neurons=16, excitatory_fraction=0.75, connectivity=0.5, dt_ms=1.0)
        inputs = gb.build_episode(config, seed, 8, 32)
        result = backend.run(inputs, 8, False, config)
        completion = int(np.flatnonzero(result.phases == 1).size)
        tried.append({"seed": seed, "n_neurons": config.n_neurons, "completion": completion})
    return {
        "witness_found": False,
        "episodes_tried": tried,
        "observability_blocker": (
            "installed PyO3 surface exposes only reduced recurrent-net summary columns "
            "(signed_sum/l1/l2), not the ordered per-neuron current components; a "
            "component-wise witness is unobservable without a forbidden fabricated matrix"
        ),
    }


# ---------------------------------------------------------------------------
# G-B negatives (real surface only)
# ---------------------------------------------------------------------------


def _check_gb_negatives(extension_sha256: str) -> None:
    backend = load_stream_backend(BackendIdentity(2, "0.1.0", extension_sha256))
    config = ModelConfig(n_neurons=16, excitatory_fraction=0.75, connectivity=0.25)
    inputs = gb.build_episode(config, 3, 8, 32)

    def evaluate(**changes: Any) -> Callable[[], object]:
        parameters: dict[str, Any] = {
            "completion_steps": 32,
            "spike_drift_ceiling": 0.05,
            "current_drift_ceiling": 0.05,
            "numerical_zero_floor": 1e-09,
        }
        parameters.update(changes)
        return lambda: evaluate_gb_descriptor(backend, inputs, 8, config, **parameters)

    _expect("multiple of eight", evaluate(completion_steps=30))
    _expect("multiple of eight", evaluate(completion_steps=24))
    _expect("exact int", evaluate(completion_steps=32.0))
    _expect("finite non-negative float", evaluate(spike_drift_ceiling=-1.0))
    _expect("finite non-negative float", evaluate(current_drift_ceiling=float("nan")))
    _expect("finite non-negative float", evaluate(numerical_zero_floor="x"))
    _expect(
        "completion phase length differs",
        lambda: evaluate_gb_descriptor(
            backend,
            inputs,
            8,
            config,
            completion_steps=40,
            spike_drift_ceiling=0.05,
            current_drift_ceiling=0.05,
            numerical_zero_floor=1e-09,
        ),
    )
    _expect(
        "verified StreamBackend is required",
        lambda: evaluate_gb_descriptor(
            _not_a_backend(),
            inputs,
            8,
            config,
            completion_steps=32,
            spike_drift_ceiling=0.05,
            current_drift_ceiling=0.05,
            numerical_zero_floor=1e-09,
        ),
    )
    # Real episode that violates the completion-phase external-L1==0 contract: drive a
    # nonzero packet inside the completion window through the real installed engine.
    active = gb.build_episode(config, 5, 8, 32)
    packets = active.packets.copy()
    packets[20, 0] = 18.0
    from dataclasses import replace

    _expect(
        "external L1 must be exactly zero at every completion row",
        lambda: evaluate_gb_descriptor(
            backend,
            replace(active, packets=packets),
            8,
            config,
            completion_steps=32,
            spike_drift_ceiling=0.05,
            current_drift_ceiling=0.05,
            numerical_zero_floor=1e-09,
        ),
    )


def _not_a_backend() -> Any:
    return object()


# ---------------------------------------------------------------------------
# Artifact/semantic/task/lane/path negatives (real surface only)
# ---------------------------------------------------------------------------


def _check_negatives(
    workspace: Path,
    d1_binding: dict[str, Any],
    d2_binding: dict[str, Any],
    record_ids: list[str],
    scoring: Any,
    calibration: Any,
    identities: dict[str, Any],
    lock: Any,
    d1_art: Any,
    d2_art: Any,
    bindings: dict[str, str],
    locked_identities: list[str],
    foreign_d1: Any,
    expected_task_set_digest: str,
    completeness: Any,
    cues: tuple[str, str],
    real: dict[str, str],
    bundle_reps: dict[str, str],
    bundle_arts: list[Any],
    foreign_lanes: dict[str, Any],
    lane_c_manifest: dict[str, Any],
    lane_h_manifest: dict[str, Any],
) -> None:
    base = _lock_payload(d1_binding, d2_binding, identities, bindings, real, foreign_lanes)
    good = el.seal_payload(base)
    assert validate_artifact_bytes(good).artifact_type == "snn-memory-experiment-lock-v2"
    _expect(
        "SHA-256 mismatch", lambda: validate_artifact_bytes(good, expected_file_sha256="0" * 64)
    )
    _expect(
        "duplicate JSON key",
        lambda: validate_artifact_bytes(b'{"artifact_type":"x","artifact_type":"y"}'),
    )
    _expect(
        "non-finite JSON constant", lambda: validate_artifact_bytes(b'{"schema_version":Infinity}')
    )
    _expect(
        "contains a non-finite JSON number",
        lambda: validate_artifact_bytes(b'{"schema_version":1e400}'),
    )
    _expect("is not strict UTF-8 JSON", lambda: validate_artifact_bytes(b"\xff\xfe\xfd"))
    _expect("root must be an object", lambda: validate_artifact_bytes(b"[1,2,3]\n"))
    _expect(
        "unknown or missing artifact_type", lambda: validate_artifact_bytes(b'{"schema_version":2}')
    )
    _expect(
        "cannot seal an unknown artifact_type",
        lambda: el.seal_payload({"artifact_type": "snn-memory-bogus-v2"}),
    )
    _expect(
        "differs from the expected type",
        lambda: validate_artifact_bytes(good, expected_type="snn-memory-scoring-target-v2"),
    )
    _expect("not canonical", lambda: validate_artifact_bytes(good[:-1] + b" \n"))
    _expect(
        "schema validation failed",
        lambda: validate_artifact_bytes(_craft(base, lambda p: p.update(extra=1))),
    )
    unsigned = json.loads(good.decode())
    unsigned["self_sha256"] = "0" * 64
    _expect("self digest mismatch", lambda: validate_artifact_bytes(el._canonical(unsigned)))

    _check_scoring_negatives(record_ids, bindings)
    _check_calibration_negatives(bindings)
    _check_foreign_lane_negatives(base, lane_c_manifest, lane_h_manifest)
    _check_lock_binding_negatives(
        d1_binding,
        d2_binding,
        scoring,
        calibration,
        identities,
        bindings,
        real,
        d1_art,
        d2_art,
        lock,
        foreign_d1,
        foreign_lanes,
        lane_c_manifest,
        bundle_arts,
    )
    _check_completeness_negatives(cues)
    _check_completeness_binding_negatives(lock, completeness, cues)
    _check_gb_evidence_negatives(workspace, identities)
    _check_dev_namespace(record_ids, locked_identities, bundle_reps)
    _check_path_negatives(workspace, record_ids, bindings)


def _check_scoring_negatives(record_ids: list[str], bindings: dict[str, str]) -> None:
    base = _scoring_target(record_ids, bindings)
    assert (
        validate_artifact_bytes(el.seal_payload(base)).artifact_type
        == "snn-memory-scoring-target-v2"
    )

    def unsorted(payload: dict[str, Any]) -> None:
        payload["candidate_order"] = list(reversed(payload["candidate_order"]))

    _expect(
        "must be lexical immutable-record-ID order",
        lambda: validate_artifact_bytes(_craft(base, unsorted)),
    )

    def short_bank(payload: dict[str, Any]) -> None:
        payload["candidate_signature_digests"] = payload["candidate_signature_digests"][:-1]

    _expect(
        "one candidate signature digest is required per candidate",
        lambda: validate_artifact_bytes(_craft(base, short_bank)),
    )

    def break_bank_digest(payload: dict[str, Any]) -> None:
        payload["identities"]["candidate_bank_digest"] = "0" * 64

    _expect(
        "candidate-bank digest does not bind the candidate bank",
        lambda: validate_artifact_bytes(_craft(base, break_bank_digest)),
    )

    def break_scorer(payload: dict[str, Any]) -> None:
        payload["scorer_digest"] = "0" * 64

    _expect(
        "scorer digest does not bind the scorer configuration",
        lambda: validate_artifact_bytes(_craft(base, break_scorer)),
    )


def _check_calibration_negatives(bindings: dict[str, str]) -> None:
    base = _calibration_spec(bindings)
    assert (
        validate_artifact_bytes(el.seal_payload(base)).artifact_type
        == "snn-memory-gb-calibration-spec-v2"
    )

    def negative_grid(payload: dict[str, Any]) -> None:
        payload["spike_drift_ceiling_grid"] = [-0.1]

    _expect(
        "must contain only finite non-negative values",
        lambda: validate_artifact_bytes(_craft(base, negative_grid)),
    )

    # Note: the non-numeric / boolean grid-item branch in _require_finite_non_negative_grid
    # is unreachable through a schema-valid artifact — the schema "grid" def already forces
    # every grid item to JSON "number", so a string/bool item fails schema first. It is a
    # redundant defensive guard, kept (not deleted) and reported as schema-unreachable.

    def tamper_positive(payload: dict[str, Any]) -> None:
        payload["synthetic_generator"]["positive_ids"][0] = "dev:tampered-000"

    _expect(
        "synthetic positive IDs are not the canonical per-seed corpus",
        lambda: validate_artifact_bytes(_craft(base, tamper_positive)),
    )

    def tamper_negative(payload: dict[str, Any]) -> None:
        payload["synthetic_generator"]["negative_ids"][0] = "dev:tampered-negative"

    _expect(
        "synthetic negative IDs are not the matched four-family corpus",
        lambda: validate_artifact_bytes(_craft(base, tamper_negative)),
    )

    def break_validation(payload: dict[str, Any]) -> None:
        payload["synthetic_generator"]["validation_partition_digest"] = "0" * 64

    _expect(
        "synthetic validation partition digest is not bound",
        lambda: validate_artifact_bytes(_craft(base, break_validation)),
    )

    def break_config(payload: dict[str, Any]) -> None:
        payload["synthetic_generator"]["config_digest"] = "0" * 64

    _expect(
        "synthetic config digest does not bind the generator configuration",
        lambda: validate_artifact_bytes(_craft(base, break_config)),
    )

    def break_selection(payload: dict[str, Any]) -> None:
        payload["synthetic_generator"]["selection_partition_digest"] = "0" * 64

    _expect(
        "synthetic selection partition digest is not bound",
        lambda: validate_artifact_bytes(_craft(base, break_selection)),
    )

    def break_task_set(payload: dict[str, Any]) -> None:
        payload["task_set_digest"] = "0" * 64

    _expect(
        "calibration task-set digest does not bind the synthetic inventory",
        lambda: validate_artifact_bytes(_craft(base, break_task_set)),
    )


def _check_lock_binding_negatives(
    d1_binding: dict[str, Any],
    d2_binding: dict[str, Any],
    scoring: Any,
    calibration: Any,
    identities: dict[str, Any],
    bindings: dict[str, str],
    real: dict[str, str],
    d1_art: Any,
    d2_art: Any,
    lock: Any,
    foreign_d1: Any,
    foreign_lanes: dict[str, Any],
    lane_c_manifest: dict[str, Any],
    bundle_arts: list[Any],
) -> None:
    def build(lane: str) -> dict[str, Any]:
        return _lock_payload(
            d1_binding, d2_binding, identities, bindings, real, foreign_lanes, lane=lane
        )

    def bind(lock_art: Any, scoring_art: Any, calibration_art: Any) -> None:
        bind_lane_isolation(lock_art, scoring_art, calibration_art, d1_art, d2_art, bundle_arts)

    base = build("lane_p")

    def break_d2_source(payload: dict[str, Any]) -> None:
        payload["d2"]["source_universe"]["repository_head"] = "c" * 40

    _expect(
        "D2 source-universe binding differs from the D1 binding",
        lambda: validate_artifact_bytes(_craft(base, break_d2_source)),
    )

    def break_head(payload: dict[str, Any]) -> None:
        payload["identities"]["repository_head"] = "d" * 40

    _expect(
        "lock repository HEAD differs from the D1 binding",
        lambda: validate_artifact_bytes(_craft(base, break_head)),
    )

    def break_candidate(payload: dict[str, Any]) -> None:
        payload["candidate_order"] = list(reversed(payload["candidate_order"]))

    _expect(
        "candidate order differs from the D1 selected record IDs",
        lambda: validate_artifact_bytes(_craft(base, break_candidate)),
    )

    def break_order_digest(payload: dict[str, Any]) -> None:
        payload["candidate_order_digest"] = "0" * 64

    _expect(
        "lock candidate-order digest is not bound",
        lambda: validate_artifact_bytes(_craft(base, break_order_digest)),
    )

    def break_lane_role_digest(payload: dict[str, Any]) -> None:
        payload["lane_role_digest"] = "0" * 64

    _expect(
        "lock lane-role digest is not bound",
        lambda: validate_artifact_bytes(_craft(base, break_lane_role_digest)),
    )

    def break_lane_domain_digest(payload: dict[str, Any]) -> None:
        payload["lane_domain_digest"] = "0" * 64

    _expect(
        "lock lane-domain digest is not bound",
        lambda: validate_artifact_bytes(_craft(base, break_lane_domain_digest)),
    )

    # Each lane-identity block field must stay consistent with the lock's own top-level
    # field (provenance, schema, task set, scorer, root); an inconsistent one is rejected.
    consistency = {
        "provenance": "lane provenance identity is not bound to the candidate order",
        "schema": "lane schema identity is not bound to the lock schema",
        "task_set": "lane task-set identity is not bound to the expected task set",
        "scorer": "lane scorer identity is not bound to the scoring target",
        "root": "lane root identity is not bound to the output root",
    }
    for consistency_field, consistency_message in consistency.items():

        def break_consistency(payload: dict[str, Any], key: str = consistency_field) -> None:
            payload["lane_identity"][key] = "0" * 64
            payload["lane_domain_digest"] = el.lane_domain_digest(
                "lane_p", el.lane_identity_fields(payload)
            )

        _expect(
            consistency_message,
            lambda mutate=break_consistency: validate_artifact_bytes(_craft(base, mutate)),
        )

    # Contaminate representative Lane-P identity classes with the bound foreign Lane-C identity
    # derived from the lock's embedded diagnostic manifest: the five distinguishing lane-identity
    # fields (with their top-level counterparts and domain resealed), a toolchain identity, and a
    # D2 provenance identity. Each carries a foreign-lane digest and is rejected by the disjointness.
    foreign_identity = el.lane_c_manifest_identity(lane_c_manifest)
    top_level = {
        "root": "output_root_digest",
        "task_set": "expected_task_set_digest",
        "scorer": "scoring_target_digest",
    }
    for field in _LANE_DISTINGUISHING:
        foreign_value = foreign_identity[field]

        def contaminate(
            payload: dict[str, Any], key: str = field, value: str = foreign_value
        ) -> None:
            payload["lane_identity"][key] = value
            if key in top_level:
                payload[top_level[key]] = value
            payload["lane_domain_digest"] = el.lane_domain_digest(
                "lane_p", el.lane_identity_fields(payload)
            )

        _expect(
            "a foreign-lane identity contaminates a Lane-P identity",
            lambda mutate=contaminate: validate_artifact_bytes(_craft(base, mutate)),
        )

    def contaminate_toolchain(payload: dict[str, Any]) -> None:
        payload["identities"]["python_wheel_sha256"] = foreign_identity["scorer"]

    _expect(
        "a foreign-lane identity contaminates a Lane-P identity",
        lambda: validate_artifact_bytes(_craft(base, contaminate_toolchain)),
    )

    def contaminate_d2(payload: dict[str, Any]) -> None:
        payload["d2"]["bundle_inventory_digest"] = foreign_identity["provenance"]

    _expect(
        "a foreign-lane identity contaminates a Lane-P identity",
        lambda: validate_artifact_bytes(_craft(base, contaminate_d2)),
    )

    # A lane_c/lane_h-ROLE lock over the Lane-P substrate (not a genuine foreign artifact) is
    # role-rejected. Its causal roots stay the Lane-P D1/D2/schema; only the role differs.
    lane_c_role = validate_artifact_bytes(el.seal_payload(build("lane_c")))
    _expect("only a Lane-P lock is admissible", lambda: bind(lane_c_role, scoring, calibration))
    lane_h_role = validate_artifact_bytes(el.seal_payload(build("lane_h")))
    _expect("only a Lane-P lock is admissible", lambda: bind(lane_h_role, scoring, calibration))
    _expect(
        "requires the exact lock/scoring/calibration types",
        lambda: bind(scoring, scoring, calibration),
    )
    bind_lane_isolation(lock, scoring, calibration, d1_art, d2_art, bundle_arts)
    bind_d1_d2(lock, d1_art, d2_art)

    # Artifact-level disjointness in a NON-FIRST authenticated bundle: a foreign Lane-C manifest
    # whose measured component equals a later bundle's file digest is caught only where the binder
    # derives every bundle identity (not just the first). The lock still seals — bundle identities
    # are not in the lock's own inventory — so this exercises the bind-boundary check specifically.
    assert len(bundle_arts) >= 2, "the cue set must declare at least two bundles"
    non_first_digest = bundle_arts[-1].file_sha256
    contaminated_manifest = dict(lane_c_manifest)
    contaminated_manifest["candidate_bank_sha256"] = non_first_digest
    contaminated_foreign = {
        "lane_c": el.build_foreign_lane_c(contaminated_manifest),
        "lane_h": foreign_lanes["lane_h"],
    }
    contaminated_lock = validate_artifact_bytes(
        el.seal_payload(
            _lock_payload(d1_binding, d2_binding, identities, bindings, real, contaminated_foreign)
        )
    )
    _expect(
        "a foreign-lane identity contaminates a bound Lane-P artifact identity",
        lambda: bind(contaminated_lock, scoring, calibration),
    )

    record_order = list(d1_binding["selected_record_ids"])

    def bind_scoring(payload: dict[str, Any], scoring_art: Any) -> None:
        # Rebind the lock's scorer and candidate-bank lane identities to the given scoring
        # so downstream isolation checks are reached rather than shadowed by the self bind.
        payload["scoring_target_digest"] = scoring_art.payload_self_sha256
        payload["lane_identity"]["scorer"] = scoring_art.payload_self_sha256
        payload["lane_identity"]["candidate_bank"] = scoring_art.payload["identities"][
            "candidate_bank_digest"
        ]
        payload["lane_domain_digest"] = el.lane_domain_digest(
            "lane_p", el.lane_identity_fields(payload)
        )

    # A scoring target whose lane role leaves Lane P is rejected before digest binding.
    scoring_lane_c = validate_artifact_bytes(
        _craft(_scoring_target(record_order, bindings), lambda p: p.update(lane_role="lane_c"))
    )
    _expect(
        "scoring-target lane role contaminates the Lane-P lock",
        lambda: bind(lock, scoring_lane_c, calibration),
    )

    # A bound scoring target whose toolchain identity block differs from the lock is rejected.
    def mutate_identity(payload: dict[str, Any]) -> None:
        payload["identities"]["schema_sha256"] = "0" * 64
        payload["identities"]["candidate_bank_digest"] = el.candidate_bank_digest(
            payload["candidate_order"], payload["candidate_signature_digests"]
        )

    scoring_bad_identity = validate_artifact_bytes(
        _craft(_scoring_target(record_order, bindings), mutate_identity)
    )
    lock_binds_bad = validate_artifact_bytes(
        _craft(base, lambda p: bind_scoring(p, scoring_bad_identity))
    )
    _expect(
        "scoring-target identity differs from the lock identity block",
        lambda: bind(lock_binds_bad, scoring_bad_identity, calibration),
    )

    # A lock whose calibration digest does not bind the calibration spec is rejected.
    def break_calibration_digest(payload: dict[str, Any]) -> None:
        payload["calibration_spec_digest"] = _framed("unbound:calibration")

    lock_bad_calibration = validate_artifact_bytes(_craft(base, break_calibration_digest))
    _expect(
        "lock calibration digest does not bind the calibration spec",
        lambda: bind(lock_bad_calibration, scoring, calibration),
    )

    # A lock whose threshold identity does not bind the calibration thresholds is rejected.
    def break_threshold(payload: dict[str, Any]) -> None:
        payload["lane_identity"]["threshold"] = _framed("unbound:threshold")
        payload["lane_domain_digest"] = el.lane_domain_digest(
            "lane_p", el.lane_identity_fields(payload)
        )

    lock_bad_threshold = validate_artifact_bytes(_craft(base, break_threshold))
    _expect(
        "threshold identity does not bind the calibration thresholds",
        lambda: bind(lock_bad_threshold, scoring, calibration),
    )

    # A bound scoring target over a different candidate set breaks the lexical candidate check.
    scoring_short = validate_artifact_bytes(
        el.seal_payload(_scoring_target(record_order[:-1], bindings))
    )
    lock_binds_short = validate_artifact_bytes(
        _craft(base, lambda p: bind_scoring(p, scoring_short))
    )
    _expect(
        "scoring candidate order must be the lexical Lane-P candidate set",
        lambda: bind(lock_binds_short, scoring_short, calibration),
    )

    # A calibration spec whose lane role leaves Lane P is rejected.
    calibration_lane_c = validate_artifact_bytes(
        _craft(_calibration_spec(bindings), lambda p: p.update(lane_role="lane_c"))
    )
    _expect(
        "calibration-spec lane role contaminates the Lane-P lock",
        lambda: bind(lock, scoring, calibration_lane_c),
    )

    # A bound calibration spec whose toolchain identity differs from the lock is rejected.
    def mutate_calibration_identity(payload: dict[str, Any]) -> None:
        payload["identities"]["schema_sha256"] = "0" * 64

    calibration_bad = validate_artifact_bytes(
        _craft(_calibration_spec(bindings), mutate_calibration_identity)
    )
    lock_binds_bad_cal = validate_artifact_bytes(
        _craft(
            base, lambda p: p.update(calibration_spec_digest=calibration_bad.payload_self_sha256)
        )
    )
    _expect(
        "calibration identity differs from the lock identity block",
        lambda: bind(lock_binds_bad_cal, scoring, calibration_bad),
    )

    # A lock whose candidate-bank identity does not bind the scoring candidate bank is rejected.
    def break_candidate_bank(payload: dict[str, Any]) -> None:
        payload["lane_identity"]["candidate_bank"] = _framed("unbound:candidate_bank")
        payload["lane_domain_digest"] = el.lane_domain_digest(
            "lane_p", el.lane_identity_fields(payload)
        )

    lock_bad_bank = validate_artifact_bytes(_craft(base, break_candidate_bank))
    _expect(
        "candidate-bank identity does not bind the scoring candidate bank",
        lambda: bind(lock_bad_bank, scoring, calibration),
    )

    # Authenticated D2 mismatch: the lock's D2 cue-set digest was altered.
    def break_cue_set(payload: dict[str, Any]) -> None:
        payload["d2"]["cue_set_file_sha256"] = "0" * 64

    forged = validate_artifact_bytes(_craft(base, break_cue_set))
    _expect(
        "lock D2 binding differs from the authenticated cue set",
        lambda: bind_d1_d2(forged, d1_art, d2_art),
    )

    # Authenticated D2 bundle-inventory mismatch.
    def break_inventory(payload: dict[str, Any]) -> None:
        payload["d2"]["bundle_inventory_digest"] = "0" * 64

    forged_inventory = validate_artifact_bytes(_craft(base, break_inventory))
    _expect(
        "lock D2 binding differs from the authenticated cue set",
        lambda: bind_d1_d2(forged_inventory, d1_art, d2_art),
    )

    # Authenticated D1 mismatch: both the D1 and nested-D2 file digests were altered.
    def break_d1_file(payload: dict[str, Any]) -> None:
        payload["d1"]["file_sha256"] = "0" * 64
        payload["d2"]["source_universe"]["file_sha256"] = "0" * 64

    forged_d1 = validate_artifact_bytes(_craft(base, break_d1_file))
    _expect(
        "lock D1 binding differs from the authenticated D1 artifact",
        lambda: bind_d1_d2(forged_d1, d1_art, d2_art),
    )

    # Foreign-root: the primary lock rejects a disjoint source-universe root.
    _expect(
        "lock D1 binding differs from the authenticated D1 artifact",
        lambda: bind_d1_d2(lock, foreign_d1, d2_art),
    )

    # Post-lock dependency mutation: a scoring target resealed after the lock bound its digest.
    mutated = _scoring_target(list(d1_binding["selected_record_ids"]), bindings)
    mutated["max_payload_utf8_bytes"] = 30000
    mutated_art = validate_artifact_bytes(el.seal_payload(mutated))
    assert mutated_art.payload_self_sha256 != lock.payload["scoring_target_digest"]
    _expect("scoring-target digest does not bind", lambda: bind(lock, mutated_art, calibration))


def _check_foreign_lane_negatives(
    base: dict[str, Any], lane_c_manifest: dict[str, Any], lane_h_manifest: dict[str, Any]
) -> None:
    good = el.seal_payload(base)

    # R6-01: a Lane-C manifest component digest changed without re-deriving its identity is rejected.
    def substitute_c_manifest(payload: dict[str, Any]) -> None:
        payload["foreign_lanes"]["lane_c"]["manifest"]["output_root_sha256"] = "0" * 64

    _expect(
        "foreign Lane-C identity is not derived from its manifest",
        lambda: validate_artifact_bytes(_craft(base, substitute_c_manifest)),
    )

    def break_c_self(payload: dict[str, Any]) -> None:
        payload["foreign_lanes"]["lane_c"]["self_sha256"] = "0" * 64

    _expect(
        "foreign Lane-C evidence self digest mismatch",
        lambda: validate_artifact_bytes(_craft(base, break_c_self)),
    )

    # R6-01: a raw measured Lane-C component digest injected into a Lane-P toolchain identity is
    # rejected — the foreign set bans the raw component, not only its domain-folded identity.
    def inject_raw_c_component(payload: dict[str, Any]) -> None:
        payload["identities"]["rust_wheel_sha256"] = lane_c_manifest["diagnostic_schema_sha256"]

    _expect(
        "a foreign-lane identity contaminates a Lane-P identity",
        lambda: validate_artifact_bytes(_craft(base, inject_raw_c_component)),
    )

    # Omission of a foreign lane and a Lane-C role change are schema-rejected before sealing.
    def omit_h(payload: dict[str, Any]) -> None:
        del payload["foreign_lanes"]["lane_h"]

    _expect("schema validation failed", lambda: validate_artifact_bytes(_craft(base, omit_h)))

    def role_change_c(payload: dict[str, Any]) -> None:
        payload["foreign_lanes"]["lane_c"]["lane_role"] = "lane_h"

    _expect(
        "schema validation failed", lambda: validate_artifact_bytes(_craft(base, role_change_c))
    )

    # Post-lock mutation of the embedded foreign evidence breaks the lock's own self digest.
    tampered = json.loads(good.decode())
    tampered["foreign_lanes"]["lane_h"]["reason"] = "tampered inadmissibility reason"
    _expect("self digest mismatch", lambda: validate_artifact_bytes(el._canonical(tampered)))

    def break_h_self(payload: dict[str, Any]) -> None:
        payload["foreign_lanes"]["lane_h"]["self_sha256"] = "0" * 64

    _expect(
        "foreign Lane-H evidence self digest mismatch",
        lambda: validate_artifact_bytes(_craft(base, break_h_self)),
    )

    # R5-02: a synthetic (non-manifest-derived) admissible Lane-H identity is rejected.
    def synthetic_h_identity(payload: dict[str, Any]) -> None:
        zero_identity = {
            key: "0" * 64
            for key in (
                "root",
                "task_set",
                "threshold",
                "candidate_bank",
                "scorer",
                "schema",
                "provenance",
            )
        }
        block = {
            "lane_role": "lane_h",
            "status": "admissible_authenticated_historical_evidence",
            "manifest": dict(lane_h_manifest),
            "identity": zero_identity,
            "self_sha256": "0" * 64,
        }
        payload["foreign_lanes"]["lane_h"] = block

    _expect(
        "admissible Lane-H identity is not derived from its manifest",
        lambda: validate_artifact_bytes(_craft(base, synthetic_h_identity)),
    )

    # An admissibility/status substitution keeping the inadmissible fields is schema-rejected.
    def status_substitution(payload: dict[str, Any]) -> None:
        payload["foreign_lanes"]["lane_h"]["status"] = (
            "admissible_authenticated_historical_evidence"
        )

    _expect(
        "schema validation failed",
        lambda: validate_artifact_bytes(_craft(base, status_substitution)),
    )

    # A partial admissible Lane-H package (manifest missing a required historical field) is rejected.
    def partial_h(payload: dict[str, Any]) -> None:
        manifest = dict(lane_h_manifest)
        del manifest["historical_git_object"]
        block = {
            "lane_role": "lane_h",
            "status": "admissible_authenticated_historical_evidence",
            "manifest": manifest,
            "identity": el.lane_h_manifest_identity(lane_h_manifest),
            "self_sha256": "0" * 64,
        }
        payload["foreign_lanes"]["lane_h"] = block

    _expect("schema validation failed", lambda: validate_artifact_bytes(_craft(base, partial_h)))

    # R6-01: with an admissible Lane H, a raw measured historical component digest injected into a
    # Lane-P identity is rejected — admissible H contributes its raw components to the foreign set.
    def inject_raw_h_component(payload: dict[str, Any]) -> None:
        payload["foreign_lanes"]["lane_h"] = el.build_foreign_lane_h_admissible(lane_h_manifest)
        payload["identities"]["backend_extension_sha256"] = lane_h_manifest[
            "historical_embedding_sha256"
        ]

    _expect(
        "a foreign-lane identity contaminates a Lane-P identity",
        lambda: validate_artifact_bytes(_craft(base, inject_raw_h_component)),
    )

    # Positive: an admissible authenticated historical Lane-H record derived from a manifest is accepted.
    def admissible_h(payload: dict[str, Any]) -> None:
        payload["foreign_lanes"]["lane_h"] = el.build_foreign_lane_h_admissible(lane_h_manifest)

    admissible_lock = validate_artifact_bytes(_craft(base, admissible_h))
    assert (
        admissible_lock.payload["foreign_lanes"]["lane_h"]["status"]
        == "admissible_authenticated_historical_evidence"
    )


def _check_completeness_negatives(cues: tuple[str, str]) -> None:
    good = _completeness_payload(True, cues)
    assert validate_artifact_bytes(el.seal_payload(good)).payload["completeness"] is True

    def duplicate_task_id(payload: dict[str, Any]) -> None:
        payload["expected_tasks"][1]["task_id"] = payload["expected_tasks"][0]["task_id"]

    _expect(
        "expected task IDs are not unique",
        lambda: validate_artifact_bytes(_craft(good, duplicate_task_id)),
    )

    def wrong_task_set(payload: dict[str, Any]) -> None:
        payload["task_set_digest"] = "0" * 64

    _expect(
        "task-set digest does not bind the expected tasks",
        lambda: validate_artifact_bytes(_craft(good, wrong_task_set)),
    )

    def missing_seed(payload: dict[str, Any]) -> None:
        payload["expected_tasks"] = [
            t for t in payload["expected_tasks"] if t["seed"] != el.SEEDS[-1]
        ]
        ids = [task["task_id"] for task in payload["expected_tasks"]]
        payload["task_set_digest"] = task_set_digest(ids)
        payload["completed_task_ids"] = ids
        payload["process_records"] = _process_records(ids)

    _expect(
        "completeness flag disagrees", lambda: validate_artifact_bytes(_craft(good, missing_seed))
    )

    def wrong_identity(payload: dict[str, Any]) -> None:
        payload["expected_tasks"][0]["condition"] = "no_match"
        ids = [task["task_id"] for task in payload["expected_tasks"]]
        payload["task_set_digest"] = task_set_digest(ids)
        payload["completed_task_ids"] = ids
        payload["process_records"] = _process_records(ids)

    _expect(
        "task ID does not bind its full task identity",
        lambda: validate_artifact_bytes(_craft(good, wrong_identity)),
    )

    def multiset_imbalance(payload: dict[str, Any]) -> None:
        payload["expected_tasks"][0]["condition"] = "zero_recurrence"
        payload["expected_tasks"][0]["task_id"] = task_identity_digest(
            payload["expected_tasks"][0]["seed"],
            "lane_p",
            "zero_recurrence",
            payload["expected_tasks"][0]["cue_digest"],
        )
        ids = [task["task_id"] for task in payload["expected_tasks"]]
        payload["task_set_digest"] = task_set_digest(ids)
        payload["completed_task_ids"] = ids
        payload["process_records"] = _process_records(ids)

    _expect(
        "completeness flag disagrees",
        lambda: validate_artifact_bytes(_craft(good, multiset_imbalance)),
    )

    def process_cardinality(payload: dict[str, Any]) -> None:
        payload["process_records"] = payload["process_records"][:-1]

    _expect(
        "completeness flag disagrees",
        lambda: validate_artifact_bytes(_craft(good, process_cardinality)),
    )

    def process_reuse(payload: dict[str, Any]) -> None:
        payload["process_records"][1]["process_instance"] = payload["process_records"][0][
            "process_instance"
        ]

    _expect(
        "completeness flag disagrees", lambda: validate_artifact_bytes(_craft(good, process_reuse))
    )

    def task_process_mismatch(payload: dict[str, Any]) -> None:
        payload["process_records"][0]["task_id"] = "f" * 64

    _expect(
        "completeness flag disagrees",
        lambda: validate_artifact_bytes(_craft(good, task_process_mismatch)),
    )

    def wrong_lane(payload: dict[str, Any]) -> None:
        payload["expected_tasks"][0]["lane_role"] = "lane_c"

    _expect(
        "expected task lane role differs", lambda: validate_artifact_bytes(_craft(good, wrong_lane))
    )

    def out_of_set_seed(payload: dict[str, Any]) -> None:
        payload["expected_tasks"][0]["seed"] = 999
        payload["expected_tasks"][0]["task_id"] = task_identity_digest(
            999,
            "lane_p",
            payload["expected_tasks"][0]["condition"],
            payload["expected_tasks"][0]["cue_digest"],
        )
        ids = [task["task_id"] for task in payload["expected_tasks"]]
        payload["task_set_digest"] = task_set_digest(ids)
        payload["completed_task_ids"] = ids
        payload["process_records"] = _process_records(ids)

    _expect(
        "outside the frozen seed set",
        lambda: validate_artifact_bytes(_craft(good, out_of_set_seed)),
    )

    def inconsistent_missing(payload: dict[str, Any]) -> None:
        payload["missing_task_ids"] = ["e" * 64]

    _expect(
        "missing task set is inconsistent",
        lambda: validate_artifact_bytes(_craft(good, inconsistent_missing)),
    )

    def unexpected_overlap(payload: dict[str, Any]) -> None:
        payload["unexpected_task_ids"] = [payload["expected_tasks"][0]["task_id"]]

    _expect(
        "unexpected tasks overlap",
        lambda: validate_artifact_bytes(_craft(good, unexpected_overlap)),
    )

    def foreign_digest_failure(payload: dict[str, Any]) -> None:
        payload["digest_failure_task_ids"] = ["d" * 64]

    _expect(
        "digest-failure tasks are not expected",
        lambda: validate_artifact_bytes(_craft(good, foreign_digest_failure)),
    )

    def clean_with_abort(payload: dict[str, Any]) -> None:
        payload["abort_reason"] = "digest_mismatch"

    _expect(
        "must not declare an abort reason",
        lambda: validate_artifact_bytes(_craft(good, clean_with_abort)),
    )

    def bad_abort(payload: dict[str, Any]) -> None:
        payload["completeness"] = False
        payload["completed_task_ids"] = payload["completed_task_ids"][:-1]
        payload["missing_task_ids"] = [payload["expected_tasks"][-1]["task_id"]]
        payload["process_records"] = payload["process_records"][:-1]
        payload["abort_reason"] = None

    _expect(
        "requires a known abort reason", lambda: validate_artifact_bytes(_craft(good, bad_abort))
    )

    def reuse_forces_abort(payload: dict[str, Any]) -> None:
        payload["process_records"][1]["process_instance"] = payload["process_records"][0][
            "process_instance"
        ]
        payload["completeness"] = False
        payload["abort_reason"] = "process_reuse"

    reuse_abort = validate_artifact_bytes(_craft(good, reuse_forces_abort))
    assert reuse_abort.payload["abort_reason"] == "process_reuse"

    def mismatch_forces_abort(payload: dict[str, Any]) -> None:
        payload["process_records"][0]["task_id"] = "f" * 64
        payload["completeness"] = False
        payload["abort_reason"] = "task_process_mismatch"

    mismatch_abort = validate_artifact_bytes(_craft(good, mismatch_forces_abort))
    assert mismatch_abort.payload["abort_reason"] == "task_process_mismatch"


def _check_completeness_binding_negatives(
    lock: Any, completeness: Any, cues: tuple[str, str]
) -> None:
    _expect(
        "requires the exact lock/completeness types", lambda: bind_task_completeness(lock, lock)
    )
    lane_c_completeness = validate_artifact_bytes(
        el.seal_payload(_completeness_payload(True, cues, lane="lane_c"))
    )
    _expect(
        "completeness lane role differs from the lock lane",
        lambda: bind_task_completeness(lock, lane_c_completeness),
    )
    other_completeness = validate_artifact_bytes(
        el.seal_payload(_completeness_payload(True, (cues[1], cues[0])))
    )
    _expect(
        "completeness task-set digest differs from the lock expected task set",
        lambda: bind_task_completeness(lock, other_completeness),
    )


def _gb_evidence_payload(
    extension_sha256: str, workspace: Path, identities: dict[str, Any]
) -> dict[str, Any]:
    backend = load_stream_backend(BackendIdentity(2, "0.1.0", extension_sha256))
    inputs = gb.build_episode(GB_CONFIG, GB_SEED, GB_CUE_STEPS, GB_COMPLETION_STEPS)
    descriptor = evaluate_gb_descriptor(
        backend,
        inputs,
        GB_CUE_STEPS,
        GB_CONFIG,
        completion_steps=GB_COMPLETION_STEPS,
        spike_drift_ceiling=GB_SPIKE_CEILING,
        current_drift_ceiling=GB_CURRENT_CEILING,
        numerical_zero_floor=GB_ZERO_FLOOR,
    )
    del identities
    return gb.descriptor_payload(
        descriptor,
        lane_role="lane_p",
        state="fixture_only",
        calibration_spec_digest=SHA,
        scoring_target_digest=SHA,
    )


def _check_gb_evidence_negatives(workspace: Path, identities: dict[str, Any]) -> None:
    ext = identities["backend_extension_sha256"]
    base = _gb_evidence_payload(ext, workspace, identities)
    assert (
        validate_artifact_bytes(el.seal_payload(base)).artifact_type
        == "snn-memory-gb-preflight-evidence-v2"
    )

    def wrong_energy_bits(payload: dict[str, Any]) -> None:
        payload["recurrent_energy_bits"] = "0000000000000000"

    _expect(
        "energy bits differ from the recomputed energy",
        lambda: validate_artifact_bytes(_craft(base, wrong_energy_bits)),
    )

    def wrong_settled(payload: dict[str, Any]) -> None:
        payload["settled"] = not payload["settled"]

    _expect(
        "settled flag disagrees with the trajectory class",
        lambda: validate_artifact_bytes(_craft(base, wrong_settled)),
    )

    def row_count_mismatch(payload: dict[str, Any]) -> None:
        payload["completion_steps"] = 40

    _expect(
        "completion rows differ from completion_steps",
        lambda: validate_artifact_bytes(_craft(base, row_count_mismatch)),
    )

    def non_contiguous(payload: dict[str, Any]) -> None:
        payload["completion_rows"][1]["timestep"] += 5

    _expect(
        "completion timesteps are not contiguous ascending",
        lambda: validate_artifact_bytes(_craft(base, non_contiguous)),
    )

    def changed_epsilon(payload: dict[str, Any]) -> None:
        payload["epsilon_bits"] = "3cb0000000000001"

    _expect(
        "schema validation failed", lambda: validate_artifact_bytes(_craft(base, changed_epsilon))
    )

    def mutate_row(payload: dict[str, Any]) -> None:
        payload["completion_rows"][10]["net_l2"] = payload["completion_rows"][10]["net_l2"] + 1.0

    _expect(
        "recurrent energy differs from the recomputed fold",
        lambda: validate_artifact_bytes(_craft(base, mutate_row)),
    )

    def raster_unsorted(payload: dict[str, Any]) -> None:
        payload["completion_spike_raster"][0] = [5, 3]

    _expect(
        "spike raster is not ascending",
        lambda: validate_artifact_bytes(_craft(base, raster_unsorted)),
    )

    def raster_out_of_range(payload: dict[str, Any]) -> None:
        payload["completion_spike_raster"][0] = [999]

    _expect(
        "spike index exceeds the population",
        lambda: validate_artifact_bytes(_craft(base, raster_out_of_range)),
    )

    def raster_extra(payload: dict[str, Any]) -> None:
        payload["completion_spike_raster"].append([])

    _expect(
        "spike raster differs from completion_steps",
        lambda: validate_artifact_bytes(_craft(base, raster_extra)),
    )

    # Individually resealed derived-field mutations reach the reader's independent
    # half-life, Hamming, and trajectory recomputation checks.
    def wrong_half_life(payload: dict[str, Any]) -> None:
        payload["half_life_steps"] = 0 if payload["half_life_steps"] != 0 else 8

    _expect(
        "half-life differs from the recomputed decay",
        lambda: validate_artifact_bytes(_craft(base, wrong_half_life)),
    )

    def wrong_hamming(payload: dict[str, Any]) -> None:
        payload["bin_spike_hamming"][0] = payload["bin_spike_hamming"][0] + 1

    _expect(
        "tail Hamming differs from the recomputed raster",
        lambda: validate_artifact_bytes(_craft(base, wrong_hamming)),
    )

    def wrong_trajectory(payload: dict[str, Any]) -> None:
        payload["trajectory_class"] = (
            "wandering_active"
            if payload["trajectory_class"] != "wandering_active"
            else "silent_decay"
        )

    _expect(
        "trajectory class differs from the recomputed dynamics",
        lambda: validate_artifact_bytes(_craft(base, wrong_trajectory)),
    )

    def negative_net_l2(payload: dict[str, Any]) -> None:
        payload["completion_rows"][0]["net_l2"] = -1.0

    _expect(
        "net-L2 must be finite and non-negative",
        lambda: validate_artifact_bytes(_craft(base, negative_net_l2)),
    )

    def overflow_net_l2(payload: dict[str, Any]) -> None:
        payload["completion_rows"][0]["net_l2"] = 1e200

    _expect(
        "recurrent energy is not finite non-negative",
        lambda: validate_artifact_bytes(_craft(base, overflow_net_l2)),
    )


def _check_dev_namespace(
    record_ids: list[str], locked_identities: list[str], bundle_reps: dict[str, str]
) -> None:
    require_dev_namespace_disjoint(["dev:mem-0", "dev:mem-1"], record_ids, locked_identities)
    _expect(
        "development IDs must use the dev: namespace",
        lambda: require_dev_namespace_disjoint(["mem-0"], record_ids, locked_identities),
    )
    _expect(
        "development and immutable record identities overlap",
        lambda: require_dev_namespace_disjoint(
            ["dev:" + record_ids[0].removeprefix("sha256:")], record_ids, locked_identities
        ),
    )
    # A synthetic development body aliasing each authenticated-bundle locked identity class.
    for representative in bundle_reps.values():
        _expect(
            "development identity overlaps a locked D2 identity",
            lambda value=representative: require_dev_namespace_disjoint(
                ["dev:" + value], record_ids, locked_identities
            ),
        )


def _check_path_negatives(workspace: Path, record_ids: list[str], bindings: dict[str, str]) -> None:
    _expect("is not a regular file", lambda: read_artifact(workspace / "d3-repo"))
    link = workspace / "path-link.json"
    link.symlink_to(workspace / "lock.json")
    _expect("cannot be opened safely", lambda: read_artifact(link))
    _expect("cannot be opened safely", lambda: read_artifact(workspace / "path-absent.json"))
    outside = workspace / "path-outside"
    outside.mkdir()
    (outside / "lock.json").write_bytes((workspace / "lock.json").read_bytes())
    parent_link = workspace / "path-parent-link"
    parent_link.symlink_to(outside)
    _expect("path traverses a symlink", lambda: read_artifact(parent_link / "lock.json"))
    # Explicit ``..`` / non-canonical spelling refused before any open.
    traversal = Path(str(workspace)) / ".." / workspace.name / "lock.json"
    _expect("path contains a traversal component", lambda: read_artifact(traversal))
    # write_artifact refuses a non-canonical (symlinked) output parent.
    _expect(
        "output parent must be an absolute canonical path",
        lambda: write_artifact(_scoring_target(record_ids, bindings), parent_link / "new.json"),
    )
    _expect(
        "no-clobber",
        lambda: write_artifact(_scoring_target(record_ids, bindings), workspace / "lock.json"),
    )
    # Substitution without a second read: the authenticated snapshot is retained and a
    # re-read under the original digest fails closed after the file is atomically replaced.
    original = read_artifact(workspace / "lock.json", expected_type="snn-memory-experiment-lock-v2")
    substitute_bytes = (workspace / "complete.json").read_bytes()
    assert original.file_sha256 != _sha(substitute_bytes)
    temporary = workspace / "lock.json.substitute"
    temporary.write_bytes(substitute_bytes)
    os.replace(temporary, workspace / "lock.json")
    assert original.file_sha256 == _sha(original.canonical_bytes)
    _expect(
        "SHA-256 mismatch",
        lambda: read_artifact(workspace / "lock.json", expected_file_sha256=original.file_sha256),
    )


# ---------------------------------------------------------------------------
# Measured environment bindings
# ---------------------------------------------------------------------------


def _dirty_tree_digest(repo: Path) -> str:
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    diff = _git(repo, "diff", "--no-color", "HEAD")
    return _sha(
        b"remanentia:snn-v2-dirty-tree:v1\0" + len(status).to_bytes(8, "big") + status + diff
    )


def _output_root_digest(workspace: Path) -> str:
    return _sha(b"remanentia:snn-v2-output-root:v1\0" + str(workspace.resolve()).encode())


def _cue_digests(d2_art: Any) -> tuple[str, str]:
    """Derive two real evaluation cue identities from the authenticated D2 cue set."""
    record = d2_art.payload["records"][0]
    trained = str(record["calibration_cue"]["sha256"])
    shuffled = str(record["evaluation_base_cues"][0]["variants"][0]["bundle"]["sha256"])
    return trained, shuffled


def _embedding_digest(bundle_art: Any) -> str:
    """Domain-separated digest of a bundle's decoded float64 embedding payload bytes."""
    raw = base64.b64decode(bundle_art.payload["embedding"]["data_base64"])
    return _sha(b"remanentia:snn-v2-embedding:v1\0" + raw)


def _bundle_class_reps(bundle_art: Any) -> dict[str, str]:
    """Return one representative hex identity per authenticated-bundle identity class."""
    payload = bundle_art.payload
    return {
        "bundle_file": bundle_art.file_sha256,
        "bundle_self": str(payload["self_sha256"]),
        "text": str(payload["text_sha256"]),
        "normalized_text": str(payload["normalized_text_sha256"]),
        "embedding": _embedding_digest(bundle_art),
        "encoder_directory": str(payload["encoder"]["directory_sha256"]),
        "encoder_config": str(payload["encoder"]["config_digest"]),
        "model_config": str(payload["model"]["config_digest"]),
        "implementation": str(payload["implementations"]["cue_materializer"]["sha256"]),
    }


def _locked_d2_inventory(d1_art: Any, d2_art: Any, bundle_arts: list[Any]) -> list[str]:
    """Build the complete authenticated D1 + D2 + bundle locked-identity inventory.

    Spans the authenticated D1 file/self/HEAD digests, its selector/split-event
    implementation paths and digests, and its selected record IDs, source paths, and
    per-record path/sha256/blob-oid; the authenticated D2 cue-set file/self digests,
    derivation, tokenizer, noise-lexicon, encoder/model, and implementation identities;
    every cue/variant identity, path, and digest; and — for every authenticated bundle —
    its file/self digests, text/normalized-text digests, an explicit domain-separated
    digest of the decoded float64 embedding payload, its encoder/model identities, and its
    three implementation/provenance paths and digests.
    """
    d1 = d1_art.payload
    inventory: set[str] = set(d1["selected_record_ids"])
    inventory.update(
        [d1_art.file_sha256, str(d1_art.payload_self_sha256), str(d1["repository"]["head"])]
    )
    for role in ("selector", "split_events"):
        impl = d1["implementations"][role]
        inventory.update([str(impl["logical_path"]), str(impl["sha256"])])
    inventory.update(str(path) for path in d1["selected_paths"])
    for record in d1["selected"]:
        inventory.update(
            [
                str(record["path"]),
                str(record["normalized_path"]),
                str(record["blob_oid"]),
                str(record["record_id"]),
                str(record["event_sha256"]),
            ]
        )

    d2 = d2_art.payload
    inventory.update([d2_art.file_sha256, str(d2_art.payload_self_sha256)])
    inventory.add(str(d2["tokenizer"]["sha256"]))
    inventory.add(str(d2["noise_lexicon"]["sha256"]))
    inventory.update([str(d2["encoder"]["directory_sha256"]), str(d2["encoder"]["config_digest"])])
    inventory.add(str(d2["model"]["config_digest"]))
    inventory.add(
        _sha(
            b"remanentia:snn-v2-derivation:v1\0"
            + json.dumps(el._thaw(d2["derivation"]), sort_keys=True).encode()
        )
    )
    for role in ("cue_materializer", "split_events", "sentence_encoder"):
        impl = d2["implementations"][role]
        inventory.update([str(impl["logical_path"]), str(impl["sha256"])])
    for record in d2["records"]:
        inventory.add(str(record["record_id"]))
        calibration = record["calibration_cue"]
        inventory.update(
            str(calibration[key]) for key in ("cue_id", "path", "sha256", "normalized_text_sha256")
        )
        for cue_base in record["evaluation_base_cues"]:
            inventory.add(str(cue_base["cue_id"]))
            for variant in cue_base["variants"]:
                inventory.update(
                    str(variant[key])
                    for key in (
                        "variant_id",
                        "path",
                        "sha256",
                        "normalized_text_sha256",
                        "tokenizer_digest",
                    )
                )
                inventory.update(str(variant["bundle"][key]) for key in ("path", "sha256"))

    for bundle_art in bundle_arts:
        payload = bundle_art.payload
        inventory.add(bundle_art.file_sha256)
        inventory.update(
            str(payload[key])
            for key in ("cue_id", "text_sha256", "normalized_text_sha256", "self_sha256")
        )
        inventory.add(_embedding_digest(bundle_art))
        inventory.update(
            str(payload["encoder"][key])
            for key in ("identity", "directory_sha256", "config_digest")
        )
        inventory.add(str(payload["model"]["config_digest"]))
        for role in ("cue_materializer", "split_events", "sentence_encoder"):
            implementation = payload["implementations"][role]
            inventory.update([str(implementation["logical_path"]), str(implementation["sha256"])])
    return sorted(inventory)


# Bounded fixture_only strong-recurrence configurations. Config 0 drives the completion
# tail to sustained/saturated activity (v_threshold sits just above v_rest with large
# recurrent weights and full connectivity), guaranteeing a non-silent trajectory so both
# the installed producer and the experiment_lock reader mirror exercise the non-silent
# trajectory, tail-silence-false, lag-drift, and current-drift paths. The remaining
# configurations vary excitation/inhibition to reach further trajectory classes.
_ACTIVE_CONFIGS = (
    # refractory 0: with no refractory period a saturated network fires the same neurons
    # every completion step, so consecutive bins match (settled_fixed at lag 1).
    ModelConfig(
        n_neurons=64,
        excitatory_fraction=0.95,
        connectivity=1.0,
        dt_ms=1.0,
        v_threshold_mv=-64.5,
        weight_max=50.0,
        refractory_ms=0.0,
        tau_m_ms=40.0,
    ),
    # refractory 1: a one-step refractory shifts which neurons fire each step, so the tail
    # never settles (wandering_active) and lag comparisons fail.
    ModelConfig(
        n_neurons=64,
        excitatory_fraction=0.95,
        connectivity=1.0,
        dt_ms=1.0,
        v_threshold_mv=-64.5,
        weight_max=50.0,
        refractory_ms=1.0,
        tau_m_ms=40.0,
    ),
    # refractory 7: a saturated network was expected to fire synchronously every eighth
    # step (settled_periodic at lag 2); in practice it settles or wanders, so the
    # settled_periodic branch is not produced by any tested bounded real configuration.
    ModelConfig(
        n_neurons=64,
        excitatory_fraction=0.95,
        connectivity=1.0,
        dt_ms=1.0,
        v_threshold_mv=-64.5,
        weight_max=50.0,
        refractory_ms=7.0,
        tau_m_ms=40.0,
    ),
    ModelConfig(
        n_neurons=128,
        excitatory_fraction=0.9,
        connectivity=1.0,
        dt_ms=1.0,
        v_threshold_mv=-64.5,
        weight_max=50.0,
        refractory_ms=1.0,
        tau_m_ms=40.0,
    ),
)


def _process_episode(
    backend: Any, config: ModelConfig, seed: int, workspace: Path, label: str
) -> tuple[str, int | None]:
    """Seal, re-read, and independently gate-recompute one real episode's descriptor."""
    inputs = gb.build_episode(config, seed, 8, 32)
    descriptor = evaluate_gb_descriptor(
        backend,
        inputs,
        8,
        config,
        completion_steps=32,
        spike_drift_ceiling=0.05,
        current_drift_ceiling=0.05,
        numerical_zero_floor=1e-09,
    )
    output = workspace / f"gb_{label}.json"
    write_artifact(
        gb.descriptor_payload(
            descriptor,
            lane_role="lane_p",
            state="fixture_only",
            calibration_spec_digest=SHA,
            scoring_target_digest=SHA,
        ),
        output,
    )
    evidence = read_artifact(output, expected_type="snn-memory-gb-preflight-evidence-v2")
    result = backend.run(inputs, 8, False, config)
    rows = _gate_rows(result, 8, 32)
    raster = _gate_raster(result, 8, 32)
    assert _gate_energy_bits(rows, config.dt_ms) == evidence.payload["recurrent_energy_bits"]
    assert _gate_half_life(rows, 4) == evidence.payload["half_life_steps"]
    assert _gate_hamming(raster, 4) == list(evidence.payload["bin_spike_hamming"])
    assert (
        _gate_class(rows, raster, 4, config.n_neurons, 0.05, 0.05, 1e-09)
        == evidence.payload["trajectory_class"]
    )
    return str(descriptor.trajectory_class), descriptor.half_life_steps


def _check_active_episode(
    workspace: Path, install_target: Path, extension_sha256: str
) -> list[str]:
    """Run bounded real fixture_only episodes; seal, read, and gate-recompute each.

    Every episode's evidence is sealed, re-read (running the experiment_lock reader
    recompute mirror), and cross-checked against a third independent gate recompute, so the
    non-silent trajectory, lag-drift, current-drift, half-life, and tail-silence paths are
    all exercised end to end. A high-threshold quiescent episode whose cue never fires keeps
    the recurrent net at zero, taking the zero-half-life branch in both recompute paths.
    """
    del install_target
    backend = load_stream_backend(BackendIdentity(2, "0.1.0", extension_sha256))
    _quiescent_class, quiescent_half_life = _process_episode(
        backend,
        ModelConfig(
            n_neurons=64,
            excitatory_fraction=0.8,
            connectivity=0.5,
            dt_ms=1.0,
            v_threshold_mv=1000.0,
        ),
        3,
        workspace,
        "quiescent",
    )
    assert quiescent_half_life == 0, "a quiescent episode must recompute a zero half-life"
    classes: list[str] = []
    for index, config in enumerate(_ACTIVE_CONFIGS):
        for seed in (3, 7):
            trajectory, _half_life = _process_episode(
                backend, config, seed, workspace, f"active_{index}_{seed}"
            )
            if trajectory != "silent_decay":
                classes.append(trajectory)
    assert classes, "no bounded active episode produced a non-silent trajectory"
    return sorted(set(classes))


# Preregistered finite real-backend witness grid for the two remaining reachable dynamics
# branches (bin-aligned periodicity and post-spike residual recurrent current). The grid is
# fixed before observation; thresholds are the frozen calibration values and are never relaxed.
_WITNESS_REFRACTORY = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0, 15.0)
_WITNESS_WEIGHT = (20.0, 40.0)
_WITNESS_SEEDS = (3, 7, 11)


def _has_residual_tail_current(result: Any) -> bool:
    """True if a tail completion step has no spikes yet recurrent net-L1 above the zero floor."""
    for step in range(16, 32):
        global_step = 8 + step
        start = int(result.spike_offsets[global_step])
        stop = int(result.spike_offsets[global_step + 1])
        if stop == start and float(result.current_values[global_step, 10]) > 1e-09:
            return True
    return False


def _witness_search(workspace: Path, extension_sha256: str) -> dict[str, Any]:
    """Run the preregistered finite real-backend witness search; seal/read any witness found."""
    backend = load_stream_backend(BackendIdentity(2, "0.1.0", extension_sha256))
    episodes = 0
    periodic: dict[str, Any] | None = None
    residual: dict[str, Any] | None = None
    for refractory in _WITNESS_REFRACTORY:
        for weight in _WITNESS_WEIGHT:
            for seed in _WITNESS_SEEDS:
                config = ModelConfig(
                    n_neurons=64,
                    excitatory_fraction=0.95,
                    connectivity=1.0,
                    dt_ms=1.0,
                    v_threshold_mv=-64.5,
                    weight_max=weight,
                    refractory_ms=refractory,
                    tau_m_ms=40.0,
                )
                episodes += 1
                inputs = gb.build_episode(config, seed, 8, 32)
                result = backend.run(inputs, 8, False, config)
                rows = _gate_rows(result, 8, 32)
                raster = _gate_raster(result, 8, 32)
                trajectory = _gate_class(rows, raster, 4, 64, 0.05, 0.05, 1e-09)
                if periodic is None and trajectory == "settled_periodic":
                    periodic = {"refractory_ms": refractory, "weight_max": weight, "seed": seed}
                    _process_episode(backend, config, seed, workspace, "witness_periodic")
                if residual is None and _has_residual_tail_current(result):
                    residual = {"refractory_ms": refractory, "weight_max": weight, "seed": seed}
                    _process_episode(backend, config, seed, workspace, "witness_residual")
    return {
        "grid": {
            "refractory_ms": list(_WITNESS_REFRACTORY),
            "weight_max": list(_WITNESS_WEIGHT),
            "seeds": list(_WITNESS_SEEDS),
            "n_neurons": 64,
            "connectivity": 1.0,
            "excitatory_fraction": 0.95,
            "v_threshold_mv": -64.5,
        },
        "episodes_evaluated": episodes,
        "settled_periodic_witness": periodic,
        "residual_tail_current_witness": residual,
    }


def _check_gb_cli_in_process(workspace: Path, extension_sha256: str) -> None:
    """Exercise the installed G-B CLI main() in-process (success + error exit)."""
    import contextlib
    import io
    import runpy

    def run(argv: list[str], expected_code: int) -> str:
        stdout, stderr = io.StringIO(), io.StringIO()
        previous = sys.argv
        sys.argv = ["python -m snn_memory.gb_preflight", *argv]
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                runpy.run_module("snn_memory.gb_preflight", run_name="__main__", alter_sys=False)
            raise AssertionError("gb_preflight __main__ did not exit")
        except SystemExit as exit_info:
            assert exit_info.code == expected_code, (
                f"gb CLI exit {exit_info.code}: {stderr.getvalue()}"
            )
        finally:
            sys.argv = previous
        return stdout.getvalue()

    out = run(_gb_child_argv(extension_sha256, workspace / "gb_inproc.json", SHA, SHA), 0)
    assert json.loads(out)["trajectory_class"] == "silent_decay"
    run(_gb_child_argv("0" * 64, workspace / "gb_inproc_err.json", SHA, SHA), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--install-target", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-digest", required=True)
    parser.add_argument("--python-wheel-sha256", required=True)
    parser.add_argument("--rust-wheel-sha256", required=True)
    parser.add_argument("--public-schema", type=Path, required=True)
    parser.add_argument("--public-license", type=Path, required=True)
    arguments = parser.parse_args()
    install_target = arguments.install_target.resolve()
    lock_origin = Path(str(el.__file__)).resolve()
    gb_origin = Path(str(gb.__file__)).resolve()
    assert lock_origin.is_relative_to(install_target)
    assert gb_origin.is_relative_to(install_target)
    packaged_schema = lock_origin.parent / "schema/snn_memory_experiment_lock_v2.schema.json"
    assert arguments.public_schema.read_bytes() == packaged_schema.read_bytes()
    assert (
        arguments.public_license.read_bytes()
        == packaged_schema.with_name(packaged_schema.name + ".license").read_bytes()
    )
    module_shas = {"lock": _sha(lock_origin.read_bytes()), "gb": _sha(gb_origin.read_bytes())}
    schema_sha256 = _sha(packaged_schema.read_bytes())

    repo = _fixture_repo(arguments.workspace)
    d1_art, d2_art, bundle_inventory_digest, bundle_arts = _acquire_d1_d2(
        arguments.workspace, repo, arguments.encoder_checkpoint, arguments.encoder_digest
    )
    d1_binding = _d1_binding(d1_art)
    d2_binding = _d2_binding(d1_binding, d2_art, bundle_inventory_digest)
    record_ids = list(d1_binding["selected_record_ids"])
    head = d1_binding["repository_head"]
    bindings: dict[str, str] = {
        "schema_sha256": schema_sha256,
        "python_wheel": arguments.python_wheel_sha256,
        "rust_wheel": arguments.rust_wheel_sha256,
        "extension": arguments.extension_sha256,
        "encoder_digest": arguments.encoder_digest,
        "lock_module": module_shas["lock"],
        "gb_module": module_shas["gb"],
        "dirty_tree_digest": _dirty_tree_digest(repo),
        "output_root_digest": _output_root_digest(arguments.workspace),
    }
    identities = _identities(bindings, head)
    cues = _cue_digests(d2_art)
    expected_task_set_digest = _expected_task_set_digest(cues)

    scoring = write_artifact(
        _scoring_target(record_ids, bindings), arguments.workspace / "scoring.json"
    )
    calibration = write_artifact(
        _calibration_spec(bindings), arguments.workspace / "calibration.json"
    )
    bundle_reps = _bundle_class_reps(bundle_arts[0])
    lane_c_manifest = _build_lane_c_diagnostic_manifest(arguments.workspace)
    lane_h_manifest = _build_lane_h_historical_manifest(arguments.workspace)
    foreign_lanes = _foreign_lanes(lane_c_manifest)
    real = _lane_real(scoring, calibration, bindings, expected_task_set_digest)
    lock = write_artifact(
        _lock_payload(d1_binding, d2_binding, identities, bindings, real, foreign_lanes),
        arguments.workspace / "lock.json",
    )
    completeness = write_artifact(
        _completeness_payload(True, cues), arguments.workspace / "complete.json"
    )
    bind_lane_isolation(lock, scoring, calibration, d1_art, d2_art, bundle_arts)
    bind_d1_d2(lock, d1_art, d2_art)
    bind_task_completeness(lock, completeness)

    foreign_d1 = _foreign_d1(arguments.workspace)
    foreign_bundle = _foreign_bundle(
        arguments.workspace, arguments.encoder_checkpoint, arguments.encoder_digest
    )
    _check_bundle_completeness_negatives(
        lock, scoring, calibration, d1_art, d2_art, bundle_arts, foreign_bundle
    )
    locked_identities = _locked_d2_inventory(d1_art, d2_art, bundle_arts)

    cli_pids = _check_cli(arguments.workspace, install_target, record_ids, lock, bindings)
    gb_evidence = _check_gb_two_child(
        arguments.workspace,
        install_target,
        arguments.extension_sha256,
        calibration.payload_self_sha256,
        scoring.payload_self_sha256,
    )
    active_class = _check_active_episode(
        arguments.workspace, install_target, arguments.extension_sha256
    )
    dynamics_witness = _witness_search(arguments.workspace, arguments.extension_sha256)
    _check_gb_cli_in_process(arguments.workspace, arguments.extension_sha256)
    witness = _u2_witness_search(arguments.extension_sha256)
    _check_gb_negatives(arguments.extension_sha256)
    _check_negatives(
        arguments.workspace,
        d1_binding,
        d2_binding,
        record_ids,
        scoring,
        calibration,
        identities,
        lock,
        d1_art,
        d2_art,
        bindings,
        locked_identities,
        foreign_d1,
        expected_task_set_digest,
        completeness,
        cues,
        real,
        bundle_reps,
        bundle_arts,
        foreign_lanes,
        lane_c_manifest,
        lane_h_manifest,
    )
    assert canonical_config_digest({"a": 1}) != model_config_digest(ModelConfig(n_neurons=64))

    print(
        json.dumps(
            {
                "status": "pass",
                "gate_pid": os.getpid(),
                "cli_pids": cli_pids,
                "cue_set_sha256": d2_art.file_sha256,
                "d1_file_sha256": d1_art.file_sha256,
                "gb_evidence": gb_evidence,
                "active_episode_class": active_class,
                "dynamics_witness": dynamics_witness,
                "u2_witness": witness,
                "lock_self_sha256": lock.payload_self_sha256,
                "lock_module_sha256": module_shas["lock"],
                "gb_module_sha256": module_shas["gb"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
