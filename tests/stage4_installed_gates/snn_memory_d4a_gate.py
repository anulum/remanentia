# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Installed-wheel D4-A checkpoint/candidate-bank materialization gate

"""Exercise D4-A only through installed wheels, real D1/D2, and the real Rust backend.

Two seeds and all five authorized conditions are materialized through the installed
public API with real fresh child-process calibration workers; every checkpoint bundle
and candidate bank is authenticated through its single-read reader; matched controls,
per-seed/per-condition isolation, the adversarial neuron-major signature witness, a
byte-determinism replay, and a real negative battery are exercised. No mock backend,
fabricated telemetry, D2 evaluation cue, or scientific outcome is produced.
"""

from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

import numpy as np

from snn_memory import candidate_bank_v2 as cbank
from snn_memory import checkpoint_bundle_v2 as cbv
from snn_memory import checkpoint_materialize_v2 as cmv
from snn_memory.candidate_bank_v2 import CandidateBankError, temporal_signature_v2
from snn_memory.checkpoint_bundle_v2 import CheckpointBundleError
from snn_memory.checkpoint_materialize_v2 import MaterializeError
from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.cue_materializer import validate_cue_set_bytes
from snn_memory.source_universe import write_source_universe

BASE_TIMESTAMP = 1_700_000_000
SEEDS = (11, 29)
CONDITIONS = ("trained", "shuffled", "random", "zero", "untrained")
COMPLETION_STEPS = 32
WORD_A = ("ledger", "orchard", "harbour", "granite", "willow", "meadow", "signal", "quarry")
WORD_B = ("morning", "evening", "quiet", "sudden", "steady", "distant", "narrow", "amber")
WORD_C = ("ridge", "valley", "coast", "plain", "summit", "delta", "basin", "moor")
FIXTURE_MODEL = ModelConfig(n_neurons=24, excitatory_fraction=0.75, connectivity=1.0, dt_ms=1.0,
                            v_threshold_mv=-64.5, weight_max=50.0, refractory_ms=1.0, tau_m_ms=40.0)
FIXTURE_ENCODER = EncoderConfig(feature_dim=16, packet_ms=5, silent_ms=2, active_fraction=0.1,
                                projection_seed=1729)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expect(substring: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except (CheckpointBundleError, CandidateBankError, MaterializeError, ValueError) as error:
        if substring not in str(error):
            raise AssertionError(f"unexpected error {error!r}, wanted {substring!r}") from error
        return
    raise AssertionError(f"expected a fail-closed error containing {substring!r}")


def _git(repo: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "-c", "gc.auto=0", *arguments],
                          capture_output=True, check=True).stdout


def _doc(record: int) -> bytes:
    lines = [
        f"Record {record} event {index:03d} explores {WORD_A[record % 8]} {WORD_B[index % 8]} "
        f"phenomena beside {WORD_C[(record + index) % 8]} landmark mark-{record}-{index}."
        for index in range(52)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_repo(repo: Path) -> None:
    docs = repo / "docs/public"
    docs.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "gc.auto", "0")
    _git(repo, "config", "user.name", "D4A Gate")
    _git(repo, "config", "user.email", "d4a@example.invalid")
    for record in range(16):
        (docs / f"record-{record:02d}.md").write_bytes(_doc(record))
    extra = docs / "manifest-source.md"
    extra.write_bytes(_doc(0).replace(b"Record 0", b"Source X"))
    mdir = repo / "experiments/snn_memory"
    mdir.mkdir(parents=True)
    development = {
        "schema_version": 1, "split": "development",
        "encoder_checkpoint": "../../.snn_models/model", "encoder_digest": "1" * 64,
        "entries": [{"label": "v1-dev-0", "path": "../../docs/public/manifest-source.md",
                     "sha256": _sha(extra.read_bytes())}],
    }
    (mdir / "development_corpus.json").write_bytes(json.dumps(development, sort_keys=True).encode())
    locked = {"schema_version": 1, "split": "locked-evaluation", "locked": True,
              "encoder_checkpoint": "../../.snn_models/model", "encoder_digest": "1" * 64, "entries": []}
    (mdir / "locked_evaluation_corpus.json").write_bytes(json.dumps(locked, sort_keys=True).encode())
    # A committed non-UTF8 regular blob at a NON-.md path: write_source_universe skips it (only .md is
    # selected + UTF-8 decoded), but a forged D1 record can reference this real committed blob so
    # materialize's live-blob UTF-8 guard (lines 512-513) is witnessed through the public CLI.
    (docs / "binary-blob.dat").write_bytes(b"\xff\xfe\x00\x80 not valid utf-8 \xff\xfe blob\n")
    _git(repo, "add", "-A")
    environment = dict(os.environ, GIT_AUTHOR_DATE=f"@{BASE_TIMESTAMP} +0000",
                       GIT_COMMITTER_DATE=f"@{BASE_TIMESTAMP} +0000")
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], env=environment, check=True)


_D2_SUBPROCESS = """
import json, sys
from pathlib import Path
from snn_memory.cue_materializer import materialize_cue_set
from snn_memory.contracts import EncoderConfig, ModelConfig
a = json.loads(sys.argv[1])
result = materialize_cue_set(
    Path(a["repo"]), Path(a["universe"]), a["d1_sha"], Path(a["encoder_checkpoint"]),
    a["encoder_digest"], Path(a["output"]),
    model=ModelConfig(**a["model"]), encoder_config=EncoderConfig(**a["encoder"]),
    input_current=a["input_current"])
sys.stdout.write(result.file_sha256)
"""


def _materialize_cue_set_subprocess(repo: Path, universe: Path, d1_sha: str, encoder_checkpoint: Path,
                                    encoder_digest: str, output: Path,
                                    environment: dict[str, str]) -> str:
    payload = {
        "repo": str(repo), "universe": str(universe), "d1_sha": d1_sha,
        "encoder_checkpoint": str(encoder_checkpoint), "encoder_digest": encoder_digest,
        "output": str(output), "input_current": 18.0, "model": FIXTURE_MODEL.to_dict(),
        "encoder": {"feature_dim": FIXTURE_ENCODER.feature_dim, "packet_ms": FIXTURE_ENCODER.packet_ms,
                    "silent_ms": FIXTURE_ENCODER.silent_ms, "active_fraction": FIXTURE_ENCODER.active_fraction,
                    "projection_seed": FIXTURE_ENCODER.projection_seed},
    }
    process = subprocess.run([sys.executable, "-c", _D2_SUBPROCESS, json.dumps(payload)],
                             capture_output=True, env=environment, check=False)
    assert process.returncode == 0, f"D2 subprocess failed: {process.stderr.decode()}"
    return process.stdout.decode()


def _acquire_d1_d2(workspace: Path, encoder_checkpoint: Path, encoder_digest: str,
                   environment: dict[str, str]) -> dict[str, Any]:
    repo = workspace / "d4a-repo"
    _build_repo(repo)
    universe_path = workspace / "universe.json"
    d1_result = write_source_universe(repo, universe_path)
    output = workspace / "cue-set"
    # Build D2 in a child process so the gate root never retains the sentence-encoder (torch) RSS;
    # the gate stays torch-free through the materialization loop and negative matrix.
    d2_file_sha256 = _materialize_cue_set_subprocess(
        repo, universe_path, d1_result.file_sha256, encoder_checkpoint, encoder_digest, output, environment)
    cue_raw = (output / "cue_set.json").read_bytes()
    cue_set = validate_cue_set_bytes(cue_raw, expected_file_sha256=d2_file_sha256)
    ids = [str(record_id) for record_id in cue_set.payload["source_universe"]["selected_record_ids"]]
    assert len(ids) >= 3, f"the fixture cue set must declare at least three candidates, got {len(ids)}"
    return {
        "repo": repo, "universe_path": universe_path, "d1_file_sha256": d1_result.file_sha256,
        "cue_set_path": output / "cue_set.json", "d2_file_sha256": d2_file_sha256,
        "ids": ids,
    }


def _base_config(arguments: argparse.Namespace, fixture: dict[str, Any]) -> dict[str, Any]:
    install = Path(arguments.install_target)
    schema_dir = install / "snn_memory/schema"

    def module(name: str) -> str:
        return _sha((install / "snn_memory" / f"{name}.py").read_bytes())

    import rust_snn_memory.rust_snn_memory as extension
    from snn_memory.experiment_lock import canonical_config_digest

    encoder_config = {
        "feature_dim": FIXTURE_ENCODER.feature_dim, "packet_ms": FIXTURE_ENCODER.packet_ms,
        "silent_ms": FIXTURE_ENCODER.silent_ms, "active_fraction": FIXTURE_ENCODER.active_fraction,
        "projection_seed": FIXTURE_ENCODER.projection_seed,
    }
    return {
        "repo": str(fixture["repo"]),
        "epochs": 1,
        "input_current": 18.0,
        "completion_steps": COMPLETION_STEPS,
        "model_config_digest": canonical_config_digest(FIXTURE_MODEL.to_dict()),
        "encoder_config_digest": canonical_config_digest(encoder_config),
        "encoder_config": encoder_config,
        "encoder_checkpoint": str(arguments.encoder_checkpoint),
        "encoder_digest": arguments.encoder_digest,
        "encoder_locator": "snn_models/all-MiniLM-L6-v2",
        "extension_sha256": arguments.extension_sha256,
        "crate_version": str(extension.CRATE_VERSION),
        "backend_version": str(extension.CRATE_VERSION),
        "backend_build_digest": arguments.extension_sha256,
        "python_wheel_sha256": arguments.python_wheel_sha256,
        "rust_wheel_sha256": arguments.rust_wheel_sha256,
        "repository_head": _git(fixture["repo"], "rev-parse", "HEAD").decode().strip(),
        "dirty_tree_digest": _sha(b"d4a-fixture-dirty-tree"),
        "patch_digest": _sha(b"d4a-fixture-patch"),
        "experiment_digest": _sha(b"d4a-fixture-experiment"),
        "task_set_digest": _sha(b"d4a-fixture-task-set"),
        "d1": str(fixture["universe_path"]),
        "d1_file_sha256": fixture["d1_file_sha256"],
        "cue_set": str(fixture["cue_set_path"]),
        "d2_file_sha256": fixture["d2_file_sha256"],
        "calibration_spec_digest": _sha(b"d4a-fixture-calibration-spec"),
        "development_artifact_digest": _sha(b"d4a-fixture-development-artifact"),
        "abstention_threshold": 0.1,
        "metric": "cosine",
        "materializer_module_sha256": module("checkpoint_materialize_v2"),
        "checkpoint_schema_sha256": _sha((schema_dir / "snn_memory_checkpoint_v2.schema.json").read_bytes()),
        "bank_schema_sha256": _sha((schema_dir / "snn_memory_candidate_bank_v2.schema.json").read_bytes()),
        "experiment_lock_schema_sha256": _sha(
            (schema_dir / "snn_memory_experiment_lock_v2.schema.json").read_bytes()),
        "experiment_lock_module_sha256": module("experiment_lock"),
        "gb_preflight_module_sha256": module("gb_preflight"),
    }


def _gate_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    # Keep child stderr controlled: silence the transformers/safetensors weight-loading progress
    # bars and advisory logging so a successful CLI child emits nothing on stderr.
    environment["TQDM_DISABLE"] = "1"
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    environment["TRANSFORMERS_VERBOSITY"] = "error"
    environment["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    return environment


def _run_cli(module_argv: list[str], environment: dict[str, str]) -> dict[str, Any]:
    """Run a CLI subcommand expecting exit 0, an empty stderr, and one canonical JSON line."""
    process = subprocess.run(
        [sys.executable, "-m", "snn_memory.checkpoint_materialize_v2", *module_argv],
        capture_output=True, env=environment, check=False,
    )
    assert process.returncode == 0, f"CLI exit {process.returncode}: {process.stderr.decode()}"
    assert process.stderr == b"", f"CLI emitted stray stderr: {process.stderr.decode()!r}"
    report = json.loads(process.stdout.decode())
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    assert process.stdout == canonical, "CLI stdout is not exactly one canonical JSON line"
    return report


def _run_cli_fail(module_argv: list[str], environment: dict[str, str], substring: str) -> None:
    process = subprocess.run(
        [sys.executable, "-m", "snn_memory.checkpoint_materialize_v2", *module_argv],
        capture_output=True, env=environment, check=False,
    )
    assert process.returncode == 2, (
        f"expected fail-closed exit 2, got {process.returncode}: {process.stdout.decode()}")
    error = process.stderr.decode()
    assert substring in error, f"unexpected CLI error {error!r}, wanted {substring!r}"


def _write_config(base: dict[str, Any], workspace: Path, seed: int, condition: str,
                  trained_bundle: Path | None, tag: str) -> Path:
    config = copy.deepcopy(base)
    config["seed"] = seed
    config["condition"] = condition
    config["output"] = str(workspace / f"out_{seed}_{condition}{tag}")
    if trained_bundle is not None:
        config["trained_bundle"] = str(trained_bundle)
    config_path = workspace / f"config_{seed}_{condition}{tag}.json"
    config_path.write_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return config_path


def _materialize(base: dict[str, Any], workspace: Path, seed: int, condition: str,
                 environment: dict[str, str], trained_bundle: Path | None = None,
                 tag: str = "") -> dict[str, Any]:
    config_path = _write_config(base, workspace, seed, condition, trained_bundle, tag)
    report = _run_cli(["materialize", "--config", str(config_path)], environment)
    report["output"] = str(workspace / f"out_{seed}_{condition}{tag}")
    return report


def _calibrate_argv(worker_dir: Path, cue: Path, record_id: str, extension_sha256: str,
                    crate_version: str, *, cue_digest: str | None = None,
                    backend_build_digest: str | None = None,
                    completion_steps: int = COMPLETION_STEPS) -> list[str]:
    def digest(path: Path) -> str:
        return _sha(path.read_bytes())

    return [
        "_calibrate",
        "--weights", str(worker_dir / "weights.bin"),
        "--weights-digest", digest(worker_dir / "weights.bin"),
        "--topology", str(worker_dir / "topology.bin"),
        "--topology-digest", digest(worker_dir / "topology.bin"),
        "--cue", str(cue), "--cue-digest", cue_digest if cue_digest is not None else digest(Path(cue)),
        "--model-config", str(worker_dir / "model_config.json"),
        "--model-config-digest", digest(worker_dir / "model_config.json"),
        "--record-id", record_id, "--nonce", os.urandom(16).hex(),
        "--completion-steps", str(completion_steps),
        "--extension-sha256", extension_sha256, "--crate-version", crate_version,
        "--backend-build-digest", backend_build_digest if backend_build_digest is not None
        else extension_sha256,
    ]


def _adversarial_signature_witness() -> None:
    n = 3
    raster = np.zeros((COMPLETION_STEPS, n), dtype=np.bool_)
    # Distinct per-neuron per-bin structure so every wrong representation differs from neuron-major.
    raster[0:4, 0] = True
    raster[4:8, 0] = True
    raster[8:12, 1] = True
    raster[0:2, 2] = True
    raster[16:20, 1] = True
    signature = temporal_signature_v2(raster, COMPLETION_STEPS, n)
    per_bin = COMPLETION_STEPS // 8
    neuron_major = signature.tolist()
    population_total = [float(raster[b * per_bin:(b + 1) * per_bin].sum()) for b in range(8)]
    bin_major = [float(raster[b * per_bin:(b + 1) * per_bin, neuron].sum())
                 for b in range(8) for neuron in range(n)]
    rate_only = [float(raster[:, neuron].sum()) for neuron in range(n)]
    assert len(neuron_major) == 8 * n
    assert neuron_major != bin_major, "neuron-major must differ from bin-major flattening"
    assert population_total != neuron_major[:8], "population-total must differ from neuron-major"
    assert rate_only != neuron_major[:n], "rate-only must differ from neuron-major"
    assert neuron_major[0:8] == [4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _expect("at least 32 steps and divisible by eight",
            lambda: temporal_signature_v2(np.zeros((24, n), dtype=np.bool_), 24, n))
    _expect("completion raster shape mismatch",
            lambda: temporal_signature_v2(np.zeros((COMPLETION_STEPS, n + 1), dtype=np.bool_),
                                          COMPLETION_STEPS, n))


def _rebind_manifest(manifest: dict[str, Any], role: str, digest: str) -> None:
    arrays, adjacency = manifest["arrays"], manifest["adjacency"]
    if role == "topology":
        manifest["topology_digest"] = digest
        adjacency["topology_digest"] = digest
    elif role == "adjacency_outgoing":
        adjacency["outgoing_digest"] = digest
    elif role == "adjacency_incoming":
        adjacency["incoming_digest"] = digest
    elif role == "weights":
        arrays["weights_digest"] = digest
    elif role == "training_final_state":
        arrays["training_final_state_digest"] = digest
    elif role == "probe_initial_state":
        arrays["probe_initial_state_digest"] = digest
    elif role == "signatures":
        arrays["signatures_digest"] = digest
    elif role == "record_ids":
        arrays["record_ids_digest"] = digest
        manifest["candidate_set_digest"] = digest
    elif role == "replay_schedule":
        manifest["replay_order_digest"] = digest


def _reseal_checkpoint(src: Path, dst: Path, role: str, new_body: bytes, new_semantic: str) -> Path:
    """Forge one component's bytes, recompute every dependent digest, and reseal consistently.

    Every digest — the component file/semantic digest, the manifest bindings, the manifest
    file/payload digest, and the descriptor self digest — is recomputed so the forgery passes all
    byte/digest checks and can only be caught by a semantic invariant in the reader.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descriptor = json.loads((dst / "descriptor.json").read_bytes())
    manifest = json.loads((dst / "manifest.json").read_bytes())
    component = next(entry for entry in descriptor["components"] if entry["role"] == role)
    (dst / component["path"]).write_bytes(new_body)
    component["file_sha256"] = _sha(new_body)
    component["semantic_digest"] = new_semantic
    component["byte_length"] = len(new_body)
    _rebind_manifest(manifest, role, new_semantic)
    manifest_bytes = cbv._canonical_bytes(manifest)
    (dst / "manifest.json").write_bytes(manifest_bytes)
    descriptor["checkpoint_manifest"]["file_sha256"] = _sha(manifest_bytes)
    descriptor["checkpoint_manifest"]["payload_self_sha256"] = cbv._manifest_payload_digest(manifest_bytes)
    if role == "replay_schedule":
        descriptor["replay_schedule_digest"] = new_semantic
    descriptor.pop("self_sha256", None)
    descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
    (dst / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))
    return dst


def _reseal_descriptor_field(src: Path, dst: Path, role: str, field: str, value: Any) -> Path:
    """Change one descriptor component metadata field and recompute the descriptor self digest."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descriptor = json.loads((dst / "descriptor.json").read_bytes())
    next(entry for entry in descriptor["components"] if entry["role"] == role)[field] = value
    descriptor.pop("self_sha256", None)
    descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
    (dst / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))
    return dst


def _reseal_manifest_field(src: Path, dst: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Mutate a manifest identity/build field and recompute manifest + descriptor digests."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descriptor = json.loads((dst / "descriptor.json").read_bytes())
    manifest = json.loads((dst / "manifest.json").read_bytes())
    mutate(manifest)
    manifest_bytes = cbv._canonical_bytes(manifest)
    (dst / "manifest.json").write_bytes(manifest_bytes)
    descriptor["checkpoint_manifest"]["file_sha256"] = _sha(manifest_bytes)
    descriptor["checkpoint_manifest"]["payload_self_sha256"] = cbv._manifest_payload_digest(manifest_bytes)
    descriptor.pop("self_sha256", None)
    descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
    (dst / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))
    return dst


def _check_resealed_forgeries(checkpoint_dir: Path, scratch: Path, untrained_dir: Path) -> None:
    """Genuinely resealed semantic forgeries: consistent digests, caught only by an invariant."""
    descriptor = json.loads((checkpoint_dir / "descriptor.json").read_bytes())
    identities = descriptor["identities"]
    n, n_exc = int(identities["n_neurons"]), int(identities["n_excitatory"])
    paths = cbv._COMPONENT_PATHS

    def body(role: str) -> bytes:
        return (checkpoint_dir / paths[role]).read_bytes()

    def semantic(role: str) -> str:
        return next(c for c in descriptor["components"] if c["role"] == role)["semantic_digest"]

    topology = np.frombuffer(body("topology"), dtype="|b1").reshape(n, n)

    # 1. Topology no longer equals the seed-derived connectivity.
    forged_topology = topology.copy()
    forged_topology[0, 1] = not forged_topology[0, 1]
    forged_topology = np.ascontiguousarray(forged_topology, dtype="|b1")
    _expect("sealed topology differs from the seed-derived connectivity",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_topology", "topology",
                forged_topology.tobytes(), cbv.topology_digest(forged_topology, n_exc))))

    # 2. Excitatory row forced negative (Dale) with digests resealed.
    weights = np.frombuffer(body("weights"), dtype="<f8").reshape(n, n).copy()
    edge = int(np.flatnonzero(topology[0])[0])
    weights[0, edge] = -1.0
    weights = np.ascontiguousarray(weights, dtype="<f8")
    _expect("an excitatory row carries a negative weight",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_dale", "weights",
                weights.tobytes(), cbv.checkpoint_component_digest("weights", "<f8", weights))))

    # 3. Outgoing CSR row reordered so it no longer ascends / matches the topology.
    values = np.frombuffer(body("adjacency_outgoing"), dtype="<u8")
    offsets, indices = values[: n + 1].copy(), values[n + 1:].copy()
    node = next(i for i in range(n) if int(offsets[i + 1]) - int(offsets[i]) >= 2)
    start, stop = int(offsets[node]), int(offsets[node + 1])
    indices[start:stop] = indices[start:stop][::-1]
    csr = cbv.CsrArrays(offsets=np.ascontiguousarray(offsets, dtype="<u8"),
                        indices=np.ascontiguousarray(indices, dtype="<u8"))
    _expect("is not strictly ascending",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_csr", "adjacency_outgoing",
                offsets.tobytes() + indices.tobytes(), cbv._csr_digest("outgoing", csr))))

    # 4. Replay order broken while the seed/positions are preserved.
    schedule = json.loads(body("replay_schedule"))
    schedule[0]["record_id"], schedule[1]["record_id"] = (
        schedule[1]["record_id"], schedule[0]["record_id"])
    _expect("replay order breaks the seeded permutation",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_replay", "replay_schedule",
                cbv._canonical_bytes(schedule), cbv.replay_schedule_digest(schedule))))

    # 5. A signature value forced non-finite with digests resealed.
    shape = tuple(int(dim) for dim in semantic_shape(descriptor, "signatures"))
    signatures = np.frombuffer(body("signatures"), dtype="<f8").reshape(shape).copy()
    signatures[0, 0] = np.nan
    signatures = np.ascontiguousarray(signatures, dtype="<f8")
    _expect("signatures carry a non-finite value",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_sig", "signatures",
                signatures.tobytes(), cbv.checkpoint_component_digest("signatures", "<f8", signatures))))

    # 6. Trained-final state substituted as the probe-initial state (non-fresh reset).
    _expect("probe-initial state voltage is not the fresh resting potential",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_probe", "probe_initial_state",
                body("training_final_state"), semantic("training_final_state"))))

    # 7. A weight magnitude driven above the sealed bound.
    weight_max = float(identities["model_config"]["weight_max"])
    over = np.frombuffer(body("weights"), dtype="<f8").reshape(n, n).copy()
    over[0, edge] = weight_max + 1.0
    over = np.ascontiguousarray(over, dtype="<f8")
    _expect("exceeds the sealed weight bound",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_bound", "weights",
                over.tobytes(), cbv.checkpoint_component_digest("weights", "<f8", over))))

    # 8. Incoming CSR row reordered so it no longer ascends / matches the topology.
    in_values = np.frombuffer(body("adjacency_incoming"), dtype="<u8")
    in_off, in_idx = in_values[: n + 1].copy(), in_values[n + 1:].copy()
    in_node = next(i for i in range(n) if int(in_off[i + 1]) - int(in_off[i]) >= 2)
    in_start, in_stop = int(in_off[in_node]), int(in_off[in_node + 1])
    in_idx[in_start:in_stop] = in_idx[in_start:in_stop][::-1]
    in_csr = cbv.CsrArrays(offsets=np.ascontiguousarray(in_off, dtype="<u8"),
                           indices=np.ascontiguousarray(in_idx, dtype="<u8"))
    _expect("incoming adjacency row",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_incsr", "adjacency_incoming",
                in_off.tobytes() + in_idx.tobytes(), cbv._csr_digest("incoming", in_csr))))

    # 9. Descriptor declares the wrong per-role dtype.
    _expect("topology component declares an unexpected dtype",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_dtype", "topology", "dtype", "<u8")))

    # 10. Descriptor declares the wrong framing version.
    _expect("topology component declares an unexpected framing version",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_framing", "topology", "framing_version", 9)))

    # 11. Topology raw byte outside {0, 1} at a set edge (digest still reseals).
    raw_bool = np.frombuffer(body("topology"), dtype=np.uint8).copy()
    raw_bool[int(np.flatnonzero(raw_bool)[0])] = 2
    boolean = np.ascontiguousarray(raw_bool.view("|b1").reshape(n, n), dtype="|b1")
    _expect("boolean bytes are not exactly zero or one",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_rawbool", "topology",
                raw_bool.tobytes(), cbv.topology_digest(boolean, n_exc))))

    # 12. Manifest seed no longer agrees with the descriptor identity.
    def swap_seed(manifest: dict[str, Any]) -> None:
        manifest["seed"] = 29 if int(manifest["seed"]) == 11 else 11

    _expect("descriptor seed disagrees with the manifest",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_manifest_field(
                checkpoint_dir, scratch / "rf_seed", swap_seed)))

    # 13. Topology self-connection on the diagonal.
    diagonal_topology = topology.copy()
    diagonal_topology[0, 0] = True
    diagonal_topology = np.ascontiguousarray(diagonal_topology, dtype="|b1")
    _expect("topology carries a self-connection on the diagonal",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_diag_topo", "topology",
                diagonal_topology.tobytes(), cbv.topology_digest(diagonal_topology, n_exc))))

    # 14. Weight self-connection on the diagonal (off the sealed topology).
    diag_weights = np.frombuffer(body("weights"), dtype="<f8").reshape(n, n).copy()
    diag_weights[0, 0] = 5.0
    diag_weights = np.ascontiguousarray(diag_weights, dtype="<f8")
    _expect("weights carry a self-connection on the diagonal",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_diag_w", "weights",
                diag_weights.tobytes(), cbv.checkpoint_component_digest("weights", "<f8", diag_weights))))

    # 15. Non-finite weight with digests resealed.
    inf_weights = np.frombuffer(body("weights"), dtype="<f8").reshape(n, n).copy()
    inf_weights[0, edge] = np.inf
    inf_weights = np.ascontiguousarray(inf_weights, dtype="<f8")
    _expect("weights carry a non-finite value",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_inf_w", "weights",
                inf_weights.tobytes(), cbv.checkpoint_component_digest("weights", "<f8", inf_weights))))

    # 16. Descriptor declares the wrong component shape.
    _expect("weights component shape mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_shape", "weights", "shape", [n, n + 1])))

    def _forge_replay(tag: str, mutate: Callable[[list[dict[str, Any]]], None], error: str) -> None:
        forged = json.loads(body("replay_schedule"))
        mutate(forged)
        _expect(error, lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
            checkpoint_dir, scratch / tag, "replay_schedule",
            cbv._canonical_bytes(forged), cbv.replay_schedule_digest(forged))))

    # 17. Replay no longer covers every completed epoch exactly once (an out-of-range epoch appears).
    _forge_replay("rf_replay_epoch",
                  lambda s: s[0].__setitem__("epoch", 9999),
                  "replay schedule does not cover every completed epoch exactly once")

    # 18. Replay positions within an epoch are not a full permutation.
    def _dup_position(schedule: list[dict[str, Any]]) -> None:
        by_epoch: dict[int, list[dict[str, Any]]] = {}
        for entry in schedule:
            by_epoch.setdefault(int(entry["epoch"]), []).append(entry)
        group = next(rows for rows in by_epoch.values() if len(rows) >= 2)
        group[1]["replay_position"] = group[0]["replay_position"]

    _forge_replay("rf_replay_perm", _dup_position,
                  "replay positions are not a full permutation")

    # 19. Replay entry carries a non-positive timestep count.
    _forge_replay("rf_replay_ts", lambda s: s[0].__setitem__("timesteps", 0),
                  "replay entry carries a non-positive timestep count")

    # 20-22. Descriptor component shape metadata disagrees with the sealed body (_decode_component).
    _expect("topology component shape mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_dc_topo", "topology", "shape", [n, n + 1])))
    _expect("adjacency_outgoing component shape mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_dc_adj", "adjacency_outgoing", "shape", [n + 999])))
    sig_shape = [int(d) for d in semantic_shape(descriptor, "signatures")]
    _expect("signatures component shape mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_dc_sig", "signatures", "shape",
                [sig_shape[0], sig_shape[1] + 8])))

    # 23-25. Descriptor identities disagree with the reconstructed model configuration.
    def _forge_identity(tag: str, mutate: Callable[[dict[str, Any]], None], error: str) -> None:
        _expect(error, lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_identities(
            checkpoint_dir, scratch / tag, mutate)))

    _forge_identity("rf_id_nn", lambda idn: idn.__setitem__("n_neurons", int(idn["n_neurons"]) + 1),
                    "descriptor neuron count disagrees with the model configuration")
    _forge_identity("rf_id_ne", lambda idn: idn.__setitem__("n_excitatory", int(idn["n_excitatory"]) + 1),
                    "descriptor excitatory count disagrees with the model configuration")
    _forge_identity("rf_id_wm", lambda idn: idn.__setitem__("weight_max", float(idn["weight_max"]) + 1.0),
                    "descriptor weight bound disagrees with the model configuration")

    # 26. A component's declared path no longer matches the fixed role path.
    _expect("component weights path mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_path", "weights", "path", "weights_forged.bin")))

    # 27. A component's declared semantic digest no longer matches its recomputed digest.
    _expect("component weights semantic digest mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_sem", "weights", "semantic_digest", "0" * 64)))

    # 28. The descriptor replay-schedule digest no longer matches the decoded schedule.
    def _wrong_replay_digest(desc: dict[str, Any]) -> None:
        desc["replay_schedule_digest"] = "0" * 64

    _expect("descriptor replay-schedule digest mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor(
                checkpoint_dir, scratch / "rf_replaydig", _wrong_replay_digest)))

    # 29-31. State-component forgeries (field-major layout: voltage|refractory|spikes|pre|post).
    def _forge_state(role: str, tag: str, offset: int, patch: bytes, error: str) -> None:
        buffer = bytearray(body(role))
        buffer[offset:offset + len(patch)] = patch
        forged = bytes(buffer)
        semantic = cbv._state_digest(cbv._decode_state(forged, n))
        _expect(error, lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
            checkpoint_dir, scratch / tag, role, forged, semantic)))

    _forge_state("training_final_state", "rf_state_nan", 0,
                 np.float64(np.nan).tobytes(), "state voltage is not finite <f8")
    _forge_state("probe_initial_state", "rf_probe_refrac", n * 8,
                 np.uint32(1).tobytes(), "refractory or spike field is not a fresh reset")
    _forge_state("probe_initial_state", "rf_probe_trace", n * 8 + n * 4 + n,
                 np.float64(1.0).tobytes(), "synaptic trace field is not a fresh reset")

    # 32-40. Malformed replay-schedule JSON reaches the strict array/entry parser branches.
    def _forge_replay_bytes(tag: str, raw: bytes, error: str) -> None:
        _expect(error, lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
            checkpoint_dir, scratch / tag, "replay_schedule", raw, "0" * 64)))

    _forge_replay_bytes("rf_rj_nonfinite", b"[NaN]", "contains a non-finite JSON constant")
    _forge_replay_bytes("rf_rj_dupkey", b'[{"a":1,"a":2}]', "contains a duplicate object key")
    _forge_replay_bytes("rf_rj_utf8", b"[\xff\xfe]", "is not strict UTF-8 JSON")
    _forge_replay_bytes("rf_rj_notarray", b"{}", "must be a JSON array")
    _forge_replay_bytes("rf_rj_noncanon", b"[ ]", "is not canonical JSON")
    _forge_replay_bytes("rf_rj_notobj", b"[1]\n", "replay entry must be a JSON object")
    _forge_replay_bytes("rf_rj_keys", b'[{"epoch":0}]\n',
                        "replay entry does not carry the exact schema keys")
    _forge_replay_bytes("rf_rj_int", b'[{"epoch":"0","record_id":"r","replay_position":0,'
                        b'"timesteps":1}]\n', "must be a JSON integer")
    _forge_replay_bytes("rf_rj_str", b'[{"epoch":0,"record_id":1,"replay_position":0,'
                        b'"timesteps":1}]\n', "must be a JSON string")

    # 41-42. Descriptor model configuration is corrupt (reconstruct/canonical/digest guards).
    def _drop_model_key(idn: dict[str, Any]) -> None:
        idn["model_config"] = {k: v for k, v in idn["model_config"].items() if k != "dt_ms"}

    _forge_identity("rf_model_incomplete", _drop_model_key,
                    "descriptor model config is not the full canonical configuration")

    def _tweak_model_value(idn: dict[str, Any]) -> None:
        idn["model_config"]["dt_ms"] = float(idn["model_config"]["dt_ms"]) + 1.0

    _forge_identity("rf_model_digest", _tweak_model_value,
                    "descriptor model config does not match its model-config digest")

    # 43. Model config that fails ModelConfig construction (n_neurons below the minimum).
    _forge_identity("rf_model_invalid",
                    lambda idn: idn["model_config"].__setitem__("n_neurons", 1),
                    "descriptor model configuration is invalid")

    # 44. An inhibitory row carries a positive weight (Dale violation on the inhibitory side).
    # (Off-topology weight 564 is class-C: connectivity=1.0 leaves no off-topology edge.)
    inh_edge = int(np.flatnonzero(topology[n_exc])[0])
    inh_weights = np.frombuffer(body("weights"), dtype="<f8").reshape(n, n).copy()
    inh_weights[n_exc, inh_edge] = 0.5
    inh_weights = np.ascontiguousarray(inh_weights, dtype="<f8")
    _expect("an inhibitory row carries a positive weight",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_w_inhpos", "weights",
                inh_weights.tobytes(), cbv.checkpoint_component_digest("weights", "<f8", inh_weights))))

    # 46. A read path carrying a '..' component is refused by the root-safe walk.
    _expect("path carries a non-normal '..' component",
            lambda: cbv.read_checkpoint_bundle_v2(checkpoint_dir / ".." / checkpoint_dir.name))

    # 47-48. Public write round-trip: a colliding target is refused no-clobber (staging cleanup
    # runs), and a target with no valid final component is refused.
    validated = cbv.read_checkpoint_bundle_v2(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_bytes())
    descriptor = json.loads((checkpoint_dir / "descriptor.json").read_bytes())

    def _write(target: Path) -> object:
        return cbv.write_checkpoint_bundle_v2(
            target, state=descriptor["state"], manifest=manifest,
            identities=descriptor["identities"], inputs=validated.inputs)

    def _write_inputs(tag: str, **overrides: Any) -> object:
        crafted = dataclasses.replace(validated.inputs, **overrides)
        return cbv.write_checkpoint_bundle_v2(
            scratch / tag, state=descriptor["state"], manifest=manifest,
            identities=descriptor["identities"], inputs=crafted)

    ostate = validated.inputs.training_final_state
    ocsr = validated.inputs.outgoing

    _expect("checkpoint bundle target already exists", lambda: _write(checkpoint_dir))
    _expect("checkpoint bundle target has no valid final component",
            lambda: _write(scratch / "bad" / ".."))

    # 55-56. Writer witnesses for CSR bracket and duplicate ordered record identifiers.
    _expect("adjacency offsets do not bracket the indices",
            lambda: _write_inputs("wt_csr_bracket", outgoing=cbv.CsrArrays(
                offsets=np.concatenate([[np.uint64(1)], ocsr.offsets[1:]]).astype(np.uint64),
                indices=np.ascontiguousarray(ocsr.indices, dtype=np.uint64))))
    dup_ids = (validated.inputs.ordered_record_ids[0],) + validated.inputs.ordered_record_ids[:-1]
    _expect("ordered record identifiers are not unique",
            lambda: _write_inputs("wt_dup_ids", ordered_record_ids=dup_ids))

    # 57. Adjacency component shorter than the offset row (reader decode guard).
    n_short = np.zeros(n, dtype=np.uint64)  # n < n+1 offset entries
    _expect("component is shorter than the offset row",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_adjacency_shape(
                checkpoint_dir, scratch / "rf_adj_short", n_short.tobytes())))

    # 58. State component whose declared shape is not (n,).
    _expect("component shape mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor_field(
                checkpoint_dir, scratch / "rf_state_shape", "training_final_state", "shape", [n, 1])))

    # 59. A non-canonical manifest whose file/payload digests still bind.
    _expect("checkpoint manifest is not canonical",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_noncanonical_manifest(
                checkpoint_dir, scratch / "rf_noncanon_manifest")))

    # 60-61. Untrained-condition replay/epoch guards (real untrained checkpoint surface):
    # place the trained (non-empty) replay schedule into the untrained checkpoint.
    u_trained_replay = json.loads((checkpoint_dir / "replay_schedule.json").read_bytes())
    # (untrained epochs!=0 = class-A: the schema conditionally pins untrained epochs_completed to 0.)
    _expect("an untrained checkpoint must carry no replay entries",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                untrained_dir, scratch / "rf_untr_replay", "replay_schedule",
                cbv._canonical_bytes(u_trained_replay),
                cbv.replay_schedule_digest(u_trained_replay))))

    # 49. Manifest payload-self-digest binding no longer matches the recomputed payload.
    _expect("checkpoint manifest payload digest mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor(
                checkpoint_dir, scratch / "rf_payload",
                lambda d: d["checkpoint_manifest"].__setitem__("payload_self_sha256", "0" * 64))))

    # 50. The declared component inventory is out of the fixed order.
    def _reorder(d: dict[str, Any]) -> None:
        d["components"] = list(reversed(d["components"]))

    _expect("component inventory is out of order",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_descriptor(
                checkpoint_dir, scratch / "rf_order", _reorder)))

    # 51. A state component whose byte length is short of the fixed per-neuron layout.
    short_state = body("training_final_state")[:-8]
    _expect("state component byte length mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_state_len", "training_final_state",
                short_state, "0" * 64)))

    # (Trained epochs<1 = class-A: the manifest schema pins epochs_completed >= 1.)

    # 52a-52h. Public-writer witnesses: write_checkpoint_bundle_v2 validates the RAW arrays via
    # _validate_topology/_csr/_weights/_state before encoding, so crafted valid-shape/wrong-dtype/
    # wrong-length inputs reach those guards through the real installed writer surface.
    _expect("topology shape differs from n_neurons",
            lambda: _write_inputs("wt_shape", topology=np.zeros((n, n + 1), dtype=np.bool_)))
    _expect("topology is not a boolean array",
            lambda: _write_inputs("wt_topo_dtype", topology=np.zeros((n, n), dtype=np.int8)))
    _expect("weights are not an <f8 square population matrix",
            lambda: _write_inputs("wt_w_dtype", weights=np.zeros((n, n), dtype=np.int64)))
    _expect("adjacency arrays are not <u8",
            lambda: _write_inputs("wt_csr_dtype", outgoing=cbv.CsrArrays(
                offsets=np.ascontiguousarray(ocsr.offsets, dtype=np.uint32),
                indices=np.ascontiguousarray(ocsr.indices, dtype=np.uint32))))
    _expect("adjacency offsets length mismatch",
            lambda: _write_inputs("wt_csr_len", outgoing=cbv.CsrArrays(
                offsets=np.zeros(n + 2, dtype=np.uint64),
                indices=np.ascontiguousarray(ocsr.indices, dtype=np.uint64))))
    _expect("float fields have the wrong length",
            lambda: _write_inputs("wt_state_flen", training_final_state=cbv.StateArrays(
                voltage_mv=np.zeros(n - 1, dtype=np.float64), refractory_steps=ostate.refractory_steps,
                spikes=ostate.spikes, pre_trace=ostate.pre_trace, post_trace=ostate.post_trace)))
    _expect("spike field has the wrong length",
            lambda: _write_inputs("wt_state_slen", training_final_state=cbv.StateArrays(
                voltage_mv=ostate.voltage_mv, refractory_steps=ostate.refractory_steps,
                spikes=np.zeros(n - 1, dtype=np.bool_), pre_trace=ostate.pre_trace,
                post_trace=ostate.post_trace)))
    _expect("refractory or trace fields have the wrong length",
            lambda: _write_inputs("wt_state_rlen", training_final_state=cbv.StateArrays(
                voltage_mv=ostate.voltage_mv, refractory_steps=np.zeros(n - 1, dtype=np.uint32),
                spikes=ostate.spikes, pre_trace=ostate.pre_trace, post_trace=ostate.post_trace)))
    _expect("refractory or spike dtype mismatch",
            lambda: _write_inputs("wt_state_dtype", training_final_state=cbv.StateArrays(
                voltage_mv=ostate.voltage_mv, refractory_steps=np.zeros(n, dtype=np.int64),
                spikes=ostate.spikes, pre_trace=ostate.pre_trace, post_trace=ostate.post_trace)))

    # 52. Signatures whose width is not eight bins per neuron.
    sig = np.frombuffer(body("signatures"), dtype="<f8").reshape(
        tuple(int(d) for d in semantic_shape(descriptor, "signatures")))
    narrow = np.ascontiguousarray(sig[:, :16], dtype="<f8")
    _expect("signatures are not an <f8 (records, eight-bins-per-neuron) block",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_narrow_signatures(
                checkpoint_dir, scratch / "rf_sig_width", narrow)))

    # 54. Outgoing CSR offsets are not monotonic.
    ov = np.frombuffer(body("adjacency_outgoing"), dtype="<u8")
    ooff, oidx = ov[: n + 1].copy(), ov[n + 1:].copy()
    ooff[1] = int(ooff[2]) + 1
    ocsr = cbv.CsrArrays(offsets=np.ascontiguousarray(ooff, dtype="<u8"),
                         indices=np.ascontiguousarray(oidx, dtype="<u8"))
    _expect("adjacency offsets are not monotonic",
            lambda: cbv.read_checkpoint_bundle_v2(_reseal_checkpoint(
                checkpoint_dir, scratch / "rf_csr_mono", "adjacency_outgoing",
                ooff.tobytes() + oidx.tobytes(), cbv._csr_digest("outgoing", ocsr))))


def _reseal_adjacency_shape(src: Path, dst: Path, new_body: bytes) -> Path:
    """Reseal a checkpoint whose outgoing adjacency body is a valid-shape but too-short u8 block."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descriptor = json.loads((dst / "descriptor.json").read_bytes())
    entry = next(c for c in descriptor["components"] if c["role"] == "adjacency_outgoing")
    entry["shape"] = [len(new_body) // 8]
    entry["byte_length"] = len(new_body)
    entry["file_sha256"] = _sha(new_body)
    (dst / (entry["path"])).write_bytes(new_body)
    descriptor.pop("self_sha256", None)
    descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
    (dst / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))
    return dst


def _reseal_noncanonical_manifest(src: Path, dst: Path) -> Path:
    """Reseal a checkpoint with an indented (non-canonical) manifest whose digests still bind."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descriptor = json.loads((dst / "descriptor.json").read_bytes())
    manifest = json.loads((dst / "manifest.json").read_bytes())
    non_canonical = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (dst / "manifest.json").write_bytes(non_canonical)
    descriptor["checkpoint_manifest"]["file_sha256"] = _sha(non_canonical)
    descriptor["checkpoint_manifest"]["payload_self_sha256"] = cbv._manifest_payload_digest(
        cbv._canonical_bytes(manifest))
    descriptor.pop("self_sha256", None)
    descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
    (dst / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))
    return dst


def _reseal_narrow_signatures(src: Path, dst: Path, narrow: Any) -> Path:
    """Reseal a checkpoint whose signatures component is a valid <f8 block of a narrower width."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descriptor = json.loads((dst / "descriptor.json").read_bytes())
    entry = next(c for c in descriptor["components"] if c["role"] == "signatures")
    entry["shape"] = [int(narrow.shape[0]), int(narrow.shape[1])]
    entry["byte_length"] = int(narrow.nbytes)
    entry["file_sha256"] = _sha(narrow.tobytes())
    entry["semantic_digest"] = cbv.checkpoint_component_digest("signatures", "<f8", narrow)
    (dst / "signatures.bin").write_bytes(narrow.tobytes())
    manifest = json.loads((dst / "manifest.json").read_bytes())
    _rebind_manifest(manifest, "signatures", entry["semantic_digest"])
    manifest_bytes = cbv._canonical_bytes(manifest)
    (dst / "manifest.json").write_bytes(manifest_bytes)
    descriptor["checkpoint_manifest"]["file_sha256"] = _sha(manifest_bytes)
    descriptor["checkpoint_manifest"]["payload_self_sha256"] = cbv._manifest_payload_digest(manifest_bytes)
    descriptor.pop("self_sha256", None)
    descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
    (dst / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))
    return dst


def _reseal_descriptor(src: Path, dst: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Mutate a top-level descriptor field and recompute only the descriptor self digest."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descriptor = json.loads((dst / "descriptor.json").read_bytes())
    mutate(descriptor)
    descriptor.pop("self_sha256", None)
    descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
    (dst / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))
    return dst


def _reseal_descriptor_identities(src: Path, dst: Path,
                                  mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Mutate a descriptor identities field and recompute only the descriptor self digest."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descriptor = json.loads((dst / "descriptor.json").read_bytes())
    mutate(descriptor["identities"])
    descriptor.pop("self_sha256", None)
    descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
    (dst / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))
    return dst


def semantic_shape(descriptor: dict[str, Any], role: str) -> list[int]:
    return next(c for c in descriptor["components"] if c["role"] == role)["shape"]


def _reseal_bank(src: Path, dst: Path, mutate: Callable[[Path, dict[str, Any]], None]) -> Path:
    """Forge bank bytes/manifest and recompute the self digest so only an invariant can catch it."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    manifest = json.loads((dst / "bank.json").read_bytes())
    mutate(dst, manifest)
    manifest.pop("self_sha256", None)
    manifest["self_sha256"] = cbank._self_digest(manifest, cbank._BANK_DOMAIN)
    (dst / "bank.json").write_bytes(cbv._canonical_bytes(manifest))
    return dst


def _check_resealed_bank_forgeries(bank_dir: Path, scratch: Path) -> None:
    def calibration_swap(_dst: Path, manifest: dict[str, Any]) -> None:
        calibration = manifest["calibration"]
        calibration[0]["record_id"], calibration[1]["record_id"] = (
            calibration[1]["record_id"], calibration[0]["record_id"])

    _expect("bank calibration order or count differs from the ordered records",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_calibration", calibration_swap)))

    def signature_nonfinite(dst: Path, manifest: dict[str, Any]) -> None:
        array = np.frombuffer((dst / "signatures.bin").read_bytes(), dtype="<f8").copy()
        array[0] = np.nan
        forged = np.ascontiguousarray(array, dtype="<f8").tobytes()
        (dst / "signatures.bin").write_bytes(forged)
        manifest["signatures"]["file_sha256"] = _sha(forged)

    _expect("non-finite value",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_signature", signature_nonfinite)))

    def layout_completion(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["signature_layout"]["completion_steps"] = COMPLETION_STEPS + 8

    _expect("signature layout and scoring completion windows disagree",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_layout", layout_completion)))

    _WRONG_SHA = "0" * 64

    def build_schema_hash(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["build"]["bank_schema_sha256"] = _WRONG_SHA

    _expect("bank build bank_schema_sha256 differs from the packaged schema hash",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_build_schema", build_schema_hash)))

    def calibration_field_disagree(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["calibration"][0]["encoder_identity"] = "forged-encoder-identity"

    _expect("bank calibration entries disagree on encoder_identity",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_cal_disagree", calibration_field_disagree)))

    def build_encoder_directory(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["build"]["encoder_directory_digest"] = _WRONG_SHA

    _expect("bank calibration encoder directory differs from the build identity",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_build_encdir", build_encoder_directory)))

    def build_encoder_config(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["build"]["encoder_config_digest"] = _WRONG_SHA

    _expect("bank calibration encoder config differs from the build identity",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_build_enccfg", build_encoder_config)))

    def build_backend(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["build"]["backend_build_digest"] = _WRONG_SHA

    _expect("bank calibration backend-build digest differs from the build identity",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_build_backend", build_backend)))

    def calibration_shape(_dst: Path, manifest: dict[str, Any]) -> None:
        for entry in manifest["calibration"]:
            entry["shape"] = [COMPLETION_STEPS, FIXTURE_MODEL.n_neurons + 1]

    _expect("bank calibration current shape is not (timesteps, n_neurons)",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_cal_shape", calibration_shape)))

    def calibration_weight_drift(_dst: Path, manifest: dict[str, Any]) -> None:
        for entry in manifest["calibration"]:
            entry["final_weights_digest"] = _WRONG_SHA

    _expect("bank calibration weights changed under disabled plasticity",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_cal_weight", calibration_weight_drift)))

    def signature_value_drift(dst: Path, manifest: dict[str, Any]) -> None:
        array = np.frombuffer((dst / "signatures.bin").read_bytes(), dtype="<f8").copy()
        array[0] = array[0] + 1.0  # finite change: passes canonical, fails the stored raw-byte digest
        forged = np.ascontiguousarray(array, dtype="<f8").tobytes()
        (dst / "signatures.bin").write_bytes(forged)
        manifest["signatures"]["file_sha256"] = _sha(forged)
        manifest["signatures"]["byte_length"] = len(forged)

    _expect("signature raw-byte digest mismatch",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_sig_rawdrift", signature_value_drift)))

    def signature_bytelength(dst: Path, manifest: dict[str, Any]) -> None:
        body = (dst / "signatures.bin").read_bytes()[:-8]  # one float64 short of the declared shape
        (dst / "signatures.bin").write_bytes(body)
        manifest["signatures"]["file_sha256"] = _sha(body)
        manifest["signatures"]["byte_length"] = len(body)

    _expect("signature byte length differs from the declared bank shape",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_sig_bytelen", signature_bytelength)))

    def scoring_bank_digest(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["scoring"]["candidate_bank_digest"] = _WRONG_SHA

    _expect("scoring candidate-bank digest does not recompute from the rows",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_scoring_digest", scoring_bank_digest)))

    def candidate_set_digest(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["identities"]["candidate_set_digest"] = _WRONG_SHA

    _expect("candidate-set digest does not recompute from the ordered records",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_candset_digest", candidate_set_digest)))

    # Crafted-path negative: a target carrying a '..' component is refused by the root-safe walk.
    _expect("path carries a non-normal '..' component",
            lambda: cbank.read_candidate_bank_v2(bank_dir / ".." / bank_dir.name))

    # Public write-API negatives: crafted signatures/manifest reach the writer's canonical guards.
    base_manifest = json.loads((bank_dir / "bank.json").read_bytes())
    base_manifest.pop("self_sha256", None)
    base_manifest.pop("signatures", None)
    n_neurons = FIXTURE_MODEL.n_neurons
    width = 8 * n_neurons
    rows = len(base_manifest["identities"]["ordered_record_ids"])
    good_signatures = np.frombuffer((bank_dir / "signatures.bin").read_bytes(),
                                    dtype="<f8").reshape(rows, width).copy()

    def _write(target_name: str, manifest: dict[str, Any], signatures: Any) -> object:
        return cbank.write_candidate_bank_v2(scratch / target_name, manifest=manifest,
                                             signatures=signatures)

    _expect("written signatures shape differs from the ordered record bank",
            lambda: _write("w_shape", json.loads(json.dumps(base_manifest)),
                           good_signatures.reshape(rows * width)))
    _expect("is not a canonical <f8 array",
            lambda: _write("w_dtype", json.loads(json.dumps(base_manifest)),
                           good_signatures.astype("<f4")))
    _expect("is not C-contiguous",
            lambda: _write("w_contig", json.loads(json.dumps(base_manifest)),
                           np.asfortranarray(good_signatures) if width > 1 else good_signatures.copy()))

    def _int_id_manifest() -> dict[str, Any]:
        forged = json.loads(json.dumps(base_manifest))
        forged["identities"]["ordered_record_ids"][0] = 12345
        return forged

    _expect("signature row record identifier is not a string",
            lambda: _write("w_intid", _int_id_manifest(), good_signatures.copy()))

    # Filesystem-fault negatives: a directory hits the regular-file guard; a symlink is refused
    # by the O_NOFOLLOW open before it can be dereferenced.
    def signatures_as_directory(dst: Path, _manifest: dict[str, Any]) -> None:
        (dst / "signatures.bin").unlink()
        (dst / "signatures.bin").mkdir()

    _expect("is not a regular file",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_sig_dir", signatures_as_directory)))

    def signatures_as_symlink(dst: Path, _manifest: dict[str, Any]) -> None:
        (dst / "signatures.bin").unlink()
        (dst / "signatures.bin").symlink_to(dst / "bank.json")

    _expect("cannot be opened safely",
            lambda: cbank.read_candidate_bank_v2(
                _reseal_bank(bank_dir, scratch / "rb_sig_symlink", signatures_as_symlink)))

    # Bad-final-component negative: a target whose final component is '..' is refused.
    _expect("candidate bank target has no valid final component",
            lambda: _write_to_bad_target(scratch, base_manifest, good_signatures))

    # renameat2 non-EEXIST errno (line 290 / arc 288->290) plus the rmdir-fail cleanup (322/323): a
    # concurrent actor strips write permission from the caller-supplied output parent the instant the
    # staging directory appears (after os.mkdir, before renameat2), so RENAME_NOREPLACE fails with EACCES
    # and best-effort cleanup cannot rmdir the staging entry under the now-unwritable parent. Fail-closed:
    # no accepted bank is installed; the empty staging entry legitimately remains as residue.
    _witness_bank_rename_eacces(scratch, base_manifest, good_signatures)


def _witness_bank_rename_eacces(scratch: Path, manifest: dict[str, Any], signatures: Any) -> None:
    """Drive candidate_bank_v2 line 290 / arc 288->290 (non-EEXIST rename errno) to EACCES on the installed
    public writer surface, and with it the rmdir-fail cleanup (322/323). A watcher thread strips write
    permission from the caller-supplied output parent once the staging directory exists, so the final
    renameat2(RENAME_NOREPLACE) fails with EACCES and the cleanup cannot remove the staging entry. Asserts
    fail-closed: no accepted bank, staging residue present under the (restored) parent."""
    parent = scratch / "bank_rename_eacces_parent"
    if parent.exists():
        os.chmod(parent, 0o755)
        shutil.rmtree(parent, ignore_errors=True)
    parent.mkdir()
    target = parent / "bank"
    final = target.name
    stop = threading.Event()
    fired = {"chmod": False}

    def _watch() -> None:
        while not stop.is_set():
            try:
                entries = os.listdir(parent)
            except OSError:
                entries = []
            if any(entry.startswith(f".{final}.staging-") for entry in entries):
                os.chmod(parent, 0o555)
                fired["chmod"] = True
                return
            time.sleep(0.0002)

    watcher = threading.Thread(target=_watch)
    watcher.start()
    message: str | None = None
    try:
        cbank.write_candidate_bank_v2(target, manifest=json.loads(json.dumps(manifest)),
                                      signatures=signatures.copy())
    except CandidateBankError as error:
        message = str(error)
    finally:
        stop.set()
        watcher.join(timeout=5)
        os.chmod(parent, 0o755)
    residue = [entry for entry in os.listdir(parent) if entry.startswith(f".{final}.staging-")]
    if not (fired["chmod"] and message is not None
            and "atomic no-replace install failed" in message and "Permission denied" in message):
        raise AssertionError(
            f"bank rename EACCES witness did not fire (chmod={fired['chmod']} message={message!r})")
    if target.exists():
        raise AssertionError("bank rename EACCES witness accepted an output despite the failed rename")
    if not residue:
        raise AssertionError("bank rename EACCES witness expected staging residue under the unwritable parent")
    shutil.rmtree(parent, ignore_errors=True)


def _witness_root_open_emfile() -> None:
    """Witness the os.open('/') root-open error handler in all three modules (candidate_bank 260-261,
    checkpoint_bundle 276-277, checkpoint_materialize 255-256). os.open('/', O_DIRECTORY|O_NOFOLLOW) is not
    infallible: under process fd exhaustion it raises EMFILE. Clamp RLIMIT_NOFILE to just above the live fd
    count, exhaust the remaining slots with /dev/null opens, then drive each module's _open_dir_from_root so
    the anchoring os.open('/') fails and the typed '… filesystem root cannot be opened safely' fires. The
    limit and the hog descriptors are always restored. Runs in the coverage-traced gate process, so each
    module's handler lines are recorded against the installed module under coverage."""
    import resource

    cases = [
        (cbank._open_dir_from_root, CandidateBankError),
        (cbv._open_dir_from_root, CheckpointBundleError),
        (cmv._open_dir_from_root, MaterializeError),
    ]
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for open_root, error_type in cases:
        message: str | None = None
        live = len(os.listdir("/proc/self/fd"))
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, live + 16), hard))
            hogs: list[int] = []
            try:
                while True:
                    hogs.append(os.open(os.devnull, os.O_RDONLY))
            except OSError:
                pass
            try:
                open_root(Path("/nonexistent/root-open-witness"), "root-open EMFILE witness")
            except error_type as error:
                message = str(error)
            finally:
                for fd in hogs:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        if not (message is not None and "filesystem root cannot be opened safely" in message):
            raise AssertionError(
                f"{error_type.__name__} root-open EMFILE witness did not fire (message={message!r})")


def _write_to_bad_target(scratch: Path, manifest: dict[str, Any], signatures: Any) -> object:
    return cbank.write_candidate_bank_v2(scratch / "bad" / "..",
                                         manifest=json.loads(json.dumps(manifest)),
                                         signatures=signatures.copy())


def _tamper(src: Path, dst: Path, mutate: Callable[[Path], None]) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    mutate(dst)
    return dst


def _check_checkpoint_negatives(bundle_dir: Path, scratch: Path) -> None:
    def break_component(directory: Path) -> None:
        (directory / "weights.bin").write_bytes(b"\x00" * (directory / "weights.bin").stat().st_size)

    _expect("weights file digest or length mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_tamper(bundle_dir, scratch / "c_weights", break_component)))

    def break_manifest(directory: Path) -> None:
        payload = json.loads((directory / "manifest.json").read_bytes())
        (directory / "manifest.json").write_bytes(json.dumps(payload, indent=2).encode())

    _expect("checkpoint manifest file digest mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_tamper(bundle_dir, scratch / "c_manifest", break_manifest)))

    def drop_component(directory: Path) -> None:
        (directory / "record_ids.json").unlink()

    _expect("directory inventory does not match the descriptor",
            lambda: cbv.read_checkpoint_bundle_v2(_tamper(bundle_dir, scratch / "c_drop", drop_component)))

    def add_component(directory: Path) -> None:
        (directory / "unexpected.bin").write_bytes(b"x")

    _expect("directory inventory does not match the descriptor",
            lambda: cbv.read_checkpoint_bundle_v2(_tamper(bundle_dir, scratch / "c_extra", add_component)))

    def symlink_component(directory: Path) -> None:
        target = directory / "weights.bin"
        target.unlink()
        target.symlink_to(directory / "topology.bin")

    _expect("cannot be opened safely",
            lambda: cbv.read_checkpoint_bundle_v2(_tamper(bundle_dir, scratch / "c_link", symlink_component)))

    def break_descriptor(directory: Path) -> None:
        payload = json.loads((directory / "descriptor.json").read_bytes())
        payload["state"] = "pre_result"
        (directory / "descriptor.json").write_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode())

    _expect("descriptor self digest mismatch",
            lambda: cbv.read_checkpoint_bundle_v2(_tamper(bundle_dir, scratch / "c_desc", break_descriptor)))

    _expect("directory cannot be opened safely",
            lambda: cbv.read_checkpoint_bundle_v2(scratch / "c_absent"))


_JSON_VARIANTS = (
    (b'{"a":Infinity}', "non-finite JSON constant"),
    (b'{"a":1,"a":2}', "duplicate object key"),
    (b"\xff\xfe", "not strict UTF-8 JSON"),
    (b"[1,2]\n", "root must be an object"),
    (b'{"schema_version":2}\n', "failed schema"),
    (b'{"schema_version":2}trailing', "not strict UTF-8 JSON"),
    (b'{"schema_version":3}\n', "failed schema"),
)


def _check_bundle_reader_negatives(bundle_dir: Path, scratch: Path) -> None:
    for index, (content, error) in enumerate(_JSON_VARIANTS):
        def mutate(directory: Path, payload: bytes = content) -> None:
            (directory / "descriptor.json").write_bytes(payload)

        _expect(error, lambda mutation=mutate, tag=index:
                cbv.read_checkpoint_bundle_v2(_tamper(bundle_dir, scratch / f"dr_{tag}", mutation)))

    def descriptor_dir(directory: Path) -> None:
        (directory / "descriptor.json").unlink()
        (directory / "descriptor.json").mkdir()

    _expect("is not a regular file",
            lambda: cbv.read_checkpoint_bundle_v2(_tamper(bundle_dir, scratch / "dr_dir", descriptor_dir)))


def _check_bank_negatives(bank_dir: Path, scratch: Path) -> None:
    real = json.loads((bank_dir / "bank.json").read_bytes())
    for index, (content, error) in enumerate(_JSON_VARIANTS):
        def mutate(directory: Path, payload: bytes = content) -> None:
            (directory / "bank.json").write_bytes(payload)

        _expect(error, lambda mutation=mutate, tag=index:
                cbank.read_candidate_bank_v2(_tamper(bank_dir, scratch / f"br_{tag}", mutation)))

    def reformat(directory: Path) -> None:
        (directory / "bank.json").write_bytes(json.dumps(real, indent=2).encode("utf-8"))

    _expect("candidate bank manifest is not canonical",
            lambda: cbank.read_candidate_bank_v2(_tamper(bank_dir, scratch / "br_canon", reformat)))

    def signature_dir(directory: Path) -> None:
        (directory / "signatures.bin").unlink()
        (directory / "signatures.bin").mkdir()

    _expect("is not a regular file",
            lambda: cbank.read_candidate_bank_v2(_tamper(bank_dir, scratch / "br_sigdir", signature_dir)))

    plain = {key: value for key, value in real.items() if key not in ("signatures", "self_sha256")}
    signatures = cbank.read_candidate_bank_v2(bank_dir).signatures
    _expect("candidate bank target already exists",
            lambda: cbank.write_candidate_bank_v2(bank_dir, manifest=plain, signatures=signatures))
    bad = copy.deepcopy(plain)
    bad["scoring"]["candidate_bank_digest"] = "0" * 64
    _expect("does not recompute from the ordered rows",
            lambda: cbank.write_candidate_bank_v2(scratch / "bw_bad", manifest=bad, signatures=signatures))
    (scratch).mkdir(parents=True, exist_ok=True)
    (scratch / "not_a_dir").write_bytes(b"x")
    _expect("candidate bank directory cannot be opened safely",
            lambda: cbank.read_candidate_bank_v2(scratch / "not_a_dir"))


def _check_signature_negatives(bank_dir: Path, scratch: Path) -> None:
    def break_signatures(directory: Path) -> None:
        path = directory / "signatures.bin"
        data = bytearray(path.read_bytes())
        data[0] ^= 0xFF
        path.write_bytes(bytes(data))

    _expect("signature file digest or length mismatch",
            lambda: cbank.read_candidate_bank_v2(_tamper(bank_dir, scratch / "b_sig", break_signatures)))

    def break_manifest(directory: Path) -> None:
        payload = json.loads((directory / "bank.json").read_bytes())
        payload["state"] = "pre_result"
        (directory / "bank.json").write_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode())

    _expect("candidate bank self digest mismatch",
            lambda: cbank.read_candidate_bank_v2(_tamper(bank_dir, scratch / "b_self", break_manifest)))

    def add_file(directory: Path) -> None:
        (directory / "extra.bin").write_bytes(b"x")

    _expect("inventory is not exactly bank.json and signatures.bin",
            lambda: cbank.read_candidate_bank_v2(_tamper(bank_dir, scratch / "b_extra", add_file)))


def _check_materialize_negatives(base: dict[str, Any], workspace: Path, trained_bundle: Path,
                                 untrained_bundle: Path, worker_dir: Path, extension_sha256: str,
                                 crate_version: str, environment: dict[str, str]) -> None:
    def run_neg(mutate: Callable[[dict[str, Any]], None], tag: str, substring: str) -> None:
        config = copy.deepcopy(base)
        config["seed"] = 11
        config["condition"] = "untrained"
        config["output"] = str(workspace / f"neg_{tag}")
        mutate(config)
        path = workspace / f"neg_{tag}.json"
        path.write_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        _run_cli_fail(["materialize", "--config", str(path)], environment, substring)

    for forbidden in ("encoder-only", "temporal-order-permuted", "leave-one-record-out", "G1-BG", "G1-STATE"):
        run_neg(lambda c, value=forbidden: c.__setitem__("condition", value),
                f"forbidden_{forbidden}", "is forbidden in D4-A")
    run_neg(lambda c: c.__setitem__("condition", "bogus"), "unknown", "is not an accepted D4-A condition")
    for field in ("evaluation_cues", "evaluation_base_cues", "expected_answers", "expected_record_id",
                  "expected_record_ids", "answer_key"):
        run_neg(lambda c, key=field: c.__setitem__(key, []), f"eval_{field}",
                "rejects the evaluation/expected-answer field")
    run_neg(lambda c: c.__setitem__("model_config_digest", "0" * 64), "modelcfg",
            "declared model-config digest differs from the D2 model configuration")
    run_neg(lambda c: c.__setitem__("encoder_digest", "0" * 64), "encdigest",
            "sentence-encoder directory digest differs")
    run_neg(lambda c: c.__setitem__("unexpected_field", 1), "unknownfield", "config carries unknown field")

    # Live-identity digest guards (_validate_live_identities): forged module/schema digests.
    run_neg(lambda c: c.__setitem__("materializer_module_sha256", "0" * 64), "matmod",
            "live materializer module hash differs from the configured digest")
    run_neg(lambda c: c.__setitem__("checkpoint_schema_sha256", "0" * 64), "cpschema",
            "live snn_memory_checkpoint_v2.schema.json hash differs from the configured digest")
    run_neg(lambda c: c.__setitem__("bank_schema_sha256", "0" * 64), "bankschema",
            "live snn_memory_candidate_bank_v2.schema.json hash differs from the configured digest")

    # Malformed config FILES (_strict_config): raw bytes that fail the strict/canonical parse.
    def run_neg_raw(transform: Callable[[bytes], bytes], tag: str, substring: str) -> None:
        cfg = copy.deepcopy(base)
        cfg["seed"] = 11
        cfg["condition"] = "untrained"
        cfg["output"] = str(workspace / f"negraw_{tag}")
        raw = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path = workspace / f"negraw_{tag}.json"
        path.write_bytes(transform(raw))
        _run_cli_fail(["materialize", "--config", str(path)], environment, substring)

    run_neg_raw(lambda r: r[:-1] + b',"seed":9}', "dupkey", "config contains a duplicate key")
    run_neg_raw(lambda r: r.replace(b'"condition":', b'"_nan":NaN,"condition":', 1), "nan",
                "config contains a non-finite JSON constant")
    run_neg_raw(lambda r: json.dumps(json.loads(r), indent=2).encode("utf-8"), "noncanon",
                "config is not canonical JSON")
    run_neg_raw(lambda r: b"[]", "nondict", "config root must be a JSON object")
    run_neg_raw(lambda r: b"{not json", "notjson", "config is not strict UTF-8 JSON")

    # Prepare/materialize cross-binding guards reachable by a single forged config field.
    run_neg(lambda c: c.__setitem__("repository_head", "0" * 40), "rephead",
            "D1 repository HEAD differs from the declared repository head")
    run_neg(lambda c: c.__setitem__("encoder_config_digest", "0" * 64), "enccfgdig",
            "declared encoder-config digest differs from the D2 encoder configuration")
    run_neg(lambda c: c["encoder_config"].__setitem__(
        "max_events", int(c["encoder_config"].get("max_events", 8)) + 1), "enccfg",
            "declared encoder config differs from its declared digest")
    run_neg(lambda c: c.__setitem__("input_current", float(c["input_current"]) + 1.0), "incur",
            "declared input current differs from the D2 model input current")
    run_neg(lambda c: c.__setitem__("d1", "/nonexistent_d4a_probe/d1.json"), "badd1parent",
            "cannot be opened safely")
    run_neg(lambda c: c.__setitem__("backend_build_digest", "0" * 64), "bbd",
            "declared backend-build digest is not the loaded extension identity")
    run_neg(lambda c: c.__setitem__("output", "/"), "outroot",
            "materialization output has no valid final component")
    run_neg(lambda c: (c.__setitem__("condition", "shuffled"), c.__setitem__("seed", 29),
                       c.__setitem__("trained_bundle", str(trained_bundle))), "controlseed",
            "control seed differs from the trained checkpoint seed")
    run_neg(lambda c: (c.__setitem__("condition", "shuffled"),
                       c.__setitem__("trained_bundle", str(untrained_bundle))), "notrained",
            "a control must derive from a trained checkpoint")
    run_neg(lambda c: c.__setitem__("output", str(untrained_bundle.parent)), "clobber",
            "materialization output already exists")

    # A config path that traverses through '..' is rejected at the root-anchored config read.
    _run_cli_fail(["materialize", "--config", str(workspace / "d4a-repo/../neg_forbidden.json")],
                  environment, "non-normal '..' component")

    # Real calibration-worker CLI negatives.
    bad_cue = workspace / "bad_cue.bin"
    bad_cue.write_bytes(b"\x00" * (FIXTURE_MODEL.n_neurons * 8 + 8))
    _run_cli_fail(_calibrate_argv(worker_dir, bad_cue, "sha256:" + "0" * 64, extension_sha256, crate_version),
                  environment, "not a whole (timesteps, n_neurons) grid")
    _run_cli_fail(_calibrate_argv(worker_dir, worker_dir / "cue_000.bin", "sha256:" + "0" * 64,
                                  extension_sha256, crate_version, cue_digest="0" * 64),
                  environment, "worker input digest mismatch")
    # A worker backend-build digest that is not the loaded extension identity is rejected.
    _run_cli_fail(_calibrate_argv(worker_dir, worker_dir / "cue_000.bin", "sha256:" + "0" * 64,
                                  extension_sha256, crate_version, backend_build_digest="0" * 64),
                  environment, "worker backend-build digest is not the loaded extension identity")

    # Worker-side calibration guards (_cmd_calibrate): reseal the authenticated worker directory
    # with one forged input; _calibrate_argv recomputes every digest from the forged files, so
    # control passes _authenticated_worker_bytes and reaches the guard under test.
    real_model = json.loads((worker_dir / "model_config.json").read_bytes())

    def _canon_model(mapping: dict[str, Any]) -> bytes:
        return json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def calib_neg(tag: str, substring: str, *, model: bytes | None = None,
                  topology: bytes | None = None, cue: bytes | None = None,
                  completion: int = COMPLETION_STEPS) -> None:
        forged = workspace / f"calibneg_{tag}"
        if forged.exists():
            shutil.rmtree(forged)
        forged.mkdir(parents=True)
        for name in ("weights.bin", "topology.bin", "model_config.json", "cue_000.bin"):
            shutil.copy(worker_dir / name, forged / name)
        if model is not None:
            (forged / "model_config.json").write_bytes(model)
        if topology is not None:
            (forged / "topology.bin").write_bytes(topology)
        if cue is not None:
            (forged / "cue_000.bin").write_bytes(cue)
        _run_cli_fail(_calibrate_argv(forged, forged / "cue_000.bin", "sha256:" + "0" * 64,
                                      extension_sha256, crate_version, completion_steps=completion),
                      environment, substring)

    calib_neg("completion", "worker completion window must be at least 32 steps and divisible by eight",
              completion=24)
    calib_neg("mc_nan", "worker model-config carries a non-finite JSON constant",
              model=b'{"_nan":NaN,' + _canon_model(real_model)[1:])
    calib_neg("mc_dupkey", "worker model-config contains a duplicate key",
              model=_canon_model(real_model)[:-1] + b',"a_minus":0.006}')
    calib_neg("mc_notjson", "worker model-config is not strict UTF-8 JSON", model=b"{not json")
    calib_neg("mc_nonobj", "worker model-config must be a JSON object", model=b"[]")
    calib_neg("mc_noncanon", "worker model-config is not canonical JSON",
              model=json.dumps(real_model, indent=2).encode("utf-8"))
    calib_neg("mc_badfield", "invalid model configuration",
              model=_canon_model({**real_model, "unexpected_field": 1}))
    calib_neg("mc_partial", "worker model config is not the full canonical configuration",
              model=_canon_model({k: v for k, v in real_model.items() if k != "connectivity"}))
    topo = bytearray((worker_dir / "topology.bin").read_bytes())
    topo[0] = 2
    calib_neg("topo_byte", "worker topology carries a byte outside", topology=bytes(topo))
    cue_nan = np.full(FIXTURE_MODEL.n_neurons, 0.5, dtype="<f8")
    cue_nan[0] = np.nan
    calib_neg("cue_nan", "calibration cue current carries a non-finite value", cue=cue_nan.tobytes())

    # D1/D2 cross-binding witnesses via consistent artifact reseal. The source-universe and cue-set
    # readers are purely payload-internal, so a reseal that keeps the payload self-consistent (target
    # field + its derived twins, self digest recomputed, D2 relinked to the new D1) is reader-valid
    # while it disagrees with the live git repo / the paired artifact, reaching materialize's
    # live-surface re-checks. This preserves the production serialization/authentication/read-back path.
    from snn_memory import cue_materializer as _cue
    from snn_memory import source_universe as _su
    d1_src = Path(base["d1"])
    d2_src = Path(base["cue_set"])
    d2_dir = d2_src.parent

    # Focused pin for the _ENCODE_MIN_EVENTS invariant (verifier Class B on line 215): every selected D1
    # record declares event_count >= MIN_EVENTS, and materialize re-verifies the actual split_events count
    # equals that declaration before _encode_events, so the no-events guard cannot be reached.
    _selected_records = json.loads(d1_src.read_bytes())["selected"]
    assert _selected_records and all(int(record["event_count"]) >= _su.MIN_EVENTS
                                     for record in _selected_records), \
        "_ENCODE_MIN_EVENTS invariant broken: a selected D1 record declares fewer than MIN_EVENTS events"

    def d1_twin(payload: dict[str, Any], field: str, value: Any) -> None:
        target = payload["selected"][0]["normalized_path"]
        for item in payload["selected"] + payload["considered"]:
            if item.get("normalized_path") == target:
                item[field] = value

    def prep_neg(tag: str, substring: str, *,
                 mut_d1: Callable[[dict[str, Any]], None] | None = None,
                 mut_d2: Callable[[dict[str, Any], Path], None] | None = None,
                 cfg_over: Callable[[dict[str, Any]], None] | None = None) -> None:
        forged = workspace / f"prepneg_{tag}"
        if forged.exists():
            shutil.rmtree(forged)
        forged.mkdir(parents=True)
        p1 = json.loads(d1_src.read_bytes())
        if mut_d1 is not None:
            mut_d1(p1)
        p1["self_sha256"] = _su._self_digest(p1)
        raw1 = _su._canonical(p1)
        (forged / "universe.json").write_bytes(raw1)
        d1_fsha = _sha(raw1)
        forged_d2 = forged / "cue-set"
        shutil.copytree(d2_dir, forged_d2)
        cue_json = forged_d2 / d2_src.name
        p2 = json.loads(cue_json.read_bytes())
        p2["source_universe"]["file_sha256"] = d1_fsha
        p2["source_universe"]["payload_self_sha256"] = p1["self_sha256"]
        p2["source_universe"]["repository_head"] = p1["repository"]["head"]
        p2["source_universe"]["selected_record_ids"] = p1["selected_record_ids"]
        if mut_d2 is not None:
            mut_d2(p2, forged_d2)
        p2["self_sha256"] = _cue._self_digest(p2, _cue._CUE_SET_SELF_DOMAIN)
        raw2 = _cue._canonical(p2)
        cue_json.write_bytes(raw2)

        def mutate(config: dict[str, Any]) -> None:
            config["d1"] = str(forged / "universe.json")
            config["d1_file_sha256"] = d1_fsha
            config["cue_set"] = str(cue_json)
            config["d2_file_sha256"] = _sha(raw2)
            if cfg_over is not None:
                cfg_over(config)
        run_neg(mutate, f"prep_{tag}", substring)
        shutil.rmtree(forged, ignore_errors=True)

    prep_neg("xbind", "cue set does not cross-bind the exact D1 artifact and HEAD",
             mut_d2=lambda p, dd: p["source_universe"].__setitem__("file_sha256", "0" * 64))
    prep_neg("encdir", "D2 encoder directory digest differs from the pinned encoder digest",
             mut_d2=lambda p, dd: p["encoder"].__setitem__("directory_sha256", "0" * 64))
    prep_neg("normdig", "calibration cue normalized digest for",
             mut_d2=lambda p, dd: p["records"][0]["calibration_cue"].__setitem__(
                 "normalized_text_sha256", "0" * 64))
    prep_neg("cuefile", "calibration cue file for",
             mut_d2=lambda p, dd: (dd / p["records"][0]["calibration_cue"]["path"]).write_bytes(
                 b"tampered\n"))
    prep_neg("bloboid", "resolves to a different blob at HEAD",
             mut_d1=lambda p: d1_twin(p, "blob_oid", "0" * 40))
    prep_neg("bytecount", "differs from its content binding",
             mut_d1=lambda p: d1_twin(p, "byte_count", int(p["selected"][0]["byte_count"]) + 1))
    prep_neg("evhash", "event hashes differ from its binding",
             mut_d1=lambda p: d1_twin(p, "event_order_digest", "0" * 64))

    def mut_evcount(payload: dict[str, Any]) -> None:
        target = payload["selected"][0]["normalized_path"]
        for item in payload["selected"] + payload["considered"]:
            if item.get("normalized_path") == target and "event_sha256" in item:
                item["event_count"] = int(item["event_count"]) + 1
                item["event_sha256"] = list(item["event_sha256"]) + ["0" * 64]
    prep_neg("evcount", "event count differs from its binding", mut_d1=mut_evcount)

    def mut_livehead(payload: dict[str, Any]) -> None:
        fake = "1" * len(str(payload["repository"]["head"]))
        payload["repository"]["head"] = fake
        for item in payload["considered"] + payload["selected"]:
            if "content_commit" in item:
                item["content_commit"] = fake
    prep_neg("livehead", "live repository HEAD differs from the authenticated D1 HEAD",
             mut_d1=mut_livehead,
             cfg_over=lambda c: c.__setitem__("repository_head", "1" * 40))

    # A selected record whose path is absent at HEAD: rename it, re-sort the considered/eligible/selected
    # order and re-align the D2 records so both artifacts stay reader-valid; the git blob capture then
    # fails at the authenticated live repository.
    def mut_gitpath(payload: dict[str, Any]) -> None:
        old_path = payload["selected"][0]["normalized_path"]
        fake = "docs/public/nonexistent-d4a-record.md"
        for item in payload["considered"]:
            if item.get("normalized_path") == old_path:
                item["path"] = fake
                item["normalized_path"] = fake
                if "selection_key" in item:
                    item["selection_key"] = _su._selection_key(fake, item["content_sha256"])
        payload["considered"].sort(key=lambda item: item["normalized_path"])
        eligible = [item for item in payload["considered"] if item.get("status") == "eligible"]
        eligible.sort(key=lambda item: item["selection_key"])
        payload["eligible_record_ids"] = [item["record_id"] for item in eligible]
        chosen = eligible[:len(payload["selected"])]
        payload["selected"] = [dict(item, rank=index) for index, item in enumerate(chosen)]
        payload["selected_record_ids"] = [item["record_id"] for item in chosen]
        payload["selected_paths"] = [item["normalized_path"] for item in chosen]

    def realign_d2_records(p2: dict[str, Any], _forged_d2: Path) -> None:
        by_id = {record["record_id"]: record for record in p2["records"]}
        p2["records"] = [by_id[rid] for rid in p2["source_universe"]["selected_record_ids"] if rid in by_id]
    prep_neg("gitpath", "nonexistent-d4a-record.md failed",
             mut_d1=mut_gitpath, mut_d2=realign_d2_records)

    # A calibration cue whose self-consistent block re-derives to a different digest than the D1 events.
    def reseal_calibration(p2: dict[str, Any], forged_d2: Path) -> None:
        cal = p2["records"][0]["calibration_cue"]
        text = "\n".join(f"zzz-tampered-line-{index}" for index in range(len(cal["event_indices"])))
        digest = _sha(text.encode("utf-8"))
        cue_id = _cue._cue_id(digest)
        cal["sha256"] = digest
        cal["cue_id"] = cue_id
        cal["path"] = f"cues/{cue_id}.txt"
        cal["normalized_text_sha256"] = _cue._normalized_sha256(text)
        (forged_d2 / cal["path"]).write_bytes(text.encode("utf-8"))
    prep_neg("calblock", "re-derived calibration block for", mut_d2=reseal_calibration)

    # A schema-valid cue path ('cues/cue-<hex>.txt', two components) whose intermediate 'cues' directory
    # is absent: the calibration-cue read walks the intermediate component with O_DIRECTORY|O_NOFOLLOW
    # and fails at the authenticated directory-relative open.
    prep_neg("cuepath", "traverses unsafely",
             mut_d2=lambda p, dd: shutil.rmtree(dd / "cues"))

    # A selected D1 record pointing at the committed non-UTF8 binary blob: validate_source_universe_bytes
    # authenticates only the D1 JSON (never opens the blob), so a consistent reseal that selects the binary
    # blob is reader-valid, and materialize's live-blob decode (lines 512-513) fails at the authenticated
    # git blob. record_id/content_sha256/selection_key are content-derived; the event fields stay as the
    # original record's (materialize's UTF-8 decode fires before the event checks).
    # Copy the fixture repo; brute-force a non-UTF8 .md content (in [MIN_BYTES, MAX_BYTES]) whose sha256
    # matches BOTH the replaced record's calibration_block_index and family_permutation_index so its D2
    # calibration is verbatim-valid; commit it as a .md in a post-write_source_universe amend (new HEAD).
    _bin_repo = workspace / "binaryblob_repo"
    if _bin_repo.exists():
        shutil.rmtree(_bin_repo)
    shutil.copytree(Path(base["repo"]), _bin_repo)
    _orig_rid0 = json.loads(d1_src.read_bytes())["selected"][0]["record_id"]
    _orig_d2 = next(r for r in json.loads(d2_src.read_bytes())["records"] if r["record_id"] == _orig_rid0)
    _tgt_block = _orig_d2["calibration_block_index"]
    _tgt_perm = _orig_d2["family_permutation_index"]
    _bin_bytes = b""
    for _n in range(500000):
        _cand = b"\xff\xfe\x00\x80 not valid utf-8 line \xff\xfe " + str(_n).encode() + b"\n"
        _cand = _cand * (1200 // len(_cand) + 2)
        _csha = _sha(_cand)
        if (_cue._calibration_block_index(_csha) == _tgt_block
                and _cue._family_permutation_index(_csha) == _tgt_perm):
            _bin_bytes = _cand
            break
    assert _bin_bytes and len(_bin_bytes) <= 20000, "no block+permutation-matching non-UTF8 content found"
    _bin_path = "docs/public/record-99.md"
    (_bin_repo / _bin_path).write_bytes(_bin_bytes)
    _git(_bin_repo, "add", "-A")
    _commit_env = dict(os.environ, GIT_AUTHOR_DATE="@1700000000 +0000", GIT_COMMITTER_DATE="@1700000000 +0000")
    subprocess.run(["git", "-C", str(_bin_repo), "commit", "-q", "--amend", "--no-edit"],
                   env=_commit_env, check=True)
    _bin_head = _git(_bin_repo, "rev-parse", "HEAD").decode("ascii").strip()
    _bin_oid = _git(_bin_repo, "rev-parse", f"HEAD:{_bin_path}").decode("ascii").strip()
    _bin_sha = _sha(_bin_bytes)
    _bin_rid = f"sha256:{_bin_sha}"

    def mut_binary(payload: dict[str, Any]) -> None:
        payload["repository"]["head"] = _bin_head
        for item in payload["considered"] + payload["selected"]:
            if "content_commit" in item:
                item["content_commit"] = _bin_head
            if "latest_path_commit" in item:
                item["latest_path_commit"] = _bin_head
        target = payload["selected"][0]["normalized_path"]
        for item in payload["considered"]:
            if item.get("normalized_path") == target:
                item["path"] = _bin_path
                item["normalized_path"] = _bin_path
                item["content_sha256"] = _bin_sha
                item["record_id"] = _bin_rid
                item["blob_oid"] = _bin_oid
                item["byte_count"] = len(_bin_bytes)
                if "selection_key" in item:
                    item["selection_key"] = _su._selection_key(_bin_path, _bin_sha)
        payload["considered"].sort(key=lambda item: item["normalized_path"])
        eligible = [item for item in payload["considered"] if item.get("status") == "eligible"]
        eligible.sort(key=lambda item: item["selection_key"])
        payload["eligible_record_ids"] = [item["record_id"] for item in eligible]
        chosen = eligible[:len(payload["selected"])]
        payload["selected"] = [dict(item, rank=index) for index, item in enumerate(chosen)]
        payload["selected_record_ids"] = [item["record_id"] for item in chosen]
        payload["selected_paths"] = [item["normalized_path"] for item in chosen]

    def mut_binary_d2(p2: dict[str, Any], _forged_d2: Path) -> None:
        by_id = {record["record_id"]: record for record in p2["records"]}
        if _orig_rid0 in by_id:  # block+perm already match by search -> repurpose VERBATIM, only record_id
            by_id[_bin_rid] = dict(by_id[_orig_rid0], record_id=_bin_rid)
        p2["records"] = [by_id[rid] for rid in p2["source_universe"]["selected_record_ids"] if rid in by_id]
    prep_neg("binaryblob", "is not strict UTF-8", mut_d1=mut_binary, mut_d2=mut_binary_d2,
             cfg_over=lambda c: (c.__setitem__("repo", str(_bin_repo)),
                                 c.__setitem__("repository_head", _bin_head)) and None)

    # renameat2 EEXIST: the output parent is caller-supplied and _cmd_materialize separates the
    # final-name absence check (os.stat) from the RENAME_NOREPLACE install; _produce_output runs in
    # that window. A concurrent creator that fills the final name after the absence check makes the
    # renameat2 EEXIST branch reachable. Witness it with a watcher thread that creates the final name
    # the instant the staging directory appears — after the absence check, well before the rename.
    race_parent = workspace / "renameat2_race"
    race_parent.mkdir(exist_ok=True)
    final_output = race_parent / "materialize_out"
    stop_watcher = threading.Event()

    def _fill_final_when_staging_appears() -> None:
        while not stop_watcher.is_set():
            if any(race_parent.glob(f".{final_output.name}.staging-*")):
                try:
                    final_output.mkdir()
                except FileExistsError:
                    pass
                return
            time.sleep(0.002)

    watcher = threading.Thread(target=_fill_final_when_staging_appears, daemon=True)
    watcher.start()
    try:
        run_neg(lambda c: c.__setitem__("output", str(final_output)), "renameat2race",
                "materialization output already exists")
    finally:
        stop_watcher.set()
        watcher.join(timeout=5)

    # renameat2 NON-EEXIST errno (EACCES): remove write permission from the caller-supplied output parent
    # once staging exists, so RENAME_NOREPLACE fails with a non-EEXIST error the OS permits. The module's
    # ignore_errors cleanup cannot unlink through the now-read-only parent, leaving an empty staging
    # residue; the gate restores permission, removes the residue, and asserts the nested parent is clean.
    eacces_parent = workspace / "renameat2_eacces"
    eacces_parent.mkdir(exist_ok=True)
    eacces_final = eacces_parent / "materialize_out"
    stop_revoke = threading.Event()

    def _revoke_write_when_staging_appears() -> None:
        while not stop_revoke.is_set():
            if any(eacces_parent.glob(f".{eacces_final.name}.staging-*")):
                os.chmod(eacces_parent, 0o555)
                return
            time.sleep(0.002)

    revoker = threading.Thread(target=_revoke_write_when_staging_appears, daemon=True)
    revoker.start()
    try:
        run_neg(lambda c: c.__setitem__("output", str(eacces_final)), "renameat2eacces",
                "atomic output install failed")
    finally:
        stop_revoke.set()
        revoker.join(timeout=5)
    # Precise EACCES contract (production is NOT residue-free): the fail-closed run leaves NO accepted
    # final target, but the module's best-effort shutil.rmtree(ignore_errors=True) cannot unlink through
    # the revoked-write parent, so an inert staging residue remains. Assert both BEFORE teardown, then
    # restore the parent mode and remove the residue as FIXTURE cleanup only.
    residue = list(eacces_parent.glob(f".{eacces_final.name}.staging-*"))
    assert residue, "EACCES witness expected an inert staging residue in the caller-supplied output parent"
    assert not eacces_final.exists(), "EACCES witness must leave no accepted final output target"
    os.chmod(eacces_parent, 0o755)
    for staging_dir in residue:
        shutil.rmtree(staging_dir, ignore_errors=True)

    nonreg_dir = workspace / "nonreg_d1_dir"
    nonreg_dir.mkdir(exist_ok=True)
    run_neg(lambda c: c.__setitem__("d1", str(nonreg_dir)), "nonreg283",
            "D1 source universe is not a regular file")

    # Control-derivation witnesses: reseal the trained checkpoint bundle (manifest + descriptor digests
    # recomputed so read_checkpoint_bundle_v2 accepts it) so a single field disagrees with the control
    # configuration, then run a control condition. Preserves the production checkpoint read-back path.
    def control_neg(tag: str, substring: str, *,
                    mut_manifest: Callable[[dict[str, Any]], None] | None = None,
                    mut_desc_ids: Callable[[dict[str, Any]], None] | None = None) -> None:
        forged = workspace / f"controlneg_{tag}"
        if forged.exists():
            shutil.rmtree(forged)
        shutil.copytree(trained_bundle, forged)
        manifest = json.loads((forged / "manifest.json").read_bytes())
        if mut_manifest is not None:
            mut_manifest(manifest)
        manifest_bytes = cbv._canonical_bytes(manifest)
        (forged / "manifest.json").write_bytes(manifest_bytes)
        descriptor = json.loads((forged / "descriptor.json").read_bytes())
        if mut_desc_ids is not None:
            mut_desc_ids(descriptor["identities"])
        descriptor["checkpoint_manifest"]["file_sha256"] = _sha(manifest_bytes)
        descriptor["checkpoint_manifest"]["payload_self_sha256"] = cbv._manifest_payload_digest(manifest_bytes)
        descriptor.pop("self_sha256", None)
        descriptor["self_sha256"] = cbv._self_digest(descriptor, cbv._DESCRIPTOR_DOMAIN)
        (forged / "descriptor.json").write_bytes(cbv._canonical_bytes(descriptor))

        def mutate(config: dict[str, Any]) -> None:
            config["condition"] = "shuffled"
            config["trained_bundle"] = str(forged)
        run_neg(mutate, f"control_{tag}", substring)
        shutil.rmtree(forged, ignore_errors=True)

    def set_experiment(mapping: dict[str, Any]) -> None:
        mapping["experiment_digest"] = "0" * 64
    control_neg("identity", "trained bundle experiment_digest differs from the control configuration",
                mut_manifest=set_experiment, mut_desc_ids=set_experiment)

    def reorder_records(manifest: dict[str, Any]) -> None:
        order = list(manifest["ordered_record_ids"])
        manifest["ordered_record_ids"] = order[1:] + order[:1]
    control_neg("order", "trained bundle record order differs from the control candidate order",
                mut_manifest=reorder_records)

    # A materialization output whose parent is a symlink is refused at the root-anchored walk.
    linked_parent = workspace / "linked_out"
    if linked_parent.is_symlink() or linked_parent.exists():
        linked_parent.unlink() if linked_parent.is_symlink() else shutil.rmtree(linked_parent)
    linked_parent.symlink_to(workspace)
    run_neg(lambda c: c.__setitem__("output", str(linked_parent / "sym_out")), "symparent",
            "cannot be opened safely")
    # No accepted partial output remains after a fail-closed materialization.
    assert not (workspace / "sym_out").exists(), "a failed materialization left an accepted output"
    leftover = list(workspace.glob(".*.staging-*"))
    assert not leftover, f"a failed materialization left staging directories: {leftover}"


def _check_bind_negatives(workspace: Path, results: dict[int, dict[str, Any]],
                          environment: dict[str, str]) -> None:
    seed_a, seed_b = SEEDS
    checkpoint_a = Path(results[seed_a]["trained"]["checkpoint_dir"])
    bank_a = Path(results[seed_a]["trained"]["bank_dir"])
    bank_b = Path(results[seed_b]["trained"]["bank_dir"])
    scoring_a = Path(results[seed_a]["trained"]["output"]) / "scoring_target.json"
    scoring_b = Path(results[seed_b]["trained"]["output"]) / "scoring_target.json"
    # A bank from a different seed does not bind the presented checkpoint bundle.
    _run_cli_fail(["bind", "--checkpoint", str(checkpoint_a), "--bank", str(bank_b),
                   "--scoring-target", str(scoring_a)], environment,
                  "candidate bank does not bind the presented checkpoint bundle")
    # A scoring target from a different seed does not match the seed-A bank binding.
    _run_cli_fail(["bind", "--checkpoint", str(checkpoint_a), "--bank", str(bank_a),
                   "--scoring-target", str(scoring_b)], environment,
                  "scoring-target digest differs from the presented target")

    # A resealed bank whose state disagrees with the checkpoint descriptor state is rejected.
    def flip_state(_dst: Path, manifest: dict[str, Any]) -> None:
        manifest["state"] = "pre_result"

    resealed_bank = _reseal_bank(bank_a, workspace / "bind_state_bank", flip_state)
    _run_cli_fail(["bind", "--checkpoint", str(checkpoint_a), "--bank", str(resealed_bank),
                   "--scoring-target", str(scoring_a)], environment,
                  "candidate bank state differs from the checkpoint descriptor state")

    # Bank↔checkpoint reconciliation guards reachable through the bind CLI: one resealed field
    # at a time, keeping the checkpoint-binding block and record order intact so control reaches
    # the target _reconcile_bank_checkpoint guard. The candidate_bank_v2 reader masks the order,
    # completion, encoder-directory/config, candidate-set-recompute and signature guards (class A
    # in the verifier); model-config and input-current are not reader-bound, so they are witnessed
    # here by mutating EVERY calibration entry (the reader pins entry-agreement first).
    def bind_neg(tag: str, mutate: Callable[[Path, dict[str, Any]], None], substring: str) -> None:
        resealed = _reseal_bank(bank_a, workspace / f"bind_{tag}", mutate)
        _run_cli_fail(["bind", "--checkpoint", str(checkpoint_a), "--bank", str(resealed),
                       "--scoring-target", str(scoring_a)], environment, substring)

    def all_calibration(field: str, value: Any) -> Callable[[Path, dict[str, Any]], None]:
        def mutate(_dst: Path, manifest: dict[str, Any]) -> None:
            for entry in manifest["calibration"]:
                entry[field] = value
        return mutate

    bind_neg("exp", lambda _d, m: m["identities"].__setitem__("experiment_digest", "0" * 64),
             "candidate bank experiment digest differs from the checkpoint")
    bind_neg("calcue", lambda _d, m: m["calibration"][0].__setitem__("source_cue_sha256", "0" * 64),
             "calibration source cue digest for")
    bind_neg("calmodel", all_calibration("model_config_digest", "0" * 64),
             "calibration model config for")
    bind_neg("calcurrent", all_calibration("input_current", 0.123456789),
             "calibration input current for")

    # Scoring-target vs bank guards (bind, 1237-1245): reseal the scoring target (mutate one field,
    # recompute scorer_digest + self via experiment_lock) and repoint the bank scoring block at the new
    # scoring self so control passes the self-digest bind (1233) and reaches the comparison. The
    # scoring-target semantic validator + schema pin candidate-bank-digest, bins, signature_dtype and
    # the abstention rule (class B), so only order/signatures/completion are witnessed here.
    from snn_memory import experiment_lock as _el
    scoring_domain = _el._ARTIFACT_DOMAINS["snn-memory-scoring-target-v2"]

    def scoring_neg(tag: str, substring: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        forged = workspace / f"scoring_{tag}"
        if forged.exists():
            shutil.rmtree(forged)
        forged.mkdir(parents=True)
        payload = json.loads(scoring_a.read_bytes())
        mutate(payload)
        payload["scorer_digest"] = _el.scorer_identity_digest(payload)
        payload["self_sha256"] = _el._self_digest(payload, scoring_domain)
        forged_scoring = forged / "scoring_target.json"
        forged_scoring.write_bytes(_el._canonical(payload))
        new_self = payload["self_sha256"]

        def repoint(_dst: Path, manifest: dict[str, Any]) -> None:
            manifest["scoring"]["scoring_target_self_sha256"] = new_self
        forged_bank = _reseal_bank(bank_a, forged / "bank", repoint)
        _run_cli_fail(["bind", "--checkpoint", str(checkpoint_a), "--bank", str(forged_bank),
                       "--scoring-target", str(forged_scoring)], environment, substring)
        shutil.rmtree(forged, ignore_errors=True)

    def reorder_candidates(payload: dict[str, Any]) -> None:
        order = list(payload["candidate_order"])
        order[-1] = "sha256:" + "f" * 64
        payload["candidate_order"] = sorted(order)
        payload["identities"]["candidate_bank_digest"] = cbank.candidate_bank_digest(
            list(payload["candidate_order"]), list(payload["candidate_signature_digests"]))

    def mutate_signature(payload: dict[str, Any]) -> None:
        signatures = list(payload["candidate_signature_digests"])
        signatures[0] = "0" * 64
        payload["candidate_signature_digests"] = signatures
        payload["identities"]["candidate_bank_digest"] = cbank.candidate_bank_digest(
            list(payload["candidate_order"]), signatures)

    scoring_neg("order", "scoring-target candidate order differs from the lexical bank order",
                reorder_candidates)
    scoring_neg("signatures", "scoring-target signature digests differ from the bank rows",
                mutate_signature)
    scoring_neg("completion", "scoring-target completion window differs from the bank layout",
                lambda p: p.__setitem__("completion_steps", int(p["completion_steps"]) + 8))


def _prove_installed_origins(arguments: argparse.Namespace) -> None:
    """Prove the three modules, schemas, and extension load from the exact install prefix."""
    install = str(Path(arguments.install_target).resolve())
    root = Path(arguments.repo_root).resolve()
    for name in ("checkpoint_bundle_v2", "candidate_bank_v2", "checkpoint_materialize_v2"):
        module = importlib.import_module(f"snn_memory.{name}")
        origin = str(Path(module.__file__).resolve())
        assert origin.startswith(install + "/"), f"{name} origin {origin} not under install prefix"
        assert _sha(Path(origin).read_bytes()) == _sha((root / "snn_memory" / f"{name}.py").read_bytes()), (
            f"{name} installed bytes differ from the authorized source SHA")
    for base in ("snn_memory_checkpoint_bundle_v2.schema.json", "snn_memory_candidate_bank_v2.schema.json"):
        packaged = files("snn_memory").joinpath("schema", base).read_bytes()
        assert packaged == (root / "docs/schema" / base).read_bytes(), f"{base} packaged != docs/schema"
        assert files("snn_memory").joinpath("schema", base + ".license").read_bytes(), f"{base} licence missing"
    import rust_snn_memory.rust_snn_memory as extension
    ext_origin = str(Path(extension.__file__).resolve())
    assert ext_origin.startswith(install + "/"), f"extension origin {ext_origin} not under install prefix"
    assert _sha(Path(ext_origin).read_bytes()) == arguments.extension_sha256, "extension digest drift"
    assert extension.STREAMED_API_VERSION == 2, extension.STREAMED_API_VERSION


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
    arguments = parser.parse_args()
    workspace = arguments.workspace
    workspace.mkdir(parents=True, exist_ok=True)

    environment = _gate_environment()
    fixture = _acquire_d1_d2(workspace, arguments.encoder_checkpoint, arguments.encoder_digest, environment)
    base = _base_config(arguments, fixture)
    _prove_installed_origins(arguments)

    results: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        per_seed: dict[str, Any] = {}
        trained = _materialize(base, workspace, seed, "trained", environment)
        per_seed["trained"] = trained
        trained_checkpoint = Path(trained["checkpoint_dir"])
        for condition in ("shuffled", "random", "zero"):
            per_seed[condition] = _materialize(
                base, workspace, seed, condition, environment, trained_bundle=trained_checkpoint)
        per_seed["untrained"] = _materialize(base, workspace, seed, "untrained", environment)
        results[seed] = per_seed
        for condition in CONDITIONS:
            cbv.read_checkpoint_bundle_v2(Path(per_seed[condition]["checkpoint_dir"]))
            cbank.read_candidate_bank_v2(Path(per_seed[condition]["bank_dir"]))

    # Per-seed and per-condition isolation: every checkpoint descriptor and bank digest is distinct.
    descriptors = [results[s][c]["checkpoint_descriptor_self_sha256"] for s in SEEDS for c in CONDITIONS]
    banks = [results[s][c]["bank_self_sha256"] for s in SEEDS for c in CONDITIONS]
    assert len(set(descriptors)) == len(descriptors), "checkpoint descriptors are not per-(seed,condition) unique"
    assert len(set(banks)) == len(banks), "candidate banks are not per-(seed,condition) unique"
    assert results[SEEDS[0]]["untrained"]["epochs_completed"] == 0
    assert results[SEEDS[0]]["trained"]["epochs_completed"] == 1

    # Matched controls retain the trained schedule's epochs_completed.
    for seed in SEEDS:
        for condition in ("shuffled", "random", "zero"):
            assert results[seed][condition]["epochs_completed"] == results[seed]["trained"]["epochs_completed"]

    # Bind the authenticated checkpoint, bank, and scoring target through the real CLI subcommand.
    seed = SEEDS[0]
    checkpoint_dir = Path(results[seed]["trained"]["checkpoint_dir"])
    bank_dir = Path(results[seed]["trained"]["bank_dir"])
    scoring_target = workspace / f"out_{seed}_trained" / "scoring_target.json"
    bind_report = _run_cli(
        ["bind", "--checkpoint", str(checkpoint_dir), "--bank", str(bank_dir),
         "--scoring-target", str(scoring_target)], environment)
    assert bind_report["bound"] is True
    assert _run_cli(["read-checkpoint", "--bundle", str(checkpoint_dir)], environment)
    assert _run_cli(["read-bank", "--bank", str(bank_dir)], environment)

    # Enumerate every materializer PID and every worker/calibration child PID across all runs.
    materialize_pids: list[int] = []
    worker_pids: list[int] = []
    for per_seed in results.values():
        for condition_report in per_seed.values():
            materialize_pids.append(int(condition_report["materialize_pid"]))
            worker_pids.extend(int(pid) for pid in condition_report["worker_pids"])

    # Real calibration-worker CLI coverage from an authenticated worker directory.
    worker_dir = workspace / f"out_{SEEDS[0]}_trained" / "workers"
    calibrate_report = _run_cli(_calibrate_argv(
        worker_dir, worker_dir / "cue_000.bin", results[SEEDS[0]]["trained"]["ordered_record_ids"][0],
        arguments.extension_sha256, str(base["crate_version"])), environment)
    assert calibrate_report["shape"] == [COMPLETION_STEPS, FIXTURE_MODEL.n_neurons]
    base64.b64decode(calibrate_report["raster_b64"], validate=True)
    worker_pids.append(int(calibrate_report["pid"]))
    _run_cli_fail(["materialize", "--config", str(workspace / "does-not-exist.json")], environment,
                  "cannot be opened safely")

    # Byte-determinism replay of one checkpoint in a fresh output through fresh child processes.
    replay = _materialize(base, workspace, SEEDS[0], "trained", environment, tag="_replay")
    materialize_pids.append(int(replay["materialize_pid"]))
    worker_pids.extend(int(pid) for pid in replay["worker_pids"])
    assert replay["checkpoint_descriptor_self_sha256"] == results[SEEDS[0]]["trained"][
        "checkpoint_descriptor_self_sha256"], "byte-determinism replay produced a different checkpoint"
    assert replay["bank_self_sha256"] == results[SEEDS[0]]["trained"]["bank_self_sha256"]

    _adversarial_signature_witness()
    untrained_bundle = Path(results[SEEDS[0]]["untrained"]["checkpoint_dir"])
    _check_bundle_reader_negatives(checkpoint_dir, workspace / "scratch")
    _check_checkpoint_negatives(checkpoint_dir, workspace / "scratch")
    _check_resealed_forgeries(checkpoint_dir, workspace / "scratch", untrained_bundle)
    _check_bank_negatives(bank_dir, workspace / "scratch")
    _check_signature_negatives(bank_dir, workspace / "scratch")
    _check_resealed_bank_forgeries(bank_dir, workspace / "scratch")
    _check_materialize_negatives(base, workspace, checkpoint_dir, untrained_bundle, worker_dir,
                                 arguments.extension_sha256, str(base["crate_version"]), environment)
    _check_bind_negatives(workspace, results, environment)
    _witness_root_open_emfile()

    print(json.dumps({
        "status": "pass",
        "gate_pid": os.getpid(),
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "n_neurons": FIXTURE_MODEL.n_neurons,
        "candidate_count": len(fixture["ids"]),
        "materializer_pids": sorted(set(materialize_pids)),
        "worker_pids": sorted(set(worker_pids)),
        "materializer_run_count": len(materialize_pids),
        "worker_child_count": len(worker_pids),
        "checkpoint_descriptors": {f"{s}_{c}": results[s][c]["checkpoint_descriptor_self_sha256"]
                                   for s in SEEDS for c in CONDITIONS},
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
