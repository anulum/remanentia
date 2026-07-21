# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN schema-v2 contract tests

"""Real-file tests for the held-out temporal-memory artifact boundary."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pytest

from snn_memory.v2_contracts import (
    ValidatedArtifactV2,
    ValidatedDatasetV2,
    canonical_json_digest,
    normalize_cue_text,
    ordered_record_ids_digest,
    read_and_validate_checkpoint_v2,
    read_and_validate_dataset_v2,
    read_and_validate_evaluation_v2,
    read_and_validate_task_output_v2,
    task_set_digest,
    token_jaccard,
)


_SEEDS = [11, 29, 47, 71, 101, 131, 167, 211, 257, 307]
_FAMILIES = ["truncation", "deletion", "masking", "sparse_noise"]
_TASK_IDS = [f"task:{sha256(f'task-{index}'.encode()).hexdigest()}" for index in range(3)]


def _digest(raw_bytes: bytes) -> str:
    return sha256(raw_bytes).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_bytes(root: Path, relative: str, raw_bytes: bytes) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    return {"path": relative, "sha256": _digest(raw_bytes)}


def _opaque_id(prefix: str, label: str) -> str:
    return f"{prefix}-{sha256(label.encode()).hexdigest()[:32]}"


def _normalized_digest(text: str) -> str:
    return _digest(normalize_cue_text(text).encode())


def _write_manifest(path: Path, value: object) -> str:
    raw_bytes = _json_bytes(value)
    path.write_bytes(raw_bytes)
    return _digest(raw_bytes)


def _dataset_fixture(root: Path) -> tuple[Path, dict[str, Any], str]:
    artifacts: dict[str, dict[str, Any]] = {
        name: _write_bytes(root, f"meta/{name}.jsonl", f"{name}-bytes\n".encode())
        for name in (
            "provenance_store",
            "temporal_edges",
            "noise_lexicon",
            "development_calibration",
        )
    }
    canonical_tasks = "remanentia:snn-memory:task-set:v2\n" + "".join(
        f"{task_id}\n" for task_id in sorted(_TASK_IDS)
    )
    artifacts["task_set"] = _write_bytes(root, "meta/task_set.txt", canonical_tasks.encode("ascii"))
    schedules = {
        "capacity_prefixes": [4, 8, 12, 16],
        **{
            name: _write_bytes(root, f"schedules/{name}.json", f"{name}-schedule\n".encode())
            for name in (
                "interference",
                "forgetting",
                "leave_one_record_out",
                "temporal_order_permutation",
            )
        },
    }
    source_universe_artifact = _write_bytes(root, "source/universe.jsonl", b"source-universe\n")
    distractor = _write_bytes(root, "distractors/distractor.txt", b"distractor\n")

    records: list[dict[str, Any]] = []
    for record_index in range(16):
        training_text = f"training-payload-{record_index:02d}\n".encode().ljust(1000, b"x")
        training = _write_bytes(root, f"training/source-{record_index:02d}.txt", training_text)
        training["event_indices"] = list(range(50))
        record_id = f"sha256:{training['sha256']}"

        calibration_id = _opaque_id("cue", f"calibration-{record_index}")
        calibration_text = f"calibrationtoken{record_index:02d}"
        calibration = _write_bytes(root, f"cues/{calibration_id}.txt", calibration_text.encode())
        calibration.update(
            {
                "cue_id": calibration_id,
                "normalized_text_sha256": _normalized_digest(calibration_text),
                "event_indices": [0],
            }
        )

        base_cues: list[dict[str, Any]] = []
        for family_index, family in enumerate(_FAMILIES):
            cue_id = _opaque_id("cue", f"base-{record_index}-{family}")
            variants: list[dict[str, Any]] = []
            for requested in (0, 10, 25, 40):
                variant_id = _opaque_id("variant", f"variant-{record_index}-{family}-{requested}")
                text = f"token{record_index:02d}{family_index}{requested:02d}"
                variant = _write_bytes(root, f"variants/{variant_id}.txt", text.encode())
                variant.update(
                    {
                        "variant_id": variant_id,
                        "normalized_text_sha256": _normalized_digest(text),
                        "requested_percent": requested,
                        "realized_fraction": requested / 100,
                        "tokenizer_digest": _digest(
                            f"tokenizer-{record_index}-{family}-{requested}".encode()
                        ),
                        "selected_positions": [] if requested == 0 else [family_index],
                    }
                )
                variants.append(variant)
            base_cue: dict[str, Any] = {
                "cue_id": cue_id,
                "task": "record_recall" if family_index % 2 == 0 else "next_event",
                "transform_family": family,
                "event_indices": [family_index + 1],
                "variants": variants,
            }
            if base_cue["task"] == "next_event":
                base_cue["expected_record_id"] = record_id
            base_cues.append(base_cue)
        records.append(
            {
                "record_id": record_id,
                "training_source": training,
                "calibration_cue": calibration,
                "evaluation_base_cues": base_cues,
                "temporal_marker": {
                    "timestamp_ns": (1_700_000_000 + record_index) * 1_000_000_000,
                    "timestamp_source": "git-commit",
                    "timestamp_precision": "seconds",
                    "source_commit": sha256(f"commit-{record_index}".encode()).hexdigest()[:40],
                },
            }
        )

    record_ids = [record["record_id"] for record in records]
    encoder_config = {
        "feature_dim": 64,
        "packet_ms": 4,
        "silent_ms": 2,
        "active_fraction": 0.1,
        "projection_seed": 11,
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": "schema-v2-development",
        "experiment_digest": _digest(b"experiment"),
        "split": "development",
        "locked": False,
        "lock_timestamp": None,
        "seeds": _SEEDS,
        "encoder": {
            "checkpoint_locator": "encoder/model",
            "directory_digest": _digest(b"encoder-directory"),
            "config_digest": canonical_json_digest(encoder_config),
            "config": encoder_config,
        },
        "model": {"config_digest": _digest(b"model-config"), "input_current": 4.0},
        "source_universe": {
            "artifact": source_universe_artifact,
            "selector_digest": _digest(b"selector"),
            "payload_min_bytes": 1000,
            "payload_max_bytes": 20000,
            "event_min": 50,
            "event_max": 256,
        },
        "candidate_set": {
            "ordered_record_ids": record_ids,
            "digest": ordered_record_ids_digest(record_ids),
        },
        "context_budget": {"top_k": 5, "max_payload_utf8_bytes": 8192},
        "cue_contract": {
            "normalizer": "unicode-nfkc-whitespace-v1",
            "near_duplicate_metric": "token-jaccard-v1",
            "near_duplicate_maximum": 0.2,
        },
        "artifacts": artifacts,
        "records": records,
        "distractor_records": [distractor],
        "schedules": schedules,
    }
    path = root / "dataset.json"
    return path, manifest, _write_manifest(path, manifest)


def _rewrite_dataset(path: Path, manifest: dict[str, Any]) -> str:
    return _write_manifest(path, manifest)


@pytest.fixture
def dataset_snapshot(tmp_path: Path) -> tuple[Path, dict[str, Any], str, ValidatedDatasetV2]:
    path, manifest, digest = _dataset_fixture(tmp_path)
    return path, manifest, digest, read_and_validate_dataset_v2(path, digest)


def _checkpoint_value(dataset: ValidatedDatasetV2) -> dict[str, Any]:
    manifest = dataset.manifest
    records = manifest["records"]
    candidates = manifest["candidate_set"]
    topology = _digest(b"topology")
    return {
        "schema_version": 2,
        "experiment_digest": manifest["experiment_digest"],
        "dataset_digest": dataset.digest,
        "candidate_set_digest": candidates["digest"],
        "task_set_digest": manifest["artifacts"]["task_set"]["sha256"],
        "seed": 11,
        "condition": "trained",
        "ordered_record_ids": candidates["ordered_record_ids"],
        "topology_digest": topology,
        "encoder": {
            "checkpoint_locator": manifest["encoder"]["checkpoint_locator"],
            "directory_digest": manifest["encoder"]["directory_digest"],
            "config_digest": manifest["encoder"]["config_digest"],
        },
        "model_config_digest": manifest["model"]["config_digest"],
        "input_current": manifest["model"]["input_current"],
        "training_source_digests": [
            {"record_id": record["record_id"], "digest": record["training_source"]["sha256"]}
            for record in records
        ],
        "calibration_cue_digests": [
            {"record_id": record["record_id"], "digest": record["calibration_cue"]["sha256"]}
            for record in records
        ],
        "adjacency": {
            "orientation": "row-pre-column-post",
            "outgoing_digest": _digest(b"outgoing"),
            "incoming_digest": _digest(b"incoming"),
            "topology_digest": topology,
        },
        "arrays": {
            "weights_digest": _digest(b"weights"),
            "training_final_state_digest": _digest(b"training-final-state"),
            "probe_initial_state_digest": _digest(b"fresh-probe-initial-state"),
            "signatures_digest": _digest(b"signatures"),
            "record_ids_digest": candidates["digest"],
        },
        "epochs_completed": 2,
        "replay_order_digest": _digest(b"replay"),
        "build": {
            "python_wheel_digest": _digest(b"python-wheel"),
            "rust_wheel_digest": _digest(b"rust-wheel"),
            "backend_version": "0.1.0",
            "backend_build_digest": _digest(b"backend-build"),
            "repository_head": sha256(b"repository-head").hexdigest()[:40],
            "dirty_tree_digest": _digest(b"dirty-tree"),
            "patch_digest": _digest(b"patch"),
        },
        "scoring_calibration": {
            "abstention_threshold": 0.3,
            "metric": "temporal-overlap",
            "development_artifact_digest": manifest["artifacts"]["development_calibration"][
                "sha256"
            ],
        },
    }


def _task_output_value(
    checkpoint: ValidatedArtifactV2, dataset: ValidatedDatasetV2
) -> dict[str, Any]:
    first = dataset.manifest["candidate_set"]["ordered_record_ids"][0]
    return {
        "schema_version": 2,
        "checkpoint_digest": checkpoint.digest,
        "cue_digest": _digest(b"cue"),
        "initial_state_digest": checkpoint.manifest["arrays"]["probe_initial_state_digest"],
        "process_instance_digest": _digest(b"process-instance"),
        "backend_digest": checkpoint.manifest["build"]["backend_build_digest"],
        "record_order_digest": dataset.manifest["candidate_set"]["digest"],
        "predicted_record_id": first,
        "ranked_record_ids": [first],
        "scores": [0.9],
        "completion": {
            "spikes": 12,
            "recurrent_energy": 1.5,
            "half_life_steps": 4,
            "cue_steps": 8,
            "completion_steps": 16,
        },
        "trajectory_digest": _digest(b"trajectory"),
    }


def _evaluation_value(dataset: ValidatedDatasetV2) -> dict[str, Any]:
    tasks = list(_TASK_IDS)
    return {
        "schema_version": 2,
        "experiment_digest": dataset.manifest["experiment_digest"],
        "dataset_digest": dataset.digest,
        "candidate_set_digest": dataset.manifest["candidate_set"]["digest"],
        "ordered_record_ids": dataset.manifest["candidate_set"]["ordered_record_ids"],
        "expected_task_count": len(tasks),
        "completed_task_count": len(tasks),
        "expected_task_set_digest": task_set_digest(tasks),
        "completed_task_set_digest": task_set_digest(tasks),
        "expected_tasks": tasks,
        "completed_tasks": list(reversed(tasks)),
        "missing_tasks": [],
        "unexpected_tasks": [],
        "digest_failures": [],
        "per_seed_metrics": [
            {
                "seed": 11,
                "arm": "trained",
                "p_at_1": 0.5,
                "records": 16,
                "base_cues": 64,
                "variants": 256,
            }
        ],
        "nested_uncertainty": {
            "method": "nested-seed-record-cue-bootstrap-v1",
            "replicates": 10000,
            "bootstrap_seed": 11,
            "mean": 0.5,
            "lower": 0.4,
            "upper": 0.6,
            "seed_denominator": 10,
        },
        "control_metrics": [],
        "baseline_metrics": [],
        "mechanism_metrics": [],
        "audit_metrics": [],
        "gates": {
            "completeness_pass": True,
            "g0_fidelity_pass": True,
            "g1_controlled_completion_pass": True,
            "g2_locked_recall_pass": False,
            "g3_mechanism_pass": False,
            "heldout_utility_pass": False,
        },
    }


def test_dataset_reader_authenticates_real_files_and_keeps_single_read_snapshot(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
) -> None:
    path, manifest, _, dataset = dataset_snapshot
    relative = manifest["records"][0]["training_source"]["path"]
    original = dataset.artifact_bytes[relative]

    (path.parent / relative).write_bytes(b"changed-after-validation")

    assert dataset.artifact_bytes[relative] == original
    assert dataset.manifest["candidate_set"]["ordered_record_ids"][0].startswith("sha256:")


def test_dataset_snapshot_is_recursively_immutable(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
) -> None:
    dataset = dataset_snapshot[3]

    with pytest.raises(TypeError):
        dataset.manifest["records"][0]["record_id"] = "changed"
    with pytest.raises(AttributeError):
        dataset.manifest["records"].append("changed")
    with pytest.raises(TypeError):
        cast(Any, dataset.artifact_bytes)["new"] = b"changed"


def test_manifest_digest_is_checked_before_json_parse(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_bytes(b"not-json")

    with pytest.raises(ValueError, match="dataset digest mismatch"):
        read_and_validate_dataset_v2(path, _digest(b"different"))


@pytest.mark.parametrize("raw_bytes", [b"{", b"[]"])
def test_dataset_reader_rejects_nonobject_or_invalid_json(tmp_path: Path, raw_bytes: bytes) -> None:
    path = tmp_path / "dataset.json"
    path.write_bytes(raw_bytes)

    with pytest.raises(ValueError, match="canonical UTF-8 JSON|JSON object"):
        read_and_validate_dataset_v2(path, _digest(raw_bytes))


def test_dataset_reader_rejects_noncanonical_expected_digest(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        read_and_validate_dataset_v2(path, "A" * 64)


@pytest.mark.parametrize("nesting", ["top-level", "nested"])
def test_dataset_reader_rejects_duplicate_json_keys_at_every_depth(
    tmp_path: Path, nesting: str
) -> None:
    path, _, _ = _dataset_fixture(tmp_path)
    raw_bytes = path.read_bytes()
    if nesting == "top-level":
        raw_bytes = b'{"schema_version":999,' + raw_bytes[1:]
    else:
        raw_bytes = raw_bytes.replace(
            b'"encoder":{"checkpoint_locator":',
            b'"encoder":{"checkpoint_locator":"duplicate","checkpoint_locator":',
            1,
        )
    path.write_bytes(raw_bytes)

    with pytest.raises(ValueError, match="duplicate object key"):
        read_and_validate_dataset_v2(path, _digest(raw_bytes))


def test_dataset_reader_rejects_nonfinite_json_number_before_schema(tmp_path: Path) -> None:
    path, _, _ = _dataset_fixture(tmp_path)
    raw_bytes = path.read_bytes().replace(
        b'"near_duplicate_maximum":0.2',
        b'"near_duplicate_maximum":NaN',
        1,
    )
    path.write_bytes(raw_bytes)

    with pytest.raises(ValueError, match="non-finite JSON number NaN"):
        read_and_validate_dataset_v2(path, _digest(raw_bytes))


def test_dataset_reader_rejects_tampered_artifact(
    tmp_path: Path,
) -> None:
    path, manifest, digest = _dataset_fixture(tmp_path)
    relative = manifest["artifacts"]["provenance_store"]["path"]
    (tmp_path / relative).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact .* digest mismatch"):
        read_and_validate_dataset_v2(path, digest)


def test_dataset_reader_rejects_symlinked_artifact(tmp_path: Path) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    ref = manifest["artifacts"]["provenance_store"]
    artifact_path = tmp_path / ref["path"]
    outside = tmp_path / "outside.txt"
    outside.write_bytes(artifact_path.read_bytes())
    artifact_path.unlink()
    artifact_path.symlink_to(outside)
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match="traverses a symlink"):
        read_and_validate_dataset_v2(path, digest)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("missing", "escapes or is missing"),
        ("directory", "not a regular file"),
        ("invalid-utf8", "not UTF-8 text"),
        ("normalized-mismatch", "normalized-text digest mismatch"),
    ],
)
def test_dataset_reader_rejects_invalid_real_artifact_surfaces(
    tmp_path: Path, corruption: str, match: str
) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    calibration = manifest["records"][0]["calibration_cue"]
    if corruption == "missing":
        (tmp_path / manifest["source_universe"]["artifact"]["path"]).unlink()
    elif corruption == "directory":
        artifact_path = tmp_path / manifest["distractor_records"][0]["path"]
        artifact_path.unlink()
        artifact_path.mkdir()
    elif corruption == "invalid-utf8":
        raw_bytes = b"\xff"
        (tmp_path / calibration["path"]).write_bytes(raw_bytes)
        calibration["sha256"] = _digest(raw_bytes)
    else:
        calibration["normalized_text_sha256"] = _digest(b"different-normalized")
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match=match):
        read_and_validate_dataset_v2(path, digest)


def test_dataset_reader_rejects_duplicate_artifact_digest(tmp_path: Path) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    manifest["artifacts"]["temporal_edges"]["sha256"] = manifest["artifacts"]["provenance_store"][
        "sha256"
    ]
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match="duplicate artifact digest"):
        read_and_validate_dataset_v2(path, digest)


def test_dataset_reader_rejects_candidate_reordering(tmp_path: Path) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    manifest["candidate_set"]["ordered_record_ids"] = list(
        reversed(manifest["candidate_set"]["ordered_record_ids"])
    )
    manifest["candidate_set"]["digest"] = ordered_record_ids_digest(
        manifest["candidate_set"]["ordered_record_ids"]
    )
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match="candidate order"):
        read_and_validate_dataset_v2(path, digest)


def test_dataset_reader_rejects_nonopaque_cue_path(tmp_path: Path) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    calibration = manifest["records"][0]["calibration_cue"]
    old_path = tmp_path / calibration["path"]
    new_relative = "cues/descriptive-label.txt"
    (tmp_path / new_relative).write_bytes(old_path.read_bytes())
    old_path.unlink()
    calibration["path"] = new_relative
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match="opaque cue ID"):
        read_and_validate_dataset_v2(path, digest)


def test_dataset_reader_rejects_calibration_evaluation_overlap(tmp_path: Path) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    manifest["records"][0]["evaluation_base_cues"][0]["event_indices"] = [0]
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match="calibration and evaluation"):
        read_and_validate_dataset_v2(path, digest)


def test_dataset_reader_rejects_near_duplicate_canonical_cues(tmp_path: Path) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    first = manifest["records"][0]["calibration_cue"]
    second = manifest["records"][1]["calibration_cue"]
    first_text = (tmp_path / first["path"]).read_text()
    changed = f"{first_text} additional"
    (tmp_path / second["path"]).write_text(changed)
    second["sha256"] = _digest(changed.encode())
    second["normalized_text_sha256"] = _normalized_digest(changed)
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match="near-duplicate cues"):
        read_and_validate_dataset_v2(path, digest)


def test_empty_cues_have_identity_jaccard_similarity() -> None:
    assert token_jaccard(" \n", "\t") == 1.0


def test_canonical_json_digest_treats_frozen_sequences_as_json_arrays() -> None:
    assert canonical_json_digest({"values": (1, 2)}) == canonical_json_digest({"values": [1, 2]})


def test_dataset_reader_rejects_normalized_duplicate_variant(tmp_path: Path) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    variants = manifest["records"][0]["evaluation_base_cues"][0]["variants"]
    first_text = (tmp_path / variants[0]["path"]).read_text()
    fullwidth = "".join(
        chr(ord(character) + 0xFEE0) if character.isascii() and character.isalpha() else character
        for character in first_text
    )
    (tmp_path / variants[1]["path"]).write_text(fullwidth)
    variants[1]["sha256"] = _digest(fullwidth.encode())
    variants[1]["normalized_text_sha256"] = _normalized_digest(fullwidth)
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match="duplicate normalized cue digest"):
        read_and_validate_dataset_v2(path, digest)


def test_dataset_reader_rejects_off_candidate_next_event_answer(tmp_path: Path) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    manifest["records"][0]["evaluation_base_cues"][1]["expected_record_id"] = (
        f"sha256:{_digest(b'off-candidate-next-event')}"
    )
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match="next-event expected record is off-candidate"):
        read_and_validate_dataset_v2(path, digest)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("payload-small", "payload byte size"),
        ("payload-large", "payload byte size"),
        ("event-count", "event count"),
        ("training-order", "training event.*strictly increasing"),
        ("calibration-outside", "calibration event indices are outside"),
        ("calibration-order", "calibration event.*strictly increasing"),
        ("evaluation-outside", "evaluation event indices are outside"),
        ("evaluation-order", "evaluation event.*strictly increasing"),
    ],
)
def test_dataset_enforces_real_payload_and_event_bounds(
    tmp_path: Path, corruption: str, match: str
) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    record = manifest["records"][0]
    training = record["training_source"]
    if corruption.startswith("payload-"):
        raw_bytes = b"small" if corruption == "payload-small" else b"x" * 20001
        (tmp_path / training["path"]).write_bytes(raw_bytes)
        training["sha256"] = _digest(raw_bytes)
        replacement = f"sha256:{training['sha256']}"
        record["record_id"] = replacement
        for base_cue in record["evaluation_base_cues"]:
            if base_cue["task"] == "next_event":
                base_cue["expected_record_id"] = replacement
        ids = [item["record_id"] for item in manifest["records"]]
        manifest["candidate_set"] = {
            "ordered_record_ids": ids,
            "digest": ordered_record_ids_digest(ids),
        }
    elif corruption == "event-count":
        training["event_indices"] = list(range(49))
    elif corruption == "training-order":
        training["event_indices"] = [1, 0, *range(2, 50)]
    elif corruption == "calibration-outside":
        record["calibration_cue"]["event_indices"] = [999]
    elif corruption == "calibration-order":
        record["calibration_cue"]["event_indices"] = [1, 0]
    elif corruption == "evaluation-outside":
        record["evaluation_base_cues"][0]["event_indices"] = [999]
    else:
        record["evaluation_base_cues"][0]["event_indices"] = [2, 1]
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match=match):
        read_and_validate_dataset_v2(path, digest)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("split-lock", "locked split"),
        ("timestamp-lock", "lock timestamp"),
        ("candidate-digest", "candidate-set digest"),
        ("encoder-config", "encoder configuration digest"),
        ("duplicate-path", "duplicate artifact path"),
        ("record-binding", "training digest"),
        ("duplicate-calibration-id", "duplicate cue ID"),
        ("calibration-leak", "calibration path leaks"),
        ("duplicate-base-id", "duplicate cue ID"),
        ("base-leak", "base cue ID leaks"),
        ("variant-order", "variants must be ordered"),
        ("duplicate-variant-id", "duplicate variant ID"),
        ("variant-leak", "variant path leaks"),
        ("variant-nonopaque-path", "opaque variant ID"),
        ("duplicate-normalized", "duplicate normalized cue digest"),
        ("missing-family", "cover all four transform families"),
        ("evaluation-overlap", "evaluation event-index blocks overlap"),
        ("timestamp-precision", "exact integer second"),
    ],
)
def test_dataset_cross_contracts_fail_closed(tmp_path: Path, corruption: str, match: str) -> None:
    path, manifest, _ = _dataset_fixture(tmp_path)
    first = manifest["records"][0]
    second = manifest["records"][1]
    first_base = first["evaluation_base_cues"][0]
    first_variant = first_base["variants"][0]
    second_variant = first_base["variants"][1]
    if corruption == "split-lock":
        manifest["locked"] = True
        manifest["lock_timestamp"] = "2026-07-14T12:00:00Z"
    elif corruption == "timestamp-lock":
        manifest["lock_timestamp"] = "2026-07-14T12:00:00Z"
    elif corruption == "candidate-digest":
        manifest["candidate_set"]["digest"] = _digest(b"wrong-order")
    elif corruption == "encoder-config":
        manifest["encoder"]["config"]["feature_dim"] = 128
    elif corruption == "duplicate-path":
        manifest["artifacts"]["temporal_edges"]["path"] = manifest["artifacts"]["provenance_store"][
            "path"
        ]
    elif corruption == "record-binding":
        replacement = f"sha256:{_digest(b'replacement-record-id')}"
        first["record_id"] = replacement
        for base in first["evaluation_base_cues"]:
            if base["task"] == "next_event":
                base["expected_record_id"] = replacement
        ids = [record["record_id"] for record in manifest["records"]]
        manifest["candidate_set"] = {
            "ordered_record_ids": ids,
            "digest": ordered_record_ids_digest(ids),
        }
    elif corruption == "duplicate-calibration-id":
        second["calibration_cue"]["cue_id"] = first["calibration_cue"]["cue_id"]
    elif corruption == "calibration-leak":
        cue = first["calibration_cue"]
        leaked_id = f"cue-{first['record_id'].removeprefix('sha256:')[:32]}"
        old_path = tmp_path / cue["path"]
        new_relative = f"cues/{leaked_id}.txt"
        (tmp_path / new_relative).write_bytes(old_path.read_bytes())
        old_path.unlink()
        cue["cue_id"] = leaked_id
        cue["path"] = new_relative
    elif corruption == "duplicate-base-id":
        first_base["cue_id"] = first["calibration_cue"]["cue_id"]
    elif corruption == "base-leak":
        first_base["cue_id"] = f"cue-{first['record_id'].removeprefix('sha256:')[:32]}"
    elif corruption == "variant-order":
        first_base["variants"][0], first_base["variants"][1] = (
            first_base["variants"][1],
            first_base["variants"][0],
        )
    elif corruption == "duplicate-variant-id":
        second_variant["variant_id"] = first_variant["variant_id"]
    elif corruption == "variant-leak":
        first_variant["variant_id"] = f"variant-{first['record_id'].removeprefix('sha256:')[:32]}"
    elif corruption == "variant-nonopaque-path":
        old_path = tmp_path / first_variant["path"]
        new_relative = "variants/descriptive.txt"
        (tmp_path / new_relative).write_bytes(old_path.read_bytes())
        old_path.unlink()
        first_variant["path"] = new_relative
    elif corruption == "duplicate-normalized":
        calibration = second["calibration_cue"]
        text = (tmp_path / first["calibration_cue"]["path"]).read_text()
        fullwidth = "".join(
            chr(ord(character) + 0xFEE0)
            if character.isascii() and character.isalpha()
            else character
            for character in text
        )
        (tmp_path / calibration["path"]).write_text(fullwidth)
        calibration["sha256"] = _digest(fullwidth.encode())
        calibration["normalized_text_sha256"] = _normalized_digest(fullwidth)
    elif corruption == "missing-family":
        first["evaluation_base_cues"][1]["transform_family"] = "truncation"
    elif corruption == "evaluation-overlap":
        first["evaluation_base_cues"][1]["event_indices"] = first_base["event_indices"]
    else:
        first["temporal_marker"]["timestamp_ns"] += 1
    digest = _rewrite_dataset(path, manifest)

    with pytest.raises(ValueError, match=match):
        read_and_validate_dataset_v2(path, digest)


@pytest.mark.parametrize("artifact_kind", ["dataset", "checkpoint", "task-output", "evaluation"])
def test_public_readers_reject_unknown_schema_fields(tmp_path: Path, artifact_kind: str) -> None:
    path, manifest, dataset_digest = _dataset_fixture(tmp_path)
    if artifact_kind == "dataset":
        manifest["unknown"] = True
        digest = _rewrite_dataset(path, manifest)
        with pytest.raises(ValueError, match="failed .*dataset_v2"):
            read_and_validate_dataset_v2(path, digest)
        return

    dataset = read_and_validate_dataset_v2(path, dataset_digest)
    checkpoint_value = _checkpoint_value(dataset)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_digest = _write_manifest(checkpoint_path, checkpoint_value)
    checkpoint = read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)
    if artifact_kind == "checkpoint":
        checkpoint_value["unknown"] = True
        digest = _write_manifest(checkpoint_path, checkpoint_value)
        with pytest.raises(ValueError, match="failed .*checkpoint_v2"):
            read_and_validate_checkpoint_v2(checkpoint_path, digest, dataset)
    elif artifact_kind == "task-output":
        value = _task_output_value(checkpoint, dataset)
        value["unknown"] = True
        output_path = tmp_path / "task-output.json"
        digest = _write_manifest(output_path, value)
        with pytest.raises(ValueError, match="failed .*task_output_v2"):
            read_and_validate_task_output_v2(output_path, digest, checkpoint, dataset)
    else:
        value = _evaluation_value(dataset)
        value["unknown"] = True
        evaluation_path = tmp_path / "evaluation.json"
        digest = _write_manifest(evaluation_path, value)
        with pytest.raises(ValueError, match="failed .*evaluation_v2"):
            read_and_validate_evaluation_v2(evaluation_path, digest, dataset)


def test_checkpoint_task_output_and_evaluation_bind_real_files(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
) -> None:
    root = dataset_snapshot[0].parent
    dataset = dataset_snapshot[3]
    checkpoint_path = root / "checkpoint.json"
    checkpoint_digest = _write_manifest(checkpoint_path, _checkpoint_value(dataset))
    checkpoint = read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)

    output_path = root / "task-output.json"
    output_digest = _write_manifest(output_path, _task_output_value(checkpoint, dataset))
    output = read_and_validate_task_output_v2(output_path, output_digest, checkpoint, dataset)

    evaluation_path = root / "evaluation.json"
    evaluation_digest = _write_manifest(evaluation_path, _evaluation_value(dataset))
    evaluation = read_and_validate_evaluation_v2(evaluation_path, evaluation_digest, dataset)

    assert output.digest == output_digest
    assert evaluation.manifest["gates"]["completeness_pass"] is True


@pytest.mark.parametrize("corruption", ["off-candidate", "rank-score-length"])
def test_task_output_rejects_invalid_rank_surface(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
    corruption: str,
) -> None:
    root = dataset_snapshot[0].parent
    dataset = dataset_snapshot[3]
    checkpoint_path = root / "checkpoint.json"
    checkpoint_digest = _write_manifest(checkpoint_path, _checkpoint_value(dataset))
    checkpoint = read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)
    value = _task_output_value(checkpoint, dataset)
    if corruption == "off-candidate":
        off_candidate = f"sha256:{_digest(b'off-candidate')}"
        value["predicted_record_id"] = off_candidate
        value["ranked_record_ids"] = [off_candidate]
    else:
        value["scores"] = []
    output_path = root / "task-output.json"
    output_digest = _write_manifest(output_path, value)

    with pytest.raises(ValueError, match="off-candidate|score/rank"):
        read_and_validate_task_output_v2(output_path, output_digest, checkpoint, dataset)


def test_probe_accepts_only_fresh_reset_state_never_training_final_state(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
) -> None:
    root = dataset_snapshot[0].parent
    dataset = dataset_snapshot[3]
    checkpoint_path = root / "checkpoint-state-separation.json"
    checkpoint_digest = _write_manifest(checkpoint_path, _checkpoint_value(dataset))
    checkpoint = read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)
    arrays = checkpoint.manifest["arrays"]
    assert arrays["training_final_state_digest"] != arrays["probe_initial_state_digest"]

    value = _task_output_value(checkpoint, dataset)
    output_path = root / "task-output-fresh-state.json"
    output_digest = _write_manifest(output_path, value)
    accepted = read_and_validate_task_output_v2(output_path, output_digest, checkpoint, dataset)
    assert accepted.manifest["initial_state_digest"] == arrays["probe_initial_state_digest"]

    value["initial_state_digest"] = arrays["training_final_state_digest"]
    output_digest = _write_manifest(output_path, value)
    with pytest.raises(ValueError, match="initial-state digest mismatch"):
        read_and_validate_task_output_v2(output_path, output_digest, checkpoint, dataset)


def test_incomplete_evaluation_forces_every_scientific_gate_false(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
) -> None:
    root = dataset_snapshot[0].parent
    dataset = dataset_snapshot[3]
    value = _evaluation_value(dataset)
    missing = value["completed_tasks"].pop()
    value["completed_task_count"] -= 1
    value["completed_task_set_digest"] = task_set_digest(value["completed_tasks"])
    value["missing_tasks"] = [missing]
    value["gates"]["completeness_pass"] = False
    evaluation_path = root / "evaluation.json"
    evaluation_digest = _write_manifest(evaluation_path, value)

    with pytest.raises(ValueError, match="incomplete evaluation"):
        read_and_validate_evaluation_v2(evaluation_path, evaluation_digest, dataset)

    for gate in value["gates"]:
        value["gates"][gate] = False
    evaluation_digest = _write_manifest(evaluation_path, value)
    evaluation = read_and_validate_evaluation_v2(evaluation_path, evaluation_digest, dataset)
    assert evaluation.manifest["gates"]["completeness_pass"] is False


def test_checkpoint_rejects_record_binding_reorder(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
) -> None:
    root = dataset_snapshot[0].parent
    dataset = dataset_snapshot[3]
    value = _checkpoint_value(dataset)
    value["training_source_digests"] = list(reversed(value["training_source_digests"]))
    checkpoint_path = root / "checkpoint.json"
    checkpoint_digest = _write_manifest(checkpoint_path, value)

    with pytest.raises(ValueError, match="training-source bindings"):
        read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("experiment", "experiment digest"),
        ("dataset", "dataset digest"),
        ("record-order", "record order"),
        ("candidate", "candidate-set digest"),
        ("array-order", "array record-order"),
        ("task-set", "task-set artifact"),
        ("encoder-locator", "encoder checkpoint locator"),
        ("encoder-directory", "encoder directory digest"),
        ("encoder-config", "encoder config digest"),
        ("model-config", "model configuration digest"),
        ("input-current", "input current"),
        ("topology", "topology digests"),
        ("calibration", "calibration-cue bindings"),
        ("development-calibration", "development-calibration artifact"),
    ],
)
def test_checkpoint_cross_bindings_fail_closed(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
    corruption: str,
    match: str,
) -> None:
    root = dataset_snapshot[0].parent
    dataset = dataset_snapshot[3]
    value = _checkpoint_value(dataset)
    if corruption == "experiment":
        value["experiment_digest"] = _digest(b"wrong-experiment")
    elif corruption == "dataset":
        value["dataset_digest"] = _digest(b"wrong-dataset")
    elif corruption == "record-order":
        value["ordered_record_ids"] = list(reversed(value["ordered_record_ids"]))
    elif corruption == "candidate":
        value["candidate_set_digest"] = _digest(b"wrong-candidates")
    elif corruption == "array-order":
        value["arrays"]["record_ids_digest"] = _digest(b"wrong-array-order")
    elif corruption == "task-set":
        value["task_set_digest"] = _digest(b"wrong-task-set")
    elif corruption == "encoder-locator":
        value["encoder"]["checkpoint_locator"] = "wrong/model"
    elif corruption == "encoder-directory":
        value["encoder"]["directory_digest"] = _digest(b"wrong-encoder-directory")
    elif corruption == "encoder-config":
        value["encoder"]["config_digest"] = _digest(b"wrong-encoder-config")
    elif corruption == "model-config":
        value["model_config_digest"] = _digest(b"wrong-model-config")
    elif corruption == "input-current":
        value["input_current"] = 99.0
    elif corruption == "topology":
        value["adjacency"]["topology_digest"] = _digest(b"wrong-topology")
    elif corruption == "calibration":
        value["calibration_cue_digests"] = list(reversed(value["calibration_cue_digests"]))
    else:
        value["scoring_calibration"]["development_artifact_digest"] = _digest(
            b"wrong-development-calibration"
        )
    checkpoint_path = root / f"checkpoint-{corruption}.json"
    checkpoint_digest = _write_manifest(checkpoint_path, value)

    with pytest.raises(ValueError, match=match):
        read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("checkpoint", "checkpoint digest"),
        ("record-order", "record-order digest"),
        ("backend", "backend build digest"),
        ("initial-state", "initial-state digest"),
        ("off-rank", "off-candidate rank"),
        ("off-prediction", "prediction is off-candidate"),
        ("prediction-order", "first ranked record"),
    ],
)
def test_task_output_cross_bindings_fail_closed(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
    corruption: str,
    match: str,
) -> None:
    root = dataset_snapshot[0].parent
    dataset = dataset_snapshot[3]
    checkpoint_path = root / "checkpoint.json"
    checkpoint_digest = _write_manifest(checkpoint_path, _checkpoint_value(dataset))
    checkpoint = read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)
    value = _task_output_value(checkpoint, dataset)
    off_candidate = f"sha256:{_digest(b'off-candidate-output')}"
    if corruption == "checkpoint":
        value["checkpoint_digest"] = _digest(b"wrong-checkpoint")
    elif corruption == "record-order":
        value["record_order_digest"] = _digest(b"wrong-record-order")
    elif corruption == "backend":
        value["backend_digest"] = _digest(b"wrong-backend")
    elif corruption == "initial-state":
        value["initial_state_digest"] = _digest(b"wrong-initial-state")
    elif corruption == "off-rank":
        value["predicted_record_id"] = None
        value["ranked_record_ids"] = [off_candidate]
    elif corruption == "off-prediction":
        value["predicted_record_id"] = off_candidate
    else:
        value["predicted_record_id"] = dataset.manifest["candidate_set"]["ordered_record_ids"][1]
    output_path = root / f"task-output-{corruption}.json"
    output_digest = _write_manifest(output_path, value)

    with pytest.raises(ValueError, match=match):
        read_and_validate_task_output_v2(output_path, output_digest, checkpoint, dataset)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("dataset", "dataset digest"),
        ("experiment", "experiment digest"),
        ("record-order", "record order"),
        ("candidate", "candidate-set digest"),
        ("expected-count", "expected task count"),
        ("completed-count", "completed task count"),
        ("expected-digest", "expected task-set digest"),
        ("dataset-task-set", "dataset task set"),
        ("completed-digest", "completed task-set digest"),
        ("missing", "missing task set"),
        ("unexpected", "unexpected task set"),
        ("unknown-failure", "digest failures reference unknown"),
        ("completeness", "reported completeness contradicts"),
    ],
)
def test_evaluation_cross_bindings_fail_closed(tmp_path: Path, corruption: str, match: str) -> None:
    path, manifest, dataset_digest = _dataset_fixture(tmp_path)
    if corruption == "dataset-task-set":
        task_ref = manifest["artifacts"]["task_set"]
        changed = b"different-authenticated-task-set\n"
        (tmp_path / task_ref["path"]).write_bytes(changed)
        task_ref["sha256"] = _digest(changed)
        dataset_digest = _rewrite_dataset(path, manifest)
    dataset = read_and_validate_dataset_v2(path, dataset_digest)
    value = _evaluation_value(dataset)
    if corruption == "dataset":
        value["dataset_digest"] = _digest(b"wrong-dataset")
    elif corruption == "experiment":
        value["experiment_digest"] = _digest(b"wrong-experiment")
    elif corruption == "record-order":
        value["ordered_record_ids"] = list(reversed(value["ordered_record_ids"]))
    elif corruption == "candidate":
        value["candidate_set_digest"] = _digest(b"wrong-candidates")
    elif corruption == "expected-count":
        value["expected_task_count"] += 1
    elif corruption == "completed-count":
        value["completed_task_count"] += 1
    elif corruption == "expected-digest":
        value["expected_task_set_digest"] = _digest(b"wrong-expected-set")
    elif corruption == "completed-digest":
        value["completed_task_set_digest"] = _digest(b"wrong-completed-set")
    elif corruption == "missing":
        value["missing_tasks"] = [value["expected_tasks"][0]]
    elif corruption == "unexpected":
        value["unexpected_tasks"] = [value["expected_tasks"][0]]
    elif corruption == "unknown-failure":
        value["digest_failures"] = [f"task:{_digest(b'unknown-task')}"]
    elif corruption == "completeness":
        value["gates"]["completeness_pass"] = False
    evaluation_path = tmp_path / f"evaluation-{corruption}.json"
    evaluation_digest = _write_manifest(evaluation_path, value)

    with pytest.raises(ValueError, match=match):
        read_and_validate_evaluation_v2(evaluation_path, evaluation_digest, dataset)


def test_untrained_checkpoint_requires_zero_completed_epochs(
    dataset_snapshot: tuple[Path, dict[str, Any], str, ValidatedDatasetV2],
) -> None:
    root = dataset_snapshot[0].parent
    dataset = dataset_snapshot[3]
    value = _checkpoint_value(dataset)
    value["condition"] = "untrained"
    checkpoint_path = root / "checkpoint.json"
    checkpoint_digest = _write_manifest(checkpoint_path, value)

    with pytest.raises(ValueError, match="failed .*checkpoint_v2"):
        read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)

    value["epochs_completed"] = 0
    checkpoint_digest = _write_manifest(checkpoint_path, value)
    checkpoint = read_and_validate_checkpoint_v2(checkpoint_path, checkpoint_digest, dataset)
    assert checkpoint.manifest["epochs_completed"] == 0


@pytest.mark.parametrize(
    "schema_name",
    [
        "snn_memory_dataset_v2.schema.json",
        "snn_memory_checkpoint_v2.schema.json",
        "snn_memory_task_output_v2.schema.json",
        "snn_memory_evaluation_v2.schema.json",
    ],
)
def test_public_and_packaged_schema_bytes_cannot_drift(schema_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "schema" / schema_name).read_bytes() == (
        root / "snn_memory" / "schema" / schema_name
    ).read_bytes()


def test_built_installed_wheel_runs_public_dataset_reader(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    outside = tmp_path / "outside"
    wheel_dir = tmp_path / "wheel"
    target = tmp_path / "target"
    dataset_root = tmp_path / "dataset"
    for directory in (outside, wheel_dir, dataset_root):
        directory.mkdir()
    dataset_path, _, dataset_digest = _dataset_fixture(dataset_root)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(root),
        ],
        cwd=outside,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheels[0]),
        ],
        cwd=outside,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    installed_environment = {**environment, "PYTHONPATH": str(target)}
    script = """
import json
import sys
from pathlib import Path
import snn_memory.v2_contracts as contracts

dataset = contracts.read_and_validate_dataset_v2(Path(sys.argv[1]), sys.argv[2])
print(json.dumps({
    "origin": str(Path(contracts.__file__).resolve()),
    "records": len(dataset.manifest["records"]),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(dataset_path), dataset_digest],
        cwd=outside,
        env=installed_environment,
        text=True,
        capture_output=True,
        check=True,
    )
    evidence = json.loads(result.stdout)
    assert Path(evidence["origin"]).is_relative_to(target.resolve())
    assert evidence["records"] == 16
