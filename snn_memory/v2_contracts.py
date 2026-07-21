# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN schema-v2 artifact contracts

"""Fail-closed readers for the schema-v2 temporal-memory artifacts.

The readers deliberately authenticate raw bytes before decoding them. Dataset
artifacts are read once, retained in the returned snapshot, and never reopened
during validation. Dataset cue references authenticate UTF-8 text; the separate
probe cue-bundle materializer/schema is intentionally deferred beyond Slice A.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib.resources import files
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, cast
import unicodedata


_DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")
_DATASET_SCHEMA = "snn_memory_dataset_v2.schema.json"
_CHECKPOINT_SCHEMA = "snn_memory_checkpoint_v2.schema.json"
_TASK_OUTPUT_SCHEMA = "snn_memory_task_output_v2.schema.json"
_EVALUATION_SCHEMA = "snn_memory_evaluation_v2.schema.json"


@dataclass(frozen=True)
class ValidatedArtifactV2:
    """An authenticated JSON artifact and the bytes used to validate it."""

    manifest: Mapping[str, Any]
    digest: str
    raw_bytes: bytes


@dataclass(frozen=True)
class ValidatedDatasetV2(ValidatedArtifactV2):
    """An authenticated dataset manifest plus its single-read byte snapshot."""

    artifact_bytes: Mapping[str, bytes]


def _framed_digest(domain: bytes, values: Sequence[str]) -> str:
    digest = sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def ordered_record_ids_digest(record_ids: Sequence[str]) -> str:
    """Hash an ordered candidate list without delimiter ambiguity."""

    return _framed_digest(b"remanentia:snn-memory:ordered-record-ids:v2", record_ids)


def task_set_digest(task_ids: Sequence[str]) -> str:
    """Hash the canonical authenticated task-set artifact bytes."""

    canonical = "remanentia:snn-memory:task-set:v2\n"
    canonical += "".join(f"{task_id}\n" for task_id in sorted(set(task_ids)))
    return _digest_bytes(canonical.encode("ascii"))


def canonical_json_digest(value: object) -> str:
    """Hash the frozen canonical-JSON representation of a configuration."""

    def thaw(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [thaw(child) for child in item]
        return item

    raw_bytes = json.dumps(
        thaw(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _digest_bytes(raw_bytes)


def normalize_cue_text(text: str) -> str:
    """Apply the frozen unicode-nfkc-whitespace-v1 normalizer."""

    return " ".join(unicodedata.normalize("NFKC", text).split())


def token_jaccard(left: str, right: str) -> float:
    """Return token-set Jaccard similarity after frozen normalization."""

    left_tokens = set(normalize_cue_text(left).split())
    right_tokens = set(normalize_cue_text(right).split())
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def _require_digest(value: str, label: str) -> None:
    if not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _digest_bytes(raw_bytes: bytes) -> str:
    return sha256(raw_bytes).hexdigest()


def _read_authenticated(path: Path, expected_digest: str, label: str) -> bytes:
    _require_digest(expected_digest, f"expected {label} digest")
    raw_bytes = path.read_bytes()
    actual_digest = _digest_bytes(raw_bytes)
    if actual_digest != expected_digest:
        raise ValueError(
            f"{label} digest mismatch: expected {expected_digest}, got {actual_digest}"
        )
    return raw_bytes


def _parse_json(raw_bytes: bytes, label: str) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate object key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw_bytes.decode("utf-8"),
            parse_constant=reject_nonfinite,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _schema_bytes(schema_name: str) -> bytes:
    return files("snn_memory").joinpath("schema", schema_name).read_bytes()


def _validate_schema(value: Mapping[str, Any], schema_name: str, label: str) -> None:
    schema = _parse_json(_schema_bytes(schema_name), f"schema {schema_name}")
    validate = cast(Callable[[object, object], None], import_module("jsonschema").validate)
    try:
        validate(value, schema)
    except Exception as exc:
        raise ValueError(f"{label} failed {schema_name}: {exc}") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _load_json_artifact(
    path: Path, expected_digest: str, schema_name: str, label: str
) -> ValidatedArtifactV2:
    raw_bytes = _read_authenticated(path, expected_digest, label)
    manifest = _parse_json(raw_bytes, label)
    _validate_schema(manifest, schema_name, label)
    return ValidatedArtifactV2(
        manifest=cast(Mapping[str, Any], _freeze_json(manifest)),
        digest=expected_digest,
        raw_bytes=raw_bytes,
    )


def _artifact_refs(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    refs: list[Mapping[str, Any]] = []
    refs.extend(cast(Mapping[str, Any], ref) for ref in manifest["artifacts"].values())
    refs.append(cast(Mapping[str, Any], manifest["source_universe"]["artifact"]))
    refs.extend(
        cast(Mapping[str, Any], manifest["schedules"][name])
        for name in (
            "interference",
            "forgetting",
            "leave_one_record_out",
            "temporal_order_permutation",
        )
    )
    refs.extend(cast(Mapping[str, Any], ref) for ref in manifest["distractor_records"])
    for record in manifest["records"]:
        refs.append(cast(Mapping[str, Any], record["training_source"]))
        refs.append(cast(Mapping[str, Any], record["calibration_cue"]))
        for base_cue in record["evaluation_base_cues"]:
            refs.extend(cast(Mapping[str, Any], variant) for variant in base_cue["variants"])
    return refs


def _secure_artifact_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    cursor = root
    for part in Path(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"artifact path traverses a symlink: {relative}")
    root_resolved = root.resolve(strict=True)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ValueError(f"artifact path escapes or is missing: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"artifact path is not a regular file: {relative}")
    return resolved


def _decode_and_check_normalized(raw_bytes: bytes, expected_digest: str, label: str) -> str:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 text") from exc
    normalized = normalize_cue_text(text)
    actual = _digest_bytes(normalized.encode("utf-8"))
    if actual != expected_digest:
        raise ValueError(f"{label} normalized-text digest mismatch")
    return normalized


def _require_strictly_increasing(values: Sequence[int], label: str) -> None:
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} indices must be strictly increasing")


def read_and_validate_dataset_v2(path: Path, expected_digest: str) -> ValidatedDatasetV2:
    """Authenticate and validate one schema-v2 dataset snapshot.

    Every referenced artifact is read exactly once after path and symlink checks.
    Its digest is checked before any text decoding.
    """

    loaded = _load_json_artifact(path, expected_digest, _DATASET_SCHEMA, "dataset")
    manifest = loaded.manifest
    root = path.resolve(strict=True).parent

    split = manifest["split"]
    locked = manifest["locked"]
    timestamp = manifest["lock_timestamp"]
    if (split == "locked") != locked:
        raise ValueError("locked split and locked boolean disagree")
    if locked != (timestamp is not None):
        raise ValueError("lock timestamp presence and locked state disagree")

    records = manifest["records"]
    record_ids = tuple(cast(str, record["record_id"]) for record in records)
    candidate_ids = set(record_ids)
    candidate_set = manifest["candidate_set"]
    if candidate_set["ordered_record_ids"] != record_ids:
        raise ValueError("candidate order differs from record order")
    if candidate_set["digest"] != ordered_record_ids_digest(record_ids):
        raise ValueError("candidate-set digest does not bind the ordered IDs")
    if manifest["encoder"]["config_digest"] != canonical_json_digest(manifest["encoder"]["config"]):
        raise ValueError("encoder configuration digest mismatch")

    refs = _artifact_refs(manifest)
    paths = [cast(str, ref["path"]) for ref in refs]
    digests = [cast(str, ref["sha256"]) for ref in refs]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate artifact path")
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate artifact digest")

    byte_snapshot: dict[str, bytes] = {}
    for ref in refs:
        relative = cast(str, ref["path"])
        expected = cast(str, ref["sha256"])
        artifact_path = _secure_artifact_path(root, relative)
        byte_snapshot[relative] = _read_authenticated(
            artifact_path, expected, f"artifact {relative}"
        )

    cue_ids: set[str] = set()
    variant_ids: set[str] = set()
    canonical_cues: list[tuple[str, str]] = []
    normalized_digests: set[str] = set()

    for record in records:
        record_id = cast(str, record["record_id"])
        record_hash = record_id.removeprefix("sha256:")
        training = record["training_source"]
        if training["sha256"] != record_hash:
            raise ValueError(f"training digest does not match record ID {record_id}")
        training_bytes = byte_snapshot[cast(str, training["path"])]
        payload_size = len(training_bytes)
        source_universe = manifest["source_universe"]
        if not (
            source_universe["payload_min_bytes"]
            <= payload_size
            <= source_universe["payload_max_bytes"]
        ):
            raise ValueError(f"training payload byte size is out of bounds for {record_id}")
        training_indices = cast(tuple[int, ...], training["event_indices"])
        _require_strictly_increasing(training_indices, "training event")
        if not (
            source_universe["event_min"] <= len(training_indices) <= source_universe["event_max"]
        ):
            raise ValueError(f"training event count is out of bounds for {record_id}")
        training_index_set = set(training_indices)

        calibration = record["calibration_cue"]
        calibration_id = cast(str, calibration["cue_id"])
        calibration_path = cast(str, calibration["path"])
        if calibration_id in cue_ids:
            raise ValueError(f"duplicate cue ID {calibration_id}")
        cue_ids.add(calibration_id)
        if record_hash[:32] in calibration_id or record_hash[:32] in calibration_path:
            raise ValueError("calibration path leaks its record ID")
        if Path(calibration_path).stem != calibration_id:
            raise ValueError("calibration path must use only its opaque cue ID")
        calibration_text = _decode_and_check_normalized(
            byte_snapshot[calibration_path],
            cast(str, calibration["normalized_text_sha256"]),
            calibration_id,
        )
        normalized_digest = cast(str, calibration["normalized_text_sha256"])
        if normalized_digest in normalized_digests:
            raise ValueError("duplicate normalized cue digest")
        normalized_digests.add(normalized_digest)
        canonical_cues.append((calibration_id, calibration_text))
        calibration_values = cast(tuple[int, ...], calibration["event_indices"])
        _require_strictly_increasing(calibration_values, "calibration event")
        calibration_indices = set(calibration_values)
        if not calibration_indices <= training_index_set:
            raise ValueError("calibration event indices are outside the training source")

        transform_families: set[str] = set()
        evaluation_blocks: list[set[int]] = []
        for base_cue in record["evaluation_base_cues"]:
            cue_id = cast(str, base_cue["cue_id"])
            if cue_id in cue_ids:
                raise ValueError(f"duplicate cue ID {cue_id}")
            cue_ids.add(cue_id)
            if record_hash[:32] in cue_id:
                raise ValueError("base cue ID leaks its record ID")
            transform_families.add(cast(str, base_cue["transform_family"]))
            if (
                base_cue["task"] == "next_event"
                and base_cue["expected_record_id"] not in candidate_ids
            ):
                raise ValueError("next-event expected record is off-candidate")
            evaluation_values = cast(tuple[int, ...], base_cue["event_indices"])
            _require_strictly_increasing(evaluation_values, "evaluation event")
            evaluation_indices = set(evaluation_values)
            if not evaluation_indices <= training_index_set:
                raise ValueError("evaluation event indices are outside the training source")
            evaluation_blocks.append(evaluation_indices)

            variants = base_cue["variants"]
            requested = [variant["requested_percent"] for variant in variants]
            if requested != [0, 10, 25, 40]:
                raise ValueError("cue variants must be ordered 0,10,25,40 percent")
            for variant in variants:
                variant_id = cast(str, variant["variant_id"])
                variant_path = cast(str, variant["path"])
                if variant_id in variant_ids:
                    raise ValueError(f"duplicate variant ID {variant_id}")
                variant_ids.add(variant_id)
                if record_hash[:32] in variant_id or record_hash[:32] in variant_path:
                    raise ValueError("variant path leaks its record ID")
                if Path(variant_path).stem != variant_id:
                    raise ValueError("variant path must use only its opaque variant ID")
                normalized_text = _decode_and_check_normalized(
                    byte_snapshot[variant_path],
                    cast(str, variant["normalized_text_sha256"]),
                    variant_id,
                )
                normalized_digest = cast(str, variant["normalized_text_sha256"])
                if normalized_digest in normalized_digests:
                    raise ValueError("duplicate normalized cue digest")
                normalized_digests.add(normalized_digest)
                if variant["requested_percent"] == 0:
                    canonical_cues.append((cue_id, normalized_text))
        if transform_families != {"truncation", "deletion", "masking", "sparse_noise"}:
            raise ValueError("each record must cover all four transform families")
        if any(calibration_indices & block for block in evaluation_blocks):
            raise ValueError("calibration and evaluation event indices overlap")
        for index, left in enumerate(evaluation_blocks):
            if any(left & right for right in evaluation_blocks[index + 1 :]):
                raise ValueError("evaluation event-index blocks overlap")
        if record["temporal_marker"]["timestamp_ns"] % 1_000_000_000:
            raise ValueError("second-precision timestamp must be an exact integer second")

    maximum = cast(float, manifest["cue_contract"]["near_duplicate_maximum"])
    for index, (left_id, left_text) in enumerate(canonical_cues):
        for right_id, right_text in canonical_cues[index + 1 :]:
            if token_jaccard(left_text, right_text) > maximum:
                raise ValueError(f"near-duplicate cues: {left_id} and {right_id}")

    return ValidatedDatasetV2(
        manifest=manifest,
        digest=loaded.digest,
        raw_bytes=loaded.raw_bytes,
        artifact_bytes=MappingProxyType(byte_snapshot),
    )


def read_and_validate_checkpoint_v2(
    path: Path, expected_digest: str, dataset: ValidatedDatasetV2
) -> ValidatedArtifactV2:
    """Authenticate a checkpoint manifest and bind it to its dataset snapshot.

    Slice A does not receive topology/array/replay-order bytes, so their digests
    remain authenticated checkpoint bindings for Stage 2 rather than byte-level
    validations here. Training-final membrane/refractory/pre/post-trace state is
    audit-only; every probe binds a distinct freshly reset initial-state digest.
    """

    loaded = _load_json_artifact(path, expected_digest, _CHECKPOINT_SCHEMA, "checkpoint")
    value = loaded.manifest
    dataset_value = dataset.manifest
    record_ids = dataset_value["candidate_set"]["ordered_record_ids"]
    candidate_digest = dataset_value["candidate_set"]["digest"]
    if value["experiment_digest"] != dataset_value["experiment_digest"]:
        raise ValueError("checkpoint experiment digest mismatch")
    if value["dataset_digest"] != dataset.digest:
        raise ValueError("checkpoint dataset digest mismatch")
    if value["ordered_record_ids"] != record_ids:
        raise ValueError("checkpoint record order mismatch")
    if value["candidate_set_digest"] != candidate_digest:
        raise ValueError("checkpoint candidate-set digest mismatch")
    if value["arrays"]["record_ids_digest"] != candidate_digest:
        raise ValueError("checkpoint array record-order digest mismatch")
    if value["task_set_digest"] != dataset_value["artifacts"]["task_set"]["sha256"]:
        raise ValueError("checkpoint task-set artifact digest mismatch")
    for field in ("checkpoint_locator", "directory_digest", "config_digest"):
        if value["encoder"][field] != dataset_value["encoder"][field]:
            raise ValueError(f"checkpoint encoder {field.replace('_', ' ')} mismatch")
    if value["model_config_digest"] != dataset_value["model"]["config_digest"]:
        raise ValueError("checkpoint model configuration digest mismatch")
    if value["input_current"] != dataset_value["model"]["input_current"]:
        raise ValueError("checkpoint input current mismatch")
    if value["adjacency"]["topology_digest"] != value["topology_digest"]:
        raise ValueError("checkpoint topology digests disagree")

    expected_training = tuple(
        {"record_id": record["record_id"], "digest": record["training_source"]["sha256"]}
        for record in dataset_value["records"]
    )
    expected_calibration = tuple(
        {"record_id": record["record_id"], "digest": record["calibration_cue"]["sha256"]}
        for record in dataset_value["records"]
    )
    if value["training_source_digests"] != expected_training:
        raise ValueError("checkpoint training-source bindings mismatch")
    if value["calibration_cue_digests"] != expected_calibration:
        raise ValueError("checkpoint calibration-cue bindings mismatch")
    if (
        value["scoring_calibration"]["development_artifact_digest"]
        != dataset_value["artifacts"]["development_calibration"]["sha256"]
    ):
        raise ValueError("checkpoint development-calibration artifact mismatch")
    return loaded


def read_and_validate_task_output_v2(
    path: Path,
    expected_digest: str,
    checkpoint: ValidatedArtifactV2,
    dataset: ValidatedDatasetV2,
) -> ValidatedArtifactV2:
    """Authenticate a probe result without exposing expected-answer material.

    Cue-bundle and trajectory bytes are Stage-2 inputs and are therefore only
    digest bindings in Slice A; this reader does authenticate the backend build
    against the checkpoint manifest.
    """

    loaded = _load_json_artifact(path, expected_digest, _TASK_OUTPUT_SCHEMA, "task output")
    value = loaded.manifest
    candidates = set(dataset.manifest["candidate_set"]["ordered_record_ids"])
    if value["checkpoint_digest"] != checkpoint.digest:
        raise ValueError("task output checkpoint digest mismatch")
    if value["record_order_digest"] != dataset.manifest["candidate_set"]["digest"]:
        raise ValueError("task output record-order digest mismatch")
    if value["backend_digest"] != checkpoint.manifest["build"]["backend_build_digest"]:
        raise ValueError("task output backend build digest mismatch")
    if value["initial_state_digest"] != checkpoint.manifest["arrays"]["probe_initial_state_digest"]:
        raise ValueError("task output initial-state digest mismatch")
    ranked = cast(list[str], value["ranked_record_ids"])
    predicted = cast(str | None, value["predicted_record_id"])
    if any(record_id not in candidates for record_id in ranked):
        raise ValueError("task output contains an off-candidate rank")
    if predicted is not None and predicted not in candidates:
        raise ValueError("task output prediction is off-candidate")
    if len(value["scores"]) != len(ranked):
        raise ValueError("task output score/rank lengths differ")
    if predicted is not None and (not ranked or ranked[0] != predicted):
        raise ValueError("prediction must equal the first ranked record")
    return loaded


def read_and_validate_evaluation_v2(
    path: Path, expected_digest: str, dataset: ValidatedDatasetV2
) -> ValidatedArtifactV2:
    """Authenticate evaluation output and derive completeness fail-closed."""

    loaded = _load_json_artifact(path, expected_digest, _EVALUATION_SCHEMA, "evaluation")
    value = loaded.manifest
    if value["dataset_digest"] != dataset.digest:
        raise ValueError("evaluation dataset digest mismatch")
    if value["experiment_digest"] != dataset.manifest["experiment_digest"]:
        raise ValueError("evaluation experiment digest mismatch")
    if value["ordered_record_ids"] != dataset.manifest["candidate_set"]["ordered_record_ids"]:
        raise ValueError("evaluation record order mismatch")
    if value["candidate_set_digest"] != dataset.manifest["candidate_set"]["digest"]:
        raise ValueError("evaluation candidate-set digest mismatch")

    expected = set(cast(list[str], value["expected_tasks"]))
    completed = set(cast(list[str], value["completed_tasks"]))
    missing = set(cast(list[str], value["missing_tasks"]))
    unexpected = set(cast(list[str], value["unexpected_tasks"]))
    failures = set(cast(list[str], value["digest_failures"]))
    if value["expected_task_count"] != len(expected):
        raise ValueError("expected task count mismatch")
    if value["completed_task_count"] != len(completed):
        raise ValueError("completed task count mismatch")
    if value["expected_task_set_digest"] != task_set_digest(list(expected)):
        raise ValueError("expected task-set digest mismatch")
    if value["expected_task_set_digest"] != dataset.manifest["artifacts"]["task_set"]["sha256"]:
        raise ValueError("evaluation expected tasks do not match the dataset task set")
    if value["completed_task_set_digest"] != task_set_digest(list(completed)):
        raise ValueError("completed task-set digest mismatch")
    if missing != expected - completed:
        raise ValueError("missing task set is contradictory")
    if unexpected != completed - expected:
        raise ValueError("unexpected task set is contradictory")
    if not failures <= expected | completed:
        raise ValueError("digest failures reference unknown tasks")

    complete = expected == completed and not missing and not unexpected and not failures
    gates = value["gates"]
    if gates["completeness_pass"] != complete:
        raise ValueError("reported completeness contradicts authenticated task sets")
    if not complete and any(
        gates[name]
        for name in (
            "g0_fidelity_pass",
            "g1_controlled_completion_pass",
            "g2_locked_recall_pass",
            "g3_mechanism_pass",
            "heldout_utility_pass",
        )
    ):
        raise ValueError("an incomplete evaluation cannot pass a scientific gate")
    return loaded
