# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Safe temporal SNN checkpoint bundle

"""Atomic NPZ checkpoints with strict JSON manifests and byte digests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from file_utils import atomic_write_bytes, atomic_write_json, atomic_write_text, fsync_directory
from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.state import BoolArray, FloatArray, validate_weights

_ARRAY_KEYS = {"weights", "topology", "signatures", "labels"}
_METADATA_KEYS = {
    "seed",
    "epochs_completed",
    "input_current",
    "encoder",
    "encoder_digest",
    "corpus_digest",
    "python",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ENCODER_KEYS = {"feature_dim", "packet_ms", "silent_ms", "active_fraction", "projection_seed"}


@dataclass(frozen=True)
class Checkpoint:
    """Validated frozen recurrent memory and calibration signatures."""

    weights: FloatArray
    topology: BoolArray
    signatures: FloatArray
    labels: tuple[str, ...]
    model: ModelConfig
    manifest: dict[str, Any]
    training_events: tuple[dict[str, Any], ...] = ()


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    """Hash array names, dtypes, shapes and contiguous bytes deterministically."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_checkpoint(
    directory: Path | str,
    weights: FloatArray,
    topology: BoolArray,
    signatures: FloatArray,
    labels: list[str],
    model: ModelConfig,
    metadata: dict[str, Any],
    training_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically write one safe checkpoint bundle and return its manifest."""
    target = Path(directory)
    _validate_metadata(metadata)
    validate_weights(weights, topology, model)
    _validate_labels(labels)
    if signatures.ndim != 2 or signatures.shape[0] != len(labels):
        raise ValueError("one two-dimensional signature row is required per label")
    arrays = {
        "weights": np.asarray(weights, dtype=np.float64),
        "topology": np.asarray(topology, dtype=np.bool_),
        "signatures": np.asarray(signatures, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.str_),
    }
    if not np.all(np.isfinite(signatures)):
        raise ValueError("signatures contain non-finite values")
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        weights=arrays["weights"],
        topology=arrays["topology"],
        signatures=arrays["signatures"],
        labels=arrays["labels"],
    )
    digest = array_digest(arrays)
    event_text = "".join(json.dumps(event, sort_keys=True) + "\n" for event in training_events)
    _validate_training_events(training_events, labels, int(metadata["epochs_completed"]))
    event_digest = hashlib.sha256(event_text.encode("utf-8")).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "orientation": "row-pre-column-post",
        "plasticity": "online-pair-stdp-e-to-e",
        "array_digest": digest,
        "event_digest": event_digest,
        "model": model.to_dict(),
        "labels": labels,
        "metadata": metadata,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_dir():
            raise ValueError("checkpoint target must be a directory path")
        if any(target.iterdir()):
            raise ValueError("checkpoint target must not contain an existing bundle")
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.staging-", dir=target.parent
    ) as staging_name:
        staging = Path(staging_name)
        atomic_write_bytes(staging / "checkpoint.npz", payload.getvalue())
        atomic_write_json(staging / "manifest.json", manifest, indent=2, sort_keys=True)
        atomic_write_text(staging / "training_events.jsonl", event_text)
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
        fsync_directory(target.parent)
    return manifest


def load_checkpoint(directory: Path | str) -> Checkpoint:
    """Load a checkpoint while rejecting contract, shape and digest drift."""
    target = Path(directory)
    manifest_data = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest root must be an object")
    required = {
        "schema_version",
        "orientation",
        "plasticity",
        "array_digest",
        "event_digest",
        "model",
        "labels",
        "metadata",
    }
    schema_version = manifest_data.get("schema_version")
    if (
        set(manifest_data) != required
        or not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError("manifest fields or schema version do not match")
    if manifest_data["orientation"] != "row-pre-column-post":
        raise ValueError("unsupported matrix orientation")
    if manifest_data["plasticity"] != "online-pair-stdp-e-to-e":
        raise ValueError("unsupported checkpoint plasticity contract")
    event_digest = manifest_data["event_digest"]
    if not isinstance(event_digest, str) or not _SHA256_PATTERN.fullmatch(event_digest):
        raise ValueError("checkpoint event digest must be lowercase SHA-256 hex")
    with np.load(target / "checkpoint.npz", allow_pickle=False) as archive:
        if set(archive.files) != _ARRAY_KEYS:
            raise ValueError("checkpoint contains missing or extra arrays")
        arrays = {name: archive[name] for name in archive.files}
    if array_digest(arrays) != manifest_data["array_digest"]:
        raise ValueError("checkpoint array digest mismatch")
    model_raw = manifest_data["model"]
    if not isinstance(model_raw, dict):
        raise ValueError("model manifest entry must be an object")
    model = ModelConfig(**model_raw)
    metadata = manifest_data["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    _validate_metadata(metadata)
    if (
        arrays["weights"].dtype != np.float64
        or arrays["signatures"].dtype != np.float64
        or arrays["topology"].dtype != np.bool_
        or arrays["labels"].dtype.kind != "U"
    ):
        raise ValueError("checkpoint arrays have an unexpected dtype")
    weights = np.asarray(arrays["weights"], dtype=np.float64)
    topology = np.asarray(arrays["topology"], dtype=np.bool_)
    signatures = np.asarray(arrays["signatures"], dtype=np.float64)
    labels = tuple(str(value) for value in arrays["labels"].tolist())
    manifest_labels = manifest_data["labels"]
    if not isinstance(manifest_labels, list):
        raise ValueError("checkpoint labels manifest entry must be an array")
    _validate_labels(manifest_labels)
    validate_weights(weights, topology, model)
    if labels != tuple(manifest_labels):
        raise ValueError("checkpoint labels differ from manifest")
    if signatures.ndim != 2 or signatures.shape[0] != len(labels):
        raise ValueError("checkpoint signatures have an invalid shape")
    if not np.all(np.isfinite(signatures)):
        raise ValueError("checkpoint signatures contain non-finite values")
    event_bytes = (target / "training_events.jsonl").read_bytes()
    if hashlib.sha256(event_bytes).hexdigest() != event_digest:
        raise ValueError("checkpoint training event digest mismatch")
    try:
        events_raw = [json.loads(line) for line in event_bytes.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint training events are invalid JSONL") from exc
    _validate_training_events(events_raw, list(labels), int(metadata["epochs_completed"]))
    return Checkpoint(
        weights, topology, signatures, labels, model, manifest_data, tuple(events_raw)
    )


def verify_run_directory(directory: Path | str) -> str:
    """Validate a bundle and return its deterministic array digest."""
    return str(load_checkpoint(directory).manifest["array_digest"])


def _validate_metadata(metadata: dict[str, Any]) -> None:
    """Enforce the checkpoint schema's reproducibility metadata contract."""
    if set(metadata) != _METADATA_KEYS:
        raise ValueError("checkpoint metadata fields do not match the schema")
    if (
        not isinstance(metadata["seed"], int)
        or isinstance(metadata["seed"], bool)
        or not isinstance(metadata["epochs_completed"], int)
        or isinstance(metadata["epochs_completed"], bool)
    ):
        raise ValueError("checkpoint seed and completed epochs must be integers")
    if not 0 <= metadata["seed"] <= 4_294_967_295 or metadata["epochs_completed"] < 1:
        raise ValueError("checkpoint seed and completed epochs are outside valid bounds")
    encoder = metadata["encoder"]
    if not isinstance(encoder, dict) or set(encoder) != _ENCODER_KEYS:
        raise ValueError("checkpoint encoder and Python version metadata are invalid")
    integer_encoder_fields = ("feature_dim", "packet_ms", "silent_ms", "projection_seed")
    if any(
        not isinstance(encoder[key], int) or isinstance(encoder[key], bool)
        for key in integer_encoder_fields
    ):
        raise ValueError("checkpoint encoder and Python version metadata are invalid")
    active_fraction = encoder["active_fraction"]
    if (
        not isinstance(active_fraction, (int, float))
        or isinstance(active_fraction, bool)
        or not np.isfinite(active_fraction)
    ):
        raise ValueError("checkpoint encoder and Python version metadata are invalid")
    try:
        EncoderConfig(**encoder)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint encoder and Python version metadata are invalid") from exc
    python_version = metadata["python"]
    if (
        not isinstance(python_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", python_version) is None
    ):
        raise ValueError("checkpoint encoder and Python version metadata are invalid")
    input_current = metadata["input_current"]
    if not isinstance(input_current, (int, float)) or isinstance(input_current, bool):
        raise ValueError("checkpoint input current must be a positive finite number")
    if not np.isfinite(input_current) or input_current <= 0.0:
        raise ValueError("checkpoint input current must be a positive finite number")
    for key in ("encoder_digest", "corpus_digest"):
        digest = metadata[key]
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"checkpoint {key.replace('_', ' ')} must be lowercase SHA-256 hex")


def _validate_labels(labels: list[str]) -> None:
    """Require the runtime label contract declared in the checkpoint schema."""
    if not labels or any(
        not isinstance(label, str) or not label.strip() or label != label.strip()
        for label in labels
    ):
        raise ValueError(
            "checkpoint labels must be non-empty strings without surrounding whitespace"
        )
    if len(set(labels)) != len(labels):
        raise ValueError("checkpoint labels must be unique")


def _validate_training_events(
    events: list[dict[str, Any]], labels: list[str], epochs_completed: int
) -> None:
    """Require one authenticated positive-timestep event per epoch and label."""
    expected = {(epoch, label) for epoch in range(epochs_completed) for label in labels}
    observed: set[tuple[int, str]] = set()
    for event in events:
        if not isinstance(event, dict) or set(event) != {"epoch", "label", "timesteps"}:
            raise ValueError("checkpoint training event fields are invalid")
        epoch, label, timesteps = event["epoch"], event["label"], event["timesteps"]
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or not isinstance(label, str)
            or not isinstance(timesteps, int)
            or isinstance(timesteps, bool)
            or timesteps < 1
        ):
            raise ValueError("checkpoint training event values are invalid")
        key = (epoch, label)
        if key not in expected or key in observed:
            raise ValueError("checkpoint training events do not match epochs and labels")
        observed.add(key)
    if observed != expected:
        raise ValueError("checkpoint training events do not cover every epoch and label")
