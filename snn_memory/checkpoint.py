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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from file_utils import atomic_write_bytes, atomic_write_json, atomic_write_text
from snn_memory.contracts import ModelConfig
from snn_memory.state import BoolArray, FloatArray, validate_weights

_ARRAY_KEYS = {"weights", "topology", "signatures", "labels"}
_METADATA_KEYS = {"seed", "epochs_completed", "encoder", "encoder_digest", "python"}


@dataclass(frozen=True)
class Checkpoint:
    """Validated frozen recurrent memory and calibration signatures."""

    weights: FloatArray
    topology: BoolArray
    signatures: FloatArray
    labels: tuple[str, ...]
    model: ModelConfig
    manifest: dict[str, Any]


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
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "orientation": "row-pre-column-post",
        "plasticity": "online-pair-stdp-e-to-e",
        "array_digest": digest,
        "model": model.to_dict(),
        "labels": labels,
        "metadata": metadata,
    }
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target / "checkpoint.npz", payload.getvalue())
    atomic_write_json(target / "manifest.json", manifest, indent=2, sort_keys=True)
    event_text = "".join(json.dumps(event, sort_keys=True) + "\n" for event in training_events)
    atomic_write_text(target / "training_events.jsonl", event_text)
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
        "model",
        "labels",
        "metadata",
    }
    if set(manifest_data) != required or manifest_data["schema_version"] != 1:
        raise ValueError("manifest fields or schema version do not match")
    if manifest_data["orientation"] != "row-pre-column-post":
        raise ValueError("unsupported matrix orientation")
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
    weights = np.asarray(arrays["weights"], dtype=np.float64)
    topology = np.asarray(arrays["topology"], dtype=np.bool_)
    signatures = np.asarray(arrays["signatures"], dtype=np.float64)
    labels = tuple(str(value) for value in arrays["labels"].tolist())
    validate_weights(weights, topology, model)
    if labels != tuple(manifest_data["labels"]):
        raise ValueError("checkpoint labels differ from manifest")
    if signatures.ndim != 2 or signatures.shape[0] != len(labels):
        raise ValueError("checkpoint signatures have an invalid shape")
    if not np.all(np.isfinite(signatures)):
        raise ValueError("checkpoint signatures contain non-finite values")
    return Checkpoint(weights, topology, signatures, labels, model, manifest_data)


def verify_run_directory(directory: Path | str) -> str:
    """Validate a bundle and return its deterministic array digest."""
    return str(load_checkpoint(directory).manifest["array_digest"])


def _validate_metadata(metadata: dict[str, Any]) -> None:
    """Enforce the checkpoint schema's reproducibility metadata contract."""
    if set(metadata) != _METADATA_KEYS:
        raise ValueError("checkpoint metadata fields do not match the schema")
    if not isinstance(metadata["seed"], int) or not isinstance(metadata["epochs_completed"], int):
        raise ValueError("checkpoint seed and completed epochs must be integers")
    if metadata["seed"] < 0 or metadata["epochs_completed"] < 1:
        raise ValueError("checkpoint seed and completed epochs are outside valid bounds")
    if not isinstance(metadata["encoder"], dict) or not isinstance(metadata["python"], str):
        raise ValueError("checkpoint encoder and Python version metadata are invalid")
    digest = metadata["encoder_digest"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("checkpoint encoder digest must be SHA-256 hex")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError("checkpoint encoder digest must be SHA-256 hex") from exc
