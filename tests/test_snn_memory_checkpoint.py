# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN checkpoint tests

"""Real NPZ/JSON bundle round-trip, metadata and corruption rejection tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from snn_memory.checkpoint import (
    array_digest,
    load_checkpoint,
    save_checkpoint,
    verify_run_directory,
)
from snn_memory.contracts import ModelConfig
from snn_memory.state import initialise_weights

SCHEMA = Path(__file__).parents[1] / "docs/schema/snn_memory_checkpoint.schema.json"


def _metadata() -> dict[str, Any]:
    return {
        "seed": 11,
        "epochs_completed": 1,
        "encoder": {
            "feature_dim": 16,
            "packet_ms": 4,
            "silent_ms": 1,
            "active_fraction": 0.05,
            "projection_seed": 1729,
        },
        "encoder_digest": "0" * 64,
        "python": "3.12.3",
    }


def _save(
    path: Path,
    *,
    signatures: np.ndarray | None = None,
    labels: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = ModelConfig(n_neurons=8, connectivity=0.5)
    weights, topology = initialise_weights(model, 11)
    return save_checkpoint(
        path,
        weights,
        topology,
        np.arange(16, dtype=np.float64).reshape(2, 8) if signatures is None else signatures,
        ["alpha", "beta"] if labels is None else labels,
        model,
        _metadata() if metadata is None else metadata,
        [{"epoch": 0, "label": "alpha"}],
    )


def _craft(path: Path, signatures: np.ndarray, labels: list[str]) -> None:
    model = ModelConfig(n_neurons=8, connectivity=0.5)
    weights, topology = initialise_weights(model, 11)
    arrays = {
        "weights": weights.astype(np.float64),
        "topology": topology.astype(np.bool_),
        "signatures": signatures.astype(np.float64),
        "labels": np.asarray(labels, dtype=np.str_),
    }
    np.savez_compressed(path / "checkpoint.npz", **arrays)
    manifest = {
        "schema_version": 1,
        "orientation": "row-pre-column-post",
        "plasticity": "online-pair-stdp-e-to-e",
        "array_digest": array_digest(arrays),
        "model": model.to_dict(),
        "labels": labels,
        "metadata": _metadata(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def _tamper(path: Path, **changes: Any) -> None:
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(changes)
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_checkpoint_round_trip_uses_safe_real_files(tmp_path: Path) -> None:
    manifest = _save(tmp_path)
    loaded = load_checkpoint(tmp_path)
    assert loaded.labels == ("alpha", "beta")
    assert verify_run_directory(tmp_path) == manifest["array_digest"]
    assert (tmp_path / "training_events.jsonl").read_text().endswith("\n")
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(loaded.manifest)


def test_checkpoint_rejects_manifest_digest_tampering(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, array_digest="0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_checkpoint(tmp_path)


def test_checkpoint_rejects_extra_npz_array(tmp_path: Path) -> None:
    _save(tmp_path)
    with np.load(tmp_path / "checkpoint.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    np.savez_compressed(tmp_path / "checkpoint.npz", **arrays, surprise=np.ones(1))
    with pytest.raises(ValueError, match="missing or extra"):
        load_checkpoint(tmp_path)


def test_save_rejects_signature_row_count_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one two-dimensional signature row"):
        _save(tmp_path, signatures=np.zeros((3, 8)))


def test_save_rejects_non_finite_signatures(tmp_path: Path) -> None:
    signatures = np.zeros((2, 8))
    signatures[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _save(tmp_path, signatures=signatures)


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"seed": 11}, "metadata fields do not match"),
        (
            {"seed": "x", "epochs_completed": 1, "encoder": {}, "encoder_digest": "0" * 64, "python": "3"},
            "must be integers",
        ),
        (
            {"seed": -1, "epochs_completed": 1, "encoder": {}, "encoder_digest": "0" * 64, "python": "3"},
            "outside valid bounds",
        ),
        (
            {"seed": 1, "epochs_completed": 1, "encoder": [], "encoder_digest": "0" * 64, "python": "3"},
            "encoder and Python version",
        ),
        (
            {"seed": 1, "epochs_completed": 1, "encoder": {}, "encoder_digest": "abc", "python": "3"},
            "SHA-256 hex",
        ),
        (
            {"seed": 1, "epochs_completed": 1, "encoder": {}, "encoder_digest": "z" * 64, "python": "3"},
            "SHA-256 hex",
        ),
    ],
)
def test_save_rejects_invalid_metadata(tmp_path: Path, metadata: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _save(tmp_path, metadata=metadata)


def test_load_rejects_non_object_manifest(tmp_path: Path) -> None:
    _save(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ValueError, match="manifest root must be an object"):
        load_checkpoint(tmp_path)


def test_load_rejects_missing_manifest_field(tmp_path: Path) -> None:
    _save(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["plasticity"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fields or schema version"):
        load_checkpoint(tmp_path)


def test_load_rejects_unsupported_orientation(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, orientation="column-pre-row-post")
    with pytest.raises(ValueError, match="unsupported matrix orientation"):
        load_checkpoint(tmp_path)


def test_load_rejects_non_object_model_entry(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, model="not-a-model")
    with pytest.raises(ValueError, match="model manifest entry must be an object"):
        load_checkpoint(tmp_path)


def test_load_rejects_non_object_metadata(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, metadata="not-a-dict")
    with pytest.raises(ValueError, match="metadata must be an object"):
        load_checkpoint(tmp_path)


def test_load_rejects_labels_that_differ_from_arrays(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, labels=["other", "names"])
    with pytest.raises(ValueError, match="labels differ from manifest"):
        load_checkpoint(tmp_path)


def test_load_rejects_signature_shape_inconsistent_with_labels(tmp_path: Path) -> None:
    _craft(tmp_path, np.zeros((3, 8)), ["alpha", "beta"])
    with pytest.raises(ValueError, match="signatures have an invalid shape"):
        load_checkpoint(tmp_path)


def test_load_rejects_non_finite_signatures(tmp_path: Path) -> None:
    signatures = np.zeros((2, 8))
    signatures[1, 1] = np.nan
    _craft(tmp_path, signatures, ["alpha", "beta"])
    with pytest.raises(ValueError, match="signatures contain non-finite"):
        load_checkpoint(tmp_path)
