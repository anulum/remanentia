# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN checkpoint tests

"""Real NPZ/JSON bundle round-trip, metadata and corruption rejection tests."""

from __future__ import annotations

import hashlib
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
        "input_current": 18.0,
        "encoder": {
            "feature_dim": 16,
            "packet_ms": 4,
            "silent_ms": 1,
            "active_fraction": 0.05,
            "projection_seed": 1729,
        },
        "encoder_digest": "0" * 64,
        "corpus_digest": "0" * 64,
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
        [
            {"epoch": 0, "label": "alpha", "timesteps": 1},
            {"epoch": 0, "label": "beta", "timesteps": 1},
        ],
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
    events = [{"epoch": 0, "label": label, "timesteps": 1} for label in labels]
    event_text = "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
    manifest = {
        "schema_version": 1,
        "orientation": "row-pre-column-post",
        "plasticity": "online-pair-stdp-e-to-e",
        "array_digest": array_digest(arrays),
        "event_digest": hashlib.sha256(event_text.encode("utf-8")).hexdigest(),
        "model": model.to_dict(),
        "labels": labels,
        "metadata": _metadata(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "training_events.jsonl").write_text(event_text)


def _tamper(path: Path, **changes: Any) -> None:
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(changes)
    (path / "manifest.json").write_text(json.dumps(manifest))


def _replace_events(path: Path, content: bytes) -> None:
    """Replace and correctly redigest an event log for structural corruption tests."""
    (path / "training_events.jsonl").write_bytes(content)
    _tamper(path, event_digest=hashlib.sha256(content).hexdigest())


def test_checkpoint_round_trip_uses_safe_real_files(tmp_path: Path) -> None:
    manifest = _save(tmp_path)
    loaded = load_checkpoint(tmp_path)
    assert loaded.labels == ("alpha", "beta")
    assert verify_run_directory(tmp_path) == manifest["array_digest"]
    assert (tmp_path / "training_events.jsonl").read_text().endswith("\n")
    assert loaded.training_events == (
        {"epoch": 0, "label": "alpha", "timesteps": 1},
        {"epoch": 0, "label": "beta", "timesteps": 1},
    )
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(loaded.manifest)


def test_save_publishes_a_complete_bundle_to_an_absent_target(tmp_path: Path) -> None:
    target = tmp_path / "new-checkpoint"
    _save(target)
    assert load_checkpoint(target).labels == ("alpha", "beta")
    assert not list(tmp_path.glob(".new-checkpoint.staging-*"))


def test_save_rejects_a_target_file_without_replacing_it(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint"
    target.write_text("owner data", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory path"):
        _save(target)
    assert target.read_text(encoding="utf-8") == "owner data"


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


@pytest.mark.parametrize("labels", [[], ["alpha", "alpha"], [" alpha"], ["alpha "]])
def test_save_rejects_invalid_labels(tmp_path: Path, labels: list[str]) -> None:
    signatures = np.zeros((len(labels), 8), dtype=np.float64)
    with pytest.raises(ValueError, match="labels"):
        _save(tmp_path, signatures=signatures, labels=labels)


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"seed": 11}, "metadata fields do not match"),
        (
            {
                "seed": "x",
                "epochs_completed": 1,
                "input_current": 18.0,
                "encoder": _metadata()["encoder"],
                "encoder_digest": "0" * 64,
                "corpus_digest": "0" * 64,
                "python": "3.12",
            },
            "must be integers",
        ),
        (
            {
                "seed": -1,
                "epochs_completed": 1,
                "input_current": 18.0,
                "encoder": _metadata()["encoder"],
                "encoder_digest": "0" * 64,
                "corpus_digest": "0" * 64,
                "python": "3.12",
            },
            "outside valid bounds",
        ),
        (
            {
                "seed": 1,
                "epochs_completed": 1,
                "input_current": 18.0,
                "encoder": [],
                "encoder_digest": "0" * 64,
                "corpus_digest": "0" * 64,
                "python": "3.12",
            },
            "encoder and Python version",
        ),
        (
            {
                "seed": 1,
                "epochs_completed": 1,
                "input_current": 18.0,
                "encoder": _metadata()["encoder"],
                "encoder_digest": "abc",
                "corpus_digest": "0" * 64,
                "python": "3.12",
            },
            "SHA-256 hex",
        ),
        (
            {
                "seed": 1,
                "epochs_completed": 1,
                "input_current": 18.0,
                "encoder": _metadata()["encoder"],
                "encoder_digest": "z" * 64,
                "corpus_digest": "0" * 64,
                "python": "3.12",
            },
            "SHA-256 hex",
        ),
        (
            {
                "seed": 1,
                "epochs_completed": 1,
                "input_current": float("nan"),
                "encoder": _metadata()["encoder"],
                "encoder_digest": "0" * 64,
                "corpus_digest": "0" * 64,
                "python": "3.12",
            },
            "positive finite number",
        ),
    ],
)
def test_save_rejects_invalid_metadata(
    tmp_path: Path, metadata: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _save(tmp_path, metadata=metadata)


def test_save_rejects_schema_drift_metadata_values(tmp_path: Path) -> None:
    invalid: list[tuple[dict[str, Any], str]] = []
    for key, value in (("seed", True), ("epochs_completed", True), ("seed", 4_294_967_296)):
        metadata = _metadata()
        metadata[key] = value
        invalid.append((metadata, "integers|valid bounds"))
    uppercase = _metadata()
    uppercase["encoder_digest"] = "A" * 64
    invalid.append((uppercase, "lowercase SHA-256"))
    arbitrary_python = _metadata()
    arbitrary_python["python"] = "arbitrary"
    invalid.append((arbitrary_python, "Python version"))
    boolean_fraction = _metadata()
    boolean_fraction["encoder"]["active_fraction"] = True
    invalid.append((boolean_fraction, "encoder and Python"))
    for metadata, match in invalid:
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


def test_load_rejects_boolean_schema_version(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, schema_version=True)
    with pytest.raises(ValueError, match="schema version"):
        load_checkpoint(tmp_path)


def test_load_rejects_changed_plasticity_contract(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, plasticity="offline")
    with pytest.raises(ValueError, match="plasticity contract"):
        load_checkpoint(tmp_path)


def test_load_rejects_malformed_event_digest(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, event_digest="A" * 64)
    with pytest.raises(ValueError, match="event digest must be lowercase"):
        load_checkpoint(tmp_path)


def test_load_rejects_scalar_manifest_labels(tmp_path: Path) -> None:
    _save(tmp_path)
    _tamper(tmp_path, labels="alpha")
    with pytest.raises(ValueError, match="labels manifest entry must be an array"):
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


def test_load_rejects_deleted_training_event_log(tmp_path: Path) -> None:
    _save(tmp_path)
    (tmp_path / "training_events.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path)


def test_load_rejects_training_event_digest_tampering(tmp_path: Path) -> None:
    _save(tmp_path)
    with (tmp_path / "training_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="training event digest mismatch"):
        load_checkpoint(tmp_path)


def test_load_rejects_redigested_incomplete_training_history(tmp_path: Path) -> None:
    _save(tmp_path)
    event_path = tmp_path / "training_events.jsonl"
    event_text = json.dumps({"epoch": 0, "label": "alpha", "timesteps": 1}, sort_keys=True) + "\n"
    event_path.write_text(event_text, encoding="utf-8")
    _tamper(tmp_path, event_digest=hashlib.sha256(event_text.encode("utf-8")).hexdigest())
    with pytest.raises(ValueError, match="cover every epoch and label"):
        load_checkpoint(tmp_path)


def test_load_rejects_redigested_invalid_event_encoding_and_json(tmp_path: Path) -> None:
    for content in (b"\xff", b"{not-json}\n"):
        target = tmp_path / hashlib.sha256(content).hexdigest()
        _save(target)
        _replace_events(target, content)
        with pytest.raises(ValueError, match="training events are invalid JSONL"):
            load_checkpoint(target)


def test_load_rejects_redigested_malformed_event_fields_values_and_duplicates(
    tmp_path: Path,
) -> None:
    cases: tuple[tuple[list[dict[str, Any]], str], ...] = (
        ([{}], "fields are invalid"),
        ([{"epoch": True, "label": "alpha", "timesteps": 1}], "values are invalid"),
        (
            [
                {"epoch": 0, "label": "alpha", "timesteps": 1},
                {"epoch": 0, "label": "alpha", "timesteps": 1},
                {"epoch": 0, "label": "beta", "timesteps": 1},
            ],
            "do not match epochs and labels",
        ),
    )
    for index, (events, match) in enumerate(cases):
        target = tmp_path / str(index)
        _save(target)
        content = "".join(json.dumps(event, sort_keys=True) + "\n" for event in events).encode()
        _replace_events(target, content)
        with pytest.raises(ValueError, match=match):
            load_checkpoint(target)


def test_save_rejects_invalid_encoder_and_boolean_input_metadata(tmp_path: Path) -> None:
    invalid_integer = _metadata()
    invalid_integer["encoder"]["feature_dim"] = True
    invalid_contract = _metadata()
    invalid_contract["encoder"]["feature_dim"] = 4
    boolean_current = _metadata()
    boolean_current["input_current"] = True
    cases: tuple[dict[str, Any], ...] = (invalid_integer, invalid_contract, boolean_current)
    for metadata in cases:
        with pytest.raises(ValueError):
            _save(tmp_path, metadata=metadata)


def test_save_refuses_to_replace_an_existing_bundle_or_sentinel(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint"
    target.mkdir()
    sentinel = target / "owner-data"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain an existing bundle"):
        _save(target)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


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


def test_load_rejects_a_correctly_redigested_wrong_dtype_bundle(tmp_path: Path) -> None:
    model = ModelConfig(n_neurons=8, connectivity=0.5)
    weights, topology = initialise_weights(model, 11)
    arrays = {
        "weights": weights.astype(np.float32),
        "topology": topology.astype(np.bool_),
        "signatures": np.zeros((2, 8), dtype=np.float64),
        "labels": np.asarray(["alpha", "beta"], dtype=np.str_),
    }
    np.savez_compressed(tmp_path / "checkpoint.npz", **arrays)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "orientation": "row-pre-column-post",
                "plasticity": "online-pair-stdp-e-to-e",
                "array_digest": array_digest(arrays),
                "event_digest": "0" * 64,
                "model": model.to_dict(),
                "labels": ["alpha", "beta"],
                "metadata": _metadata(),
            }
        )
    )
    with pytest.raises(ValueError, match="unexpected dtype"):
        load_checkpoint(tmp_path)
