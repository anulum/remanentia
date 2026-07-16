# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Model-free CLI surface tests

"""In-process tests for the model-free CLI: inspect, verify-manifest and dispatch.

No embedding model is required. The model-command dispatch is covered by driving a
model command that fails closed at the missing encoder checkpoint, before any model
load, so the dispatch line is measured without the pinned model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from snn_memory import cli
from snn_memory.checkpoint import save_checkpoint
from snn_memory.contracts import ModelConfig
from snn_memory.state import initialise_weights


def _checkpoint(path: Path) -> dict[str, object]:
    model = ModelConfig(n_neurons=8, connectivity=0.5)
    weights, topology = initialise_weights(model, 11)
    return save_checkpoint(
        path,
        weights,
        topology,
        np.zeros((2, 8), dtype=np.float64),
        ["alpha", "beta"],
        model,
        {
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
        },
        [
            {"epoch": 0, "label": "alpha", "timesteps": 1},
            {"epoch": 0, "label": "beta", "timesteps": 1},
        ],
    )


def _model_command_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "a.txt"
    source.write_text("alpha content")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": {"n_neurons": 24, "connectivity": 0.25},
                "encoder": {"feature_dim": 24, "packet_ms": 5, "silent_ms": 1},
                "train": {"seed": 11, "epochs": 1, "input_current": 18.0},
            }
        )
    )
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder_checkpoint": str(tmp_path / "no_such_model"),
                "encoder_digest": "0" * 64,
                "entries": [
                    {
                        "label": "a",
                        "path": "a.txt",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    return config, corpus


def test_inspect_prints_the_checkpoint_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _checkpoint(tmp_path)
    assert cli.main(["inspect", "--checkpoint", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema_version"] == 1
    assert printed["labels"] == ["alpha", "beta"]


def test_verify_manifest_prints_the_array_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _checkpoint(tmp_path)
    assert cli.main(["verify-manifest", "--run-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == manifest["array_digest"]


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_main_dispatches_a_model_command_and_fails_closed_without_the_model(tmp_path: Path) -> None:
    config, corpus = _model_command_inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        cli.main(
            [
                "train",
                "--config",
                str(config),
                "--corpus-manifest",
                str(corpus),
                "--output",
                str(tmp_path / "out"),
            ]
        )
