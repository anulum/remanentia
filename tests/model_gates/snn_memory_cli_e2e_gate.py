# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN process-boundary tests

"""Fresh source-tree module-process train/checkpoint/probe real-surface tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from snn_memory.checkpoint import load_checkpoint
from tests.model_gates.model_precondition import MODEL, ROOT, require_pinned_model

PINNED_DIGEST = require_pinned_model()


def _write_inputs(tmp_path: Path, epochs: int) -> tuple[Path, Path]:
    config = tmp_path / f"config-{epochs}.json"
    config.write_text(
        json.dumps(
            {
                "model": {"n_neurons": 24, "connectivity": 0.25},
                "encoder": {"feature_dim": 24, "packet_ms": 5, "silent_ms": 1},
                "train": {"seed": 11, "epochs": epochs, "input_current": 18.0},
            }
        )
    )
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder_checkpoint": str(MODEL),
                "encoder_digest": PINNED_DIGEST,
                "entries": [
                    {
                        "label": "snn",
                        "path": str(ROOT / "docs/research/snn_consolidation.md"),
                        "sha256": hashlib.sha256(
                            (ROOT / "docs/research/snn_consolidation.md").read_bytes()
                        ).hexdigest(),
                    },
                    {
                        "label": "retrieval",
                        "path": str(ROOT / "docs/adr/0004-dual-retrieval-stacks.md"),
                        "sha256": hashlib.sha256(
                            (ROOT / "docs/adr/0004-dual-retrieval-stacks.md").read_bytes()
                        ).hexdigest(),
                    },
                ],
            }
        )
    )
    return config, manifest


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "snn_memory.cli", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_fresh_process_probe_accepts_cue_but_no_corpus_argument(tmp_path: Path) -> None:
    config, manifest = _write_inputs(tmp_path, 1)
    checkpoint = tmp_path / "checkpoint"
    result_path = tmp_path / "probe.json"
    _run(
        "train",
        "--config",
        str(config),
        "--corpus-manifest",
        str(manifest),
        "--output",
        str(checkpoint),
    )
    _run(
        "probe",
        "--checkpoint",
        str(checkpoint),
        "--cue",
        str(ROOT / "docs/research/snn_consolidation.md"),
        "--encoder-checkpoint",
        str(MODEL),
        "--output",
        str(result_path),
        "--completion-steps",
        "8",
    )
    result = json.loads(result_path.read_text())
    schema = json.loads((ROOT / "docs/schema/snn_memory_result.schema.json").read_text())
    importlib.import_module("jsonschema").Draft202012Validator(schema).validate(result)
    assert result["label"] is None or result["label"] in {"snn", "retrieval"}
    assert result["weight_digest_unchanged"] is True
    help_text = _run("probe", "--help").stdout
    assert "--corpus" not in help_text


def test_resumed_training_matches_uninterrupted_weight_digest(tmp_path: Path) -> None:
    config_one, manifest = _write_inputs(tmp_path, 1)
    config_three, _ = _write_inputs(tmp_path, 3)
    first = tmp_path / "first"
    resumed = tmp_path / "resumed"
    direct = tmp_path / "direct"
    _run(
        "train",
        "--config",
        str(config_one),
        "--corpus-manifest",
        str(manifest),
        "--output",
        str(first),
    )
    _run(
        "train",
        "--config",
        str(config_three),
        "--corpus-manifest",
        str(manifest),
        "--resume-checkpoint",
        str(first),
        "--output",
        str(resumed),
    )
    _run(
        "train",
        "--config",
        str(config_three),
        "--corpus-manifest",
        str(manifest),
        "--output",
        str(direct),
    )
    resumed_checkpoint = load_checkpoint(resumed)
    direct_checkpoint = load_checkpoint(direct)
    assert resumed_checkpoint.manifest["array_digest"] == direct_checkpoint.manifest["array_digest"]
    assert resumed_checkpoint.manifest["event_digest"] == direct_checkpoint.manifest["event_digest"]
    assert resumed_checkpoint.training_events == direct_checkpoint.training_events


def test_cli_rejects_non_forward_resume_target(tmp_path: Path) -> None:
    config_one, manifest = _write_inputs(tmp_path, 1)
    first = tmp_path / "first"
    _run(
        "train",
        "--config",
        str(config_one),
        "--corpus-manifest",
        str(manifest),
        "--output",
        str(first),
    )
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _run(
            "train",
            "--config",
            str(config_one),
            "--corpus-manifest",
            str(manifest),
            "--resume-checkpoint",
            str(first),
            "--output",
            str(tmp_path / "invalid"),
        )
    assert "resume target epochs must exceed completed epochs" in raised.value.stderr
