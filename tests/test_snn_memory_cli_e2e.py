# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN process-boundary tests

"""Installed CLI train-to-checkpoint-to-fresh-probe real-surface tests."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from snn_memory.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".snn_models" / "all-MiniLM-L6-v2"


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
                "encoder_checkpoint": str(MODEL),
                "entries": [
                    {
                        "label": "snn",
                        "path": str(ROOT / "docs/research/snn_consolidation.md"),
                        "sha256": hashlib.sha256((ROOT / "docs/research/snn_consolidation.md").read_bytes()).hexdigest(),
                    },
                    {
                        "label": "retrieval",
                        "path": str(ROOT / "docs/adr/0004-dual-retrieval-stacks.md"),
                        "sha256": hashlib.sha256((ROOT / "docs/adr/0004-dual-retrieval-stacks.md").read_bytes()).hexdigest(),
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


@pytest.mark.skipif(  # type: ignore[untyped-decorator] # Pytest decorator.
    not MODEL.is_dir(), reason="pinned local encoder not provisioned"
)
def test_fresh_process_probe_accepts_cue_but_no_corpus_argument(tmp_path: Path) -> None:
    config, manifest = _write_inputs(tmp_path, 1)
    checkpoint = tmp_path / "checkpoint"
    result_path = tmp_path / "probe.json"
    _run("train", "--config", str(config), "--corpus-manifest", str(manifest), "--output", str(checkpoint))
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
    Draft202012Validator(schema).validate(result)
    assert result["label"] is None or result["label"] in {"snn", "retrieval"}
    assert result["weight_digest_unchanged"] is True
    help_text = _run("probe", "--help").stdout
    assert "--corpus" not in help_text


@pytest.mark.skipif(  # type: ignore[untyped-decorator] # Pytest decorator.
    not MODEL.is_dir(), reason="pinned local encoder not provisioned"
)
def test_resumed_training_matches_uninterrupted_weight_digest(tmp_path: Path) -> None:
    config_one, manifest = _write_inputs(tmp_path, 1)
    config_three, _ = _write_inputs(tmp_path, 3)
    first = tmp_path / "first"
    resumed = tmp_path / "resumed"
    direct = tmp_path / "direct"
    _run("train", "--config", str(config_one), "--corpus-manifest", str(manifest), "--output", str(first))
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
    _run("train", "--config", str(config_three), "--corpus-manifest", str(manifest), "--output", str(direct))
    assert load_checkpoint(resumed).manifest["array_digest"] == load_checkpoint(direct).manifest["array_digest"]
