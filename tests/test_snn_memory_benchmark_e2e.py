# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN benchmark process tests

"""Fresh-process matched-control benchmark over tracked Markdown."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".snn_models" / "all-MiniLM-L6-v2"


@pytest.mark.skipif(  # type: ignore[untyped-decorator] # Pytest decorator.
    not MODEL.is_dir(), reason="pinned local encoder not provisioned"
)
def test_benchmark_runs_each_matched_control_in_fresh_process(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": {"n_neurons": 16, "connectivity": 0.25},
                "encoder": {"feature_dim": 16, "packet_ms": 4, "silent_ms": 1},
                "train": {"seed": 11, "epochs": 1, "input_current": 18.0},
            }
        )
    )
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
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
    checkpoint = tmp_path / "checkpoint"
    report = tmp_path / "report.json"
    base = [sys.executable, "-m", "snn_memory.cli"]
    subprocess.run(
        base + ["train", "--config", str(config), "--corpus-manifest", str(corpus), "--output", str(checkpoint)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        base
        + [
            "benchmark",
            "--checkpoint",
            str(checkpoint),
            "--corpus-manifest",
            str(corpus),
            "--output",
            str(report),
            "--seeds",
            "11,29",
            "--completion-steps",
            "6",
        ],
        cwd=ROOT,
        check=True,
    )
    result = json.loads(report.read_text())
    schema = json.loads((ROOT / "docs/schema/snn_memory_result.schema.json").read_text())
    Draft202012Validator(schema).validate(result)
    assert set(result["conditions"]) == {
        "trained",
        "shuffled",
        "random",
        "zero",
        "encoder-only",
    }
    assert result["seeds"] == [11, 29]
    assert result["gates"]["g2_pass"] is False
