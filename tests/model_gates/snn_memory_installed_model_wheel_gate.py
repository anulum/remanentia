# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Exact-wheel pinned-model installed CLI gate

"""Build one wheel and run real pinned-model train/probe outside the checkout."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.model_gates.model_precondition import MODEL, ROOT, require_pinned_model

PINNED_DIGEST = require_pinned_model()


def _run(
    arguments: list[str], *, cwd: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run one exact installed-wheel surface and preserve its real exit status."""
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )


def _entry(label: str, path: Path) -> dict[str, str]:
    """Create one byte-authenticated real corpus entry."""
    return {
        "label": label,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_exact_installed_wheel_trains_and_probes_with_pinned_model(tmp_path: Path) -> None:
    """Prove installed origins and drive real model-backed train then probe."""
    outside = tmp_path / "outside"
    wheel_dir = tmp_path / "wheel"
    prefix = tmp_path / "prefix"
    home = tmp_path / "home"
    for directory in (outside, wheel_dir, home):
        directory.mkdir()
    clean_environment = {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONNOUSERSITE": "1",
    }
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(ROOT),
        ],
        cwd=outside,
        environment=clean_environment,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
    wheel = wheels[0]
    assert len(hashlib.sha256(wheel.read_bytes()).hexdigest()) == 64
    _run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--prefix", str(prefix), str(wheel)],
        cwd=outside,
        environment=clean_environment,
    )
    sites = list((prefix / "lib").glob("python*/site-packages"))
    assert len(sites) == 1, f"expected one installed site-packages, found {sites}"
    site = sites[0].resolve()
    console = prefix / "bin" / "remanentia-snn-memory"
    assert console.is_file()
    installed_environment = {**clean_environment, "PYTHONPATH": str(site)}
    origins = json.loads(
        _run(
            [
                sys.executable,
                "-c",
                (
                    "import file_utils,json,snn_memory;"
                    "from pathlib import Path;"
                    "print(json.dumps({"
                    "'package':str(Path(snn_memory.__file__).resolve()),"
                    "'file_utils':str(Path(file_utils.__file__).resolve())}))"
                ),
            ],
            cwd=outside,
            environment=installed_environment,
        ).stdout
    )
    assert Path(origins["package"]).is_relative_to(site)
    assert Path(origins["file_utils"]).is_relative_to(site)

    config = outside / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": {"n_neurons": 24, "connectivity": 0.25},
                "encoder": {"feature_dim": 24, "packet_ms": 5, "silent_ms": 1},
                "train": {"seed": 11, "epochs": 1, "input_current": 18.0},
            }
        ),
        encoding="utf-8",
    )
    snn_doc = ROOT / "docs/research/snn_consolidation.md"
    retrieval_doc = ROOT / "docs/adr/0004-dual-retrieval-stacks.md"
    corpus = outside / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder_checkpoint": str(MODEL),
                "encoder_digest": PINNED_DIGEST,
                "entries": [_entry("snn", snn_doc), _entry("retrieval", retrieval_doc)],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = outside / "checkpoint"
    probe = outside / "probe.json"
    _run(
        [
            str(console),
            "train",
            "--config",
            str(config),
            "--corpus-manifest",
            str(corpus),
            "--output",
            str(checkpoint),
        ],
        cwd=outside,
        environment=installed_environment,
    )
    _run(
        [
            str(console),
            "probe",
            "--checkpoint",
            str(checkpoint),
            "--cue",
            str(snn_doc),
            "--encoder-checkpoint",
            str(MODEL),
            "--completion-steps",
            "8",
            "--output",
            str(probe),
        ],
        cwd=outside,
        environment=installed_environment,
    )
    payload = json.loads(probe.read_text(encoding="utf-8"))
    assert payload["weight_digest_unchanged"] is True
    assert payload["label"] is None or payload["label"] in {"snn", "retrieval"}
