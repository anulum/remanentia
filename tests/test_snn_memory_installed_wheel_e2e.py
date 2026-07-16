# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Exact-wheel model-free installed CLI gate

"""Build one wheel and exercise only its installed model-free public surfaces."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one installed-surface command with captured deterministic output."""
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )


def test_exact_wheel_model_free_cli_outside_checkout(tmp_path: Path) -> None:
    """Prove wheel origin, checkpoint API, help, inspect and verification."""
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
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert len(wheel_sha256) == 64

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--prefix",
            str(prefix),
            str(wheel),
        ],
        cwd=outside,
        environment=clean_environment,
    )
    sites = list((prefix / "lib").glob("python*/site-packages"))
    assert len(sites) == 1, f"expected one installed site-packages, found {sites}"
    site = sites[0].resolve()
    console = prefix / "bin" / "remanentia-snn-memory"
    assert console.is_file()
    installed_environment = {
        **clean_environment,
        "PYTHONPATH": str(site),
    }
    checkpoint = outside / "checkpoint"
    create_script = """
import json
from pathlib import Path
import file_utils
import snn_memory
from snn_memory.checkpoint import save_checkpoint
from snn_memory.contracts import EncoderConfig, ModelConfig
from snn_memory.state import initialise_weights

model = ModelConfig(n_neurons=4, connectivity=1.0)
weights, topology = initialise_weights(model, 11)
encoder = EncoderConfig(feature_dim=8, packet_ms=2, silent_ms=0)
manifest = save_checkpoint(
    Path(__import__('sys').argv[1]),
    weights,
    topology,
    __import__('numpy').zeros((1, 32), dtype='float64'),
    ['memory'],
    model,
    {
        'seed': 11,
        'epochs_completed': 1,
        'input_current': 18.0,
        'encoder': encoder.__dict__,
        'encoder_digest': '0' * 64,
        'corpus_digest': '0' * 64,
        'python': __import__('platform').python_version(),
    },
    [{'epoch': 0, 'label': 'memory', 'timesteps': 1}],
)
print(json.dumps({
    'snn_memory_origin': str(Path(snn_memory.__file__).resolve()),
    'file_utils_origin': str(Path(file_utils.__file__).resolve()),
    'array_digest': manifest['array_digest'],
}))
"""
    created = _run(
        [sys.executable, "-c", create_script, str(checkpoint)],
        cwd=outside,
        environment=installed_environment,
    )
    evidence = json.loads(created.stdout)
    assert Path(evidence["snn_memory_origin"]).is_relative_to(site)
    assert Path(evidence["file_utils_origin"]).is_relative_to(site)

    help_result = _run([str(console), "--help"], cwd=outside, environment=installed_environment)
    assert "inspect" in help_result.stdout
    assert "verify-manifest" in help_result.stdout
    inspected = _run(
        [str(console), "inspect", "--checkpoint", str(checkpoint)],
        cwd=outside,
        environment=installed_environment,
    )
    inspected_manifest = json.loads(inspected.stdout)
    verified = _run(
        [str(console), "verify-manifest", "--run-dir", str(checkpoint)],
        cwd=outside,
        environment=installed_environment,
    )
    assert inspected_manifest["array_digest"] == evidence["array_digest"]
    assert verified.stdout.strip() == evidence["array_digest"]
