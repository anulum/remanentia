# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Clean installed-wheel streamed SNN Stage-1 verifier

"""Build, install, and cover the compiled streamed backend without fallback."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(
    os.environ.get(
        "REMANENTIA_STAGE1_RUNTIME_ROOT", str(Path.home() / ".cache" / "remanentia" / "runtime")
    )
)
GATE = ROOT / "tests/installed_gates/snn_memory_stream_stage1_gate.py"
EXPECTED_INSTALLED_GATES = {"snn_memory_stream_stage1_gate.py"}


def _run(arguments: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> None:
    """Run one mandatory build, install, gate, or coverage command."""
    subprocess.run(arguments, cwd=cwd, env=environment, check=True)


def _only(directory: Path, pattern: str) -> Path:
    """Return the sole fresh artifact matching a required pattern."""
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {matches}")
    return matches[0].resolve(strict=True)


def _sha256(path: Path) -> str:
    """Hash one wheel or installed extension."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Run the mandatory clean installed line-and-branch coverage gate."""
    actual_gates = {path.name for path in (ROOT / "tests/installed_gates").glob("*_gate.py")}
    if actual_gates != EXPECTED_INSTALLED_GATES:
        raise RuntimeError(
            "installed-gate inventory drift: "
            f"expected={sorted(EXPECTED_INSTALLED_GATES)} actual={sorted(actual_gates)}"
        )
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="remanentia-stage1-", dir=RUNTIME_ROOT))
    rust_wheels = workspace / "rust-wheels"
    python_wheels = workspace / "python-wheels"
    install_target = workspace / "installed"
    for directory in (rust_wheels, python_wheels, install_target):
        directory.mkdir()

    _run(
        [
            "maturin",
            "build",
            "--release",
            "--locked",
            "--offline",
            "--out",
            str(rust_wheels),
        ],
        cwd=ROOT / "rust_snn_memory",
    )
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(python_wheels),
            str(ROOT),
        ],
        cwd=workspace,
    )
    rust_wheel = _only(rust_wheels, "rust_snn_memory-*.whl")
    python_wheel = _only(python_wheels, "remanentia-*.whl")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(install_target),
            str(python_wheel),
            str(rust_wheel),
        ],
        cwd=workspace,
    )
    extension = _only(install_target / "rust_snn_memory", "rust_snn_memory*.so")
    adapter = (install_target / "snn_memory/stream_backend.py").resolve(strict=True)
    extension_sha256 = _sha256(extension)
    coverage_data = workspace / ".coverage"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(install_target)
    _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--branch",
            f"--data-file={coverage_data}",
            f"--include={adapter}",
            str(GATE),
            "--extension-sha256",
            extension_sha256,
            "--repo-root",
            str(ROOT),
            "--install-target",
            str(install_target),
        ],
        cwd=workspace,
        environment=environment,
    )
    _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            "--rcfile=/dev/null",
            f"--data-file={coverage_data}",
            f"--include={adapter}",
            "--show-missing",
            "--fail-under=100",
        ],
        cwd=workspace,
        environment=environment,
    )
    print(
        json.dumps(
            {
                "adapter_origin": str(adapter),
                "extension_origin": str(extension),
                "extension_sha256": extension_sha256,
                "install_target": str(install_target),
                "python_wheel_sha256": _sha256(python_wheel),
                "rust_wheel_sha256": _sha256(rust_wheel),
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
