# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Clean installed-wheel D3 experiment-lock and G-B preflight verifier

"""Build fresh Python and Rust wheels, install them, and cover D3 to 100%."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("REMANENTIA_STAGE3_RUNTIME_ROOT", "[workspace]/_runtime"))
GATE = ROOT / "tests/stage3_installed_gates/snn_memory_d3_gate.py"
EXPECTED_GATES = {"snn_memory_d3_gate.py"}
MODEL = ROOT / ".snn_models" / "all-MiniLM-L6-v2"
MANIFESTS = (
    ROOT / "experiments/snn_memory/development_corpus.json",
    ROOT / "experiments/snn_memory/locked_evaluation_corpus.json",
)


def _run(arguments: list[str], cwd: Path, environment: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, env=environment, check=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {matches}")
    return matches[0].resolve(strict=True)


def _pinned_encoder_digest() -> str:
    if not MODEL.is_dir():
        raise RuntimeError(f"pinned local encoder not provisioned: {MODEL}; report the blocker")
    declared = tuple(
        json.loads(path.read_text(encoding="utf-8"))["encoder_digest"] for path in MANIFESTS
    )
    if len(set(declared)) != 1:
        raise RuntimeError(f"tracked corpus manifests disagree on encoder digest: {declared}")
    digest = hashlib.sha256()
    digest.update(b"remanentia.encoder-directory.v1\x00")
    entries = sorted(item for item in MODEL.rglob("*") if item.is_file())
    if not entries:
        raise RuntimeError(f"pinned local encoder is empty: {MODEL}")
    for item in entries:
        relative = item.relative_to(MODEL).as_posix().encode("utf-8")
        content = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    live = digest.hexdigest()
    if live != declared[0]:
        raise RuntimeError(f"pinned local encoder digest drift: tracked={declared[0]} live={live}")
    return live


def main() -> int:
    actual = {path.name for path in (ROOT / "tests/stage3_installed_gates").glob("*_gate.py")}
    if actual != EXPECTED_GATES:
        raise RuntimeError(f"D3 gate inventory drift: expected={sorted(EXPECTED_GATES)} actual={sorted(actual)}")
    encoder_digest = _pinned_encoder_digest()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="remanentia-d3-", dir=RUNTIME))
    rust_wheels = workspace / "rust-wheels"
    python_wheels = workspace / "python-wheels"
    install_target = workspace / "installed"
    fixtures = workspace / "fixtures"
    for directory in (rust_wheels, python_wheels, install_target, fixtures):
        directory.mkdir()
    _run(
        ["maturin", "build", "--release", "--locked", "--offline", "--out", str(rust_wheels)],
        ROOT / "rust_snn_memory",
    )
    _run(
        [
            sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
            "--no-build-isolation", "--wheel-dir", str(python_wheels), str(ROOT),
        ],
        workspace,
    )
    rust_wheel = _one(rust_wheels, "rust_snn_memory-*.whl")
    python_wheel = _one(python_wheels, "remanentia-*.whl")
    _run(
        [
            sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
            "--target", str(install_target), str(python_wheel), str(rust_wheel),
        ],
        workspace,
    )
    extension = _one(install_target / "rust_snn_memory", "rust_snn_memory*.so")
    lock_module = (install_target / "snn_memory/experiment_lock.py").resolve(strict=True)
    gb_module = (install_target / "snn_memory/gb_preflight.py").resolve(strict=True)
    packaged_schema = (
        install_target / "snn_memory/schema/snn_memory_experiment_lock_v2.schema.json"
    ).resolve(strict=True)
    extension_sha256 = _sha(extension)
    coverage_data = workspace / ".coverage"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(install_target)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    _run(
        [
            sys.executable, "-m", "coverage", "run", "--rcfile=/dev/null", "--branch",
            f"--data-file={coverage_data}",
            f"--include={lock_module},{gb_module}",
            str(GATE),
            "--workspace", str(fixtures),
            "--install-target", str(install_target),
            "--repo-root", str(ROOT),
            "--extension-sha256", extension_sha256,
            "--encoder-checkpoint", str(MODEL),
            "--encoder-digest", encoder_digest,
            "--python-wheel-sha256", _sha(python_wheel),
            "--rust-wheel-sha256", _sha(rust_wheel),
            "--public-schema", str(ROOT / "docs/schema/snn_memory_experiment_lock_v2.schema.json"),
            "--public-license", str(ROOT / "docs/schema/snn_memory_experiment_lock_v2.schema.json.license"),
        ],
        workspace,
        environment,
    )
    _run(
        [
            sys.executable, "-m", "coverage", "report", "--rcfile=/dev/null",
            f"--data-file={coverage_data}",
            f"--include={lock_module},{gb_module}",
            "--show-missing", "--fail-under=100",
        ],
        workspace,
        environment,
    )
    print(
        json.dumps(
            {
                "encoder_digest": encoder_digest,
                "experiment_lock_module_origin": str(lock_module),
                "experiment_lock_module_sha256": _sha(lock_module),
                "gb_preflight_module_origin": str(gb_module),
                "gb_preflight_module_sha256": _sha(gb_module),
                "extension_origin": str(extension),
                "extension_sha256": extension_sha256,
                "install_prefix": str(install_target),
                "packaged_schema_sha256": _sha(packaged_schema),
                "python_wheel_sha256": _sha(python_wheel),
                "rust_wheel_sha256": _sha(rust_wheel),
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
