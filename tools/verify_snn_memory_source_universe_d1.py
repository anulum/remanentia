# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Clean installed-wheel D1 source-universe verifier

"""Build and cover D1 from a fresh installed production wheel."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("REMANENTIA_STAGE2_RUNTIME_ROOT", "[workspace]/_runtime"))
GATES = {
    "snn_memory_source_universe_d1_gate.py",
    "snn_memory_cue_materializer_d2_gate.py",
}


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
        raise RuntimeError(f"expected one {pattern}, found {matches}")
    return matches[0].resolve(strict=True)


def main() -> int:
    actual = {path.name for path in (ROOT / "tests/stage2_installed_gates").glob("*_gate.py")}
    if actual != GATES:
        raise RuntimeError(
            f"D1 gate inventory drift: expected={sorted(GATES)} actual={sorted(actual)}"
        )
    RUNTIME.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="remanentia-d1-", dir=RUNTIME))
    wheels = workspace / "wheels"
    prefix = workspace / "prefix"
    fixtures = workspace / "fixtures"
    for directory in (wheels, prefix, fixtures):
        directory.mkdir()
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
            str(wheels),
            str(ROOT),
        ],
        workspace,
    )
    wheel = _one(wheels, "remanentia-*.whl")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--prefix",
            str(prefix),
            str(wheel),
        ],
        workspace,
    )
    sites = list((prefix / "lib").glob("python*/site-packages"))
    if len(sites) != 1:
        raise RuntimeError(f"expected one installed site-packages, found {sites}")
    site = sites[0].resolve(strict=True)
    module = (site / "snn_memory/source_universe.py").resolve(strict=True)
    coverage_data = workspace / ".coverage"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(site)
    gate = ROOT / "tests/stage2_installed_gates/snn_memory_source_universe_d1_gate.py"
    _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--branch",
            f"--data-file={coverage_data}",
            f"--include={module}",
            str(gate),
            "--workspace",
            str(fixtures),
            "--install-target",
            str(prefix),
            "--public-schema",
            str(ROOT / "docs/schema/snn_memory_source_universe_v2.schema.json"),
            "--public-license",
            str(ROOT / "docs/schema/snn_memory_source_universe_v2.schema.json.license"),
        ],
        workspace,
        environment,
    )
    _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            "--rcfile=/dev/null",
            f"--data-file={coverage_data}",
            f"--include={module}",
            "--show-missing",
            "--fail-under=100",
        ],
        workspace,
        environment,
    )
    schema = module.parent / "schema/snn_memory_source_universe_v2.schema.json"
    print(
        json.dumps(
            {
                "install_prefix": str(prefix),
                "module_origin": str(module),
                "module_sha256": _sha(module),
                "packaged_schema_sha256": _sha(schema),
                "status": "pass",
                "wheel_sha256": _sha(wheel),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
