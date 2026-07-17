# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Clean installed-wheel D2 cue-materializer verifier

"""Build and cover D2 from a fresh installed production wheel with the real encoder."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(
    os.environ.get(
        "REMANENTIA_STAGE2_RUNTIME_ROOT", str(Path.home() / ".cache" / "remanentia" / "runtime")
    )
)
GATES = {
    "snn_memory_source_universe_d1_gate.py",
    "snn_memory_cue_materializer_d2_gate.py",
}
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
        raise RuntimeError(f"expected one {pattern}, found {matches}")
    return matches[0].resolve(strict=True)


def _pinned_encoder_digest() -> str:
    if not MODEL.is_dir():
        raise RuntimeError(
            f"pinned local encoder not provisioned: {MODEL}; stop and report the blocker"
        )
    declared = tuple(
        json.loads(path.read_text(encoding="utf-8"))["encoder_digest"] for path in MANIFESTS
    )
    if len(set(declared)) != 1:
        raise RuntimeError(f"tracked corpus manifests disagree on encoder digest: {declared}")
    digest = hashlib.sha256()
    digest.update(b"remanentia.encoder-directory.v1\x00")
    files = sorted(item for item in MODEL.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"pinned local encoder is empty: {MODEL}")
    for item in files:
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
    actual = {path.name for path in (ROOT / "tests/stage2_installed_gates").glob("*_gate.py")}
    if actual != GATES:
        raise RuntimeError(
            f"D2 gate inventory drift: expected={sorted(GATES)} actual={sorted(actual)}"
        )
    encoder_digest = _pinned_encoder_digest()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="remanentia-d2-", dir=RUNTIME))
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
    module = (site / "snn_memory/cue_materializer.py").resolve(strict=True)
    coverage_data = workspace / ".coverage"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(site)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    gate = ROOT / "tests/stage2_installed_gates/snn_memory_cue_materializer_d2_gate.py"
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
            "--public-cue-set-schema",
            str(ROOT / "docs/schema/snn_memory_cue_set_v2.schema.json"),
            "--public-cue-set-license",
            str(ROOT / "docs/schema/snn_memory_cue_set_v2.schema.json.license"),
            "--public-cue-bundle-schema",
            str(ROOT / "docs/schema/snn_memory_cue_bundle_v2.schema.json"),
            "--public-cue-bundle-license",
            str(ROOT / "docs/schema/snn_memory_cue_bundle_v2.schema.json.license"),
            "--encoder-checkpoint",
            str(MODEL),
            "--encoder-digest",
            encoder_digest,
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
    print(
        json.dumps(
            {
                "encoder_digest": encoder_digest,
                "install_prefix": str(prefix),
                "module_origin": str(module),
                "module_sha256": _sha(module),
                "packaged_cue_bundle_schema_sha256": _sha(
                    module.parent / "schema/snn_memory_cue_bundle_v2.schema.json"
                ),
                "packaged_cue_set_schema_sha256": _sha(
                    module.parent / "schema/snn_memory_cue_set_v2.schema.json"
                ),
                "status": "pass",
                "wheel_sha256": _sha(wheel),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
