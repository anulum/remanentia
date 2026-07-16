# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Explicit pinned-model gate verifier

"""Run every non-default temporal-SNN pinned-model gate with exact exits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.model_gates.model_precondition import require_pinned_model  # noqa: E402

EXPECTED_GATES = {
    "snn_memory_benchmark_e2e_gate.py",
    "snn_memory_cli_e2e_gate.py",
    "snn_memory_cli_runners_gate.py",
    "snn_memory_installed_model_wheel_gate.py",
    "snn_memory_sentence_encoder_gate.py",
}


def _run(arguments: list[str]) -> None:
    """Run an exact gate command and fail immediately on its real exit code."""
    subprocess.run(arguments, cwd=ROOT, check=True)


def _coverage(python: str, test_path: str, module_path: str) -> None:
    """Run one isolated real gate and enforce exact line-and-branch coverage."""
    _run([python, "-m", "coverage", "erase"])
    _run(
        [
            python,
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--branch",
            "-m",
            "pytest",
            test_path,
        ]
    )
    _run(
        [
            python,
            "-m",
            "coverage",
            "report",
            "--rcfile=/dev/null",
            f"--include={module_path}",
            "--show-missing",
            "--fail-under=100",
        ]
    )


def main() -> int:
    """Verify the pin, isolated adapter coverage, and all process boundaries."""
    actual = {path.name for path in (ROOT / "tests/model_gates").glob("*_gate.py")}
    if actual != EXPECTED_GATES:
        raise AssertionError(
            f"model-gate verifier inventory drift: expected={sorted(EXPECTED_GATES)} actual={sorted(actual)}"
        )
    require_pinned_model()
    python = sys.executable
    _coverage(
        python,
        "tests/model_gates/snn_memory_sentence_encoder_gate.py",
        "snn_memory/sentence_encoder.py",
    )
    _coverage(
        python,
        "tests/model_gates/snn_memory_cli_runners_gate.py",
        "snn_memory/cli_runners.py",
    )
    _run(
        [
            python,
            "-m",
            "pytest",
            "tests/model_gates/snn_memory_cli_e2e_gate.py",
            "tests/model_gates/snn_memory_benchmark_e2e_gate.py",
            "tests/model_gates/snn_memory_installed_model_wheel_gate.py",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
