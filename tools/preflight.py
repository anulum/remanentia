# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Pre-push preflight checks
"""Pre-push preflight: lint, format, tests, coverage, credentials scan."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = [
    "memory_index.py",
    "mcp_server.py",
    "consolidation_engine.py",
    "cli.py",
    "api.py",
    "api_server.py",
    "answer_extractor.py",
    "answer_normalizer.py",
    "knowledge_store.py",
    "temporal_graph.py",
    "entity_extractor.py",
    "observer.py",
    "reflector.py",
    "memory_recall.py",
    "arcane_retriever.py",
    "fact_decomposer.py",
]
TARGETS = MODULES + ["tests/"]


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"  FAIL: {label}")
        return False
    print(f"  PASS: {label}")
    return True


def main() -> int:
    checks: list[tuple[list[str], str]] = [
        (
            [sys.executable, "-m", "ruff", "check", *TARGETS, "--output-format=concise"],
            "ruff check (lint)",
        ),
        (
            [sys.executable, "-m", "ruff", "format", "--check", *TARGETS],
            "ruff format (formatting)",
        ),
        (
            [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                *MODULES,
                "-s",
                "B101,B110,B112,B301,B310,B311,B324,B403,B404,B603,B607",
                "-q",
            ],
            "bandit (security)",
        ),
        (
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q"],
            "pytest (tests)",
        ),
    ]

    # Credential scan: ensure no secrets in tracked files
    cred_patterns = [".env", "credentials", "secret", "token", "api_key"]
    print(f"\n{'=' * 60}")
    print("  Credential scan")
    print(f"{'=' * 60}")
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tracked = result.stdout.strip().splitlines()
    flagged = [f for f in tracked if any(p in f.lower() for p in cred_patterns)]
    if flagged:
        print(f"  FAIL: tracked files match credential patterns: {flagged}")
        return 1
    print("  PASS: no credential files tracked")

    # CLAUDE.md / .claude/ must be gitignored
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    missing_gitignore = []
    if "CLAUDE.md" not in gitignore:
        missing_gitignore.append("CLAUDE.md")
    if ".claude/" not in gitignore:
        missing_gitignore.append(".claude/")
    if missing_gitignore:
        print(f"  FAIL: missing from .gitignore: {missing_gitignore}")
        return 1
    print("  PASS: CLAUDE.md and .claude/ in .gitignore")

    passed = 0
    failed = 0
    for cmd, label in checks:
        if _run(cmd, label):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  PREFLIGHT: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
