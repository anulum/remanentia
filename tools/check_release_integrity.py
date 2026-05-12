# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Release workflow integrity checker

"""Validate release workflow supply-chain controls.

The checker is intentionally stdlib-only and conservative: it scans the
tracked workflow text for controls that are easy to accidentally remove
while editing release automation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


@dataclass(frozen=True)
class CheckResult:
    """Aggregated workflow integrity check result."""

    checks: tuple[tuple[str, bool], ...]

    @property
    def ok(self) -> bool:
        return all(passed for _, passed in self.checks)

    def format(self) -> str:
        lines = []
        for label, passed in self.checks:
            status = "PASS" if passed else "FAIL"
            lines.append(f"{status}: {label}")
        return "\n".join(lines)


def _contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def _release_files_block(text: str) -> str:
    match = re.search(
        r"uses:\s+softprops/action-gh-release@[^\n]+\n"
        r"(?:[ \t]+[^\n]*\n)*?"
        r"[ \t]+files:\s*\|\n"
        r"(?P<files>(?:[ \t]{12}[^\n]+\n)+)",
        text,
        flags=re.MULTILINE,
    )
    return match.group("files") if match else ""


def _sigstore_block(text: str) -> str:
    match = re.search(
        r"uses:\s+sigstore/gh-action-sigstore-python@[^\n]+\n"
        r"[ \t]+with:\n"
        r"(?P<block>(?:[ \t]{10}[^\n]+\n|[ \t]{12}[^\n]+\n)+)",
        text,
        flags=re.MULTILINE,
    )
    return match.group("block") if match else ""


def _action_refs_are_pinned(text: str) -> bool:
    refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", text)
    if not refs:
        return False
    return all(re.fullmatch(r"[0-9a-f]{40}", ref) is not None for ref in refs)


def check_release_workflow(path: str | Path = DEFAULT_WORKFLOW) -> CheckResult:
    """Return release workflow integrity status."""

    workflow = Path(path)
    text = workflow.read_text(encoding="utf-8")
    sigstore_block = _sigstore_block(text)
    release_files = _release_files_block(text)

    checks = [
        (
            "release job can write contents, OIDC tokens, and attestations",
            _contains(text, r"contents:\s+write")
            and _contains(text, r"id-token:\s+write")
            and _contains(text, r"attestations:\s+write"),
        ),
        ("workflow action refs are pinned to immutable SHAs", _action_refs_are_pinned(text)),
        (
            "CycloneDX SBOM is generated from installed release artefact",
            "pip install dist/*.tar.gz" in text
            and "cyclonedx-py environment" in text
            and "sbom.cyclonedx.json" in text,
        ),
        (
            "sha256 digest manifest covers sdist and wheel",
            "sha256sum *.tar.gz *.whl > ../sha256sums.txt" in text,
        ),
        (
            "sigstore signs sdist, wheel, SBOM, and digest manifest",
            "dist/*.tar.gz" in sigstore_block
            and "dist/*.whl" in sigstore_block
            and "sbom.cyclonedx.json" in sigstore_block
            and "sha256sums.txt" in sigstore_block,
        ),
        (
            "sigstore action verifies generated bundles before release",
            "verify: true" in sigstore_block
            and "verify-cert-identity:" in sigstore_block
            and "https://token.actions.githubusercontent.com" in sigstore_block,
        ),
        (
            "SLSA provenance attests built sdist and wheel",
            "actions/attest-build-provenance@" in text
            and "dist/*.tar.gz" in text
            and "dist/*.whl" in text,
        ),
        (
            "GitHub Release uploads sdist, wheel, SBOM, digest manifest, and .sigstore bundles",
            "dist/*.tar.gz" in release_files
            and "dist/*.whl" in release_files
            and "sbom.cyclonedx.json" in release_files
            and "sha256sums.txt" in release_files
            and "*.sigstore" in release_files,
        ),
    ]
    return CheckResult(tuple(checks))


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    workflow = Path(args[0]) if args else DEFAULT_WORKFLOW
    result = check_release_workflow(workflow)
    print(result.format())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
