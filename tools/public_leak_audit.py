# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Public release leak audit

"""Audit tracked public files for private workspace and agentic labels.

The audit intentionally scans tracked public surfaces, not ignored internal
planning notes. It is a release gate for accidental private path, workspace,
and agent identity leakage in files that would ship to users.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 2_000_000
PUBLIC_TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".css",
    ".dockerignore",
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "site",
}
INTERNAL_PATH_PREFIXES = (
    "docs/internal/",
    "00_SAFETY_BACKUPS/",
)
ALLOW_DIRECTIVE = "public-leak-audit: allow"
AGENT_IDENTITY_PATTERN = (
    r"\b(?:"
    r"Claude(?!\s+(?:Haiku|Sonnet))"  # public-leak-audit: allow
    r"|Gemini"  # public-leak-audit: allow
    r"|Codex"  # public-leak-audit: allow
    r"|ChatGPT"  # public-leak-audit: allow
    r")\b"
)


@dataclass(frozen=True)
class LeakRule:
    """One public leak detection rule."""

    label: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class LeakFinding:
    """One matched leak finding in a public file."""

    path: Path
    line: int
    column: int
    label: str
    match: str
    excerpt: str

    def format(self, root: Path) -> str:
        """Return a compact, deterministic finding line."""

        relpath = _display_path(self.path, root)
        return (
            f"{relpath}:{self.line}:{self.column}: {self.label}: {self.match!r} in {self.excerpt!r}"
        )


@dataclass(frozen=True)
class LeakAuditResult:
    """Aggregated public leak audit result."""

    findings: tuple[LeakFinding, ...]
    scanned: int

    @property
    def ok(self) -> bool:
        """Return whether the scan found no public leak findings."""

        return not self.findings

    def format(self, root: Path = ROOT) -> str:
        """Return a human-readable audit report."""

        if self.ok:
            return f"PASS: scanned {self.scanned} public text files; no internal leaks found"
        lines = [f"FAIL: found {len(self.findings)} public leak finding(s)"]
        lines.extend(finding.format(root) for finding in self.findings)
        return "\n".join(lines)


DEFAULT_RULES = (
    LeakRule(
        "private workspace path",
        re.compile(r"/(?:media|home)/anulum(?:/[A-Za-z0-9._@%+=:,/ -]+)?"),
    ),
    LeakRule(
        "private workspace label",
        re.compile(r"\b(?:project-workspace|workspace-internal|shared-coordination)\b"),  # public-leak-audit: allow
    ),
    LeakRule(
        "agent identity label",
        re.compile(AGENT_IDENTITY_PATTERN),
    ),
)


def tracked_public_paths(root: Path = ROOT) -> list[Path]:
    """Return tracked public text-file candidates under ``root``."""

    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[Path] = []
    for raw_path in result.stdout.splitlines():
        path = root / raw_path
        if _is_public_text_candidate(path, root):
            paths.append(path)
    return paths


def audit_public_surface(root: Path = ROOT) -> LeakAuditResult:
    """Audit all tracked public text-file candidates under ``root``."""

    return audit_paths(tracked_public_paths(root), root=root)


def audit_paths(paths: Iterable[Path], *, root: Path = ROOT) -> LeakAuditResult:
    """Audit explicit paths and return all public leak findings.

    Parameters
    ----------
    paths:
        Files to scan. Internal docs, binary files, and oversized files are
        filtered the same way as tracked public paths.
    root:
        Base path used for internal-path filtering and display.
    """

    findings: list[LeakFinding] = []
    scanned = 0
    for path in paths:
        candidate = path if path.is_absolute() else root / path
        if not _is_public_text_candidate(candidate, root):
            continue
        text = _read_text(candidate)
        if text is None:
            continue
        scanned += 1
        findings.extend(_scan_text(candidate, text))
    return LeakAuditResult(tuple(findings), scanned)


def _scan_text(path: Path, text: str) -> list[LeakFinding]:
    """Scan one text payload for all default leak rules."""

    findings: list[LeakFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if ALLOW_DIRECTIVE in line:
            continue
        for rule in DEFAULT_RULES:
            for match in rule.pattern.finditer(line):
                findings.append(
                    LeakFinding(
                        path=path,
                        line=line_number,
                        column=match.start() + 1,
                        label=rule.label,
                        match=match.group(0).strip(),
                        excerpt=line.strip()[:180],
                    )
                )
    return findings


def _read_text(path: Path) -> str | None:
    """Read a bounded UTF-8 text file or return ``None`` for non-text files."""

    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_public_text_candidate(path: Path, root: Path) -> bool:
    """Return whether ``path`` belongs to the public text audit surface."""

    try:
        relpath = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relpath = path.as_posix()
    if any(
        relpath == prefix.rstrip("/") or relpath.startswith(prefix)
        for prefix in INTERNAL_PATH_PREFIXES
    ):
        return False
    if any(part in SKIP_PARTS for part in Path(relpath).parts):
        return False
    return path.suffix.lower() in PUBLIC_TEXT_SUFFIXES or path.name in {
        "Makefile",
        "LICENSE",
        "NOTICE",
    }


def _display_path(path: Path, root: Path) -> str:
    """Return a stable display path relative to ``root`` when possible."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public leak audit CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root used for git ls-files and relative display paths.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional explicit paths to scan instead of all tracked public files.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    result = audit_paths(args.paths, root=root) if args.paths else audit_public_surface(root)
    print(result.format(root))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
