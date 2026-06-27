# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — API documentation wiring checker

"""Validate parity between packaged Python modules and API reference pages.

The package surface in ``pyproject.toml`` and the MkDocs API pages must move
together. A module shipped in the wheel without an API page is undocumented;
an API page for a module absent from ``py-modules`` is stale public
documentation. This checker makes both cases a local gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from collections.abc import Sequence

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ApiDocsAuditResult:
    """API documentation wiring audit result."""

    packaged_modules: tuple[str, ...]
    api_pages: tuple[str, ...]
    mkdocs_pages: tuple[str, ...]
    missing_pages: tuple[str, ...]
    orphan_pages: tuple[str, ...]
    missing_nav: tuple[str, ...]
    orphan_nav: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether package modules and API pages are in parity."""

        return (
            not self.missing_pages
            and not self.orphan_pages
            and not self.missing_nav
            and not self.orphan_nav
        )

    def format(self) -> str:
        """Return a deterministic human-readable report."""

        if self.ok:
            return (
                "PASS: "
                f"{len(self.packaged_modules)} packaged modules match "
                f"{len(self.api_pages)} API pages and "
                f"{len(self.mkdocs_pages)} MkDocs nav entries"
            )
        lines = [
            (
                "FAIL: "
                f"{len(self.packaged_modules)} packaged modules, "
                f"{len(self.api_pages)} API pages, "
                f"{len(self.mkdocs_pages)} MkDocs nav entries"
            )
        ]
        if self.missing_pages:
            lines.append("missing API pages: " + ", ".join(self.missing_pages))
        if self.orphan_pages:
            lines.append("API pages without packaged modules: " + ", ".join(self.orphan_pages))
        if self.missing_nav:
            lines.append("API pages missing from MkDocs nav: " + ", ".join(self.missing_nav))
        if self.orphan_nav:
            lines.append("MkDocs API nav entries without pages: " + ", ".join(self.orphan_nav))
        return "\n".join(lines)


def audit_api_docs(root: Path = ROOT) -> ApiDocsAuditResult:
    """Audit packaged module names against ``docs/api/*.md`` page stems.

    Parameters
    ----------
    root:
        Repository root containing ``pyproject.toml`` and ``docs/api``.

    Returns
    -------
    ApiDocsAuditResult
        Sorted parity inventory and mismatch lists.
    """

    packaged_modules = tuple(sorted(_packaged_modules(root / "pyproject.toml")))
    api_pages = tuple(sorted(_api_page_names(root / "docs" / "api")))
    mkdocs_pages = tuple(sorted(_mkdocs_api_refs(root / "mkdocs.yml")))
    packaged_set = set(packaged_modules)
    page_set = set(api_pages)
    nav_set = set(mkdocs_pages)
    return ApiDocsAuditResult(
        packaged_modules=packaged_modules,
        api_pages=api_pages,
        mkdocs_pages=mkdocs_pages,
        missing_pages=tuple(sorted(packaged_set - page_set)),
        orphan_pages=tuple(sorted(page_set - packaged_set)),
        missing_nav=tuple(sorted(page_set - nav_set)),
        orphan_nav=tuple(sorted(nav_set - page_set)),
    )


def _packaged_modules(pyproject_path: Path) -> set[str]:
    """Read top-level ``py-modules`` from a ``pyproject.toml`` file."""

    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    raw_modules = data.get("tool", {}).get("setuptools", {}).get("py-modules", [])
    modules: set[str] = set()
    for item in raw_modules:
        if isinstance(item, str) and item.strip():
            modules.add(item.strip())
    return modules


def _api_page_names(api_dir: Path) -> set[str]:
    """Return API page names from Markdown files in ``api_dir``."""

    if not api_dir.exists():
        return set()
    return {path.stem for path in api_dir.glob("*.md") if path.is_file()}


def _mkdocs_api_refs(mkdocs_path: Path) -> set[str]:
    """Return API page names referenced by ``mkdocs.yml`` navigation."""

    if not mkdocs_path.exists():
        return set()
    text = mkdocs_path.read_text(encoding="utf-8")
    return set(re.findall(r"\bapi/([A-Za-z0-9_]+)\.md\b", text))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the API documentation wiring checker CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root containing pyproject.toml and docs/api",
    )
    args = parser.parse_args(argv)
    result = audit_api_docs(args.root)
    print(result.format())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
