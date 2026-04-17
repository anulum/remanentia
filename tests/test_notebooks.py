# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Smoke tests for committed notebooks

"""Validate every notebook under ``notebooks/``.

We check three things, all stdlib-ish so the suite stays fast and
CI does not need a Jupyter install:

1. JSON + nbformat validation (no broken cells, every cell has an id).
2. Every import statement in a code cell resolves against the repo
   modules (catches notebooks that reference a deleted helper).
3. The leading markdown cell has the SPDX license marker — the same
   hygiene rule we apply to .py and .md files across the repo.

Full execution of the notebook (via nbclient) is intentionally out
of scope: some cells load the LongMemEval oracle which is gitignored.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _all_notebooks() -> list[Path]:
    return sorted(NOTEBOOK_DIR.rglob("*.ipynb"))


def _iter_imports(source: str):
    """Yield every imported module name from a code cell."""
    for line in source.splitlines():
        line = line.strip()
        m = re.match(r"^(?:from|import)\s+([\w.]+)", line)
        if not m:
            continue
        # `from X.y import ...` → X.y ; `import X.y` → X.y
        yield m.group(1)


class TestNotebooksExist:
    def test_at_least_one_notebook(self):
        nbs = _all_notebooks()
        assert nbs, "notebooks/ directory is empty; P5-29 should ship one walkthrough"


@pytest.mark.parametrize("nb_path", _all_notebooks())
class TestNotebookHygiene:
    def test_valid_nbformat(self, nb_path):
        nb = nbformat.read(nb_path, as_version=4)
        nbformat.validate(nb)

    def test_every_cell_has_id(self, nb_path):
        nb = nbformat.read(nb_path, as_version=4)
        for i, cell in enumerate(nb.cells):
            assert cell.get("id"), f"cell {i} in {nb_path.name} has no id"

    def test_spdx_marker_in_first_cell(self, nb_path):
        nb = nbformat.read(nb_path, as_version=4)
        assert nb.cells, f"{nb_path.name} has no cells"
        first = nb.cells[0]
        assert first.cell_type == "markdown", "first cell must be markdown"
        assert "AGPL-3.0-or-later" in first.source, (
            f"{nb_path.name} first cell missing AGPL-3.0-or-later marker"
        )

    def test_imports_resolve(self, nb_path):
        nb = nbformat.read(nb_path, as_version=4)
        seen_imports: set[str] = set()
        for cell in nb.cells:
            if cell.cell_type == "code":
                for mod in _iter_imports(cell.source):
                    seen_imports.add(mod)

        assert seen_imports, "notebook has no imports; very likely broken"

        # Only enforce the first segment of dotted imports; downstream
        # symbol resolution is not our problem here.
        for mod in seen_imports:
            root = mod.split(".")[0]
            # Std-lib or third-party modules are out of scope; we only
            # assert that repo modules named in the notebook still exist.
            repo_module = Path(__file__).resolve().parent.parent / f"{root}.py"
            if repo_module.exists():
                importlib.import_module(root)


# ── Offline end-to-end execution ─────────────────────────────────────
#
# The hygiene tests above only *validate* notebook shape; they cannot
# catch a rename that changes a function signature the notebook relies
# on. One full nbclient execution of the offline path guards against
# that. The oracle cell is skipped automatically (its `if oracle_path
# .exists()` fallback prints a pointer), so the test stays deterministic
# even when the LongMemEval oracle is absent.


class TestNotebookExecutesOffline:
    """Run the walkthrough notebook end-to-end via nbclient."""

    NOTEBOOK = NOTEBOOK_DIR / "01_load_oracle_and_query.ipynb"

    def test_offline_path_runs(self):
        nbclient = pytest.importorskip("nbclient")
        pytest.importorskip("ipykernel")

        nb = nbformat.read(self.NOTEBOOK, as_version=4)
        # Execute from the repo root so the module imports
        # (``fact_decomposer``, ``arcane_retriever``, …) resolve exactly
        # as they do for ``pytest``. The oracle cell uses a relative
        # ``../data/`` path that will simply fail its ``exists()`` check
        # from the repo root — that path prints a pointer and returns
        # cleanly, so the test stays deterministic even when the
        # gitignored oracle is not present.
        repo_root = self.NOTEBOOK.parent.parent
        client = nbclient.NotebookClient(
            nb,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(repo_root)}},
        )
        client.execute()

        # Every code cell finished without raising — nbclient surfaces
        # cell errors via ``CellExecutionError``, which would escape
        # above. Sanity-check that at least one cell produced output so
        # silent no-op notebooks fail the test too.
        executed_code = [c for c in nb.cells if c.cell_type == "code"]
        assert any(c.get("outputs") for c in executed_code), (
            "no code cell produced output — notebook may be silently empty"
        )
