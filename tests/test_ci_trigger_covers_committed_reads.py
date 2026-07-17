# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — CI-trigger meta-guard: committed-read surfaces must be in-trigger

"""Meta-guard closing the path-filter false-green class self-maintainingly.

CEO ruling 2026-07-17 (option B): a workflow that TESTS a surface must RUN when that surface
changes. A test that reads committed content from a repo-root-anchored path depends on that
surface; if the surface sits outside ``ci.yml``'s ``push.paths`` filter, a change to it silently
skips CI while unrelated workflows stay green — a false green. This guard fails the moment a test
introduces a ``__file__``-anchored read of a committed surface not covered by the trigger.

Honest scope (CEO refinement 1): only ``__file__``-anchored ``Path`` *constructions* that resolve
to a git-tracked path count. ``tmp_path`` fixtures and path-STRING equality asserts (e.g.
``report["path"] == "benchmarks/..."``) are correctly excluded — they are neither ``__file__``-
anchored nor a committed-content dependency. Intermediate path-join steps (``a`` when a test builds
``a / "b" / "c"``) are dropped; only the maximal read of each chain is judged. The guard is robust
to the model-gate skip/fail question: it only ensures the right workflow RUNS when the surface
changes; whatever the gate then does (skip or fail) is the honest CI verdict.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TESTS_DIR = REPO_ROOT / "tests"

# A directory read is covered iff a trigger glob matches an arbitrary child; this sentinel child
# carries no extension so an extension-scoped glob (``**/*.py``) correctly fails to cover a
# directory whose validated content is not Python.
_SENTINEL_CHILD = "￿"


# ---- GitHub Actions path-filter glob matching ----------------------------------------------------


def gh_glob_to_regex(glob: str) -> re.Pattern[str]:
    """Translate a GitHub Actions path-filter glob into an anchored regex.

    GitHub semantics: ``*`` matches a run of characters that never crosses ``/``; ``**`` matches
    across ``/``; a leading ``**/`` segment matches zero or more whole path segments; ``?`` matches
    exactly one non-``/`` character. Every other character is a literal.
    """
    out = ["^"]
    index = 0
    length = len(glob)
    while index < length:
        if glob.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
        elif glob.startswith("**", index):
            out.append(".*")
            index += 2
        elif glob[index] == "*":
            out.append("[^/]*")
            index += 1
        elif glob[index] == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(glob[index]))
            index += 1
    out.append("$")
    return re.compile("".join(out))


def push_paths(workflow_text: str) -> list[str]:
    """Return the ``on.push.paths`` globs of a workflow without importing a YAML parser.

    Handles the block form used by ``ci.yml`` (``paths:`` then ``- "glob"`` items) and the inline
    ``paths: ["a", "b"]`` form; keeping this dependency-free preserves the test suite's hermetic
    profile (PyYAML is not a declared test dependency).
    """
    inside = False
    globs: list[str] = []
    for raw in workflow_text.splitlines():
        stripped = raw.strip()
        if not inside:
            if stripped == "paths:":
                inside = True
            elif stripped.startswith("paths:") and "[" in stripped:
                return re.findall(r"""["']([^"']+)["']""", stripped)
            continue
        if stripped.startswith("- "):
            globs.append(stripped[2:].strip().strip("\"'"))
        elif stripped == "":
            continue
        else:
            break
    return globs


def path_covered(repo_relative: str, globs: list[str]) -> bool:
    """True when at least one trigger glob matches the repo-relative file path."""
    return any(glob.match(repo_relative) for glob in map(gh_glob_to_regex, globs))


def directory_covered(directory: str, globs: list[str]) -> bool:
    """True when a trigger glob covers an arbitrary child of the directory (whole-subtree cover)."""
    child = f"{directory}/{_SENTINEL_CHILD}"
    return any(glob.match(child) for glob in map(gh_glob_to_regex, globs))


# ---- committed-read discovery via AST ------------------------------------------------------------


def _drop(path: PurePosixPath, count: int) -> PurePosixPath:
    """Walk ``count`` parents up (``.`` once the repository root is reached)."""
    parts = path.parts
    if count >= len(parts):
        return PurePosixPath(".")
    return PurePosixPath(*parts[: len(parts) - count])


def _is_path_call(node: ast.expr) -> bool:
    """``Path(__file__)`` or ``pathlib.Path(__file__)``."""
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return False
    target = node.func
    named_path = (isinstance(target, ast.Name) and target.id == "Path") or (
        isinstance(target, ast.Attribute) and target.attr == "Path"
    )
    arg = node.args[0]
    return named_path and isinstance(arg, ast.Name) and arg.id == "__file__"


def _literal_int(node: ast.expr) -> int | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, int) else None


def _literal_str(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


class _FileAnchoredReads(ast.NodeVisitor):
    """Resolve ``__file__``-anchored ``Path`` expressions in one module to repo-relative paths."""

    def __init__(self, module_rel: PurePosixPath) -> None:
        self._module_rel = module_rel
        self.env: dict[str, PurePosixPath] = {}
        self.reads: set[PurePosixPath] = set()

    def resolve(self, node: ast.expr) -> PurePosixPath | None:
        """Return the repo-relative path a node denotes, or ``None`` if it is not anchored."""
        if _is_path_call(node):
            return self._module_rel
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve":
                return self.resolve(node.func.value)
            return None
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            if base is not None and node.attr == "parent":
                return _drop(base, 1)
            return None
        if isinstance(node, ast.Subscript):
            target = node.value
            if isinstance(target, ast.Attribute) and target.attr == "parents":
                base = self.resolve(target.value)
                index = _literal_int(node.slice)
                if base is not None and index is not None:
                    return _drop(base, index + 1)
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            base = self.resolve(node.left)
            suffix = _literal_str(node.right)
            if base is not None and suffix is not None:
                return base / suffix
            return None
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        return None

    def bind(self, node: ast.Assign) -> None:
        value = self.resolve(node.value)
        if value is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.env[target.id] = value

    def visit_BinOp(self, node: ast.BinOp) -> None:
        resolved = self.resolve(node)
        if resolved is not None:
            self.reads.add(resolved)
        self.generic_visit(node)


def file_anchored_reads(source: str, module_rel: str) -> set[PurePosixPath]:
    """Every repo-relative path a module builds from ``__file__`` (bind names, then collect reads)."""
    tree = ast.parse(source)
    visitor = _FileAnchoredReads(PurePosixPath(module_rel))
    for node in ast.walk(tree):  # pass 1: bind names regardless of source order
        if isinstance(node, ast.Assign):
            visitor.bind(node)
    visitor.visit(tree)  # pass 2: collect reads against the fully-populated environment
    return visitor.reads


# ---- committed-surface predicate -----------------------------------------------------------------


def _is_ancestor(candidate: PurePosixPath, other: PurePosixPath) -> bool:
    return other.as_posix().startswith(candidate.as_posix() + "/")


def maximal_reads(reads: set[PurePosixPath]) -> set[PurePosixPath]:
    """Drop intermediate path-join steps: keep only reads that no other read descends from."""
    return {read for read in reads if not any(_is_ancestor(read, other) for other in reads)}


def _tracked_files() -> frozenset[str]:
    output = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return frozenset(line for line in output.splitlines() if line)


def uncovered_read(read: PurePosixPath, tracked: frozenset[str], globs: list[str]) -> str | None:
    """Return the offending repo-relative path if it is committed and not in-trigger, else ``None``."""
    text = read.as_posix()
    if text in ("", "."):
        return None
    if text in tracked:
        return None if path_covered(text, globs) else text
    if any(item.startswith(text + "/") for item in tracked):  # tracked directory
        return None if directory_covered(text, globs) else text
    return None  # untracked (tmp/scratch/generated) — not a committed-content dependency


def collect_offenders(
    reads_by_module: dict[str, set[PurePosixPath]], tracked: frozenset[str], globs: list[str]
) -> dict[str, set[str]]:
    """Reduce per-module reads to the uncovered committed paths and the tests that introduce them."""
    offenders: dict[str, set[str]] = {}
    for module_rel, reads in reads_by_module.items():
        for read in maximal_reads(reads):
            offending = uncovered_read(read, tracked, globs)
            if offending is not None:
                offenders.setdefault(offending, set()).add(module_rel)
    return offenders


def uncovered_committed_reads() -> dict[str, set[str]]:
    """Map every committed read not covered by ci.yml's trigger to the tests that introduce it."""
    tracked = _tracked_files()
    globs = push_paths(CI_WORKFLOW.read_text(encoding="utf-8"))
    reads_by_module = {
        test_file.relative_to(REPO_ROOT).as_posix(): file_anchored_reads(
            test_file.read_text(encoding="utf-8"),
            test_file.relative_to(REPO_ROOT).as_posix(),
        )
        for test_file in sorted(TESTS_DIR.rglob("*.py"))
    }
    return collect_offenders(reads_by_module, tracked, globs)


# ---- unit coverage: GH glob matcher --------------------------------------------------------------


def test_star_matches_root_only_not_nested() -> None:
    assert path_covered("cli.py", ["*.py"])
    assert not path_covered("snn_memory/held_out_probe.py", ["*.py"])


def test_double_star_slash_matches_root_and_any_depth() -> None:
    pattern = ["**/*.py"]
    assert path_covered("cli.py", pattern)
    assert path_covered("snn_memory/held_out_probe.py", pattern)
    assert path_covered("a/b/c/deep.py", pattern)


def test_trailing_double_star_matches_everything_under() -> None:
    assert path_covered("experiments/snn_memory/corpus.json", ["experiments/**"])
    assert not path_covered("experimental/x.json", ["experiments/**"])


def test_exact_file_and_question_mark() -> None:
    assert path_covered("pyproject.toml", ["pyproject.toml"])
    assert path_covered("a.py", ["?.py"])
    assert not path_covered("ab.py", ["?.py"])


def test_bare_double_star_matches_across_slashes() -> None:
    assert path_covered("any/deep/thing.txt", ["**"])


def test_directory_cover_needs_subtree_glob_not_extension_glob() -> None:
    assert directory_covered("notebooks", ["notebooks/**"])
    assert directory_covered("tests", ["tests/**"])
    assert not directory_covered("notebooks", ["**/*.py"])
    assert not directory_covered("docs/schema", ["*.py", "tests/**"])


# ---- unit coverage: push.paths parser ------------------------------------------------------------


def test_push_paths_reads_block_form() -> None:
    text = 'on:\n  push:\n    paths:\n      - "*.py"\n      - "tests/**"\n  pull_request:\n    branches: [main]\n'
    assert push_paths(text) == ["*.py", "tests/**"]


def test_push_paths_reads_inline_form() -> None:
    assert push_paths('    paths: ["docs/**", "*.py"]\n') == ["docs/**", "*.py"]


def test_push_paths_skips_blank_lines_within_block() -> None:
    text = '    paths:\n      - "a.py"\n\n      - "b.py"\n    branches: [main]\n'
    assert push_paths(text) == ["a.py", "b.py"]


def test_push_paths_absent_returns_empty() -> None:
    assert push_paths("on:\n  push:\n    branches: [main]\n") == []


# ---- unit coverage: AST committed-read resolver --------------------------------------------------


def test_resolver_handles_root_constant_and_join() -> None:
    source = 'from pathlib import Path\nROOT = Path(__file__).resolve().parents[2]\nX = ROOT / "experiments/corpus.json"\n'
    reads = file_anchored_reads(source, "tests/model_gates/model_precondition.py")
    assert PurePosixPath("experiments/corpus.json") in reads


def test_resolver_handles_parent_chain_and_forward_reference() -> None:
    # Read appears before the anchor assignment in source: pass-1 binding must still resolve it.
    source = (
        'from pathlib import Path\nY = REPO / "tools/x.py"\nREPO = Path(__file__).parent.parent\n'
    )
    reads = file_anchored_reads(source, "tests/test_example.py")
    assert PurePosixPath("tools/x.py") in reads


def test_resolver_ignores_tmp_path_and_string_asserts() -> None:
    source = (
        "def test(tmp_path):\n"
        '    p = tmp_path / "experiments" / "x.json"\n'
        '    assert report["path"] == "benchmarks/report.json"\n'
        "    return p\n"
    )
    assert file_anchored_reads(source, "tests/test_x.py") == set()


def test_resolver_ignores_non_literal_join_and_accepts_pathlib_prefix() -> None:
    source = (
        "import pathlib\n"
        "ROOT = pathlib.Path(__file__).resolve().parents[1]\n"
        'name = "dynamic.json"\n'
        "a = ROOT / name\n"  # non-literal right operand -> not resolvable
        'b = ROOT / "data/f.jsonl"\n'
    )
    reads = file_anchored_reads(source, "tests/test_x.py")
    assert reads == {PurePosixPath("data/f.jsonl")}


def test_resolver_returns_nothing_for_unanchored_calls_and_subscripts() -> None:
    source = 'x = open("f").read()\ny = some_list[0]\nz = obj.parents[1]\n'
    assert file_anchored_reads(source, "tests/test_x.py") == set()


def test_resolver_binds_only_name_targets_but_still_reads_the_join() -> None:
    # An attribute/subscript assignment target is not a bindable name; the join is still a read.
    source = (
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        'obj.field = ROOT / "data/skip.jsonl"\n'  # non-Name target -> not bound, join still read
        'X = ROOT / "data/keep.jsonl"\n'
    )
    reads = file_anchored_reads(source, "tests/test_x.py")
    assert {PurePosixPath("data/skip.jsonl"), PurePosixPath("data/keep.jsonl")} <= reads


def test_drop_reaches_repo_root_dot() -> None:
    assert _drop(PurePosixPath("tests/x.py"), 5) == PurePosixPath(".")


def test_literal_helpers_reject_wrong_types() -> None:
    assert _literal_int(ast.parse("1.5", mode="eval").body) is None
    assert _literal_str(ast.parse("42", mode="eval").body) is None
    assert _is_path_call(ast.parse("open(x)", mode="eval").body) is False
    assert _is_path_call(ast.parse("Path(a, b)", mode="eval").body) is False


# ---- unit coverage: maximal-read + coverage predicate --------------------------------------------


def test_maximal_reads_drops_intermediate_join_steps() -> None:
    reads = {
        PurePosixPath(".github"),
        PurePosixPath(".github/workflows"),
        PurePosixPath(".github/workflows/ci.yml"),
    }
    assert maximal_reads(reads) == {PurePosixPath(".github/workflows/ci.yml")}


def test_uncovered_read_classifies_file_dir_and_untracked() -> None:
    tracked = frozenset({"experiments/corpus.json", "notebooks/demo.ipynb"})
    globs = ["**/*.py", "tests/**"]
    assert (
        uncovered_read(PurePosixPath("experiments/corpus.json"), tracked, globs)
        == "experiments/corpus.json"
    )
    assert uncovered_read(PurePosixPath("notebooks"), tracked, globs) == "notebooks"
    assert (
        uncovered_read(PurePosixPath("experiments/corpus.json"), tracked, ["experiments/**"])
        is None
    )
    assert uncovered_read(PurePosixPath(".codex_scratch/tmp.json"), tracked, globs) is None
    assert uncovered_read(PurePosixPath("."), tracked, globs) is None


def test_collect_offenders_records_uncovered_reads_and_clears_when_covered() -> None:
    reads_by_module = {"tests/test_x.py": {PurePosixPath("experiments/corpus.json")}}
    tracked = frozenset({"experiments/corpus.json"})
    assert collect_offenders(reads_by_module, tracked, ["**/*.py"]) == {
        "experiments/corpus.json": {"tests/test_x.py"}
    }
    assert collect_offenders(reads_by_module, tracked, ["experiments/**"]) == {}


# ---- the invariant --------------------------------------------------------------------------------


def test_ci_trigger_covers_every_committed_read_surface() -> None:
    offenders = uncovered_committed_reads()
    assert offenders == {}, (
        "committed paths read by tests but absent from ci.yml push.paths (false-green risk); "
        "add a covering glob to .github/workflows/ci.yml: "
        + "; ".join(f"{path} <- {sorted(tests)}" for path, tests in sorted(offenders.items()))
    )
