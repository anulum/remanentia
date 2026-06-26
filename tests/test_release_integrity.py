# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for release integrity workflow checks

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_release_integrity.py"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_release_integrity", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestReleaseIntegrityWorkflow:
    def test_current_release_workflow_passes_integrity_contract(self):
        checker = _load_module()

        result = checker.check_release_workflow(RELEASE_WORKFLOW)

        assert result.ok, result.format()

    def test_rejects_release_upload_without_sigstore_bundles(self, tmp_path):
        checker = _load_module()
        workflow = tmp_path / "release.yml"
        workflow.write_text(
            RELEASE_WORKFLOW.read_text(encoding="utf-8").replace(
                "            *.sigstore\n",
                "",
            ),
            encoding="utf-8",
        )

        result = checker.check_release_workflow(workflow)

        assert not result.ok
        assert (
            "GitHub Release uploads sdist, wheel, SBOM, digest manifest, and .sigstore bundles"
            in result.format()
        )

    def test_rejects_sigstore_step_without_local_verify(self, tmp_path):
        checker = _load_module()
        workflow = tmp_path / "release.yml"
        workflow.write_text(
            RELEASE_WORKFLOW.read_text(encoding="utf-8").replace(
                "          verify: true\n",
                "          verify: false\n",
            ),
            encoding="utf-8",
        )

        result = checker.check_release_workflow(workflow)

        assert not result.ok
        assert "sigstore action verifies generated bundles before release" in result.format()

    def test_action_ref_pin_check_fails_when_no_actions_are_referenced(self):
        checker = _load_module()

        assert checker._action_refs_are_pinned("name: release\n") is False

    def test_indented_block_after_returns_empty_for_missing_key(self):
        checker = _load_module()

        assert checker._indented_block_after(["name: release"], 0, "files") == ""

    def test_main_returns_zero_for_current_workflow(self, capsys):
        checker = _load_module()

        code = checker.main([str(RELEASE_WORKFLOW)])

        assert code == 0
        assert "PASS: workflow action refs are pinned" in capsys.readouterr().out

    def test_main_returns_one_for_broken_workflow_from_sys_argv(
        self, tmp_path, monkeypatch, capsys
    ):
        checker = _load_module()
        workflow = tmp_path / "release.yml"
        workflow.write_text("name: release\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["check_release_integrity.py", str(workflow)])

        code = checker.main()

        assert code == 1
        assert "FAIL: workflow action refs are pinned" in capsys.readouterr().out
