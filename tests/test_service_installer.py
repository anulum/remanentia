# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Service installer tests

from __future__ import annotations

from pathlib import Path

import tools.install_user_services as installer
from tools.install_user_services import main


def _record_systemctl(monkeypatch) -> list[tuple[str, ...]]:
    """Capture systemctl calls so the activation policy can be asserted offline."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(installer, "_systemctl", lambda *a: calls.append(a))
    return calls


def test_installer_writes_api_and_vector_worker_units(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    user_dir = tmp_path / "systemd"

    code = main(
        [
            "--repo",
            str(repo),
            "--user-dir",
            str(user_dir),
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--interval-s",
            "60",
            "--embedding-base-url",
            "http://127.0.0.1:8082/v1",
            "--embedding-model",
            "nomic-embed",
            "--embedding-vector-size",
            "768",
            "--no-systemctl",
        ]
    )

    assert code == 0
    api_unit = (user_dir / "remanentia-api.service").read_text(encoding="utf-8")
    worker_unit = (user_dir / "remanentia-vector-worker.service").read_text(encoding="utf-8")
    assert f"WorkingDirectory={repo}" in api_unit
    assert "cli.py serve --host 127.0.0.1 --port 8765" in api_unit
    assert "REMANENTIA_EMBEDDING_BASE_URL=http://127.0.0.1:8082/v1" in api_unit
    assert "REMANENTIA_EMBEDDING_MODEL=nomic-embed" in api_unit
    assert "REMANENTIA_PUBLIC_VECTOR_SOURCES=paper" in api_unit
    assert "REMANENTIA_PUBLIC_VECTOR_PATH_PREFIXES=paper" in api_unit
    assert f"WorkingDirectory={repo}" in worker_unit
    assert "-m vector_pipeline watch --interval-s 60.0" in worker_unit
    assert "REMANENTIA_EMBEDDING_BASE_URL=http://127.0.0.1:8082/v1" in worker_unit


def test_installer_writes_freshness_watchdog_units(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    user_dir = tmp_path / "systemd"

    code = main(["--repo", str(repo), "--user-dir", str(user_dir), "--no-systemctl"])

    assert code == 0
    service = (user_dir / "remanentia-index-freshness.service").read_text(encoding="utf-8")
    timer = (user_dir / "remanentia-index-freshness.timer").read_text(encoding="utf-8")

    # The oneshot reads only mtimes and persists its verdict; it never rebuilds.
    assert "Type=oneshot" in service
    assert f"WorkingDirectory={repo}" in service
    assert f"REMANENTIA_BASE={repo}" in service
    expected_report = repo / "snn_state" / "index_freshness.json"
    assert f"-m index_freshness --report {expected_report}" in service
    # A failing oneshot must not be auto-restarted into a loop.
    assert "Restart=" not in service

    # The timer fires daily and catches up a missed run after the host wakes.
    assert "OnCalendar=daily" in timer
    assert "Persistent=true" in timer
    assert "Unit=remanentia-index-freshness.service" in timer
    assert "WantedBy=timers.target" in timer


class TestActivationPolicy:
    """Worker activation (a possible heavy rebuild) is opt-in; the watchdog is not."""

    def _args(self, tmp_path: Path) -> list[str]:
        repo = tmp_path / "repo"
        repo.mkdir()
        return ["--repo", str(repo), "--user-dir", str(tmp_path / "systemd")]

    def test_default_enables_watchdog_not_worker(self, tmp_path: Path, monkeypatch) -> None:
        calls = _record_systemctl(monkeypatch)

        assert main(self._args(tmp_path)) == 0

        assert ("daemon-reload",) in calls
        assert ("enable", "remanentia-api.service", "remanentia-index-freshness.timer") in calls
        assert ("start", "remanentia-index-freshness.timer") in calls
        # The refresh worker is never enabled or restarted without an explicit flag.
        assert not any("remanentia-vector-worker.service" in c for c in calls)

    def test_enable_worker_enables_but_does_not_start(self, tmp_path: Path, monkeypatch) -> None:
        calls = _record_systemctl(monkeypatch)

        assert main([*self._args(tmp_path), "--enable-worker"]) == 0

        assert ("enable", "remanentia-vector-worker.service") in calls
        assert not any(c[0] == "restart" for c in calls)

    def test_start_restarts_api_and_worker(self, tmp_path: Path, monkeypatch) -> None:
        calls = _record_systemctl(monkeypatch)

        assert main([*self._args(tmp_path), "--start"]) == 0

        assert ("enable", "remanentia-vector-worker.service") in calls
        assert ("restart", "remanentia-api.service", "remanentia-vector-worker.service") in calls

    def test_no_systemctl_skips_all_activation(self, tmp_path: Path, monkeypatch) -> None:
        calls = _record_systemctl(monkeypatch)

        assert main([*self._args(tmp_path), "--no-systemctl"]) == 0

        assert calls == []


def test_units_use_checkout_venv_python_when_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    user_dir = tmp_path / "systemd"

    assert main(["--repo", str(repo), "--user-dir", str(user_dir), "--no-systemctl"]) == 0

    service = (user_dir / "remanentia-index-freshness.service").read_text(encoding="utf-8")
    assert f"ExecStart={repo / '.venv' / 'bin' / 'python'} -m index_freshness" in service


def test_systemctl_invokes_user_scope(monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_run(cmd, check):
        captured.append(cmd)
        assert check is True

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    installer._systemctl("daemon-reload")
    assert captured == [["systemctl", "--user", "daemon-reload"]]
