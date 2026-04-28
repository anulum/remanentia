# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Service installer tests

from __future__ import annotations

from pathlib import Path

from tools.install_user_services import main


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
