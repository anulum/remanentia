# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — container deployment contract tests

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_repo_file(name: str) -> str:
    """Read a repository deployment file as UTF-8 text."""
    return (ROOT / name).read_text(encoding="utf-8")


def test_docker_lock_is_fully_hash_pinned() -> None:
    """Every requirement in the container lock must carry a sha256 hash.

    This is what lets the Dockerfile install with --require-hashes and satisfies
    the OpenSSF pinned-dependencies check. A floated (hashless) requirement here
    would silently re-open that alert.
    """
    lock = _read_repo_file("requirements/docker.txt")
    pinned = [
        line
        for line in lock.splitlines()
        if line and not line.startswith((" ", "\t", "#")) and "==" in line
    ]
    assert pinned, "lock has no pinned requirements"
    # The build toolchain must be pinned too, not just runtime deps.
    joined = "\n".join(pinned)
    for required in ("pip==", "setuptools==", "wheel==", "fastapi==", "numpy=="):
        assert required in joined, f"{required} missing from container lock"
    # Each pinned requirement line must be backed by at least one hash.
    assert lock.count("--hash=sha256:") >= len(pinned)


def test_dockerfile_runs_rest_api_as_non_root_with_healthcheck() -> None:
    """Validate the production Dockerfile REST API runtime contract."""
    dockerfile = _read_repo_file("Dockerfile")

    assert "FROM python:3.12-slim-bookworm" in dockerfile
    assert "USER remanentia" in dockerfile
    assert "ENV PYTHONDONTWRITEBYTECODE=1 \\" in dockerfile
    assert "REMANENTIA_BASE=/data \\" in dockerfile
    # Hash-pinned install: the runtime closure resolves only from the
    # --require-hashes lock, and the local package builds without an unpinned
    # build-time fetch (--no-build-isolation against the pinned toolchain).
    assert "COPY requirements/docker.txt requirements/docker.txt" in dockerfile
    assert (
        "python -m pip install --no-cache-dir --require-hashes -r requirements/docker.txt"
        in dockerfile
    )
    assert "python -m pip install --no-cache-dir --no-build-isolation --no-deps ." in dockerfile
    assert "HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \\" in dockerfile
    assert "http://127.0.0.1:8001/health" in dockerfile
    assert 'ENTRYPOINT ["remanentia"]' in dockerfile
    assert 'CMD ["serve", "--host", "0.0.0.0", "--port", "8001", "--require-auth"]' in dockerfile
    assert "COPY data/compiled_seed_facts.jsonl data/compiled_seed_facts.jsonl" in dockerfile


def test_compose_wires_secret_token_data_volume_and_localhost_port() -> None:
    """Validate the Compose deployment uses the real CLI and secret token file."""
    compose = _read_repo_file("docker-compose.yml")

    assert "remanentia-api:" in compose
    assert "dockerfile: Dockerfile" in compose
    assert "- --require-auth" in compose
    assert "- --token-file" in compose
    assert "- /run/secrets/remanentia_api_token" in compose
    assert "REMANENTIA_BASE: /data" in compose
    assert '"127.0.0.1:8001:8001"' in compose
    assert "remanentia-data:/data" in compose
    assert 'uid: "10001"' in compose
    assert 'gid: "10001"' in compose
    assert "mode: 0400" in compose
    assert "restart: unless-stopped" in compose
    assert "file: ./secrets/remanentia_api_token" in compose


def test_compose_healthcheck_matches_dockerfile_health_endpoint() -> None:
    """Keep Compose and Dockerfile health probes on the same public endpoint."""
    dockerfile = _read_repo_file("Dockerfile")
    compose = _read_repo_file("docker-compose.yml")

    endpoint = "http://127.0.0.1:8001/health"
    assert endpoint in dockerfile
    assert endpoint in compose
    assert "timeout=3" in dockerfile
    assert "timeout=3" in compose


def test_dockerignore_excludes_runtime_stores_and_secrets() -> None:
    """Prevent local memory stores, credentials, and build artifacts entering context."""
    entries = {
        line.strip()
        for line in _read_repo_file(".dockerignore").splitlines()
        if line.strip() and not line.startswith("#")
    }

    required_entries = {
        ".git",
        ".env",
        ".env.*",
        "credentials*",
        "*.key",
        "*.pem",
        "secrets/",
        "snn_state/",
        "reasoning_traces/",
        "memory/",
        "consolidation/",
        "models/",
        "training/datasets/",
        "site/",
        ".venv*",
        "venv*",
    }
    assert required_entries <= entries


def test_pyproject_packages_container_runtime_imports() -> None:
    """Ensure the wheel installed in the image carries REST runtime dependencies."""
    pyproject = _read_repo_file("pyproject.toml")

    required_modules = {
        "api",
        "api_security",
        "cli",
        "context_builder",
        "device_utils",
        "file_utils",
        "knowledge_store",
        "memory_index",
        "mcp_server",
        "pii_redactor",
        "signal_detector",
    }
    for module in required_modules:
        assert f'    "{module}",' in pyproject
