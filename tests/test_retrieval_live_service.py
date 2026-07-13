# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real filesystem live-retrieval protocol tests

"""Exercise a real producer/consumer request-response filesystem roundtrip."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from REMANENTIA.retrieval_live_service import (
    LiveServiceConfig,
    load_live_service_config,
    retrieve_via_live_service,
)
from REMANENTIA.retrieval_network_io import write_json_atomic


def _config(root: Path, *, timeout_s: float = 0.2) -> LiveServiceConfig:
    request_dir = root / "requests"
    response_dir = root / "responses"
    request_dir.mkdir(parents=True)
    response_dir.mkdir(parents=True)
    return {
        "request_dir": request_dir,
        "response_dir": response_dir,
        "timeout_s": timeout_s,
        "transport": "filesystem",
        "cycle": 17,
        "n_neurons": 256,
    }


def _start_consumer(
    config: LiveServiceConfig,
    handler: Callable[[Path, dict[str, object]], None],
) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def consume() -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            requests = list(config["request_dir"].glob("*.json"))
            if requests:
                request_path = requests[0]
                try:
                    payload = json.loads(request_path.read_text(encoding="utf-8"))
                    handler(request_path, payload)
                except BaseException as exc:
                    errors.append(exc)
                return
            time.sleep(0.002)
        errors.append(TimeoutError("live retrieval request was not produced"))

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    return thread, errors


def test_live_config_requires_enabled_current_real_directories(tmp_path: Path) -> None:
    """Disabled, stale, or missing queue directories cannot advertise a live service."""
    state_path = tmp_path / "state" / "current_state.json"
    request_dir = tmp_path / "requests"
    response_dir = tmp_path / "responses"

    assert (
        load_live_service_config(
            state_path,
            base_dir=tmp_path,
            default_request_dir=request_dir,
            default_response_dir=response_dir,
            minimum_timeout_s=1.0,
            stale_after_s=60.0,
        )
        is None
    )
    write_json_atomic(state_path, {"live_retrieval_available": False})
    assert (
        load_live_service_config(
            state_path,
            base_dir=tmp_path,
            default_request_dir=request_dir,
            default_response_dir=response_dir,
            minimum_timeout_s=1.0,
            stale_after_s=60.0,
        )
        is None
    )
    write_json_atomic(
        state_path,
        {"live_retrieval_available": True, "timestamp": time.time() - 120.0},
    )
    assert (
        load_live_service_config(
            state_path,
            base_dir=tmp_path,
            default_request_dir=request_dir,
            default_response_dir=response_dir,
            minimum_timeout_s=1.0,
            stale_after_s=60.0,
        )
        is None
    )

    write_json_atomic(state_path, {"live_retrieval_available": True, "timestamp": 0.0})
    assert (
        load_live_service_config(
            state_path,
            base_dir=tmp_path,
            default_request_dir=request_dir,
            default_response_dir=response_dir,
            minimum_timeout_s=1.0,
            stale_after_s=60.0,
        )
        is None
    )
    request_dir.mkdir()
    assert (
        load_live_service_config(
            state_path,
            base_dir=tmp_path,
            default_request_dir=request_dir,
            default_response_dir=response_dir,
            minimum_timeout_s=1.0,
            stale_after_s=60.0,
        )
        is None
    )


def test_live_config_resolves_relative_paths_and_metadata(tmp_path: Path) -> None:
    """A current manifest resolves its real queues and enforces the timeout floor."""
    state_path = tmp_path / "state" / "current_state.json"
    request_dir = tmp_path / "queues" / "requests"
    response_dir = tmp_path / "queues" / "responses"
    request_dir.mkdir(parents=True)
    response_dir.mkdir(parents=True)
    write_json_atomic(
        state_path,
        {
            "live_retrieval_available": True,
            "timestamp": time.time(),
            "live_retrieval_request_dir": "queues/requests",
            "live_retrieval_response_dir": "queues/responses",
            "live_retrieval_timeout_s": 0.1,
            "live_retrieval_transport": "filesystem-v2",
            "cycle": 42,
            "n_neurons": 512,
        },
    )

    config = load_live_service_config(
        state_path,
        base_dir=tmp_path,
        default_request_dir=tmp_path / "unused-requests",
        default_response_dir=tmp_path / "unused-responses",
        minimum_timeout_s=1.0,
        stale_after_s=60.0,
    )

    assert config == {
        "request_dir": request_dir.resolve(),
        "response_dir": response_dir.resolve(),
        "timeout_s": 1.0,
        "transport": "filesystem-v2",
        "cycle": 42,
        "n_neurons": 512,
    }

    write_json_atomic(
        state_path,
        {"live_retrieval_available": True, "live_retrieval_timeout_s": 2.5},
    )
    default_requests = tmp_path / "default-requests"
    default_responses = tmp_path / "default-responses"
    default_requests.mkdir()
    default_responses.mkdir()
    defaults = load_live_service_config(
        state_path,
        base_dir=tmp_path,
        default_request_dir=default_requests,
        default_response_dir=default_responses,
        minimum_timeout_s=1.0,
        stale_after_s=60.0,
    )
    assert defaults is not None
    assert defaults["timeout_s"] == 2.5
    assert defaults["transport"] == "filesystem"


def test_live_retrieval_completes_real_producer_consumer_roundtrip(tmp_path: Path) -> None:
    """A consumer reads the real request and atomically writes the matching response."""
    config = _config(tmp_path)
    expected_results = [
        {"trace": "tokamak-control.md", "score": 0.92, "content": "measured result"}
    ]

    def respond(request_path: Path, payload: dict[str, object]) -> None:
        assert payload["query"] == "tokamak disruption"
        assert payload["top_k"] == 3
        assert payload["include_content"] is True
        assert isinstance(payload["created_at"], float)
        response_path = config["response_dir"] / request_path.name
        write_json_atomic(
            response_path,
            {"status": "ok", "results": expected_results},
        )

    consumer, errors = _start_consumer(config, respond)
    results = retrieve_via_live_service(
        config,
        query="tokamak disruption",
        top_k=3,
        include_content=True,
        minimum_wait_s=0.01,
        poll_interval_s=0.002,
    )
    consumer.join(timeout=2.0)

    assert errors == []
    assert results == expected_results
    assert list(config["request_dir"].iterdir()) == []
    assert list(config["response_dir"].iterdir()) == []


def test_live_retrieval_retries_partial_response_then_accepts_valid_json(
    tmp_path: Path,
) -> None:
    """A partially written response is retried until the consumer replaces it atomically."""
    config = _config(tmp_path)

    def respond(request_path: Path, payload: dict[str, object]) -> None:
        response_path = config["response_dir"] / request_path.name
        response_path.write_text("{partial", encoding="utf-8")
        time.sleep(0.02)
        write_json_atomic(response_path, {"status": "ok", "results": [{"trace": "ok.md"}]})

    consumer, errors = _start_consumer(config, respond)
    results = retrieve_via_live_service(
        config,
        query="memory",
        top_k=1,
        include_content=False,
        minimum_wait_s=0.01,
        poll_interval_s=0.002,
    )
    consumer.join(timeout=2.0)

    assert errors == []
    assert results == [{"trace": "ok.md"}]


def test_live_retrieval_rejects_protocol_error_response(tmp_path: Path) -> None:
    """A syntactically valid non-success response cleans up and returns no results."""
    config = _config(tmp_path)

    def respond(request_path: Path, payload: dict[str, object]) -> None:
        write_json_atomic(
            config["response_dir"] / request_path.name,
            {"status": "error", "results": []},
        )

    consumer, errors = _start_consumer(config, respond)
    result = retrieve_via_live_service(
        config,
        query="unavailable",
        top_k=2,
        include_content=False,
        minimum_wait_s=0.01,
        poll_interval_s=0.002,
    )
    consumer.join(timeout=2.0)

    assert errors == []
    assert result is None


def test_live_retrieval_timeout_cleans_request_and_tolerates_unlink_error(
    tmp_path: Path,
) -> None:
    """Timeout cleanup removes its request and tolerates a non-file response path."""
    config = _config(tmp_path, timeout_s=0.02)

    def create_non_file_response(request_path: Path, payload: dict[str, object]) -> None:
        (config["response_dir"] / request_path.name).mkdir()

    consumer, errors = _start_consumer(config, create_non_file_response)
    result = retrieve_via_live_service(
        config,
        query="timeout",
        top_k=1,
        include_content=False,
        minimum_wait_s=0.01,
        poll_interval_s=0.002,
    )
    consumer.join(timeout=2.0)

    assert errors == []
    assert result is None
    assert list(config["request_dir"].iterdir()) == []
    response_entries = list(config["response_dir"].iterdir())
    assert len(response_entries) == 1
    assert response_entries[0].is_dir()


def test_live_retrieval_returns_none_when_request_directory_is_not_a_directory(
    tmp_path: Path,
) -> None:
    """A real filesystem parent collision exercises the request-write failure path."""
    request_parent = tmp_path / "request-parent"
    request_parent.write_text("occupied by a file", encoding="utf-8")
    response_dir = tmp_path / "responses"
    response_dir.mkdir()
    config: LiveServiceConfig = {
        "request_dir": request_parent,
        "response_dir": response_dir,
        "timeout_s": 0.01,
        "transport": "filesystem",
        "cycle": None,
        "n_neurons": None,
    }

    assert (
        retrieve_via_live_service(
            config,
            query="cannot write",
            top_k=1,
            include_content=False,
            minimum_wait_s=0.01,
            poll_interval_s=0.002,
        )
        is None
    )
