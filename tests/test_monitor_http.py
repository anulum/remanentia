# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real TCP tests for the monitor dashboard

"""Exercise the production monitor handler through a real localhost HTTP server."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from REMANENTIA.monitor import DASHBOARD_PATH, MonitorHandler  # type: ignore[import]


@contextmanager
def running_monitor() -> Iterator[str]:
    """Run the production request handler on an ephemeral localhost port."""
    server = HTTPServer(("127.0.0.1", 0), MonitorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = server.server_address[0]
        port = server.server_address[1]
        assert isinstance(host, str)
        assert isinstance(port, int)
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def read_url(url: str) -> tuple[int, str, bytes]:
    """Read status, content type, and body through the stdlib HTTP client."""
    with urlopen(url, timeout=5) as response:
        return response.status, response.headers.get_content_type(), response.read()


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_dashboard_asset_is_served_over_real_http(path: str) -> None:
    """Both dashboard routes return the tracked static asset byte-for-byte."""
    with running_monitor() as base_url:
        status, content_type, body = read_url(base_url + path)

    assert status == 200
    assert content_type == "text/html"
    assert body == DASHBOARD_PATH.read_bytes()
    assert b"Arcane Sapience" in body
    assert b"/api/current" in body


def test_health_and_service_worker_routes_cross_real_tcp() -> None:
    """Dynamic JSON and JavaScript endpoints retain their production protocols."""
    with running_monitor() as base_url:
        health_status, health_type, health_body = read_url(base_url + "/api/health")
        sw_status, sw_type, sw_body = read_url(base_url + "/sw.js")

    health = json.loads(health_body)
    assert health_status == 200
    assert health_type == "application/json"
    assert health["status"] == "ok"
    assert health["version"] == "4.0"
    assert isinstance(health["uptime_s"], int)
    assert sw_status == 200
    assert sw_type == "application/javascript"
    assert b"serviceWorker" not in sw_body
    assert b"addEventListener('fetch'" in sw_body


def test_invalid_and_empty_stimulus_requests_return_real_http_errors() -> None:
    """Malformed POST bodies are rejected before reaching stimulus persistence."""
    with running_monitor() as base_url:
        invalid = Request(
            base_url + "/api/inject_stimulus",
            data=b"not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as invalid_error:
            urlopen(invalid, timeout=5)
        invalid_body = json.loads(invalid_error.value.read())

        empty = Request(
            base_url + "/api/inject_stimulus",
            data=json.dumps({"content": ""}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as empty_error:
            urlopen(empty, timeout=5)
        empty_body = json.loads(empty_error.value.read())

    assert invalid_error.value.code == 400
    assert invalid_body == {"error": "invalid JSON"}
    assert empty_error.value.code == 400
    assert empty_body == {"error": "empty text"}


def test_unknown_get_and_post_routes_return_404_over_real_http() -> None:
    """The production handler rejects unsupported methods and paths."""
    with running_monitor() as base_url:
        with pytest.raises(HTTPError) as get_error:
            urlopen(base_url + "/not-a-route", timeout=5)
        post = Request(base_url + "/not-a-route", data=b"{}", method="POST")
        with pytest.raises(HTTPError) as post_error:
            urlopen(post, timeout=5)

    assert get_error.value.code == 404
    assert post_error.value.code == 404
