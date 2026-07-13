# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Live filesystem retrieval protocol

"""Request live retrieval through durable filesystem producer/consumer queues."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TypedDict, cast

try:
    from .retrieval_network_io import (  # type: ignore[import-not-found]
        read_json,
        resolve_path,
        write_json_atomic,
    )
except ImportError:
    from retrieval_network_io import read_json, resolve_path, write_json_atomic


class LiveServiceConfig(TypedDict):
    """Validated live-retrieval filesystem transport configuration."""

    request_dir: Path
    response_dir: Path
    timeout_s: float
    transport: str
    cycle: object
    n_neurons: object


def load_live_service_config(
    state_path: Path,
    *,
    base_dir: Path,
    default_request_dir: Path,
    default_response_dir: Path,
    minimum_timeout_s: float,
    stale_after_s: float,
) -> LiveServiceConfig | None:
    """Load a live service config only when its state and directories are current."""
    state = read_json(state_path)
    if not state or not state.get("live_retrieval_available"):
        return None
    timestamp = float(state.get("timestamp", 0.0) or 0.0)
    if timestamp and time.time() - timestamp > stale_after_s:
        return None
    request_dir = resolve_path(
        str(state.get("live_retrieval_request_dir") or default_request_dir),
        base_dir=base_dir,
    )
    response_dir = resolve_path(
        str(state.get("live_retrieval_response_dir") or default_response_dir),
        base_dir=base_dir,
    )
    if not request_dir.is_dir() or not response_dir.is_dir():
        return None
    advertised_timeout = float(state.get("live_retrieval_timeout_s", minimum_timeout_s))
    return {
        "request_dir": request_dir,
        "response_dir": response_dir,
        "timeout_s": max(advertised_timeout, minimum_timeout_s),
        "transport": str(state.get("live_retrieval_transport", "filesystem")),
        "cycle": state.get("cycle"),
        "n_neurons": state.get("n_neurons"),
    }


def _remove_protocol_files(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def retrieve_via_live_service(
    config: LiveServiceConfig,
    *,
    query: str,
    top_k: int,
    include_content: bool,
    minimum_wait_s: float = 1.0,
    poll_interval_s: float = 0.05,
) -> list[dict[str, object]] | None:
    """Complete one request/response transaction through real filesystem queues."""
    request_id = hashlib.md5(
        f"{query}:{top_k}:{include_content}:{time.time_ns()}".encode()
    ).hexdigest()
    request_path = config["request_dir"] / f"{request_id}.json"
    response_path = config["response_dir"] / f"{request_id}.json"
    payload: dict[str, object] = {
        "id": request_id,
        "query": query,
        "top_k": int(top_k),
        "include_content": bool(include_content),
        "created_at": time.time(),
    }
    try:
        write_json_atomic(request_path, payload)
    except OSError:
        return None

    deadline = time.time() + max(config["timeout_s"], minimum_wait_s)
    while time.time() < deadline:
        if response_path.exists():
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(poll_interval_s)
                continue
            _remove_protocol_files(response_path, request_path)
            if (
                isinstance(response, dict)
                and response.get("status") == "ok"
                and isinstance(response.get("results"), list)
            ):
                return cast(list[dict[str, object]], response["results"])
            return None
        time.sleep(poll_interval_s)

    _remove_protocol_files(request_path, response_path)
    return None
