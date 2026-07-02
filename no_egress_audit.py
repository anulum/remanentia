# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — sovereign no-egress audit

"""Prove a run is sovereign: no cloud LLM call left the machine.

The "governed / sovereign / no-egress local-first" axis (roadmap W1/W3) is a
category no public memory leaderboard scores: can the system produce its answer
without any cloud model in the loop, and at what accuracy cost. To make that
claim auditable rather than asserted, every model endpoint a run touches is
classified local vs cloud and the run earns a verdict: **pure-local iff zero
cloud calls were made**.

The stance is deliberately conservative — an unknown or empty endpoint counts as
cloud. Sovereignty must be *proven* (the endpoint demonstrably points at a
loopback address or an on-device runtime), never assumed; a misconfigured run
that silently reached a cloud API must fail the audit, not pass it by default.
A network URL (http/https/ws/wss) is therefore judged by its *host alone* —
local iff the host is a loopback address; a runtime name in the URL cannot
rescue it, because ``https://ollama.com/v1`` is egress no matter what it runs.
"""

from __future__ import annotations

import ipaddress
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

EndpointClass = Literal["local", "cloud"]

# URL schemes whose endpoint is a network host: judged by loopback host only.
_NETWORK_SCHEMES = ("http", "https", "ws", "wss")

# Substrings that demonstrate an on-device runtime or loopback descriptor for
# NON-network endpoints (runtime labels, socket paths, scheme-less hosts).
# Anything not matching is treated as cloud (prove-local, not assume-local).
_LOCAL_MARKERS = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "ollama",
    "llama.cpp",
    "llamacpp",
    "file://",
    "unix:",
    "null",
    "in-process",
)


def _is_loopback_host(host: str) -> bool:
    """Whether *host* is provably this machine (loopback or unspecified)."""
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal — a DNS name like ``localhost.evil.com`` or
        # ``127.0.0.1.evil.com`` proves nothing about locality.
        return False
    return ip.is_loopback or ip.is_unspecified


def classify_endpoint(endpoint: object) -> EndpointClass:
    """Classify a model endpoint as ``local`` or ``cloud``.

    Local only when the endpoint visibly points at a loopback address or an
    on-device runtime; an empty or unrecognised endpoint is ``cloud`` so an
    unproven endpoint can never silently pass a sovereignty audit. Network
    URLs are judged by host alone — a runtime name in the hostname or path
    (``https://ollama.com/v1``, ``https://proxy.example.com/ollama``) does not
    make the call local, and a loopback *substring* in a DNS name
    (``https://localhost.evil.com``) does not either.
    """
    text = str(endpoint).strip().lower()
    if not text:
        return "cloud"
    parts = urlsplit(text)
    if parts.scheme in _NETWORK_SCHEMES:
        return "local" if _is_loopback_host(parts.hostname or "") else "cloud"
    return "local" if any(marker in text for marker in _LOCAL_MARKERS) else "cloud"


@dataclass(frozen=True)
class EgressVerdict:
    """Outcome of a no-egress audit over the endpoints a run touched."""

    pure_local: bool
    total_calls: int
    cloud_calls: int
    by_endpoint: dict[str, int]
    cloud_endpoints: tuple[str, ...]

    def __bool__(self) -> bool:
        """Truthy when the run was sovereign (no cloud call)."""
        return self.pure_local


@dataclass
class EgressMonitor:
    """Accumulate the model endpoints a run touches and render the verdict."""

    total_calls: int = 0
    _by_endpoint: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _cloud: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, endpoint: object) -> EndpointClass:
        """Record one model call to *endpoint*; return how it was classified."""
        key = str(endpoint).strip() or "(empty)"
        self.total_calls += 1
        self._by_endpoint[key] += 1
        klass = classify_endpoint(endpoint)
        if klass == "cloud":
            self._cloud[key] += 1
        return klass

    def verdict(self) -> EgressVerdict:
        """Return the sovereignty verdict for everything recorded so far."""
        cloud_calls = sum(self._cloud.values())
        return EgressVerdict(
            pure_local=cloud_calls == 0,
            total_calls=self.total_calls,
            cloud_calls=cloud_calls,
            by_endpoint=dict(self._by_endpoint),
            cloud_endpoints=tuple(sorted(self._cloud)),
        )


def audit_endpoints(endpoints: Iterable[object]) -> EgressVerdict:
    """Audit a sequence of model endpoints and return the sovereignty verdict."""
    monitor = EgressMonitor()
    for endpoint in endpoints:
        monitor.record(endpoint)
    return monitor.verdict()
