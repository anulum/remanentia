# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — API security primitives

"""Authentication, rate limiting, and request-size guards for Remanentia servers.

Designed to be reused across `api_server.py`, `mcp_server.py`, and any
other Remanentia network surface. No external dependencies (stdlib-only)
so the `api` optional-dependency remains minimal.

Three primitives:

- :class:`BearerAuth` — constant-time bearer-token check.
- :class:`TokenBucketLimiter` — per-key (IP) token bucket.
- :func:`enforce_body_size` — reject requests exceeding a byte cap.

All three are defensive by default: when a token is not configured,
:class:`BearerAuth` emits a loud warning to stderr on construction and
then passes every request. This keeps local development frictionless
while making "I forgot to set the token" visible in operator logs.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import sys
import threading
import time
from pathlib import Path


class BearerAuth:
    """Constant-time bearer-token check.

    Parameters
    ----------
    token:
        The expected token. If None or empty, auth is disabled and a
        warning is emitted once to stderr. Use :meth:`from_env` or
        :meth:`from_file` for production loading.
    warn_on_disabled:
        Emit a one-line warning to stderr when token is missing.
        Disable for tests that intentionally construct an open instance.
    """

    _WARNED: set[int] = set()  # avoid repeated warnings in same process

    def __init__(self, token: str | None, *, warn_on_disabled: bool = True):
        self._token = token or None
        if self._token is None and warn_on_disabled and id(self) not in self._WARNED:
            print(
                "[SECURITY] REMANENTIA_API_TOKEN is not set — API auth is DISABLED.",
                file=sys.stderr,
                flush=True,
            )
            self._WARNED.add(id(self))

    @classmethod
    def from_env(cls, var: str = "REMANENTIA_API_TOKEN") -> BearerAuth:
        return cls(os.environ.get(var))

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> BearerAuth:
        with open(path, encoding="utf-8") as f:
            token = f.read().strip()
        if not token:
            raise ValueError(f"token file {path!s} is empty")
        return cls(token)

    @property
    def enabled(self) -> bool:
        return self._token is not None

    def check_header(self, authorization_header: str | None) -> bool:
        """Return True iff the header holds a matching Bearer token.

        If auth is disabled, every call returns True. When enabled, the
        header must be exactly ``Bearer <token>`` and the comparison is
        constant-time to avoid timing oracles.
        """
        token = self._token
        if token is None:
            return True
        if not authorization_header:
            return False
        prefix = "Bearer "
        if not authorization_header.startswith(prefix):
            return False
        candidate = authorization_header[len(prefix) :]
        if not candidate:
            return False
        return hmac.compare_digest(candidate, token)


class TokenBucketLimiter:
    """Per-key token-bucket rate limiter (thread-safe, in-memory).

    Tokens refill at ``rate_per_second``; capacity is ``burst``. Each
    :meth:`allow` call spends one token and returns True if a token was
    available. Keys not seen for :attr:`ttl_seconds` are evicted.

    Parameters
    ----------
    rate_per_minute:
        Steady-state rate. Default 60 (one per second).
    burst:
        Maximum tokens in the bucket. Default 10.
    ttl_seconds:
        Evict bucket state for keys inactive for this long. Default 3600.

    The limiter is **not** a distributed rate limiter: each process has
    its own state. Use a shared backend (Redis, memcached) for
    multi-process deployments.
    """

    def __init__(
        self,
        *,
        rate_per_minute: float = 60.0,
        burst: int = 10,
        ttl_seconds: float = 3600.0,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")
        self._rate_per_minute = rate_per_minute
        self._rate = rate_per_minute / 60.0
        self._burst = float(burst)
        self._ttl = ttl_seconds
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Try to spend one token for *key*. Return True on success."""
        t = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key, (self._burst, t))
            tokens = min(self._burst, tokens + (t - last) * self._rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, t)
                return True
            self._buckets[key] = (tokens, t)
            return False

    def peek(self, key: str) -> float:
        """Return the current token count for *key* without spending."""
        t = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self._burst, t))
            return min(self._burst, tokens + (t - last) * self._rate)

    def evict_stale(self, *, now: float | None = None) -> int:
        """Drop buckets idle longer than ``ttl_seconds``. Returns count."""
        t = time.monotonic() if now is None else now
        cutoff = t - self._ttl
        with self._lock:
            stale = [k for k, (_, last) in self._buckets.items() if last < cutoff]
            for k in stale:
                del self._buckets[k]
            return len(stale)

    def retry_after_seconds(self) -> str:
        """Return a conservative whole-second wait before retrying."""
        return retry_after_seconds(self._rate_per_minute)


def enforce_body_size(declared_length: int, limit_bytes: int) -> None:
    """Raise :class:`ValueError` if *declared_length* exceeds *limit_bytes*.

    The caller is expected to read *Content-Length* from the request
    headers, pass it here, and return HTTP 413 to the client on
    :class:`ValueError`. Refusing on the declared length (before reading
    the body) is the whole point — a hostile client can otherwise stream
    arbitrary bytes.
    """
    if limit_bytes < 0:
        raise ValueError("limit_bytes must be non-negative")
    if declared_length < 0:
        raise ValueError("declared_length must be non-negative")
    if declared_length > limit_bytes:
        raise ValueError(f"request body {declared_length} B exceeds limit {limit_bytes} B")


def retry_after_seconds(rate_per_minute: float) -> str:
    """Return a whole-second Retry-After value for one replenished token."""
    if rate_per_minute <= 0:
        raise ValueError("rate_per_minute must be positive")
    return str(max(1, math.ceil(60.0 / rate_per_minute)))


def _disabled_env_value(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"", "0", "false", "off", "no"}


def _read_int_env(var: str, default: int) -> int:
    value = os.environ.get(var)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{var} must be non-negative")
    return parsed


class _RotatingJsonlWriter:
    """Thread-safe append-only JSONL writer with size-capped backups."""

    def __init__(
        self,
        path: str | os.PathLike | None,
        *,
        max_bytes: int = 0,
        backups: int = 0,
    ) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if backups < 0:
            raise ValueError("backups must be non-negative")
        self.path = Path(path) if path else None
        self.max_bytes = int(max_bytes)
        self.backups = int(backups)
        self._lock = threading.Lock()

    def write_payload(self, payload: dict[str, object]) -> None:
        if self.path is None:
            return
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        encoded_size = len(line.encode("utf-8"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(encoded_size)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if self.path is None or self.max_bytes <= 0 or not self.path.exists():
            return
        current_size = self.path.stat().st_size
        if current_size + incoming_bytes <= self.max_bytes:
            return
        if self.backups <= 0:
            self.path.unlink()
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
        if oldest.exists():
            oldest.unlink()
        for idx in range(self.backups - 1, 0, -1):
            src = self.path.with_name(f"{self.path.name}.{idx}")
            if src.exists():
                src.replace(self.path.with_name(f"{self.path.name}.{idx + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))


class RequestAuditLogger:
    """Append-only JSONL audit logger for API request metadata.

    The logger records routing and response metadata only. It deliberately
    has no fields for request bodies or authorisation headers.
    """

    def __init__(
        self,
        path: str | os.PathLike | None,
        *,
        max_bytes: int = 0,
        backups: int = 0,
    ):
        self._writer = _RotatingJsonlWriter(path, max_bytes=max_bytes, backups=backups)

    @classmethod
    def from_env(
        cls,
        default_path: str | os.PathLike,
        *,
        var: str = "REMANENTIA_API_AUDIT_LOG",
        max_bytes_var: str = "REMANENTIA_API_AUDIT_MAX_BYTES",
        backups_var: str = "REMANENTIA_API_AUDIT_BACKUPS",
    ) -> RequestAuditLogger:
        configured = os.environ.get(var)
        if _disabled_env_value(configured):
            return cls(None)
        return cls(
            configured.strip() if configured else default_path,
            max_bytes=_read_int_env(max_bytes_var, 0),
            backups=_read_int_env(backups_var, 0),
        )

    @property
    def path(self) -> Path | None:
        return self._writer.path

    @property
    def max_bytes(self) -> int:
        return self._writer.max_bytes

    @property
    def backups(self) -> int:
        return self._writer.backups

    def record(
        self,
        *,
        server: str,
        method: str,
        path: str,
        client: str,
        status: int,
        outcome: str,
        auth_enabled: bool,
    ) -> None:
        payload = {
            "timestamp_unix": time.time(),
            "server": server,
            "method": method,
            "path": path,
            "client": client,
            "status": int(status),
            "outcome": outcome,
            "auth_enabled": bool(auth_enabled),
        }
        self._writer.write_payload(payload)


class ToolAuditLogger:
    """Append-only JSONL audit logger for MCP tool-call metadata.

    The logger records invocation metadata only. It deliberately stores
    argument names instead of argument values so user memory content,
    prompts, and credentials cannot land in the audit stream.
    """

    def __init__(
        self,
        path: str | os.PathLike | None,
        *,
        max_bytes: int = 0,
        backups: int = 0,
    ):
        self._writer = _RotatingJsonlWriter(path, max_bytes=max_bytes, backups=backups)

    @classmethod
    def from_env(
        cls,
        default_path: str | os.PathLike,
        *,
        var: str = "REMANENTIA_MCP_AUDIT_LOG",
        max_bytes_var: str = "REMANENTIA_MCP_AUDIT_MAX_BYTES",
        backups_var: str = "REMANENTIA_MCP_AUDIT_BACKUPS",
    ) -> ToolAuditLogger:
        configured = os.environ.get(var)
        if _disabled_env_value(configured):
            return cls(None)
        return cls(
            configured.strip() if configured else default_path,
            max_bytes=_read_int_env(max_bytes_var, 0),
            backups=_read_int_env(backups_var, 0),
        )

    @property
    def path(self) -> Path | None:
        return self._writer.path

    @property
    def max_bytes(self) -> int:
        return self._writer.max_bytes

    @property
    def backups(self) -> int:
        return self._writer.backups

    def record(
        self,
        *,
        server: str,
        method: str,
        tool: str,
        request_id: str,
        argument_keys: list[str],
        outcome: str,
        duration_ms: float,
        error_type: str = "",
    ) -> None:
        payload = {
            "timestamp_unix": time.time(),
            "server": server,
            "method": method,
            "tool": tool,
            "request_id": request_id,
            "argument_keys": sorted(argument_keys),
            "outcome": outcome,
            "duration_ms": round(float(duration_ms), 3),
        }
        if error_type:
            payload["error_type"] = error_type
        self._writer.write_payload(payload)


DEFAULT_BODY_LIMIT = 1 * 1024 * 1024  # 1 MiB
DEFAULT_RATE_PER_MINUTE = 60.0
DEFAULT_BURST = 10
