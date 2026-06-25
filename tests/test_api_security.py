# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for api_security primitives

from __future__ import annotations

import json
import os
import tempfile

import pytest

from api_security import (
    DEFAULT_BODY_LIMIT,
    DEFAULT_BURST,
    DEFAULT_RATE_PER_MINUTE,
    BearerAuth,
    RequestAuditLogger,
    ToolAuditLogger,
    TokenBucketLimiter,
    _RotatingJsonlWriter,
    _read_int_env,
    enforce_body_size,
    retry_after_seconds,
)


# ── BearerAuth ───────────────────────────────────────────────────────


class TestBearerAuth:
    def test_disabled_when_token_none(self):
        a = BearerAuth(None, warn_on_disabled=False)
        assert not a.enabled
        assert a.check_header(None) is True
        assert a.check_header("Bearer anything") is True  # open

    def test_enabled_when_token_set(self):
        a = BearerAuth("secret", warn_on_disabled=False)
        assert a.enabled

    def test_correct_bearer_passes(self):
        a = BearerAuth("secret", warn_on_disabled=False)
        assert a.check_header("Bearer secret") is True

    def test_wrong_token_fails(self):
        a = BearerAuth("secret", warn_on_disabled=False)
        assert a.check_header("Bearer wrong") is False

    def test_missing_header_fails(self):
        a = BearerAuth("secret", warn_on_disabled=False)
        assert a.check_header(None) is False
        assert a.check_header("") is False

    def test_non_bearer_scheme_fails(self):
        a = BearerAuth("secret", warn_on_disabled=False)
        assert a.check_header("Basic secret") is False
        assert a.check_header("Token secret") is False

    def test_empty_bearer_value_fails(self):
        a = BearerAuth("secret", warn_on_disabled=False)
        assert a.check_header("Bearer ") is False

    def test_constant_time_compare(self):
        """Smoke: we use hmac.compare_digest, not ==."""
        a = BearerAuth("a" * 64, warn_on_disabled=False)
        # correctness only; timing is not testable here deterministically
        assert a.check_header("Bearer " + "a" * 64) is True
        assert a.check_header("Bearer " + "a" * 63 + "b") is False

    def test_from_env_unset(self, monkeypatch):
        monkeypatch.delenv("REMANENTIA_API_TOKEN", raising=False)
        a = BearerAuth.from_env()
        assert not a.enabled

    def test_from_env_set(self, monkeypatch):
        monkeypatch.setenv("REMANENTIA_API_TOKEN", "tok")
        a = BearerAuth.from_env()
        assert a.enabled
        assert a.check_header("Bearer tok") is True

    def test_from_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tok") as f:
            f.write("file-token-123\n")
            path = f.name
        try:
            a = BearerAuth.from_file(path)
            assert a.enabled
            assert a.check_header("Bearer file-token-123") is True
        finally:
            os.unlink(path)

    def test_from_file_empty(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tok") as f:
            f.write("")
            path = f.name
        try:
            with pytest.raises(ValueError):
                BearerAuth.from_file(path)
        finally:
            os.unlink(path)


# ── TokenBucketLimiter ───────────────────────────────────────────────


class TestTokenBucketLimiter:
    def test_rejects_invalid_rate(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(rate_per_minute=0)
        with pytest.raises(ValueError):
            TokenBucketLimiter(rate_per_minute=-1)

    def test_rejects_invalid_burst(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(burst=0)
        with pytest.raises(ValueError):
            TokenBucketLimiter(burst=-5)

    def test_first_burst_allowed(self):
        lim = TokenBucketLimiter(rate_per_minute=60, burst=5)
        for _ in range(5):
            assert lim.allow("ip-a") is True
        # 6th is refused (no time advanced, no refill)
        assert lim.allow("ip-a", now=0.0) is False

    def test_refill_over_time(self):
        lim = TokenBucketLimiter(rate_per_minute=60, burst=1)  # 1/s refill
        assert lim.allow("k", now=0.0) is True
        assert lim.allow("k", now=0.1) is False
        assert lim.allow("k", now=1.0) is True

    def test_keys_isolated(self):
        lim = TokenBucketLimiter(rate_per_minute=60, burst=1)
        assert lim.allow("a", now=0.0) is True
        assert lim.allow("a", now=0.0) is False
        assert lim.allow("b", now=0.0) is True  # b has own bucket

    def test_peek_does_not_spend(self):
        lim = TokenBucketLimiter(rate_per_minute=60, burst=3)
        lim.allow("k")  # spend one
        tokens = lim.peek("k")
        assert tokens > 0
        # Peek a second time — still positive (no spend)
        assert lim.peek("k") == pytest.approx(tokens, abs=0.1)

    def test_evict_stale(self):
        lim = TokenBucketLimiter(rate_per_minute=60, burst=1, ttl_seconds=10)
        lim.allow("old", now=0.0)
        lim.allow("fresh", now=1000.0)
        n = lim.evict_stale(now=1005.0)
        assert n == 1  # "old" evicted, "fresh" kept

    def test_burst_cap(self):
        lim = TokenBucketLimiter(rate_per_minute=60, burst=3)
        # Advance time to refill above burst cap
        assert lim.allow("k", now=0.0) is True  # spends → bucket 2
        # big wait: refill should cap at burst=3, not accumulate to 1000
        # so after 3 more allows with small delta, 4th fails
        for _ in range(3):
            assert lim.allow("k", now=10000.0) is True
        assert lim.allow("k", now=10000.0) is False

    def test_retry_after_uses_configured_rate(self):
        assert TokenBucketLimiter(rate_per_minute=60).retry_after_seconds() == "1"
        assert TokenBucketLimiter(rate_per_minute=30).retry_after_seconds() == "2"
        assert TokenBucketLimiter(rate_per_minute=7).retry_after_seconds() == "9"

    def test_retry_after_rejects_non_positive_rate(self):
        with pytest.raises(ValueError, match="positive"):
            retry_after_seconds(0)


class TestEnvParsing:
    def test_read_int_env_returns_default_for_unset_or_blank(self, monkeypatch):
        monkeypatch.delenv("REMANENTIA_TEST_INT", raising=False)
        assert _read_int_env("REMANENTIA_TEST_INT", 7) == 7

        monkeypatch.setenv("REMANENTIA_TEST_INT", " ")
        assert _read_int_env("REMANENTIA_TEST_INT", 7) == 7

    def test_read_int_env_rejects_negative_values(self, monkeypatch):
        monkeypatch.setenv("REMANENTIA_TEST_INT", "-1")
        with pytest.raises(ValueError, match="non-negative"):
            _read_int_env("REMANENTIA_TEST_INT", 7)


# ── enforce_body_size ────────────────────────────────────────────────


class TestEnforceBodySize:
    def test_allows_small(self):
        enforce_body_size(500, 1024)  # no raise

    def test_allows_exact(self):
        enforce_body_size(1024, 1024)  # boundary: equal is ok

    def test_rejects_over(self):
        with pytest.raises(ValueError, match="exceeds limit"):
            enforce_body_size(1025, 1024)

    def test_rejects_negative_declared(self):
        with pytest.raises(ValueError):
            enforce_body_size(-1, 1024)

    def test_rejects_negative_limit(self):
        with pytest.raises(ValueError):
            enforce_body_size(100, -1)

    def test_zero_body_ok(self):
        enforce_body_size(0, 1024)


# ── RequestAuditLogger ────────────────────────────────────────────────


class TestRequestAuditLogger:
    def test_disabled_logger_writes_nothing(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        logger = RequestAuditLogger(None)
        logger.record(
            server="fastapi",
            method="GET",
            path="/status",
            client="127.0.0.1",
            status=200,
            outcome="ok",
            auth_enabled=True,
        )

        assert not path.exists()

    def test_record_writes_jsonl_without_sensitive_fields(self, tmp_path):
        path = tmp_path / "nested" / "audit.jsonl"
        logger = RequestAuditLogger(path)

        logger.record(
            server="fastapi",
            method="GET",
            path="/status",
            client="127.0.0.1",
            status=401,
            outcome="authentication_required",
            auth_enabled=True,
        )

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["server"] == "fastapi"
        assert payload["method"] == "GET"
        assert payload["path"] == "/status"
        assert payload["client"] == "127.0.0.1"
        assert payload["status"] == 401
        assert payload["outcome"] == "authentication_required"
        assert payload["auth_enabled"] is True
        assert "timestamp_unix" in payload
        assert "authorization" not in payload
        assert "body" not in payload

    def test_from_env_can_disable_logging(self, monkeypatch, tmp_path):
        monkeypatch.setenv("REMANENTIA_API_AUDIT_LOG", "off")

        logger = RequestAuditLogger.from_env(tmp_path / "audit.jsonl")

        assert logger.path is None

    def test_rotates_when_size_cap_would_be_exceeded(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        logger = RequestAuditLogger(path, max_bytes=220, backups=2)

        for idx in range(6):
            logger.record(
                server="fastapi",
                method="GET",
                path=f"/status/{idx}",
                client="127.0.0.1",
                status=200,
                outcome="ok",
                auth_enabled=True,
            )

        assert path.exists()
        assert path.with_name("audit.jsonl.1").exists()
        assert path.with_name("audit.jsonl.2").exists()
        assert not path.with_name("audit.jsonl.3").exists()
        assert path.stat().st_size <= 220
        assert path.with_name("audit.jsonl.1").stat().st_size <= 220

    def test_from_env_reads_rotation_controls(self, monkeypatch, tmp_path):
        path = tmp_path / "api_audit.jsonl"
        monkeypatch.setenv("REMANENTIA_API_AUDIT_LOG", str(path))
        monkeypatch.setenv("REMANENTIA_API_AUDIT_MAX_BYTES", "4096")
        monkeypatch.setenv("REMANENTIA_API_AUDIT_BACKUPS", "4")

        logger = RequestAuditLogger.from_env(tmp_path / "fallback.jsonl")

        assert logger.path == path
        assert logger.max_bytes == 4096
        assert logger.backups == 4


class TestRotatingJsonlWriter:
    def test_rejects_negative_rotation_controls(self, tmp_path):
        with pytest.raises(ValueError, match="max_bytes"):
            _RotatingJsonlWriter(tmp_path / "audit.jsonl", max_bytes=-1)
        with pytest.raises(ValueError, match="backups"):
            _RotatingJsonlWriter(tmp_path / "audit.jsonl", backups=-1)

    def test_rotation_without_backups_replaces_active_file(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        path.write_text("x" * 20, encoding="utf-8")
        writer = _RotatingJsonlWriter(path, max_bytes=21, backups=0)

        writer.write_payload({"event": "new"})

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {"event": "new"}

    def test_no_rotation_when_record_fits_cap(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        path.write_text('{"event":"old"}\n', encoding="utf-8")
        writer = _RotatingJsonlWriter(path, max_bytes=200, backups=1)

        writer.write_payload({"event": "new"})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["event"] for line in lines] == ["old", "new"]
        assert not path.with_name("audit.jsonl.1").exists()


# ── ToolAuditLogger ──────────────────────────────────────────────────


class TestToolAuditLogger:
    def test_disabled_logger_writes_nothing(self, tmp_path):
        path = tmp_path / "mcp_audit.jsonl"
        logger = ToolAuditLogger(None)
        logger.record(
            server="mcp",
            method="tools/call",
            tool="remanentia_recall",
            request_id="1",
            argument_keys=["query"],
            outcome="ok",
            duration_ms=1.0,
        )

        assert not path.exists()

    def test_record_writes_argument_names_without_values(self, tmp_path):
        path = tmp_path / "nested" / "mcp_audit.jsonl"
        logger = ToolAuditLogger(path)
        logger.record(
            server="mcp",
            method="tools/call",
            tool="remanentia_remember",
            request_id="abc",
            argument_keys=["content", "project"],
            outcome="ok",
            duration_ms=2.5,
        )

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["server"] == "mcp"
        assert payload["method"] == "tools/call"
        assert payload["tool"] == "remanentia_remember"
        assert payload["request_id"] == "abc"
        assert payload["argument_keys"] == ["content", "project"]
        assert payload["outcome"] == "ok"
        assert payload["duration_ms"] == 2.5
        assert "timestamp_unix" in payload
        assert "arguments" not in payload
        assert "authorization" not in payload
        assert "body" not in payload

    def test_record_includes_error_type_when_present(self, tmp_path):
        path = tmp_path / "mcp_audit.jsonl"
        logger = ToolAuditLogger(path)

        logger.record(
            server="mcp",
            method="tools/call",
            tool="remanentia_recall",
            request_id="err",
            argument_keys=["query"],
            outcome="error",
            duration_ms=3.25,
            error_type="ValueError",
        )

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["error_type"] == "ValueError"

    def test_from_env_can_disable_logging(self, monkeypatch, tmp_path):
        monkeypatch.setenv("REMANENTIA_MCP_AUDIT_LOG", "off")

        logger = ToolAuditLogger.from_env(tmp_path / "mcp_audit.jsonl")

        assert logger.path is None

    def test_rotates_when_size_cap_would_be_exceeded(self, tmp_path):
        path = tmp_path / "mcp_audit.jsonl"
        logger = ToolAuditLogger(path, max_bytes=240, backups=1)

        for idx in range(5):
            logger.record(
                server="mcp",
                method="tools/call",
                tool="remanentia_recall",
                request_id=str(idx),
                argument_keys=["query", "top_k"],
                outcome="ok",
                duration_ms=1.0,
            )

        assert path.exists()
        assert path.with_name("mcp_audit.jsonl.1").exists()
        assert not path.with_name("mcp_audit.jsonl.2").exists()
        assert path.stat().st_size <= 240

    def test_from_env_reads_rotation_controls(self, monkeypatch, tmp_path):
        path = tmp_path / "mcp_audit.jsonl"
        monkeypatch.setenv("REMANENTIA_MCP_AUDIT_LOG", str(path))
        monkeypatch.setenv("REMANENTIA_MCP_AUDIT_MAX_BYTES", "8192")
        monkeypatch.setenv("REMANENTIA_MCP_AUDIT_BACKUPS", "3")

        logger = ToolAuditLogger.from_env(tmp_path / "fallback.jsonl")

        assert logger.path == path
        assert logger.max_bytes == 8192
        assert logger.backups == 3


# ── Defaults sanity ──────────────────────────────────────────────────


class TestDefaults:
    def test_body_limit_is_mib(self):
        assert DEFAULT_BODY_LIMIT == 1 * 1024 * 1024

    def test_rate_default(self):
        assert DEFAULT_RATE_PER_MINUTE == 60.0

    def test_burst_default(self):
        assert DEFAULT_BURST == 10
