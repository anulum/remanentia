# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for api_security primitives

from __future__ import annotations

import os
import tempfile

import pytest

from api_security import (
    DEFAULT_BODY_LIMIT,
    DEFAULT_BURST,
    DEFAULT_RATE_PER_MINUTE,
    BearerAuth,
    TokenBucketLimiter,
    enforce_body_size,
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


# ── Defaults sanity ──────────────────────────────────────────────────


class TestDefaults:
    def test_body_limit_is_mib(self):
        assert DEFAULT_BODY_LIMIT == 1 * 1024 * 1024

    def test_rate_default(self):
        assert DEFAULT_RATE_PER_MINUTE == 60.0

    def test_burst_default(self):
        assert DEFAULT_BURST == 10
