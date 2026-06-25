# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for pii_redactor

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from pii_redactor import (
    DEFAULT_POLICY,
    RedactionPolicy,
    redact,
    redact_texts,
)


class TestEmail:
    def test_simple_email(self):
        r = redact("Contact me at alice@example.com tomorrow.")
        assert "alice@example.com" not in r.text
        assert "[REDACTED:EMAIL]" in r.text
        assert r.counts["EMAIL"] == 1

    def test_multiple_emails(self):
        r = redact("From: a@b.co To: c@d.io CC: e@f.org")
        assert r.counts["EMAIL"] == 3
        assert r.text.count("[REDACTED:EMAIL]") == 3

    def test_dotted_local_part(self):
        r = redact("Miroslav.Sotek+flag@anulum.li picked up")
        assert "anulum.li" not in r.text

    def test_does_not_redact_bare_domain(self):
        r = redact("Visit www.anulum.li for info.")
        assert r.counts.get("EMAIL", 0) == 0
        assert "anulum.li" in r.text

    def test_no_false_positive_on_bibliographic_at(self):
        r = redact("Meeting @ 10 AM in the conference room.")
        assert r.counts.get("EMAIL", 0) == 0


class TestPhone:
    def test_e164_format(self):
        r = redact("Call +421 902 123 456 when ready.")
        assert "[REDACTED:PHONE]" in r.text
        assert "902" not in r.text

    def test_us_with_parens(self):
        r = redact("Ring (555) 123-4567 anytime.")
        assert r.counts["PHONE"] == 1

    def test_dashes_only(self):
        r = redact("555-555-5555 is the line.")
        assert r.counts["PHONE"] == 1

    def test_no_false_positive_on_date(self):
        r = redact("The review happened on 2026-03-15 in Zurich.")
        assert r.counts.get("PHONE", 0) == 0

    def test_no_false_positive_on_version(self):
        r = redact("Release v3.14.0 shipped to PyPI.")
        assert r.counts.get("PHONE", 0) == 0


class TestIBAN:
    def test_skr_iban(self):
        r = redact("Please wire to CH9300762011623852957 by Friday.")
        assert "CH9300762011623852957" not in r.text
        assert r.counts["IBAN"] == 1

    def test_de_iban(self):
        r = redact("DE89370400440532013000")
        assert r.counts["IBAN"] == 1


class TestCreditCard:
    def test_four_groups(self):
        r = redact("Charge 4111 1111 1111 1111 today.")
        assert "4111" not in r.text
        assert r.counts["CREDIT_CARD"] == 1

    def test_hyphenated(self):
        r = redact("Card: 5500-0000-0000-0004")
        assert r.counts["CREDIT_CARD"] == 1

    def test_contiguous(self):
        r = redact("Digits: 4111111111111111")
        assert r.counts["CREDIT_CARD"] == 1


class TestAPIKeys:
    def test_openai_project(self):
        key = "sk-proj-" + "A" * 40
        r = redact(f"export OPENAI_API_KEY={key}")
        assert key not in r.text
        assert r.counts["OPENAI_KEY"] == 1

    def test_anthropic(self):
        key = "sk-ant-api03-" + "B" * 95
        r = redact(key)
        assert r.counts["ANTHROPIC_KEY"] == 1

    def test_huggingface(self):
        key = "hf_" + "C" * 35
        r = redact(f"token = '{key}'")
        assert r.counts["HUGGINGFACE_KEY"] == 1

    def test_github_pat(self):
        key = "ghp_" + "D" * 35
        r = redact(key)
        assert r.counts["GITHUB_PAT"] == 1

    def test_aws_access_key(self):
        r = redact("AWS: AKIAIOSFODNN7EXAMPLE = our access key")
        assert "AKIAIOSFODNN7EXAMPLE" not in r.text
        assert r.counts["AWS_ACCESS_KEY"] == 1

    def test_slack_token(self):
        r = redact("xoxb-12345-abcdef-ghi")
        assert r.counts["SLACK_TOKEN"] == 1

    def test_hex_token(self):
        r = redact("session=" + "a" * 32)
        assert r.counts["HEX_TOKEN"] == 1

    def test_short_hex_not_redacted(self):
        r = redact("commit abc1234")
        assert r.counts.get("HEX_TOKEN", 0) == 0


class TestPolicy:
    def test_disable_emails(self):
        pol = RedactionPolicy(emails=False)
        r = redact("alice@example.com", pol)
        assert "[REDACTED" not in r.text

    def test_disable_phones(self):
        pol = RedactionPolicy(phones=False)
        r = redact("+421 902 123 456", pol)
        assert "[REDACTED" not in r.text

    def test_extra_pattern(self):
        pol = RedactionPolicy(
            emails=False,
            phones=False,
            iban=False,
            credit_cards=False,
            api_keys=False,
            extra=(("EMPLOYEE_ID", re.compile(r"EMP-\d{6}")),),
        )
        r = redact("Looking up EMP-123456 in HR.", pol)
        assert "EMP-123456" not in r.text
        assert r.counts["EMPLOYEE_ID"] == 1

    def test_default_policy_all_on(self):
        assert DEFAULT_POLICY.emails
        assert DEFAULT_POLICY.phones
        assert DEFAULT_POLICY.iban
        assert DEFAULT_POLICY.credit_cards
        assert DEFAULT_POLICY.api_keys


class TestResult:
    def test_total_property(self):
        r = redact("a@b.co +421 902 123 456")
        assert r.total == r.counts["EMAIL"] + r.counts["PHONE"]

    def test_empty_text(self):
        r = redact("")
        assert r.text == ""
        assert r.counts == {}

    def test_no_pii(self):
        r = redact("The sky is blue and the grass is green.")
        assert r.text == "The sky is blue and the grass is green."
        assert r.counts == {}

    def test_preserves_shape(self):
        r = redact("Email alice@example.com and call 555-555-5555.")
        # Structure preserved: sentence still has 'and'
        assert "and" in r.text
        assert r.text.endswith(".")


class TestBatch:
    def test_vectorised(self):
        results = redact_texts(["a@b.co", "555-555-5555", "plain"])
        assert len(results) == 3
        assert results[0].counts["EMAIL"] == 1
        assert results[1].counts["PHONE"] == 1
        assert results[2].counts == {}


class TestOrdering:
    def test_api_key_redacted_before_hex_catches_it(self):
        """``sk-`` prefixed keys should be tagged OPENAI_KEY, not HEX_TOKEN."""
        key = "sk-proj-" + "f" * 32
        r = redact(key)
        assert r.counts.get("OPENAI_KEY", 0) == 1
        assert r.counts.get("HEX_TOKEN", 0) == 0


class TestRustPythonParity:
    """Rust and Python paths must return identical counts for the fixed detector set."""

    @staticmethod
    def _available() -> bool:
        try:
            import remanentia_pii_redactor  # noqa: F401
        except ImportError:
            return False
        return True

    def _parity(self, text: str) -> None:
        if not self._available():
            pytest.skip("Rust crate not built; maturin develop required for parity test")
        # Force Python path via a policy that disables all Rust-side detectors
        # then compare against Rust-enabled via fresh call.
        rust_result = redact(text)
        # Python-only path: temporarily hide the Rust module.
        import sys

        real = sys.modules.pop("remanentia_pii_redactor", None)
        try:
            # Also poison it so re-import inside redact() fails.
            sys.modules["remanentia_pii_redactor"] = None
            py_result = redact(text)
        finally:
            if real is not None:
                sys.modules["remanentia_pii_redactor"] = real
            else:
                sys.modules.pop("remanentia_pii_redactor", None)

        assert rust_result.counts == py_result.counts, (
            f"counts differ: rust={rust_result.counts} py={py_result.counts}"
        )
        assert rust_result.text == py_result.text, (
            f"text differs: rust={rust_result.text!r} py={py_result.text!r}"
        )

    def test_parity_email(self):
        self._parity("Contact alice@example.com tomorrow.")

    def test_parity_phone(self):
        self._parity("Call +421 902 123 456 when ready.")

    def test_parity_api_key_anthropic(self):
        self._parity(f"key=sk-ant-api03-{'B' * 95} end")

    def test_parity_iban(self):
        self._parity("wire CH9300762011623852957 now")

    def test_parity_credit_card(self):
        self._parity("charge 4111 1111 1111 1111 today")

    def test_parity_mixed(self):
        self._parity(
            "Contact alice@example.com or +421 902 123 456. "
            "Wire CH9300762011623852957. Card 4111 1111 1111 1111."
        )

    def test_parity_no_pii(self):
        self._parity("The quick brown fox jumps over the lazy dog.")

    def test_parity_utf8(self):
        self._parity("Tokyo ¥500 🎯 alice@example.com")


class TestCustomPolicy:
    def test_extra_patterns_are_redacted_and_counted(self):
        policy = RedactionPolicy(extra=(("PROJECT", re.compile(r"\bProject\s+Artemis\b")),))

        with patch("pii_redactor._try_rust_redact", return_value=None):
            result = redact("Project Artemis owner is alice@example.com", policy)

        assert "Project Artemis" not in result.text
        assert result.counts["PROJECT"] == 1
        assert result.counts["EMAIL"] == 1
