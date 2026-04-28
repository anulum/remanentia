# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — PII redaction for memory-bound content

"""Redact personally identifiable data before it lands in a memory store.

Remanentia persists arbitrary user text. Without a redaction pass, any
email, phone number, IBAN, credit-card number, or leaked API-key-shaped
string sticks on disk forever, in plain text, indexed for retrieval.
This module runs a conservative regex sweep before the memory write.

Design choices:

1. **Placeholder tokens, not blanks.** A redacted memory still reveals
   the *shape* of what was removed ("[REDACTED:EMAIL]") so the context
   stays meaningful for retrieval, while the raw value is gone. The
   LLM that later reads the memory knows an email was there without
   seeing which email.
2. **Regex-only by default.** No ML dependency, no false-positive
   cascade, no model-download. Named-entity detection via the
   (optional) `gliner` pipeline is a separate opt-in layer callers
   can compose.
3. **Configurable policy.** Callers can turn individual detectors off
   (audit trail needs emails redacted, test fixtures may not). The
   default policy errs on the side of redacting.
4. **No side effects.** :func:`redact` returns a new string and a
   dict of counts. The caller decides what to log or rate-limit on.

What this module does **not** do:

- No Luhn check on credit-card candidates (reduces false positives
  but also reduces recall; the redactor prefers over-redaction here).
- No locale-specific phone-number parsing (E.164 + permissive
  punctuation is good enough for the common cases).
- No reversible de-redaction. A redacted memory cannot be "un-redacted"
  because we never stored the original.
- No content-filtering (hate speech, illegal content). Separate concern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import import_module
from typing import Iterable


# ─── Patterns ──────────────────────────────────────────────────────────


_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)

# E.164 and common local forms. Two shapes sharing the ``[ .\-]`` separator class:
#   A) three digit groups: ``+?[country] ?2-4 2-4 2-4`` → +421 902 123 456, 555 555 5555
#   B) parenthesised area code + two body groups: ``(555) 123-4567``
# Length-bounded to keep bibliographic years (e.g. "2026") out of scope.
_PHONE = re.compile(
    r"(?<![\w@])"
    r"(?:"
    # Three groups of 3-4 digits each, optional country code. Bumping
    # the minimum from 2 to 3 digits per group keeps ISO dates like
    # ``2026-03-15`` out: those have 4+2+2 digits.
    r"(?:\+\d{1,3}[ .\-]?)?\d{3,4}[ .\-]\d{3,4}[ .\-]\d{3,4}"
    r"|"
    # Parenthesised area code (3-4 digits) + two body groups (3-4 each).
    r"\(\d{3,4}\)[ .\-]?\d{3,4}[ .\-]\d{3,4}"
    r")"
    r"(?!\w)"
)

# IBAN: 2 country letters + 2 check digits + up to 30 alphanumerics.
_IBAN = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{10,30}(?![A-Z0-9])")

# Credit card: 13-19 digits, optional spaces/hyphens in groups of four.
_CREDIT_CARD = re.compile(
    r"(?<!\d)"
    r"(?:\d{4}[ \-]?){3,4}\d{1,4}"
    r"(?!\d)"
)

# API-key-shaped strings. Conservative on length; very high entropy in
# practice but we do not measure entropy here.
_API_KEYS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # ANTHROPIC_KEY first — its ``sk-ant-api*`` prefix would otherwise
    # be shadowed by the more generic ``sk-`` pattern used by OPENAI_KEY.
    ("ANTHROPIC_KEY", re.compile(r"sk-ant-api\d\d-[A-Za-z0-9_-]{80,}")),
    ("OPENAI_KEY", re.compile(r"sk-(?:proj-|svcacct-|user-)?[A-Za-z0-9_-]{20,}")),
    ("HUGGINGFACE_KEY", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("GITHUB_PAT", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS_ACCESS_KEY", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    ("SLACK_TOKEN", re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}")),
    # Generic high-entropy token: 32+ hex chars (UUID-like) flanked by
    # non-word characters. Catches home-rolled session tokens.
    ("HEX_TOKEN", re.compile(r"(?<!\w)[a-f0-9]{32,}(?!\w)")),
)


# ─── Policy & API ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class RedactionPolicy:
    """Which detectors are active for a given call site."""

    emails: bool = True
    phones: bool = True
    iban: bool = True
    credit_cards: bool = True
    api_keys: bool = True
    # Custom patterns: list of (tag, regex) pairs. ``tag`` becomes the
    # token name in the output placeholder.
    extra: tuple[tuple[str, re.Pattern[str]], ...] = field(default_factory=tuple)


DEFAULT_POLICY = RedactionPolicy()


@dataclass
class RedactionResult:
    """Outcome of a single :func:`redact` call."""

    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _placeholder(tag: str) -> str:
    return f"[REDACTED:{tag}]"


def _try_rust_redact(
    text: str, policy: RedactionPolicy
) -> RedactionResult | None:  # pragma: no cover
    """Rust fast-path dispatch. Returns None when the extension is absent.

    Everything inside this function is covered by integration in a Rust-
    aware environment (parity tests in ``tests/test_pii_redactor.py``).
    CI does not compile the PyO3 wheel, so we exclude the whole
    function from the coverage budget rather than pepper it with
    per-line pragmas.
    """
    try:
        _rust_redact = import_module("remanentia_pii_redactor").redact
    except ImportError:
        return None

    policy_dict = {
        "emails": policy.emails,
        "phones": policy.phones,
        "iban": policy.iban,
        "credit_cards": policy.credit_cards,
        "api_keys": policy.api_keys,
    }
    text, counts_raw = _rust_redact(text, policy_dict)
    counts: dict[str, int] = {k: int(v) for k, v in counts_raw.items()}
    for tag, pat in policy.extra:
        text = _apply(text, tag, pat, counts)
    return RedactionResult(text=text, counts=counts)


def _apply(text: str, tag: str, pattern: re.Pattern[str], counts: dict[str, int]) -> str:
    def _sub(_m: re.Match[str]) -> str:
        counts[tag] = counts.get(tag, 0) + 1
        return _placeholder(tag)

    return pattern.sub(_sub, text)


def redact(text: str, policy: RedactionPolicy | None = None) -> RedactionResult:
    """Return *text* with PII replaced by ``[REDACTED:TAG]`` tokens.

    The result's ``counts`` dict reports how many matches each tag
    produced. Callers that care about the number fed to metrics /
    alerts can inspect it directly.

    Uses the Rust-accelerated ``remanentia_pii_redactor`` crate when
    available (compiled by ``maturin develop``). Falls back to the
    pure-Python path below when the crate is missing — same results,
    just slower on hot paths like memory writes.

    Custom ``policy.extra`` patterns are always handled in Python;
    the Rust side covers the fixed detector set only.
    """
    policy = policy or DEFAULT_POLICY

    # Try the Rust fast path first. The crate mirrors the detector
    # order and output format exactly, so the only difference is
    # absence of `extra` custom patterns (handled here post-hoc).
    rust_result = _try_rust_redact(text, policy)
    if rust_result is not None:
        return rust_result  # pragma: no cover

    counts: dict[str, int] = {}

    # Order matters: API keys before HEX_TOKEN (prefix-specific before
    # generic), API keys before emails (they can contain @-like bytes
    # in theory), credit cards before phones (phone regex is more
    # permissive and would swallow the grouping).
    if policy.api_keys:
        for tag, pat in _API_KEYS:
            text = _apply(text, tag, pat, counts)
    if policy.emails:
        text = _apply(text, "EMAIL", _EMAIL, counts)
    if policy.iban:
        text = _apply(text, "IBAN", _IBAN, counts)
    if policy.credit_cards:
        text = _apply(text, "CREDIT_CARD", _CREDIT_CARD, counts)
    if policy.phones:
        text = _apply(text, "PHONE", _PHONE, counts)
    for tag, pat in policy.extra:
        text = _apply(text, tag, pat, counts)

    return RedactionResult(text=text, counts=counts)


def redact_texts(
    texts: Iterable[str], policy: RedactionPolicy | None = None
) -> list[RedactionResult]:
    """Vectorised convenience wrapper for batch writers."""
    return [redact(t, policy) for t in texts]
