// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust-accelerated PII redactor

//! Rust implementation of Remanentia's PII redaction.
//!
//! Mirrors the detector set in `pii_redactor.py`:
//! email, phone (international + parenthesised + dashed), IBAN,
//! credit card, API-key-shaped tokens (Anthropic / OpenAI /
//! HuggingFace / GitHub / AWS / Slack / generic hex).
//!
//! UTF-8-safe throughout — replacements walk `char_indices` so
//! multi-byte sequences (¥, €, CJK, emoji) are never split.
//!
//! Detector ordering matters for correctness:
//!   1. Anthropic sk-ant-api* (must come before generic OpenAI sk-*)
//!   2. OpenAI sk-proj- / sk- generic
//!   3. HuggingFace hf_
//!   4. GitHub ghp_ / ghs_ / ...
//!   5. AWS AKIA...
//!   6. Slack xox[abprs]-
//!   7. Generic 32+ hex (fallback)
//!   8. Email
//!   9. IBAN
//!   10. Credit card
//!   11. Phone (most permissive, runs last)

use pyo3::prelude::*;
use pyo3::types::PyDict;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// ─── Patterns (mirror pii_redactor.py) ───────────────────────────────

static RE_ANTHROPIC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"sk-ant-api\d\d-[A-Za-z0-9_-]{80,}").unwrap());
static RE_OPENAI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"sk-(?:proj-|svcacct-|user-)?[A-Za-z0-9_-]{20,}").unwrap());
static RE_HUGGINGFACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"hf_[A-Za-z0-9]{30,}").unwrap());
static RE_GITHUB: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"gh[pousr]_[A-Za-z0-9]{30,}").unwrap());
// `regex` crate does not support lookaround. Where Python uses
// `(?<![\w@])` / `(?!\w)`, we rely on the pattern's structural
// specificity (e.g. `\d{3,4}[ .\-]\d{3,4}[ .\-]\d{3,4}` for phone
// naturally rejects ISO dates because `03` has 2 digits). Boundary
// anchors are kept only where they do not need lookbehind.
static RE_AWS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"AKIA[0-9A-Z]{16}").unwrap());
static RE_SLACK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"xox[abprs]-[A-Za-z0-9-]{10,}").unwrap());
static RE_HEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-f0-9]{32,}").unwrap());
static RE_EMAIL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}").unwrap());
static RE_IBAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}").unwrap());
static RE_CREDIT_CARD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:\d{4}[ \-]?){3,4}\d{1,4}").unwrap());
static RE_PHONE: LazyLock<Regex> = LazyLock::new(|| {
    // Matches Python _PHONE's structure. 3+ digits per group keeps ISO
    // dates out. Without `\b` the optional country code + digit body
    // all match as one run (the Python lookbehind-anchored version
    // would otherwise leave `+421` un-redacted).
    Regex::new(
        r"(?:(?:\+\d{1,3}[ .\-]?)?\d{3,4}[ .\-]\d{3,4}[ .\-]\d{3,4}|\(\d{3,4}\)[ .\-]?\d{3,4}[ .\-]\d{3,4})",
    )
    .unwrap()
});

// Detector entries: tag, regex, and whether this tag category is in
// the active policy. The Python side filters via `RedactionPolicy`;
// here the Python caller passes a `HashMap<&str, bool>` slice of the
// eight toggleable categories so we respect the same policy.
struct Detector {
    tag: &'static str,
    re: &'static LazyLock<Regex>,
}

static DETECTORS: LazyLock<Vec<Detector>> = LazyLock::new(|| {
    vec![
        // API keys first (order matters — Anthropic before OpenAI).
        Detector {
            tag: "ANTHROPIC_KEY",
            re: &RE_ANTHROPIC,
        },
        Detector {
            tag: "OPENAI_KEY",
            re: &RE_OPENAI,
        },
        Detector {
            tag: "HUGGINGFACE_KEY",
            re: &RE_HUGGINGFACE,
        },
        Detector {
            tag: "GITHUB_PAT",
            re: &RE_GITHUB,
        },
        Detector {
            tag: "AWS_ACCESS_KEY",
            re: &RE_AWS,
        },
        Detector {
            tag: "SLACK_TOKEN",
            re: &RE_SLACK,
        },
        Detector {
            tag: "HEX_TOKEN",
            re: &RE_HEX,
        },
        // Then structured PII.
        Detector {
            tag: "EMAIL",
            re: &RE_EMAIL,
        },
        Detector {
            tag: "IBAN",
            re: &RE_IBAN,
        },
        Detector {
            tag: "CREDIT_CARD",
            re: &RE_CREDIT_CARD,
        },
        Detector {
            tag: "PHONE",
            re: &RE_PHONE,
        },
    ]
});

// Which category the Python RedactionPolicy controls each tag by.
// "api_keys" is a compound toggle covering the seven API-key tags.
fn policy_key_for_tag(tag: &str) -> &'static str {
    match tag {
        "ANTHROPIC_KEY" | "OPENAI_KEY" | "HUGGINGFACE_KEY" | "GITHUB_PAT" | "AWS_ACCESS_KEY"
        | "SLACK_TOKEN" | "HEX_TOKEN" => "api_keys",
        "EMAIL" => "emails",
        "IBAN" => "iban",
        "CREDIT_CARD" => "credit_cards",
        "PHONE" => "phones",
        _ => "",
    }
}

// ─── Core redaction ──────────────────────────────────────────────────

fn placeholder(tag: &str) -> String {
    format!("[REDACTED:{}]", tag)
}

/// Redact `text` under the given per-category policy.
///
/// Returns the redacted text plus a per-tag count map. The regex
/// engine is UTF-8-safe and replacements are char-boundary-safe
/// (Regex::replace_all works on &str directly).
fn redact_inner(text: &str, policy: &HashMap<String, bool>) -> (String, HashMap<String, u32>) {
    let mut out = text.to_string();
    let mut counts: HashMap<String, u32> = HashMap::new();
    for d in DETECTORS.iter() {
        let key = policy_key_for_tag(d.tag);
        if !key.is_empty() && !policy.get(key).copied().unwrap_or(true) {
            continue;
        }
        let mut n: u32 = 0;
        let replaced = d.re.replace_all(&out, |_caps: &regex::Captures| {
            n += 1;
            placeholder(d.tag)
        });
        if n > 0 {
            counts.insert(d.tag.to_string(), n);
            out = replaced.into_owned();
        }
    }
    (out, counts)
}

// ─── PyO3 surface ────────────────────────────────────────────────────

/// Python: `redact(text, policy) -> (str, dict[str, int])`.
///
/// `policy` is a dict with string keys matching the five toggle
/// categories used by Python's `RedactionPolicy`:
/// ``emails``, ``phones``, ``iban``, ``credit_cards``, ``api_keys``.
/// Missing keys default to True (detector active).
#[pyfunction]
#[pyo3(signature = (text, policy=None))]
fn redact(
    py: Python<'_>,
    text: &str,
    policy: Option<&Bound<'_, PyDict>>,
) -> PyResult<(String, Py<PyDict>)> {
    let mut pol_map: HashMap<String, bool> = HashMap::new();
    if let Some(p) = policy {
        for (k, v) in p.iter() {
            let key: String = k.extract()?;
            let val: bool = v.extract()?;
            pol_map.insert(key, val);
        }
    }
    let (red, counts) = redact_inner(text, &pol_map);
    let out = PyDict::new(py);
    for (k, v) in counts {
        out.set_item(k, v)?;
    }
    Ok((red, out.unbind()))
}

#[pymodule]
fn remanentia_pii_redactor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(redact, m)?)?;
    Ok(())
}

// ─── Unit tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn full_policy() -> HashMap<String, bool> {
        let mut p = HashMap::new();
        for k in ["emails", "phones", "iban", "credit_cards", "api_keys"] {
            p.insert(k.to_string(), true);
        }
        p
    }

    #[test]
    fn email_redacted() {
        let (r, c) = redact_inner("contact alice@example.com tomorrow", &full_policy());
        assert!(!r.contains("alice@example.com"));
        assert!(r.contains("[REDACTED:EMAIL]"));
        assert_eq!(c.get("EMAIL"), Some(&1));
    }

    #[test]
    fn phone_international() {
        let (r, _) = redact_inner("+421 902 123 456", &full_policy());
        assert!(r.contains("[REDACTED:PHONE]"));
    }

    #[test]
    fn iso_date_not_mistaken_for_phone() {
        let (_, c) = redact_inner("2026-03-15", &full_policy());
        assert_eq!(c.get("PHONE"), None);
    }

    #[test]
    fn anthropic_beats_openai_prefix() {
        let key = format!("sk-ant-api03-{}", "B".repeat(95));
        let (_, c) = redact_inner(&key, &full_policy());
        assert_eq!(c.get("ANTHROPIC_KEY"), Some(&1));
        assert_eq!(c.get("OPENAI_KEY"), None);
    }

    #[test]
    fn iban_redacted() {
        let (_, c) = redact_inner("wire to CH9300762011623852957 soon", &full_policy());
        assert_eq!(c.get("IBAN"), Some(&1));
    }

    #[test]
    fn credit_card_redacted() {
        let (_, c) = redact_inner("card 4111 1111 1111 1111", &full_policy());
        assert_eq!(c.get("CREDIT_CARD"), Some(&1));
    }

    #[test]
    fn policy_disables_emails() {
        let mut p = full_policy();
        p.insert("emails".to_string(), false);
        let (r, c) = redact_inner("alice@example.com", &p);
        assert!(r.contains("alice@example.com"));
        assert_eq!(c.get("EMAIL"), None);
    }

    #[test]
    fn utf8_safe_on_yen_and_emoji() {
        // ¥ (2 bytes) + 🎯 (4 bytes). Replacements must not panic.
        let text = "Tokyo ¥500 🎯 alice@example.com";
        let (r, c) = redact_inner(text, &full_policy());
        assert!(r.contains("[REDACTED:EMAIL]"));
        assert!(r.contains("¥500"));
        assert!(r.contains("🎯"));
        assert_eq!(c.get("EMAIL"), Some(&1));
    }

    #[test]
    fn empty_text() {
        let (r, c) = redact_inner("", &full_policy());
        assert_eq!(r, "");
        assert!(c.is_empty());
    }

    #[test]
    fn multiple_emails() {
        let (_, c) = redact_inner("a@b.co c@d.io e@f.org", &full_policy());
        assert_eq!(c.get("EMAIL"), Some(&3));
    }

    #[test]
    fn aws_access_key() {
        let (_, c) = redact_inner("AWS: AKIAIOSFODNN7EXAMPLE used", &full_policy());
        assert_eq!(c.get("AWS_ACCESS_KEY"), Some(&1));
    }

    #[test]
    fn hex_token_short_skipped() {
        let (_, c) = redact_inner("commit abc1234", &full_policy());
        assert_eq!(c.get("HEX_TOKEN"), None);
    }
}
