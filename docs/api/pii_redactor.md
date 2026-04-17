# pii_redactor

Regex-driven PII redaction pass run before any user text lands in a
Remanentia memory store. Rust-accelerated hot path; pure-Python
fallback when the extension is absent.

## Why this module exists

Remanentia persists arbitrary user text. Without a redaction pass,
anything the user types — email addresses, phone numbers, IBANs,
credit-card numbers, leaked API keys — sticks on disk forever, in
plain text, indexed for retrieval. A later retrieval matches on those
strings and leaks them back into the LLM prompt.

The module runs a conservative regex sweep before every memory write.
Eleven detectors run in a fixed order so specific prefixes (Anthropic
`sk-ant-api…`) beat generic ones (OpenAI `sk-…`).

## Public surface

```python
from pii_redactor import RedactionPolicy, RedactionResult, redact, redact_texts
```

### `redact(text: str, policy: RedactionPolicy | None = None) -> RedactionResult`

Return a new string with every matched PII span replaced by a
`[REDACTED:TAG]` placeholder. `RedactionResult.counts` reports how many
matches each tag produced, so callers can rate-limit or alert when a
single memory write contains suspiciously many secrets.

```python
from pii_redactor import redact

out = redact("Mail me at alice@example.com or +421 902 123 456")
print(out.text)
# "Mail me at [REDACTED:EMAIL] or [REDACTED:PHONE]"
print(out.counts)
# {"EMAIL": 1, "PHONE": 1}
```

### `RedactionPolicy(**toggles, extra=())`

Per-call-site policy:

| Field | Default | Effect when False |
| --- | --- | --- |
| `emails` | True | email regex skipped |
| `phones` | True | phone regex skipped |
| `iban` | True | IBAN regex skipped |
| `credit_cards` | True | credit-card regex skipped |
| `api_keys` | True | all seven API-key regexes skipped |
| `extra` | `()` | tuple of `(tag, re.Pattern)` pairs to apply after built-ins |

The extra-pattern list is always handled in Python; the Rust fast path
covers only the fixed detector set.

### `redact_texts(texts, policy) -> list[RedactionResult]`

Batch convenience wrapper for ingest workers. Iterable-in,
list-of-results-out.

## Detectors and ordering

Order matters for correctness. The fixed sequence is:

| # | Tag | Shape |
| --- | --- | --- |
| 1 | `ANTHROPIC_KEY` | `sk-ant-api\d\d-…{80,}` |
| 2 | `OPENAI_KEY` | `sk-(?:proj-\|svcacct-\|user-)?…{20,}` |
| 3 | `HUGGINGFACE_KEY` | `hf_…{30,}` |
| 4 | `GITHUB_PAT` | `gh[pousr]_…{30,}` |
| 5 | `AWS_ACCESS_KEY` | `AKIA[0-9A-Z]{16}` |
| 6 | `SLACK_TOKEN` | `xox[abprs]-…{10,}` |
| 7 | `HEX_TOKEN` | `[a-f0-9]{32,}` (fallback) |
| 8 | `EMAIL` | RFC-ish address |
| 9 | `IBAN` | 2 letters + 2 digits + 10-30 alnum |
| 10 | `CREDIT_CARD` | 13-19 digit groups |
| 11 | `PHONE` | 3-4 digit groups with separators |

Anthropic beats OpenAI because both begin `sk-`. HEX_TOKEN is last among
the API-key tags so it only catches home-rolled session tokens that
none of the named detectors recognise.

## Rust fast path

`pii_redactor` imports the `remanentia_pii_redactor` PyO3 extension at
module load. When present, `redact` delegates to the Rust
`redact(text, policy)` binding; the Python fallback runs only on
`ImportError`. Rust path is **~2.9× faster** on typical memory-ingest
text. Both paths return identical output; parity tests in
`tests/test_pii_redactor.py::TestRustPythonParity` compare them on
every CI run so drift is caught immediately.

## Invariants

- **No reversible de-redaction.** The original bytes are never stored
  anywhere; you cannot recover them.
- **UTF-8 safe.** Multi-byte sequences (¥, €, CJK, emoji) are never
  split. The Rust path works on `&str`; the Python path uses the
  `re` module which is byte-safe on `str`.
- **No over-redaction of ISO dates.** `2026-03-15` and similar do not
  match the phone regex: phone digit groups are 3-4, ISO year groups
  are 4-2-2.
- **No Luhn check.** Deliberate: increases recall at the cost of some
  false positives. The module prefers over-redaction.

## Out of scope

- Named-entity detection (GLiNER pipeline) — separate opt-in layer.
- Content filtering (hate speech, illegal content) — not a PII problem.
- Hashing / tokenisation for analytics — we replace with a token, we
  do not preserve any reversible form.

## See also

- [`aggregate_precompute`](aggregate_precompute.md) — sister module,
  also Rust-accelerated with a Python fallback.
- `remanentia_pii_redactor` crate at
  `../../../../workspace-internal/rust_pii_redactor/` — Rust
  implementation.
- `PRIVACY.md` (repo root) — data-handling policy this module
  enforces at ingest.
