# Privacy Statement

Remanentia persists conversational content that may contain personal
data. This document states what we store, how it is protected, and
how a person whose data is stored can exercise their rights.

## What gets stored

| Data class | Where | Example | Retention |
|------------|-------|---------|-----------|
| Conversation turns | `memory/semantic/*.md` | "On March 15, Alice said she moved to Zurich." | Until explicit deletion |
| Entity graph | `memory/graph/entities.jsonl`, `relations.jsonl` | `{"id": "alice", "label": "person"}` | Same as above |
| Reasoning traces | `reasoning_traces/*.md` | Intermediate LLM reasoning over the above | Same as above |
| Knowledge store | `memory/knowledge/notes.jsonl` | Free-form user-supplied notes | Same as above |
| SNN consolidation state | `snn_state/*.npz` | Derived; contains hashed tokens, not text | Same as above |

Remanentia does **not** store LLM API request/response pairs, model
weights containing user data, or any telemetry off the host.

## Controls on write

Every write path runs the `pii_redactor` before the bytes hit disk:

- Emails → `[REDACTED:EMAIL]`
- Phone numbers → `[REDACTED:PHONE]`
- IBAN → `[REDACTED:IBAN]`
- Credit-card-shaped digit groups → `[REDACTED:CREDIT_CARD]`
- API-key-shaped tokens (OpenAI, Anthropic, HuggingFace, GitHub,
  AWS, Slack, generic 32+ hex) → `[REDACTED:<CATEGORY>]`

Policy is configurable per call site; operators who want the raw
text (e.g. audit trails) pass `redact_pii=False` explicitly.

## Controls at rest

- **File locking + atomic writes:** every memory-store writer uses
  `file_utils.atomic_write_*` with a `fcntl.flock` around the
  transaction. Concurrent writers cannot tear files or lose updates.
- **No pickle:** the legacy pickle fallback was removed 2026-04-17
  (ADR-0002). Stored state is `.npz` (numeric) or `.json.gz` (text)
  only.

## Controls on the network

- **Bearer token auth** via `REMANENTIA_API_TOKEN` on every HTTP
  request except `/health`. The operator controls the token lifetime
  — there is no SaaS component that sees it.
- **Rate limiting** per IP (token bucket, 60/min burst 10 default).
- **Body size cap** (1 MiB default) before the server reads a byte.

## Right to access, correct, delete

There is no multi-tenant namespace yet (tracked as P4-22). On a
single-tenant deployment the person exercising their rights is
necessarily the operator:

- **Access:** `python -c "from knowledge_store import KnowledgeStore;
  ks = KnowledgeStore(); ks.load(); print(ks.notes)"`
- **Correct:** edit the corresponding `memory/semantic/*.md` file
  and run `python tools/rebuild_index.py` (or equivalently, delete
  and re-ingest).
- **Delete:** remove the file on disk; the knowledge store and
  memory index drop the record on next load. There is no trash; a
  deleted note is gone.

Multi-tenant isolation is on the roadmap. Until it lands, operators
running Remanentia on behalf of multiple end-users need to partition
the file tree themselves and apply OS-level permissions.

## Data export

- Every stored record is plaintext / JSON on disk. A simple `tar` of
  `memory/` + `reasoning_traces/` gives a complete export.
- There is no proprietary format — any editor or scripting language
  can consume the exported bundle.

## Third parties

When configured to use hosted LLMs (OpenAI, Anthropic), the contents
of the retrieved memories that form part of a prompt are transmitted
to that provider per their terms. Operators should disclose this to
their users. `LocalLLMBackend` (Ollama) keeps everything on-host.

## Disclosure

Security incidents that involve personal data are handled per
`SECURITY.md` § "Reporting a Vulnerability". We will notify
affected operators within 72 hours of confirmation.

Last change: 2026-04-17.
