# Threat Model

Remanentia stores arbitrary user conversations and exposes retrieval
over them through an HTTP API and an MCP server. This document names
the assets, the adversaries, the trust boundaries and the controls in
place on each. It is intentionally short; controls live in
``SECURITY.md`` and the ADR log.

## Assets

| ID | Asset | Sensitivity |
|----|-------|-------------|
| A1 | User conversation text persisted in `memory/semantic/` | **Personal** |
| A2 | Entity graph in `memory/graph/entities.jsonl` and `relations.jsonl` | Personal |
| A3 | Reasoning traces in `reasoning_traces/*.md` | Personal (often richer than A1) |
| A4 | Model weights under `models/` | Licensing only |
| A5 | Bearer token for the HTTP API | Secret |
| A6 | Hosted-LLM API keys in the operator's environment | Secret |
| A7 | SNN state files in `snn_state/` | Low |

## Adversaries

| ID | Actor | Capability |
|----|-------|-----------|
| T1 | Unauthenticated network peer | Reach the HTTP API on whatever port it binds to |
| T2 | Authenticated-but-malicious client | Send large or adversarial payloads to `/recall` and `/remember` |
| T3 | Concurrent process on same host | Race on memory-store writes |
| T4 | Operator with a legacy `.pkl` file | May run `pickle.load` implicitly |
| T5 | Supply-chain attacker | Injects a malicious dependency version or a tampered release artefact |

## Trust boundaries

```
┌──────────────┐   TLS / mTLS    ┌──────────────┐    stdio   ┌─────────────┐
│  MCP client  │ ──────────────▶ │  api_server  │ ─────────▶ │ mcp_server  │
│              │                 │   (HTTP)     │            │ (subprocess)│
└──────────────┘                 └──────────────┘            └─────────────┘
                                        │                          │
                                        ▼                          ▼
                                  ┌──────────────────────────────────┐
                                  │  memory/, reasoning_traces/      │
                                  │  knowledge_store, consolidation  │
                                  └──────────────────────────────────┘
```

Trust is assumed for: local filesystem, operator shell, configured
hosted-LLM endpoints over HTTPS. Untrusted by default: any network
peer reaching the HTTP API.

## Threat → control mapping

| Threat | Control | Commit / module |
|--------|---------|-----------------|
| T1 unauthenticated POST to `/remember` | Bearer-token auth via `REMANENTIA_API_TOKEN`; exempt paths list is hard-coded to `/health` only | `77ab128` · `api_security.BearerAuth` |
| T1 flooding POSTs | Per-IP token-bucket rate limiter, 60 req/min burst 10 default | `api_security.TokenBucketLimiter` |
| T1 oversized bodies | Content-Length declared bytes checked against `REMANENTIA_API_MAX_BODY` (1 MiB default) before reading | `api_security.enforce_body_size` |
| T2 PII/secret exfil in a future retrieval leak | Regex redaction of emails, phone numbers, IBAN, credit cards, API-key-shaped tokens on every `knowledge_store.add_note` / `_write_semantic_memory` / `handle_remember` call | `0751b4e` · `pii_redactor` |
| T3 concurrent writers tearing files | POSIX `fcntl.flock` + atomic `os.replace` on every memory-store writer | `c0ee8b5` · `file_utils` |
| T4 legacy pickle loaded | All runtime `pickle.load` sites replaced by a `ValueError` pointing at the one-shot migrator | `9e31575` · see ADR-0002 |
| T5 tampered release artefact | CycloneDX SBOM + sigstore keyless signatures + SLSA build provenance attached to every GitHub Release | `bacccd6` · `SECURITY.md` § "Verifying a Released Artefact" |
| Audit-log disk growth | Optional byte-capped rotation with bounded numbered backups via `*_AUDIT_MAX_BYTES` and `*_AUDIT_BACKUPS` | `api_security.RequestAuditLogger` · `api_security.ToolAuditLogger` |
| MCP tool-call abuse or unexpected write volume | Metadata-only JSONL audit of every `tools/call`, including tool name, request id, sorted argument names, outcome, duration, and exception type without argument values; same rotation controls as HTTP audit logs | `api_security.ToolAuditLogger` · `mcp_server` |

## Residual risk

- **Authenticated malicious clients** can still abuse the API within
  their rate budget. Per-user quotas are not implemented (tracked as
  P4-22 multi-tenant).
- **LLM prompt injection.** Memory fragments are routed through the
  LLM at synthesis time; an adversary who plants instructions in a
  stored memory can influence subsequent LLM output. No defence
  beyond the PII redactor today; a grounding / guardrail layer is
  on the roadmap (P2-13 Director-AI Guarded tier).

## Out of scope

- Host-level hardening (SELinux / AppArmor profiles).
- Network-level attacks below HTTP (OS, TLS stack).
- Cryptographic analysis of the Bearer token generation — operators
  choose their own token; no key derivation is performed.

This document is updated whenever a new control ships. Last change:
2026-05-12.
