<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# api

FastAPI surface for Remanentia.

## Authentication

Set `REMANENTIA_API_TOKEN` in production. When the token is set, every
private endpoint must send `Authorization: Bearer <token>` or the server
returns `401`.

The intentionally public endpoints are:

- `GET /health`
- `POST /vector/search/public`

Unset tokens keep local development open, but the server emits the
standard `api_security` warning so operators can see that auth is disabled.

For fail-closed production startup, use:

```bash
remanentia serve --require-auth --token-file /run/secrets/remanentia_api_token
```

`--token-file` loads `REMANENTIA_API_TOKEN` before `uvicorn` imports `api:app`.
`--require-auth` refuses to start unless a token exists in either the environment
or the configured token file.

## OpenAPI Export

Generate the published REST API schema with:

```bash
remanentia openapi --output docs/openapi/remanentia_openapi.json
```

The exporter builds from the live FastAPI app and then annotates each operation
with the same bearer-auth policy enforced by runtime middleware:

- `GET /health` is public.
- `POST /vector/search/public` is public.
- All other operations require the `BearerAuth` HTTP bearer scheme when
  `REMANENTIA_API_TOKEN` is configured.

The committed schema is deterministic JSON for client generation, contract
reviews, and release audits.

## Request Limits

The FastAPI surface applies the shared `api_security` request gates before
endpoint handlers run:

| Variable | Default | Purpose |
|---|---:|---|
| `REMANENTIA_API_BODY_LIMIT_BYTES` | 1048576 | Maximum declared request body size |
| `REMANENTIA_API_RATE_PER_MINUTE` | 60 | Per-client steady-state request rate |
| `REMANENTIA_API_RATE_BURST` | 10 | Per-client token-bucket burst |
| `REMANENTIA_CORS_ORIGINS` | `*` | Comma-separated allowed browser origins |

`GET /health` is excluded from rate limiting so health checks cannot consume
application request quota. Public vector search is unauthenticated, but still
uses the body-size and rate-limit gates.

Rate-limited responses return `429` with a `Retry-After` header in seconds,
computed from `REMANENTIA_API_RATE_PER_MINUTE`.

Private endpoint responses are written as JSONL request metadata to
`.coordination/runtime/api_audit.jsonl` by default. Set
`REMANENTIA_API_AUDIT_LOG` to a path to relocate the log, or to `off` to
disable it. Audit records include method, path, client address, status,
outcome, and auth-enabled state; request bodies and authorisation headers are
never recorded. Set `REMANENTIA_API_AUDIT_MAX_BYTES` to rotate before an
append would exceed the configured byte cap, and
`REMANENTIA_API_AUDIT_BACKUPS` to keep numbered backups such as
`api_audit.jsonl.1`.

Set `REMANENTIA_CORS_ORIGINS` for browser-exposed deployments, for example
`https://remanentia.com,https://www.remanentia.com`. Leaving it unset preserves
the local-development wildcard.

## Public Vector Search

`POST /vector/search/public` exposes the public-safe vector result view.
It never returns raw vector search objects directly.

Request:

```json
{
  "query": "hybrid retrieval decision",
  "top_k": 5,
  "source": ""
}
```

The server controls public allowlists and redaction policy through
environment/configuration:

| Variable | Purpose |
|---|---|
| `REMANENTIA_PUBLIC_VECTOR_SOURCES` | Comma-separated allowed source labels |
| `REMANENTIA_PUBLIC_VECTOR_PATH_PREFIXES` | Comma-separated allowed metadata path prefixes |
| `REMANENTIA_PUBLIC_VECTOR_REDACTION_FILE` | Local newline-delimited redaction term file |
| `REMANENTIA_PUBLIC_VECTOR_MAX_TEXT_CHARS` | Maximum public text length |
| `REMANENTIA_VECTOR_INDEX_DIR` | Optional vector index directory override |

No allowlist means no public vector results are emitted.

The caller may narrow by `source`, but cannot widen the server-controlled
public corpus.

## API Reference

::: api
