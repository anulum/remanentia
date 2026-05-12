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

## Request Limits

The FastAPI surface applies the shared `api_security` request gates before
endpoint handlers run:

| Variable | Default | Purpose |
|---|---:|---|
| `REMANENTIA_API_BODY_LIMIT_BYTES` | 1048576 | Maximum declared request body size |
| `REMANENTIA_API_RATE_PER_MINUTE` | 60 | Per-client steady-state request rate |
| `REMANENTIA_API_RATE_BURST` | 10 | Per-client token-bucket burst |

`GET /health` is excluded from rate limiting so health checks cannot consume
application request quota. Public vector search is unauthenticated, but still
uses the body-size and rate-limit gates.

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
