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
