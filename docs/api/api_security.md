# api_security

Authentication, rate limiting, and request-size guards for every
Remanentia network surface (HTTP API, MCP server, future WebSocket).

## Purpose

Remanentia's network servers persist arbitrary user text to disk.
A naïve deploy without auth accepts writes from anyone who can reach
the port, leaks private memories through search, and lets a single
abusive client exhaust memory via unbounded request bodies.

`api_security` is the shared defence layer behind every listener.
Three orthogonal primitives — bearer auth, token-bucket rate limiting,
body-size enforcement — so a caller composes exactly the guarantees
it needs without pulling in a framework.

## Public surface

```python
from api_security import BearerAuth, TokenBucketLimiter, enforce_body_size
```

### `BearerAuth(token: str | None = None)`

Constant-time bearer-token check. Pass an explicit token, or use
`BearerAuth.from_env()` to read `REMANENTIA_API_TOKEN`, or
`BearerAuth.from_file(path)` to load a local token file.
**Configuration posture is intentionally loud-on-default-missing**:

- Token set → every request must carry `Authorization: Bearer <token>`;
  mismatches return `401` and do not leak whether the token was too
  long, too short, or plain wrong (the check uses `hmac.compare_digest`).
- Token unset → stderr warns once at construction time, then every
  request passes. This keeps `remanentia serve` frictionless during
  local development while making "I forgot to set the token" visible
  in operator logs.

```python
auth = BearerAuth.from_env()              # reads REMANENTIA_API_TOKEN
if not auth.check_header(request.headers.get("Authorization")):
    return Response(401)
```

### `TokenBucketLimiter(rate_per_minute: float = 60.0, burst: int = 10)`

Per-key token bucket. The key is caller-supplied (typically
`request.client.host`), so `limiter.allow(key)` returns `True` when
a token is available and `False` when the caller exceeded the cap.

The bucket refills continuously from `rate_per_minute`, so burst traffic
is absorbed up to `burst` before throttling kicks in. The state is
per-process; behind a load balancer, each replica maintains its own bucket
(intentional — distributed rate limiting needs a shared backend and is out of
scope for this module).

```python
limiter = TokenBucketLimiter(rate_per_minute=60, burst=10)
if not limiter.allow(request.client.host):
    return Response(429, headers={"Retry-After": "1"})
```

### `enforce_body_size(declared_length: int, limit_bytes: int)`

Reject requests whose `Content-Length` exceeds the cap. Called before
reading the body so an attacker cannot force the server to buffer a
gigabyte to discover the request was too big.

```python
enforce_body_size(int(request.headers["content-length"]), 1_048_576)
# raises ValueError if too large — wrap in the framework's 413 handler
```

## Invariants

- **Constant-time token compare**: `hmac.compare_digest` throughout,
  so timing channels cannot reveal token prefixes.
- **Stateless primitives**: `BearerAuth` and `enforce_body_size` carry
  no per-request state. `TokenBucketLimiter` mutates its internal map
  only under a `threading.Lock` — safe for WSGI / ASGI workers.
- **No third-party deps**: stdlib-only. Adding `api_security` to a
  listener does not change the install footprint.
- **Never log tokens**: the warning on missing tokens is fixed text;
  the `check_header` method returns bool only.

## When to use what

| Threat | Primitive |
| --- | --- |
| Unauthorised writes from the internet | `BearerAuth` |
| Single client hammering `/search` | `TokenBucketLimiter` |
| Attacker streaming a 10 GB body to OOM the host | `enforce_body_size` |
| Layered defence for a public endpoint | all three, in that order |

## See also

- [`api`](api.md) — FastAPI surface that applies bearer auth to private
  endpoints while keeping health and public vector search open.
- [`api_server`](api_server.md) — HTTP surface that wires these in.
- [`mcp_server`](mcp_server.md) — stdio bridge; uses `BearerAuth` when
  transported over a socket.
- `THREAT_MODEL.md` (repo root) — which attacks these primitives
  mitigate.
