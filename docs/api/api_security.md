# api_security

Authentication, rate limiting, request-size guards, and metadata-only
audit logging for every Remanentia network surface (HTTP API, MCP
server, future WebSocket).

## Purpose

Remanentia's network servers persist arbitrary user text to disk.
A naïve deploy without auth accepts writes from anyone who can reach
the port, leaks private memories through search, and lets a single
abusive client exhaust memory via unbounded request bodies.

`api_security` is the shared defence layer behind every listener.
Five orthogonal primitives — bearer auth, token-bucket rate limiting,
body-size enforcement, append-only request audit logging, and append-only
MCP tool-call audit logging — so a caller composes exactly the guarantees it
needs without pulling in a framework.

## Public surface

```python
from api_security import (
    BearerAuth,
    RequestAuditLogger,
    ToolAuditLogger,
    TokenBucketLimiter,
    enforce_body_size,
)
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
    return Response(429, headers={"Retry-After": limiter.retry_after_seconds()})
```

`retry_after_seconds()` returns a conservative whole-second wait for one
token to refill and is used by both HTTP API surfaces for `429` responses.

### `enforce_body_size(declared_length: int, limit_bytes: int)`

Reject requests whose `Content-Length` exceeds the cap. Called before
reading the body so an attacker cannot force the server to buffer a
gigabyte to discover the request was too big.

```python
enforce_body_size(int(request.headers["content-length"]), 1_048_576)
# raises ValueError if too large — wrap in the framework's 413 handler
```

### `RequestAuditLogger(path: str | os.PathLike | None)`

Append-only JSONL audit logger for request metadata. It records routing and
response fields only: timestamp, server, method, path, client, status,
outcome, and whether auth was enabled. It does not accept request bodies or
authorisation headers as inputs.

```python
audit = RequestAuditLogger.from_env(".coordination/runtime/api_audit.jsonl")
audit.record(
    server="fastapi",
    method="GET",
    path="/status",
    client=request.client.host,
    status=200,
    outcome="ok",
    auth_enabled=auth.enabled,
)
```

Set `REMANENTIA_API_AUDIT_LOG=off` to disable audit logging, or set it to a
path to relocate the JSONL file.

### `ToolAuditLogger(path: str | os.PathLike | None)`

Append-only JSONL audit logger for MCP tool-call metadata. It records the
server, method, tool name, request id, sorted argument names, outcome,
duration, and exception type when present. It deliberately stores argument
names only, never argument values, so `remanentia_remember` content and
`remanentia_recall` queries do not leak into audit files.

```python
audit = ToolAuditLogger.from_env(".coordination/runtime/mcp_tool_audit.jsonl")
audit.record(
    server="mcp",
    method="tools/call",
    tool="remanentia_recall",
    request_id="42",
    argument_keys=["query", "top_k"],
    outcome="ok",
    duration_ms=3.4,
)
```

Set `REMANENTIA_MCP_AUDIT_LOG=off` to disable MCP audit logging, or set it to
a path to relocate the JSONL file.

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
- **Never log tool arguments**: MCP audit records include argument names
  only, not argument values.

## When to use what

| Threat | Primitive |
| --- | --- |
| Unauthorised writes from the internet | `BearerAuth` |
| Single client hammering `/search` | `TokenBucketLimiter` |
| Attacker streaming a 10 GB body to OOM the host | `enforce_body_size` |
| Operational traceability for private API endpoints | `RequestAuditLogger` |
| Operational traceability for MCP tool use | `ToolAuditLogger` |
| Layered defence for a public endpoint | body-size and rate-limit gates, plus audit where appropriate |

## See also

- [`api`](api.md) — FastAPI surface that applies bearer auth to private
  endpoints while keeping health and public vector search open.
- [`api_server`](api_server.md) — HTTP surface that wires these in.
- [`mcp_server`](mcp_server.md) — stdio bridge; uses `BearerAuth` when
  transported over a socket.
- `THREAT_MODEL.md` (repo root) — which attacks these primitives
  mitigate.
