# api_server

Lightweight stdlib HTTP API server for cross-service integration (SYNAPSE bridge, SPO, etc). No FastAPI dependency.

Endpoints: GET `/health`, `/status`; POST `/recall`, `/consolidate`, `/remember`.

Private endpoint responses are written as JSONL request metadata to
`.coordination/runtime/api_server_audit.jsonl` by default. Set
`REMANENTIA_API_AUDIT_LOG` to a path to relocate the log, or to `off` to
disable it. Audit records include method, path, client address, status,
outcome, and auth-enabled state; request bodies and authorisation headers are
never recorded.

::: api_server.RemanentiaHandler
    options:
      show_source: true
      members_order: source

::: api_server._json_default
