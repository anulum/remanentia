# mcp_server

Thread-safe MCP server (stdio JSON-RPC) for memory retrieval and persistence.

## Tool-call audit log

Every `tools/call` request is audited by default to
`.coordination/runtime/mcp_tool_audit.jsonl`. Set
`REMANENTIA_MCP_AUDIT_LOG=/path/to/audit.jsonl` to relocate it, or
`REMANENTIA_MCP_AUDIT_LOG=off` to disable it for a local-only session.

Audit records are metadata-only JSONL. They include timestamp, server,
method, tool name, request id, sorted argument names, outcome, duration,
and exception type when a handler fails. They do not include tool argument
values, memory content, recall queries, authorisation headers, or request
bodies.

Set `REMANENTIA_MCP_AUDIT_MAX_BYTES` to rotate before an append would exceed
the configured byte cap, and `REMANENTIA_MCP_AUDIT_BACKUPS` to keep numbered
backups such as `mcp_tool_audit.jsonl.1`.

Handler exceptions return a generic JSON-RPC error (`-32000`) and are
recorded with `outcome="error"` plus the exception class name. Unknown
tools remain successful JSON-RPC responses for compatibility, but are
audited with `outcome="unknown_tool"`.

## Tool-call rate limiting

`tools/call` requests pass through a per-process token bucket before any
tool handler runs. The default is intentionally generous for local stdio
sessions: 600 calls per minute with a burst of 120. Operators that expose MCP
through a socket bridge should set tighter values for their deployment.

| Variable | Default | Purpose |
| --- | --- | --- |
| `REMANENTIA_MCP_RATE_LIMIT` | enabled | Set to `off`, `0`, `false`, or `no` to disable the limiter. |
| `REMANENTIA_MCP_RATE` | `600.0` | Sustained calls per minute for the MCP client bucket. |
| `REMANENTIA_MCP_BURST` | `120` | Maximum burst calls before throttling. |
| `REMANENTIA_MCP_CLIENT_ID` | `stdio` | Token-bucket key for this MCP client/session. |

When the bucket is empty, the server returns a JSON-RPC error with code
`-32029` and `data.retry_after_seconds`. The blocked call is audited with
`outcome="rate_limited"`; argument values are still never written to the audit
log.

::: mcp_server.handle_recall

::: mcp_server.handle_remember

::: mcp_server.handle_status

::: mcp_server.handle_graph
