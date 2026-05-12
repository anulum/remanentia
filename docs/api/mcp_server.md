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

::: mcp_server.handle_recall

::: mcp_server.handle_remember

::: mcp_server.handle_status

::: mcp_server.handle_graph
