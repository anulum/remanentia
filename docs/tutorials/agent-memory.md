# Tutorial: Give an Agent Durable Memory

This tutorial connects an MCP-compatible AI client to a local Remanentia store.
It uses the model-free retrieval path and does not require an API key.

## 1. Install and initialise

```bash
python -m venv .venv
. .venv/bin/activate
pip install remanentia
export REMANENTIA_BASE="$PWD/.remanentia-data"
remanentia init
```

## 2. Add a safe test memory

Create `$REMANENTIA_BASE/reasoning_traces/onboarding.md`:

```markdown
# Onboarding decisions

- The development API binds to 127.0.0.1:8001.
- Security reports go through the private process described in SECURITY.md.
- Benchmark claims must link to committed evidence.
```

Confirm that local retrieval sees it:

```bash
remanentia search "where does the development API bind"
```

## 3. Configure the MCP client

Add this server definition to the client's MCP configuration:

```json
{
  "mcpServers": {
    "remanentia": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "mcp_server"],
      "env": {
        "REMANENTIA_BASE": "/absolute/path/to/your/store"
      }
    }
  }
}
```

Use absolute paths for both the virtual-environment interpreter and
`REMANENTIA_BASE`; GUI clients may not inherit an activated shell. Run
`python -c 'import sys; print(sys.executable)'` inside the intended environment
to discover the interpreter path. Restart the MCP client after changing its
configuration.

## 4. Verify read and write paths

Ask the client to call `remanentia_status`, then
`remanentia_recall` with:

```json
{"query": "where does the development API bind", "top_k": 3}
```

The result should include the onboarding trace and its source context. Next,
store a non-sensitive test note through `remanentia_remember`, then recall it.
Inspect the selected store on disk to confirm where the write landed.

## 5. Record retrieval quality

Use `remanentia_recall_feedback` for usefulness feedback and
`remanentia_recall_correctness` when you have a labelled correctness outcome.
These records support later calibration; they do not make an incorrect answer
safe by themselves.

## 6. Harden the integration

- Keep the store outside repositories that should not receive generated memory.
- Do not put secrets in traces merely because the server runs locally.
- Set MCP audit-log and rotation options described in the
  [MCP API documentation](../api/mcp_server.md).
- Review source roots before enabling cross-project retrieval.
- Add hosted synthesis only after reviewing its egress and retention impact.

## Troubleshooting

| Symptom | Check |
|---|---|
| Client cannot start the server | Run `python -m mcp_server --help` in the same environment |
| Recall returns no result | Run `remanentia status`; verify the trace is under a configured source root |
| Writes appear in the wrong directory | Set an absolute `REMANENTIA_BASE` in the MCP environment |
| JSON-RPC parse errors | Ensure nothing writes ordinary logs to stdout |

Continue with the [Knowledge Store Tutorial](knowledge_store_tutorial.md) or the
[Integration Guide](../guides/INTEGRATION_GUIDE.md).
