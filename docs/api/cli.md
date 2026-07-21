# CLI Reference

The `remanentia` command exposes local recall, store maintenance, service, and
schema operations. Set an explicit writable store before running data commands:

```bash
export REMANENTIA_BASE="$PWD/.remanentia-data"
remanentia init
```

## Command map

| Command | Purpose |
|---|---|
| `recall` | Search memory and render summary, context, or JSON |
| `search` | Alias of `recall` |
| `consolidate` | Consolidate new or all traces |
| `status` | Report selected-store state |
| `store-manifest` | Inspect or write the selected-store manifest |
| `store-sources` | Inspect or write MemoryIndex source configuration |
| `openapi` | Export the FastAPI OpenAPI schema |
| `claim-schema` | Export or check the claim-axis schema |
| `graph` | Show top entity relationships |
| `entities` | List known entities |
| `daemon` | Start, stop, or inspect the legacy experimental daemon |
| `init` | Create the selected-store directory layout |
| `observe` | Watch source files or run one observation cycle |
| `reflect` | Run optional LLM-powered reflection |
| `setup-llm` | Detect hardware and configure a local model |
| `serve` | Start the FastAPI service |
| `serve-llm` | Start the configured local LLM server |
| `notes` | List knowledge notes |

`remanentia --help` and `remanentia <command> --help` are the canonical option
reference for the installed version.

## Recall and search

```bash
remanentia search "deployment decision" --top 5
remanentia recall "deployment decision" --top 5 --format context
remanentia recall "deployment decision" --project example --after 2026-01-01 --format json
```

Both commands accept `--top`, not `--top-k`. Optional model synthesis is
controlled by `--llm` and `--llm-backend`; it is not used by default.

## Consolidation and inspection

```bash
remanentia consolidate
remanentia consolidate --force
remanentia status
remanentia graph --top 15
remanentia entities
remanentia notes --top 20
```

## Store configuration

```bash
remanentia store-manifest --json
remanentia store-manifest --write --json
remanentia store-sources --write --json
```

## API and schemas

```bash
remanentia serve --host 127.0.0.1 --port 8001
remanentia serve --host 127.0.0.1 --port 8001 \
  --require-auth --token-file /run/secrets/remanentia_api_token
remanentia openapi --output docs/openapi/remanentia_openapi.json
remanentia claim-schema --output docs/schema/remanentia_claim_axes.schema.json
```

Use `--require-auth` when the service must refuse to start without a token.
Binding outside loopback also requires a trusted transport-security boundary.

## Python entry point

::: cli.main
