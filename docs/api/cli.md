# CLI Reference

Command-line interface for Remanentia.

## Commands

### search

```bash
remanentia search "query" [--top-k N] [--project NAME] [--after DATE] [--before DATE]
```

### recall

```bash
remanentia recall "query" [--top-k N] [--format context|json]
```

### status

```bash
remanentia status
```

### consolidate

```bash
remanentia consolidate [--force]
```

### init

```bash
remanentia init
```

Creates the directory structure for a new Remanentia installation.

### serve

```bash
remanentia serve [--port PORT] [--host HOST] [--token-file PATH] [--require-auth]
```

Starts the FastAPI REST server. Use `--require-auth` in deployments that must
refuse open private endpoints, and use `--token-file` to load
`REMANENTIA_API_TOKEN` before `api:app` imports.

### openapi

```bash
remanentia openapi [--output PATH]
```

Writes the deterministic FastAPI OpenAPI schema, including bearer-auth
annotations for private endpoints.

::: cli.main
