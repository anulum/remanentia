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
remanentia serve [--port PORT] [--host HOST]
```

Starts the FastAPI REST server.

::: cli.main
