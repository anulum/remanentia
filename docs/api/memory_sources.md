# memory_sources

Configuration loader for `memory_index.MemoryIndex` source roots.

## Purpose

`memory_sources.py` keeps the public package independent of any private
workspace layout. It ships neutral, repository-local defaults and lets operators
add deployment-specific archives through JSON configuration.

## Defaults

| Source | Path | Extensions |
|--------|------|------------|
| `traces` | `reasoning_traces/` | `.md` |
| `paper` | `paper/` | `.md`, `.tex`, `.bib`, `.txt` |
| `semantic` | `memory/semantic/` | `.md`, `.txt`, `.json`, `.jsonl`, `.yaml`, `.yml` |
| `compiled` | `memory/compiled/` | `.md`, `.jsonl` |
| `code_remanentia` | `.` | `.py` |

## Configuration

Use `REMANENTIA_MEMORY_SOURCES_CONFIG` for a JSON file:

```json
{
  "sources": {
    "decision_archive": {
      "path": "../archives/decisions",
      "extensions": [".md", ".jsonl"]
    }
  }
}
```

Use `REMANENTIA_MEMORY_SOURCES_JSON` for the same JSON inline. Set
`"extends_defaults": false` to run with only configured roots.

Labels must be stable identifiers (`A-Z`, `a-z`, digits, `_`, `-`) because they
flow into document `source` fields and downstream filters.

## API Reference

::: memory_sources.SourceConfig

::: memory_sources.build_source_config

::: memory_sources.load_source_config
