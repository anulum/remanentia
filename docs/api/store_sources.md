# store_sources

`store_sources.py` writes the `memory_sources.py` JSON config for an operator's
selected memory store. Use it before rebuilding `MemoryIndex` over the MS.0
backfill corpus so the retriever indexes selected `reasoning_traces`, semantic
memory, compiled facts, and the external `snn_stimuli` firehose through the
normal production build path.

```bash
remanentia store-sources --base /path/to/store \
  --stimuli-dir /path/to/snn_stimuli --write --json

REMANENTIA_MEMORY_SOURCES_CONFIG=/path/to/store/snn_state/memory_sources.json \
  python -m memory_index --build
```

::: store_sources
