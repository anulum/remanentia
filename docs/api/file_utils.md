# file_utils

Atomic writes and advisory locks for Remanentia's persistent knowledge
stores. Stdlib-only, no third-party dependencies.

## Why this module exists

Before 2026-04-17 Remanentia had **no writer-concurrency story at all**.
Two `remanentia ingest` processes writing to the same knowledge store
silently clobbered each other: the second write overwrote everything
the first had just added, and the user noticed only when a memory
"vanished" weeks later. A power cut during a write left the store half
full, half old, with no magic number to tell the loader which half it
had.

Two primitives fix both failure modes:

- **Atomic writes** — the target file either has the old bytes or the
  new bytes, never half of each.
- **Advisory locks** — concurrent writers block on each other instead
  of clobbering.

A distributed multi-host setup still needs a shared lock backend
(Redis / etcd / DynamoDB) on top of this module; `file_utils` is the
single-host baseline.

## Public surface

```python
from file_utils import (
    atomic_write_text,
    atomic_write_bytes,
    atomic_write_json,
    FileLock,
)
```

### `atomic_write_text(path, content, *, encoding="utf-8")`

Write `content` to `path` via an `os.replace` rename of a same-
directory tmpfile. Key properties:

- Same filesystem, so `os.replace` is a real atomic rename.
- Tmpfile name is `{target}.tmp.{pid}` → crashes leave an obvious
  artefact the operator can clean up (or ignore; the next successful
  write overwrites it).
- Parent directory is `os.fsync`'d after the rename so the rename
  commit survives an unclean shutdown.

### `atomic_write_bytes(path, data)`

Binary variant. Same crash-safety properties.

### `atomic_write_json(path, obj, *, indent=2, sort_keys=False)`

Convenience wrapper: `json.dumps` then `atomic_write_text`. Used for
`snn_state/content_hashes.json`, `knowledge_store.json`, and the
benchmarks hypothesis file.

### `FileLock(path: Path | str, *, timeout: float | None = None)`

POSIX advisory `fcntl.flock` context manager. Blocks until it acquires
the lock, or raises `TimeoutError` when `timeout` is set and expires.

```python
with FileLock("snn_state/.lock", timeout=30):
    # exclusive access to snn_state/ — other writers block
    rebuild_index()
```

On Windows (no `fcntl`), the lock degrades to a no-op with a one-line
stderr warning on first use. This is a deliberate choice: Remanentia's
Windows story is "it works for single-user development"; concurrent
writers on Windows need `msvcrt.locking`, which we can add when a real
user asks for it.

## Invariants

- **Atomic writes never truncate on failure.** If the disk is full, the
  tmpfile write fails and `path` is untouched.
- **FileLock is reentrant.** The same process acquiring the lock twice
  does not deadlock (wrapped in thread-local reference count).
- **No silent data loss.** Every path either raises or succeeds; there
  is no "returned False on a race" code path.
- **Compatible with symlinks.** `os.replace` follows symlinks; the
  target of the link is replaced, not the link itself.

## When to use what

| Scenario | Primitive |
| --- | --- |
| Writing a config or cache that must never be half-written | `atomic_write_*` |
| Two processes might write the same file | `FileLock` + `atomic_write_*` |
| Appending to a log | **neither** — use `open("...", "a")` with `os.O_APPEND` |
| Writing a temp file that will be deleted anyway | **neither** — use `tempfile.NamedTemporaryFile` |

## See also

- `knowledge_store.py` — uses `FileLock` + `atomic_write_json`.
- `memory_index.py` — `atomic_write_json` for the content-hash manifest.
