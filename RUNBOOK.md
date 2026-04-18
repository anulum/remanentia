# Operational Runbook

This runbook covers the operational paths most likely to break on
a live Remanentia deployment. Each section has the symptom, the
quick diagnosis, and the fix. When a fix lands in code, the ADR
log (`docs/adr/`) records the rationale.

## Quick health check

```bash
curl -sS http://127.0.0.1:8001/health
# Expected: {"status": "ok", "timestamp": 1.71e9}

curl -sS -H "Authorization: Bearer $REMANENTIA_API_TOKEN" \
    http://127.0.0.1:8001/status
# Expected: {"status": "ok", "entities": N, "memories": N, ...}
```

If `/health` returns anything other than 200 + `status: ok`, the
process is down. Check the unit / container logs first.

## Symptom map

### "401 authentication required"

Cause: `REMANENTIA_API_TOKEN` not set on the client OR mismatch with
the server's environment.

```bash
# On the server host:
echo "${REMANENTIA_API_TOKEN}" | sha256sum   # compare to client's

# If empty server-side, the API is running with auth DISABLED:
grep "API auth is DISABLED" /path/to/stderr.log
```

Fix: set `REMANENTIA_API_TOKEN` in the server's unit-file
`[Service] Environment=...` stanza (or use `--token-file`), restart,
confirm the banner says "auth: ENABLED (Bearer)".

### "429 rate limit exceeded"

Cause: a client exceeded 60 req/min on one source IP.

```bash
# Defaults: 60 req/min, burst 10, 1-hour bucket TTL.
# Check current settings:
grep REMANENTIA_API_RATE /etc/systemd/system/remanentia.service.d/*.conf
```

Fix: either rate-limit the client, or raise
`REMANENTIA_API_RATE`/`REMANENTIA_API_BURST` on the server. The
limiter is in-process; multi-worker deployments need a shared
backend (Redis) — not implemented today.

### "413 request body N B exceeds limit M B"

Cause: POST body too big. Default cap is 1 MiB.

Fix: raise `--body-limit` (or `REMANENTIA_API_MAX_BODY`) to match
expected payload size. Investigate the client first — a 1 MiB
`/remember` payload is usually a bug, not a legitimate memory.

### Bench hangs with no progress print

Cause: most likely a hosted-LLM request stalled (pre-2026-04-17 the
client had no timeout; fixed in commit `e2c1868`). Since the fix,
the request times out after 30 s (configurable via
`REMANENTIA_OPENAI_TIMEOUT`, kept under that name for backward
compatibility).

Diagnosis:

```bash
# Tail the bench stdout: it prints [N/500] every _PROGRESS_EVERY
# questions and a [slow Q] heartbeat for any question above timeout.
tail -f bench.log

# Also check the hosted-LLM provider status page — 5xx on their
# side is invisible to our code except as timeouts.
```

Fix: raise `REMANENTIA_OPENAI_TIMEOUT`, or add `--progress-every 5`
to bench invocation so milestones come sooner.

### "unsupported legacy pickle format"

Cause: runtime loader hit a `.pkl` state file after the 2026-04-17
pickle removal.

Fix:

```bash
python tools/migrate_pickle_to_npz.py --path snn_state/
python tools/migrate_pickle_to_npz.py --path memory/
# Dry-run first if unsure:
python tools/migrate_pickle_to_npz.py --dry-run
```

The migrator leaves `.pkl.bak` files for rollback. Delete once
verified.

### CUDA kernel panic during retrieval

Cause: GPU compute capability below what PyTorch was compiled for.
The default torch build dropped sm_61 (GTX 10-series) at some point;
`device_utils.safe_device` now detects and falls back to CPU
automatically — but only for **production** call sites wired in
commit `acfff30`. Third-party plugins that pick a device manually
will still panic.

Fix: either update the plugin to call `safe_device()`, or force
`CUDA_VISIBLE_DEVICES=""` to turn CUDA off entirely.

### Concurrent writers corrupt the knowledge store

Cause: pre-2026-04-17 writers were not locked. Fixed in commit
`c0ee8b5`. If the store is already corrupt at upgrade time:

```bash
# The lock is advisory. Check for stale lock files:
ls memory/.knowledge_store.lock  # safe to delete if no process
                                  # holds it
```

Fix: restore from `BACKUP/` or `00_SAFETY_BACKUPS/REMANENTIA/`.
Atomic writes land via `os.replace`, so mid-crash partial files
have the shape `.<name>.<suffix>.tmp` in the same directory — safe
to delete.

### Test-suite flakes

Three classes of flake are documented and intentional:

- **Performance tests** (`tests/test_pipeline_performance.py`,
  `tests/test_tier3_rust.py::test_performance`): scale every budget
  via `REMANENTIA_PERF_BUDGET_SCALE`. Slow-HW devs use `30`; CI uses
  `1.0`.
- **Cross-test pollution** (`TestCrossTierPipeline::test_triggers_consolidation`):
  pre-existing flake; passes in isolation. Tracked in
  docs/internal/AUDIT_INDEX.md.
- **heartbeat-performance** on NTFS: filesystem latency. Document on
  the operator side; do not try to tighten the budget.

## Backup and restore

- **In-repo backups:** `BACKUP/` (gitignored, quick local).
- **Off-repo backups:** `00_SAFETY_BACKUPS/REMANENTIA/` (per
  `SHARED_CONTEXT.md` § Backup Policy).

Rotate both. Tested restore path:

```bash
# Stop the server
systemctl stop remanentia-api

# Restore a backup
rsync -a --delete 00_SAFETY_BACKUPS/REMANENTIA/memory/ memory/
rsync -a --delete 00_SAFETY_BACKUPS/REMANENTIA/snn_state/ snn_state/

# Verify atomic-write + migrator state is consistent
python tools/migrate_pickle_to_npz.py --dry-run
python -c "from knowledge_store import KnowledgeStore; ks = KnowledgeStore(); ks.load(); print(len(ks.notes))"

# Start the server
systemctl start remanentia-api
curl -sS http://127.0.0.1:8001/health
```

## Upgrade path

1. Read the CHANGELOG entry for the target version.
2. Stop the running server.
3. Upgrade the Python package: `pip install -U remanentia` (or
   `pip install <tarball>` for sigstore-verified installs — see
   SECURITY.md).
4. Run the pickle migrator if the CHANGELOG mentions state-format
   changes.
5. Start the server. Confirm `/health` and `/status` respond as
   expected.

Rollbacks work by reinstalling the prior version; the on-disk
memory format is backward-compatible since 0.4.0.

Last change: 2026-04-17.
