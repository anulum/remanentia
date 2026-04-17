# ADR-0002: Remove pickle at runtime; ship one-shot migrator

- **Status:** Accepted
- **Date:** 2026-04-17
- **Commits:** `9e31575` (migrator + removal), `bacccd6` (Bandit re-enable)

## Context

Seven production loaders — ``snn_backend._load_state``,
``snn_daemon._load_daemon_state``, ``retrieve._load_pickle_safe``,
``retrieve._load_checkpoint``, ``extractors/physical._extract_from_snn``,
``memory_index._load_legacy_meta`` and
``build_memory_standalone.MemoryIndex.load`` — fell back to
``pickle.load`` for pre-0.4 saves. Each carried a ``# noqa: S301 —
legacy format migration`` marker and had been deferred since the
2026-04-05 Gemini audit. ``pickle.load`` on an attacker-controlled
file is an arbitrary-code-execution primitive. Keeping the fallback
alive "for compat" traded security for convenience indefinitely.

## Decision

Remove the pickle fallback from every runtime loader. Operators with
legacy ``.pkl`` state run ``python tools/migrate_pickle_to_npz.py``
to convert files to npz / gzip-JSON (atomic, with ``.pkl.bak``
backups). Bandit B301 and B403 re-enabled globally in CI; the
migrator itself carries per-line ``# nosec`` with a documented
rationale.

## Options considered

- **Keep pickle fallback with a warning.** Ongoing CVE-exposure; the
  warning was already in place and had not motivated migration in six
  months.
- **``RestrictedUnpickler`` allow-listing.** Non-trivial maintenance
  burden for a migration path that only needs to run once per
  deployment.
- **Remove + migrator (chosen).** One clear error message ("run the
  migrator") beats a silent load of suspect bytes.

## Consequences

- Positive: zero runtime pickle surface. Bandit clean across 24
  production modules. Clear upgrade path for operators.
- Negative: a one-shot operator action is required on upgrade;
  the migrator runs in seconds but has to be invoked.
- Follow-up: audit other deserialisers for the same shape
  (`json.loads`, `numpy.load(allow_pickle=True)` — already set to
  `False` everywhere in production).
