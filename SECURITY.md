# Security Policy

## Supported Versions

| Version | Supported          | Notes |
|---------|--------------------|-------|
| 0.3.x   | :white_check_mark: | Current development (alpha) |
| < 0.3   | :x:                | Superseded |

Only the latest `0.x` patch receives security fixes.

## Reporting a Vulnerability

If you discover a security vulnerability in Remanentia, please report it
responsibly:

1. **GitHub Security Advisories** (preferred):
   [Report a vulnerability](https://github.com/anulum/remanentia/security/advisories/new)
2. **Email:** remanentia@anulum.li or protoscience@anulum.li — Subject: `[SECURITY] Remanentia — <brief description>`
3. **Do not** open a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and aim to provide a fix within
7 days for critical issues.

## Scope

The following are in scope:

- Memory data exposure (traces, semantic memories, entity graphs)
- MCP server authentication bypass
- Path traversal in file indexing
- Arbitrary code execution via temporal code evaluation
- Credential leakage in reasoning traces

The following are out of scope:

- Denial of service via large index builds (known limitation)
- Information disclosure via MCP tools (by design — MCP clients are trusted)

## Security Design

- No LLM calls in core retrieval path — no prompt injection vector
- Temporal code execution (`temporal_graph.py`) limited to datetime arithmetic — no eval/exec
- No network access from retrieval pipeline
- AGPL-3.0 license ensures source availability for audit
- Legacy pickle load path removed from every runtime (2026-04-17); see
  `tools/migrate_pickle_to_npz.py` for the one-shot migrator
- PII redaction (`pii_redactor.py`) scrubs emails, phone numbers, IBAN,
  credit cards, and API-key-shaped tokens from every memory write
- API server requires Bearer auth (`REMANENTIA_API_TOKEN`), per-IP rate
  limit, and a 1 MiB body cap by default; `/health` is the only exempt
  path
- Memory-store writes use atomic `os.replace` and advisory `flock` so
  concurrent writers cannot tear files or lose updates
- Public-release leak audit (`python tools/public_leak_audit.py`) scans
  tracked public text files for private workspace paths, private workspace
  labels, and agent-identity labels; it is wired into pre-commit and the
  release checklist.

## Verifying a Released Artefact

Every release publishes three verifiable attestations alongside the
source distribution and wheel:

1. **CycloneDX SBOM** — `sbom.cyclonedx.json`, lists every transitive
   dependency and its version.
2. **Sigstore keyless signatures** — `.sigstore` bundles are published
   for every artefact. Verify identity and digest with::

       pip install sigstore
       sigstore verify identity \
         --cert-identity 'https://github.com/anulum/remanentia/.github/workflows/release.yml@refs/tags/vX.Y.Z' \
         --cert-oidc-issuer 'https://token.actions.githubusercontent.com' \
         remanentia-X.Y.Z.tar.gz

3. **SLSA build provenance** — GitHub-native attestation emitted by
   ``actions/attest-build-provenance``. Verify with::

       gh attestation verify remanentia-X.Y.Z.tar.gz \
         --repo anulum/remanentia

A release that fails any of these checks should not be installed.

Maintainers can run `python tools/check_release_integrity.py` before tagging
to verify that the tracked release workflow still builds the SBOM, signs and
locally verifies release artefacts, emits SLSA provenance, and uploads the
`.sigstore` bundles advertised above.

Maintainers must also run `python tools/public_leak_audit.py` before tagging.
The audit must report zero findings before publishing a release candidate.
