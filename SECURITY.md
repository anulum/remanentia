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
2. **Email:** protoscience@anulum.li — Subject: `[SECURITY] Remanentia — <brief description>`
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
- Temporal code execution (`temporal_graph.py`) runs in restricted sandbox
- No network access from retrieval pipeline
- AGPL-3.0 license ensures source availability for audit
