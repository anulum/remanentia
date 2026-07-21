# no_egress_audit

Sovereign no-egress audit: prove a run made **zero cloud LLM calls**. Part of the
multi-axis evaluation harness (roadmap W1/W3) for the local-first/sovereign axis
no public memory leaderboard scores.

## Why this module exists

The "governed / sovereign / no-egress local-first" claim must be *auditable*, not
asserted: can the system answer with no cloud model in the loop, and at what
accuracy cost. Every model endpoint a run touches is classified local vs cloud,
and the run is **pure-local iff zero cloud calls were made**.

The stance is conservative — an unknown or empty endpoint counts as **cloud**.
Sovereignty must be proven (the endpoint visibly points at a loopback address or
an on-device runtime); a misconfigured run that silently reached a cloud API
fails the audit rather than passing by default. A network URL (http/https/ws/wss)
is judged by its **host alone**: local iff the host is a loopback IP literal or
`localhost`. A runtime name in the URL cannot rescue it (`https://ollama.com/v1`
is egress no matter what it runs), and a loopback substring in a DNS name
(`https://localhost.evil.com`) proves nothing.

## Public surface

```python
from no_egress_audit import classify_endpoint, EgressMonitor, EgressVerdict, audit_endpoints
```

- `classify_endpoint(endpoint) -> "local" | "cloud"` — network URLs: local only
  when the host is loopback (`localhost`, `127.0.0.0/8`, `::1`, unspecified);
  non-network descriptors: local only for on-device markers (`ollama://…`,
  `llama.cpp`, `file://`, `unix:`, `in-process`, …); empty/unknown → cloud.
- `EgressMonitor.record(endpoint)` / `.verdict()` — accumulate calls during a run.
- `audit_endpoints(endpoints) -> EgressVerdict` — audit a recorded list.
- `EgressVerdict(pure_local, total_calls, cloud_calls, by_endpoint, cloud_endpoints)`;
  truthy when sovereign.

## Invariants

- **Prove-local, not assume-local.** Unknown endpoint = cloud.
- **Deterministic**, no model calls. Empty run is vacuously pure-local.

## See also

- `world_class_scorecard` — folds the verdict into the comparable scorecard.
- `plan_2026-06-29_sota_world_class_roadmap.md` — W1/W3 context.
