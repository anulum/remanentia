# memory_guarded

Director-AI bridge for Remanentia's Guarded memory tier. Grounds LLM
answers on retrieved memories before they reach the user.

## Why this module exists

Memory retrieval hands the LLM a set of facts pulled from the store;
the LLM synthesises an answer. Nothing in that pipeline checks that
the answer is *consistent* with the facts. A hallucination slips past.

Remanentia's Guarded tier wires **Director-AI** into the post-
synthesis step: it scores the candidate answer against the retrieved
evidence on two dimensions (logical coherence + factual grounding),
flags prompt-injection attempts, and optionally runs an NLI check. The
server can then return the answer as-is, mark it unverified, or block
it outright.

Director-AI is an **optional dependency**. Installs without it use
the Fast or Accurate tiers and this module's `score_memory_answer`
returns `None`, letting the caller degrade gracefully.

## Public surface

```python
from memory_guarded import (
    is_available,
    GuardedPolicy,
    GuardedResult,
    score_memory_answer,
    facts_from_results,
)
```

### `is_available() -> bool`

Probe whether `director_ai` imports successfully. Cheap: one
`importlib.util.find_spec`. Callers branch on this to skip the scoring
call when Director-AI is not installed.

### `GuardedPolicy(approve_threshold=0.3, block_below=0.15, use_nli=None, injection_detection=True)`

Frozen dataclass of scoring thresholds. Defaults come from the
Director-AI v3.12 calibration:

- `approve_threshold=0.3` — answers at or above this score are
  explicitly approved.
- `block_below=0.15` — answers strictly below this are blocked.
  Between the two values the answer is returned with `approved=False,
  blocked=False`, meaning "unverified; show it but don't vouch".
- `use_nli=None` — leave NLI off (CPU-cheap path). `True` runs the
  NLI head; `False` explicitly disables.
- `injection_detection=True` — run prompt-injection classifier.

### `score_memory_answer(question, answer, facts, *, policy=None) -> GuardedResult | None`

End-to-end grounding check. `facts` is a `dict[name, text]` (built via
`facts_from_results` from a list of retrieval hits). Returns `None`
when Director-AI is unavailable, a populated `GuardedResult` otherwise.

```python
from memory_guarded import score_memory_answer, facts_from_results

facts = facts_from_results(retrieval.results)
r = score_memory_answer(question, answer, facts)
if r is None:
    return answer  # Guarded tier not installed; trust the LLM
if r.blocked:
    return "I don't have enough information to answer that."
if not r.approved:
    return f"{answer} (unverified)"
return answer
```

### `GuardedResult`

| Field | Type | Meaning |
| --- | --- | --- |
| `score` | float | Fused coherence score, 0..1 |
| `approved` | bool | `score ≥ approve_threshold` |
| `blocked` | bool | `score < block_below` |
| `h_logical` | float | Logical-coherence sub-score |
| `h_factual` | float | Factual-grounding sub-score |
| `injection_risk` | float \| None | `None` when detection disabled |
| `evidence` | list[dict] | Supporting chunks with `text` / `distance` / `source` |
| `reason` | str | Free-text explanation for `blocked`/unapproved answers |

`to_dict()` produces a JSON-serialisable form for API responses.

### `facts_from_results(results) -> dict[str, str]`

Adapter: take a list of retrieval-hit dataclasses (anything exposing
`name`, `snippet`, `answer`) and return the `{name: text}` dict
Director-AI expects. Prefers `answer` over `snippet`, dedupes by
first-seen `name`, truncates each fact to 400 chars.

## Policy tuning

The defaults (0.3 / 0.15) come from Director-AI's CalibrationSet v1.
Teams that prefer higher recall (more answers approved at the cost of
more hallucinations slipping through) can widen the gap:

```python
policy = GuardedPolicy(approve_threshold=0.5, block_below=0.2)
```

Teams that prefer higher precision (fewer approved answers, fewer
false positives) tighten it:

```python
policy = GuardedPolicy(approve_threshold=0.2, block_below=0.1)
```

## Invariants

- **Non-destructive.** The module never modifies the answer; it returns
  a verdict and lets the caller decide.
- **Optional-dep safe.** A CI host without Director-AI can still run
  the whole `memory_guarded` test module; tests are split into
  always-run (shape / fact-extractor / policy) and skip-if-unavailable
  (real scoring, mocked) groups.
- **Frozen policy.** `GuardedPolicy` is a `@dataclass(frozen=True)`;
  mutating at runtime raises.

## See also

- [`api_server`](api_server.md) — wires the Guarded check into the
  `/answer` endpoint.
- [`mcp_server`](mcp_server.md) — also wraps answers in Guarded.
- `director_ai` — upstream scoring model.
