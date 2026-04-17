# ADR-0003: Pluggable LLM backend via Protocol

- **Status:** Accepted
- **Date:** 2026-04-04 (retroactive ADR, 2026-04-17)
- **Module:** `llm_backend.py`

## Context

Remanentia needs to reason over retrieved memories when the rule-based
answer extractor is not enough. The bench and the MCP server both
call into "the LLM", but the project targets four different deploy
modes — OpenAI, Anthropic, local Ollama, and pure retrieval (no LLM
for offline or air-gapped installs). Hard-coding any of those would
have prevented the others from working.

## Decision

Define a narrow `LLMBackend` Protocol in `llm_backend.py` with
``complete(prompt, max_tokens) -> str | None`` and
``is_available() -> bool``. Concrete backends (``OpenAIBackend``,
``AnthropicBackend``, ``LocalLLMBackend``, ``NullBackend``) implement
the Protocol; ``resolve_backend()`` picks one at runtime based on
env vars and installed dependencies. Call sites hold only the
Protocol type.

## Options considered

- **Vendor-specific hard import.** Locks the project to one provider.
- **LangChain / LlamaIndex wrapper.** Large transitive dep for a
  four-method surface.
- **Protocol (chosen).** Zero runtime deps, clean typing, trivially
  mockable in tests.

## Consequences

- Positive: plug-in a new backend by subclassing the Protocol; tests
  use ``NullBackend`` for offline runs; the bench can switch providers
  with an env var.
- Negative: the Protocol is the **only** extension seam today; there
  is no plugin framework for retrievers, rankers, or consolidation
  strategies. Third-party extensions of those areas require a fork.
- Follow-up: when a second extension point is actually needed (a
  third-party asks), lift this pattern into a proper plugin loader;
  until then, the Protocol alone is correct.
