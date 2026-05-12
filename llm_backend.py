# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Pluggable LLM backend abstraction

# Repository: https://github.com/anulum/remanentia
"""Pluggable LLM backend abstraction.

Provides a ``LLMBackend`` protocol with four implementations:

* **AnthropicBackend** — hosted LLM via the ``anthropic`` Python SDK
* **LocalLLMBackend** — local chat-completions-compatible server (llama.cpp / Ollama)
* **NullBackend** — always returns ``None`` (explicit no-LLM sentinel)
* **AutoBackend** — tries local → hosted → Null in order
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

# ── Protocol ──────────────────────────────────────────────────────


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal contract every LLM backend must satisfy."""

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        system: str = "",
    ) -> str | None:
        """Return a short text completion, or ``None`` on failure."""
        ...  # pragma: no cover


# ── Config ────────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    """Centralised LLM configuration.

    Defaults target a local Ollama server running ``gemma3:4b``. See
    ``docs/internal/benchmark_2026-04-10_local_llm_evaluation.md`` for the
    rationale behind this choice over Qwen 2.5 7B and Gemma 4.
    """

    backend: str = "auto"
    local_url: str = "http://localhost:11434/v1"
    local_model: str = "gemma3:4b"
    local_api_key: str = ""
    local_timeout: float = 60.0
    anthropic_model: str = "claude-haiku-4-5-20251001"
    max_tokens_extract: int = 100
    max_tokens_generate: int = 200
    max_tokens_synthesise: int = 200


_DEFAULT_CONFIG_DIR = Path.home() / ".remanentia"
_CONFIG_FILENAME = "llm.toml"


def load_config(path: Path | None = None) -> LLMConfig:
    """Load configuration from TOML file.

    Resolution order: *path* argument → ``REMANENTIA_LLM_CONFIG`` env var →
    ``~/.remanentia/llm.toml`` → defaults.
    """
    if path is None:
        env = os.environ.get("REMANENTIA_LLM_CONFIG")
        if env:
            path = Path(env)
        else:
            path = _DEFAULT_CONFIG_DIR / _CONFIG_FILENAME

    if not path.is_file():
        return LLMConfig()

    text = path.read_text(encoding="utf-8")
    data = _parse_toml(text)

    llm_section = data.get("llm", {})
    tokens = llm_section.pop("tokens", {})

    cfg = LLMConfig()
    for key in ("backend", "local_url", "local_model", "local_api_key", "anthropic_model"):
        if key in llm_section:
            setattr(cfg, key, llm_section[key])
    if "local_timeout" in llm_section:
        cfg.local_timeout = float(llm_section["local_timeout"])
    if "extract" in tokens:
        cfg.max_tokens_extract = int(tokens["extract"])
    if "generate" in tokens:
        cfg.max_tokens_generate = int(tokens["generate"])
    if "synthesise" in tokens:
        cfg.max_tokens_synthesise = int(tokens["synthesise"])

    return cfg


def _parse_toml(text: str) -> dict:
    """Parse TOML using stdlib tomllib (3.11+) or tomli fallback."""
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef,import-untyped]
        except ModuleNotFoundError:
            log.warning("No TOML parser available; using defaults")
            return {}
    return tomllib.loads(text)


# ── Backends ──────────────────────────────────────────────────────


class NullBackend:
    """No-op backend — every call returns ``None``."""

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        system: str = "",
    ) -> None:
        return None


class AnthropicBackend:
    """Hosted-LLM backend using the ``anthropic`` Python SDK.

    Lazy import and cached client — no network call at construction.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: object | None = None

    def _get_client(self):
        """Lazy-initialise the hosted-LLM client."""
        if self._client is not None:
            return self._client
        if not self._api_key:
            return None
        try:
            import anthropic  # type: ignore[import-untyped]

            self._client = anthropic.Anthropic(api_key=self._api_key)
            return self._client
        except ImportError:  # pragma: no cover
            return None

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        system: str = "",
    ) -> str | None:
        client = self._get_client()
        if client is None:
            return None
        try:  # pragma: no cover
            messages = [{"role": "user", "content": prompt}]
            kwargs: dict = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system
            response = client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception:  # pragma: no cover
            return None


class LocalLLMBackend:
    """OpenAI-compatible local server backend (llama.cpp / Ollama).

    Communicates via ``urllib`` — zero external dependencies.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "gemma3:4b",
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        parsed = urllib.parse.urlparse(self._base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Local LLM base_url must use http or https")
        self._model = model
        self._timeout = timeout
        self._api_key = api_key or os.environ.get("REMANENTIA_LOCAL_LLM_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def is_available(self) -> bool:
        """Health-check: GET /v1/models succeeds."""
        try:
            req = urllib.request.Request(
                f"{self._base_url}/models",
                headers=self._headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        system: str = "",
    ) -> str | None:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
        except Exception:
            log.debug("Local LLM request failed", exc_info=True)
            return None


class AutoBackend:
    """Tries local → hosted → Null, caches the resolved backend."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        self._resolved: LLMBackend | None = None

    def _resolve(self) -> LLMBackend:
        if self._resolved is not None:
            return self._resolved

        # Try local first
        local = LocalLLMBackend(
            base_url=self._config.local_url,
            model=self._config.local_model,
            timeout=self._config.local_timeout,
            api_key=self._config.local_api_key,
        )
        if local.is_available():
            log.info("AutoBackend: using local LLM at %s", self._config.local_url)
            self._resolved = local
            return local

        # Try the hosted backend
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            log.info("AutoBackend: using the hosted LLM backend")
            anthropic = AnthropicBackend(
                model=self._config.anthropic_model,
                api_key=api_key,
            )
            self._resolved = anthropic
            return anthropic

        # Fallback
        log.info("AutoBackend: no LLM available, using NullBackend")
        self._resolved = NullBackend()
        return self._resolved

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        system: str = "",
    ) -> str | None:
        backend = self._resolve()
        return backend.complete(prompt, max_tokens=max_tokens, system=system)


# ── Factory ───────────────────────────────────────────────────────


def resolve_backend(
    name: str = "auto",
    config: LLMConfig | None = None,
) -> LLMBackend:
    """Create a backend instance from a name string.

    Valid names: ``auto``, ``local``, ``anthropic``, ``none``.
    """
    cfg = config or load_config()
    name = name.lower().strip()

    if name == "none":
        return NullBackend()
    if name == "local":
        return LocalLLMBackend(
            base_url=cfg.local_url,
            model=cfg.local_model,
            timeout=cfg.local_timeout,
            api_key=cfg.local_api_key,
        )
    if name == "anthropic":
        return AnthropicBackend(model=cfg.anthropic_model)
    if name == "auto":
        return AutoBackend(config=cfg)

    log.warning("Unknown backend %r, falling back to auto", name)
    return AutoBackend(config=cfg)
