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

* **AnthropicBackend** — hosted LLM via the Anthropic Messages HTTP API
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
from importlib import import_module
from pathlib import Path
from typing import Mapping, Protocol, Sequence, cast, runtime_checkable

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

    llm_section = dict(_mapping(data.get("llm")))
    tokens = _mapping(llm_section.pop("tokens", {}))

    cfg = LLMConfig()
    for key in ("backend", "local_url", "local_model", "local_api_key", "anthropic_model"):
        value = llm_section.get(key)
        if isinstance(value, str):
            setattr(cfg, key, value)
    timeout = llm_section.get("local_timeout")
    if isinstance(timeout, (float, int, str)):
        cfg.local_timeout = float(timeout)
    extract = tokens.get("extract")
    if isinstance(extract, (float, int, str)):
        cfg.max_tokens_extract = int(extract)
    generate = tokens.get("generate")
    if isinstance(generate, (float, int, str)):
        cfg.max_tokens_generate = int(generate)
    synthesise = tokens.get("synthesise")
    if isinstance(synthesise, (float, int, str)):
        cfg.max_tokens_synthesise = int(synthesise)

    return cfg


class _TomlModule(Protocol):
    """Typed view of TOML parser modules."""

    def loads(self, text: str) -> object:
        """Parse TOML text into Python objects."""


def _parse_toml(text: str) -> dict[str, object]:
    """Parse TOML using stdlib tomllib (3.11+) or tomli fallback."""
    try:
        toml_module = import_module("tomllib")
    except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
        try:
            toml_module = import_module("tomli")
        except ModuleNotFoundError:
            log.warning("No TOML parser available; using defaults")
            return {}
    parser = cast(_TomlModule, toml_module)
    return cast(dict[str, object], parser.loads(text))


def _mapping(value: object) -> Mapping[str, object]:
    """Return a string-keyed mapping view for nested configuration objects."""
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _chat_completion_content(body: object) -> str | None:
    """Extract the first OpenAI-compatible chat completion message string."""
    response = _mapping(body)
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        return None
    first_choice = _mapping(choices[0])
    message = _mapping(first_choice.get("message"))
    content = message.get("content")
    if not isinstance(content, str):
        return None
    return content.strip()


def _anthropic_content(body: object) -> str | None:
    """Extract the first text block from an Anthropic Messages response."""
    response = _mapping(body)
    content = response.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str) or not content:
        return None
    first_block = _mapping(content[0])
    text = first_block.get("text")
    if not isinstance(text, str):
        return None
    return text.strip()


# ── Backends ──────────────────────────────────────────────────────


class NullBackend:
    """No-op backend — every call returns ``None``."""

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        system: str = "",
    ) -> str | None:
        """Return no completion for explicit no-LLM operation."""
        return None


class AnthropicBackend:
    """Hosted backend using the Anthropic Messages HTTP API directly."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialise hosted model metadata without opening a connection."""
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        ).rstrip("/")
        parsed = urllib.parse.urlparse(self._base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Anthropic base_url must use http or https")

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        system: str = "",
    ) -> str | None:
        """Return a hosted completion, or ``None`` when unavailable or failed."""
        if not self._api_key:
            return None
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        request = urllib.request.Request(
            f"{self._base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = cast(object, json.loads(response.read().decode("utf-8")))
                return _anthropic_content(body)
        except Exception:
            log.debug("Anthropic request failed", exc_info=True)
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
        """Initialise an OpenAI-compatible local chat completion endpoint."""
        self._base_url = base_url.rstrip("/")
        parsed = urllib.parse.urlparse(self._base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Local LLM base_url must use http or https")
        self._model = model
        self._timeout = timeout
        self._api_key = api_key or os.environ.get("REMANENTIA_LOCAL_LLM_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        """Build JSON request headers with an optional bearer token."""
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
                status = cast(object, getattr(resp, "status", None))
                return status == 200
        except Exception:
            return False

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        system: str = "",
    ) -> str | None:
        """Return a local chat completion, or ``None`` on transport failure."""
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
                body = cast(object, json.loads(resp.read().decode("utf-8")))
                return _chat_completion_content(body)
        except Exception:
            log.debug("Local LLM request failed", exc_info=True)
            return None


class AutoBackend:
    """Tries local → hosted → Null, caches the resolved backend."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialise auto-selection with an optional resolved configuration."""
        self._config = config or LLMConfig()
        self._resolved: LLMBackend | None = None

    def _resolve(self) -> LLMBackend:
        """Resolve and cache the first available backend in preference order."""
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
        """Delegate completion to the cached auto-selected backend."""
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
