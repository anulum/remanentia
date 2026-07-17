# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — security-hardening regression tests

"""Regression tests for hardened optional-backend and identifier paths."""

from __future__ import annotations

import hashlib
import logging
import types
import asyncio
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import pytest

import answer_normalizer
from bus_recall import BusRecallEmitter
from entity_extractor import extract_entities
from knowledge_store import KnowledgeStore, _note_id


class LegacySentenceTransformer:
    """SentenceTransformer stand-in for the legacy constructor fallback path."""

    def __init__(self, *_args: object, **kwargs: object) -> None:
        """Reject keyword constructors to force the legacy loader path."""
        if kwargs:
            raise TypeError("legacy constructor")

    def to(self, device: str) -> "LegacySentenceTransformer":
        """Raise during device placement so the loader must log the failure."""
        raise RuntimeError(f"cannot move to {device}")


def test_legacy_embedding_cpu_move_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The semantic-similarity loader logs recoverable CPU-placement failures."""
    fake_module = types.SimpleNamespace(SentenceTransformer=LegacySentenceTransformer)
    old_model = answer_normalizer._embed_model
    answer_normalizer._embed_model = None
    try:
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            caplog.at_level(logging.DEBUG, logger="answer_normalizer"),
        ):
            model = answer_normalizer._get_embed_model()
    finally:
        answer_normalizer._embed_model = old_model

    assert isinstance(model, LegacySentenceTransformer)
    assert "Embedding model CPU placement failed" in caplog.text


class ChunkFailingModel:
    """GLiNER stand-in that fails first chunk and succeeds on the second."""

    def __init__(self) -> None:
        """Track the number of public prediction calls."""
        self.calls = 0

    def predict_entities(
        self,
        _text: str,
        _labels: list[str],
        *,
        threshold: float = 0.4,
    ) -> list[dict[str, str | float]]:
        """Return a real entity after one recoverable chunk failure."""
        assert threshold == 0.4
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first chunk failed")
        return [{"text": "BM25", "label": "algorithm", "score": 0.91, "start": 2, "end": 6}]


def test_gliner_chunk_failure_is_logged_and_later_chunks_continue(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The GLiNER path logs chunk failures while preserving later predictions."""
    model = ChunkFailingModel()
    long_text = ("x" * 1501) + " BM25"
    with (
        patch("entity_extractor._load_gliner", return_value=model),
        caplog.at_level(logging.DEBUG, logger="entity_extractor"),
    ):
        entities = extract_entities(long_text)

    assert [entity.text for entity in entities] == ["BM25"]
    assert "GLiNER entity extraction failed for chunk" in caplog.text


class ClosingAgent:
    """Minimal agent that lets :class:`BusRecallEmitter` start and close."""

    def __init__(self, name: str, *, uri: str, verbose: bool = False) -> None:
        """Store connection metadata and runtime state."""
        self.name = name
        self.uri = uri
        self.verbose = verbose
        self.running = True
        self._ready = asyncio.Event()

    async def connect(self) -> None:
        """Signal readiness and then idle until cancelled."""
        self._ready.set()
        while self.running:
            await asyncio.sleep(0.005)

    async def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Return true once the fake connection has signalled readiness."""
        await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        return True

    async def log_recall(
        self,
        _query_text: str,
        *,
        returned_claim_ids: tuple[str, ...] = (),
        was_used: bool = False,
        abstained: bool = False,
    ) -> None:
        """Accept recall logs without contacting a hub."""
        assert returned_claim_ids or not was_used or abstained


class RaisingAwaitableTask:
    """Task wrapper whose awaited shutdown path raises a non-cancellation error."""

    def __init__(self, original: asyncio.Future[Any]) -> None:
        """Keep the original task so cancellation still reaches the loop."""
        self._original = original

    def cancel(self) -> bool:
        """Cancel the real task before simulating a drain failure."""
        return bool(self._original.cancel())

    def done(self) -> bool:
        """Mirror the wrapped Future so shutdown's done()-guard works."""
        return bool(self._original.done())

    def __await__(self) -> Iterator[Any]:
        """Raise while the emitter drains the connection task."""

        async def _raise() -> None:
            raise RuntimeError("shutdown drain failed")

        return _raise().__await__()


def test_cancelled_connect_task_error_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Bus shutdown logs non-cancellation errors from the connect task drain."""
    factory: Callable[..., ClosingAgent] = ClosingAgent
    emitter = BusRecallEmitter(
        name="TEST-recall",
        agent_factory=factory,
        connect_timeout=1.0,
        shutdown_timeout=1.0,
    )
    try:
        assert emitter.emit("q", returned_claim_ids=["a:1"]) is True
        original_task = emitter._connect_task
        assert original_task is not None
        emitter._connect_task = RaisingAwaitableTask(original_task)  # type: ignore[assignment]
        with caplog.at_level(logging.DEBUG, logger="bus_recall"):
            emitter.close()
    finally:
        emitter.close()

    assert "Recall bus connect task shutdown failed" in caplog.text


def test_note_and_trigger_ids_use_sha256_contract() -> None:
    """Knowledge-store identifiers use SHA-256, not MD5."""
    note_expected = hashlib.sha256(b"content:source").hexdigest()[:12]
    assert _note_id("content", "source") == note_expected

    store = KnowledgeStore()
    trigger = store.add_trigger("scpn-control", "Check weights")
    trigger_expected = hashlib.sha256(b"scpn-control:Check weights").hexdigest()[:12]
    assert trigger.id == trigger_expected
