# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the recall query-stream bus emitter

"""Behavioural tests for :mod:`bus_recall`.

The real ``synapse_channel`` agent is replaced by an in-process fake so the
suite exercises the sync→async bridge, the connection lifecycle and every
failure mode without a hub or a network. The three invariants under test:
the emitter is a no-op when the client is absent or not ready, it never
raises into the caller, and a successful emit reaches the agent's
``log_recall`` with the recall-time fields intact.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

import bus_recall
from bus_recall import BusRecallEmitter, _load_agent_factory, default_emitter


class FakeAgent:
    """Stand-in for ``synapse_channel.SynapseAgent``.

    ``connect`` mimics the real coroutine that runs the inbound listener for
    the connection's lifetime: it signals readiness (unless ``never_ready``),
    then idles until ``running`` is cleared or the task is cancelled.
    """

    def __init__(
        self,
        name,
        *,
        uri,
        verbose=False,
        connect_raises=False,
        log_raises=False,
        never_ready=False,
    ):
        self.name = name
        self.uri = uri
        self.verbose = verbose
        self.running = True
        self.logged: list[dict] = []
        self._connect_raises = connect_raises
        self._log_raises = log_raises
        self._never_ready = never_ready
        self._ready_evt = asyncio.Event()

    async def connect(self):
        if self._connect_raises:
            raise RuntimeError("connect boom")
        if not self._never_ready:
            self._ready_evt.set()
        while self.running:
            await asyncio.sleep(0.005)

    async def wait_until_ready(self, timeout=5.0):
        try:
            await asyncio.wait_for(self._ready_evt.wait(), timeout=max(timeout, 0.1))
            return True
        except asyncio.TimeoutError:
            return False

    async def log_recall(
        self, query_text, *, returned_claim_ids=(), was_used=False, abstained=False
    ):
        if self._log_raises:
            raise RuntimeError("log boom")
        self.logged.append(
            {
                "query": query_text,
                "ids": list(returned_claim_ids),
                "was_used": was_used,
                "abstained": abstained,
            }
        )


def make_factory(**agent_kwargs):
    """Build an agent factory that records every agent it constructs."""
    created: list[FakeAgent] = []

    def factory(name, *, uri, verbose=False):
        agent = FakeAgent(name, uri=uri, verbose=verbose, **agent_kwargs)
        created.append(agent)
        return agent

    factory.created = created  # type: ignore[attr-defined]
    return factory


def _wait_for(predicate, timeout=2.0):
    """Poll *predicate* until true or *timeout* elapses; return its result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def emitter_factory():
    """Yield a builder that tracks emitters and closes them on teardown."""
    built: list[BusRecallEmitter] = []

    def build(**kwargs):
        kwargs.setdefault("name", "TEST-recall")
        kwargs.setdefault("connect_timeout", 1.0)
        kwargs.setdefault("shutdown_timeout", 1.0)
        em = BusRecallEmitter(**kwargs)
        built.append(em)
        return em

    yield build
    for em in built:
        em.close()


class TestEmit:
    def test_successful_emit_reaches_log_recall(self, emitter_factory):
        factory = make_factory()
        em = emitter_factory(agent_factory=factory)

        assert em.emit("what is K_nm?", returned_claim_ids=["a:1", "b:2"], abstained=False) is True
        agent = factory.created[0]
        assert _wait_for(lambda: bool(agent.logged))
        rec = agent.logged[0]
        assert rec["query"] == "what is K_nm?"
        assert rec["ids"] == ["a:1", "b:2"]
        assert rec["was_used"] is False
        assert rec["abstained"] is False
        assert em.active is True

    def test_abstained_flag_propagates(self, emitter_factory):
        factory = make_factory()
        em = emitter_factory(agent_factory=factory)

        assert em.emit("unknown query", returned_claim_ids=[], abstained=True) is True
        agent = factory.created[0]
        assert _wait_for(lambda: bool(agent.logged))
        assert agent.logged[0]["abstained"] is True
        assert agent.logged[0]["ids"] == []

    def test_reuses_single_connection(self, emitter_factory):
        factory = make_factory()
        em = emitter_factory(agent_factory=factory)

        em.emit("first", returned_claim_ids=["x:1"])
        em.emit("second", returned_claim_ids=["y:2"])
        agent = factory.created[0]
        assert _wait_for(lambda: len(agent.logged) == 2)
        # One agent built, one connection — no per-event handshake.
        assert len(factory.created) == 1
        assert [r["query"] for r in agent.logged] == ["first", "second"]

    def test_not_ready_is_noop(self, emitter_factory):
        factory = make_factory(never_ready=True)
        em = emitter_factory(agent_factory=factory)

        assert em.emit("q", returned_claim_ids=["a:1"]) is False
        assert em.active is False
        agent = factory.created[0]
        assert agent.logged == []

    def test_connect_failure_is_noop(self, emitter_factory):
        factory = make_factory(connect_raises=True)
        em = emitter_factory(agent_factory=factory)

        assert em.emit("q", returned_claim_ids=["a:1"]) is False
        assert em.active is False

    def test_log_failure_is_swallowed(self, emitter_factory):
        factory = make_factory(log_raises=True)
        em = emitter_factory(agent_factory=factory)

        # Scheduling succeeds; the failing send is swallowed on the loop thread.
        assert em.emit("q", returned_claim_ids=["a:1"]) is True
        agent = factory.created[0]
        # Give the loop a moment; logged stays empty, no exception surfaces.
        time.sleep(0.1)
        assert agent.logged == []


class TestNoSynapse:
    def test_missing_client_disables_emitter(self, emitter_factory, monkeypatch):
        monkeypatch.setattr(bus_recall, "_load_agent_factory", lambda: None)
        em = emitter_factory(agent_factory=None)

        assert em.emit("q", returned_claim_ids=["a:1"]) is False
        assert em.active is False
        # Stays disabled — a second emit short-circuits without restarting.
        assert em.emit("q2", returned_claim_ids=["b:2"]) is False

    def test_load_agent_factory_returns_none_without_module(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "synapse_channel", None)
        assert _load_agent_factory() is None

    def test_load_agent_factory_resolves_real_class(self):
        # synapse-channel is a dev dependency, so the real class resolves here.
        factory = _load_agent_factory()
        assert factory is not None
        assert factory.__name__ == "SynapseAgent"


class TestShutdown:
    def test_close_is_idempotent(self, emitter_factory):
        factory = make_factory()
        em = emitter_factory(agent_factory=factory)
        em.emit("q", returned_claim_ids=["a:1"])

        em.close()
        assert em.active is False
        em.close()  # second close is a no-op
        assert em.active is False

    def test_emit_after_close_is_noop(self, emitter_factory):
        factory = make_factory()
        em = emitter_factory(agent_factory=factory)
        em.emit("q", returned_claim_ids=["a:1"])
        em.close()

        assert em.emit("late", returned_claim_ids=["b:2"]) is False

    def test_close_before_start_is_noop(self, emitter_factory):
        factory = make_factory()
        em = emitter_factory(agent_factory=factory)
        # Never emitted → loop never started → close short-circuits.
        em.close()
        assert em.active is False


class _FlipLock:
    """A lock that flips an emitter flag the moment it is acquired.

    Simulates the race the double-checked locking in ``_ensure_started``
    guards against: the state changes between the lock-free fast path and the
    critical section, so the re-validation inside the lock must catch it.
    """

    def __init__(self, emitter, attr):
        self._emitter = emitter
        self._attr = attr
        self._real = __import__("threading").Lock()

    def __enter__(self):
        setattr(self._emitter, self._attr, True)
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)


class TestRaceGuards:
    def test_closed_between_fastpath_and_lock(self, emitter_factory):
        em = emitter_factory(agent_factory=make_factory())
        em._lock = _FlipLock(em, "_closed")
        assert em._ensure_started() is False

    def test_started_between_fastpath_and_lock(self, emitter_factory):
        em = emitter_factory(agent_factory=make_factory())
        em._lock = _FlipLock(em, "_started")
        assert em._ensure_started() is True


class TestStartupAndEmitFailures:
    def test_factory_raising_disables_emitter(self, emitter_factory):
        def boom_factory(name, *, uri, verbose=False):
            raise RuntimeError("factory boom")

        em = emitter_factory(agent_factory=boom_factory)
        assert em.emit("q", returned_claim_ids=["a:1"]) is False
        assert em.active is False

    def test_emit_swallows_argument_failure(self, emitter_factory):
        factory = make_factory()
        em = emitter_factory(agent_factory=factory)
        # Prime a ready connection so emit reaches the scheduling try-block.
        assert em.emit("warmup", returned_claim_ids=["a:1"]) is True
        assert _wait_for(lambda: bool(factory.created[0].logged))

        class BadSeq:
            def __iter__(self):
                raise RuntimeError("bad sequence")

        # ``list(returned_claim_ids)`` raises inside emit; it must be swallowed.
        assert em.emit("q", returned_claim_ids=BadSeq()) is False


class TestDefaultEmitter:
    def test_disabled_via_env_returns_none(self, monkeypatch):
        monkeypatch.setenv("REMANENTIA_RECALL_BUS_DISABLE", "1")
        assert default_emitter() is None

    def test_builds_with_env_identity(self, monkeypatch):
        monkeypatch.delenv("REMANENTIA_RECALL_BUS_DISABLE", raising=False)
        monkeypatch.setenv("REMANENTIA_RECALL_BUS_NAME", "custom-recall")
        monkeypatch.setenv("REMANENTIA_SYNAPSE_URI", "ws://example:9999")

        em = default_emitter()
        assert em is not None
        try:
            assert em._name == "custom-recall"
            assert em._uri == "ws://example:9999"
        finally:
            em.close()

    def test_defaults_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("REMANENTIA_RECALL_BUS_DISABLE", raising=False)
        monkeypatch.delenv("REMANENTIA_RECALL_BUS_NAME", raising=False)
        monkeypatch.delenv("REMANENTIA_SYNAPSE_URI", raising=False)

        em = default_emitter()
        assert em is not None
        try:
            assert em._name == "REMANENTIA-recall"
            assert em._uri == bus_recall.DEFAULT_HUB_URI
        finally:
            em.close()
