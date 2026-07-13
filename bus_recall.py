# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — recall query-stream bus emitter (MS.1: fleet-wide sink)

"""Bridge the recall query-stream onto the SYNAPSE fleet bus.

The MCP server (:mod:`mcp_server`) is fully synchronous, but the fleet hub
client (``synapse-channel`` ≥ 0.37.0) is asyncio-based:
``SynapseAgent.connect`` opens a websocket and *runs the inbound listener for
the lifetime of the connection*, and ``log_recall`` is a coroutine that sends
one ``recall_log`` event over that connection. This module owns the only
bridge between the two worlds — a single persistent ``SynapseAgent`` living on
a dedicated daemon-thread event loop — so a synchronous ``emit`` call from the
recall path schedules the async send without ever blocking the recall or
holding the caller's thread.

It is the **second sink** for the same query-stream the local
:mod:`recall_ledger` records: one recall, two sinks. The local ledger is the
durable per-process calibration substrate; the bus carries the same
query-weighted distribution to the fleet feed so any agent's persistent-memory
layer can calibrate against the real questions asked across the whole fleet,
not just its own.

Three hard invariants, in priority order:

1. **Optional.** Remanentia never hard-depends on the bus. If
   ``synapse_channel`` is not importable, every :meth:`BusRecallEmitter.emit`
   is a silent no-op and nothing is started.
2. **Non-blocking.** ``emit`` returns immediately; the websocket send runs
   fire-and-forget on the loop thread. The recall return path never waits on
   the network.
3. **Never raises into recall.** Telemetry must not break memory: every
   failure here is swallowed and logged at ``debug``.

What the wire carries is honestly bounded by what the 0.37.0 protocol exposes.
``log_recall`` is fire-once with no event identifier, so the emitter sends the
*recall-time* facts only — ``query_text``, ``returned_claim_ids`` and
``abstained`` (objectively, whether anything came back). The realised
``was_used`` outcome stays a separate, recency-linked record in the local
ledger; propagating it to the fleet waits on an outcome/event-id seam on the
wire (requested from SYNAPSE-CHANNEL), so no weakly-linked duplicate events
pollute the feed in the meantime.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any, cast

log = logging.getLogger(__name__)

DEFAULT_HUB_URI = "ws://localhost:8876"
"""Hub websocket URI used when ``REMANENTIA_SYNAPSE_URI`` is unset; matches
``synapse_channel.DEFAULT_HUB_URI`` so the emitter reaches the same hub the
rest of the fleet uses."""

AgentFactory = Callable[..., Any]
"""Callable that builds a hub agent: ``factory(name, *, uri, verbose) -> agent``
where the agent exposes the async ``connect`` / ``wait_until_ready`` /
``log_recall`` surface and a ``running`` flag. Injected in tests; resolved from
``synapse_channel.SynapseAgent`` in production."""


def _load_agent_factory() -> AgentFactory | None:
    """Soft-import the hub agent class; return ``None`` if unavailable.

    The import is deliberately local and guarded: a missing or broken
    ``synapse_channel`` must degrade the bus to a no-op, never crash the
    import of the recall path that depends on this module.
    """
    try:
        agent_module = import_module("synapse_channel.client.agent")
        factory = getattr(agent_module, "SynapseAgent", None)
    except Exception:  # pragma: no cover — exercised via injected factory in tests
        return None
    return cast(AgentFactory, factory) if callable(factory) else None


class BusRecallEmitter:
    """Persistent, fire-and-forget emitter of recall events to the fleet bus.

    A single :class:`BusRecallEmitter` owns one daemon thread running one
    asyncio event loop on which one ``SynapseAgent`` stays connected. The
    connection is established lazily on the first :meth:`emit`; subsequent
    emits reuse it (no per-event handshake, no roster churn). All public
    methods are safe to call from the synchronous recall path.

    Parameters
    ----------
    name:
        Identity the emitter registers under on the hub. Kept distinct from
        the main project agent so recall telemetry is visible as its own seat
        and never evicts the project's primary connection.
    uri:
        Hub websocket URI.
    agent_factory:
        Override for the hub agent constructor (tests inject a fake). When
        ``None``, :func:`_load_agent_factory` resolves the real class, and an
        unavailable ``synapse_channel`` disables the emitter.
    connect_timeout:
        Seconds to wait for the hub's welcome before treating the connection
        as not-ready (emits no-op until a later attempt succeeds).
    shutdown_timeout:
        Seconds :meth:`close` waits for the loop thread to drain and exit.
    """

    def __init__(
        self,
        *,
        name: str,
        uri: str = DEFAULT_HUB_URI,
        agent_factory: AgentFactory | None = None,
        connect_timeout: float = 2.0,
        shutdown_timeout: float = 2.0,
    ) -> None:
        self._name = name
        self._uri = uri
        self._explicit_factory = agent_factory
        self._connect_timeout = connect_timeout
        self._shutdown_timeout = shutdown_timeout

        self._lock = threading.Lock()
        self._started = False
        self._disabled = False
        self._ready = False
        self._closed = False
        self._startup_done = threading.Event()

        self._factory: AgentFactory | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._agent: Any = None
        self._connect_task: asyncio.Future[Any] | None = None

    # ── lifecycle ────────────────────────────────────────────────

    def _ensure_started(self) -> bool:
        """Start the loop thread once; return whether the bus is usable.

        Returns ``False`` (and stays a no-op forever) when the hub client is
        not importable or the emitter has been closed.
        """
        if self._disabled or self._closed:
            return False
        if self._started:
            return True
        with self._lock:
            if self._disabled or self._closed:
                return False
            if self._started:
                return True
            factory = self._explicit_factory or _load_agent_factory()
            if factory is None:
                self._disabled = True
                return False
            self._factory = factory
            self._startup_done.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="remanentia-recall-bus", daemon=True
            )
            self._thread.start()
            # Bounded wait: the welcome handshake or its timeout, plus slack.
            self._startup_done.wait(timeout=self._connect_timeout + 1.0)
            self._started = True
            return True

    def _run_loop(self) -> None:
        """Daemon-thread entry point: own one event loop for the connection."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._startup())
        except Exception:  # pragma: no cover — startup guards its own failures
            log.debug("Recall bus loop startup crashed", exc_info=True)
        finally:
            self._startup_done.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _startup(self) -> None:
        """Construct the agent, launch its connect loop, await readiness."""
        try:
            # _ensure_started sets _factory before this thread is started.
            assert self._factory is not None
            self._agent = self._factory(self._name, uri=self._uri, verbose=False)
            self._connect_task = asyncio.ensure_future(self._agent.connect())
            self._ready = await self._agent.wait_until_ready(timeout=self._connect_timeout)
        except Exception:
            self._ready = False
            log.debug("Recall bus connect failed", exc_info=True)

    # ── emission ─────────────────────────────────────────────────

    def emit(
        self,
        query_text: str,
        *,
        returned_claim_ids: Sequence[str],
        was_used: bool = False,
        abstained: bool = False,
    ) -> bool:
        """Schedule one recall event onto the bus; never block, never raise.

        Returns ``True`` when the send was scheduled, ``False`` for any no-op
        (hub client absent, connection not ready, or emitter closed). The
        return value is advisory — callers must treat the bus as best-effort.
        """
        try:
            if not self._ensure_started():
                return False
            loop = self._loop
            if not self._ready or loop is None or self._closed:
                return False
            coro = self._safe_log(
                query_text,
                returned_claim_ids=list(returned_claim_ids),
                was_used=was_used,
                abstained=abstained,
            )
            asyncio.run_coroutine_threadsafe(coro, loop)
            return True
        except Exception:
            log.debug("Recall bus emit failed", exc_info=True)
            return False

    async def _safe_log(
        self,
        query_text: str,
        *,
        returned_claim_ids: list[str],
        was_used: bool,
        abstained: bool,
    ) -> None:
        """Await the agent's ``log_recall``, swallowing any send failure."""
        try:
            await self._agent.log_recall(
                query_text,
                returned_claim_ids=returned_claim_ids,
                was_used=was_used,
                abstained=abstained,
            )
        except Exception:
            log.debug("Recall bus log_recall failed", exc_info=True)

    # ── shutdown ─────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """Whether the emitter has a started, ready, open connection."""
        return self._started and self._ready and not self._closed

    def close(self) -> None:
        """Stop the connection and join the loop thread. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._ashutdown(), loop)
        except Exception:  # pragma: no cover — loop already gone
            log.debug("Recall bus shutdown scheduling failed", exc_info=True)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._shutdown_timeout)

    async def _ashutdown(self) -> None:
        """Cancel the connect loop, drain it, then stop the event loop."""
        if self._agent is not None:
            self._agent.running = False
        task = self._connect_task
        if task is not None:
            task.cancel()
            try:
                await task
            except Exception:
                log.debug("Recall bus connect task shutdown failed", exc_info=True)
        if self._loop is not None:
            self._loop.stop()


def default_emitter() -> BusRecallEmitter | None:
    """Build the process-wide recall bus emitter from the environment.

    Returns ``None`` when ``REMANENTIA_RECALL_BUS_DISABLE`` is set, so callers
    can treat the bus as entirely absent without a started thread. Identity
    defaults to ``"<project>-recall"`` (``REMANENTIA_RECALL_BUS_NAME``) and the
    hub URI to ``REMANENTIA_SYNAPSE_URI`` or :data:`DEFAULT_HUB_URI`.
    """
    if os.environ.get("REMANENTIA_RECALL_BUS_DISABLE"):
        return None
    name = os.environ.get("REMANENTIA_RECALL_BUS_NAME", "REMANENTIA-recall")
    uri = os.environ.get("REMANENTIA_SYNAPSE_URI", DEFAULT_HUB_URI)
    return BusRecallEmitter(name=name, uri=uri)
