# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the recall query-stream bus emitter

"""Real-hub lifecycle and emission tests for :mod:`bus_recall`."""

from __future__ import annotations

import json
import socket
import sqlite3
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast, overload

import pytest

import bus_recall
from bus_recall import BusRecallEmitter, _load_agent_factory, default_emitter


@contextmanager
def _real_hub(tmp_path: Path) -> Iterator[tuple[str, Path, subprocess.Popen[str]]]:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = cast(int, probe.getsockname()[1])
    probe.close()
    db_path = tmp_path / "hub.db"
    process = subprocess.Popen(
        [
            "synapse",
            "hub",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--db",
            str(db_path),
            "--log-level",
            "ERROR",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(100):
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise RuntimeError(f"Synapse hub exited during startup: {stderr}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                    break
            except OSError:
                time.sleep(0.02)
        else:
            raise RuntimeError("Synapse hub did not bind its test port")
        yield f"ws://127.0.0.1:{port}", db_path, process
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _payloads(db_path: Path, expected: int) -> list[dict[str, object]]:
    for _ in range(150):
        if db_path.exists():
            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    "select payload from events where kind = 'recall' order by seq"
                ).fetchall()
            if len(rows) >= expected:
                return [dict(json.loads(raw)) for (raw,) in rows]
        time.sleep(0.02)
    raise AssertionError(f"expected {expected} durable recall events")


def _closed_uri() -> str:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = cast(int, probe.getsockname()[1])
    probe.close()
    return f"ws://127.0.0.1:{port}"


class TestRealHubEmission:
    def test_success_reuse_and_abstention_reach_durable_store(self, tmp_path: Path) -> None:
        with _real_hub(tmp_path) as (uri, db_path, _process):
            emitter = BusRecallEmitter(
                name="REMANENTIA-bus-test",
                uri=uri,
                connect_timeout=3,
                shutdown_timeout=2,
            )
            try:
                assert emitter.emit("first", returned_claim_ids=["a:1", "b:2"]) is True
                assert emitter.active is True
                assert emitter.emit("second", returned_claim_ids=["c:3"]) is True
                assert emitter.emit("unknown", returned_claim_ids=[], abstained=True) is True
                payloads = _payloads(db_path, 3)
            finally:
                emitter.close()

        assert [payload["query_text"] for payload in payloads] == [
            "first",
            "second",
            "unknown",
        ]
        assert payloads[0]["returned_claim_ids"] == ["a:1", "b:2"]
        assert payloads[0]["was_used"] is False
        assert payloads[2]["abstained"] is True
        assert {payload["by"] for payload in payloads} == {"REMANENTIA-bus-test"}
        assert emitter.active is False

    def test_real_connection_failure_is_noop(self) -> None:
        emitter = BusRecallEmitter(
            name="REMANENTIA-unreachable",
            uri=_closed_uri(),
            connect_timeout=0.1,
            shutdown_timeout=0.2,
        )
        try:
            assert emitter.emit("q", returned_claim_ids=["a:1"]) is False
            assert emitter.active is False
            assert emitter.emit("q2", returned_claim_ids=["b:2"]) is False
        finally:
            emitter.close()

    def test_hub_shutdown_after_connection_never_raises(self, tmp_path: Path) -> None:
        with _real_hub(tmp_path) as (uri, db_path, process):
            emitter = BusRecallEmitter(name="REMANENTIA-drop-test", uri=uri, connect_timeout=3)
            assert emitter.emit("before-drop", returned_claim_ids=["a:1"]) is True
            _payloads(db_path, 1)
            process.terminate()
            process.wait(timeout=5)
            assert emitter.emit("after-drop", returned_claim_ids=["b:2"]) is True
            time.sleep(0.1)
            emitter.close()


class _BrokenSequence(Sequence[str]):
    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: int | slice) -> str | Sequence[str]:
        raise RuntimeError("caller sequence failed")

    def __len__(self) -> int:
        return 1


class TestLifecycleAndInputFailure:
    def test_invalid_sequence_is_swallowed_after_real_connection(self, tmp_path: Path) -> None:
        with _real_hub(tmp_path) as (uri, db_path, _process):
            emitter = BusRecallEmitter(name="REMANENTIA-input-test", uri=uri, connect_timeout=3)
            try:
                assert emitter.emit("warmup", returned_claim_ids=["a:1"]) is True
                _payloads(db_path, 1)
                assert emitter.emit("broken", returned_claim_ids=_BrokenSequence()) is False
            finally:
                emitter.close()

    def test_close_before_start_and_after_close_are_idempotent(self) -> None:
        emitter = BusRecallEmitter(name="REMANENTIA-close-test", uri=_closed_uri())
        emitter.close()
        emitter.close()
        assert emitter.active is False
        assert emitter.emit("late", returned_claim_ids=["a:1"]) is False


class TestDependencyResolution:
    def test_load_agent_factory_resolves_real_class(self) -> None:
        factory = _load_agent_factory()
        assert factory is not None
        assert factory.__name__ == "SynapseAgent"

    def test_isolated_interpreter_has_real_missing_dependency_fallback(self) -> None:
        module_path = Path(bus_recall.__file__).resolve()
        script = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('bus_recall_isolated', {str(module_path)!r})
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module._load_agent_factory() is None
emitter = module.BusRecallEmitter(name='isolated')
assert emitter.emit('q', returned_claim_ids=['a:1']) is False
"""
        completed = subprocess.run(
            ["/usr/bin/python3", "-I", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr


class TestDefaultEmitter:
    def test_disabled_via_env_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REMANENTIA_RECALL_BUS_DISABLE", "1")
        assert default_emitter() is None

    def test_builds_with_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REMANENTIA_RECALL_BUS_DISABLE", raising=False)
        monkeypatch.setenv("REMANENTIA_RECALL_BUS_NAME", "custom-recall")
        monkeypatch.setenv("REMANENTIA_SYNAPSE_URI", "ws://example:9999")

        emitter = default_emitter()
        assert emitter is not None
        try:
            assert emitter._name == "custom-recall"
            assert emitter._uri == "ws://example:9999"
        finally:
            emitter.close()

    def test_defaults_when_environment_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REMANENTIA_RECALL_BUS_DISABLE", raising=False)
        monkeypatch.delenv("REMANENTIA_RECALL_BUS_NAME", raising=False)
        monkeypatch.delenv("REMANENTIA_SYNAPSE_URI", raising=False)

        emitter = default_emitter()
        assert emitter is not None
        try:
            assert emitter._name == "REMANENTIA-recall"
            assert emitter._uri == bus_recall.DEFAULT_HUB_URI
        finally:
            emitter.close()
