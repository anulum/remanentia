# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for atomic writes + advisory file locking

from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from typing import Protocol

import pytest

from file_utils import (
    FileLock,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)


class _Event(Protocol):
    """Minimal event contract shared by thread and process synchronizers."""

    def set(self) -> None:
        """Mark the event as ready."""

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until the event is ready or *timeout* expires."""


# ── atomic writes ────────────────────────────────────────────────────


class TestAtomicWriteText:
    def test_writes_to_target(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        atomic_write_text(p, "hello")
        assert p.read_text() == "hello"

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        p.write_text("old")
        atomic_write_text(p, "new")
        assert p.read_text() == "new"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "c" / "out.txt"
        atomic_write_text(p, "x")
        assert p.read_text() == "x"

    def test_encoding_utf8_default(self, tmp_path: Path) -> None:
        p = tmp_path / "utf.txt"
        atomic_write_text(p, "Žižkov ¥500 🎯")
        assert p.read_text(encoding="utf-8") == "Žižkov ¥500 🎯"

    def test_no_tmp_leak_on_success(self, tmp_path: Path) -> None:
        p = tmp_path / "clean.txt"
        atomic_write_text(p, "x")
        leftovers = [x for x in tmp_path.iterdir() if x != p]
        assert leftovers == []

    def test_real_replace_failure_preserves_target_and_cleans_temp(self, tmp_path: Path) -> None:
        target = tmp_path / "existing-target"
        target.mkdir()
        sentinel = target / "sentinel.txt"
        sentinel.write_text("old content")

        # Replacing a non-empty directory with a file fails at the real
        # filesystem boundary after the same-directory temporary file exists.
        with pytest.raises(OSError) as exc_info:
            atomic_write_text(target, "new content")

        assert isinstance(exc_info.value.filename, str)
        assert exc_info.value.filename2 == str(target)
        temporary_path = Path(exc_info.value.filename)
        assert temporary_path.parent == tmp_path
        assert temporary_path.name.startswith(".existing-target.")
        assert temporary_path.name.endswith(".tmp")
        assert not temporary_path.exists()
        assert target.is_dir()
        assert sentinel.read_text() == "old content"
        assert list(tmp_path.iterdir()) == [target]


class TestAtomicWriteBytes:
    def test_writes_raw_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "out.bin"
        atomic_write_bytes(p, b"\x00\x01\x02\xff")
        assert p.read_bytes() == b"\x00\x01\x02\xff"


class TestAtomicWriteJson:
    def test_writes_json(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        atomic_write_json(p, {"a": 1, "b": [2, 3]})
        assert json.loads(p.read_text()) == {"a": 1, "b": [2, 3]}

    def test_indent_and_sort(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        atomic_write_json(p, {"b": 2, "a": 1}, indent=2, sort_keys=True)
        text = p.read_text()
        assert text.index('"a"') < text.index('"b"')
        assert "  " in text  # indent=2

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        atomic_write_json(p, {"name": "Žižkov ¥"})
        reloaded = json.loads(p.read_text())
        assert reloaded["name"] == "Žižkov ¥"


# ── FileLock ─────────────────────────────────────────────────────────


class TestFileLockSingleProcess:
    def test_acquire_and_release(self, tmp_path: Path) -> None:
        lock = FileLock(tmp_path / "x.lock")
        lock.acquire()
        lock.release()  # no error = success

    def test_context_manager(self, tmp_path: Path) -> None:
        with FileLock(tmp_path / "x.lock") as lk:
            assert lk is not None

    def test_double_release_is_noop(self, tmp_path: Path) -> None:
        lock = FileLock(tmp_path / "x.lock")
        lock.acquire()
        lock.release()
        lock.release()  # second release does nothing

    def test_lock_file_persists(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "x.lock"
        with FileLock(lock_path):
            assert lock_path.exists()
        # Not deleted intentionally — delete races with other processes.
        assert lock_path.exists()


def _hold_lock(
    path_str: str,
    duration: float,
    ready_event: _Event,
    started_event: _Event,
) -> None:
    with FileLock(path_str):
        started_event.set()
        ready_event.wait(timeout=5)
        time.sleep(duration)


class TestFileLockConcurrency:
    def test_blocking_waits_for_release(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "blocking.lock"
        started = threading.Event()
        release = threading.Event()

        def _holder() -> None:
            with FileLock(lock_path):
                started.set()
                release.wait(timeout=2)

        t = threading.Thread(target=_holder, daemon=True)
        t.start()
        assert started.wait(timeout=2)

        # Second acquirer should NOT immediately succeed. Use non-blocking
        # to prove contention without blocking the test thread.
        lock2 = FileLock(lock_path, blocking=False)
        with pytest.raises(BlockingIOError):
            lock2.acquire()

        release.set()
        t.join(timeout=2)

        # After release, new acquirer succeeds.
        lock3 = FileLock(lock_path, blocking=False)
        lock3.acquire()
        lock3.release()

    def test_non_blocking_raises_immediately(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "nb.lock"
        holder = FileLock(lock_path)
        holder.acquire()
        try:
            with pytest.raises(BlockingIOError):
                FileLock(lock_path, blocking=False).acquire()
        finally:
            holder.release()

    def test_timeout_raises(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "to.lock"
        holder = FileLock(lock_path)
        holder.acquire()
        try:
            t0 = time.monotonic()
            with pytest.raises(TimeoutError, match="timed out"):
                FileLock(lock_path, timeout=0.2).acquire()
            elapsed = time.monotonic() - t0
            # At least the timeout, but never more than 1 s of slack.
            assert 0.15 < elapsed < 1.5
        finally:
            holder.release()

    def test_timeout_path_succeeds_after_wait(self, tmp_path: Path) -> None:
        """Polling loop exits successfully when the holder releases."""
        lock_path = tmp_path / "wait.lock"
        holder = FileLock(lock_path)
        holder.acquire()

        released = threading.Event()

        def _release_after_delay() -> None:
            time.sleep(0.15)
            holder.release()
            released.set()

        t = threading.Thread(target=_release_after_delay, daemon=True)
        t.start()

        # timeout=2 s so the polling path has room to retry-and-succeed.
        waiter = FileLock(lock_path, timeout=2.0)
        waiter.acquire()
        try:
            assert released.wait(timeout=1)
        finally:
            waiter.release()
            t.join(timeout=1)

    def test_shared_allows_multiple_readers(self, tmp_path: Path) -> None:
        if os.name == "nt":
            pytest.skip("fcntl not available on Windows")
        lock_path = tmp_path / "shared.lock"
        a = FileLock(lock_path, exclusive=False)
        b = FileLock(lock_path, exclusive=False)
        a.acquire()
        b.acquire()  # both shared locks coexist
        a.release()
        b.release()


@pytest.mark.skipif(os.name == "nt", reason="fcntl not available on Windows")
class TestFileLockCrossProcess:
    def test_blocking_across_processes(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "xprocess.lock"
        ready = mp.Event()
        started = mp.Event()

        p = mp.Process(target=_hold_lock, args=(str(lock_path), 0.5, ready, started))
        p.start()
        try:
            assert started.wait(timeout=3)
            # Child holds the lock; our non-blocking attempt should fail
            with pytest.raises(BlockingIOError):
                FileLock(lock_path, blocking=False).acquire()
            ready.set()
            p.join(timeout=3)
            # After child exits, we can grab it
            ok = FileLock(lock_path, blocking=False)
            ok.acquire()
            ok.release()
        finally:
            if p.is_alive():
                p.terminate()
                p.join()
