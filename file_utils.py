# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Atomic file writes and advisory file locking

"""Atomic writes and advisory locks for Remanentia persistent stores.

Two small primitives, stdlib-only:

- :func:`atomic_write_text`, :func:`atomic_write_bytes`,
  :func:`atomic_write_json` — write to a same-directory tmpfile then
  ``os.replace`` so the target either has the old content or the new
  content, never half of each. A crash mid-write leaves ``file.tmp.PID``
  behind (safe to delete) instead of corrupting ``file``.

- :class:`FileLock` — POSIX advisory ``flock`` context manager. Two
  Remanentia processes writing to the same knowledge store used to
  silently clobber each other; now one blocks (or times out) while the
  other finishes. Windows has no ``fcntl``; there the lock becomes a
  no-op with a one-line stderr warning the first time it's used.

Pre-2026-04-17 the project had no writer-concurrency story at all.
This module is the minimum viable fix; a distributed or multi-host
setup needs a shared lock backend (Redis, etcd, DynamoDB) on top.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import TracebackType

try:
    import fcntl  # POSIX only

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover — Windows
    _HAVE_FCNTL = False


# ─── Atomic writes ────────────────────────────────────────────────────


def fsync_directory(directory: Path) -> None:
    """fsync *directory* so the rename that just landed is itself durable.

    ``os.replace`` makes the swap atomic, and we already fsync the file's
    data before renaming — but the *directory entry* the rename creates
    lives in the directory's own metadata, which is not on stable storage
    until the directory is flushed. A power loss in that window can lose
    the rename, and for a brand-new target the entire entry, even though
    the file contents were safely fsynced. That is exactly the torn-write
    class this module exists to prevent, so we close it here.

    Opening a directory for fsync is POSIX-only and not universally
    permitted (Windows has no directory fds; some filesystems reject the
    fsync). Durability is therefore best-effort: a failure here cannot
    corrupt or half-write the target — the atomic swap has already
    happened — so it is swallowed rather than raised.
    """
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)  # codeql[py/path-injection]
    except OSError:  # pragma: no cover — platform without directory fds (Windows)
        return
    try:
        os.fsync(dir_fd)
    except OSError:  # pragma: no cover — filesystem/platform rejects dir fsync
        pass
    finally:
        os.close(dir_fd)


def _atomic_write_raw(path: Path, data: bytes) -> None:
    """Core write-then-rename. Caller hands us bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # codeql[py/path-injection]
    # NamedTemporaryFile writes in the same directory so os.replace is
    # an atomic rename on the same filesystem. delete=False because we
    # hand the path to os.replace ourselves.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),  # codeql[py/path-injection]
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)  # codeql[py/path-injection]
        fsync_directory(path.parent)
    except Exception:
        # Leave the partial file for post-mortem but do not clobber target.
        try:
            os.unlink(tmp_name)  # codeql[py/path-injection]
        except OSError:  # pragma: no cover — race with a parallel cleanup
            pass
        raise


def atomic_write_text(path: Path | str, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace *path* with *content*."""
    _atomic_write_raw(Path(path), content.encode(encoding))


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Atomically replace *path* with *data*."""
    _atomic_write_raw(Path(path), data)


def atomic_write_json(
    path: Path | str,
    obj: object,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> None:
    """Atomically replace *path* with ``json.dumps(obj)``."""
    encoded = json.dumps(
        obj,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
    ).encode("utf-8")
    _atomic_write_raw(Path(path), encoded)


# ─── Advisory file locking ────────────────────────────────────────────


_WINDOWS_WARNED = False


class FileLock:
    """Advisory POSIX file lock, context-manager style.

    Usage::

        with FileLock("memory/knowledge.lock"):
            knowledge_store.save()

    The lock file is created if missing and never deleted (delete races
    with a parallel locker). Acquisition blocks by default; pass
    ``blocking=False`` to raise :class:`BlockingIOError` on contention,
    or ``timeout=N`` to give up after N seconds.

    On Windows (``fcntl`` unavailable) this is a no-op with a one-time
    stderr warning so operators can switch to a different coordination
    mechanism. Production multi-host deployments need a shared backend
    regardless.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        blocking: bool = True,
        timeout: float | None = None,
        exclusive: bool = True,
    ) -> None:
        self._path = Path(path)
        self._blocking = blocking
        self._timeout = timeout
        self._exclusive = exclusive
        self._fd: int | None = None

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    # ---- public API ---------------------------------------------------

    def acquire(self) -> None:
        global _WINDOWS_WARNED
        if not _HAVE_FCNTL:  # pragma: no cover — Windows
            if not _WINDOWS_WARNED:
                print(
                    "[FileLock] fcntl unavailable (Windows); locking is a NO-OP.",
                    file=sys.stderr,
                    flush=True,
                )
                _WINDOWS_WARNED = True
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self._flock(fd)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        if _HAVE_FCNTL:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
        else:  # pragma: no cover — Windows
            os.close(self._fd)
        self._fd = None

    # ---- internals ----------------------------------------------------

    def _flock(self, fd: int) -> None:
        """Take the lock, honouring blocking / timeout / exclusive flags."""
        mode = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH
        if self._blocking and self._timeout is None:
            fcntl.flock(fd, mode)
            return

        # Non-blocking or timeout path: poll with short sleeps.
        import time

        mode_nb = mode | fcntl.LOCK_NB
        if not self._blocking:
            fcntl.flock(fd, mode_nb)  # raises BlockingIOError if contested
            return

        deadline = time.monotonic() + (self._timeout or 0)
        while True:
            try:
                fcntl.flock(fd, mode_nb)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"FileLock timed out after {self._timeout} s on {self._path}"
                    ) from None
                time.sleep(0.05)
