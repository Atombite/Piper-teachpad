# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Process lifecycle helpers with no hardware work in signal context."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import os
from pathlib import Path
import signal
from threading import Event
from types import FrameType
from typing import Callable, Protocol


@dataclass(frozen=True, slots=True)
class ShutdownResult:
    homed: bool
    timed_out: bool = False
    faulted: bool = False
    errors: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return (
            0
            if self.homed and not self.timed_out and not self.faulted and not self.errors
            else 1
        )


class _GracefulController(Protocol):
    def shutdown_gracefully(
        self, *, wait: Callable[[float], object] | None = None
    ) -> ShutdownResult: ...


def shutdown_gracefully(
    controller: _GracefulController,
    *,
    wait: Callable[[float], object] | None = None,
) -> ShutdownResult:
    """Run the controller's safe shutdown sequence."""

    return controller.shutdown_gracefully(wait=wait)


def install_signal_handlers(shutdown_event: Event) -> dict[int, object]:
    """Install handlers that do nothing except request orderly shutdown."""

    def request_shutdown(_number: int, _frame: FrameType | None) -> None:
        shutdown_event.set()

    return {
        number: signal.signal(number, request_shutdown)
        for number in (signal.SIGINT, signal.SIGTERM)
    }


class SingletonLock:
    """Advisory nonblocking single-process lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = None

    def acquire(self) -> None:
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+", encoding="ascii")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError("piper-control is already running") from exc
            raise
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        self._file = lock_file

    def release(self) -> None:
        if self._file is None:
            return
        lock_file = self._file
        self._file = None
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    def __enter__(self) -> "SingletonLock":
        self.acquire()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.release()
