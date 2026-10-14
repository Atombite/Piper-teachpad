# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe storage for a single replaceable value."""

from __future__ import annotations

from threading import Lock
from typing import Generic, TypeVar


T = TypeVar("T")


class LatestValue(Generic[T]):
    """A one-slot mailbox where a new value replaces the previous value."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._value: T | None = None

    def put(self, value: T) -> None:
        with self._lock:
            self._value = value

    def take(self) -> T | None:
        with self._lock:
            value = self._value
            self._value = None
            return value
