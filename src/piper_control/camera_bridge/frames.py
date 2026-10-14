# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Immutable camera frame values and a latest-only bounded queue."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from threading import Condition
from typing import Generic, Literal, TypeVar

from piper_control.camera_bridge.config import CameraName, StreamName


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    stream: StreamName
    payload: bytes
    width: int
    height: int
    encoding: Literal["jpeg", "raw"]
    pixel_format: Literal["bgr8", "z16", "y8"]
    device_timestamp_ms: float | None = None
    dtype: Literal["uint8", "uint16"] | None = None
    byte_order: Literal["little"] | None = None
    stride_bytes: int | None = None
    depth_scale: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("frame payload must be non-empty bytes")
        if self.device_timestamp_ms is not None and not math.isfinite(
            self.device_timestamp_ms
        ):
            raise ValueError("device timestamp must be finite")
        if self.stride_bytes is not None and self.stride_bytes <= 0:
            raise ValueError("frame stride must be positive")
        if self.depth_scale is not None and (
            not math.isfinite(self.depth_scale) or self.depth_scale <= 0
        ):
            raise ValueError("depth scale must be positive and finite")


@dataclass(frozen=True, slots=True)
class CameraFrameSet:
    camera: CameraName
    frameset_sequence: int
    host_monotonic_ns: int
    host_unix_ns: int
    device_timestamp_ms: float | None
    frames: tuple[EncodedFrame, ...]

    def __post_init__(self) -> None:
        if self.frameset_sequence < 0:
            raise ValueError("frameset sequence must be non-negative")
        if self.host_monotonic_ns < 0 or self.host_unix_ns < 0:
            raise ValueError("host timestamps must be non-negative")
        if self.device_timestamp_ms is not None and not math.isfinite(
            self.device_timestamp_ms
        ):
            raise ValueError("device timestamp must be finite")
        if not self.frames:
            raise ValueError("frameset must contain at least one frame")
        streams = [frame.stream for frame in self.frames]
        if len(streams) != len(set(streams)):
            raise ValueError("frameset contains duplicate streams")


T = TypeVar("T")


class LatestFrameQueue(Generic[T]):
    """Thread-safe capacity-two queue whose consumer takes only the newest item."""

    def __init__(self) -> None:
        self._items: deque[T] = deque(maxlen=2)
        self._condition = Condition()
        self._closed = False

    def put(self, item: T) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("latest frame queue is closed")
            self._items.append(item)
            self._condition.notify()

    def pop_latest(self, timeout_s: float | None = None) -> T | None:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("queue timeout must be non-negative")
        with self._condition:
            self._condition.wait_for(
                lambda: bool(self._items) or self._closed,
                timeout=timeout_s,
            )
            if not self._items:
                return None
            latest = self._items[-1]
            self._items.clear()
            return latest

    def clear(self) -> None:
        with self._condition:
            self._items.clear()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._items.clear()
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)
