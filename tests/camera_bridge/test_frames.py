# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError
from threading import Event, Thread
import time

import pytest

from piper_control.camera_bridge.config import CameraName, StreamName
from piper_control.camera_bridge.frames import (
    CameraFrameSet,
    EncodedFrame,
    LatestFrameQueue,
)


def color_frame(payload: bytes = b"jpeg") -> EncodedFrame:
    return EncodedFrame(
        stream=StreamName.COLOR,
        payload=payload,
        width=640,
        height=480,
        encoding="jpeg",
        pixel_format="bgr8",
        device_timestamp_ms=12.5,
    )


def frameset(sequence: int) -> CameraFrameSet:
    return CameraFrameSet(
        camera=CameraName.LEFT_WRIST,
        frameset_sequence=sequence,
        host_monotonic_ns=sequence * 100,
        host_unix_ns=sequence * 1000,
        device_timestamp_ms=float(sequence),
        frames=(color_frame(str(sequence).encode()),),
    )


def test_frame_records_are_immutable_and_reject_invalid_shape() -> None:
    frame = color_frame()
    with pytest.raises(FrozenInstanceError):
        frame.width = 320
    with pytest.raises(ValueError, match="positive"):
        EncodedFrame(
            stream=StreamName.COLOR,
            payload=b"x",
            width=0,
            height=480,
            encoding="jpeg",
            pixel_format="bgr8",
        )
    with pytest.raises(ValueError, match="payload"):
        EncodedFrame(
            stream=StreamName.COLOR,
            payload=b"",
            width=640,
            height=480,
            encoding="jpeg",
            pixel_format="bgr8",
        )


def test_frameset_rejects_duplicate_or_empty_streams() -> None:
    base = {
        "camera": CameraName.HEAD,
        "frameset_sequence": 1,
        "host_monotonic_ns": 1,
        "host_unix_ns": 2,
        "device_timestamp_ms": 3.0,
    }
    with pytest.raises(ValueError, match="at least one"):
        CameraFrameSet(**base, frames=())
    with pytest.raises(ValueError, match="duplicate"):
        CameraFrameSet(**base, frames=(color_frame(), color_frame()))


def test_latest_queue_keeps_exactly_two_and_pops_only_newest() -> None:
    queue = LatestFrameQueue[CameraFrameSet]()
    queue.put(frameset(1))
    queue.put(frameset(2))
    assert len(queue) == 2
    queue.put(frameset(3))
    assert len(queue) == 2

    latest = queue.pop_latest(timeout_s=0)

    assert latest is not None
    assert latest.frameset_sequence == 3
    assert len(queue) == 0


def test_independent_queues_do_not_share_storage() -> None:
    left = LatestFrameQueue[CameraFrameSet]()
    right = LatestFrameQueue[CameraFrameSet]()
    left.put(frameset(1))
    assert len(left) == 1
    assert len(right) == 0


def test_clear_removes_pending_frames() -> None:
    queue = LatestFrameQueue[CameraFrameSet]()
    queue.put(frameset(1))
    queue.clear()
    assert queue.pop_latest(timeout_s=0) is None


def test_close_wakes_waiter_and_rejects_new_frames() -> None:
    queue = LatestFrameQueue[CameraFrameSet]()
    entered = Event()
    result: list[CameraFrameSet | None] = []

    def wait_for_frame() -> None:
        entered.set()
        result.append(queue.pop_latest(timeout_s=5.0))

    thread = Thread(target=wait_for_frame)
    thread.start()
    assert entered.wait(0.2)
    time.sleep(0.01)
    queue.close()
    thread.join(0.5)

    assert not thread.is_alive()
    assert result == [None]
    with pytest.raises(RuntimeError, match="closed"):
        queue.put(frameset(2))
