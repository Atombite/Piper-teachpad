# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
import json
from uuid import UUID

import pytest
import zmq

from piper_control.camera_bridge.client import CameraSubscriber
from piper_control.camera_bridge.config import CameraName, StreamName
from piper_control.camera_bridge.frames import CameraFrameSet, EncodedFrame
from piper_control.camera_bridge.protocol import encode_frame_message


SESSION = UUID("12345678-1234-5678-1234-567812345678")


def message(sequence: int, payload: bytes) -> tuple[bytes, bytes, bytes]:
    frame = EncodedFrame(
        stream=StreamName.COLOR,
        payload=payload,
        width=640,
        height=480,
        encoding="jpeg",
        pixel_format="bgr8",
    )
    frameset = CameraFrameSet(
        camera=CameraName.LEFT_WRIST,
        frameset_sequence=sequence,
        host_monotonic_ns=sequence,
        host_unix_ns=sequence,
        device_timestamp_ms=None,
        frames=(frame,),
    )
    return encode_frame_message(SESSION, sequence, frameset, frame)


class ReceiveSocket:
    def __init__(self, messages: list[list[bytes]]) -> None:
        self.messages = deque(messages)
        self.options: list[tuple[int, object]] = []
        self.connected: list[str] = []
        self.closed: list[int] = []

    def setsockopt(self, option: int, value: object) -> None:
        self.options.append((option, value))

    def connect(self, endpoint: str) -> None:
        self.connected.append(endpoint)

    def recv_multipart(self, flags: int = 0) -> list[bytes]:
        if not self.messages:
            raise zmq.Again()
        return self.messages.popleft()

    def close(self, linger: int = -1) -> None:
        self.closed.append(linger)


class ReceivePoller:
    def __init__(self, socket: ReceiveSocket) -> None:
        self.socket = socket
        self.registered: list[tuple[object, int]] = []

    def register(self, socket: object, flags: int) -> None:
        self.registered.append((socket, flags))

    def unregister(self, _socket: object) -> None:
        return None

    def poll(self, _timeout_ms: int) -> list[tuple[object, int]]:
        return [(self.socket, zmq.POLLIN)] if self.socket.messages else []


def make_client(messages: list[list[bytes]]) -> tuple[CameraSubscriber, ReceiveSocket]:
    socket = ReceiveSocket(messages)
    poller = ReceivePoller(socket)
    client = CameraSubscriber(
        "tcp://127.0.0.1:15555",
        topics=("camera/left_wrist/color",),
        socket_factory=lambda: socket,
        poller_factory=lambda: poller,
    )
    return client, socket


def test_client_sets_hwm_subscriptions_and_drains_to_newest_per_topic() -> None:
    client, socket = make_client(
        [list(message(1, b"old")), list(message(2, b"new"))]
    )
    client.open()

    latest = client.recv_latest(timeout_ms=50)

    assert (zmq.RCVHWM, 14) in socket.options
    assert (zmq.LINGER, 0) in socket.options
    assert (zmq.SUBSCRIBE, b"camera/left_wrist/color") in socket.options
    assert socket.connected == ["tcp://127.0.0.1:15555"]
    assert latest["camera/left_wrist/color"].sequence == 2
    assert latest["camera/left_wrist/color"].payload == b"new"


def test_client_rejects_malformed_frame_without_returning_corrupt_data() -> None:
    malformed = list(message(1, b"valid"))
    malformed[2] = b"wrong-size"
    client, _socket = make_client([malformed])
    client.open()

    assert client.recv_latest(timeout_ms=0) == {}
    assert client.invalid_messages == 1


def test_client_timeout_and_close_are_bounded_and_idempotent() -> None:
    client, socket = make_client([])
    client.open()
    assert client.recv_latest(timeout_ms=0) == {}
    with pytest.raises(ValueError, match="non-negative"):
        client.recv_latest(timeout_ms=-1)
    client.close()
    client.close()
    assert socket.closed == [0]


def test_client_rejects_older_and_retired_session_frames_across_reads() -> None:
    client, socket = make_client([list(message(10, b"newest"))])
    client.open()

    assert client.recv_latest(timeout_ms=0)["camera/left_wrist/color"].sequence == 10
    socket.messages.append(list(message(9, b"older")))
    assert client.recv_latest(timeout_ms=0) == {}

    next_session = UUID("22345678-1234-5678-1234-567812345678")
    next_parts = list(message(0, b"next-session"))
    metadata = json.loads(next_parts[1])
    metadata["session_id"] = str(next_session)
    metadata["host_monotonic_ns"] = 20
    next_parts[1] = json.dumps(metadata).encode()
    socket.messages.append(next_parts)
    assert client.recv_latest(timeout_ms=0)["camera/left_wrist/color"].session_id == next_session

    delayed_old = list(message(11, b"delayed-old-session"))
    delayed_metadata = json.loads(delayed_old[1])
    delayed_metadata["host_monotonic_ns"] = 30
    delayed_old[1] = json.dumps(delayed_metadata).encode()
    socket.messages.append(delayed_old)
    assert client.recv_latest(timeout_ms=0) == {}
