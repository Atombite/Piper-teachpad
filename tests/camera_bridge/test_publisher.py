# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from threading import Thread
from uuid import UUID

import zmq

from piper_control.camera_bridge.config import CameraName, StreamName
from piper_control.camera_bridge.frames import CameraFrameSet, EncodedFrame
from piper_control.camera_bridge.publisher import CameraPublisher, CameraStatus


SESSION = UUID("12345678-1234-5678-1234-567812345678")


class RecordingSocket:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.options: list[tuple[int, object]] = []
        self.bound: list[str] = []
        self.sent: list[tuple[list[bytes], int]] = []
        self.closed: list[int] = []
        self.fail_first = fail_first

    def setsockopt(self, option: int, value: object) -> None:
        self.options.append((option, value))

    def bind(self, endpoint: str) -> None:
        self.bound.append(endpoint)

    def send_multipart(self, parts: list[bytes], flags: int = 0) -> None:
        if self.fail_first:
            self.fail_first = False
            raise zmq.Again()
        self.sent.append((parts, flags))

    def close(self, linger: int = -1) -> None:
        self.closed.append(linger)


def frameset() -> CameraFrameSet:
    color = EncodedFrame(
        stream=StreamName.COLOR,
        payload=b"jpeg",
        width=640,
        height=480,
        encoding="jpeg",
        pixel_format="bgr8",
    )
    depth = EncodedFrame(
        stream=StreamName.DEPTH,
        payload=b"\x01\x00",
        width=1,
        height=1,
        encoding="raw",
        pixel_format="z16",
        dtype="uint16",
        byte_order="little",
        stride_bytes=2,
        depth_scale=0.001,
    )
    return CameraFrameSet(
        camera=CameraName.RIGHT_WRIST,
        frameset_sequence=5,
        host_monotonic_ns=10,
        host_unix_ns=20,
        device_timestamp_ms=1.5,
        frames=(color, depth),
    )


def test_publisher_sets_low_latency_options_and_sends_nonblocking_frames() -> None:
    socket = RecordingSocket()
    publisher = CameraPublisher(
        "tcp://127.0.0.1:15555", socket_factory=lambda: socket
    )
    publisher.open()

    sent = publisher.publish_frameset(SESSION, frameset())

    assert (zmq.SNDHWM, 14) in socket.options
    assert (zmq.LINGER, 0) in socket.options
    assert socket.bound == ["tcp://127.0.0.1:15555"]
    assert sent == 2
    assert [entry[0][0] for entry in socket.sent] == [
        b"camera/right_wrist/color",
        b"camera/right_wrist/depth",
    ]
    assert all(flags == zmq.NOBLOCK for _parts, flags in socket.sent)
    assert [json.loads(entry[0][1])["sequence"] for entry in socket.sent] == [
        0,
        0,
    ]


def test_eagain_drops_message_without_faulting_and_sequence_keeps_advancing() -> None:
    socket = RecordingSocket(fail_first=True)
    publisher = CameraPublisher("tcp://127.0.0.1:15555", socket_factory=lambda: socket)
    publisher.open()

    assert publisher.publish_frameset(SESSION, frameset()) == 1
    assert publisher.dropped_messages == 1
    assert publisher.publish_frameset(SESSION, frameset()) == 2
    color_sequences = [
        json.loads(parts[1])["sequence"]
        for parts, _flags in socket.sent
        if parts[0].endswith(b"color")
    ]
    assert color_sequences == [1]


def test_status_is_two_part_json_and_reset_restarts_topic_sequences() -> None:
    socket = RecordingSocket()
    publisher = CameraPublisher("tcp://127.0.0.1:15555", socket_factory=lambda: socket)
    publisher.open()
    publisher.publish_frameset(SESSION, frameset())
    publisher.publish_frameset(SESSION, frameset())
    publisher.reset_sequences()
    publisher.publish_frameset(SESSION, frameset())

    status = CameraStatus(
        session_id=SESSION,
        camera=CameraName.HEAD,
        state="online",
        host_monotonic_ns=30,
        host_unix_ns=40,
        identity="HEAD_SERIAL_EXAMPLE",
    )
    assert publisher.publish_status(status) is True

    assert [
        json.loads(socket.sent[index][0][1])["frameset_sequence"]
        for index in (0, 2, 4)
    ] == [0, 1, 0]
    assert json.loads(socket.sent[4][0][1])["sequence"] == 0
    status_parts, flags = socket.sent[-1]
    assert status_parts[0] == b"camera/status/head"
    assert json.loads(status_parts[1])["state"] == "online"
    assert len(status_parts) == 2
    assert flags == zmq.NOBLOCK


def test_only_first_publishing_thread_may_use_socket_and_close_is_idempotent() -> None:
    socket = RecordingSocket()
    publisher = CameraPublisher("tcp://127.0.0.1:15555", socket_factory=lambda: socket)
    publisher.open()
    publisher.publish_frameset(SESSION, frameset())
    errors: list[Exception] = []

    def publish_from_other_thread() -> None:
        try:
            publisher.publish_frameset(SESSION, frameset())
        except Exception as exc:
            errors.append(exc)

    thread = Thread(target=publish_from_other_thread)
    thread.start()
    thread.join()
    publisher.close()
    publisher.close()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "owner thread" in str(errors[0])
    assert socket.closed == [0]
