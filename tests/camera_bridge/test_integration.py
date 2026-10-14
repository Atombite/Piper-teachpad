# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from threading import Event, Lock, Thread, enumerate as enumerate_threads
import time

import zmq

from piper_control.camera_bridge.config import CameraName, StreamName
from piper_control.camera_bridge.frames import CameraFrameSet, EncodedFrame
from piper_control.camera_bridge.publisher import CameraPublisher
from piper_control.camera_bridge.service import CameraBridgeService
from piper_control.camera_bridge.sources.base import (
    DeviceDisconnected,
    FatalFrameError,
)

from camera_bridge.test_service import service_config, wait_until


STREAMS = {
    CameraName.LEFT_WRIST: (StreamName.COLOR,),
    CameraName.RIGHT_WRIST: (StreamName.COLOR, StreamName.DEPTH),
    CameraName.HEAD: (
        StreamName.COLOR,
        StreamName.DEPTH,
        StreamName.INFRARED_1,
        StreamName.INFRARED_2,
    ),
}


def make_frameset(camera: CameraName, sequence: int) -> CameraFrameSet:
    frames: list[EncodedFrame] = []
    for stream in STREAMS[camera]:
        if stream is StreamName.COLOR:
            frames.append(
                EncodedFrame(
                    stream=stream,
                    payload=f"jpeg-{sequence}".encode(),
                    width=640,
                    height=480,
                    encoding="jpeg",
                    pixel_format="bgr8",
                )
            )
        elif stream is StreamName.DEPTH:
            frames.append(
                EncodedFrame(
                    stream=stream,
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
            )
        else:
            frames.append(
                EncodedFrame(
                    stream=stream,
                    payload=b"\x01",
                    width=1,
                    height=1,
                    encoding="raw",
                    pixel_format="y8",
                    dtype="uint8",
                    stride_bytes=1,
                )
            )
    now = time.monotonic_ns()
    return CameraFrameSet(
        camera=camera,
        frameset_sequence=sequence,
        host_monotonic_ns=now,
        host_unix_ns=time.time_ns(),
        device_timestamp_ms=float(sequence),
        frames=tuple(frames),
    )


class ControllableSource:
    def __init__(self, camera: CameraName) -> None:
        self.camera = camera
        self.identity = f"fake-{camera.value}"
        self.disconnect_requested = Event()
        self.fatal_requested = Event()
        self._did_disconnect = False
        self._sequence = 0
        self._lock = Lock()

    def open(self, _timeout_ms: int) -> CameraFrameSet:
        with self._lock:
            self._sequence += 1
            return make_frameset(self.camera, self._sequence)

    def read(self, _timeout_ms: int) -> CameraFrameSet:
        time.sleep(0.0005)
        with self._lock:
            if self.fatal_requested.is_set():
                raise FatalFrameError("injected terminal profile error")
            if self.disconnect_requested.is_set() and not self._did_disconnect:
                self._did_disconnect = True
                raise DeviceDisconnected("injected USB disconnect")
            self._sequence += 1
            return make_frameset(self.camera, self._sequence)

    def read_startup(self, timeout_ms: int) -> CameraFrameSet:
        return self.read(timeout_ms)

    def close(self) -> None:
        return None


class SlowRecordingSocket:
    def __init__(self) -> None:
        self.sent: list[list[bytes]] = []
        self._lock = Lock()

    def setsockopt(self, _option: int, _value: object) -> None:
        return None

    def bind(self, _endpoint: str) -> None:
        return None

    def send_multipart(self, parts: list[bytes], flags: int = 0) -> None:
        assert flags == zmq.NOBLOCK
        time.sleep(0.001)
        with self._lock:
            self.sent.append(parts)

    def close(self, linger: int = -1) -> None:
        assert linger == 0

    def snapshot(self) -> list[list[bytes]]:
        with self._lock:
            return list(self.sent)


def test_fake_end_to_end_topics_latest_frames_reconnect_and_terminal_exit() -> None:
    camera_sources = {camera: ControllableSource(camera) for camera in CameraName}
    socket = SlowRecordingSocket()
    publisher = CameraPublisher(
        "tcp://127.0.0.1:15555", socket_factory=lambda: socket
    )
    service = CameraBridgeService(service_config(), camera_sources, publisher)
    shutdown = Event()
    result: list[int] = []
    thread = Thread(target=lambda: result.append(service.run(shutdown)))
    thread.start()

    expected_topics = {
        f"camera/{camera.value}/{stream.value}".encode()
        for camera, streams in STREAMS.items()
        for stream in streams
    }
    wait_until(
        lambda: expected_topics
        <= {parts[0] for parts in socket.snapshot() if len(parts) == 3}
    )
    first_messages = [parts for parts in socket.snapshot() if len(parts) == 3]
    first_session = json.loads(first_messages[0][1])["session_id"]
    assert any(len(parts) == 2 for parts in socket.snapshot())

    camera_sources[CameraName.RIGHT_WRIST].disconnect_requested.set()
    wait_until(
        lambda: any(
            len(parts) == 3
            and json.loads(parts[1])["session_id"] != first_session
            for parts in socket.snapshot()
        )
    )

    def left_wrist_has_overwritten_frames() -> bool:
        left_capture_sequences = [
            int(parts[2].decode().removeprefix("jpeg-"))
            for parts in socket.snapshot()
            if len(parts) == 3 and parts[0] == b"camera/left_wrist/color"
        ]
        return any(
            later - earlier > 1
            for earlier, later in zip(
                left_capture_sequences, left_capture_sequences[1:]
            )
        )

    wait_until(left_wrist_has_overwritten_frames)
    assert left_wrist_has_overwritten_frames()

    camera_sources[CameraName.HEAD].fatal_requested.set()
    thread.join(1.0)

    assert result == [1]
    assert not thread.is_alive()
    assert service.worker_threads_alive == 0
    assert service.faults[-1].camera is CameraName.HEAD
    assert not [
        thread
        for thread in enumerate_threads()
        if thread.name.startswith("camera-") and thread.is_alive()
    ]
