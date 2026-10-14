# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Nonblocking ZeroMQ publisher for camera frames and status."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import json
from threading import get_ident
from typing import Any, Callable, Literal
from uuid import UUID

import zmq

from piper_control.camera_bridge.config import CameraName
from piper_control.camera_bridge.frames import CameraFrameSet
from piper_control.camera_bridge.protocol import NETWORK_HWM, encode_frame_message


SocketFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class CameraStatus:
    session_id: UUID
    camera: CameraName
    state: Literal["online", "reconnecting", "fault"]
    host_monotonic_ns: int
    host_unix_ns: int
    identity: str
    error_category: str | None = None
    error: str | None = None


class CameraPublisher:
    """Own a PUB socket and drop output instead of blocking capture."""

    def __init__(
        self,
        endpoint: str,
        *,
        socket_factory: SocketFactory | None = None,
    ) -> None:
        self.endpoint = endpoint
        self._socket_factory = socket_factory or self._default_socket_factory
        self._socket: Any | None = None
        self._owner_thread_id: int | None = None
        self._topic_sequences: defaultdict[bytes, int] = defaultdict(int)
        self._frameset_sequences: defaultdict[CameraName, int] = defaultdict(int)
        self.dropped_messages = 0

    @staticmethod
    def _default_socket_factory() -> Any:
        return zmq.Context.instance().socket(zmq.PUB)

    def open(self) -> None:
        if self._socket is not None:
            return
        socket = self._socket_factory()
        socket.setsockopt(zmq.SNDHWM, NETWORK_HWM)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)
        self._socket = socket

    def publish_frameset(self, session_id: UUID, frameset: CameraFrameSet) -> int:
        self._claim_owner()
        logical_sequence = self._frameset_sequences[frameset.camera]
        self._frameset_sequences[frameset.camera] += 1
        frameset = replace(frameset, frameset_sequence=logical_sequence)
        sent = 0
        for frame in frameset.frames:
            topic = f"camera/{frameset.camera.value}/{frame.stream.value}".encode(
                "ascii"
            )
            sequence = self._topic_sequences[topic]
            self._topic_sequences[topic] += 1
            if self._send(
                list(encode_frame_message(session_id, sequence, frameset, frame))
            ):
                sent += 1
        return sent

    def publish_status(self, status: CameraStatus) -> bool:
        self._claim_owner()
        data = asdict(status)
        data["protocol_version"] = 1
        data["session_id"] = str(status.session_id)
        data["camera"] = status.camera.value
        body = json.dumps(
            {key: value for key, value in data.items() if value is not None},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return self._send(
            [f"camera/status/{status.camera.value}".encode("ascii"), body]
        )

    def reset_sequences(self) -> None:
        self._topic_sequences.clear()
        self._frameset_sequences.clear()

    def close(self) -> None:
        socket, self._socket = self._socket, None
        self._owner_thread_id = None
        self._topic_sequences.clear()
        self._frameset_sequences.clear()
        if socket is not None:
            socket.close(linger=0)

    def _claim_owner(self) -> None:
        if self._socket is None:
            raise RuntimeError("camera publisher is not open")
        current = get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = current
        elif self._owner_thread_id != current:
            raise RuntimeError("camera publisher socket may only be used by owner thread")

    def _send(self, parts: list[bytes]) -> bool:
        assert self._socket is not None
        try:
            self._socket.send_multipart(parts, flags=zmq.NOBLOCK)
        except zmq.Again:
            self.dropped_messages += 1
            return False
        return True
