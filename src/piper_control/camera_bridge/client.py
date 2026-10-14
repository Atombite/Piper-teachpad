# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Reference latest-frame ZeroMQ subscriber."""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

import zmq

from piper_control.camera_bridge.protocol import (
    CameraProtocolError,
    NETWORK_HWM,
    ReceivedFrame,
    decode_frame_message,
)


class CameraSubscriber:
    """Subscribe to selected topics and drain queued messages to their newest value."""

    def __init__(
        self,
        endpoint: str,
        *,
        topics: Iterable[str] = ("camera/",),
        socket_factory: Callable[[], Any] | None = None,
        poller_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.topics = tuple(topics)
        self._socket_factory = socket_factory or self._default_socket_factory
        self._poller_factory = poller_factory or zmq.Poller
        self._socket: Any | None = None
        self._poller: Any | None = None
        self.invalid_messages = 0
        self.statuses: dict[str, dict[str, object]] = {}
        self._delivered: dict[str, ReceivedFrame] = {}
        self._retired_sessions: dict[str, set[object]] = {}

    @staticmethod
    def _default_socket_factory() -> Any:
        return zmq.Context.instance().socket(zmq.SUB)

    def open(self) -> None:
        if self._socket is not None:
            return
        socket = self._socket_factory()
        socket.setsockopt(zmq.RCVHWM, NETWORK_HWM)
        socket.setsockopt(zmq.LINGER, 0)
        for topic in self.topics:
            socket.setsockopt(zmq.SUBSCRIBE, topic.encode("ascii"))
        socket.connect(self.endpoint)
        poller = self._poller_factory()
        poller.register(socket, zmq.POLLIN)
        self._socket = socket
        self._poller = poller

    def recv_latest(self, timeout_ms: int) -> dict[str, ReceivedFrame]:
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if self._socket is None or self._poller is None:
            raise RuntimeError("camera subscriber is not open")
        latest: dict[str, ReceivedFrame] = {}
        timeout = timeout_ms
        while self._poller.poll(timeout):
            timeout = 0
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) == 2 and parts[0].startswith(b"camera/status/"):
                self._remember_status(parts)
                continue
            try:
                frame = decode_frame_message(parts)
            except CameraProtocolError:
                self.invalid_messages += 1
                continue
            retired = self._retired_sessions.setdefault(frame.topic, set())
            if frame.session_id in retired:
                continue
            current = latest.get(frame.topic, self._delivered.get(frame.topic))
            if current is None or _is_newer(frame, current):
                if current is not None and frame.session_id != current.session_id:
                    retired.add(current.session_id)
                latest[frame.topic] = frame
        self._delivered.update(latest)
        return latest

    def close(self) -> None:
        socket, self._socket = self._socket, None
        poller, self._poller = self._poller, None
        self._delivered.clear()
        self._retired_sessions.clear()
        if socket is None:
            return
        if poller is not None:
            poller.unregister(socket)
        socket.close(linger=0)

    def _remember_status(self, parts: list[bytes]) -> None:
        try:
            topic = parts[0].decode("ascii")
            body = json.loads(parts[1].decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self.invalid_messages += 1
            return
        self.statuses[topic] = body


def _is_newer(candidate: ReceivedFrame, current: ReceivedFrame) -> bool:
    if candidate.session_id != current.session_id:
        return candidate.host_monotonic_ns >= current.host_monotonic_ns
    return candidate.sequence > current.sequence
