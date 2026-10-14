#!/usr/bin/env python3
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Small asynchronous client for the Piper control WebSocket protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import itertools
import json
from typing import Any, Literal

import websockets


class PiperControlError(RuntimeError):
    """A structured error returned by the Piper control service."""

    def __init__(self, code: str, message: str, command: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.command = command


class ProtocolMismatchError(RuntimeError):
    """The peer returned a response that cannot match the sent request."""


class PiperControlClient:
    """One-connection, one-request-at-a-time Piper control client."""

    def __init__(
        self, url: str, *, identity: Literal["local", "bridge"] = "local"
    ) -> None:
        self.url = url
        self.identity = identity
        self._websocket: Any | None = None
        self._ids = itertools.count(1)
        self._request_lock = asyncio.Lock()

    async def connect(self) -> "PiperControlClient":
        if self._websocket is None:
            self._websocket = await websockets.connect(self.url)
            try:
                hello = await self._request("hello", identity=self.identity)
                if not isinstance(hello, dict) or hello.get("identity") != self.identity:
                    raise ProtocolMismatchError(
                        "hello response identity does not match requested identity"
                    )
            except BaseException:
                await self.close()
                raise
        return self

    async def close(self) -> None:
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            await websocket.close()

    async def __aenter__(self) -> "PiperControlClient":
        return await self.connect()

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def _request(self, command: str, **fields: object) -> Any:
        async with self._request_lock:
            if self._websocket is None:
                raise RuntimeError("client is not connected")
            request_id = str(next(self._ids))
            await self._websocket.send(
                json.dumps({"id": request_id, "cmd": command, **fields})
            )
            raw_response = await self._websocket.recv(decode=True)
            try:
                response = json.loads(raw_response)
            except (TypeError, ValueError):
                raise ProtocolMismatchError("response must be valid JSON") from None
            if not isinstance(response, dict):
                raise ProtocolMismatchError("response must be a JSON object")
            if response.get("id") != request_id:
                raise ProtocolMismatchError("response id does not match request id")
            if response.get("cmd") != command:
                raise ProtocolMismatchError(
                    "response command does not match request command"
                )
            if response.get("ok") is not True:
                error = response.get("error")
                if not isinstance(error, dict):
                    raise ProtocolMismatchError(
                        "error response has no structured error"
                    )
                code = error.get("code")
                message = error.get("message")
                if not isinstance(code, str) or not isinstance(message, str):
                    raise ProtocolMismatchError(
                        "error response fields must be strings"
                    )
                raise PiperControlError(code, message, command)
            if "data" not in response:
                raise ProtocolMismatchError("successful response has no data")
            return response["data"]

    async def get_state(self) -> dict[str, Any]:
        return await self._request("get_state")

    async def get_base_info(self) -> dict[str, Any]:
        return await self._request("get_base_info")

    async def servo_j(
        self,
        target_joints: Sequence[float],
        speed: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, object] = {"target_joints": list(target_joints)}
        if speed is not None:
            fields["speed"] = list(speed)
        return await self._request("servoJ", **fields)

    async def move_gripper(
        self,
        target_position: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, object] = {"target_position": target_position}
        if speed is not None:
            fields["speed"] = speed
        return await self._request("move_gripper", **fields)
