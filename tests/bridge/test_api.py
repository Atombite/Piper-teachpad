# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

import pytest

from piper_control.bridge.api import create_bridge_app
from piper_control.bridge.arbitration import EnqueueResult
from piper_control.protocol import ProtocolError


class FakeService:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, object]] = []
        self.releases: list[tuple[str, int]] = []
        self.disconnects: list[str] = []
        self.reject: ProtocolError | None = None

    async def submit(self, connection_id: str, request: object) -> EnqueueResult:
        if self.reject is not None:
            raise self.reject
        self.submissions.append((connection_id, request))
        return EnqueueResult(accepted=True, queued=True)

    async def release(self, connection_id: str, priority: int) -> None:
        self.releases.append((connection_id, priority))

    async def disconnect(self, connection_id: str) -> None:
        self.disconnects.append(connection_id)

    async def state(self) -> dict:
        await asyncio.sleep(0)
        return {"side": "left", "bridge": {"upstream_available": True}}

    async def base_info(self) -> dict:
        await asyncio.sleep(0)
        return {"protocol_version": "3.0", "side": "left"}


async def open_socket(app):
    incoming: asyncio.Queue = asyncio.Queue()
    outgoing: asyncio.Queue = asyncio.Queue()

    async def receive():
        return await incoming.get()

    async def send(message):
        await outgoing.put(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "scheme": "ws",
        "path": "/ws/control",
        "raw_path": b"/ws/control",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8766),
        "subprotocols": [],
        "state": {},
    }
    task = asyncio.create_task(app(scope, receive, send))
    await incoming.put({"type": "websocket.connect"})
    assert (await outgoing.get())["type"] == "websocket.accept"
    return task, incoming, outgoing


async def request(incoming, outgoing, payload: object) -> dict:
    await incoming.put({"type": "websocket.receive", "text": json.dumps(payload)})
    return json.loads((await outgoing.get())["text"])


@pytest.mark.asyncio
async def test_multiple_clients_route_priority_release_queries_and_disconnect() -> None:
    service = FakeService()
    app = create_bridge_app(service)
    first_task, first_in, first_out = await open_socket(app)
    second_task, second_in, second_out = await open_socket(app)
    try:
        first_response, second_response = await asyncio.gather(
            request(
                first_in,
                first_out,
                {"id": "a", "cmd": "servoJ", "target_joints": [1] * 6, "priority": 7},
            ),
            request(
                second_in,
                second_out,
                {"id": "b", "cmd": "move_gripper", "target_position": 0.2, "priority": 99},
            ),
        )
        assert first_response["data"] == {"accepted": True, "queued": True, "priority": 7}
        assert second_response["data"] == {"accepted": True, "queued": True, "priority": 99}
        first_id, second_id = [item[0] for item in service.submissions]
        assert first_id != second_id

        released = await request(
            first_in, first_out, {"id": "r", "cmd": "release", "priority": 7}
        )
        assert released["data"] == {"released": True, "priority": 7}
        assert service.releases == [(first_id, 7)]

        state, base = await asyncio.gather(
            request(first_in, first_out, {"id": "s", "cmd": "get_state"}),
            request(second_in, second_out, {"id": "i", "cmd": "get_base_info"}),
        )
        assert state["data"]["bridge"]["upstream_available"] is True
        assert base["data"]["protocol_version"] == "3.0"
    finally:
        await first_in.put({"type": "websocket.disconnect", "code": 1000})
        await second_in.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.gather(first_task, second_task)
    assert set(service.disconnects) == {first_id, second_id}


@pytest.mark.asyncio
async def test_errors_are_structured_and_do_not_close_other_clients() -> None:
    service = FakeService()
    app = create_bridge_app(service)
    task, incoming, outgoing = await open_socket(app)
    try:
        malformed = await request(incoming, outgoing, {"id": "x", "cmd": "servoJ"})
        assert malformed["error"]["code"] == "INVALID_REQUEST"

        service.reject = ProtocolError("UPSTREAM_UNAVAILABLE", "controller is local")
        rejected = await request(
            incoming,
            outgoing,
            {"id": "y", "cmd": "move_gripper", "target_position": 0.1},
        )
        assert rejected == {
            "id": "y",
            "cmd": "move_gripper",
            "ok": False,
            "error": {"code": "UPSTREAM_UNAVAILABLE", "message": "controller is local"},
        }
        service.reject = None
        assert (await request(incoming, outgoing, {"id": "z", "cmd": "get_state"}))["ok"] is True
    finally:
        await incoming.put({"type": "websocket.disconnect", "code": 1000})
        await task
