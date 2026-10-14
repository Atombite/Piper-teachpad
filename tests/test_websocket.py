# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import replace
from threading import Event, Thread
import time

import pytest
import httpx
import uvicorn
import websockets

from piper_control.api import create_app
from piper_control.config import ArmConfig, ArmSide
from piper_control.controller import StateConflict
from piper_control.types import ArmStateSnapshot, ControlMode, ServiceState

from test_api import arm_config


class WsController:
    def __init__(
        self,
        side: ArmSide = ArmSide.LEFT,
        mode: ControlMode = ControlMode.LOCAL,
    ) -> None:
        self._snapshot = replace(
            ArmStateSnapshot(side=side),
            control_mode=mode,
            state=ServiceState.RUNNING,
            arm=replace(
                ArmStateSnapshot(side=side).arm,
                connected=True,
                enabled=True,
                feedback_fresh=True,
            ),
        )
        self.servo_commands = []
        self.gripper_commands = []
        self.disconnect_calls = 0
        self.disconnect_entered = Event()
        self.allow_disconnect = Event()
        self.allow_disconnect.set()

    def snapshot(self) -> ArmStateSnapshot:
        return self._snapshot

    def submit_servo(self, command) -> None:
        self.servo_commands.append(command)

    def submit_gripper(self, command) -> None:
        self.gripper_commands.append(command)

    def client_disconnected(self) -> None:
        self.disconnect_entered.set()
        self.allow_disconnect.wait(timeout=2)
        self.disconnect_calls += 1

    def request_hold(self) -> None:
        self._snapshot = replace(
            self._snapshot,
            state=ServiceState.HOLD,
            moving=False,
            hold=True,
        )

    def change_control_mode(self, mode: ControlMode) -> None:
        if self._snapshot.state is not ServiceState.HOLD:
            raise StateConflict("control mode change requires HOLD state")
        self._snapshot = replace(
            self._snapshot,
            control_mode=mode,
            state=ServiceState.RUNNING,
            moving=False,
            hold=True,
        )


class DisconnectAbort(BaseException):
    pass


class AbortingDisconnectController(WsController):
    def __init__(self) -> None:
        super().__init__()
        self.failure = DisconnectAbort("disconnect aborted")

    def client_disconnected(self) -> None:
        self.disconnect_calls += 1
        if self.disconnect_calls == 1:
            raise self.failure


def _free_ports(count: int = 1) -> tuple[int, ...]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return tuple(sock.getsockname()[1] for sock in sockets)
    finally:
        for sock in sockets:
            sock.close()


class LiveServer:
    def __init__(
        self,
        controller: WsController,
        side: ArmSide,
        config: ArmConfig,
        *,
        port: int | None = None,
    ) -> None:
        self.port = _free_ports()[0] if port is None else port
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(controller, side, config),
                host="127.0.0.1",
                port=self.port,
                log_level="error",
            )
        )
        self.thread = Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "LiveServer":
        self.thread.start()
        deadline = time.monotonic() + 3
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert self.server.started
        return self

    def __exit__(self, *_args) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=3)
        assert not self.thread.is_alive()


async def _start_asgi_websocket(app):
    incoming = asyncio.Queue()
    outgoing = asyncio.Queue()

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
        "server": ("127.0.0.1", 8765),
        "subprotocols": [],
        "state": {},
    }
    task = asyncio.create_task(app(scope, receive, send))
    await incoming.put({"type": "websocket.connect"})
    assert (await outgoing.get())["type"] == "websocket.accept"
    return task, incoming, outgoing


async def _hello_asgi(incoming, outgoing, identity: str = "local") -> None:
    await incoming.put(
        {
            "type": "websocket.receive",
            "text": json.dumps(
                {"id": "hello", "cmd": "hello", "identity": identity}
            ),
        }
    )
    response = json.loads((await outgoing.get())["text"])
    assert response["ok"] is True
    assert response["data"] == {"identity": identity}


async def _hello_socket(ws, identity: str = "local") -> None:
    await ws.send(json.dumps({"id": "hello", "cmd": "hello", "identity": identity}))
    response = json.loads(await ws.recv(decode=True))
    assert response["ok"] is True
    assert response["data"] == {"identity": identity}


async def _wait_until(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_websocket_requires_matching_identity_before_commands() -> None:
    controller = WsController()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/ws/control"
        ) as ws:
            await ws.send('{"id":"h1","cmd":"hello","identity":"local"}')
            hello = json.loads(await ws.recv(decode=True))
            assert hello == {
                "id": "h1",
                "cmd": "hello",
                "ok": True,
                "data": {"identity": "local"},
            }
            await ws.send('{"id":"s1","cmd":"get_state"}')
            state = json.loads(await ws.recv(decode=True))
            assert state["ok"] is True
            assert state["data"]["ws_controller_identity"] == "local"


@pytest.mark.asyncio
async def test_websocket_rejects_command_before_hello_and_closes() -> None:
    controller = WsController()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/ws/control"
        ) as ws:
            await ws.send('{"id":"s1","cmd":"get_state"}')
            response = json.loads(await ws.recv(decode=True))
            assert response["error"]["code"] == "HANDSHAKE_REQUIRED"
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                await ws.recv(decode=True)
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 4401


@pytest.mark.asyncio
async def test_websocket_rejects_identity_that_does_not_match_mode() -> None:
    controller = WsController(mode=ControlMode.REMOTE)
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/ws/control"
        ) as ws:
            await ws.send('{"id":"h1","cmd":"hello","identity":"local"}')
            response = json.loads(await ws.recv(decode=True))
            assert response["error"]["code"] == "IDENTITY_MISMATCH"
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                await ws.recv(decode=True)
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 4403


@pytest.mark.asyncio
async def test_unverified_bridge_socket_does_not_occupy_local_lease() -> None:
    controller = WsController(mode=ControlMode.LOCAL)
    app = create_app(controller, ArmSide.LEFT, arm_config())
    bridge, bridge_incoming, bridge_outgoing = await _start_asgi_websocket(app)
    local, local_incoming, local_outgoing = await _start_asgi_websocket(app)

    await _hello_asgi(local_incoming, local_outgoing, "local")
    await bridge_incoming.put(
        {
            "type": "websocket.receive",
            "text": json.dumps(
                {"id": "bridge", "cmd": "hello", "identity": "bridge"}
            ),
        }
    )
    rejected = json.loads((await bridge_outgoing.get())["text"])
    assert rejected["error"]["code"] == "IDENTITY_MISMATCH"
    assert (await bridge_outgoing.get())["code"] == 4403
    await bridge

    await local_incoming.put({"type": "websocket.disconnect", "code": 1000})
    await local


@pytest.mark.asyncio
async def test_hold_rejects_new_control_socket_without_occupying_lease() -> None:
    controller = WsController(mode=ControlMode.REMOTE)
    controller.request_hold()
    app = create_app(controller, ArmSide.LEFT, arm_config())
    task, _incoming, outgoing = await _start_asgi_websocket(app)

    response = json.loads((await asyncio.wait_for(outgoing.get(), 0.1))["text"])
    assert response["error"]["code"] == "STATE_CONFLICT"
    assert (await outgoing.get())["type"] == "websocket.close"
    await task
    assert controller.disconnect_calls == 0


@pytest.mark.asyncio
async def test_stale_feedback_rejects_new_control_socket_without_occupying_lease() -> None:
    controller = WsController()
    controller._snapshot = replace(
        controller._snapshot,
        arm=replace(controller._snapshot.arm, feedback_fresh=False),
    )
    app = create_app(controller, ArmSide.LEFT, arm_config())
    task, _incoming, outgoing = await _start_asgi_websocket(app)

    response = json.loads((await asyncio.wait_for(outgoing.get(), 0.1))["text"])
    assert response["error"]["code"] == "STATE_CONFLICT"
    assert response["error"]["message"] == "Robot feedback is stale."
    assert (await outgoing.get())["type"] == "websocket.close"
    await task
    assert controller.disconnect_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [b'{"cmd":"hello","identity":"local"}', "{"],
)
async def test_websocket_rejects_invalid_first_frame_and_closes(frame: object) -> None:
    controller = WsController()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/ws/control"
        ) as ws:
            await ws.send(frame)
            response = json.loads(await ws.recv(decode=True))
            assert response["error"]["code"] == "INVALID_REQUEST"
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                await ws.recv(decode=True)
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 4400


@pytest.mark.asyncio
async def test_second_hello_is_rejected_without_replacing_identity() -> None:
    controller = WsController()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/ws/control"
        ) as ws:
            await _hello_socket(ws)
            await ws.send('{"id":"h2","cmd":"hello","identity":"bridge"}')
            response = json.loads(await ws.recv(decode=True))
            assert response["error"]["code"] == "HANDSHAKE_ALREADY_COMPLETED"
            await ws.send('{"id":"s1","cmd":"get_state"}')
            state = json.loads(await ws.recv(decode=True))
            assert state["data"]["ws_controller_identity"] == "local"


@pytest.mark.asyncio
async def test_handshake_timeout_returns_stable_error_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("piper_control.api.HANDSHAKE_TIMEOUT_S", 0.01)
    controller = WsController()
    app = create_app(controller, ArmSide.LEFT, arm_config())
    task, _incoming, outgoing = await _start_asgi_websocket(app)

    response = json.loads((await outgoing.get())["text"])
    assert response["error"]["code"] == "HANDSHAKE_TIMEOUT"
    close = await outgoing.get()
    assert close["code"] == 4408
    await task

    next_task, incoming, next_outgoing = await _start_asgi_websocket(app)
    await _hello_asgi(incoming, next_outgoing)
    await incoming.put({"type": "websocket.disconnect", "code": 1000})
    await next_task


@pytest.mark.asyncio
async def test_hold_waits_for_inflight_command_then_closes_old_identity() -> None:
    class BlockingController(WsController):
        def __init__(self) -> None:
            super().__init__()
            self.command_entered = Event()
            self.release_command = Event()

        def submit_servo(self, command) -> None:
            self.command_entered.set()
            assert self.release_command.wait(2.0)
            super().submit_servo(command)

    controller = BlockingController()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        uri = f"ws://127.0.0.1:{server.port}/ws/control"
        ws = await websockets.connect(uri)
        await _hello_socket(ws)
        await ws.send(
            '{"id":"move","cmd":"servoJ","target_joints":[0,1,-1,2,3,4]}'
        )
        assert await asyncio.to_thread(controller.command_entered.wait, 1.0)

        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}"
        ) as client:
            hold_task = asyncio.create_task(
                client.post("/change_state", json={"state": "hold"})
            )
            await asyncio.sleep(0.02)
            assert not hold_task.done()
            controller.release_command.set()
            assert json.loads(await ws.recv(decode=True))["ok"] is True
            hold_response = await asyncio.wait_for(hold_task, 2.0)
            assert hold_response.status_code == 200
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await ws.recv(decode=True)

            state = (await client.get("/api/state")).json()
            assert state["state"] == "HOLD"
            assert state["ws_controller_connected"] is False
            assert state["ws_controller_identity"] is None

            mode = await client.post("/control_mode", json={"mode": "remote"})
            assert mode.status_code == 200
            assert mode.json()["data"] == {
                "accepted": True,
                "mode": "remote",
                "state": "RUNNING",
            }
        await ws.close()


@pytest.mark.asyncio
async def test_mode_change_excludes_new_old_identity_connection() -> None:
    class BlockingModeController(WsController):
        def __init__(self) -> None:
            super().__init__()
            self.request_hold()
            self.mode_entered = Event()
            self.release_mode = Event()

        def change_control_mode(self, mode: ControlMode) -> None:
            self.mode_entered.set()
            assert self.release_mode.wait(2.0)
            super().change_control_mode(mode)

    controller = BlockingModeController()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        uri = f"ws://127.0.0.1:{server.port}/ws/control"
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}"
        ) as client:
            mode_task = asyncio.create_task(
                client.post("/control_mode", json={"mode": "remote"})
            )
            assert await asyncio.to_thread(controller.mode_entered.wait, 1.0)

            async def connect() -> object:
                return await websockets.connect(uri)

            connect_task = asyncio.create_task(connect())
            await asyncio.sleep(0.02)
            ws = connect_task.result() if connect_task.done() else None
            receive_task = None
            if ws is not None:
                await ws.send('{"id":"h1","cmd":"hello","identity":"local"}')
                receive_task = asyncio.create_task(ws.recv(decode=True))
                await asyncio.sleep(0.02)
                assert not receive_task.done()

            controller.release_mode.set()
            response = await asyncio.wait_for(mode_task, 2.0)
            assert response.status_code == 200
            if ws is None:
                ws = await asyncio.wait_for(connect_task, 2.0)
            try:
                if receive_task is None:
                    await ws.send(
                        '{"id":"h1","cmd":"hello","identity":"local"}'
                    )
                    receive_task = asyncio.create_task(ws.recv(decode=True))
                rejected = json.loads(await receive_task)
                assert rejected["error"]["code"] == "IDENTITY_MISMATCH"
            finally:
                await ws.close()


@pytest.mark.asyncio
async def test_websocket_accepts_commands_echoes_ids_and_returns_stable_errors() -> None:
    controller = WsController()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/ws/control"
        ) as ws:
            await _hello_socket(ws)
            await ws.send('{"id":7,"cmd":"get_state"}')
            state = await ws.recv(decode=True)
            assert '"id":7' in state
            assert '"ok":true' in state
            assert '"ActualQ"' in state

            await ws.send(
                '{"id":"s1","cmd":"servoJ","target_joints":'
                + str([0] * 6).replace(" ", "")
                + "}"
            )
            assert '"accepted":true' in await ws.recv(decode=True)
            assert len(controller.servo_commands) == 1

            await ws.send('{"id":"bad","cmd":"recover"}')
            error = await ws.recv(decode=True)
            assert '"code":"UNKNOWN_COMMAND"' in error
            assert "Traceback" not in error

            await ws.send(
                '{"cmd":"servoJ","target_joints":'
                + str([999] + [0] * 5).replace(" ", "")
                + "}"
            )
            assert '"code":"OUT_OF_RANGE"' in await ws.recv(decode=True)

            await ws.send('{"cmd":"homing"}')
            assert '"code":"UNKNOWN_COMMAND"' in await ws.recv(decode=True)


@pytest.mark.asyncio
async def test_websocket_dispatches_scalar_gripper_command_end_to_end() -> None:
    controller = WsController()
    app = create_app(controller, ArmSide.LEFT, arm_config())
    task, incoming, outgoing = await _start_asgi_websocket(app)
    await _hello_asgi(incoming, outgoing)

    await incoming.put(
        {
            "type": "websocket.receive",
            "text": (
                '{"id":"grip","cmd":"move_gripper",'
                '"target_position":0.03,"speed":0.02}'
            ),
        }
    )

    response = json.loads((await outgoing.get())["text"])
    assert response == {
        "id": "grip",
        "cmd": "move_gripper",
        "ok": True,
        "data": {"accepted": True},
    }
    assert len(controller.gripper_commands) == 1
    assert controller.gripper_commands[0].target == 0.03
    assert controller.gripper_commands[0].speed == 0.02

    await incoming.put({"type": "websocket.disconnect", "code": 1000})
    await task


@pytest.mark.asyncio
async def test_app_instances_have_independent_commands_and_client_leases() -> None:
    left = WsController(ArmSide.LEFT)
    right = WsController(ArmSide.RIGHT)
    left_port, right_port = _free_ports(2)
    with (
        LiveServer(
            left,
            ArmSide.LEFT,
            arm_config(ArmSide.LEFT),
            port=left_port,
        ) as left_server,
        LiveServer(
            right,
            ArmSide.RIGHT,
            arm_config(ArmSide.RIGHT, port=8766),
            port=right_port,
        ) as right_server,
    ):
        left_ws = await websockets.connect(
            f"ws://127.0.0.1:{left_server.port}/ws/control"
        )
        right_ws = await websockets.connect(
            f"ws://127.0.0.1:{right_server.port}/ws/control"
        )
        try:
            await _hello_socket(left_ws)
            await _hello_socket(right_ws)
            await left_ws.send(
                '{"id":"left","cmd":"servoJ",'
                '"target_joints":[0,1,-1,2,3,4]}'
            )
            await right_ws.send(
                '{"id":"right","cmd":"servoJ",'
                '"target_joints":[1,2,-2,3,4,5]}'
            )
            assert json.loads(await left_ws.recv(decode=True))["ok"] is True
            assert json.loads(await right_ws.recv(decode=True))["ok"] is True
            assert left.servo_commands[0].target == (0, 1, -1, 2, 3, 4)
            assert right.servo_commands[0].target == (1, 2, -2, 3, 4, 5)

            conflicting_left = await websockets.connect(
                f"ws://127.0.0.1:{left_server.port}/ws/control"
            )
            await conflicting_left.send(
                '{"id":"hello","cmd":"hello","identity":"local"}'
            )
            conflict = json.loads(await conflicting_left.recv(decode=True))
            assert conflict["error"]["code"] == "CLIENT_CONFLICT"
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await conflicting_left.recv()

            await right_ws.send('{"id":"still-right","cmd":"get_state"}')
            right_state = json.loads(await right_ws.recv(decode=True))
            assert right_state["id"] == "still-right"
            assert right_state["ok"] is True

            await left_ws.close()
            await _wait_until(lambda: left.disconnect_calls == 1)
            assert right.disconnect_calls == 0

            await right_ws.send('{"id":"right-again","cmd":"get_state"}')
            right_again = json.loads(await right_ws.recv(decode=True))
            assert right_again["id"] == "right-again"
            assert right_again["ok"] is True
        finally:
            await left_ws.close()
            await right_ws.close()


@pytest.mark.asyncio
async def test_binary_and_malformed_frames_return_error_without_disconnect() -> None:
    controller = WsController()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/ws/control"
        ) as ws:
            await _hello_socket(ws)
            await ws.send(b'{"cmd":"get_state"}')
            binary_error = json.loads(await ws.recv(decode=True))
            assert binary_error["error"]["code"] == "INVALID_REQUEST"

            await ws.send("{" + "[" * 2000)
            malformed_error = json.loads(await ws.recv(decode=True))
            assert malformed_error["error"]["code"] == "INVALID_REQUEST"

            await ws.send('{"cmd":"get_state"}')
            assert json.loads(await ws.recv(decode=True))["ok"] is True


@pytest.mark.asyncio
async def test_structured_rejection_and_lease_logs_exclude_payload_and_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="piper_control.api")
    app = create_app(WsController(), ArmSide.LEFT, arm_config())
    first, first_incoming, first_outgoing = await _start_asgi_websocket(app)
    await _hello_asgi(first_incoming, first_outgoing)
    await first_incoming.put(
        {
            "type": "websocket.receive",
            "text": '{"cmd":"bogus","secret":"do-not-log"}',
        }
    )
    rejection = json.loads((await first_outgoing.get())["text"])
    assert rejection["error"]["code"] == "UNKNOWN_COMMAND"

    rejected, rejected_incoming, rejected_outgoing = await _start_asgi_websocket(app)
    await rejected_incoming.put(
        {
            "type": "websocket.receive",
            "text": '{"id":"hello","cmd":"hello","identity":"local"}',
        }
    )
    assert json.loads((await rejected_outgoing.get())["text"])["error"]["code"] == "CLIENT_CONFLICT"
    await rejected_outgoing.get()
    await rejected
    await first_incoming.put({"type": "websocket.disconnect", "code": 1000})
    await first

    events = [getattr(record, "event", None) for record in caplog.records]
    assert events.count("ws_lease_acquire") == 1
    assert events.count("ws_lease_conflict") == 1
    assert events.count("ws_lease_release") == 1
    rejected_record = next(
        record for record in caplog.records
        if getattr(record, "event", None) == "command_rejected"
    )
    assert rejected_record.code == "UNKNOWN_COMMAND"
    assert rejected_record.cmd == "bogus"
    assert "do-not-log" not in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.asyncio
async def test_cancelled_handler_keeps_lease_until_disconnect_hold_finishes() -> None:
    controller = WsController()
    controller.allow_disconnect.clear()
    app = create_app(controller, ArmSide.LEFT, arm_config())
    first, first_incoming, first_outgoing = await _start_asgi_websocket(app)
    await _hello_asgi(first_incoming, first_outgoing)

    first.cancel()
    await _wait_until(controller.disconnect_entered.is_set)
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done()

    rejected, rejected_incoming, rejected_outgoing = await _start_asgi_websocket(app)
    await rejected_incoming.put(
        {
            "type": "websocket.receive",
            "text": '{"id":"hello","cmd":"hello","identity":"local"}',
        }
    )
    conflict = json.loads((await rejected_outgoing.get())["text"])
    assert conflict["error"]["code"] == "CLIENT_CONFLICT"
    assert (await rejected_outgoing.get())["code"] == 4409
    await rejected

    controller.allow_disconnect.set()
    with pytest.raises(asyncio.CancelledError):
        await first

    third, third_incoming, third_outgoing = await _start_asgi_websocket(app)
    await _hello_asgi(third_incoming, third_outgoing)
    await third_incoming.put(
        {"type": "websocket.receive", "text": '{"cmd":"get_state"}'}
    )
    assert json.loads((await third_outgoing.get())["text"])["ok"] is True
    await third_incoming.put({"type": "websocket.disconnect", "code": 1000})
    await third


@pytest.mark.asyncio
async def test_disconnect_base_exception_releases_lease_then_propagates_unchanged() -> None:
    controller = AbortingDisconnectController()
    app = create_app(controller, ArmSide.LEFT, arm_config())
    first, first_incoming, first_outgoing = await _start_asgi_websocket(app)
    await _hello_asgi(first_incoming, first_outgoing)

    await first_incoming.put({"type": "websocket.disconnect", "code": 1000})
    with pytest.raises(DisconnectAbort) as caught:
        await first
    assert caught.value is controller.failure

    second, second_incoming, second_outgoing = await _start_asgi_websocket(app)
    await _hello_asgi(second_incoming, second_outgoing)
    await second_incoming.put(
        {"type": "websocket.receive", "text": '{"cmd":"get_state"}'}
    )
    assert json.loads((await second_outgoing.get())["text"])["ok"] is True
    await second_incoming.put({"type": "websocket.disconnect", "code": 1000})
    await second


@pytest.mark.asyncio
async def test_real_uvicorn_enforces_atomic_single_client_lease_until_hold_finishes() -> None:
    controller = WsController()
    controller.allow_disconnect.clear()
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        uri = f"ws://127.0.0.1:{server.port}/ws/control"
        first = await websockets.connect(uri)
        await _hello_socket(first)
        second = await websockets.connect(uri)
        await second.send('{"id":"hello","cmd":"hello","identity":"local"}')
        conflict = await second.recv(decode=True)
        assert "controller already connected" in conflict
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            await second.recv()
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.code == 4409

        await first.close()
        assert await asyncio.to_thread(controller.disconnect_entered.wait, 1)

        while_hold = await websockets.connect(uri)
        await while_hold.send('{"id":"hello","cmd":"hello","identity":"local"}')
        assert "controller already connected" in await while_hold.recv(decode=True)
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed_hold:
            await while_hold.recv()
        assert closed_hold.value.rcvd is not None
        assert closed_hold.value.rcvd.code == 4409

        controller.allow_disconnect.set()
        deadline = time.monotonic() + 2
        while controller.disconnect_calls != 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert controller.disconnect_calls == 1

        third = await websockets.connect(uri)
        await _hello_socket(third)
        await third.send('{"cmd":"get_base_info"}')
        assert '"ok":true' in await third.recv(decode=True)
        await third.close()


@pytest.mark.asyncio
async def test_websocket_state_conflict_and_internal_error_are_sanitized() -> None:
    controller = WsController()

    def conflict(_command) -> None:
        raise StateConflict("servo commands require RUNNING state")

    controller.submit_servo = conflict
    with LiveServer(controller, ArmSide.LEFT, arm_config()) as server:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/ws/control"
        ) as ws:
            await _hello_socket(ws)
            await ws.send(
                '{"cmd":"servoJ","target_joints":'
                + str([0] * 6).replace(" ", "")
                + "}"
            )
            error = await ws.recv(decode=True)
            assert '"code":"STATE_CONFLICT"' in error
            assert "Traceback" not in error

            controller.snapshot = lambda: (_ for _ in ()).throw(
                RuntimeError("sensitive /secret/path")
            )
            await ws.send('{"cmd":"get_state"}')
            internal = await ws.recv(decode=True)
            assert '"code":"INTERNAL_ERROR"' in internal
            assert "sensitive" not in internal
            assert "/secret/path" not in internal
