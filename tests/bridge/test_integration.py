# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import websockets
import yaml

from examples.ws_client import PiperControlClient
from test_config import valid_config


ROOT = Path(__file__).parents[2]


def free_ports(count: int) -> tuple[int, ...]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return tuple(item.getsockname()[1] for item in sockets)
    finally:
        for item in sockets:
            item.close()


def request_json(url: str, body: dict[str, str] | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urlopen(request, timeout=0.5) as response:
        return json.load(response)


class BridgeClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.socket = None
        self.ids = itertools.count(1)

    async def connect(self) -> BridgeClient:
        self.socket = await websockets.connect(self.url)
        return self

    async def close(self) -> None:
        if self.socket is not None:
            await self.socket.close()
            self.socket = None

    async def request(self, command: str, **fields: object) -> dict:
        assert self.socket is not None
        request_id = str(next(self.ids))
        await self.socket.send(
            json.dumps({"id": request_id, "cmd": command, **fields})
        )
        response = json.loads(await self.socket.recv(decode=True))
        assert response["id"] == request_id
        assert response["cmd"] == command
        assert response["ok"] is True, response
        return response["data"]

    async def state(self) -> dict:
        return await self.request("get_state")

    async def servo(self, target: float, priority: int) -> dict:
        return await self.request(
            "servoJ",
            target_joints=[target, 0.0, 0.0, 0.0, 0.0, 0.0],
            speed=[10.0] * 6,
            priority=priority,
        )

    async def gripper(self, target: float, priority: int) -> dict:
        return await self.request(
            "move_gripper",
            target_position=target,
            speed=0.05,
            priority=priority,
        )

    async def release(self, priority: int) -> dict:
        return await self.request("release", priority=priority)


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=2)


def wait_controller(process: subprocess.Popen[str], http_url: str) -> None:
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"controller exited\n{stdout}\n{stderr}")
        try:
            if request_json(f"{http_url}/api/state")["state"] == "RUNNING":
                return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(0.02)
    raise AssertionError("controller did not become RUNNING")


def switch_remote(http_url: str) -> None:
    assert request_json(f"{http_url}/change_state", {"state": "hold"})["ok"]
    response = request_json(f"{http_url}/control_mode", {"mode": "remote"})
    assert response["data"] == {
        "accepted": True,
        "mode": "remote",
        "state": "RUNNING",
    }


@contextmanager
def live_pairs(tmp_path: Path, sides: tuple[str, ...]) -> Iterator[dict[str, dict]]:
    ports = free_ports(len(sides) * 2)
    control_ports = dict(zip(sides, ports[: len(sides)], strict=True))
    bridge_ports = dict(zip(sides, ports[len(sides) :], strict=True))
    control = valid_config()
    for side, arm in control["arms"].items():
        arm["server"] = {"host": "127.0.0.1", "port": control_ports.get(side, free_ports(1)[0])}
        arm["control_hz"] = 100
        arm["shutdown_home_timeout_s"] = 0.5
    control_path = tmp_path / "control.yaml"
    control_path.write_text(yaml.safe_dump(control), encoding="utf-8")
    bridge = {
        "bridges": {
            side: {
                "listen": {"host": "127.0.0.1", "port": bridge_ports[side]},
                "upstream_url": f"ws://127.0.0.1:{control_ports[side]}/ws/control",
                "control_hz": 100,
                "command_ttl_s": 1.0,
                "owner_timeout_s": 3.0,
                "reconnect_initial_s": 0.05,
                "reconnect_max_s": 0.2,
            }
            for side in sides
        }
    }
    bridge_path = tmp_path / "bridge.yaml"
    bridge_path.write_text(yaml.safe_dump(bridge), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    result: dict[str, dict] = {}
    processes: list[subprocess.Popen[str]] = []
    try:
        for side in sides:
            controller = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "piper_control.main",
                    "--config",
                    str(control_path),
                    "--simulate",
                    "--side",
                    side,
                    "--lock-path",
                    str(tmp_path / f"controller-{side}.lock"),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            processes.append(controller)
            http_url = f"http://127.0.0.1:{control_ports[side]}"
            wait_controller(controller, http_url)
            switch_remote(http_url)
            bridge_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "piper_control.bridge.main",
                    "--config",
                    str(bridge_path),
                    "--side",
                    side,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            processes.append(bridge_process)
            result[side] = {
                "controller": controller,
                "bridge": bridge_process,
                "http": http_url,
                "controller_ws": f"ws://127.0.0.1:{control_ports[side]}/ws/control",
                "ws": f"ws://127.0.0.1:{bridge_ports[side]}/ws/control",
            }
        yield result
    finally:
        for process in reversed(processes):
            stop_process(process)


async def connect_bridge(url: str) -> BridgeClient:
    deadline = time.monotonic() + 6
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client = BridgeClient(url)
        try:
            await client.connect()
            state = await client.state()
            if state["bridge"]["upstream_available"]:
                return client
            await client.close()
        except Exception as exc:
            last_error = exc
            await client.close()
        await asyncio.sleep(0.02)
    raise AssertionError(f"Bridge did not become available: {last_error}")


async def wait_state(client: BridgeClient, predicate, description: str) -> dict:
    deadline = time.monotonic() + 4
    last = None
    while time.monotonic() < deadline:
        last = await client.state()
        if predicate(last):
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"did not reach {description}; last={last}")


@pytest.mark.asyncio
async def test_live_bridge_preemption_expiry_release_and_shared_ownership(tmp_path: Path) -> None:
    with live_pairs(tmp_path, ("left",)) as pairs:
        a = await connect_bridge(pairs["left"]["ws"])
        b = await connect_bridge(pairs["left"]["ws"])
        try:
            base_info = await a.request("get_base_info")
            assert base_info["arm"]["gripper"]["backend"] == "piper"
            assert base_info["units"]["gripper_position"] == "meter"
            assert base_info["units"]["gripper_speed"] == "meter/second"
            initial = await a.state()
            assert initial["bridge"]["owner"] is None
            assert initial["TargetQ"] == pytest.approx(initial["ActualQ"])
            assert initial["TargetGripper"] == pytest.approx(initial["ActualGripper"])

            await a.servo(10.0, priority=0)
            moving = await wait_state(a, lambda state: state["ActualQ"][0] > 0.1, "A motion")
            await b.gripper(0.05, priority=100)
            await wait_state(
                b,
                lambda state: state["bridge"]["owner"] is not None
                and state["bridge"]["owner"]["priority"] == 100,
                "B ownership",
            )
            joint_at_preemption = (await b.state())["ActualQ"][0]
            await a.servo(-10.0, priority=0)
            await asyncio.sleep(0.25)
            held = await b.state()
            assert held["ActualQ"][0] <= joint_at_preemption + 0.35
            assert held["ActualGripper"] > initial["ActualGripper"]

            await asyncio.sleep(0.8)
            await a.servo(15.0, priority=0)
            before_release = (await a.state())["ActualQ"][0]
            await b.release(100)
            samples = []
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                state = await a.state()
                samples.append(state["ActualQ"][0])
                if state["ActualQ"][0] > before_release + 0.2:
                    break
                await asyncio.sleep(0.02)
            assert samples[-1] > before_release + 0.2
            assert min(samples) >= before_release - 0.15
            assert moving["side"] == "left"
        finally:
            await asyncio.gather(a.close(), b.close())


@pytest.mark.asyncio
async def test_bridge_client_reacquires_priority_ten_after_owner_expiry(
    tmp_path: Path,
) -> None:
    with live_pairs(tmp_path, ("left",)) as pairs:
        client = await connect_bridge(pairs["left"]["ws"])
        try:
            assert (await client.state())["side"] == "left"
            await client.request(
                "servoJ", target_joints=[2.0, 0, 0, 0, 0, 0],
                speed=[2.0] * 6, priority=10,
            )
            await client.request(
                "move_gripper", target_position=0.02, speed=0.01, priority=10,
            )
            await asyncio.sleep(3.1)
            assert (await client.state())["bridge"]["owner"] is None
            await client.request(
                "servoJ", target_joints=[3.0, 0, 0, 0, 0, 0],
                speed=[2.0] * 6, priority=10,
            )
            state = await client.state()
            assert state["side"] == "left"
            assert state["bridge"]["owner"]["priority"] == 10
            await client.release(10)
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_two_bridge_pairs_stay_isolated_when_one_upstream_disconnects(tmp_path: Path) -> None:
    with live_pairs(tmp_path, ("left", "right")) as pairs:
        left = await connect_bridge(pairs["left"]["ws"])
        right = await connect_bridge(pairs["right"]["ws"])
        try:
            assert (await left.state())["bridge"]["owner"] is None
            assert (await right.state())["bridge"]["owner"] is None
            await left.servo(4.0, priority=10)
            await right.servo(8.0, priority=20)
            await wait_state(left, lambda state: state["ActualQ"][0] > 0.1, "left motion")
            right_before = await wait_state(
                right, lambda state: state["ActualQ"][0] > 0.1, "right motion"
            )

            stop_process(pairs["left"]["controller"])
            await wait_state(
                left,
                lambda state: state["bridge"]["upstream_available"] is False,
                "left upstream outage",
            )
            right_owner = (await right.state())["bridge"]["owner"]
            assert right_owner is not None and right_owner["priority"] == 20
            await right.servo(12.0, priority=20)
            right_after = await wait_state(
                right,
                lambda state: state["ActualQ"][0] > right_before["ActualQ"][0] + 0.2,
                "right motion after left outage",
            )
            assert right_after["side"] == "right"
            assert right_after["bridge"]["upstream_available"] is True
            assert (await left.state())["side"] == "left"
        finally:
            await asyncio.gather(left.close(), right.close())


@pytest.mark.asyncio
async def test_running_bridge_does_not_block_remote_local_mode_round_trip(tmp_path: Path) -> None:
    with live_pairs(tmp_path, ("left",)) as pairs:
        details = pairs["left"]
        client = await connect_bridge(details["ws"])
        try:
            assert request_json(
                f"{details['http']}/change_state", {"state": "hold"}
            )["ok"]
            local = request_json(
                f"{details['http']}/control_mode", {"mode": "local"}
            )
            assert local["data"] == {
                "accepted": True,
                "mode": "local",
                "state": "RUNNING",
            }
            local_client = PiperControlClient(
                details["controller_ws"], identity="local"
            )
            await local_client.connect()
            assert (await local_client.get_state())["control_mode"] == "local"
            await local_client.close()
            await wait_state(
                client,
                lambda state: state["bridge"]["upstream_available"] is False,
                "Bridge unavailable in local mode",
            )

            assert request_json(
                f"{details['http']}/change_state", {"state": "hold"}
            )["ok"]
            remote = request_json(
                f"{details['http']}/control_mode", {"mode": "remote"}
            )
            assert remote["data"]["state"] == "RUNNING"
            recovered = await wait_state(
                client,
                lambda state: state["bridge"]["upstream_available"] is True,
                "Bridge reconnect after remote mode",
            )
            assert recovered["control_mode"] == "remote"
            assert recovered["bridge"]["owner"] is None
            gripper_before = recovered["ActualGripper"]
            target = 0.06 if gripper_before < 0.06 else 0.01
            await client.gripper(target, priority=10)
            await wait_state(
                client,
                lambda state: abs(state["ActualGripper"] - gripper_before) > 0.001,
                "native Piper gripper motion after reconnect",
            )
        finally:
            await client.close()
