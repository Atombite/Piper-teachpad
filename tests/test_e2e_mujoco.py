# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Callable, Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml

pytest.importorskip("mujoco")

from examples.ws_client import PiperControlClient
from test_config import valid_config, valid_mujoco_config


ROOT = Path(__file__).parents[1]
MODEL = ROOT / "thirdparty" / "mujoco_menagerie" / "agilex_piper" / "piper.xml"
SAFE_HOME_TIMEOUT_S = 30.0
# The longest API ramp is 0.10 m / 0.02 m/s = 5 s. Allow another
# 5 s for physical settling plus 2 s for process and CI scheduling.
GRIPPER_CONVERGENCE_TIMEOUT_S = 12.0


def _free_ports() -> tuple[int, int]:
    """Reserve both ephemeral choices concurrently so they cannot collide."""

    sockets = [socket.socket(), socket.socket()]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return sockets[0].getsockname()[1], sockets[1].getsockname()[1]
    finally:
        for sock in sockets:
            sock.close()


def _request_json(url: str, body: dict[str, str] | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urlopen(request, timeout=0.5) as response:
        return json.load(response)


def _service_command(config: Path, lock: Path) -> list[str]:
    bootstrap = (
        "from unittest.mock import Mock; "
        "import piper_control.main as service; "
        "service.PiperArmDriver = Mock("
        "side_effect=AssertionError('PiperArmDriver constructor touched')); "
        "raise SystemExit(service.main())"
    )
    return [
        sys.executable,
        "-c",
        bootstrap,
        "--config",
        str(config),
        "--mujoco-headless",
        "--lock-path",
        str(lock),
    ]


@contextmanager
def _headless_service(
    tmp_path: Path,
) -> Iterator[tuple[subprocess.Popen[str], dict[str, tuple[str, str]]]]:
    left_port, right_port = _free_ports()
    ports = {"left": left_port, "right": right_port}
    data = valid_config()
    mujoco_config = valid_mujoco_config()
    mujoco_config.update(
        {
            "model_path": str(MODEL.resolve()),
            "viewer": False,
            "realtime": True,
        }
    )
    data["mujoco"] = mujoco_config
    for side, port in ports.items():
        arm = data["arms"][side]
        arm["server"] = {"host": "127.0.0.1", "port": port}
        arm["control_hz"] = 100
        arm["shutdown_home_timeout_s"] = SAFE_HOME_TIMEOUT_S
        arm["min_joints_deg"] = [-150, 0, -154, -100, -69, -120]
        arm["max_joints_deg"] = [150, 179, 0, 100, 69, 120]
        arm["default_joints_speed_deg_s"] = [90] * 6
        arm["max_joints_speed_deg_s"] = [180] * 6

    config = tmp_path / "mujoco-e2e.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    lock = tmp_path / "piper-control.lock"
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    process = subprocess.Popen(
        _service_command(config, lock),
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    urls = {
        side: (
            f"http://127.0.0.1:{port}",
            f"ws://127.0.0.1:{port}/ws/control",
        )
        for side, port in ports.items()
    }
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(f"service exited during startup\n{stdout}\n{stderr}")
        try:
            if all(
                _request_json(f"{http_url}/api/state")["state"] == "RUNNING"
                for http_url, _ws_url in urls.values()
            ):
                break
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(0.03)
    else:
        process.terminate()
        stdout, stderr = process.communicate(timeout=6)
        raise AssertionError(f"service did not become RUNNING\n{stdout}\n{stderr}")
    try:
        yield process, urls
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=2)


async def _wait_for_http(
    http_url: str,
    predicate: Callable[[dict], bool],
    description: str,
    *,
    timeout_s: float = 6.0,
) -> dict:
    deadline = time.monotonic() + timeout_s
    last_state: dict | None = None
    while time.monotonic() < deadline:
        last_state = await asyncio.to_thread(
            _request_json, f"{http_url}/api/state"
        )
        if predicate(last_state):
            return last_state
        await asyncio.sleep(0.03)
    raise AssertionError(f"service did not reach {description}; last={last_state}")


async def _wait_for_independent_targets(
    left: PiperControlClient,
    right: PiperControlClient,
    left_target: list[float],
    right_target: list[float],
) -> tuple[dict, dict]:
    deadline = time.monotonic() + GRIPPER_CONVERGENCE_TIMEOUT_S
    last: tuple[dict, dict] | None = None
    while time.monotonic() < deadline:
        left_state, right_state = await asyncio.gather(
            left.get_state(), right.get_state()
        )
        last = left_state, right_state
        assert left_state["TargetQ"] != pytest.approx(right_target)
        assert right_state["TargetQ"] != pytest.approx(left_target)
        assert left_state["TargetGripper"] != pytest.approx(0.05)
        assert right_state["TargetGripper"] != pytest.approx(0.10)
        if (
            left_state["TargetQ"] == pytest.approx(left_target)
            and right_state["TargetQ"] == pytest.approx(right_target)
            and left_state["TargetGripper"] == pytest.approx(0.10)
            and right_state["TargetGripper"] == pytest.approx(0.05)
            and max(
                abs(actual - target)
                for actual, target in zip(left_state["ActualQ"], left_target)
            )
            < 8.0
            and max(
                abs(actual - target)
                for actual, target in zip(right_state["ActualQ"], right_target)
            )
            < 8.0
            and left_state["ActualGripper"] == pytest.approx(0.10, abs=0.01)
            and right_state["ActualGripper"] == pytest.approx(0.05, abs=0.01)
        ):
            return left_state, right_state
        await asyncio.sleep(0.03)
    raise AssertionError(f"arms did not converge independently; last={last}")


async def _wait_for_right_progress_while_left_holds(
    left_http: str,
    right_http: str,
    left_qpos: list[float],
    right_joint_before: float,
) -> tuple[dict, dict]:
    deadline = time.monotonic() + 6
    last: tuple[dict, dict] | None = None
    while time.monotonic() < deadline:
        left_state, right_state = await asyncio.gather(
            asyncio.to_thread(_request_json, f"{left_http}/api/state"),
            asyncio.to_thread(_request_json, f"{right_http}/api/state"),
        )
        last = left_state, right_state
        assert left_state["hold"] is True
        assert left_state["ActualQ"] == pytest.approx(left_qpos, abs=0.05)
        if right_state["ActualQ"][1] > right_joint_before + 1:
            return left_state, right_state
        await asyncio.sleep(0.03)
    raise AssertionError(f"right did not progress while left held; last={last}")


@pytest.mark.asyncio
async def test_headless_dual_arm_websocket_control_is_isolated_and_stops_cleanly(
    tmp_path: Path,
) -> None:
    with _headless_service(tmp_path) as (process, urls):
        left_http, left_ws = urls["left"]
        right_http, right_ws = urls["right"]
        left = PiperControlClient(left_ws)
        right = PiperControlClient(right_ws)
        await asyncio.gather(left.connect(), right.connect())

        assert (await left.get_base_info())["side"] == "left"
        assert (await right.get_base_info())["side"] == "right"
        left_target = [5, 45, -45, 5, 5, 5]
        right_target = [-5, 55, -55, -5, -5, -5]
        await left.servo_j(left_target)
        await right.servo_j(right_target)
        await left.move_gripper(0.10)
        await right.move_gripper(0.05)
        _left_state, right_state = await _wait_for_independent_targets(
            left, right, left_target, right_target
        )

        await left.close()
        left_held = await _wait_for_http(
            left_http, lambda state: state["hold"] is True, "left HOLD"
        )
        left_stopped_at = left_held["ActualQ"]
        assert (await right.get_state())["state"] == "RUNNING"
        continued_target = [-10, 65, -65, -10, -10, -10]
        right_joint_before = right_state["ActualQ"][1]
        await right.servo_j(continued_target)
        _left_held, right_state = await _wait_for_right_progress_while_left_holds(
            left_http,
            right_http,
            left_stopped_at,
            right_joint_before,
        )

        emergency = _request_json(f"{left_http}/api/emergency_stop")
        assert emergency["ok"] is True
        await _wait_for_http(
            left_http,
            lambda state: state["state"] == "EMERGENCY_STOP",
            "left emergency stop",
        )
        right_before_stop = right_state["ActualQ"][1]
        await right.servo_j([-15, 75, -75, -15, -15, -15])
        right_after_stop = await _wait_for_http(
            right_http,
            lambda state: state["state"] == "RUNNING"
            and state["ActualQ"][1] > right_before_stop + 1,
            "right motion after left emergency stop",
        )
        assert right_after_stop["state"] == "RUNNING"
        left_after_right_motion = _request_json(f"{left_http}/api/state")
        assert left_after_right_motion["ActualQ"] == pytest.approx(
            left_stopped_at, abs=1.0
        )

        recovery = _request_json(
            f"{left_http}/change_state", {"state": "recover"}
        )
        assert recovery["ok"] is True
        recovered = await _wait_for_http(
            left_http,
            lambda state: state["state"] == "RUNNING"
            and state["emergency_stopped"] is False
            and state["fault_reason"] is None
            and state["arm"]["connected"] is True
            and state["arm"]["enabled"] is True,
            "healthy left recovery",
        )
        assert recovered["hold"] is True
        await left.connect()
        assert (await left.get_state())["state"] == "RUNNING"
        homing = _request_json(
            f"{left_http}/change_state", {"state": "homing"}
        )
        assert homing["ok"] is True
        await _wait_for_http(
            left_http,
            lambda state: state["state"] == "RUNNING",
            "left RUNNING after reconnect homing",
            timeout_s=SAFE_HOME_TIMEOUT_S + 2.0,
        )
        left_after_recovery = await left.get_state()
        resumed_target = [0, 30, -30, 0, 0, 0]
        await left.servo_j(resumed_target)
        await left.move_gripper(0.02)
        await _wait_for_http(
            left_http,
            lambda state: state["TargetQ"] == pytest.approx(resumed_target)
            and state["TargetGripper"] == pytest.approx(0.02)
            and state["ActualQ"][1]
            > left_after_recovery["ActualQ"][1] + 1
            and state["ActualGripper"]
            < left_after_recovery["ActualGripper"] - 0.005,
            "left motion after reconnect and recovery",
            timeout_s=8.0,
        )

        process.send_signal(signal.SIGTERM)
        stdout, stderr = await asyncio.to_thread(process.communicate, timeout=40)
        assert process.returncode == 0, stdout + stderr
        assert stderr.count("event=shutdown_result result=success") == 2
        assert "component_failure" not in stderr
        assert "mujoco_world_shutdown_timeout" not in stderr
        assert "PiperArmDriver constructor touched" not in stderr
        await asyncio.gather(left.close(), right.close())
