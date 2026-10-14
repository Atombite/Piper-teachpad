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
from typing import Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml

from examples.ws_client import PiperControlClient
from test_config import valid_config


ROOT = Path(__file__).parents[1]


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
    return [
        sys.executable,
        "-m",
        "piper_control.main",
        "--config",
        str(config),
        "--simulate",
        "--lock-path",
        str(lock),
    ]


@contextmanager
def _simulated_service(
    tmp_path: Path,
) -> Iterator[tuple[subprocess.Popen[str], dict[str, tuple[str, str]], Path, Path]]:
    left_port, right_port = _free_ports()
    ports = {"left": left_port, "right": right_port}
    data = valid_config()
    for side, port in ports.items():
        data["arms"][side]["server"] = {"host": "127.0.0.1", "port": port}
        data["arms"][side]["control_hz"] = 50
        data["arms"][side]["shutdown_home_timeout_s"] = 3.0
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    lock = tmp_path / "piper-control.lock"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
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
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
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
        stdout, stderr = process.communicate(timeout=5)
        raise AssertionError(f"service did not become RUNNING\n{stdout}\n{stderr}")
    try:
        yield process, urls, config, lock
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=8)


async def _wait_for(http_url: str, predicate, description: str) -> dict:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = await asyncio.to_thread(_request_json, f"{http_url}/api/state")
        if predicate(state):
            return state
        await asyncio.sleep(0.03)
    raise AssertionError(f"service did not reach {description}")


@pytest.mark.asyncio
async def test_two_simulated_arms_remain_independent_and_shutdown_safely(tmp_path: Path) -> None:
    with _simulated_service(tmp_path) as (process, urls, config, lock):
        left_http, left_ws = urls["left"]
        right_http, right_ws = urls["right"]
        assert _request_json(f"{left_http}/api/base_info")["side"] == "left"
        assert _request_json(f"{right_http}/api/base_info")["side"] == "right"

        duplicate = subprocess.run(
            _service_command(config, lock),
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
        assert duplicate.returncode != 0
        assert "already running" in duplicate.stderr

        left = PiperControlClient(left_ws)
        right = PiperControlClient(right_ws)
        await asyncio.gather(left.connect(), right.connect())
        await asyncio.gather(
            left.servo_j([1.0, 0, 0, 0, 0, 0]),
            right.servo_j([0, 2.0, 0, 0, 0, 0]),
        )
        await asyncio.gather(
            _wait_for(left_http, lambda state: state["ActualQ"][0] > 0.1, "left motion"),
            _wait_for(right_http, lambda state: state["ActualQ"][1] > 0.1, "right motion"),
        )

        await left.close()
        await _wait_for(left_http, lambda state: state["hold"] is True, "left HOLD")
        assert (await right.get_state())["state"] == "RUNNING"
        await right.servo_j([0, 3.0, 0, 0, 0, 0])
        await _wait_for(right_http, lambda state: state["ActualQ"][1] > 2.1, "continued right motion")

        emergency = _request_json(f"{left_http}/api/emergency_stop")
        assert emergency["ok"] is True
        await _wait_for(left_http, lambda state: state["state"] == "EMERGENCY_STOP", "left emergency stop")
        right_before = await right.get_state()
        assert right_before["state"] == "RUNNING"
        await right.servo_j([0, 4.0, 0, 0, 0, 0])
        await _wait_for(right_http, lambda state: state["ActualQ"][1] > 3.1, "right motion after left stop")
        await right.close()

        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 1, stdout + stderr
        assert "event=shutdown_result result=failure side=left" in stderr
        assert "event=shutdown_result result=success side=right" in stderr
        assert stderr.count("hold=True") >= 2
