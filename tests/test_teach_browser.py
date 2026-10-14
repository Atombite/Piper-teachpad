# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import asyncio
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

import pytest
import websockets


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_SCRIPT = ROOT / "tests/browser/teach-acceptance.test.js"


class TeachingTestHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        requested = unquote(urlsplit(path).path)
        if requested.startswith("/static/teach/"):
            relative = requested.removeprefix("/static/teach/")
            target = ROOT / "src/piper_control/web" / relative
        elif requested == "/teach":
            target = ROOT / "src/piper_control/web/teach.html"
        elif requested == "/generated/teach-tests.bundle.js":
            target = ROOT / "tests/browser/teach-tests.bundle.js"
        elif requested == "/tests/browser/harness.html":
            target = ROOT / "tests/browser/harness.html"
        elif requested == "/fixture/missing-mesh.urdf":
            target = ROOT / "tests/browser/missing-mesh.urdf"
        else:
            target = ROOT / "__not_found__"
        return str(target)

    def log_message(self, format: str, *args: object) -> None:
        return


class TeachingTestServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


@contextmanager
def teaching_test_server():
    server = TeachingTestServer(("127.0.0.1", 0), TeachingTestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_browser_kinematics_state_and_viewer() -> None:
    if shutil.which("google-chrome") is None:
        pytest.skip("google-chrome is required for browser teaching tests")
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("npm is required for browser teaching tests")
    bundle = ROOT / "tests/browser/teach-tests.bundle.js"
    env = os.environ.copy()
    bundle.unlink(missing_ok=True)
    try:
        build = subprocess.run(
            [npm, "run", "build:teach-tests"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert build.returncode == 0, build.stdout + build.stderr

        with teaching_test_server() as base_url:
            chrome = subprocess.run(
                [
                    "google-chrome",
                    "--headless",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--virtual-time-budget=10000",
                    "--dump-dom",
                    f"{base_url}/tests/browser/harness.html",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
    finally:
        bundle.unlink(missing_ok=True)

    diagnostic = chrome.stdout + "\n--- chrome stderr ---\n" + chrome.stderr
    assert chrome.returncode == 0, diagnostic
    assert '<body data-status="passed">' in chrome.stdout, diagnostic
    assert not bundle.exists(), "browser test bundle must not pollute the worktree"


async def _assert_teaching_footer_clears_fixed_command_bar(cdp_url: str, base_url: str) -> None:
    async with CdpPage(cdp_url) as page:
        overlaps = []
        for width in (1440, 390):
            await page.command(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": width,
                    "height": 900,
                    "deviceScaleFactor": 1,
                    "mobile": width < 560,
                },
            )
            await page.navigate(f"{base_url}/teach")
            metrics = await page.evaluate(
                r"""
                new Promise(resolve => {
                  window.scrollTo(0, document.documentElement.scrollHeight);
                  requestAnimationFrame(() => requestAnimationFrame(() => {
                    const footer = document.querySelector('.company-footer').getBoundingClientRect();
                    const commandElement = document.querySelector('.command-bar');
                    const command = commandElement.getBoundingClientRect();
                    const shadowLengths = getComputedStyle(commandElement).boxShadow
                      .match(/-?\d+(?:\.\d+)?px/g)
                      .map(value => Number.parseFloat(value));
                    const [, shadowY, shadowBlur] = shadowLengths;
                    resolve({
                      footerTop: footer.top,
                      footerBottom: footer.bottom,
                      commandTop: command.top,
                      commandBottom: command.bottom,
                      commandPaintTop: command.top + Math.min(0, shadowY - shadowBlur),
                    });
                  }));
                })
                """
            )
            if metrics["footerBottom"] > metrics["commandPaintTop"]:
                overlaps.append(f"{width}px: {metrics}")
        assert not overlaps, "footer is covered by the fixed command bar or its shadow: " + "; ".join(overlaps)


def test_teaching_company_footer_clears_fixed_command_bar_in_browser(tmp_path: Path) -> None:
    if shutil.which("google-chrome") is None:
        pytest.skip("google-chrome is required for browser teaching tests")
    with teaching_test_server() as base_url, _chrome(tmp_path) as cdp_url:
        asyncio.run(_assert_teaching_footer_clears_fixed_command_bar(cdp_url, base_url))


class CdpPage:
    def __init__(self, websocket_url: str) -> None:
        self.websocket_url = websocket_url
        self.websocket: Any = None
        self.reader: asyncio.Task[None] | None = None
        self.next_id = 0
        self.pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self.websocket_urls: dict[str, str] = {}
        self.servo_frames: dict[str, list[dict[str, Any]]] = {}
        self.gripper_frames: dict[str, list[dict[str, Any]]] = {}
        self.http_requests: list[dict[str, Any]] = []
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def __aenter__(self) -> "CdpPage":
        self.websocket = await websockets.connect(self.websocket_url, max_size=8 * 1024 * 1024)
        self.reader = asyncio.create_task(self._read())
        await self.command("Page.enable")
        await self.command("Runtime.enable")
        await self.command("Network.enable")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.websocket.close()
        if self.reader is not None:
            await self.reader

    async def _read(self) -> None:
        async for raw in self.websocket:
            message = json.loads(raw)
            if "id" in message:
                future = self.pending.pop(message["id"], None)
                if future is not None and not future.done():
                    future.set_result(message)
                continue
            await self.events.put(message)
            if message.get("method") == "Network.webSocketCreated":
                params = message.get("params", {})
                self.websocket_urls[params.get("requestId", "")] = params.get("url", "")
                continue
            if message.get("method") == "Network.requestWillBeSent":
                self.http_requests.append(message.get("params", {}).get("request", {}))
                continue
            if message.get("method") != "Network.webSocketFrameSent":
                continue
            params = message.get("params", {})
            url = self.websocket_urls.get(params.get("requestId", ""))
            if not url:
                continue
            payload = params.get("response", {}).get("payloadData", "")
            try:
                frame = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                continue
            if frame.get("cmd") == "servoJ":
                self.servo_frames.setdefault(url, []).append(frame)
            elif frame.get("cmd") == "move_gripper":
                self.gripper_frames.setdefault(url, []).append(frame)

    def frames_for(self, websocket_url: str) -> list[dict[str, Any]]:
        return self.servo_frames.setdefault(websocket_url, [])

    def clear_frames(self, websocket_url: str) -> None:
        self.servo_frames[websocket_url] = []
        self.gripper_frames[websocket_url] = []

    def gripper_frames_for(self, websocket_url: str) -> list[dict[str, Any]]:
        return self.gripper_frames.setdefault(websocket_url, [])

    async def command(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 10,
    ) -> dict[str, Any]:
        self.next_id += 1
        message_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[message_id] = future
        await self.websocket.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        response = await asyncio.wait_for(future, timeout=timeout_s)
        if "error" in response:
            raise AssertionError(f"CDP {method} failed: {response['error']}")
        return response.get("result", {})

    async def wait_for_event(self, method: str, *, timeout_s: float = 10) -> dict[str, Any]:
        async def next_matching() -> dict[str, Any]:
            while True:
                event = await self.events.get()
                if event.get("method") == method:
                    return event.get("params", {})

        return await asyncio.wait_for(next_matching(), timeout=timeout_s)

    async def evaluate(self, expression: str, *, await_promise: bool = True) -> Any:
        result = await self.command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
            },
            timeout_s=45 if await_promise else 10,
        )
        detail = result.get("exceptionDetails")
        if detail:
            raise AssertionError(f"browser evaluation failed: {detail}")
        return result.get("result", {}).get("value")

    async def navigate(self, url: str) -> None:
        await self.command("Page.navigate", {"url": url})
        await self.evaluate(
            "new Promise(resolve => document.readyState === 'complete' ? resolve() : "
            "window.addEventListener('load', resolve, {once: true}))"
        )


def _request_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=1) as response:
        return json.load(response)


def _port_is_available(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


@contextmanager
def _real_mujoco_service(tmp_path: Path):
    command = _mujoco_service_command(tmp_path / "piper-control-teach.lock")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=2)
                raise AssertionError(f"MuJoCo service exited during startup\n{stdout}\n{stderr}")
            try:
                states = [_request_json(f"http://127.0.0.1:{port}/api/state") for port in (8765, 9876)]
                if all(state["state"] == "RUNNING" for state in states):
                    break
            except OSError:
                pass
            time.sleep(0.05)
        else:
            raise AssertionError("MuJoCo service did not expose two RUNNING arms")
        yield process
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.communicate(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=2)


def _mujoco_service_command(lock_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "piper_control.main",
        "--config",
        str(ROOT / "config/piper-control.mujoco.yaml"),
        "--mujoco-headless",
        "--lock-path",
        str(lock_path),
    ]


def test_mujoco_browser_acceptance_uses_headless_cli(tmp_path: Path) -> None:
    command = _mujoco_service_command(tmp_path / "acceptance.lock")
    assert "--mujoco-headless" in command
    assert "--mujoco" not in command


@contextmanager
def _chrome(tmp_path: Path):
    profile = tmp_path / "chrome-profile"
    process = subprocess.Popen(
        [
            "google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--remote-allow-origins=*",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=2)
                raise AssertionError(f"Chrome exited during startup\n{stdout}\n{stderr}")
            active_port = profile / "DevToolsActivePort"
            try:
                debug_port = int(active_port.read_text(encoding="utf-8").splitlines()[0])
                pages = _request_json(f"http://127.0.0.1:{debug_port}/json/list")
                page = next(item for item in pages if item["type"] == "page")
                break
            except (OSError, ValueError, IndexError, StopIteration):
                time.sleep(0.05)
        else:
            raise AssertionError("Chrome DevTools endpoint did not start")
        yield page["webSocketDebuggerUrl"]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=2)
        shutil.rmtree(profile, ignore_errors=True)


async def _run_teach_acceptance(cdp_url: str) -> None:
    helper = ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
    targets = {
        8765: [0, 75, -80, 5, 5, 5],
        9876: [0, 70, -75, -5, 5, -5],
    }
    async with CdpPage(cdp_url) as page:
        for port, expected_side in ((8765, "LEFT"), (9876, "RIGHT")):
            control_url = f"ws://127.0.0.1:{port}/ws/control"
            page.clear_frames(control_url)
            await page.navigate(f"http://127.0.0.1:{port}/teach")
            await page.evaluate(helper)
            ready = await page.evaluate(f"__teachAcceptance.waitReady('{expected_side}')")
            assert ready["side"] == expected_side
            assert len(ready["actual"]) == len(ready["target"]) == 6
            actual_on_load = [float(value.removesuffix("°")) for value in ready["actual"]]
            assert max(abs(actual - target) for actual, target in zip(actual_on_load, ready["target"])) <= 0.1

            before = len(page.frames_for(control_url))
            target = targets[port]
            edited = await page.evaluate(f"__teachAcceptance.setJointTarget({json.dumps(target)})")
            assert edited == target
            await asyncio.sleep(0.3)
            assert len(page.frames_for(control_url)) == before, "preview edit sent servoJ"

            await page.evaluate("__teachAcceptance.selectMode('reviewed')")
            await page.evaluate("__teachAcceptance.waitForAvailable('#execute-target')")
            await page.evaluate("__teachAcceptance.click('#execute-target')")
            await page.evaluate("__teachAcceptance.waitForReviewedCompletion()")
            reviewed_frames = page.frames_for(control_url)[before:]
            assert reviewed_frames, f"{expected_side} reviewed execution sent no servoJ"
            assert reviewed_frames[0]["target_joints"] == target
            state = await page.evaluate("__teachAcceptance.fetchState()")
            assert max(abs(actual - requested) for actual, requested in zip(state["ActualQ"], target)) <= 2

            await page.evaluate("__teachAcceptance.selectMode('live')")
            await page.evaluate("__teachAcceptance.waitForAvailable('#acquire-control')")
            await page.evaluate("__teachAcceptance.click('#acquire-control')")
            await page.evaluate("__teachAcceptance.waitForLiveArmed()")
            live_before = len(page.frames_for(control_url))
            live_target = await page.evaluate("__teachAcceptance.moveTcpXBy(10)")
            await page.evaluate("__teachAcceptance.holdDeadman()")
            deadline = time.monotonic() + 4
            while len(page.frames_for(control_url)) == live_before and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            live_frames = page.frames_for(control_url)[live_before:]
            assert live_frames, f"{expected_side} live follow sent no servoJ"
            assert live_frames[-1]["target_joints"] == pytest.approx(live_target, abs=0.01)
            await page.evaluate("__teachAcceptance.releaseDeadman()")
            released_count = len(page.frames_for(control_url))
            await asyncio.sleep(0.5)
            assert len(page.frames_for(control_url)) == released_count
            state = await page.evaluate("__teachAcceptance.fetchState()")
            assert state["ws_controller_connected"] is False

            await page.evaluate("__teachAcceptance.waitForAvailable('#acquire-control')")
            await page.evaluate("__teachAcceptance.click('#acquire-control')")
            await page.evaluate("__teachAcceptance.waitForLiveArmed()")
            second_live_before = len(page.frames_for(control_url))
            await page.evaluate("__teachAcceptance.holdDeadman()")
            deadline = time.monotonic() + 4
            while len(page.frames_for(control_url)) == second_live_before and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert len(page.frames_for(control_url)) > second_live_before, (
                f"{expected_side} fresh reacquisition sent no second live servoJ"
            )

            emergency_requests_before = len(page.http_requests)
            click_task = asyncio.create_task(page.evaluate("__teachAcceptance.click('#emergency-stop')"))
            dialog = await page.wait_for_event("Page.javascriptDialogOpening")
            assert "Emergency-stop" in dialog.get("message", "")
            await page.command("Page.handleJavaScriptDialog", {"accept": True})
            await click_task
            emergency_state = await page.evaluate("__teachAcceptance.waitForEmergencyStop()")
            assert emergency_state["emergency_stopped"] is True
            assert emergency_state["state"] != "RUNNING"
            assert emergency_state["ws_controller_connected"] is False
            new_requests = page.http_requests[emergency_requests_before:]
            assert any(
                request.get("method") == "GET"
                and request.get("url", "").endswith("/api/emergency_stop")
                for request in new_requests
            )


async def _run_gripper_acceptance(cdp_url: str) -> None:
    helper = ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
    async with CdpPage(cdp_url) as page:
        for port, expected_side, target_mm in ((8765, "LEFT", 20), (9876, "RIGHT", 30)):
            control_url = f"ws://127.0.0.1:{port}/ws/control"
            page.clear_frames(control_url)
            await page.navigate(f"http://127.0.0.1:{port}/teach")
            await page.evaluate(helper)
            await page.evaluate(f"__teachAcceptance.waitReady('{expected_side}')")
            await page.evaluate("__teachAcceptance.waitForGripperAvailable()")
            await page.evaluate(f"__teachAcceptance.dragGripper({target_mm}, 40)")
            deadline = time.monotonic() + 4
            while not page.gripper_frames_for(control_url) and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            frames = page.gripper_frames_for(control_url)
            assert frames, f"{expected_side} gripper drag sent no move_gripper"
            assert frames[-1]["target_position"] == pytest.approx(target_mm / 1000)
            assert frames[-1]["speed"] == pytest.approx(0.04)
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                state = await page.evaluate("__teachAcceptance.fetchState()")
                if abs(state["ActualGripper"] - target_mm / 1000) <= 0.005:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError(f"{expected_side} gripper did not reach its browser target")
            await page.evaluate("__teachAcceptance.releaseGripper()")
            released_count = len(frames)
            await asyncio.sleep(0.5)
            assert len(page.gripper_frames_for(control_url)) == released_count
            state = await page.evaluate("__teachAcceptance.waitForLeaseReleased()")
            assert state["ws_controller_connected"] is False


@pytest.mark.mujoco_acceptance
def test_teach_acceptance_real_dual_arm_mujoco(tmp_path: Path) -> None:
    if os.environ.get("PIPER_RUN_MUJOCO_ACCEPTANCE") != "1":
        pytest.skip("set PIPER_RUN_MUJOCO_ACCEPTANCE=1 to run real MuJoCo browser acceptance")
    if shutil.which("google-chrome") is None:
        pytest.skip("google-chrome is required for MuJoCo browser acceptance")
    pytest.importorskip("mujoco")
    occupied = [port for port in (8765, 9876) if not _port_is_available(port)]
    if occupied:
        pytest.skip(f"ports {occupied} are occupied; stop their services manually before acceptance")
    with _real_mujoco_service(tmp_path), _chrome(tmp_path) as cdp_url:
        asyncio.run(_run_teach_acceptance(cdp_url))


@pytest.mark.mujoco_acceptance
def test_teach_gripper_acceptance_real_dual_arm_mujoco(tmp_path: Path) -> None:
    if os.environ.get("PIPER_RUN_MUJOCO_ACCEPTANCE") != "1":
        pytest.skip("set PIPER_RUN_MUJOCO_ACCEPTANCE=1 to run real MuJoCo browser acceptance")
    if shutil.which("google-chrome") is None:
        pytest.skip("google-chrome is required for MuJoCo browser acceptance")
    pytest.importorskip("mujoco")
    occupied = [port for port in (8765, 9876) if not _port_is_available(port)]
    if occupied:
        pytest.skip(f"ports {occupied} are occupied; stop their services manually before acceptance")
    with _real_mujoco_service(tmp_path), _chrome(tmp_path) as cdp_url:
        asyncio.run(_run_gripper_acceptance(cdp_url))
