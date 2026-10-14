# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import builtins
import math
from threading import Event, enumerate as enumerate_threads
import time
from typing import Any

import pytest
import examples.gello_ws_teleop as gello

from examples.gello_ws_teleop import (
    LEFT_CALIBRATION,
    LEFT_PORT,
    MOTION_CONFIRMATION,
    RIGHT_CALIBRATION,
    RIGHT_PORT,
    async_main,
    build_parser,
    make_gello_agent,
)


def _base_info(side: str) -> dict[str, object]:
    return {
        "side": side,
        "protocol_version": "2.0",
        "arm": {
            "name": f"{side}_arm",
            "joint_limits_deg": {
                "min": [-150, 0, -170, -100, -70, -120],
                "max": [150, 180, 0, 100, 70, 120],
            },
            "joint_speed_deg_s": {"max": [30] * 6},
        },
    }


def _running_state(actual_q: list[float] | None = None) -> dict[str, object]:
    return {"state": "RUNNING", "ActualQ": actual_q or [0.0] * 6}


class FakeAgent:
    def __init__(self, samples: list[tuple[float, ...]]) -> None:
        self.samples = samples
        self.calls = 0

    def act(self, observation: dict[str, object]) -> tuple[float, ...]:
        assert observation == {}
        sample = self.samples[min(self.calls, len(self.samples) - 1)]
        self.calls += 1
        return sample


class FakeClient:
    def __init__(self, side: str, *, fail_after: int | None = None) -> None:
        self.side = side
        self.fail_after = fail_after
        self.closed = False
        self.servo_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.closed = True

    async def get_base_info(self) -> dict[str, Any]:
        return _base_info(self.side)

    async def get_state(self) -> dict[str, Any]:
        return _running_state()

    async def servo_j(
        self, target_joints: tuple[float, ...], speed: tuple[float, ...]
    ) -> dict[str, bool]:
        if self.fail_after is not None and len(self.servo_calls) >= self.fail_after:
            raise RuntimeError(f"{self.side} failed")
        self.servo_calls.append((tuple(target_joints), tuple(speed)))
        return {"accepted": True}


async def _no_sleep(_delay: float) -> None:
    return None


def test_calibrations_use_documented_device_placeholders() -> None:
    assert LEFT_CALIBRATION.port == (
        "/dev/serial/by-id/REPLACE_WITH_LEFT_GELLO_SERIAL_DEVICE"
    )
    assert LEFT_CALIBRATION.joint_ids == (1, 2, 3, 4, 5, 6)
    assert RIGHT_CALIBRATION.port == (
        "/dev/serial/by-id/REPLACE_WITH_RIGHT_GELLO_SERIAL_DEVICE"
    )
    assert RIGHT_CALIBRATION.joint_ids == (1, 2, 3, 4, 5, 6)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side", "sample", "expected"),
    [
        ("left", (0, math.pi / 180, 0, 0, 0, 0, 0.25), (0, 1, 0, 0, 0, 0)),
        ("right", (0, 0, -math.pi / 180, 0, 0, 0, 0.75), (0, 0, -1, 0, 0, 0)),
    ],
)
async def test_side_session_preserves_joint_order_and_ignores_gripper(
    side: str, sample: tuple[float, ...], expected: tuple[float, ...]
) -> None:
    client = FakeClient(side)

    await gello.run_side(
        side,
        f"ws://{side}",
        FakeAgent([sample]),
        control_hz=50,
        speed_deg_s=10,
        alignment_tolerance_deg=5,
        state_check_interval_s=1,
        max_cycles=1,
        client_factory=lambda _url: client,
        sleep=_no_sleep,
    )

    assert client.servo_calls == [(expected, (10.0,) * 6)]
    assert client.closed is True


@pytest.mark.asyncio
async def test_side_session_requires_matching_service_side() -> None:
    client = FakeClient("right")
    with pytest.raises(ValueError, match="side"):
        await gello.run_side(
            "left",
            "ws://left",
            FakeAgent([(0.0,) * 7]),
            control_hz=50,
            speed_deg_s=10,
            alignment_tolerance_deg=5,
            state_check_interval_s=1,
            max_cycles=1,
            client_factory=lambda _url: client,
        )
    assert client.closed is True


@pytest.mark.asyncio
async def test_one_side_failure_does_not_cancel_other_side(caplog) -> None:
    args = build_parser().parse_args(
        ["--allow-motion", "--confirm-motion", MOTION_CONFIRMATION]
    )
    clients = {"left": FakeClient("left", fail_after=1), "right": FakeClient("right")}
    agents: dict[str, FakeAgent] = {}

    def agent_factory(calibration: object, _port: str) -> FakeAgent:
        side = "left" if calibration is LEFT_CALIBRATION else "right"
        agents[side] = FakeAgent([(0.0,) * 7])
        return agents[side]

    async def runner(side: str, url: str, agent: Any, **kwargs: object) -> None:
        await gello.run_side(
            side,
            url,
            agent,
            **kwargs,
            max_cycles=3,
            client_factory=lambda _url: clients[side],
            sleep=_no_sleep,
        )

    await async_main(
        args,
        agent_factory=agent_factory,
        path_exists=lambda _path: True,
        runner=runner,
    )

    assert len(clients["left"].servo_calls) == 1
    assert len(clients["right"].servo_calls) == 3
    assert clients["left"].closed and clients["right"].closed
    assert "left GELLO task failed" in caplog.text


@pytest.mark.asyncio
async def test_side_failure_is_logged_while_other_side_is_still_running(caplog) -> None:
    args = build_parser().parse_args(
        ["--allow-motion", "--confirm-motion", MOTION_CONFIRMATION]
    )
    started = {"left": asyncio.Event(), "right": asyncio.Event()}
    release_right = asyncio.Event()

    async def runner(side: str, _url: str, _agent: Any, **_kwargs: object) -> None:
        started[side].set()
        await asyncio.gather(*(event.wait() for event in started.values()))
        if side == "left":
            raise RuntimeError("left failed immediately")
        await release_right.wait()

    task = asyncio.create_task(
        async_main(
            args,
            agent_factory=lambda _calibration, _port: object(),
            path_exists=lambda _path: True,
            runner=runner,
        )
    )
    await asyncio.gather(*(event.wait() for event in started.values()))
    await asyncio.sleep(0)

    try:
        assert task.done() is False
        assert "left GELLO task failed" in caplog.text
    finally:
        release_right.set()
        await task


@pytest.mark.asyncio
async def test_cancellation_closes_both_side_sessions() -> None:
    args = build_parser().parse_args(
        ["--allow-motion", "--confirm-motion", MOTION_CONFIRMATION]
    )
    clients = {"left": FakeClient("left"), "right": FakeClient("right")}
    started = {"left": asyncio.Event(), "right": asyncio.Event()}

    async def runner(side: str, _url: str, _agent: Any, **_kwargs: object) -> None:
        async with clients[side]:
            started[side].set()
            await asyncio.Future()

    task = asyncio.create_task(
        async_main(
            args,
            agent_factory=lambda _calibration, _port: object(),
            path_exists=lambda _path: True,
            runner=runner,
        )
    )
    await asyncio.gather(*(event.wait() for event in started.values()))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert clients["left"].closed and clients["right"].closed


@pytest.mark.asyncio
async def test_real_side_cancellation_does_not_wait_for_blocked_agent_reads() -> None:
    clients = {"left": FakeClient("left"), "right": FakeClient("right")}
    entered = {"left": Event(), "right": Event()}
    release = Event()

    class BlockingAgent:
        def __init__(self, side: str) -> None:
            self.side = side

        def act(self, _observation: dict[str, object]) -> tuple[float, ...]:
            entered[self.side].set()
            release.wait(0.5)
            return (0.0,) * 7

    tasks = [
        asyncio.create_task(
            gello.run_side(
                side,
                f"ws://{side}",
                BlockingAgent(side),
                control_hz=50,
                speed_deg_s=10,
                alignment_tolerance_deg=5,
                state_check_interval_s=1,
                client_factory=lambda _url, side=side: clients[side],
            )
        )
        for side in ("left", "right")
    ]
    try:
        deadline = time.monotonic() + 0.2
        while not all(event.is_set() for event in entered.values()):
            assert time.monotonic() < deadline
            await asyncio.sleep(0.001)
        started = time.monotonic()
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert clients["left"].closed and clients["right"].closed
    finally:
        release.set()
        await asyncio.sleep(0.02)

    assert not any(
        thread.name.startswith("gello-") for thread in enumerate_threads()
    )


def test_cli_defaults_use_independent_arm_endpoints() -> None:
    args = build_parser().parse_args([])
    assert args.left_url == gello.LEFT_URL == "ws://127.0.0.1:8765/ws/control"
    assert args.right_url == gello.RIGHT_URL == "ws://127.0.0.1:9876/ws/control"
    assert args.left_port == LEFT_PORT
    assert args.right_port == RIGHT_PORT
    assert args.allow_motion is False
    assert "ls -l /dev/serial/by-id/" in build_parser().format_help()


@pytest.mark.asyncio
async def test_cli_rejects_inexact_confirmation_before_touching_devices() -> None:
    args = build_parser().parse_args(
        ["--allow-motion", "--confirm-motion", "I_understand"]
    )

    def forbidden_agent_factory(*_args: object) -> object:
        raise AssertionError("serial devices must not be opened")

    with pytest.raises(ValueError, match=MOTION_CONFIRMATION):
        await async_main(
            args,
            agent_factory=forbidden_agent_factory,
            path_exists=lambda _path: True,
        )


def test_missing_gello_dependency_reports_official_install_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_gello(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("gello"):
            raise ImportError("forced missing gello")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gello)
    with pytest.raises(RuntimeError, match="github.com/wuphilipp/gello_software"):
        make_gello_agent(LEFT_CALIBRATION, LEFT_PORT)
