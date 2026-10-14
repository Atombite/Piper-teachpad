#!/usr/bin/env python3
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Guarded independent GELLO teleoperation for two Piper arm services."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
import logging
import math
from numbers import Real
import os
from queue import Empty, Queue
from threading import Lock, Thread
import time
from typing import Any, Awaitable, Callable

if __package__:
    from examples.ws_client import PiperControlClient
else:  # Support ``python examples/gello_ws_teleop.py``.
    from ws_client import PiperControlClient


LOGGER = logging.getLogger(__name__)
LEFT_PORT = "/dev/serial/by-id/REPLACE_WITH_LEFT_GELLO_SERIAL_DEVICE"
RIGHT_PORT = "/dev/serial/by-id/REPLACE_WITH_RIGHT_GELLO_SERIAL_DEVICE"
LEFT_URL = "ws://127.0.0.1:8765/ws/control"
RIGHT_URL = "ws://127.0.0.1:9876/ws/control"
MOTION_CONFIRMATION = "I_UNDERSTAND"


@dataclass(frozen=True, slots=True)
class GelloCalibration:
    port: str
    joint_ids: tuple[int, ...]
    joint_offsets: tuple[float, ...]
    joint_signs: tuple[int, ...]
    gripper_config: tuple[int, int, int]


LEFT_CALIBRATION = GelloCalibration(
    LEFT_PORT,
    (1, 2, 3, 4, 5, 6),
    (math.pi, math.pi / 2, math.pi / 2, 3 * math.pi / 2, 0, math.pi),
    (1, -1, -1, -1, 1, 1),
    (7, 60, 18),
)
RIGHT_CALIBRATION = GelloCalibration(
    RIGHT_PORT,
    (1, 2, 3, 4, 5, 6),
    (
        3 * math.pi / 2,
        math.pi,
        2 * math.pi,
        3 * math.pi / 2,
        3 * math.pi / 2,
        3 * math.pi / 2,
    ),
    (1, -1, -1, -1, 1, 1),
    (7, 330, 289),
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_vector(value: object, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain exactly {length} values")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{label} must contain exactly {length} values") from exc
    if len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, Real)
        or not math.isfinite(float(item))
        for item in values
    ):
        raise ValueError(f"{label} must contain finite numbers")
    return tuple(float(item) for item in values)


def _targets_deg(sample: Sequence[float], side: str) -> tuple[float, ...]:
    try:
        values = tuple(sample)
    except TypeError as exc:
        raise ValueError(f"{side} GELLO sample must contain at least six values") from exc
    if len(values) < 6:
        raise ValueError(f"{side} GELLO sample must contain at least six values")
    joints = _finite_vector(values[:6], 6, f"{side} GELLO joints")
    return tuple(math.degrees(value) for value in joints)


def _validate_startup(
    side: str,
    info: object,
    state: object,
    targets: tuple[float, ...],
    speed_deg_s: float,
    alignment_tolerance_deg: float,
) -> None:
    base_info = _mapping(info, "base info")
    if base_info.get("side") != side:
        raise ValueError(f"Piper service side must be {side}")
    arm_info = _mapping(base_info.get("arm"), "arm")
    current = _mapping(state, "state")
    if current.get("state") != "RUNNING":
        raise ValueError("Piper service must be RUNNING")
    _validate_targets(side, arm_info, targets)
    if (
        not isinstance(speed_deg_s, Real)
        or isinstance(speed_deg_s, bool)
        or not math.isfinite(speed_deg_s)
        or speed_deg_s <= 0
    ):
        raise ValueError("speed must be a positive finite number")
    speeds = _mapping(arm_info.get("joint_speed_deg_s"), "joint speeds")
    maximums = _finite_vector(speeds.get("max"), 6, "maximum speeds")
    if any(speed_deg_s > maximum for maximum in maximums):
        raise ValueError("speed exceeds an advertised joint maximum")
    if (
        not isinstance(alignment_tolerance_deg, Real)
        or isinstance(alignment_tolerance_deg, bool)
        or not math.isfinite(alignment_tolerance_deg)
        or alignment_tolerance_deg < 0
    ):
        raise ValueError("alignment tolerance must be a finite non-negative number")
    actual = _finite_vector(current.get("ActualQ"), 6, "ActualQ")
    for index, (target, present) in enumerate(zip(targets, actual, strict=True), 1):
        if abs(target - present) > alignment_tolerance_deg:
            raise ValueError(
                f"{side} joint {index} alignment exceeds "
                f"{alignment_tolerance_deg:g} degrees"
            )


def _validate_targets(
    side: str,
    base_info: Mapping[str, object],
    targets: tuple[float, ...],
) -> None:
    limits = _mapping(base_info.get("joint_limits_deg"), "joint limits")
    minimums = _finite_vector(limits.get("min"), 6, "minimum limits")
    maximums = _finite_vector(limits.get("max"), 6, "maximum limits")
    limits_by_joint = zip(targets, minimums, maximums, strict=True)
    for index, (target, minimum, maximum) in enumerate(limits_by_joint, 1):
        if not minimum <= target <= maximum:
            raise ValueError(
                f"{side} joint {index} target {target:g} is outside "
                f"[{minimum:g}, {maximum:g}] degrees"
            )


async def _future_result(future: Future[Any]) -> Any:
    while not future.done():
        await asyncio.sleep(0.001)
    return future.result()


SerialWorkItem = tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]


class _DaemonSerialExecutor:
    """Run serial reads one at a time without making cancellation block.

    Python cannot interrupt a thread stuck in a device read. Closing this
    executor cancels queued reads and leaves an in-flight read on a daemon
    thread, so the peer session and process shutdown are not held hostage.
    The vendor call may retain its serial resource until it eventually returns.
    """

    def __init__(self, side: str) -> None:
        self._queue: Queue[SerialWorkItem | None] = Queue()
        self._lock = Lock()
        self._closed = False
        self._thread = Thread(
            target=self._worker,
            name=f"gello-{side}",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self, function: Callable[..., Any], *args: Any
    ) -> Future[Any]:
        future: Future[Any] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("GELLO serial executor is closed")
            self._queue.put((future, function, args))
        return future

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            while True:
                try:
                    item = self._queue.get_nowait()
                except Empty:
                    break
                if item is not None:
                    item[0].cancel()
            self._queue.put(None)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, function, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = function(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)


async def run_side(
    side: str,
    url: str,
    agent: Any,
    *,
    control_hz: float,
    speed_deg_s: float,
    alignment_tolerance_deg: float,
    state_check_interval_s: float,
    max_cycles: int | None = None,
    client_factory: Callable[[str], Any] = PiperControlClient,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> None:
    """Stream one GELLO's six joints to its matching arm service."""

    if (
        not isinstance(control_hz, Real)
        or isinstance(control_hz, bool)
        or not math.isfinite(control_hz)
        or control_hz <= 0
    ):
        raise ValueError("control_hz must be a positive finite number")
    if (
        not isinstance(state_check_interval_s, Real)
        or isinstance(state_check_interval_s, bool)
        or not math.isfinite(state_check_interval_s)
        or state_check_interval_s < 0
    ):
        raise ValueError("state check interval must be finite and non-negative")
    if max_cycles is not None and max_cycles < 0:
        raise ValueError("max_cycles must be non-negative")

    executor = _DaemonSerialExecutor(side)
    try:
        async with client_factory(url) as client:
            info = await client.get_base_info()
            state = await client.get_state()
            targets = _targets_deg(await _future_result(executor.submit(agent.act, {})), side)
            _validate_startup(side, info, state, targets, speed_deg_s, alignment_tolerance_deg)
            speed = (float(speed_deg_s),) * 6
            period = 1.0 / float(control_hz)
            next_deadline = clock()
            last_state_check = next_deadline
            cycles = 0
            while max_cycles is None or cycles < max_cycles:
                arm_info = _mapping(
                    _mapping(info, "base info").get("arm"), "arm"
                )
                _validate_targets(side, arm_info, targets)
                now = clock()
                if now - last_state_check >= state_check_interval_s:
                    state = await client.get_state()
                    if _mapping(state, "state").get("state") != "RUNNING":
                        raise RuntimeError("Piper service left RUNNING state")
                    last_state_check = now
                await client.servo_j(targets, speed=speed)
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                next_deadline += period
                await sleep(max(0.0, next_deadline - clock()))
                targets = _targets_deg(await _future_result(executor.submit(agent.act, {})), side)
    finally:
        executor.close()


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream two GELLO leaders to independent Piper arm services."
    )
    parser.add_argument("--left-url", default=LEFT_URL)
    parser.add_argument("--right-url", default=RIGHT_URL)
    parser.add_argument(
        "--left-port",
        default=LEFT_PORT,
        help="stable device path from: ls -l /dev/serial/by-id/",
    )
    parser.add_argument(
        "--right-port",
        default=RIGHT_PORT,
        help="stable device path from: ls -l /dev/serial/by-id/",
    )
    parser.add_argument("--control-hz", type=_positive_float, default=50.0)
    parser.add_argument("--speed-deg-s", type=_positive_float, default=10.0)
    parser.add_argument("--alignment-tolerance-deg", type=_non_negative_float, default=5.0)
    parser.add_argument("--state-check-interval-s", type=_non_negative_float, default=0.5)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--confirm-motion", metavar=MOTION_CONFIRMATION)
    return parser


def make_gello_agent(calibration: GelloCalibration, port: str) -> Any:
    try:
        from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent
    except ImportError as exc:
        raise RuntimeError(
            "GELLO support is not installed; follow "
            "https://github.com/wuphilipp/gello_software for installation"
        ) from exc
    config = DynamixelRobotConfig(
        joint_ids=calibration.joint_ids,
        joint_offsets=calibration.joint_offsets,
        joint_signs=calibration.joint_signs,
        gripper_config=calibration.gripper_config,
    )
    return GelloAgent(port=port, dynamixel_config=config)


async def _supervise_side(side: str, operation: Awaitable[None]) -> None:
    try:
        await operation
    except asyncio.CancelledError:
        raise
    except BaseException:
        LOGGER.exception("%s GELLO task failed", side)


async def async_main(
    args: argparse.Namespace,
    *,
    agent_factory: Callable[[GelloCalibration, str], Any] = make_gello_agent,
    path_exists: Callable[[str], bool] = os.path.exists,
    runner: Callable[..., Awaitable[None]] = run_side,
) -> None:
    if not args.allow_motion or args.confirm_motion != MOTION_CONFIRMATION:
        raise ValueError(f"motion requires --allow-motion --confirm-motion {MOTION_CONFIRMATION}")
    if args.left_port == args.right_port:
        raise ValueError("left and right GELLO ports must be different")
    for side, port in (("left", args.left_port), ("right", args.right_port)):
        if not path_exists(port):
            raise ValueError(f"{side} GELLO device does not exist: {port}")
    agents = {
        "left": agent_factory(LEFT_CALIBRATION, args.left_port),
        "right": agent_factory(RIGHT_CALIBRATION, args.right_port),
    }
    urls = {"left": args.left_url, "right": args.right_url}
    tasks = {
        side: asyncio.create_task(
            _supervise_side(
                side,
                runner(
                    side,
                    urls[side],
                    agents[side],
                    control_hz=args.control_hz,
                    speed_deg_s=args.speed_deg_s,
                    alignment_tolerance_deg=args.alignment_tolerance_deg,
                    state_check_interval_s=args.state_check_interval_s,
                ),
            )
        )
        for side in ("left", "right")
    }
    try:
        await asyncio.gather(*tasks.values())
    except asyncio.CancelledError:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        raise


def main() -> None:
    parser = build_parser()
    try:
        asyncio.run(async_main(parser.parse_args()))
    except KeyboardInterrupt:
        print("GELLO teleoperation stopped; WebSockets closed.")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"GELLO teleoperation failed: {exc}\n")


if __name__ == "__main__":
    main()
