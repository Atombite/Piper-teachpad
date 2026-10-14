#!/usr/bin/env python3
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Explicitly guarded single-arm hardware feedback and motion probe."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from piper_control.config import ArmSide, ServiceConfig, load_config
from piper_control.drivers import PiperArmDriver
from piper_control.interpolation import step_axes
from piper_control.types import ArmFeedback
from piper_control.vectors import validate_gripper, validate_servo


MOTION_CONFIRMATION = "I_UNDERSTAND"
JOINT_ARRIVAL_TOLERANCE_DEG = 0.01
GRIPPER_ARRIVAL_TOLERANCE_M = 1e-5
ARRIVAL_STABLE_CYCLES = 3


def _positive_finite(raw: str) -> float:
    value = float(raw)
    if not isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read one Piper arm; motion requires double confirmation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--confirm-motion")
    parser.add_argument("--target-joints", nargs=6, type=float)
    parser.add_argument("--target-gripper", type=float)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--motion-timeout", type=_positive_finite)
    return parser


def validate_motion_targets(
    config: ServiceConfig,
    side: ArmSide,
    target_joints: Sequence[float],
    target_gripper: float | None = None,
) -> None:
    arm = config.arms[side]
    validate_servo(arm, target_joints)
    if target_gripper is not None:
        validate_gripper(arm, target_gripper)


def _feedback_json(sample: ArmFeedback) -> dict[str, object]:
    return {
        "joints_deg": list(sample.joints_deg),
        "gripper_position_m": sample.gripper_position,
        "tcp_pose": list(sample.tcp_pose),
        "connected": sample.connected,
        "enabled": sample.enabled,
        "fault_code": sample.fault_code,
        "stale": sample.stale,
        "received_at_s": sample.received_at_s,
    }


def _valid_hold_feedback(sample: ArmFeedback) -> bool:
    return (
        sample.connected
        and len(sample.joints_deg) == 6
        and all(isfinite(value) for value in sample.joints_deg)
        and isfinite(sample.gripper_position)
    )


def _validate_motion_feedback(side: str, sample: ArmFeedback) -> None:
    numeric = (
        *sample.joints_deg,
        sample.gripper_position,
        *sample.tcp_pose,
        sample.received_at_s,
    )
    reason: str | None = None
    if not sample.connected:
        reason = "disconnected"
    elif not sample.enabled:
        reason = "disabled"
    elif sample.stale:
        reason = "stale"
    elif sample.fault_code != 0:
        reason = f"fault code {sample.fault_code}"
    elif len(sample.joints_deg) != 6 or len(sample.tcp_pose) != 6:
        reason = "invalid dimensions"
    elif not all(isfinite(value) for value in numeric):
        reason = "non-finite values"
    if reason is not None:
        raise RuntimeError(f"{side} unhealthy feedback: {reason}")


def _default_motion_timeout(
    arm: Any,
    current: ArmFeedback,
    servo: Any,
    gripper: Any | None,
) -> float:
    longest = max(
        abs(goal - actual) / speed
        for actual, goal, speed in zip(
            current.joints_deg,
            servo.target,
            arm.default_joints_speed_deg_s,
            strict=True,
        )
    )
    if gripper is not None:
        travel_s = (
            abs(gripper.target - current.gripper_position)
            / arm.gripper.default_speed_m_s
        )
        longest = max(longest, travel_s)
    return longest + max(1.0, ARRIVAL_STABLE_CYCLES / arm.control_hz)


def run_probe(
    config: ServiceConfig,
    *,
    side: ArmSide,
    allow_motion: bool,
    target_joints: Sequence[float] | None,
    target_gripper: float | None,
    settle_seconds: float,
    motion_timeout_s: float | None = None,
    drivers: Mapping[str, Any] | None = None,
    driver_factory: Callable[[Any], Any] = PiperArmDriver,
    wait: Callable[[float], object] = time.sleep,
    output: Callable[[str], object] = print,
) -> None:
    """Probe exactly one configured arm and never touch the other side."""

    if settle_seconds < 0:
        raise ValueError("settle_seconds must not be negative")
    if motion_timeout_s is not None and (
        not isfinite(motion_timeout_s) or motion_timeout_s <= 0
    ):
        raise ValueError("motion_timeout_s must be a positive finite number")
    arm = config.arms[side]
    servo = gripper = None
    if allow_motion:
        if target_joints is None:
            raise ValueError("motion requires an explicit 6-value joint target")
        servo = validate_servo(arm, target_joints)
        if target_gripper is not None:
            gripper = validate_gripper(arm, target_gripper)

    side_name = side.value
    driver = drivers[side_name] if drivers is not None else driver_factory(arm)
    latest: ArmFeedback | None = None
    try:
        driver.connect()
        current = driver.read_feedback()
        if _valid_hold_feedback(current):
            latest = current
        output(
            json.dumps(
                {"mode": "read_only", "feedback": {side_name: _feedback_json(current)}},
                indent=2,
            )
        )
        if not allow_motion:
            return

        assert servo is not None
        driver.enable()
        current = driver.read_feedback()
        if _valid_hold_feedback(current):
            latest = current
        _validate_motion_feedback(side_name, current)
        period = 1.0 / arm.control_hz
        timeout = motion_timeout_s or _default_motion_timeout(arm, current, servo, gripper)
        elapsed = 0.0
        stable_cycles = 0
        while True:
            joints_arrived = all(
                abs(actual - target) <= JOINT_ARRIVAL_TOLERANCE_DEG
                for actual, target in zip(
                    current.joints_deg, servo.target, strict=True
                )
            )
            arrived = joints_arrived
            if not joints_arrived:
                driver.command_joints(
                    step_axes(current.joints_deg, servo.target, servo.speed, period)
                )
            if gripper is not None:
                gripper_arrived = (
                    abs(current.gripper_position - gripper.target)
                    <= GRIPPER_ARRIVAL_TOLERANCE_M
                )
                arrived = arrived and gripper_arrived
                if not gripper_arrived:
                    next_gripper = step_axes(
                        (current.gripper_position,),
                        (gripper.target,),
                        (gripper.speed,),
                        period,
                    )[0]
                    driver.command_gripper(next_gripper)
            stable_cycles = stable_cycles + 1 if arrived else 0
            if stable_cycles >= ARRIVAL_STABLE_CYCLES:
                break
            if elapsed + period > timeout:
                raise TimeoutError(f"motion timeout after {timeout:.3f}s")
            wait(period)
            elapsed += period
            current = driver.read_feedback()
            if _valid_hold_feedback(current):
                latest = current
            _validate_motion_feedback(side_name, current)

        wait(settle_seconds)
        after = driver.read_feedback()
        if _valid_hold_feedback(after):
            latest = after
        _validate_motion_feedback(side_name, after)
        try:
            driver.hold(latest)
        except BaseException as exc:
            raise RuntimeError(f"final HOLD failed: {side_name}: {exc}") from exc
        output(
            json.dumps(
                {
                    "mode": "motion_confirmed",
                    "feedback": {side_name: _feedback_json(after)},
                },
                indent=2,
            )
        )
    except BaseException:
        if allow_motion:
            try:
                driver.emergency_stop()
            except BaseException:
                pass
            if latest is not None:
                try:
                    driver.hold(latest)
                except BaseException:
                    pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    side = ArmSide(args.side)
    motion_requested = args.allow_motion or args.confirm_motion is not None
    if motion_requested and not (
        args.allow_motion and args.confirm_motion == MOTION_CONFIRMATION
    ):
        raise SystemExit("motion requires both --allow-motion and --confirm-motion I_UNDERSTAND")
    if args.allow_motion and args.target_joints is None:
        raise SystemExit("motion requires an explicit 6-value --target-joints")
    if args.settle_seconds < 0:
        raise SystemExit("--settle-seconds must not be negative")
    run_probe(
        config,
        side=side,
        allow_motion=args.allow_motion,
        target_joints=args.target_joints,
        target_gripper=args.target_gripper,
        settle_seconds=args.settle_seconds,
        motion_timeout_s=args.motion_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
