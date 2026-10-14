# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Read-only raw-joint calibration utility for Piper arms."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Protocol, TextIO

import yaml

from piper_control.config import ArmConfig, ArmSide, ServiceConfig, load_config
from piper_control.drivers.piper import PiperArmDriver
from piper_control.types import RawJointFeedback


JointMdeg = tuple[int, int, int, int, int, int]
ZERO_SAMPLE_COUNT = 50


class CalibrationError(RuntimeError):
    """Calibration cannot safely produce a result."""


class RawJointDriver(Protocol):
    def connect(self) -> None: ...

    def read_raw_joint_feedback(self) -> RawJointFeedback: ...


@dataclass(frozen=True, slots=True)
class ArmCalibrationResult:
    name: str
    can_port: str
    calibration_mdeg: JointMdeg
    raw_min_mdeg: JointMdeg
    raw_max_mdeg: JointMdeg
    min_joints_mdeg: JointMdeg
    max_joints_mdeg: JointMdeg
    range_frame_count: int


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    arms: Mapping[str, ArmCalibrationResult]


@dataclass(slots=True)
class _ArmCollection:
    config: ArmConfig
    zero_samples: list[JointMdeg]
    last_timestamp: float | None
    last_new_frame_at: float
    calibration: JointMdeg | None = None
    raw_min: JointMdeg | None = None
    raw_max: JointMdeg | None = None
    range_frame_count: int = 0


def calibration_mode_mdeg(samples: Sequence[JointMdeg]) -> JointMdeg:
    """Return each joint's mode, using tied-mode median in millidegrees."""

    if not samples:
        raise ValueError("at least one calibration sample is required")
    values: list[int] = []
    for joint_index in range(6):
        counts = Counter(sample[joint_index] for sample in samples)
        highest = max(counts.values())
        tied = sorted(value for value, count in counts.items() if count == highest)
        values.append(round(statistics.median(tied)))
    return tuple(values)  # type: ignore[return-value]


def _validate_feedback(
    arm: ArmConfig,
    feedback: RawJointFeedback,
    *,
    allow_startup_stale: bool = False,
) -> JointMdeg | None:
    if not feedback.connected:
        raise CalibrationError(f"{arm.name}: arm is disconnected")
    if feedback.fault_code:
        raise CalibrationError(
            f"{arm.name}: arm reported fault {feedback.fault_code}"
        )
    if not math.isfinite(feedback.received_at_s):
        raise CalibrationError(f"{arm.name}: joint timestamp is not finite")
    if feedback.stale or feedback.received_at_s <= 0:
        if allow_startup_stale:
            return None
        raise CalibrationError(f"{arm.name}: joint feedback is stale")
    if len(feedback.joints_mdeg) != 6:
        raise CalibrationError(f"{arm.name}: joint feedback must contain six values")

    joints: list[int] = []
    for index, value in enumerate(feedback.joints_mdeg):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CalibrationError(
                f"{arm.name}: joint {index + 1} is not an integer millidegree value"
            )
        if not math.isfinite(float(value)):
            raise CalibrationError(f"{arm.name}: joint {index + 1} is not finite")
        physical_min = round(ArmConfig.PHYSICAL_MIN_JOINTS_DEG[index] * 1000)
        physical_max = round(ArmConfig.PHYSICAL_MAX_JOINTS_DEG[index] * 1000)
        if not physical_min <= value <= physical_max:
            raise CalibrationError(
                f"{arm.name}: joint {index + 1} exceeds Piper physical range"
            )
        joints.append(value)
    return tuple(joints)  # type: ignore[return-value]


def _accept_new_frame(
    collection: _ArmCollection,
    feedback: RawJointFeedback,
    now: float,
) -> JointMdeg | None:
    joints = _validate_feedback(
        collection.config,
        feedback,
        allow_startup_stale=collection.last_timestamp is None,
    )
    if joints is None:
        return None
    if (
        collection.last_timestamp is not None
        and feedback.received_at_s <= collection.last_timestamp
    ):
        return None
    collection.last_timestamp = feedback.received_at_s
    collection.last_new_frame_at = now
    return joints


def _check_timeout(
    collection: _ArmCollection, now: float, timeout_s: float
) -> None:
    if now - collection.last_new_frame_at >= timeout_s:
        raise CalibrationError(
            f"{collection.config.name}: timed out waiting for a new joint frame"
        )


def run_calibration(
    config: ServiceConfig,
    drivers: Mapping[str, RawJointDriver],
    *,
    should_stop: Callable[[], bool] = lambda: False,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] | None = None,
) -> CalibrationReport:
    """Collect zero and travel frames from already configured Piper arms."""

    notify = progress or (lambda _: None)
    expected_sides = {side.value for side, _arm in config.items()}
    if set(drivers) != expected_sides:
        raise CalibrationError("driver sides do not match configured arm sides")

    started_at = monotonic()
    collections = {
        side.value: _ArmCollection(arm, [], None, started_at)
        for side, arm in config.items()
    }
    for side, arm in config.items():
        try:
            drivers[side.value].connect()
        except Exception as exc:
            raise CalibrationError(f"{arm.name}: failed to connect: {exc}") from exc

    connected_at = monotonic()
    for collection in collections.values():
        collection.last_new_frame_at = connected_at

    periods = {side.value: 1.0 / arm.control_hz for side, arm in config.items()}
    scheduled_at = 0.0
    next_reads = {side: 0.0 for side in collections}
    notify(
        f"Keep all arms at reset pose; collecting {ZERO_SAMPLE_COUNT} new frames."
    )
    while any(
        len(collection.zero_samples) < ZERO_SAMPLE_COUNT
        for collection in collections.values()
    ):
        if should_stop():
            raise CalibrationError("calibration stopped before zero sampling completed")
        due = [
            side
            for side, deadline in next_reads.items()
            if deadline <= scheduled_at + 1e-12
        ]
        for side in due:
            collection = collections[side]
            _check_timeout(
                collection,
                monotonic(),
                collection.config.feedback_timeout_s,
            )
            try:
                feedback = drivers[side].read_raw_joint_feedback()
            except Exception as exc:
                if isinstance(exc, CalibrationError):
                    raise
                raise CalibrationError(
                    f"{collection.config.name}: failed to read joint feedback: {exc}"
                ) from exc
            now = monotonic()
            _check_timeout(collection, now, collection.config.feedback_timeout_s)
            joints = _accept_new_frame(collection, feedback, now)
            if joints is not None and len(collection.zero_samples) < ZERO_SAMPLE_COUNT:
                collection.zero_samples.append(joints)
            next_reads[side] += periods[side]
        delay = max(0.0, min(next_reads.values()) - scheduled_at)
        sleep(delay)
        scheduled_at += delay

    for collection in collections.values():
        calibration = calibration_mode_mdeg(collection.zero_samples)
        collection.calibration = calibration
        collection.raw_min = calibration
        collection.raw_max = calibration

    notify("Zero sampling complete; move joints safely, then press Ctrl+C to finish.")
    try:
        while not should_stop():
            due = [
                side
                for side, deadline in next_reads.items()
                if deadline <= scheduled_at + 1e-12
            ]
            for side in due:
                collection = collections[side]
                _check_timeout(
                    collection,
                    monotonic(),
                    collection.config.feedback_timeout_s,
                )
                try:
                    feedback = drivers[side].read_raw_joint_feedback()
                except Exception as exc:
                    raise CalibrationError(
                        f"{collection.config.name}: failed to read joint feedback: {exc}"
                    ) from exc
                now = monotonic()
                _check_timeout(collection, now, collection.config.feedback_timeout_s)
                joints = _accept_new_frame(collection, feedback, now)
                if joints is not None:
                    assert collection.raw_min is not None
                    assert collection.raw_max is not None
                    collection.raw_min = tuple(
                        min(old, new)
                        for old, new in zip(collection.raw_min, joints, strict=True)
                    )  # type: ignore[assignment]
                    collection.raw_max = tuple(
                        max(old, new)
                        for old, new in zip(collection.raw_max, joints, strict=True)
                    )  # type: ignore[assignment]
                    collection.range_frame_count += 1
                next_reads[side] += periods[side]
            delay = max(0.0, min(next_reads.values()) - scheduled_at)
            sleep(delay)
            scheduled_at += delay
    except KeyboardInterrupt:
        pass

    results: dict[str, ArmCalibrationResult] = {}
    for side, arm in config.items():
        collection = collections[side.value]
        assert collection.calibration is not None
        assert collection.raw_min is not None
        assert collection.raw_max is not None
        logical_min = tuple(
            raw - zero
            for raw, zero in zip(
                collection.raw_min, collection.calibration, strict=True
            )
        )
        logical_max = tuple(
            raw - zero
            for raw, zero in zip(
                collection.raw_max, collection.calibration, strict=True
            )
        )
        results[side.value] = ArmCalibrationResult(
            name=arm.name,
            can_port=arm.can_port,
            calibration_mdeg=collection.calibration,
            raw_min_mdeg=collection.raw_min,
            raw_max_mdeg=collection.raw_max,
            min_joints_mdeg=logical_min,  # type: ignore[arg-type]
            max_joints_mdeg=logical_max,  # type: ignore[arg-type]
            range_frame_count=collection.range_frame_count,
        )
    return CalibrationReport(results)


def _degrees(values: JointMdeg) -> list[float]:
    return [round(value / 1000.0, 3) for value in values]


def render_calibration_yaml(report: CalibrationReport) -> str:
    """Render calculated vectors and raw measurement metadata as YAML."""

    data = {
        "arms": {
            side: {
                "name": arm.name,
                "can_port": arm.can_port,
                "calibration_joints_pos_deg": _degrees(arm.calibration_mdeg),
                "min_joints_deg": _degrees(arm.min_joints_mdeg),
                "max_joints_deg": _degrees(arm.max_joints_mdeg),
            }
            for side, arm in report.arms.items()
        },
        "calibration_metadata": {
            side: {
                "name": arm.name,
                "range_frame_count": arm.range_frame_count,
                "raw_min_joints_mdeg": list(arm.raw_min_mdeg),
                "raw_max_joints_mdeg": list(arm.raw_max_mdeg),
            }
            for side, arm in report.arms.items()
        },
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read raw Piper joint angles and calculate calibration YAML"
    )
    parser.add_argument("--config", required=True, help="service YAML configuration")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(Path(args.config))
        drivers = {side.value: PiperArmDriver(arm) for side, arm in config.items()}
        report = run_calibration(
            config,
            drivers,
            progress=lambda message: print(message, file=stderr),
        )
    except KeyboardInterrupt:
        print("piper-calibrate: interrupted before calibration completed", file=stderr)
        return 1
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"piper-calibrate: {exc}", file=stderr)
        return 1
    print(render_calibration_yaml(report), end="", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
