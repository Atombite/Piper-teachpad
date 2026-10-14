# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Validation for external single-arm commands."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .config import ArmConfig
from .types import Joint6


class CommandValidationError(ValueError):
    """Raised when an external command is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class ServoCommand:
    target: Joint6
    speed: Joint6


@dataclass(frozen=True, slots=True)
class GripperCommand:
    target: float
    speed: float


def _finite_vector(
    values: Sequence[float], length: int, label: str
) -> tuple[float, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise CommandValidationError(f"{label} must contain {length} values") from exc
    if len(result) != length:
        raise CommandValidationError(f"{label} must contain exactly {length} values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        for value in result
    ):
        raise CommandValidationError(f"{label} values must be finite numbers")
    return tuple(float(value) for value in result)


def _finite_scalar(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise CommandValidationError(f"{label} must be a finite number")
    return float(value)


def validate_servo(
    arm: ArmConfig,
    target: Sequence[float],
    speed: Sequence[float] | None = None,
) -> ServoCommand:
    """Validate one arm's joint target and optional per-axis speed."""

    checked_target = _finite_vector(target, 6, "joint target")
    checked_speed = (
        tuple(float(value) for value in arm.default_joints_speed_deg_s)
        if speed is None
        else _finite_vector(speed, 6, "joint speed")
    )

    for index, (value, minimum, maximum) in enumerate(
        zip(
            checked_target,
            arm.min_joints_deg,
            arm.max_joints_deg,
            strict=True,
        ),
        start=1,
    ):
        if not minimum <= value <= maximum:
            raise CommandValidationError(
                f"joint {index} target is outside configured range"
            )
    for index, (value, maximum) in enumerate(
        zip(checked_speed, arm.max_joints_speed_deg_s, strict=True), start=1
    ):
        if not 0 < value <= maximum:
            raise CommandValidationError(
                f"joint {index} speed must be positive and not exceed configured maximum"
            )

    return ServoCommand(target=checked_target, speed=checked_speed)


def validate_gripper(
    arm: ArmConfig,
    target: float,
    speed: float | None = None,
) -> GripperCommand:
    """Validate one arm's scalar gripper target and optional speed."""

    if not arm.gripper.enabled:
        raise CommandValidationError("gripper is unavailable")

    checked_target = _finite_scalar(target, "gripper target")
    checked_speed = (
        arm.gripper.default_speed
        if speed is None
        else _finite_scalar(speed, "gripper speed")
    )
    if not (
        arm.gripper.command_min_position
        <= checked_target
        <= arm.gripper.command_max_position
    ):
        raise CommandValidationError("gripper target is outside configured range")
    if not 0 < checked_speed <= arm.gripper.max_speed:
        raise CommandValidationError(
            "gripper speed must be positive and not exceed configured maximum"
        )

    return GripperCommand(target=checked_target, speed=checked_speed)
