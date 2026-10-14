# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Deadline-bounded pure motion planning for Bridge commands."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import copysign, isfinite
from numbers import Real

from piper_control.interpolation import step_axes
from piper_control.types import Joint6


@dataclass(frozen=True, slots=True)
class JointMotionPlan:
    target: Joint6
    speed: Joint6
    deadline: float


@dataclass(frozen=True, slots=True)
class GripperMotionPlan:
    target: float
    speed: float
    deadline: float


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise TypeError(f"{label} must be a finite number")
    return float(value)


def _positive(value: object, label: str) -> float:
    checked = _finite(value, label)
    if checked <= 0:
        raise ValueError(f"{label} must be positive")
    return checked


def _six(values: Sequence[float], label: str) -> Joint6:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must contain exactly six values")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must contain exactly six values") from exc
    if len(items) != 6:
        raise ValueError(f"{label} must contain exactly six values")
    return tuple(_finite(value, f"{label} value") for value in items)  # type: ignore[return-value]


def _reachable(current: float, requested: float, speed: float, remaining: float) -> float:
    delta = requested - current
    distance = min(abs(delta), speed * remaining)
    return current if delta == 0 else current + copysign(distance, delta)


def plan_joint_motion(
    current: Sequence[float],
    requested: Sequence[float],
    requested_speed: Sequence[float],
    maximum_speed: Sequence[float],
    deadline: float,
    now: float,
) -> JointMotionPlan | None:
    current_q = _six(current, "current joints")
    requested_q = _six(requested, "requested joints")
    requested_qd = _six(requested_speed, "requested joint speed")
    maximum_qd = _six(maximum_speed, "maximum joint speed")
    speed = tuple(
        min(
            _positive(requested_value, "requested joint speed"),
            _positive(maximum_value, "maximum joint speed"),
        )
        for requested_value, maximum_value in zip(
            requested_qd, maximum_qd, strict=True
        )
    )
    checked_deadline = _finite(deadline, "deadline")
    checked_now = _finite(now, "now")
    remaining = checked_deadline - checked_now
    if remaining <= 0:
        return None
    target = tuple(
        _reachable(current_value, requested_value, speed_value, remaining)
        for current_value, requested_value, speed_value in zip(
            current_q, requested_q, speed, strict=True
        )
    )
    return JointMotionPlan(
        target=target,  # type: ignore[arg-type]
        speed=speed,  # type: ignore[arg-type]
        deadline=checked_deadline,
    )


def plan_gripper_motion(
    current: float,
    requested: float,
    requested_speed: float,
    maximum_speed: float,
    deadline: float,
    now: float,
) -> GripperMotionPlan | None:
    checked_current = _finite(current, "current gripper position")
    checked_requested = _finite(requested, "requested gripper position")
    speed = min(
        _positive(requested_speed, "requested gripper speed"),
        _positive(maximum_speed, "maximum gripper speed"),
    )
    checked_deadline = _finite(deadline, "deadline")
    checked_now = _finite(now, "now")
    remaining = checked_deadline - checked_now
    if remaining <= 0:
        return None
    return GripperMotionPlan(
        target=_reachable(
            checked_current,
            checked_requested,
            speed,
            remaining,
        ),
        speed=speed,
        deadline=checked_deadline,
    )


def step_joint_motion(
    current: Sequence[float],
    plan: JointMotionPlan,
    *,
    dt: float,
    now: float,
) -> Joint6 | None:
    current_q = _six(current, "current joints")
    step_s = min(
        _positive(dt, "control period"),
        plan.deadline - _finite(now, "now"),
    )
    if step_s <= 0:
        return None
    return step_axes(current_q, plan.target, plan.speed, step_s)  # type: ignore[return-value]


def step_gripper_motion(
    current: float,
    plan: GripperMotionPlan,
    *,
    dt: float,
    now: float,
) -> float | None:
    checked_current = _finite(current, "current gripper position")
    step_s = min(
        _positive(dt, "control period"),
        plan.deadline - _finite(now, "now"),
    )
    if step_s <= 0:
        return None
    return step_axes(
        (checked_current,),
        (plan.target,),
        (plan.speed,),
        step_s,
    )[0]
