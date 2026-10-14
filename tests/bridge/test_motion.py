# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from piper_control.bridge.motion import (
    plan_gripper_motion,
    plan_joint_motion,
    step_gripper_motion,
    step_joint_motion,
)


def test_joint_plan_caps_speed_and_truncates_large_target_to_deadline() -> None:
    plan = plan_joint_motion(
        current=(0.0,) * 6,
        requested=(100.0, -100.0, 4.0, -4.0, 1.0, -1.0),
        requested_speed=(50.0,) * 6,
        maximum_speed=(30.0, 20.0, 10.0, 10.0, 2.0, 2.0),
        deadline=11.0,
        now=10.0,
    )

    assert plan is not None
    assert plan.speed == (30.0, 20.0, 10.0, 10.0, 2.0, 2.0)
    assert plan.target == (30.0, -20.0, 4.0, -4.0, 1.0, -1.0)
    assert plan.deadline == 11.0


def test_small_joint_target_finishes_early_without_overshoot() -> None:
    plan = plan_joint_motion(
        current=(0.0,) * 6,
        requested=(0.5,) * 6,
        requested_speed=(10.0,) * 6,
        maximum_speed=(30.0,) * 6,
        deadline=1.0,
        now=0.0,
    )
    assert plan is not None and plan.target == (0.5,) * 6
    assert step_joint_motion((0.0,) * 6, plan, dt=0.1, now=0.0) == (0.5,) * 6


def test_queue_time_reduces_remaining_joint_travel_budget() -> None:
    plan = plan_joint_motion(
        current=(0.0,) * 6,
        requested=(100.0,) * 6,
        requested_speed=(10.0,) * 6,
        maximum_speed=(30.0,) * 6,
        deadline=1.0,
        now=0.4,
    )
    assert plan is not None
    assert plan.target == pytest.approx((6.0,) * 6)


def test_expired_joint_and_gripper_commands_produce_no_plan_or_tick() -> None:
    assert plan_joint_motion(
        (0.0,) * 6,
        (1.0,) * 6,
        (1.0,) * 6,
        (1.0,) * 6,
        deadline=1.0,
        now=1.0,
    ) is None
    plan = plan_gripper_motion(0.0, 1.0, 1.0, 1.0, deadline=1.0, now=0.0)
    assert plan is not None
    assert step_gripper_motion(0.0, plan, dt=0.1, now=1.0) is None


@pytest.mark.parametrize(
    ("current", "requested", "requested_speed", "maximum_speed", "expected"),
    [
        (0.0, 0.07, 0.05, 0.04, 0.04),
        (0.06, 0.0, 0.05, 0.04, 0.02),
    ],
)
def test_gripper_planner_preserves_piper_metres_in_both_directions(
    current: float,
    requested: float,
    requested_speed: float,
    maximum_speed: float,
    expected: float,
) -> None:
    plan = plan_gripper_motion(
        current,
        requested,
        requested_speed,
        maximum_speed,
        deadline=1.0,
        now=0.0,
    )
    assert plan is not None
    assert plan.speed == maximum_speed
    assert plan.target == pytest.approx(expected)
    assert step_gripper_motion(current, plan, dt=0.5, now=0.0) == pytest.approx(
        (current + expected) / 2
    )


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), True, "1"],
)
def test_planners_reject_nonfinite_or_coerced_values(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        plan_gripper_motion(0.0, 1.0, bad, 1.0, deadline=1.0, now=0.0)
    with pytest.raises((TypeError, ValueError)):
        plan_joint_motion(
            (0.0,) * 5 + (bad,),
            (1.0,) * 6,
            (1.0,) * 6,
            (1.0,) * 6,
            deadline=1.0,
            now=0.0,
        )


def test_planners_require_positive_speeds_and_six_axes() -> None:
    with pytest.raises(ValueError, match="positive"):
        plan_gripper_motion(0.0, 1.0, 0.0, 1.0, deadline=1.0, now=0.0)
    with pytest.raises(ValueError, match="six"):
        plan_joint_motion(
            (0.0,) * 5,
            (1.0,) * 6,
            (1.0,) * 6,
            (1.0,) * 6,
            deadline=1.0,
            now=0.0,
        )
