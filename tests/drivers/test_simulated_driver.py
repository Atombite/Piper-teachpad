# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from piper_control.drivers.simulated import SimulatedArmDriver
from piper_control.types import ArmFeedback


def test_simulator_records_commands_and_updates_feedback() -> None:
    driver = SimulatedArmDriver()
    driver.connect()
    driver.enable()
    driver.command_joints((1, 2, 3, 4, 5, 6))
    driver.command_gripper(0.025)
    feedback = driver.read_feedback()
    assert feedback.joints_deg == (1, 2, 3, 4, 5, 6)
    assert feedback.gripper_position == 0.025
    assert feedback.connected and feedback.enabled
    assert driver.joint_commands == [(1, 2, 3, 4, 5, 6)]
    assert driver.gripper_commands == [0.025]


def test_simulator_safety_calls_and_hold_are_deterministic() -> None:
    driver = SimulatedArmDriver()
    driver.set_feedback(
        ArmFeedback(joints_deg=(6, 5, 4, 3, 2, 1), enabled=True, connected=True)
    )
    driver.emergency_stop()
    driver.recover_emergency()
    driver.hold(driver.read_feedback())
    assert driver.emergency_stop_calls == 1
    assert driver.recover_emergency_calls == 1
    assert driver.hold_calls == 1
    assert driver.joint_commands == [(6, 5, 4, 3, 2, 1)]
    assert driver.read_feedback().enabled


def test_simulator_rejects_motion_while_emergency_is_latched() -> None:
    driver = SimulatedArmDriver()
    driver.connect()
    driver.enable()
    before = driver.read_feedback()

    driver.emergency_stop()
    with pytest.raises(RuntimeError, match="emergency"):
        driver.command_joints((1, 2, 3, 4, 5, 6))
    with pytest.raises(RuntimeError, match="emergency"):
        driver.command_gripper(0.02)
    driver.hold(before)

    assert driver.read_feedback().joints_deg == before.joints_deg
    assert driver.hold_calls == 1
    assert driver.joint_commands == [before.joints_deg]

    driver.recover_emergency()
    driver.command_joints((1, 2, 3, 4, 5, 6))
    assert driver.read_feedback().joints_deg == (1, 2, 3, 4, 5, 6)


def test_simulator_fault_enable_and_stale_hooks_replace_frozen_feedback() -> None:
    driver = SimulatedArmDriver()
    before = driver.read_feedback()
    driver.set_fault(7)
    driver.set_enabled(False)
    driver.set_feedback_stale(True)
    after = driver.read_feedback()
    assert after.fault_code == 7
    assert not after.enabled
    assert after.stale
    assert before.fault_code == 0 and not before.stale
    with pytest.raises(FrozenInstanceError):
        after.stale = False  # type: ignore[misc]


def test_simulator_exposes_no_power_off_or_reset_operations() -> None:
    driver = SimulatedArmDriver()
    assert not hasattr(driver, "disable")
    assert not hasattr(driver, "reset")
    assert not hasattr(driver, "DisableArm")
    assert not hasattr(driver, "ResetPiper")


def test_simulator_keeps_explicit_feedback_timestamp_by_default() -> None:
    driver = SimulatedArmDriver(ArmFeedback(received_at_s=7.0))

    assert driver.read_feedback().received_at_s == 7.0
    assert driver.read_feedback().received_at_s == 7.0


def test_live_simulator_feedback_timestamp_is_strictly_increasing() -> None:
    clock_values = iter((10.0, 10.0, 9.0))
    driver = SimulatedArmDriver(
        ArmFeedback(received_at_s=20.0),
        auto_timestamp=True,
        clock=lambda: next(clock_values),
    )

    timestamps = [driver.read_feedback().received_at_s for _ in range(3)]

    assert timestamps[0] > 20.0
    assert timestamps[0] < timestamps[1] < timestamps[2]
