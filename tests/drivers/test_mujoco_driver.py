# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from piper_control.config import ArmConfig, ArmSide
from piper_control.drivers.mujoco import MujocoArmDriver
from piper_control.mujoco_world import MujocoArmSample
from piper_control.types import ArmFeedback


ZERO6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _config(
    side: ArmSide,
    *,
    gripper_min_m: float = 0.0,
    gripper_max_m: float = 0.10,
) -> ArmConfig:
    return ArmConfig.model_validate(
        {
            "server": {"host": "127.0.0.1", "port": 8765},
            "control_hz": 100,
            "feedback_timeout_s": 0.5,
            "shutdown_home_timeout_s": 1.0,
            "name": f"{side.value}_arm",
            "can_port": "unused",
            "auto_initialize": False,
            "receive_latency_s": 0.0,
            "calibration_joints_pos_deg": ZERO6,
            "reset_joints_pos_deg": ZERO6,
            "min_joints_deg": (-150, 0, -154, -100, -69, -120),
            "max_joints_deg": (150, 179, 0, 100, 69, 120),
            "max_joints_speed_deg_s": (30, 30, 30, 30, 30, 30),
            "default_joints_speed_deg_s": (10, 10, 10, 10, 10, 10),
            "gripper": {
                "enabled": True,
                "min_position_m": gripper_min_m,
                "max_position_m": gripper_max_m,
                "max_speed_m_s": 0.05,
                "default_speed_m_s": 0.02,
                "effort_nm": 1.0,
            },
        }
    )


def _sample(**changes: object) -> MujocoArmSample:
    sample = MujocoArmSample(
        joints_rad=ZERO6,
        gripper_opening_m=0.0,
        tcp_position_m=(0.1, 0.2, 0.3),
        tcp_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        simulation_time_s=1.25,
        received_at_s=500.0,
        connected=True,
        enabled=False,
        emergency_latched=False,
        command_generation=0,
    )
    return replace(sample, **changes)


class FakeWorld:
    def __init__(self) -> None:
        self.failure: Exception | None = None
        self.samples = {
            ArmSide.LEFT: _sample(),
            ArmSide.RIGHT: _sample(),
        }
        self.targets = {
            side: (sample.joints_rad, sample.gripper_opening_m)
            for side, sample in self.samples.items()
        }
        self.emergency_latched = {side: False for side in ArmSide}
        self.enabled_calls: list[tuple[ArmSide, bool]] = []
        self.emergency_calls: list[ArmSide] = []
        self.recover_calls: list[ArmSide] = []
        self.hold_calls: list[ArmSide] = []

    def read_sample(self, side: ArmSide) -> MujocoArmSample:
        return self.samples[side]

    def set_enabled(self, side: ArmSide, enabled: bool) -> None:
        self.enabled_calls.append((side, enabled))
        self.samples[side] = replace(self.samples[side], enabled=enabled)

    def set_target(
        self,
        side: ArmSide,
        joints_rad: tuple[float, ...],
        gripper_opening_m: float,
    ) -> None:
        if self.emergency_latched[side]:
            return
        self.targets[side] = (tuple(joints_rad), gripper_opening_m)

    def set_joint_target(
        self, side: ArmSide, joints_rad: tuple[float, ...]
    ) -> None:
        if self.emergency_latched[side]:
            raise RuntimeError(f"{side.value} MuJoCo emergency stop is latched")
        self.targets[side] = (tuple(joints_rad), self.targets[side][1])

    def set_gripper_target(self, side: ArmSide, gripper_opening_m: float) -> None:
        if self.emergency_latched[side]:
            raise RuntimeError(f"{side.value} MuJoCo emergency stop is latched")
        self.targets[side] = (self.targets[side][0], gripper_opening_m)

    def emergency_stop(self, side: ArmSide) -> None:
        self.emergency_calls.append(side)
        self.emergency_latched[side] = True
        sample = self.samples[side]
        self.targets[side] = (sample.joints_rad, sample.gripper_opening_m)

    def recover(self, side: ArmSide) -> None:
        self.recover_calls.append(side)
        self.emergency_latched[side] = False

    def hold(self, side: ArmSide) -> None:
        self.hold_calls.append(side)
        sample = self.samples[side]
        self.targets[side] = (sample.joints_rad, sample.gripper_opening_m)


def _driver(
    side: ArmSide = ArmSide.LEFT,
    *,
    world: FakeWorld | None = None,
    config: ArmConfig | None = None,
) -> tuple[MujocoArmDriver, FakeWorld]:
    fake = world or FakeWorld()
    return (
        MujocoArmDriver(
            side,
            config or _config(side),
            fake,  # type: ignore[arg-type]
            gripper_model_max_m=0.07,
        ),
        fake,
    )


def test_joint_commands_convert_degrees_to_radians_without_changing_gripper() -> None:
    driver, world = _driver()

    driver.command_joints((0, 90, -90, 10, 20, 30))

    joints, gripper = world.targets[ArmSide.LEFT]
    assert joints == pytest.approx(
        (
            0,
            math.pi / 2,
            -math.pi / 2,
            math.radians(10),
            math.radians(20),
            math.radians(30),
        )
    )
    assert gripper == 0.0


@pytest.mark.parametrize(
    ("api_position_m", "model_opening_m"),
    [(0.0, 0.0), (0.05, 0.035), (0.10, 0.07)],
)
def test_gripper_commands_map_api_endpoints_and_midpoint(
    api_position_m: float, model_opening_m: float
) -> None:
    driver, world = _driver()

    driver.command_gripper(api_position_m)

    assert world.targets[ArmSide.LEFT][1] == pytest.approx(model_opening_m)
    world.samples[ArmSide.LEFT] = replace(
        world.samples[ArmSide.LEFT], gripper_opening_m=model_opening_m
    )
    assert driver.read_feedback().gripper_position == pytest.approx(api_position_m)


def test_gripper_mapping_uses_configured_nonzero_minimum() -> None:
    config = _config(ArmSide.LEFT, gripper_min_m=0.02, gripper_max_m=0.10)
    driver, world = _driver(config=config)

    driver.command_gripper(0.06)

    assert world.targets[ArmSide.LEFT][1] == pytest.approx(0.035)
    world.samples[ArmSide.LEFT] = replace(
        world.samples[ArmSide.LEFT], gripper_opening_m=0.035
    )
    assert driver.read_feedback().gripper_position == pytest.approx(0.06)


@pytest.mark.parametrize(
    ("api_position_m", "expected_m"),
    [(-0.00005, 0.0), (0.10005, 0.10)],
)
def test_gripper_feedback_clamps_small_simulation_overshoot_at_api_boundaries(
    api_position_m: float, expected_m: float
) -> None:
    driver, world = _driver()
    world.samples[ArmSide.LEFT] = replace(
        world.samples[ArmSide.LEFT],
        gripper_opening_m=api_position_m / 0.10 * 0.07,
    )

    assert driver.read_feedback().gripper_position == expected_m


@pytest.mark.parametrize("api_position_m", [-0.001, 0.101])
def test_gripper_feedback_preserves_material_simulation_range_violations(
    api_position_m: float,
) -> None:
    driver, world = _driver()
    world.samples[ArmSide.LEFT] = replace(
        world.samples[ArmSide.LEFT],
        gripper_opening_m=api_position_m / 0.10 * 0.07,
    )

    assert driver.read_feedback().gripper_position == pytest.approx(api_position_m)


def test_feedback_snaps_integrator_noise_to_latest_hold_target() -> None:
    driver, world = _driver()
    held_joints = (5.0, 45.0, -45.0, 5.0, 5.0, 5.0)
    held_gripper = 0.09
    held = ArmFeedback(
        joints_deg=held_joints,
        gripper_position=held_gripper,
    )

    driver.hold(held)
    world.samples[ArmSide.LEFT] = _sample(
        joints_rad=tuple(
            math.radians(value + 0.000005) for value in held_joints
        ),
        gripper_opening_m=(held_gripper + 0.00000005) / 0.10 * 0.07,
    )

    feedback = driver.read_feedback()
    assert feedback.joints_deg == held_joints
    assert feedback.gripper_position == held_gripper


def test_feedback_preserves_material_error_from_latest_hold_target() -> None:
    driver, world = _driver()
    held_joints = (5.0, 45.0, -45.0, 5.0, 5.0, 5.0)
    held_gripper = 0.09
    driver.hold(
        ArmFeedback(
            joints_deg=held_joints,
            gripper_position=held_gripper,
        )
    )
    world.samples[ArmSide.LEFT] = _sample(
        joints_rad=tuple(
            math.radians(value + 0.001) for value in held_joints
        ),
        gripper_opening_m=(held_gripper + 0.00001) / 0.10 * 0.07,
    )

    feedback = driver.read_feedback()
    assert feedback.joints_deg == pytest.approx(
        tuple(value + 0.001 for value in held_joints)
    )
    assert feedback.gripper_position == pytest.approx(held_gripper + 0.00001)


def test_joint_command_after_hold_releases_feedback_snapping() -> None:
    driver, world = _driver()
    held_joints = (5.0, 45.0, -45.0, 5.0, 5.0, 5.0)
    driver.hold(ArmFeedback(joints_deg=held_joints, gripper_position=0.09))

    commanded = (6.0, 45.0, -45.0, 5.0, 5.0, 5.0)
    driver.command_joints(commanded)
    actual = tuple(value + 0.000005 for value in held_joints)
    world.samples[ArmSide.LEFT] = _sample(
        joints_rad=tuple(math.radians(value) for value in actual),
        gripper_opening_m=0.09 / 0.10 * 0.07,
    )

    assert driver.read_feedback().joints_deg == pytest.approx(
        actual, abs=1e-10, rel=0
    )
    assert world.targets[ArmSide.LEFT][0] == pytest.approx(
        tuple(math.radians(value) for value in commanded)
    )


def test_gripper_command_after_hold_releases_feedback_snapping() -> None:
    driver, world = _driver()
    held_gripper = 0.09
    driver.hold(
        ArmFeedback(
            joints_deg=(5.0, 45.0, -45.0, 5.0, 5.0, 5.0),
            gripper_position=held_gripper,
        )
    )

    driver.command_gripper(0.08)
    actual = held_gripper + 0.00000005
    world.samples[ArmSide.LEFT] = _sample(
        gripper_opening_m=actual / 0.10 * 0.07,
    )

    assert driver.read_feedback().gripper_position == pytest.approx(
        actual, abs=1e-12, rel=0
    )
    assert world.targets[ArmSide.LEFT][1] == pytest.approx(0.056)


@pytest.mark.parametrize("joints", [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)])
def test_joint_commands_reject_non_six_axis_shapes(joints: tuple[float, ...]) -> None:
    driver, _ = _driver()

    with pytest.raises(ValueError, match="six"):
        driver.command_joints(joints)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "joints",
    [
        (float("nan"), 0, 0, 0, 0, 0),
        (150.0 + 1.1e-9, 0, 0, 0, 0, 0),
    ],
)
def test_joint_commands_reject_nonfinite_or_materially_out_of_range_values(
    joints: tuple[float, ...],
) -> None:
    driver, _ = _driver()

    with pytest.raises(ValueError):
        driver.command_joints(joints)  # type: ignore[arg-type]


def test_joint_command_clamps_only_one_nanodegree_of_roundoff() -> None:
    driver, world = _driver()

    driver.command_joints((150.0 + 1e-9, 0, 0, 0, 0, 0))

    assert world.targets[ArmSide.LEFT][0][0] == pytest.approx(math.radians(150.0))


@pytest.mark.parametrize("position_m", [float("inf"), -1.1e-9, 0.10 + 1.1e-9])
def test_gripper_commands_reject_nonfinite_or_materially_out_of_range_values(
    position_m: float,
) -> None:
    driver, _ = _driver()

    with pytest.raises(ValueError):
        driver.command_gripper(position_m)


@pytest.mark.parametrize("position_m", [-1e-9, 0.10 + 1e-9])
def test_gripper_command_clamps_only_one_nanometre_of_roundoff(
    position_m: float,
) -> None:
    driver, world = _driver()

    driver.command_gripper(position_m)

    expected = 0.0 if position_m < 0 else 0.07
    assert world.targets[ArmSide.LEFT][1] == pytest.approx(expected)


def test_lifecycle_and_safety_calls_are_forwarded_for_only_the_driver_side() -> None:
    driver, world = _driver(ArmSide.LEFT)

    driver.connect()
    driver.enable()
    driver.emergency_stop()
    driver.recover_emergency()
    driver.hold(ArmFeedback(joints_deg=(10, 20, 30, 40, 50, 60)))

    assert world.enabled_calls == [(ArmSide.LEFT, True)]
    assert world.emergency_calls == [ArmSide.LEFT]
    assert world.recover_calls == [ArmSide.LEFT]
    assert world.hold_calls == [ArmSide.LEFT]


def test_left_and_right_drivers_keep_targets_isolated() -> None:
    world = FakeWorld()
    left, _ = _driver(ArmSide.LEFT, world=world)
    right, _ = _driver(ArmSide.RIGHT, world=world)

    left.command_joints((10, 20, -30, 40, 50, 60))
    right.command_gripper(0.05)

    assert world.targets[ArmSide.LEFT][0] == pytest.approx(
        tuple(math.radians(value) for value in (10, 20, -30, 40, 50, 60))
    )
    assert world.targets[ArmSide.LEFT][1] == 0.0
    assert world.targets[ArmSide.RIGHT][0] == ZERO6
    assert world.targets[ArmSide.RIGHT][1] == pytest.approx(0.035)


def test_rejected_joint_during_emergency_cannot_replay_on_gripper_command() -> None:
    driver, world = _driver()
    driver.command_joints((10, 20, -30, 40, 50, 60))
    driver.emergency_stop()

    with pytest.raises(RuntimeError, match="emergency stop"):
        driver.command_joints((20, 30, -40, 50, 60, 70))

    driver.recover_emergency()
    driver.command_gripper(0.05)
    assert world.targets[ArmSide.LEFT][0] == ZERO6
    assert world.targets[ArmSide.LEFT][1] == pytest.approx(0.035)


def test_rejected_gripper_during_emergency_cannot_replay_on_joint_command() -> None:
    driver, world = _driver()
    driver.command_gripper(0.08)
    driver.emergency_stop()

    with pytest.raises(RuntimeError, match="emergency stop"):
        driver.command_gripper(0.09)

    driver.recover_emergency()
    driver.command_joints((10, 20, -30, 40, 50, 60))
    assert world.targets[ArmSide.LEFT][0] == pytest.approx(
        tuple(math.radians(value) for value in (10, 20, -30, 40, 50, 60))
    )
    assert world.targets[ArmSide.LEFT][1] == 0.0


def test_feedback_converts_radians_quaternion_and_simulation_timestamp() -> None:
    driver, world = _driver()
    # q = Rz(60 deg) * Ry(-30 deg) * Rx(20 deg), in MuJoCo wxyz order.
    world.samples[ArmSide.LEFT] = _sample(
        joints_rad=(
            0,
            math.pi / 2,
            -math.pi / 2,
            math.pi / 18,
            math.pi / 9,
            math.pi / 6,
        ),
        gripper_opening_m=0.035,
        tcp_quat_wxyz=(
            0.801336013243125,
            0.272703032854837,
            -0.136872989216772,
            0.514547795502811,
        ),
    )

    feedback = driver.read_feedback()

    assert feedback.joints_deg == pytest.approx((0, 90, -90, 10, 20, 30))
    assert feedback.tcp_pose == pytest.approx((0.1, 0.2, 0.3, 20, -30, 60))
    assert feedback.received_at_s == 500.0
    assert feedback.robot_timestamp_s == 1.25
    assert all(math.isfinite(value) for value in feedback.tcp_pose)


def test_world_failure_is_exposed_as_stale_fault_feedback() -> None:
    driver, world = _driver()
    world.failure = RuntimeError("physics failed")

    feedback = driver.read_feedback()

    assert feedback.stale is True
    assert feedback.fault_code != 0
    assert feedback.connected is False
    with pytest.raises(RuntimeError, match="physics failed"):
        driver.connect()
