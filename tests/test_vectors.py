# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from piper_control.config import ArmConfig
from piper_control.vectors import (
    CommandValidationError,
    GripperCommand,
    ServoCommand,
    validate_gripper,
    validate_servo,
)


@pytest.fixture
def arm_config() -> ArmConfig:
    return ArmConfig.model_validate(
        {
            "server": {"host": "127.0.0.1", "port": 8765},
            "control_hz": 100,
            "feedback_timeout_s": 0.5,
            "shutdown_home_timeout_s": 30,
            "name": "left_arm",
            "can_port": "can1",
            "auto_initialize": True,
            "calibration_joints_pos_deg": [0] * 6,
            "reset_joints_pos_deg": [0] * 6,
            "min_joints_deg": [-150, 0, -170, -100, -70, -120],
            "max_joints_deg": [150, 180, 0, 100, 70, 120],
            "max_joints_speed_deg_s": [30, 31, 32, 33, 34, 35],
            "default_joints_speed_deg_s": [10, 11, 12, 13, 14, 15],
            "gripper": {
                "enabled": True,
                "min_position_m": 0.01,
                "max_position_m": 0.07,
                "max_speed_m_s": 0.05,
                "default_speed_m_s": 0.02,
                "effort_nm": 1.0,
            },
        }
    )


def test_validate_servo_applies_arm_defaults(arm_config: ArmConfig) -> None:
    command = validate_servo(arm_config, [0, 1, -1, 2, 3, 4])
    assert command.target == (0, 1, -1, 2, 3, 4)
    assert command.speed == arm_config.default_joints_speed_deg_s


def test_validate_servo_returns_immutable_single_arm_command(
    arm_config: ArmConfig,
) -> None:
    command = validate_servo(
        arm_config,
        [0, 1, -1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6],
    )
    assert command == ServoCommand(
        target=(0, 1, -1, 2, 3, 4),
        speed=(1, 2, 3, 4, 5, 6),
    )
    with pytest.raises(Exception, match="cannot assign"):
        command.target = (0,) * 6


@pytest.mark.parametrize("length", [5, 7])
@pytest.mark.parametrize("field", ["target", "speed"])
def test_validate_servo_requires_exactly_six_values(
    arm_config: ArmConfig, length: int, field: str
) -> None:
    target = [0.0] * (length if field == "target" else 6)
    speed = [1.0] * length if field == "speed" else None
    with pytest.raises(CommandValidationError, match="exactly 6"):
        validate_servo(arm_config, target, speed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
@pytest.mark.parametrize("field", ["target", "speed"])
def test_validate_servo_rejects_nonfinite_and_boolean_values(
    arm_config: ArmConfig, value: float, field: str
) -> None:
    target = [0.0] * 6
    speed = [1.0] * 6
    (target if field == "target" else speed)[0] = value
    with pytest.raises(CommandValidationError, match="finite"):
        validate_servo(arm_config, target, speed)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("direction", [-1, 1])
def test_validate_servo_checks_every_joint_bound(
    arm_config: ArmConfig, index: int, direction: int
) -> None:
    target = [0.0] * 6
    bound = (
        arm_config.min_joints_deg[index]
        if direction < 0
        else arm_config.max_joints_deg[index]
    )
    target[index] = bound + direction
    with pytest.raises(CommandValidationError, match=rf"joint {index + 1}.*range"):
        validate_servo(arm_config, target)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("speed", [0.0, -1.0])
def test_validate_servo_requires_every_speed_to_be_positive(
    arm_config: ArmConfig, index: int, speed: float
) -> None:
    speeds = [1.0] * 6
    speeds[index] = speed
    with pytest.raises(CommandValidationError, match=rf"joint {index + 1}.*speed"):
        validate_servo(arm_config, [0.0] * 6, speeds)


@pytest.mark.parametrize("index", range(6))
def test_validate_servo_checks_every_speed_maximum(
    arm_config: ArmConfig, index: int
) -> None:
    speeds = [1.0] * 6
    speeds[index] = arm_config.max_joints_speed_deg_s[index] + 1
    with pytest.raises(CommandValidationError, match=rf"joint {index + 1}.*speed"):
        validate_servo(arm_config, [0.0] * 6, speeds)


def test_validate_gripper_is_scalar(arm_config: ArmConfig) -> None:
    command = validate_gripper(arm_config, 0.05)
    assert command == GripperCommand(
        target=0.05,
        speed=arm_config.gripper.default_speed_m_s,
    )


def test_validate_gripper_accepts_explicit_speed(arm_config: ArmConfig) -> None:
    assert validate_gripper(arm_config, 0.04, 0.03) == GripperCommand(
        target=0.04,
        speed=0.03,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", float("nan")),
        ("target", float("inf")),
        ("target", float("-inf")),
        ("target", True),
        ("target", "0.02"),
        ("target", None),
        ("speed", float("nan")),
        ("speed", float("inf")),
        ("speed", float("-inf")),
        ("speed", True),
        ("speed", "0.02"),
    ],
)
def test_validate_gripper_rejects_non_numeric_nonfinite_and_boolean_scalars(
    arm_config: ArmConfig, field: str, value: object
) -> None:
    target: object = value if field == "target" else 0.05
    speed: object = value if field == "speed" else 0.02
    with pytest.raises(CommandValidationError, match="finite"):
        validate_gripper(arm_config, target, speed)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [0.009, 0.071])
def test_validate_gripper_checks_configured_range(
    arm_config: ArmConfig, target: float
) -> None:
    with pytest.raises(CommandValidationError, match="range"):
        validate_gripper(arm_config, target)


@pytest.mark.parametrize("speed", [0.0, -0.01, 0.051])
def test_validate_gripper_checks_speed(
    arm_config: ArmConfig, speed: float
) -> None:
    with pytest.raises(CommandValidationError, match="speed"):
        validate_gripper(arm_config, 0.05, speed)


def test_validate_gripper_rejects_disabled_gripper(arm_config: ArmConfig) -> None:
    disabled = arm_config.model_copy(
        update={"gripper": arm_config.gripper.model_copy(update={"enabled": False})}
    )
    with pytest.raises(CommandValidationError, match="unavailable"):
        validate_gripper(disabled, 0.05)
