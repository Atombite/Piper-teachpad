# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-arm adapter between the controller API and a shared MuJoCo world."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from piper_control.config import ArmConfig, ArmSide, PiperGripperConfig
from piper_control.types import ArmFeedback, GripperDiagnostics, Joint6, TcpPose6

if TYPE_CHECKING:
    from piper_control.mujoco_world import MujocoWorld


_RANGE_TOLERANCE = 1e-9
_GRIPPER_FEEDBACK_TOLERANCE_M = 1e-4
_HOLD_JOINT_NOISE_TOLERANCE_DEG = 1e-5
_HOLD_GRIPPER_NOISE_TOLERANCE_M = 1e-7
_WORLD_FAULT_CODE = 1


class MujocoArmDriver:
    """Expose one side of a shared :class:`MujocoWorld` as an arm driver."""

    def __init__(
        self,
        side: ArmSide,
        config: ArmConfig,
        world: MujocoWorld,
        *,
        gripper_model_max_m: float,
    ) -> None:
        model_max = float(gripper_model_max_m)
        if not math.isfinite(model_max) or model_max <= 0.0:
            raise ValueError("gripper_model_max_m must be a positive finite number")
        self._side = side
        self._config = config
        if not isinstance(config.gripper, PiperGripperConfig):
            raise ValueError("MuJoCo driver requires a Piper gripper backend")
        self._gripper_config = config.gripper
        self._world = world
        self._gripper_model_max_m = model_max
        self._held_joints_deg: Joint6 | None = None
        self._held_gripper_m: float | None = None

    def connect(self) -> None:
        self._raise_world_failure()

    def enable(self) -> None:
        self._raise_world_failure()
        self._world.set_enabled(self._side, True)

    def read_feedback(self) -> ArmFeedback:
        sample = self._world.read_sample(self._side)
        roll, pitch, yaw = _quaternion_to_rpy_deg(sample.tcp_quat_wxyz)
        failure = self._world.failure
        joints_deg = cast(
            Joint6,
            tuple(math.degrees(value) for value in sample.joints_rad),
        )
        held_joints = self._held_joints_deg
        if held_joints is not None and all(
            abs(actual - target) <= _HOLD_JOINT_NOISE_TOLERANCE_DEG
            for actual, target in zip(joints_deg, held_joints, strict=True)
        ):
            joints_deg = held_joints
        gripper_position = self._model_to_api_gripper(
            sample.gripper_opening_m
        )
        held_gripper = self._held_gripper_m
        if (
            held_gripper is not None
            and abs(gripper_position - held_gripper)
            <= _HOLD_GRIPPER_NOISE_TOLERANCE_M
        ):
            gripper_position = held_gripper
        return ArmFeedback(
            joints_deg=joints_deg,
            gripper_position=gripper_position,
            gripper=GripperDiagnostics(
                backend="piper",
                enabled=sample.enabled and self._gripper_config.enabled,
                communication_ok=failure is None,
                fault_code=_WORLD_FAULT_CODE if failure is not None else 0,
                warning_code=0,
                feedback_age_s=0.0,
            ),
            tcp_pose=cast(
                TcpPose6,
                (*sample.tcp_position_m, roll, pitch, yaw),
            ),
            connected=sample.connected and failure is None,
            enabled=sample.enabled,
            fault_code=_WORLD_FAULT_CODE if failure is not None else 0,
            stale=failure is not None,
            received_at_s=sample.received_at_s,
            robot_timestamp_s=sample.simulation_time_s,
        )

    def command_joints(self, joints_deg: Joint6) -> None:
        values = self._validated_joints(joints_deg)
        joints_rad = cast(Joint6, tuple(math.radians(value) for value in values))
        self._raise_world_failure()
        self._world.set_joint_target(self._side, joints_rad)
        held = self._held_joints_deg
        if held is not None and any(
            abs(value - target) > _HOLD_JOINT_NOISE_TOLERANCE_DEG
            for value, target in zip(values, held, strict=True)
        ):
            self._held_joints_deg = None

    def command_gripper(self, position: float) -> None:
        if not self._gripper_config.enabled:
            raise ValueError("gripper is disabled")
        position = _bounded_value(
            position,
            self._gripper_config.min_position_m,
            self._gripper_config.max_position_m,
            "gripper position",
        )
        opening = self._api_to_model_gripper(position)
        self._raise_world_failure()
        self._world.set_gripper_target(self._side, opening)
        held = self._held_gripper_m
        if (
            held is not None
            and abs(position - held) > _HOLD_GRIPPER_NOISE_TOLERANCE_M
        ):
            self._held_gripper_m = None

    def emergency_stop(self) -> None:
        self._world.emergency_stop(self._side)

    def recover_emergency(self) -> None:
        self._raise_world_failure()
        self._world.recover(self._side)

    def hold(self, feedback: ArmFeedback) -> None:
        self._world.hold(self._side)
        self._held_joints_deg = feedback.joints_deg
        self._held_gripper_m = feedback.gripper_position

    def shutdown(self) -> None:
        """The shared world owns its own process-level lifetime."""

    def _validated_joints(self, joints_deg: Joint6) -> Joint6:
        if len(joints_deg) != 6:
            raise ValueError("joint command must contain exactly six values")
        values = tuple(float(value) for value in joints_deg)
        bounded = tuple(
            _bounded_value(value, minimum, maximum, f"joint {index}")
            for index, (value, minimum, maximum) in enumerate(
                zip(
                    values,
                    self._config.min_joints_deg,
                    self._config.max_joints_deg,
                    strict=True,
                ),
                1,
            )
        )
        return cast(Joint6, bounded)

    def _api_to_model_gripper(self, position_m: float) -> float:
        minimum = self._gripper_config.min_position_m
        span = self._gripper_config.max_position_m - minimum
        return (position_m - minimum) / span * self._gripper_model_max_m

    def _model_to_api_gripper(self, opening_m: float) -> float:
        minimum = self._gripper_config.min_position_m
        maximum = self._gripper_config.max_position_m
        span = maximum - minimum
        position = minimum + float(opening_m) / self._gripper_model_max_m * span
        if minimum - _GRIPPER_FEEDBACK_TOLERANCE_M <= position < minimum:
            return minimum
        if maximum < position <= maximum + _GRIPPER_FEEDBACK_TOLERANCE_M:
            return maximum
        return position

    def _raise_world_failure(self) -> None:
        failure = self._world.failure
        if failure is not None:
            raise RuntimeError(f"MuJoCo world failed: {failure}") from failure


def _bounded_value(value: float, minimum: float, maximum: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if number < minimum - _RANGE_TOLERANCE or number > maximum + _RANGE_TOLERANCE:
        raise ValueError(f"{label} is outside the configured range")
    return min(max(number, minimum), maximum)


def _quaternion_to_rpy_deg(
    quaternion_wxyz: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    """Convert MuJoCo's wxyz quaternion to XYZ roll, pitch, yaw degrees."""

    w, x, y, z = quaternion_wxyz
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(min(1.0, max(-1.0, sin_pitch)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
