# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable value types for Piper control."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

from piper_control.config import ArmSide


Joint6: TypeAlias = tuple[float, float, float, float, float, float]
TcpPose6: TypeAlias = tuple[float, float, float, float, float, float]

ZERO_JOINTS: Joint6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ZERO_TCP_POSE: TcpPose6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class ServiceState(str, Enum):
    """Externally visible controller lifecycle states."""

    STARTING = "STARTING"
    HOMING = "HOMING"
    RUNNING = "RUNNING"
    HOLD = "HOLD"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class ControlMode(str, Enum):
    """Identity class currently allowed to control one arm."""

    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True, slots=True)
class GripperDiagnostics:
    """Backend-neutral gripper health attached to each feedback sample."""

    backend: str = "unknown"
    enabled: bool = False
    communication_ok: bool = False
    fault_code: int = 0
    warning_code: int = 0
    feedback_age_s: float | None = None

    def to_json(self) -> dict[str, str | bool | int | float | None]:
        return {
            "backend": self.backend,
            "enabled": self.enabled,
            "communication_ok": self.communication_ok,
            "fault_code": self.fault_code,
            "warning_code": self.warning_code,
            "feedback_age_s": self.feedback_age_s,
        }


@dataclass(frozen=True, slots=True)
class ArmFeedback:
    """A single arm feedback sample in logical service units."""

    joints_deg: Joint6 = ZERO_JOINTS
    gripper_position: float = 0.0
    gripper: GripperDiagnostics = field(default_factory=GripperDiagnostics)
    tcp_pose: TcpPose6 = ZERO_TCP_POSE
    connected: bool = False
    enabled: bool = False
    fault_code: int = 0
    stale: bool = False
    received_at_s: float = 0.0
    robot_timestamp_s: float | None = None


@dataclass(frozen=True, slots=True)
class RawJointFeedback:
    """One uncalibrated Piper joint frame in native millidegrees."""

    joints_mdeg: tuple[int, int, int, int, int, int]
    connected: bool
    fault_code: int
    stale: bool
    received_at_s: float


@dataclass(frozen=True, slots=True)
class ArmState:
    """Diagnostics retained in a single-arm state snapshot."""

    receive_timestamp: float
    robot_timestamp: float
    connected: bool
    enabled: bool
    fault_code: int
    feedback_fresh: bool

    @property
    def stale(self) -> bool:
        return not self.feedback_fresh

    def to_json(self) -> dict[str, bool | float | int]:
        """Return the public, JSON-native diagnostic representation."""

        return {
            "receive_timestamp": float(self.receive_timestamp),
            "robot_timestamp": float(self.robot_timestamp),
            "connected": self.connected,
            "enabled": self.enabled,
            "fault_code": self.fault_code,
            "feedback_fresh": self.feedback_fresh,
            "stale": self.stale,
        }


def _empty_arm_state() -> ArmState:
    return ArmState(
        receive_timestamp=0.0,
        robot_timestamp=0.0,
        connected=False,
        enabled=False,
        fault_code=0,
        feedback_fresh=False,
    )


@dataclass(frozen=True, slots=True)
class ArmStateSnapshot:
    """Immutable single-arm state published atomically by a controller."""

    side: ArmSide
    control_mode: ControlMode = ControlMode.LOCAL
    state: ServiceState = ServiceState.STARTING
    emergency_stopped: bool = False
    moving: bool = False
    hold: bool = True
    fault_reason: str | None = None
    actual_q: Joint6 = ZERO_JOINTS
    actual_qd: Joint6 = ZERO_JOINTS
    target_q: Joint6 = ZERO_JOINTS
    target_qd: Joint6 = ZERO_JOINTS
    actual_tcp_pose: TcpPose6 = ZERO_TCP_POSE
    actual_tcp_speed: TcpPose6 = ZERO_TCP_POSE
    actual_gripper: float = 0.0
    actual_gripper_speed: float = 0.0
    target_gripper: float = 0.0
    target_gripper_speed: float = 0.0
    robot_receive_timestamp: float = 0.0
    robot_timestamp: float = 0.0
    arm: ArmState = field(default_factory=_empty_arm_state)
    gripper: GripperDiagnostics = field(default_factory=GripperDiagnostics)

    @property
    def control_block_reason(self) -> str | None:
        """Return the canonical fail-closed reason that motion is unavailable."""

        if self.emergency_stopped:
            return "Emergency stop is active."
        if self.state != ServiceState.RUNNING:
            return f"Service is {self.state.value}; RUNNING is required."
        if not self.arm.connected:
            return "Arm is disconnected."
        if not self.arm.enabled:
            return "Arm is not enabled."
        if self.arm.fault_code != 0:
            return f"Arm fault code {self.arm.fault_code} is active."
        if not self.arm.feedback_fresh:
            return "Robot feedback is stale."
        return None

    @property
    def control_ready(self) -> bool:
        """Whether this snapshot is safe and ready to accept motion."""

        return self.control_block_reason is None

    def to_json(self) -> dict[str, Any]:
        """Return the stable public state schema using only JSON-native values."""

        def numbers(values: tuple[float, ...]) -> list[float]:
            return [float(value) for value in values]

        return {
            "side": self.side.value,
            "control_mode": self.control_mode.value,
            "state": self.state.value,
            "emergency_stopped": self.emergency_stopped,
            "moving": self.moving,
            "hold": self.hold,
            "fault_reason": self.fault_reason,
            "control_ready": self.control_ready,
            "control_block_reason": self.control_block_reason,
            "ActualQ": numbers(self.actual_q),
            "ActualQd": numbers(self.actual_qd),
            "TargetQ": numbers(self.target_q),
            "TargetQd": numbers(self.target_qd),
            "ActualTCPPose": numbers(self.actual_tcp_pose),
            "ActualTCPSpeed": numbers(self.actual_tcp_speed),
            "ActualGripper": float(self.actual_gripper),
            "ActualGripperSpeed": float(self.actual_gripper_speed),
            "TargetGripper": float(self.target_gripper),
            "TargetGripperSpeed": float(self.target_gripper_speed),
            "robot_receive_timestamp": float(self.robot_receive_timestamp),
            "robot_timestamp": float(self.robot_timestamp),
            "arm": self.arm.to_json(),
            "gripper": self.gripper.to_json(),
        }
