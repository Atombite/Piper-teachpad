# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict configuration models for the Piper control service."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    Strict,
    StrictBool,
    StrictInt,
    model_validator,
)


StrictFiniteFloat = Annotated[FiniteFloat, Strict()]
Vector3 = tuple[StrictFiniteFloat, StrictFiniteFloat, StrictFiniteFloat]
JointVector = tuple[
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
]


class FrozenModel(BaseModel):
    """Base model shared by all configuration sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArmSide(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class MujocoBaseConfig(FrozenModel):
    position_m: Vector3
    rpy_deg: Vector3


class MujocoCupConfig(FrozenModel):
    enabled: StrictBool = True
    position_m: Vector3 = (0.30, 0.0, 0.065)
    mouth_diameter_m: StrictFiniteFloat = Field(default=0.0878, gt=0)
    height_m: StrictFiniteFloat = Field(default=0.130, gt=0)
    mass_kg: StrictFiniteFloat = Field(default=0.35, gt=0)


class MujocoOpenBoxConfig(FrozenModel):
    enabled: StrictBool = True
    position_m: Vector3 = (0.38, 0.0, 0.10)
    yaw_deg: StrictFiniteFloat = 0.0
    size_m: Vector3 = (0.20, 0.10, 0.20)
    wall_thickness_m: StrictFiniteFloat = Field(default=0.004, gt=0)
    mass_kg: StrictFiniteFloat = Field(default=0.30, gt=0)
    friction: Vector3 = (1.0, 0.005, 0.0001)

    @model_validator(mode="after")
    def validate_geometry(self) -> "MujocoOpenBoxConfig":
        if any(value <= 0 for value in self.size_m):
            raise ValueError("open-box size_m values must be positive")
        if self.wall_thickness_m >= min(self.size_m) / 2.0:
            raise ValueError(
                "open-box wall thickness must be less than half every dimension"
            )
        if any(value <= 0 for value in self.friction):
            raise ValueError("open-box friction values must be positive")
        return self


class MujocoConfig(FrozenModel):
    model_path: Path
    timestep_s: StrictFiniteFloat = Field(gt=0)
    realtime: StrictBool
    viewer: StrictBool
    gripper_model_max_m: StrictFiniteFloat = Field(gt=0)
    cup: MujocoCupConfig = Field(default_factory=MujocoCupConfig)
    open_box: MujocoOpenBoxConfig = Field(default_factory=MujocoOpenBoxConfig)
    bases: dict[ArmSide, MujocoBaseConfig]

    @model_validator(mode="after")
    def freeze_bases(self) -> "MujocoConfig":
        object.__setattr__(self, "bases", MappingProxyType(dict(self.bases)))
        return self


class ServerConfig(FrozenModel):
    host: str = Field(min_length=1)
    port: StrictInt = Field(ge=1, le=65535)


class PiperGripperConfig(FrozenModel):
    backend: Literal["piper"] = "piper"
    enabled: StrictBool
    min_position_m: StrictFiniteFloat
    max_position_m: StrictFiniteFloat
    max_speed_m_s: StrictFiniteFloat
    default_speed_m_s: StrictFiniteFloat
    effort_nm: StrictFiniteFloat

    @model_validator(mode="after")
    def validate_bounds(self) -> "PiperGripperConfig":
        if self.min_position_m < 0 or self.max_position_m < 0:
            raise ValueError("gripper positions must be non-negative")
        if self.min_position_m >= self.max_position_m:
            raise ValueError(
                "gripper min_position_m must be less than max_position_m"
            )
        if self.max_speed_m_s <= 0:
            raise ValueError("gripper max_speed_m_s must be positive")
        if not 0 < self.default_speed_m_s <= self.max_speed_m_s:
            raise ValueError(
                "gripper default_speed_m_s must be positive and not exceed "
                "max_speed_m_s"
            )
        if self.effort_nm <= 0:
            raise ValueError("gripper effort_nm must be positive")
        return self

    @property
    def position_unit(self) -> str:
        return "meter"

    @property
    def speed_unit(self) -> str:
        return "meter_per_second"

    @property
    def command_min_position(self) -> float:
        return float(self.min_position_m)

    @property
    def command_max_position(self) -> float:
        return float(self.max_position_m)

    @property
    def physical_min_position(self) -> float:
        return float(self.min_position_m)

    @property
    def physical_max_position(self) -> float:
        return float(self.max_position_m)

    @property
    def default_speed(self) -> float:
        return float(self.default_speed_m_s)

    @property
    def max_speed(self) -> float:
        return float(self.max_speed_m_s)

    @property
    def command_lead_limit(self) -> float:
        return 0.005


GripperConfig = PiperGripperConfig


class ArmConfig(FrozenModel):
    PHYSICAL_MIN_JOINTS_DEG: ClassVar[JointVector] = (
        -150,
        0,
        -170,
        -100,
        -70,
        -120,
    )
    PHYSICAL_MAX_JOINTS_DEG: ClassVar[JointVector] = (
        150,
        180,
        0,
        100,
        70,
        120,
    )

    server: ServerConfig
    control_hz: StrictFiniteFloat = Field(gt=0)
    feedback_timeout_s: StrictFiniteFloat = Field(gt=0)
    homing_timeout_s: StrictFiniteFloat = Field(default=10.0, gt=0)
    shutdown_home_timeout_s: StrictFiniteFloat = Field(gt=0)
    name: str = Field(min_length=1)
    can_port: str = Field(min_length=1)
    auto_initialize: StrictBool
    receive_latency_s: StrictFiniteFloat = Field(default=0.0, ge=0)
    calibration_joints_pos_deg: JointVector
    reset_joints_pos_deg: JointVector
    min_joints_deg: JointVector
    max_joints_deg: JointVector
    max_joints_speed_deg_s: JointVector
    default_joints_speed_deg_s: JointVector
    gripper: PiperGripperConfig

    @model_validator(mode="after")
    def validate_joint_configuration(self) -> "ArmConfig":
        if any(value != 0 for value in self.reset_joints_pos_deg):
            raise ValueError("reset_joints_pos_deg must contain six logical zeros")

        for index, (minimum, maximum) in enumerate(
            zip(self.min_joints_deg, self.max_joints_deg)
        ):
            if minimum > maximum:
                raise ValueError(
                    "min_joints_deg must not exceed max_joints_deg "
                    f"at joint {index + 1}"
                )
            if not minimum <= self.reset_joints_pos_deg[index] <= maximum:
                raise ValueError(
                    "reset_joints_pos_deg must lie between min_joints_deg and "
                    f"max_joints_deg at joint {index + 1}"
                )

        for index, (default_speed, max_speed) in enumerate(
            zip(
                self.default_joints_speed_deg_s,
                self.max_joints_speed_deg_s,
            )
        ):
            if max_speed <= 0:
                raise ValueError(
                    f"max_joints_speed_deg_s must be positive at joint {index + 1}"
                )
            if not 0 < default_speed <= max_speed:
                raise ValueError(
                    "default_joints_speed_deg_s must be positive and not exceed "
                    f"max_joints_speed_deg_s at joint {index + 1}"
                )

        calibrated_limits = zip(
            self.calibration_joints_pos_deg,
            self.min_joints_deg,
            self.max_joints_deg,
            self.PHYSICAL_MIN_JOINTS_DEG,
            self.PHYSICAL_MAX_JOINTS_DEG,
        )
        for index, limits in enumerate(calibrated_limits):
            (
                calibration,
                logical_min,
                logical_max,
                physical_min,
                physical_max,
            ) = limits
            calibrated_min = logical_min + calibration
            calibrated_max = logical_max + calibration
            if calibrated_min < physical_min or calibrated_max > physical_max:
                raise ValueError(
                    "calibrated limits exceed Piper physical joint limits "
                    f"at joint {index + 1}"
                )
        return self


class ServiceConfig(FrozenModel):
    arms: dict[ArmSide, ArmConfig]
    mujoco: MujocoConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_runtime_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and any(
            field in data
            for field in (
                "server",
                "control_hz",
                "feedback_timeout_s",
                "homing_timeout_s",
                "shutdown_home_timeout_s",
            )
        ):
            raise ValueError(
                "legacy configuration must move server and runtime settings "
                "under arms.<side>"
            )
        return data

    @model_validator(mode="after")
    def validate_arm_relationships(self) -> "ServiceConfig":
        if not self.arms:
            raise ValueError("configuration must contain at least one arm")

        names = [arm.name for arm in self.arms.values()]
        if len(names) != len(set(names)):
            raise ValueError("arm names must be unique")
        can_ports = [arm.can_port for arm in self.arms.values()]
        if len(can_ports) != len(set(can_ports)):
            raise ValueError("arm CAN ports must be unique")
        tcp_ports = [arm.server.port for arm in self.arms.values()]
        if len(tcp_ports) != len(set(tcp_ports)):
            raise ValueError("arm TCP ports must be unique")

        if self.mujoco is not None:
            missing_bases = set(self.arms) - set(self.mujoco.bases)
            if missing_bases:
                sides = ", ".join(sorted(side.value for side in missing_bases))
                raise ValueError(
                    f"MuJoCo bases must include every configured arm; missing: {sides}"
                )

        object.__setattr__(self, "arms", MappingProxyType(dict(self.arms)))
        return self

    def items(self) -> tuple[tuple[ArmSide, ArmConfig], ...]:
        return tuple(
            (side, self.arms[side])
            for side in (ArmSide.RIGHT, ArmSide.LEFT)
            if side in self.arms
        )


def load_config(path: Path) -> ServiceConfig:
    """Load and validate a service configuration from a YAML file."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse YAML configuration {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"YAML configuration {path} must contain a mapping")

    config_data = dict(raw)
    mujoco = config_data.get("mujoco")
    if isinstance(mujoco, dict):
        mujoco_data = dict(mujoco)
        model_path = mujoco_data.get("model_path")
        if isinstance(model_path, (str, Path)) and not Path(model_path).is_absolute():
            mujoco_data["model_path"] = (path.parent / model_path).resolve()
        config_data["mujoco"] = mujoco_data

    return ServiceConfig.model_validate(config_data)
