# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Composition and validation of the shared dual-arm MuJoCo model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TypeAlias, cast

from .config import ArmConfig, ArmSide, MujocoConfig


JointRange6: TypeAlias = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


@dataclass(frozen=True, slots=True)
class ArmModelBindings:
    joint_qpos_adr: tuple[int, int, int, int, int, int]
    joint_dof_adr: tuple[int, int, int, int, int, int]
    joint_actuator_ids: tuple[int, int, int, int, int, int]
    gripper_qpos_adr: int
    gripper_dof_adr: int
    gripper_actuator_id: int
    wrist_body_id: int
    finger_body_ids: tuple[int, int]
    joint_ranges_rad: JointRange6
    joint_safety_ranges_rad: JointRange6


@dataclass(frozen=True, slots=True)
class OpenBoxModelBindings:
    body_id: int
    freejoint_id: int
    qpos_adr: int
    dof_adr: int
    geom_ids: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class DualArmModel:
    model: object
    bindings: Mapping[ArmSide, ArmModelBindings]
    open_box: OpenBoxModelBindings | None


_SIDES = (ArmSide.LEFT, ArmSide.RIGHT)
_JOINT_NAMES = tuple(f"joint{number}" for number in range(1, 7))
_ACTUATOR_NAMES = (*_JOINT_NAMES, "gripper")
_BODY_NAMES = ("base_link", "link6", "link7", "link8")
_LIMIT_TOLERANCE_RAD = 1e-6
_OPEN_BOX_GEOM_NAMES = (
    "open_box_bottom",
    "open_box_front",
    "open_box_rear",
    "open_box_left",
    "open_box_right",
)


def _rpy_degrees_to_quaternion(rpy_deg: tuple[float, float, float]) -> list[float]:
    roll, pitch, yaw = (math.radians(value) for value in rpy_deg)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _require_unique_element(child: object, side: ArmSide, kind: str, name: str) -> None:
    collection_name = {
        "joint": "joints",
        "actuator": "actuators",
        "body": "bodies",
    }[kind]
    matches = sum(
        element.name == name for element in getattr(child, collection_name)
    )
    if matches != 1:
        raise ValueError(
            f"{side.value} model element {kind} {name!r} must occur exactly once; "
            f"found {matches}"
        )


def _load_child_spec(mujoco: object, path: Path, side: ArmSide) -> object:
    if not path.is_file():
        raise ValueError(
            f"{side.value} model_path does not exist or is not a file: {path}"
        )
    assets_path = path.parent / "assets"
    if not assets_path.is_dir():
        raise ValueError(
            f"{side.value} model assets directory is missing: {assets_path}"
        )
    try:
        child = mujoco.MjSpec.from_file(str(path))
    except Exception as exc:
        raise ValueError(
            f"{side.value} model_path {path} could not load model assets: {exc}"
        ) from exc

    for name in (*_JOINT_NAMES, "joint7", "joint8"):
        _require_unique_element(child, side, "joint", name)
    for name in _ACTUATOR_NAMES:
        _require_unique_element(child, side, "actuator", name)
    for name in _BODY_NAMES:
        _require_unique_element(child, side, "body", name)

    for key in list(child.keys):
        child.delete(key)
    return child


def _element_id(
    mujoco: object,
    model: object,
    side: ArmSide,
    kind: object,
    name: str,
) -> int:
    identifier = int(mujoco.mj_name2id(model, kind, name))
    if identifier < 0:
        raise ValueError(f"{side.value} compiled model element {name!r} is missing")
    return identifier


def _required_id(mujoco: object, model: object, kind: object, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, kind, name))
    if identifier < 0:
        raise ValueError(f"compiled model element {name!r} is missing")
    return identifier


def _add_open_box(parent: object, mujoco: object, config: MujocoConfig) -> None:
    box_config = config.open_box
    if not box_config.enabled:
        return

    length, width, height = box_config.size_m
    thickness = box_config.wall_thickness_m
    wall_height = height - thickness
    wall_center_z = thickness / 2.0
    plate_specs = (
        (
            "open_box_bottom",
            (0.0, 0.0, -height / 2.0 + thickness / 2.0),
            (length, width, thickness),
            (0.58, 0.62, 0.66, 0.72),
        ),
        (
            "open_box_front",
            (-length / 2.0 + thickness / 2.0, 0.0, wall_center_z),
            (thickness, width, wall_height),
            (0.85, 0.12, 0.10, 0.88),
        ),
        (
            "open_box_rear",
            (length / 2.0 - thickness / 2.0, 0.0, wall_center_z),
            (thickness, width, wall_height),
            (0.10, 0.22, 0.85, 0.88),
        ),
        (
            "open_box_left",
            (0.0, width / 2.0 - thickness / 2.0, wall_center_z),
            (length - 2.0 * thickness, thickness, wall_height),
            (0.58, 0.62, 0.66, 0.72),
        ),
        (
            "open_box_right",
            (0.0, -width / 2.0 + thickness / 2.0, wall_center_z),
            (length - 2.0 * thickness, thickness, wall_height),
            (0.58, 0.62, 0.66, 0.72),
        ),
    )
    volumes = tuple(x * y * z for _name, _pos, (x, y, z), _rgba in plate_specs)
    total_volume = sum(volumes)
    box = parent.worldbody.add_body(
        name="open_box",
        pos=list(box_config.position_m),
        quat=_rpy_degrees_to_quaternion((0.0, 0.0, box_config.yaw_deg)),
    )
    box.add_freejoint(name="open_box_freejoint")
    for spec, volume in zip(plate_specs, volumes, strict=True):
        name, position, dimensions, rgba = spec
        box.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(position),
            size=[dimension / 2.0 for dimension in dimensions],
            mass=box_config.mass_kg * volume / total_volume,
            friction=list(box_config.friction),
            rgba=list(rgba),
        )


def _validate_joint_bounds(
    model: object,
    side: ArmSide,
    arm: ArmConfig,
    joint_ids: tuple[int, int, int, int, int, int],
) -> JointRange6:
    ranges: list[tuple[float, float]] = []
    for index, joint_id in enumerate(joint_ids):
        model_min, model_max = (float(value) for value in model.jnt_range[joint_id])
        configured_min = math.radians(arm.min_joints_deg[index])
        configured_max = math.radians(arm.max_joints_deg[index])
        joint_name = _JOINT_NAMES[index]
        if configured_min < model_min - _LIMIT_TOLERANCE_RAD:
            raise ValueError(
                f"{side.value} min_joints_deg for {joint_name} is outside compiled "
                f"model range [{model_min}, {model_max}] rad"
            )
        if configured_max > model_max + _LIMIT_TOLERANCE_RAD:
            raise ValueError(
                f"{side.value} max_joints_deg for {joint_name} is outside compiled "
                f"model range [{model_min}, {model_max}] rad"
            )
        ranges.append((model_min, model_max))
    return cast(JointRange6, tuple(ranges))


def build_dual_arm_model(
    config: MujocoConfig,
    arms: Mapping[ArmSide, ArmConfig],
) -> DualArmModel:
    """Compile two independently prefixed Piper specs into one MuJoCo model."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise RuntimeError(
            "MuJoCo support requires the optional 'mujoco' dependency"
        ) from exc

    for side in _SIDES:
        if side not in arms:
            raise ValueError(f"{side.value} arm configuration is missing")
        if side not in config.bases:
            raise ValueError(f"{side.value} MuJoCo base configuration is missing")

    parent = mujoco.MjSpec()
    parent.modelname = "dual_piper"
    parent.option.timestep = config.timestep_s
    parent.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    parent.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    parent.option.impratio = 10
    parent.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.1],
        rgba=[0.8, 0.8, 0.8, 1.0],
    )
    parent.worldbody.add_light(
        name="world_light",
        pos=[0.0, 0.0, 2.0],
        dir=[0.0, 0.0, -1.0],
    )
    parent.worldbody.add_camera(
        name="overview",
        pos=[1.4, 0.0, 1.0],
        xyaxes=[0.0, 1.0, 0.0, -0.55, 0.0, 0.84],
    )
    if config.cup.enabled:
        cup = parent.worldbody.add_body(
            name="milk_tea_cup",
            pos=list(config.cup.position_m),
        )
        cup.add_freejoint(name="milk_tea_cup_freejoint")
        radius = config.cup.mouth_diameter_m / 2.0
        half_height = config.cup.height_m / 2.0
        cup.add_geom(
            name="milk_tea_cup_collision",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, half_height],
            mass=config.cup.mass_kg,
            friction=[1.0, 0.005, 0.0001],
            rgba=[0.72, 0.43, 0.22, 1.0],
        )
        cup.add_geom(
            name="milk_tea_cup_lid",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, half_height - 0.002],
            size=[radius, 0.002],
            mass=0.0,
            contype=0,
            conaffinity=0,
            rgba=[0.12, 0.09, 0.07, 1.0],
        )

    _add_open_box(parent, mujoco, config)

    model_path = Path(config.model_path)
    for side in _SIDES:
        child = _load_child_spec(mujoco, model_path, side)
        base = config.bases[side]
        frame = parent.worldbody.add_frame(
            name=f"{side.value}_mount",
            pos=list(base.position_m),
            quat=_rpy_degrees_to_quaternion(base.rpy_deg),
        )
        try:
            parent.attach(child, prefix=f"{side.value}_", frame=frame)
        except Exception as exc:
            raise ValueError(
                f"{side.value} model attachment failed for model_path "
                f"{model_path}: {exc}"
            ) from exc

    try:
        model = parent.compile()
    except Exception as exc:
        raise ValueError(f"dual-arm model compilation failed: {exc}") from exc

    bindings: dict[ArmSide, ArmModelBindings] = {}
    for side in _SIDES:
        prefix = f"{side.value}_"
        joint_ids = cast(
            tuple[int, int, int, int, int, int],
            tuple(
                _element_id(
                    mujoco,
                    model,
                    side,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{prefix}{name}",
                )
                for name in _JOINT_NAMES
            ),
        )
        actuator_ids = cast(
            tuple[int, int, int, int, int, int],
            tuple(
                _element_id(
                    mujoco,
                    model,
                    side,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{prefix}{name}",
                )
                for name in _JOINT_NAMES
            ),
        )
        gripper_joint_id = _element_id(
            mujoco,
            model,
            side,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{prefix}joint7",
        )
        gripper_actuator_id = _element_id(
            mujoco,
            model,
            side,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{prefix}gripper",
        )
        model_gripper_max = 2.0 * float(model.jnt_range[gripper_joint_id, 1])
        if config.gripper_model_max_m > model_gripper_max:
            raise ValueError(
                f"{side.value} gripper_model_max_m exceeds twice the positive "
                f"joint7 range ({model_gripper_max} m)"
            )

        bindings[side] = ArmModelBindings(
            joint_qpos_adr=cast(
                tuple[int, int, int, int, int, int],
                tuple(int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids),
            ),
            joint_dof_adr=cast(
                tuple[int, int, int, int, int, int],
                tuple(int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids),
            ),
            joint_actuator_ids=actuator_ids,
            gripper_qpos_adr=int(model.jnt_qposadr[gripper_joint_id]),
            gripper_dof_adr=int(model.jnt_dofadr[gripper_joint_id]),
            gripper_actuator_id=gripper_actuator_id,
            wrist_body_id=_element_id(
                mujoco,
                model,
                side,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{prefix}link6",
            ),
            finger_body_ids=(
                _element_id(
                    mujoco,
                    model,
                    side,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{prefix}link7",
                ),
                _element_id(
                    mujoco,
                    model,
                    side,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{prefix}link8",
                ),
            ),
            joint_ranges_rad=_validate_joint_bounds(
                model, side, arms[side], joint_ids
            ),
            joint_safety_ranges_rad=cast(
                JointRange6,
                tuple(
                    (
                        math.radians(minimum),
                        math.radians(maximum),
                    )
                    for minimum, maximum in zip(
                        arms[side].min_joints_deg,
                        arms[side].max_joints_deg,
                        strict=True,
                    )
                ),
            ),
        )

    open_box: OpenBoxModelBindings | None = None
    if config.open_box.enabled:
        freejoint_id = _required_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "open_box_freejoint",
        )
        open_box = OpenBoxModelBindings(
            body_id=_required_id(
                mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "open_box"
            ),
            freejoint_id=freejoint_id,
            qpos_adr=int(model.jnt_qposadr[freejoint_id]),
            dof_adr=int(model.jnt_dofadr[freejoint_id]),
            geom_ids=cast(
                tuple[int, int, int, int, int],
                tuple(
                    _required_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name)
                    for name in _OPEN_BOX_GEOM_NAMES
                ),
            ),
        )

    return DualArmModel(
        model=model,
        bindings=MappingProxyType(bindings),
        open_box=open_box,
    )
