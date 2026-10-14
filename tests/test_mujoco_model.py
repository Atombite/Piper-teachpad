# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from piper_control.config import ArmConfig, ArmSide, MujocoConfig, load_config
from piper_control.mujoco_model import DualArmModel, build_dual_arm_model


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "config" / "piper-control.mujoco.yaml"
MODEL_PATH = ROOT / "thirdparty" / "mujoco_menagerie" / "agilex_piper" / "piper.xml"


@pytest.fixture
def model_inputs() -> tuple[MujocoConfig, dict[ArmSide, ArmConfig]]:
    service = load_config(CONFIG_PATH)
    assert service.mujoco is not None
    return service.mujoco, dict(service.arms)


def test_builds_prefixed_dual_arm_model_in_one_world(model_inputs):
    config, arms = model_inputs

    dual = build_dual_arm_model(config, arms)

    assert dual.model.nq == 30
    assert dual.model.nu == 14
    assert dual.model.nkey == 0
    assert set(dual.bindings) == {ArmSide.LEFT, ArmSide.RIGHT}
    left = dual.bindings[ArmSide.LEFT]
    right = dual.bindings[ArmSide.RIGHT]
    assert left.joint_qpos_adr != right.joint_qpos_adr
    assert set(left.joint_actuator_ids).isdisjoint(right.joint_actuator_ids)

    for side in ArmSide:
        prefix = f"{side.value}_"
        binding = dual.bindings[side]
        assert tuple(
            dual.model.joint(f"{prefix}joint{number}").qposadr[0]
            for number in range(1, 7)
        ) == binding.joint_qpos_adr
        assert tuple(
            dual.model.joint(f"{prefix}joint{number}").dofadr[0]
            for number in range(1, 7)
        ) == binding.joint_dof_adr
        assert tuple(
            dual.model.actuator(f"{prefix}joint{number}").id
            for number in range(1, 7)
        ) == binding.joint_actuator_ids
        assert dual.model.actuator(f"{prefix}gripper").id == binding.gripper_actuator_id
        assert (
            dual.model.joint(f"{prefix}joint7").qposadr[0]
            == binding.gripper_qpos_adr
        )
        assert (
            dual.model.joint(f"{prefix}joint7").dofadr[0]
            == binding.gripper_dof_adr
        )
        assert dual.model.body(f"{prefix}link6").id == binding.wrist_body_id
        assert (
            dual.model.body(f"{prefix}link7").id,
            dual.model.body(f"{prefix}link8").id,
        ) == binding.finger_body_ids
        for bound, number in zip(binding.joint_ranges_rad, range(1, 7)):
            assert bound == pytest.approx(
                dual.model.joint(f"{prefix}joint{number}").range
            )
        for bound, minimum, maximum in zip(
            binding.joint_safety_ranges_rad,
            arms[side].min_joints_deg,
            arms[side].max_joints_deg,
            strict=True,
        ):
            assert bound == pytest.approx(
                (math.radians(minimum), math.radians(maximum))
            )

    assert left.wrist_body_id != right.wrist_body_id
    assert set(left.finger_body_ids).isdisjoint(right.finger_body_ids)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        left.gripper_qpos_adr = 999
    with pytest.raises(TypeError):
        dual.bindings[ArmSide.LEFT] = right


def test_applies_base_poses_in_world_coordinates(model_inputs):
    config, arms = model_inputs
    dual = build_dual_arm_model(config, arms)
    data = mujoco.MjData(dual.model)

    mujoco.mj_forward(dual.model, data)

    left_base = data.body("left_base_link")
    right_base = data.body("right_base_link")
    assert left_base.xpos.tolist() == pytest.approx([0.0, 0.263, 0.0])
    assert right_base.xpos.tolist() == pytest.approx([0.0, -0.263, 0.0])
    assert left_base.xquat.tolist() == pytest.approx(right_base.xquat.tolist())
    assert math.dist(left_base.xpos, right_base.xpos) == pytest.approx(0.526)


def test_adds_configured_dynamic_milk_tea_cup(model_inputs):
    config, arms = model_inputs

    dual = build_dual_arm_model(config, arms)

    cup = dual.model.body("milk_tea_cup")
    geom = dual.model.geom("milk_tea_cup_collision")
    freejoint = dual.model.joint("milk_tea_cup_freejoint")
    assert cup.pos.tolist() == pytest.approx([0.65, 0.65, 0.065])
    assert cup.mass[0] == pytest.approx(0.35)
    assert geom.size.tolist() == pytest.approx([0.0878 / 2, 0.130 / 2, 0.0])
    assert freejoint.type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_open_box_is_a_five_geom_dynamic_body(model_inputs) -> None:
    config, arms = model_inputs

    dual: DualArmModel = build_dual_arm_model(config, arms)

    assert dual.open_box is not None
    box = dual.open_box
    names = {
        mujoco.mj_id2name(dual.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in box.geom_ids
    }
    assert names == {
        "open_box_bottom",
        "open_box_front",
        "open_box_rear",
        "open_box_left",
        "open_box_right",
    }
    assert mujoco.mj_name2id(
        dual.model, mujoco.mjtObj.mjOBJ_GEOM, "open_box_top"
    ) == -1
    assert dual.model.body("open_box").id == box.body_id
    assert dual.model.joint("open_box_freejoint").id == box.freejoint_id
    assert dual.model.jnt_qposadr[box.freejoint_id] == box.qpos_adr
    assert dual.model.jnt_dofadr[box.freejoint_id] == box.dof_adr
    assert dual.model.body_mass[box.body_id] == pytest.approx(0.30)
    assert dual.model.body_pos[box.body_id] == pytest.approx([0.38, 0.0, 0.10])

    local_min = [float("inf")] * 3
    local_max = [float("-inf")] * 3
    for geom_id in box.geom_ids:
        assert dual.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        for axis in range(3):
            local_min[axis] = min(
                local_min[axis],
                float(dual.model.geom_pos[geom_id, axis] - dual.model.geom_size[geom_id, axis]),
            )
            local_max[axis] = max(
                local_max[axis],
                float(dual.model.geom_pos[geom_id, axis] + dual.model.geom_size[geom_id, axis]),
            )
    assert local_min == pytest.approx([-0.10, -0.05, -0.10])
    assert local_max == pytest.approx([0.10, 0.05, 0.10])

    front = dual.model.geom("open_box_front")
    rear = dual.model.geom("open_box_rear")
    assert front.pos[0] == pytest.approx(-0.098)
    assert rear.pos[0] == pytest.approx(0.098)
    assert front.rgba.tolist() == pytest.approx([0.85, 0.12, 0.10, 0.88])
    assert rear.rgba.tolist() == pytest.approx([0.10, 0.22, 0.85, 0.88])


def test_disabled_open_box_has_no_body_or_bindings(model_inputs) -> None:
    config, arms = model_inputs
    disabled = config.model_copy(
        update={"open_box": config.open_box.model_copy(update={"enabled": False})}
    )

    dual = build_dual_arm_model(disabled, arms)

    assert dual.open_box is None
    assert mujoco.mj_name2id(
        dual.model, mujoco.mjtObj.mjOBJ_BODY, "open_box"
    ) == -1


def test_missing_model_file_names_side_and_model_path(model_inputs, tmp_path):
    config, arms = model_inputs
    missing = tmp_path / "missing-piper.xml"

    with pytest.raises(ValueError, match=r"left.*model_path.*missing-piper\.xml"):
        build_dual_arm_model(config.model_copy(update={"model_path": missing}), arms)


def test_missing_submodule_assets_names_side_and_assets(model_inputs, tmp_path):
    config, arms = model_inputs
    xml_only = tmp_path / "piper.xml"
    shutil.copyfile(MODEL_PATH, xml_only)

    with pytest.raises(ValueError, match=r"left.*assets"):
        build_dual_arm_model(config.model_copy(update={"model_path": xml_only}), arms)


@pytest.mark.parametrize(
    ("old", "new", "element"),
    [
        ('<joint name="joint6"', '<joint name="renamed_joint6"', "joint6"),
        ('<position name="joint6"', '<position name="renamed_actuator6"', "joint6"),
    ],
)
def test_renamed_model_element_names_side_and_element(
    model_inputs, tmp_path, old, new, element
):
    config, arms = model_inputs
    copied_model = tmp_path / "model"
    shutil.copytree(MODEL_PATH.parent, copied_model)
    xml_path = copied_model / "piper.xml"
    xml = xml_path.read_text(encoding="utf-8")
    assert old in xml
    xml_path.write_text(xml.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"left.*{element}"):
        build_dual_arm_model(config.model_copy(update={"model_path": xml_path}), arms)


@pytest.mark.parametrize(
    ("margin_rad", "should_pass"),
    [(0.5e-6, True), (2e-6, False)],
)
def test_yaml_joint_bound_uses_one_microradian_margin(
    model_inputs, margin_rad, should_pass
):
    config, arms = model_inputs
    left = arms[ArmSide.LEFT]
    maxima = list(left.max_joints_deg)
    maxima[1] = math.degrees(3.14 + margin_rad)
    arms[ArmSide.LEFT] = left.model_copy(update={"max_joints_deg": tuple(maxima)})

    if should_pass:
        build_dual_arm_model(config, arms)
    else:
        with pytest.raises(
            ValueError, match=r"left.*max_joints_deg.*joint2"
        ):
            build_dual_arm_model(config, arms)


def test_yaml_joint_minimum_outside_model_names_side_field_and_joint(model_inputs):
    config, arms = model_inputs
    right = arms[ArmSide.RIGHT]
    minima = list(right.min_joints_deg)
    minima[1] = math.degrees(-2e-6)
    arms[ArmSide.RIGHT] = right.model_copy(update={"min_joints_deg": tuple(minima)})

    with pytest.raises(ValueError, match=r"right.*min_joints_deg.*joint2"):
        build_dual_arm_model(config, arms)


def test_gripper_opening_above_model_travel_names_side_and_field(model_inputs):
    config, arms = model_inputs
    invalid = config.model_copy(update={"gripper_model_max_m": 0.070001})

    with pytest.raises(ValueError, match=r"left.*gripper_model_max_m.*joint7"):
        build_dual_arm_model(invalid, arms)
