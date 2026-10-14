# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from piper_control import config as config_module
from piper_control.config import (
    ArmSide,
    PiperGripperConfig,
    ServiceConfig,
    load_config,
)


PHYSICAL_MIN = [-150, 0, -170, -100, -70, -120]
PHYSICAL_MAX = [150, 180, 0, 100, 70, 120]


def arm_config(name: str, port: int, can_port: str) -> dict[str, object]:
    return {
        "server": {"host": "127.0.0.1", "port": port},
        "control_hz": 100,
        "feedback_timeout_s": 0.5,
        "homing_timeout_s": 10.0,
        "shutdown_home_timeout_s": 30.0,
        "name": name,
        "can_port": can_port,
        "auto_initialize": True,
        "receive_latency_s": 0.0,
        "calibration_joints_pos_deg": [0, 0, 0, 0, 0, 0],
        "reset_joints_pos_deg": [0, 0, 0, 0, 0, 0],
        "min_joints_deg": PHYSICAL_MIN.copy(),
        "max_joints_deg": PHYSICAL_MAX.copy(),
        "max_joints_speed_deg_s": [30, 30, 30, 30, 30, 30],
        "default_joints_speed_deg_s": [10, 10, 10, 10, 10, 10],
        "gripper": {
            "enabled": True,
            "min_position_m": 0.0,
            "max_position_m": 0.1,
            "max_speed_m_s": 0.05,
            "default_speed_m_s": 0.02,
            "effort_nm": 1.0,
        },
    }


def valid_config() -> dict[str, object]:
    return {
        "arms": {
            "left": arm_config("left_arm", 8765, "can1"),
            "right": arm_config("right_arm", 9876, "can0"),
        },
    }


def unsupported_gripper_config() -> dict[str, object]:
    return {
        "backend": "unsupported_backend",
        "enabled": True,
    }


@pytest.mark.parametrize("side", [ArmSide.LEFT, ArmSide.RIGHT])
def test_existing_gripper_defaults_to_piper_backend(side: ArmSide) -> None:
    config = ServiceConfig.model_validate(valid_config())
    gripper = config.arms[side].gripper
    assert isinstance(gripper, PiperGripperConfig)
    assert gripper.backend == "piper"
    assert gripper.position_unit == "meter"
    assert gripper.speed_unit == "meter_per_second"
    assert gripper.command_min_position == 0.0
    assert gripper.command_max_position == 0.1
    assert gripper.physical_min_position == 0.0
    assert gripper.physical_max_position == 0.1
    assert gripper.default_speed == 0.02
    assert gripper.max_speed == 0.05
    assert gripper.command_lead_limit == 0.005


def test_explicit_piper_backend_is_accepted() -> None:
    data = valid_config()
    data["arms"]["right"]["gripper"]["backend"] = "piper"  # type: ignore[index]
    assert ServiceConfig.model_validate(data).arms[ArmSide.RIGHT].gripper.backend == "piper"


@pytest.mark.parametrize("side", ["left", "right"])
def test_unsupported_backend_is_rejected(side: str) -> None:
    data = valid_config()
    data["arms"][side]["gripper"] = unsupported_gripper_config()  # type: ignore[index]
    with pytest.raises(ValidationError, match="backend"):
        ServiceConfig.model_validate(data)


def test_arm_homing_timeout_is_configured_per_arm() -> None:
    raw = valid_config()
    raw["arms"]["left"]["homing_timeout_s"] = 10.0  # type: ignore[index]
    raw["arms"]["right"]["homing_timeout_s"] = 12.0  # type: ignore[index]

    config = ServiceConfig.model_validate(raw)

    assert config.arms[ArmSide.LEFT].homing_timeout_s == 10.0
    assert config.arms[ArmSide.RIGHT].homing_timeout_s == 12.0


def valid_mujoco_config() -> dict[str, object]:
    return {
        "model_path": "../models/piper.xml",
        "timestep_s": 0.002,
        "realtime": True,
        "viewer": True,
        "gripper_model_max_m": 0.07,
        "bases": {
            "left": {"position_m": [0, 0.2, 0], "rpy_deg": [0, 0, 0]},
            "right": {"position_m": [0, -0.2, 0], "rpy_deg": [0, 0, 0]},
        },
    }


def valid_open_box_payload() -> dict[str, object]:
    return {
        "enabled": True,
        "position_m": [0.38, 0.0, 0.10],
        "yaw_deg": 0.0,
        "size_m": [0.20, 0.10, 0.20],
        "wall_thickness_m": 0.004,
        "mass_kg": 0.30,
        "friction": [1.0, 0.005, 0.0001],
    }


def write_yaml(tmp_path: Path, data: object, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def two_arm_config() -> dict[str, object]:
    """Return the complete dual-arm fixture used by downstream migration tests."""

    return valid_config()


def test_console_scripts_target_service_and_can_entry_points():
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(project_file.read_text(encoding="utf-8"))

    assert metadata["project"]["scripts"] == {
        "piper-control": "piper_control.main:main",
        "piper-control-bridge": "piper_control.bridge.main:main",
        "piper-control-can": "piper_control.can_cli:main",
        "piper-calibrate": "piper_control.calibration_cli:main",
        "camera-bridge": "piper_control.camera_bridge.main:main",
    }


def test_loads_valid_config_into_frozen_typed_models(tmp_path):
    config = load_config(write_yaml(tmp_path, valid_config()))

    assert config.arms[ArmSide.LEFT].server.host == "127.0.0.1"
    assert config.arms[ArmSide.LEFT].server.port == 8765
    assert config.arms[ArmSide.LEFT].control_hz == 100
    assert config.arms[ArmSide.RIGHT].receive_latency_s == 0.0
    assert config.arms[ArmSide.RIGHT].reset_joints_pos_deg == (0, 0, 0, 0, 0, 0)
    with pytest.raises(ValidationError, match="frozen"):
        config.arms[ArmSide.RIGHT].control_hz = 200


def test_each_arm_owns_independent_runtime_parameters(tmp_path):
    raw = valid_config()
    raw["arms"]["left"]["control_hz"] = 50
    raw["arms"]["right"]["control_hz"] = 200
    raw["arms"]["left"]["auto_initialize"] = False
    config = load_config(write_yaml(tmp_path, raw))
    assert config.arms[ArmSide.LEFT].control_hz == 50
    assert config.arms[ArmSide.RIGHT].control_hz == 200
    assert config.arms[ArmSide.LEFT].auto_initialize is False
    assert config.arms[ArmSide.RIGHT].auto_initialize is True


def test_single_left_arm_is_valid(tmp_path):
    raw = valid_config()
    del raw["arms"]["right"]
    assert set(load_config(write_yaml(tmp_path, raw)).arms) == {ArmSide.LEFT}


def test_single_right_arm_is_valid(tmp_path):
    raw = valid_config()
    del raw["arms"]["left"]
    assert set(load_config(write_yaml(tmp_path, raw)).arms) == {ArmSide.RIGHT}


def test_items_are_ordered_right_then_left(tmp_path):
    config = load_config(write_yaml(tmp_path, valid_config()))

    assert tuple(side for side, _arm in config.items()) == (
        ArmSide.RIGHT,
        ArmSide.LEFT,
    )


def test_arms_mapping_is_copied_and_frozen(tmp_path):
    raw = valid_config()
    config = load_config(write_yaml(tmp_path, raw))

    with pytest.raises(TypeError):
        config.arms[ArmSide.LEFT] = config.arms[ArmSide.RIGHT]


def test_receive_latency_defaults_to_zero(tmp_path):
    data = valid_config()
    del data["arms"]["right"]["receive_latency_s"]

    assert (
        load_config(write_yaml(tmp_path, data))
        .arms[ArmSide.RIGHT]
        .receive_latency_s
        == 0.0
    )


def test_reset_is_logical_zero(tmp_path):
    data = valid_config()
    data["arms"]["right"]["reset_joints_pos_deg"][2] = 1

    with pytest.raises(ValueError, match="reset_joints_pos_deg"):
        load_config(write_yaml(tmp_path, data))


def test_requires_at_least_one_arm(tmp_path):
    data = valid_config()
    data["arms"] = {}

    with pytest.raises(ValueError, match="at least one arm"):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("name",), "right_arm", "arm names must be unique"),
        (("can_port",), "can0", "CAN ports must be unique"),
        (("server", "port"), 9876, "TCP ports must be unique"),
    ],
)
def test_two_arms_require_unique_identity_fields(tmp_path, path, value, message):
    data = valid_config()
    if path == ("server", "port"):
        data["arms"]["left"]["server"]["host"] = "0.0.0.0"
    target = data["arms"]["left"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_config(write_yaml(tmp_path, data))


def test_unknown_arm_side_is_rejected(tmp_path):
    data = valid_config()
    data["arms"]["center"] = data["arms"].pop("left")

    with pytest.raises(ValueError, match="center"):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    "field",
    [
        "server",
        "control_hz",
        "feedback_timeout_s",
        "homing_timeout_s",
        "shutdown_home_timeout_s",
    ],
)
def test_legacy_top_level_runtime_fields_show_migration_message(tmp_path, field):
    data = valid_config()
    data[field] = deepcopy(data["arms"]["right"][field])

    with pytest.raises(
        ValueError,
        match=r"move server and runtime settings under arms\.<side>",
    ):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    "field",
    [
        "calibration_joints_pos_deg",
        "reset_joints_pos_deg",
        "min_joints_deg",
        "max_joints_deg",
        "max_joints_speed_deg_s",
        "default_joints_speed_deg_s",
    ],
)
@pytest.mark.parametrize("length", [5, 7])
def test_joint_vectors_have_exactly_six_items(tmp_path, field, length):
    data = valid_config()
    data["arms"]["right"][field] = [0] * length

    with pytest.raises(ValueError, match=field):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("arms", "right", "control_hz"), float("nan")),
        (("arms", "right", "feedback_timeout_s"), float("inf")),
        (("arms", "right", "homing_timeout_s"), float("inf")),
        (("arms", "right", "shutdown_home_timeout_s"), float("-inf")),
        (("arms", "right", "receive_latency_s"), float("nan")),
        (("arms", "right", "calibration_joints_pos_deg", 0), float("inf")),
        (("arms", "right", "reset_joints_pos_deg", 0), float("nan")),
        (("arms", "right", "min_joints_deg", 0), float("-inf")),
        (("arms", "right", "max_joints_deg", 0), float("inf")),
        (("arms", "right", "max_joints_speed_deg_s", 0), float("nan")),
        (("arms", "right", "default_joints_speed_deg_s", 0), float("inf")),
        (("arms", "right", "gripper", "min_position_m"), float("nan")),
        (("arms", "right", "gripper", "max_position_m"), float("inf")),
        (("arms", "right", "gripper", "max_speed_m_s"), float("-inf")),
        (("arms", "right", "gripper", "default_speed_m_s"), float("nan")),
        (("arms", "right", "gripper", "effort_nm"), float("inf")),
    ],
)
def test_numeric_values_must_be_finite(tmp_path, path, value):
    data = valid_config()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="finite"):
        load_config(write_yaml(tmp_path, data))


def test_homing_timeout_must_be_positive(tmp_path) -> None:
    data = valid_config()
    data["arms"]["right"]["homing_timeout_s"] = 0.0

    with pytest.raises(ValueError, match="greater than 0"):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("arms", "right", "control_hz"), True),
        (("arms", "right", "server", "port"), "8765"),
        (("arms", "right", "calibration_joints_pos_deg", 0), True),
        (("arms", "right", "gripper", "effort_nm"), "1.0"),
    ],
)
def test_numeric_values_are_not_coerced(tmp_path, path, value):
    data = valid_config()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("arms", "right", "auto_initialize"), "true"),
        (("arms", "right", "auto_initialize"), 1),
        (("arms", "right", "gripper", "enabled"), "false"),
        (("arms", "right", "gripper", "enabled"), 0),
    ],
)
def test_boolean_values_are_not_coerced(tmp_path, path, value):
    data = valid_config()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(10, 9), (1, -1)],
)
def test_joint_limits_must_not_be_inverted(tmp_path, minimum, maximum):
    data = valid_config()
    data["arms"]["right"]["min_joints_deg"][0] = minimum
    data["arms"]["right"]["max_joints_deg"][0] = maximum

    with pytest.raises(ValueError, match="min_joints_deg.*max_joints_deg"):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize("default_speed", [0, -1, 31])
def test_default_joint_speed_is_positive_and_not_above_max(tmp_path, default_speed):
    data = valid_config()
    data["arms"]["right"]["default_joints_speed_deg_s"][0] = default_speed

    with pytest.raises(ValueError, match="default_joints_speed_deg_s"):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize("max_speed", [0, -1])
def test_max_joint_speed_is_positive(tmp_path, max_speed):
    data = valid_config()
    data["arms"]["right"]["max_joints_speed_deg_s"][0] = max_speed

    with pytest.raises(ValueError, match="max_joints_speed_deg_s"):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_position_m", -0.001),
        ("max_position_m", -0.001),
        ("max_speed_m_s", 0),
        ("max_speed_m_s", -0.01),
        ("default_speed_m_s", 0),
        ("default_speed_m_s", -0.01),
        ("default_speed_m_s", 0.06),
        ("effort_nm", 0),
        ("effort_nm", -1),
    ],
)
def test_gripper_bounds_speeds_and_effort_are_valid(tmp_path, field, value):
    data = valid_config()
    data["arms"]["right"]["gripper"][field] = value

    with pytest.raises(ValueError, match="gripper"):
        load_config(write_yaml(tmp_path, data))


def test_gripper_minimum_cannot_exceed_maximum(tmp_path):
    data = valid_config()
    data["arms"]["right"]["gripper"]["min_position_m"] = 0.11

    with pytest.raises(ValueError, match="min_position_m.*max_position_m"):
        load_config(write_yaml(tmp_path, data))


def test_gripper_requires_nonzero_position_range(tmp_path):
    data = valid_config()
    data["arms"]["right"]["gripper"]["min_position_m"] = 0.1

    with pytest.raises(ValueError, match="min_position_m.*max_position_m"):
        load_config(write_yaml(tmp_path, data))


def test_disabled_gripper_still_requires_complete_valid_configuration(tmp_path):
    data = valid_config()
    data["arms"]["right"]["gripper"]["enabled"] = False
    del data["arms"]["right"]["gripper"]["effort_nm"]

    with pytest.raises(ValueError, match="effort_nm"):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("joint", "calibration"),
    [(0, -1), (1, 1), (2, -1), (3, 1), (4, -1), (5, 1)],
)
def test_calibrated_limits_must_stay_within_piper_physical_limits(
    tmp_path, joint, calibration
):
    data = valid_config()
    data["arms"]["right"]["calibration_joints_pos_deg"][joint] = calibration

    with pytest.raises(ValueError, match="Piper physical joint limits"):
        load_config(write_yaml(tmp_path, data))


def test_unknown_fields_are_rejected_at_every_level(tmp_path):
    data = valid_config()
    data["arms"]["right"]["gripper"]["unexpected"] = True

    with pytest.raises(ValueError, match="unexpected"):
        load_config(write_yaml(tmp_path, data))


def test_yaml_error_includes_file_path(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("arms: [", encoding="utf-8")

    with pytest.raises(ValueError, match=str(path)):
        load_config(path)


def test_example_yaml_is_complete_and_valid():
    example = Path(__file__).parents[1] / "config" / "piper-control.example.yaml"

    config = load_config(example)

    assert config.arms[ArmSide.LEFT].name == "left_arm"
    assert config.arms[ArmSide.RIGHT].gripper.enabled is True


@pytest.mark.parametrize(
    "filename", ["piper-control.example.yaml", "piper-control.mujoco.yaml"]
)
@pytest.mark.parametrize("side", [ArmSide.LEFT, ArmSide.RIGHT])
def test_example_grippers_use_piper_meter_units(filename: str, side: ArmSide) -> None:
    config = load_config(Path(__file__).parents[1] / "config" / filename)
    gripper = config.arms[side].gripper
    assert isinstance(gripper, PiperGripperConfig)
    assert gripper.backend == "piper"
    assert gripper.position_unit == "meter"
    assert gripper.speed_unit == "meter_per_second"


def test_load_config_resolves_mujoco_model_relative_to_yaml(tmp_path):
    data = valid_config()
    data["mujoco"] = valid_mujoco_config()

    config = load_config(write_yaml(tmp_path, data))

    assert config.mujoco is not None
    assert config.mujoco.model_path == (tmp_path / "../models/piper.xml").resolve()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timestep_s", 0),
        ("timestep_s", float("inf")),
        ("gripper_model_max_m", -0.1),
        ("gripper_model_max_m", float("nan")),
    ],
)
def test_mujoco_config_rejects_invalid_scalars(field, value):
    data = valid_config()
    data["mujoco"] = valid_mujoco_config()
    data["mujoco"][field] = value

    with pytest.raises(ValueError):
        ServiceConfig.model_validate(data)


@pytest.mark.parametrize("length", [2, 4])
def test_mujoco_base_vectors_require_exactly_three_components(length):
    data = valid_config()
    data["mujoco"] = valid_mujoco_config()
    data["mujoco"]["bases"]["left"]["position_m"] = [0] * length

    with pytest.raises(ValueError, match="position_m"):
        ServiceConfig.model_validate(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_mujoco_base_vectors_reject_non_finite_components(value):
    data = valid_config()
    data["mujoco"] = valid_mujoco_config()
    data["mujoco"]["bases"]["right"]["rpy_deg"][1] = value

    with pytest.raises(ValueError, match="finite"):
        ServiceConfig.model_validate(data)


@pytest.mark.parametrize("side", ["left", "right"])
def test_mujoco_requires_a_base_for_each_configured_arm(side):
    data = valid_config()
    data["mujoco"] = valid_mujoco_config()
    del data["mujoco"]["bases"][side]

    with pytest.raises(ValueError, match=side):
        ServiceConfig.model_validate(data)


def test_mujoco_bases_mapping_is_copied_and_frozen():
    data = valid_config()
    data["mujoco"] = valid_mujoco_config()

    config = ServiceConfig.model_validate(data)

    assert config.mujoco is not None
    with pytest.raises(TypeError):
        config.mujoco.bases[ArmSide.LEFT] = config.mujoco.bases[ArmSide.RIGHT]


def test_hardware_config_can_omit_mujoco():
    config = ServiceConfig.model_validate(valid_config())

    assert config.mujoco is None


def test_hardware_example_defaults_both_arms_to_twenty_degrees_per_second():
    example = Path(__file__).parents[1] / "config" / "piper-control.example.yaml"

    config = load_config(example)

    assert config.arms[ArmSide.LEFT].default_joints_speed_deg_s == (20.0,) * 6
    assert config.arms[ArmSide.RIGHT].default_joints_speed_deg_s == (20.0,) * 6
    assert config.arms[ArmSide.LEFT].max_joints_speed_deg_s == (30.0,) * 6
    assert config.arms[ArmSide.RIGHT].max_joints_speed_deg_s == (30.0,) * 6


def test_mujoco_optional_dependency_is_declared():
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(project_file.read_text(encoding="utf-8"))

    assert metadata["project"]["optional-dependencies"]["mujoco"] == [
        "mujoco>=3.3,<4"
    ]


def test_camera_dependencies_are_optional_and_version_bounded():
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(project_file.read_text(encoding="utf-8"))

    assert metadata["project"]["optional-dependencies"]["camera"] == [
        "pyzmq>=26,<28",
        "numpy>=1.26,<3",
        "opencv-python-headless>=4.10,<5",
        "pyrealsense2>=2.55,<3",
    ]


def test_mujoco_example_yaml_has_dual_arm_runtime_values():
    example = Path(__file__).parents[1] / "config" / "piper-control.mujoco.yaml"

    config = load_config(example)

    assert config.arms[ArmSide.LEFT].server.port == 8765
    assert config.arms[ArmSide.RIGHT].server.port == 9876
    assert config.arms[ArmSide.LEFT].min_joints_deg == (
        -150,
        0,
        -154,
        -100,
        -69,
        -120,
    )
    assert config.arms[ArmSide.RIGHT].max_joints_deg == (150, 179, 0, 100, 69, 120)
    assert config.arms[ArmSide.LEFT].gripper.max_position_m == 0.10
    assert config.arms[ArmSide.RIGHT].gripper.max_position_m == 0.10
    assert config.mujoco is not None
    assert config.mujoco.timestep_s == 0.002
    assert config.mujoco.gripper_model_max_m == 0.07
    assert config.mujoco.cup.position_m == (0.65, 0.65, 0.065)
    assert config.mujoco.cup.mouth_diameter_m == 0.0878
    assert config.mujoco.cup.height_m == 0.130
    assert config.mujoco.cup.mass_kg == 0.35
    assert config.mujoco.bases[ArmSide.LEFT].position_m == (0.0, 0.263, 0.0)
    assert config.mujoco.bases[ArmSide.RIGHT].position_m == (0.0, -0.263, 0.0)
    assert config.mujoco.model_path == (
        example.parent / "../thirdparty/mujoco_menagerie/agilex_piper/piper.xml"
    ).resolve()


def test_mujoco_open_box_defaults_are_physical() -> None:
    example = Path(__file__).parents[1] / "config" / "piper-control.mujoco.yaml"
    config = load_config(example).mujoco

    assert config is not None
    assert config.open_box.enabled is True
    assert config.open_box.position_m == (0.38, 0.0, 0.10)
    assert config.open_box.yaw_deg == 0.0
    assert config.open_box.size_m == (0.20, 0.10, 0.20)
    assert config.open_box.wall_thickness_m == pytest.approx(0.004)
    assert config.open_box.mass_kg == pytest.approx(0.30)
    assert config.open_box.friction == (1.0, 0.005, 0.0001)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("size_m", (0.20, 0.0, 0.20)),
        ("size_m", (0.20, float("inf"), 0.20)),
        ("position_m", (0.38, float("nan"), 0.10)),
        ("wall_thickness_m", 0.050),
        ("mass_kg", 0.0),
        ("friction", (1.0, -0.001, 0.0001)),
        ("yaw_deg", float("nan")),
    ],
)
def test_mujoco_open_box_rejects_invalid_values(field: str, value: object) -> None:
    payload = valid_open_box_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        config_module.MujocoOpenBoxConfig.model_validate(payload)


def test_mujoco_open_box_rejects_unknown_fields() -> None:
    payload = valid_open_box_payload()
    payload["top_enabled"] = True

    with pytest.raises(ValidationError, match="top_enabled"):
        config_module.MujocoOpenBoxConfig.model_validate(payload)
