# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree

import httpx
import numpy as np
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from piper_control.api import create_app
from piper_control.config import ArmConfig, ArmSide
from piper_control.types import ArmStateSnapshot


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "src/piper_control/web/models/piper/piper.urdf"
MODEL_PATH = ROOT / "thirdparty/mujoco_menagerie/agilex_piper/piper.xml"


class RecordingController:
    def snapshot(self) -> ArmStateSnapshot:
        return ArmStateSnapshot(side=ArmSide.LEFT)


def arm_config() -> ArmConfig:
    return ArmConfig.model_validate(
        {
            "server": {"host": "127.0.0.1", "port": 8765},
            "control_hz": 100,
            "feedback_timeout_s": 0.5,
            "shutdown_home_timeout_s": 30,
            "name": "left_arm",
            "can_port": "can1",
            "auto_initialize": False,
            "receive_latency_s": 0.1,
            "calibration_joints_pos_deg": [0] * 6,
            "reset_joints_pos_deg": [0] * 6,
            "min_joints_deg": [-150, 0, -170, -100, -70, -120],
            "max_joints_deg": [150, 180, 0, 100, 70, 120],
            "max_joints_speed_deg_s": [30] * 6,
            "default_joints_speed_deg_s": [10] * 6,
            "gripper": {
                "enabled": True,
                "min_position_m": 0,
                "max_position_m": 0.07,
                "max_speed_m_s": 0.05,
                "default_speed_m_s": 0.02,
                "effort_nm": 1,
            },
        }
    )


async def request(method: str, path: str) -> httpx.Response:
    config = arm_config()
    app = create_app(RecordingController(), ArmSide.LEFT, config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        return await client.request(method, path)


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3) * math.cos(angle)
        + (1 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )


def _transform(
    translation: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    result = np.eye(4)
    if translation is not None:
        result[:3, 3] = translation
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def evaluate_urdf_chain(
    path: Path, joints_deg: tuple[float, ...], target_link: str
) -> np.ndarray:
    root = ElementTree.parse(path).getroot()
    joints_by_child = {
        joint.find("child").attrib["link"]: joint
        for joint in root.findall("joint")
    }
    chain = []
    link = target_link
    while link in joints_by_child:
        joint = joints_by_child[link]
        chain.append(joint)
        link = joint.find("parent").attrib["link"]

    positions = {
        f"joint{index}": math.radians(value)
        for index, value in enumerate(joints_deg, start=1)
    }
    result = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        result = result @ _transform(xyz, _rotation_from_rpy(rpy))

        joint_type = joint.attrib["type"]
        if joint_type in {"revolute", "continuous", "prismatic"}:
            axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            value = positions.get(joint.attrib["name"], 0.0)
            if joint_type == "prismatic":
                result = result @ _transform(axis * value)
            else:
                result = result @ _transform(rotation=_axis_angle(axis, value))
    return result


def evaluate_mujoco_chain(
    path: Path, joints_deg: tuple[float, ...], target_body: str
) -> np.ndarray:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[: len(joints_deg)] = np.radians(joints_deg)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
    return _transform(data.xpos[body_id], data.xmat[body_id].reshape(3, 3))


@pytest.mark.asyncio
async def test_teaching_page_and_local_assets_are_served() -> None:
    page = await request("GET", "/teach")
    css = await request("GET", "/static/teach/teach.css")
    script = await request("GET", "/static/teach/teach.bundle.js")
    urdf = await request("GET", "/static/teach/models/piper/piper.urdf")
    notices = await request(
        "GET", "/static/teach/vendor/THIRD_PARTY_NOTICES.txt"
    )

    assert (
        page.status_code
        == css.status_code
        == script.status_code
        == urdf.status_code
        == notices.status_code
        == 200
    )
    assert 'href="/"' in page.text
    footer = re.search(r'<footer class="company-footer">([^<]+)</footer>', page.text)
    assert footer is not None
    assert footer.group(1) == "Copyright © 2026-2026 atombite.ai"
    assert page.text.index(footer.group(0)) > page.text.index("</main>")
    assert 'id="teach-viewport"' in page.text
    assert 'data-arm-side="left"' in page.text
    assert 'data-server-port="8765"' in page.text
    assert page.text.index('class="control-card status-card"') < page.text.index('id="gripper-controls"')
    assert 'id="gripper-position"' in page.text
    assert 'id="gripper-position-number"' in page.text
    assert 'id="gripper-speed"' in page.text
    assert 'id="gripper-speed-number"' in page.text
    assert "three 0.186.0" in notices.text
    assert "urdf-loader 0.13.1" in notices.text
    assert "THREE-LICENSE.txt" in notices.text
    assert "URDF-LOADER-LICENSE.txt" in notices.text
    assert "PIPER-MODEL-LICENSE.txt" in notices.text

    three_license = await request("GET", "/static/teach/vendor/THREE-LICENSE.txt")
    urdf_license = await request("GET", "/static/teach/vendor/URDF-LOADER-LICENSE.txt")
    model_license = await request("GET", "/static/teach/vendor/PIPER-MODEL-LICENSE.txt")
    assert three_license.status_code == urdf_license.status_code == model_license.status_code == 200
    assert "Copyright © 2010-2026 three.js authors" in three_license.text
    assert "Permission is hereby granted" in three_license.text
    assert "Apache License" in urdf_license.text
    assert "Copyright © 2020 California Institute of Technology" in urdf_license.text
    assert model_license.text == (
        ROOT / "thirdparty/mujoco_menagerie/agilex_piper/LICENSE"
    ).read_text()


def test_teaching_company_footer_is_visible_outside_motion_controls() -> None:
    page = (ROOT / "src/piper_control/web/teach.html").read_text(encoding="utf-8")
    footer = re.search(r'<footer class="company-footer">([^<]+)</footer>', page)
    assert footer is not None
    assert footer.group(1) == "Copyright © 2026-2026 atombite.ai"
    assert page.index("</main>") < page.index(footer.group(0))
    assert page.index(footer.group(0)) < page.index('<footer class="command-bar"')


def test_committed_teaching_bundle_has_no_whitespace_errors() -> None:
    bundle = (ROOT / "src/piper_control/web/teach.bundle.js").read_text()
    assert all(line.rstrip() == line for line in bundle.splitlines())


def test_teaching_gripper_metadata_uses_piper_metres() -> None:
    app = create_app(RecordingController(), ArmSide.LEFT, arm_config())
    route = next(
        route for route in app.routes
        if getattr(route, "path", None) == "/api/base_info"
    )
    response = route.endpoint()

    assert response.status_code == 200
    metadata = json.loads(response.body)
    assert metadata["units"]["gripper_position"] == "meter"
    assert metadata["units"]["gripper_speed"] == "meter/second"
    gripper = metadata["arm"]["gripper"]
    assert gripper["backend"] == "piper"
    assert gripper["enabled"] is True
    assert gripper["position_m"] == {"min": 0, "max": 0.07}
    assert gripper["physical_position_m"] == {"min": 0, "max": 0.07}
    assert gripper["speed_m_s"] == {"default": 0.02, "max": 0.05}


def test_teaching_emergency_stop_is_fixed_at_every_breakpoint() -> None:
    css = (ROOT / "src/piper_control/web/teach.css").read_text()
    rule = css.split(".emergency-stop", 1)[1].split("}", 1)[0]

    assert "position: fixed" in rule
    assert "z-index: 100" in rule


@pytest.mark.asyncio
async def test_dashboard_links_to_teaching_page() -> None:
    response = await request("GET", "/")

    assert response.status_code == 200
    assert 'href="/teach"' in response.text


@pytest.mark.asyncio
async def test_teaching_assets_reject_path_traversal() -> None:
    response = await request("GET", "/static/teach/%2e%2e/api.py")

    assert response.status_code == 404
    assert '"FastAPI REST' not in response.text


def test_nested_teaching_assets_are_in_package_data() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = metadata["tool"]["setuptools"]["package-data"]["piper_control"]

    assert "web/vendor/*" in patterns
    assert "web/models/piper/*" in patterns
    assert "web/models/piper/meshes/*" in patterns


def test_piper_urdf_has_expected_control_chain_and_packaged_meshes() -> None:
    root = ElementTree.parse(URDF_PATH).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    expected_joints = {
        "joint1": (
            "revolute",
            "0 0 0.123",
            "0 0 -1.57080056944",
            "0 0 1",
            "-2.618",
            "2.618",
            "100",
            "1",
        ),
        "joint2": (
            "revolute",
            "0 0 0",
            "1.57080032679 0 -1.57080032679",
            "0 0 1",
            "0",
            "3.14",
            "100",
            "1",
        ),
        "joint3": (
            "revolute",
            "0.28358 0.028726 0",
            "0 0 0.100950101079",
            "0 0 1",
            "-2.697",
            "0",
            "100",
            "1",
        ),
        "joint4": (
            "revolute",
            "-0.24221 0.068514 0",
            "-1.57079940748 0.000000713697318 1.382599809",
            "0 0 1",
            "-1.832",
            "1.832",
            "100",
            "1",
        ),
        "joint5": (
            "revolute",
            "0 0 0",
            "1.57080056944 0 0",
            "0 0 1",
            "-1.22",
            "1.22",
            "100",
            "1",
        ),
        "joint6": (
            "revolute",
            "0 0.091 0.0014165",
            "1.57079208415 0 -3.14159265359",
            "0 0 1",
            "-3.14",
            "3.14",
            "100",
            "1",
        ),
        "joint7": (
            "prismatic",
            "0 0 0.13503",
            "1.57080056944 0 3.1415926",
            "0 0 -1",
            "0",
            "0.035",
            "10",
            "0.05",
        ),
        "joint8": (
            "prismatic",
            "0 0 0.13503",
            "-1.57080056944 0 3.1415926",
            "0 0 1",
            "-0.035",
            "0",
            "10",
            "0.05",
        ),
    }
    assert expected_joints.keys() <= joints.keys()
    for name, metadata in expected_joints.items():
        joint_type, xyz, rpy, axis, lower, upper, effort, velocity = metadata
        joint = joints[name]
        assert joint.attrib["type"] == joint_type
        assert joint.find("origin").attrib == {"xyz": xyz, "rpy": rpy}
        assert joint.find("axis").attrib == {"xyz": axis}
        assert joint.find("limit").attrib == {
            "lower": lower,
            "upper": upper,
            "effort": effort,
            "velocity": velocity,
        }
    assert joints["joint7"].find("mimic") is None
    assert joints["joint8"].find("mimic").attrib == {
        "joint": "joint7",
        "multiplier": "-1",
        "offset": "0",
    }
    tool_joint = joints["tool0_fixed"]
    assert tool_joint.attrib["type"] == "fixed"
    assert tool_joint.find("parent").attrib == {"link": "link6"}
    assert tool_joint.find("child").attrib == {"link": "tool0"}
    assert tool_joint.find("origin").attrib == {
        "xyz": "0 0 0",
        "rpy": "0 0 0",
    }
    for link in root.findall("link"):
        for mesh_container_name in ("visual", "collision"):
            mesh_container = link.find(mesh_container_name)
            if mesh_container is None or mesh_container.find("geometry/mesh") is None:
                continue
            origin = mesh_container.find("origin")
            assert origin is not None, (link.attrib["name"], mesh_container_name)
            expected_rpy = (
                "0 0 1.57080056944"
                if link.attrib["name"] == "link6"
                else "0 0 0"
            )
            assert origin.attrib == {"xyz": "0 0 0", "rpy": expected_rpy}
    for mesh in root.findall(".//mesh"):
        assert (URDF_PATH.parent / mesh.attrib["filename"]).is_file()


@pytest.mark.parametrize(
    "joints_deg",
    [
        (0, 0, 0, 0, 0, 0),
        (0, 90, -77.263, 0, 0, 0),
        (25, 60, -90, 20, -30, 45),
        (-40, 110, -120, -35, 25, -50),
    ],
)
def test_urdf_tool_transform_matches_mujoco(
    joints_deg: tuple[float, ...],
) -> None:
    urdf_transform = evaluate_urdf_chain(URDF_PATH, joints_deg, "tool0")
    mujoco_transform = evaluate_mujoco_chain(MODEL_PATH, joints_deg, "link6")

    assert urdf_transform[:3, 3] == pytest.approx(
        mujoco_transform[:3, 3], abs=1e-5
    )
    assert urdf_transform[:3, :3] == pytest.approx(
        mujoco_transform[:3, :3], abs=1e-5
    )
