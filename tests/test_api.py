# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from piper_control.api import create_app
from piper_control.config import ArmConfig, ArmSide
from piper_control.controller import ArmController, StateConflict
from piper_control.drivers.simulated import SimulatedArmDriver
from piper_control.types import (
    ArmFeedback,
    ArmState,
    ArmStateSnapshot,
    ControlMode,
    ServiceState,
    GripperDiagnostics,
)


def arm_config(side: ArmSide = ArmSide.LEFT, *, port: int = 8765) -> ArmConfig:
    return ArmConfig.model_validate(
        {
            "server": {"host": "127.0.0.1", "port": port},
            "control_hz": 100,
            "feedback_timeout_s": 0.5,
            "shutdown_home_timeout_s": 30,
            "name": f"{side.value}_arm",
            "can_port": "can1" if side is ArmSide.LEFT else "can0",
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


class RecordingController:
    def __init__(self, side: ArmSide = ArmSide.LEFT) -> None:
        self.emergency_calls: list[str] = []
        self.transitions: list[str] = []
        self._snapshot = replace(
            ArmStateSnapshot(side=side),
            state=ServiceState.RUNNING,
            hold=False,
            actual_q=tuple(float(value) for value in range(6)),
            actual_gripper=0.01,
            arm=ArmState(
                receive_timestamp=123.0,
                robot_timestamp=122.9,
                connected=True,
                enabled=True,
                fault_code=0,
                feedback_fresh=True,
            ),
        )

    def snapshot(self) -> ArmStateSnapshot:
        return self._snapshot

    def emergency_stop(self, reason: str) -> None:
        self.emergency_calls.append(reason)

    def request_recovery(self) -> None:
        self.transitions.append("recover")

    def request_homing(self) -> None:
        self.transitions.append("homing")

    def request_hold(self) -> None:
        self.transitions.append("hold")
        self._snapshot = replace(
            self._snapshot,
            state=ServiceState.HOLD,
            moving=False,
            hold=True,
        )

    def change_control_mode(self, mode: ControlMode) -> None:
        if self._snapshot.state is not ServiceState.HOLD:
            raise StateConflict("control mode change requires HOLD state")
        self.transitions.append(f"mode:{mode.value}")
        self._snapshot = replace(
            self._snapshot,
            state=ServiceState.RUNNING,
            control_mode=mode,
            hold=True,
        )


async def request(
    controller: RecordingController,
    method: str,
    path: str,
    *,
    side: ArmSide = ArmSide.LEFT,
    **kwargs,
):
    config = arm_config(side)
    transport = httpx.ASGITransport(app=create_app(controller, side, config))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_base_info_describes_only_the_served_arm() -> None:
    response = await request(RecordingController(), "GET", "/api/base_info")
    data = response.json()

    assert response.status_code == 200
    assert data["protocol_version"] == "3.0"
    assert data["control_mode"] == "local"
    assert data["side"] == "left"
    assert data["arm"]["name"] == "left_arm"
    assert data["arm"]["joint_limits_deg"]["min"] == [
        -150,
        0,
        -170,
        -100,
        -70,
        -120,
    ]
    assert data["config"]["server"]["port"] == 8765
    assert data["supported_commands"] == [
        "hello",
        "servoJ",
        "move_gripper",
        "get_state",
        "get_base_info",
    ]
    assert data["units"]["joint_position"] == "degree"
    assert data["units"]["gripper_position"] == "meter"
    assert isinstance(data["service_version"], str)
    assert isinstance(data["piper_sdk_version"], str)
    assert "arms" not in data
    assert "side_order" not in data


@pytest.mark.asyncio
async def test_base_info_exposes_only_native_piper_metre_metadata() -> None:
    response = await request(RecordingController(), "GET", "/api/base_info")
    data = response.json()
    assert data["units"]["gripper_position"] == "meter"
    assert data["units"]["gripper_speed"] == "meter/second"
    assert data["arm"]["gripper"] == {
        "backend": "piper",
        "enabled": True,
        "position_m": {"min": 0.0, "max": 0.07},
        "physical_position_m": {"min": 0.0, "max": 0.07},
        "speed_m_s": {"default": 0.02, "max": 0.05},
        "effort_nm": 1.0,
    }
    assert data["config"]["gripper"]["backend"] == "piper"


@pytest.mark.parametrize("side", [ArmSide.LEFT, ArmSide.RIGHT])
def test_base_info_native_gripper_bounds_match_the_served_configuration(side: ArmSide) -> None:
    payload = arm_config(side).model_dump()
    payload["gripper"].update({
        "enabled": False,
        "min_position_m": 0.01,
        "max_position_m": 0.06,
        "default_speed_m_s": 0.015,
        "max_speed_m_s": 0.03,
        "effort_nm": 0.5,
    })
    config = ArmConfig.model_validate(payload)
    app = create_app(RecordingController(side), side, config)
    endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", None) == "/api/base_info"
    )
    response = endpoint()
    data = json.loads(response.body)

    assert response.status_code == 200
    assert data["side"] == side.value
    assert data["units"]["gripper_position"] == "meter"
    assert data["units"]["gripper_speed"] == "meter/second"
    assert data["arm"]["gripper"] == {
        "backend": "piper", "enabled": False,
        "position_m": {"min": 0.01, "max": 0.06},
        "physical_position_m": {"min": 0.01, "max": 0.06},
        "speed_m_s": {"default": 0.015, "max": 0.03},
        "effort_nm": 0.5,
    }


@pytest.mark.asyncio
async def test_state_uses_six_axes_and_scalar_gripper() -> None:
    response = await request(RecordingController(), "GET", "/api/state")
    data = response.json()

    assert response.status_code == 200
    assert data["side"] == "left"
    assert len(data["ActualQ"]) == len(data["ActualQd"]) == 6
    assert len(data["ActualTCPPose"]) == len(data["ActualTCPSpeed"]) == 6
    assert isinstance(data["ActualGripper"], float)
    assert isinstance(data["TargetGripper"], float)
    assert data["arm"]["connected"] is True
    assert data["control_ready"] is True
    assert data["control_block_reason"] is None
    assert data["control_mode"] == "local"
    assert data["ws_controller_connected"] is False
    assert data["ws_controller_identity"] is None
    assert set(data["gripper"]) == {
        "backend", "enabled", "communication_ok", "fault_code",
        "warning_code", "feedback_age_s",
    }
    assert "arms" not in data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot_update", "arm_update", "reason"),
    [
        ({"state": ServiceState.HOMING}, {}, "Service is HOMING; RUNNING is required."),
        ({"emergency_stopped": True}, {}, "Emergency stop is active."),
        ({}, {"connected": False}, "Arm is disconnected."),
        ({}, {"enabled": False}, "Arm is not enabled."),
        ({}, {"fault_code": 42}, "Arm fault code 42 is active."),
        ({}, {"feedback_fresh": False}, "Robot feedback is stale."),
        (
            {"state": ServiceState.HOMING, "emergency_stopped": True},
            {"connected": False, "enabled": False, "fault_code": 42, "feedback_fresh": False},
            "Emergency stop is active.",
        ),
    ],
)
async def test_state_exposes_one_fail_closed_control_readiness_verdict(
    snapshot_update: dict[str, object],
    arm_update: dict[str, object],
    reason: str,
) -> None:
    controller = RecordingController()
    controller._snapshot = replace(
        controller._snapshot,
        arm=replace(controller._snapshot.arm, **arm_update),
        **snapshot_update,
    )

    response = await request(controller, "GET", "/api/state")

    assert response.status_code == 200
    assert response.json()["control_ready"] is False
    assert response.json()["control_block_reason"] == reason


@pytest.mark.parametrize(
    "invalid_field",
    ["tcp_pose", "received_at_s"],
)
def test_state_remains_strict_json_after_nonfinite_feedback(
    invalid_field: str,
) -> None:
    config = arm_config()
    valid = ArmFeedback(
        joints_deg=(1, 2, 3, 4, 5, 6),
        gripper_position=0.02,
        tcp_pose=(10, 20, 30, 40, 50, 60),
        received_at_s=10.0,
    )
    driver = SimulatedArmDriver(valid)
    controller = ArmController(ArmSide.LEFT, config, driver)
    controller.tick(dt=0.01)
    replacement = (
        (float("nan"), 20, 30, 40, 50, 60)
        if invalid_field == "tcp_pose"
        else float("inf")
    )
    driver.set_feedback(replace(valid, **{invalid_field: replacement}))

    controller.tick(dt=0.01)
    app = create_app(controller, ArmSide.LEFT, config)
    state_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/state"
    )
    response = state_endpoint()

    assert response.status_code == 200
    payload = json.loads(
        response.body,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    assert payload["state"] == "EMERGENCY_STOP"
    assert payload["ActualTCPPose"] == [10, 20, 30, 40, 50, 60]
    assert payload["robot_receive_timestamp"] == 10.0


@pytest.mark.asyncio
async def test_rest_actions_call_only_the_served_controller() -> None:
    left = RecordingController(ArmSide.LEFT)
    right = RecordingController(ArmSide.RIGHT)

    emergency = await request(left, "GET", "/api/emergency_stop")
    recover = await request(
        left, "POST", "/change_state", json={"state": "recover"}
    )
    homing = await request(
        left, "POST", "/change_state", json={"state": "homing"}
    )
    hold = await request(left, "POST", "/change_state", json={"state": "hold"})
    mode = await request(left, "POST", "/control_mode", json={"mode": "remote"})

    assert (
        emergency.status_code
        == recover.status_code
        == homing.status_code
        == hold.status_code
        == mode.status_code
        == 200
    )
    assert emergency.headers["cache-control"] == "no-store"
    assert left.emergency_calls == ["HTTP emergency stop"]
    assert left.transitions == ["recover", "homing", "hold", "mode:remote"]
    assert right.emergency_calls == []
    assert right.transitions == []


@pytest.mark.asyncio
async def test_control_mode_requires_hold_and_rejects_invalid_body() -> None:
    controller = RecordingController()

    conflict = await request(
        controller, "POST", "/control_mode", json={"mode": "remote"}
    )
    invalid = await request(
        controller, "POST", "/control_mode", json={"mode": "bridge"}
    )
    extra = await request(
        controller,
        "POST",
        "/control_mode",
        json={"mode": "local", "force": True},
    )

    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "STATE_CONFLICT"
    assert invalid.status_code == extra.status_code == 422
    assert controller.snapshot().control_mode is ControlMode.LOCAL


@pytest.mark.asyncio
async def test_change_state_rejects_unknown_or_extra_fields() -> None:
    controller = RecordingController()
    invalid = await request(
        controller, "POST", "/change_state", json={"state": "shutdown"}
    )
    extra = await request(
        controller,
        "POST",
        "/change_state",
        json={"state": "recover", "force": True},
    )

    assert controller.transitions == []
    assert invalid.status_code == extra.status_code == 422
    assert invalid.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_change_state_maps_conflicts_without_traceback() -> None:
    controller = RecordingController()

    def conflict() -> None:
        raise StateConflict("homing requires RUNNING state")

    controller.request_homing = conflict
    response = await request(
        controller, "POST", "/change_state", json={"state": "homing"}
    )

    assert response.status_code == 409
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "STATE_CONFLICT",
            "message": "homing requires RUNNING state",
        },
    }
