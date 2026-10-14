# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from piper_control.bridge.upstream import (
    PiperControlClient,
    UpstreamSnapshot,
    probe_control_mode,
)
from piper_control.config import ArmSide
from piper_control.protocol import ProtocolError


class Socket:
    def __init__(self) -> None:
        self.sent = []
        self.closed = False

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self, **_kwargs) -> str:
        request = self.sent[-1]
        data = {"identity": "bridge"} if request["cmd"] == "hello" else {"accepted": True}
        return json.dumps({"id": request["id"], "cmd": request["cmd"], "ok": True, "data": data})

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_client_hello_is_first_and_serializes_motion_requests() -> None:
    socket = Socket()

    async def connect(_url: str):
        return socket

    async with PiperControlClient("ws://controller/ws/control", connector=connect) as client:
        await client.servo_j((0.0,) * 6, (1.0,) * 6)
        await client.move_gripper(0.2, 0.1)
    assert socket.sent[0]["cmd"] == "hello"
    assert socket.sent[0]["identity"] == "bridge"
    assert [item["cmd"] for item in socket.sent] == ["hello", "servoJ", "move_gripper"]
    assert socket.closed


@pytest.mark.asyncio
async def test_control_mode_probe_uses_read_only_base_info_endpoint() -> None:
    captured: dict[str, object] = {}

    async def requester(url: str, timeout: float):
        captured["url"] = url
        captured["timeout"] = timeout
        return {"control_mode": "local"}

    mode = await probe_control_mode(
        "ws://127.0.0.1:8765/ws/control",
        0.75,
        requester=requester,
    )

    assert mode == "local"
    assert captured == {
        "url": "http://127.0.0.1:8765/api/base_info",
        "timeout": 0.75,
    }


def base() -> dict:
    return {
        "protocol_version": "3.0", "side": "left", "control_mode": "remote",
        "units": {"gripper_position": "meter", "gripper_speed": "meter/second"},
        "arm": {"joint_speed_deg_s": {"default": [10] * 6, "max": [30] * 6},
                "gripper": {"backend": "piper", "speed_m_s": {"default": 0.02, "max": 0.05}}},
    }


def state() -> dict:
    return {"side": "left", "control_mode": "remote", "state": "RUNNING",
            "emergency_stopped": False, "control_ready": True, "ActualQ": [0] * 6,
            "ActualGripper": 0.025}


def test_snapshot_preserves_native_piper_gripper_metadata_units() -> None:
    snapshot = UpstreamSnapshot.build(
        state(), base(), 1.0, expected_side=ArmSide.LEFT
    )
    assert snapshot.gripper_max_speed == 0.05
    assert snapshot.gripper_default_speed == 0.02
    assert snapshot.gripper_backend == "piper"
    assert snapshot.gripper_position_unit == "meter"
    assert snapshot.gripper_speed_unit == "meter/second"
    assert snapshot.joint_max_speed == (30.0,) * 6
    assert snapshot.actual_gripper == 0.025
    assert snapshot.joint_default_speed == (10.0,) * 6


@pytest.mark.parametrize("backend", ["unsupported_backend", "unknown", None])
def test_snapshot_rejects_non_piper_gripper_backend(backend: object) -> None:
    payload = base()
    payload["arm"]["gripper"]["backend"] = backend
    if backend == "unsupported_backend":
        payload["units"] = {"gripper_position": "radian", "gripper_speed": "radian/second"}
        payload["arm"]["gripper"] = {
            "backend": backend, "speed_rad_s": {"default": 1.0, "max": 2.0}
        }
    with pytest.raises(ProtocolError, match="backend"):
        UpstreamSnapshot.build(state(), payload, 1.0, expected_side=ArmSide.LEFT)


def test_snapshot_requires_explicit_piper_backend() -> None:
    payload = base()
    del payload["arm"]["gripper"]["backend"]
    with pytest.raises(ProtocolError, match="backend"):
        UpstreamSnapshot.build(state(), payload, 1.0, expected_side=ArmSide.LEFT)


@pytest.mark.parametrize(
    ("field", "unit"),
    [("gripper_position", "radian"), ("gripper_speed", "radian/second"),
     ("gripper_position", None), ("gripper_speed", None)],
)
def test_snapshot_rejects_non_metre_gripper_units(field: str, unit: object) -> None:
    payload = base()
    payload["units"][field] = unit
    with pytest.raises(ProtocolError, match="units"):
        UpstreamSnapshot.build(state(), payload, 1.0, expected_side=ArmSide.LEFT)


def test_snapshot_rejects_side_mode_and_safety_mismatch() -> None:
    for mutate in (
        lambda value: value.update({"side": "right"}),
        lambda value: value.update({"control_mode": "local"}),
        lambda value: value.update({"emergency_stopped": True}),
    ):
        payload = state()
        mutate(payload)
        with pytest.raises(ProtocolError):
            UpstreamSnapshot.build(
                payload, base(), 1.0, expected_side=ArmSide.LEFT
            )


def test_snapshot_rejects_missing_or_nonfinite_gripper_feedback() -> None:
    for bad in (None, float("nan"), True):
        payload = state()
        payload["ActualGripper"] = bad
        with pytest.raises(ProtocolError, match="ActualGripper"):
            UpstreamSnapshot.build(
                payload, base(), 1.0, expected_side=ArmSide.LEFT
            )


def test_snapshot_rejects_an_upstream_from_the_wrong_arm_side() -> None:
    with pytest.raises(ProtocolError, match="expected right"):
        UpstreamSnapshot.build(
            state(), base(), 1.0, expected_side=ArmSide.RIGHT
        )
