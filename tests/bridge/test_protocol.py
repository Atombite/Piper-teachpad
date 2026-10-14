# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from piper_control.bridge.protocol import (
    BridgeGripperRequest,
    BridgeServoRequest,
    ReleaseRequest,
    accepted_response_data,
    parse_bridge_request,
)
from piper_control.protocol import GetBaseInfoRequest, GetStateRequest, ProtocolError


@pytest.mark.parametrize(
    ("payload", "request_type"),
    [
        ({"cmd": "servoJ", "target_joints": [0] * 6}, BridgeServoRequest),
        (
            {"cmd": "servoJ", "target_joints": [0] * 6, "priority": 100},
            BridgeServoRequest,
        ),
        (
            {"cmd": "move_gripper", "target_position": 0.2},
            BridgeGripperRequest,
        ),
        (
            {"cmd": "move_gripper", "target_position": 0.2, "priority": -7},
            BridgeGripperRequest,
        ),
        ({"cmd": "release", "priority": 7}, ReleaseRequest),
        ({"cmd": "get_state"}, GetStateRequest),
        ({"cmd": "get_base_info"}, GetBaseInfoRequest),
    ],
)
def test_parses_motion_release_and_query_commands(payload, request_type) -> None:
    request = parse_bridge_request(payload)
    assert isinstance(request, request_type)
    if isinstance(request, (BridgeServoRequest, BridgeGripperRequest)):
        assert request.priority == payload.get("priority", 0)


@pytest.mark.parametrize("command", ["servoJ", "move_gripper"])
@pytest.mark.parametrize(
    "priority",
    [True, False, 1.0, "1", -2147483649, 2147483648],
)
def test_priority_is_strict_signed_32_bit(command: str, priority: object) -> None:
    payload: dict[str, object] = {"cmd": command, "priority": priority}
    if command == "servoJ":
        payload["target_joints"] = [0] * 6
    else:
        payload["target_position"] = 0.2

    with pytest.raises(ProtocolError) as caught:
        parse_bridge_request(payload)
    assert caught.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize(
    "payload",
    [
        {"cmd": "release"},
        {"cmd": "release", "priority": True},
        {"cmd": "release", "priority": 1, "extra": "no"},
        {"cmd": "servoJ", "target_joints": [0] * 5},
        {"cmd": "servoJ", "target_joints": [0] * 5 + [True]},
        {"cmd": "move_gripper", "target_position": True},
        {"cmd": "get_state", "priority": 0},
    ],
)
def test_rejects_missing_extra_or_malformed_fields(payload: object) -> None:
    with pytest.raises(ProtocolError) as caught:
        parse_bridge_request(payload)
    assert caught.value.code == "INVALID_REQUEST"


def test_unknown_command_has_stable_code() -> None:
    with pytest.raises(ProtocolError) as caught:
        parse_bridge_request({"cmd": "hello", "identity": "bridge"})
    assert caught.value.code == "UNKNOWN_COMMAND"


def test_accepted_response_marks_priority_and_queueing() -> None:
    assert accepted_response_data(100) == {
        "accepted": True,
        "queued": True,
        "priority": 100,
    }
