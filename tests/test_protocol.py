# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from piper_control.protocol import (
    GetBaseInfoRequest,
    GetStateRequest,
    HelloRequest,
    MoveGripperRequest,
    ProtocolError,
    ServoJRequest,
    error_response,
    parse_json_text,
    parse_hello,
    parse_request,
    success_response,
)


@pytest.mark.parametrize("identity", ["local", "bridge"])
def test_parse_hello_accepts_only_explicit_control_identities(identity: str) -> None:
    request = parse_hello({"id": "h1", "cmd": "hello", "identity": identity})

    assert isinstance(request, HelloRequest)
    assert request.id == "h1"
    assert request.identity == identity


@pytest.mark.parametrize(
    "payload",
    [
        {"cmd": "hello"},
        {"cmd": "hello", "identity": "operator"},
        {"cmd": "hello", "identity": True},
        {"cmd": "hello", "identity": "local", "token": "unused"},
        {"cmd": "get_state", "identity": "local"},
    ],
)
def test_parse_hello_rejects_missing_unknown_or_extra_identity_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ProtocolError) as caught:
        parse_hello(payload)

    assert caught.value.code in {"INVALID_REQUEST", "HANDSHAKE_REQUIRED"}


def test_parse_servo_requires_exactly_six_joints() -> None:
    request = parse_request({"cmd": "servoJ", "target_joints": [0] * 6})
    assert request.target_joints == (0.0,) * 6
    with pytest.raises(ProtocolError, match="exactly 6"):
        parse_request({"cmd": "servoJ", "target_joints": [0] * 12})


def test_parse_gripper_uses_scalars() -> None:
    request = parse_request(
        {"cmd": "move_gripper", "target_position": 0.05, "speed": 0.02}
    )
    assert request.target_position == 0.05
    assert request.speed == 0.02
    with pytest.raises(ProtocolError):
        parse_request({"cmd": "move_gripper", "target_positions": [0.05, 0]})


def test_protocol_models_are_strict_frozen_and_forbid_extra_fields() -> None:
    request = parse_request({"cmd": "get_state", "id": 7})
    assert isinstance(request, GetStateRequest)
    with pytest.raises(Exception, match="frozen"):
        request.cmd = "get_base_info"
    with pytest.raises(ProtocolError) as caught:
        parse_request({"cmd": "get_state", "surprise": True})
    assert caught.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize(
    ("payload", "request_type"),
    [
        (
            {"cmd": "servoJ", "target_joints": [0] * 6, "speed": [1] * 6},
            ServoJRequest,
        ),
        (
            {"cmd": "move_gripper", "target_position": 0.01, "speed": 0.02},
            MoveGripperRequest,
        ),
        ({"cmd": "get_state"}, GetStateRequest),
        ({"cmd": "get_base_info"}, GetBaseInfoRequest),
    ],
)
def test_parse_supported_commands(payload: dict[str, object], request_type: type) -> None:
    assert isinstance(parse_request(payload), request_type)


@pytest.mark.parametrize(
    "payload",
    [
        {"cmd": "servoJ", "target_joints": [0] * 6, "speed": [1] * 5},
        {"cmd": "servoJ", "target_joints": [0] * 5 + [float("nan")]},
        {"cmd": "servoJ", "target_joints": [0] * 5 + [True]},
        {"cmd": "move_gripper", "target_position": float("inf")},
        {"cmd": "move_gripper", "target_position": True},
        {"cmd": "move_gripper", "target_position": 0.01, "speed": False},
    ],
)
def test_rejects_malformed_nonfinite_or_coerced_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ProtocolError) as caught:
        parse_request(payload)
    assert caught.value.code == "INVALID_REQUEST"


def test_unknown_command_has_stable_code() -> None:
    with pytest.raises(ProtocolError, match="unsupported command") as caught:
        parse_request({"cmd": "recover", "id": "r1"})
    assert caught.value.code == "UNKNOWN_COMMAND"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"cmd":"get_state"}',
        "{",
        "[" * 2000 + "]" * 2000,
        "1" * 5000,
        '{"id":"\\ud800","cmd":"get_state"}',
        '{"id":NaN,"cmd":"get_state"}',
        '{"id":1e9999,"cmd":"get_state"}',
    ],
)
def test_json_frame_parser_normalizes_expected_malformed_or_extreme_input(raw) -> None:
    with pytest.raises(ProtocolError, match="valid JSON") as caught:
        parse_json_text(raw)
    assert caught.value.code == "INVALID_REQUEST"


def test_json_frame_parser_remains_usable_after_protocol_errors() -> None:
    with pytest.raises(ProtocolError):
        parse_json_text("{")

    assert parse_json_text('{"cmd":"get_state"}') == {"cmd": "get_state"}


def test_response_envelopes_echo_json_id_without_coercion() -> None:
    assert success_response(7, "get_state", {"ready": True}) == {
        "id": 7,
        "cmd": "get_state",
        "ok": True,
        "data": {"ready": True},
    }
    assert error_response("abc", "servoJ", "OUT_OF_RANGE", "bad target") == {
        "id": "abc",
        "cmd": "servoJ",
        "ok": False,
        "error": {"code": "OUT_OF_RANGE", "message": "bad target"},
    }
    assert "id" not in success_response(None, "get_state", {})
