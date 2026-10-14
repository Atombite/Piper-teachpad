# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict downstream command protocol for the multi-client Bridge."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StrictInt, TypeAdapter, ValidationError

from piper_control.protocol import (
    GetBaseInfoRequest,
    GetStateRequest,
    MoveGripperRequest,
    ProtocolError,
    RequestModel,
    ServoJRequest,
)


Priority = Annotated[StrictInt, Field(ge=-2147483648, le=2147483647)]


class BridgeServoRequest(ServoJRequest):
    priority: Priority = 0


class BridgeGripperRequest(MoveGripperRequest):
    priority: Priority = 0


class ReleaseRequest(RequestModel):
    cmd: Literal["release"]
    priority: Priority


BridgeRequest: TypeAlias = Annotated[
    BridgeServoRequest
    | BridgeGripperRequest
    | ReleaseRequest
    | GetStateRequest
    | GetBaseInfoRequest,
    Field(discriminator="cmd"),
]
_BRIDGE_REQUEST_ADAPTER = TypeAdapter(BridgeRequest)
BRIDGE_COMMANDS = (
    "servoJ",
    "move_gripper",
    "release",
    "get_state",
    "get_base_info",
)


def parse_bridge_request(payload: object) -> BridgeRequest:
    """Parse one downstream command without coercing priority lane keys."""

    if not isinstance(payload, dict):
        raise ProtocolError("INVALID_REQUEST", "request must be a JSON object")
    command = payload.get("cmd")
    if not isinstance(command, str):
        raise ProtocolError("INVALID_REQUEST", "cmd must be a string")
    if command not in BRIDGE_COMMANDS:
        raise ProtocolError("UNKNOWN_COMMAND", f"unsupported command: {command}")
    for field in ("target_joints", "speed") if command == "servoJ" else ():
        if field not in payload or payload[field] is None and field == "speed":
            continue
        value = payload[field]
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ProtocolError(
                "INVALID_REQUEST", f"{field} must contain exactly 6 values"
            )
    try:
        return _BRIDGE_REQUEST_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"] if part != command)
        detail = first["msg"]
        message = (
            f"invalid {location}: {detail}"
            if location
            else f"invalid request: {detail}"
        )
        raise ProtocolError("INVALID_REQUEST", message) from None


def accepted_response_data(priority: int) -> dict[str, object]:
    return {"accepted": True, "queued": True, "priority": priority}
