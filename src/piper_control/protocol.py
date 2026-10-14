# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict version-two JSON request protocol and response envelopes."""

from __future__ import annotations

import json
from math import isfinite
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    Strict,
    TypeAdapter,
    ValidationError,
)


StrictFiniteFloat = Annotated[FiniteFloat, Strict()]
Joint6: TypeAlias = tuple[
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
]


class ProtocolError(ValueError):
    """A client-visible protocol error with a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Any | None = None


class HelloRequest(RequestModel):
    cmd: Literal["hello"]
    identity: Literal["local", "bridge"]


class ServoJRequest(RequestModel):
    cmd: Literal["servoJ"]
    target_joints: Joint6
    speed: Joint6 | None = None


class MoveGripperRequest(RequestModel):
    cmd: Literal["move_gripper"]
    target_position: StrictFiniteFloat
    speed: StrictFiniteFloat | None = None


class GetStateRequest(RequestModel):
    cmd: Literal["get_state"]


class GetBaseInfoRequest(RequestModel):
    cmd: Literal["get_base_info"]


Request: TypeAlias = Annotated[
    ServoJRequest | MoveGripperRequest | GetStateRequest | GetBaseInfoRequest,
    Field(discriminator="cmd"),
]
_REQUEST_ADAPTER = TypeAdapter(Request)
CONTROL_COMMANDS = (
    "servoJ",
    "move_gripper",
    "get_state",
    "get_base_info",
)
SUPPORTED_COMMANDS = ("hello", *CONTROL_COMMANDS)


def parse_hello(payload: object) -> HelloRequest:
    """Parse the mandatory first-frame identity handshake."""

    if not isinstance(payload, dict):
        raise ProtocolError("INVALID_REQUEST", "request must be a JSON object")
    if payload.get("cmd") != "hello":
        raise ProtocolError(
            "HANDSHAKE_REQUIRED",
            "first request must be hello",
        )
    try:
        return HelloRequest.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        detail = first["msg"]
        message = f"invalid {location}: {detail}" if location else f"invalid request: {detail}"
        raise ProtocolError("INVALID_REQUEST", message) from None


def parse_json_text(raw: object) -> object:
    """Decode one strict UTF-8 JSON text frame into JSON-native values."""

    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite JSON number")

    try:
        if not isinstance(raw, str):
            raise TypeError("text frame required")
        raw.encode("utf-8", errors="strict")
        value = json.loads(raw, parse_constant=reject_constant)
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                item.encode("utf-8", errors="strict")
            elif isinstance(item, float) and not isfinite(item):
                raise ValueError("non-finite JSON number")
            elif isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        raise ProtocolError("INVALID_REQUEST", "request must be valid JSON") from None
    return value


def parse_request(payload: object) -> Request:
    """Parse one strict request, distinguishing unknown commands from bad input."""

    if not isinstance(payload, dict):
        raise ProtocolError("INVALID_REQUEST", "request must be a JSON object")
    command = payload.get("cmd")
    if not isinstance(command, str):
        raise ProtocolError("INVALID_REQUEST", "cmd must be a string")
    if command not in CONTROL_COMMANDS:
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
        return _REQUEST_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"] if part != command)
        detail = first["msg"]
        message = f"invalid {location}: {detail}" if location else f"invalid request: {detail}"
        raise ProtocolError("INVALID_REQUEST", message) from None


def success_response(
    request_id: object | None, command: str, data: object
) -> dict[str, object]:
    response: dict[str, object] = {"cmd": command, "ok": True, "data": data}
    if request_id is not None:
        response["id"] = request_id
    return response


def error_response(
    request_id: object | None,
    command: str,
    code: str,
    message: str,
) -> dict[str, object]:
    response: dict[str, object] = {
        "cmd": command,
        "ok": False,
        "error": {"code": code, "message": message},
    }
    if request_id is not None:
        response["id"] = request_id
    return response
