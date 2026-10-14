# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Serialized protocol-3 client and validated metadata cache for piper-control."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import itertools
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

import websockets

from piper_control.config import ArmSide
from piper_control.protocol import ProtocolError


class UpstreamError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


async def _fetch_json(endpoint: str, timeout_s: float) -> object:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise UpstreamError("UPSTREAM_INVALID", "mode probe URL is invalid")

    async def exchange() -> object:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        reader, writer = await asyncio.open_connection(
            parsed.hostname,
            port,
            ssl=parsed.scheme == "https",
        )
        try:
            path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            request = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {parsed.netloc}\r\n"
                "Accept: application/json\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(request.encode("ascii"))
            await writer.drain()
            status_line = await reader.readline()
            parts = status_line.decode("ascii", errors="replace").split()
            if len(parts) < 2 or parts[1] != "200":
                raise UpstreamError("UPSTREAM_PROTOCOL", "mode probe request failed")
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b""}:
                    break
                name, separator, value = line.decode("ascii", errors="replace").partition(":")
                if separator:
                    headers[name.strip().lower()] = value.strip()
            length = headers.get("content-length")
            body = await reader.readexactly(int(length)) if length is not None else await reader.read()
            return json.loads(body)
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    try:
        return await asyncio.wait_for(exchange(), timeout_s)
    except asyncio.TimeoutError:
        raise UpstreamError("UPSTREAM_TIMEOUT", "control mode probe timed out") from None


async def probe_control_mode(
    websocket_url: str,
    timeout_s: float,
    *,
    requester: Callable[[str, float], Awaitable[object]] = _fetch_json,
) -> str:
    """Read controller mode without opening or occupying a control lease."""

    parsed = urlsplit(websocket_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    endpoint = urlunsplit((scheme, parsed.netloc, "/api/base_info", "", ""))

    payload = await requester(endpoint, timeout_s)
    mode = payload.get("control_mode") if isinstance(payload, dict) else None
    if mode not in {"local", "remote"}:
        raise UpstreamError("UPSTREAM_INVALID", "control mode probe returned invalid data")
    return mode


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _finite_six(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ProtocolError("UPSTREAM_INVALID", f"{label} must contain six values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(item) for item in value):
        raise ProtocolError("UPSTREAM_INVALID", f"{label} must contain finite numbers")
    return tuple(float(item) for item in value)


@dataclass(frozen=True, slots=True)
class UpstreamSnapshot:
    state: Mapping[str, Any]
    base_info: Mapping[str, Any]
    received_at: float
    side: ArmSide
    joint_max_speed: tuple[float, ...]
    joint_default_speed: tuple[float, ...]
    gripper_max_speed: float
    gripper_default_speed: float
    actual_joints: tuple[float, ...]
    actual_gripper: float
    gripper_backend: str
    gripper_position_unit: str
    gripper_speed_unit: str

    @classmethod
    def build(
        cls,
        state: object,
        base_info: object,
        received_at: float,
        *,
        expected_side: ArmSide,
    ) -> "UpstreamSnapshot":
        if not isinstance(state, dict) or not isinstance(base_info, dict):
            raise ProtocolError("UPSTREAM_INVALID", "upstream state and base info must be objects")
        if not isinstance(received_at, (int, float)) or isinstance(received_at, bool) or not isfinite(received_at):
            raise ProtocolError("UPSTREAM_INVALID", "received_at must be finite")
        if base_info.get("protocol_version") != "3.0":
            raise ProtocolError("UPSTREAM_INVALID", "upstream protocol must be 3.0")
        try:
            side = ArmSide(base_info.get("side"))
        except ValueError:
            raise ProtocolError("UPSTREAM_INVALID", "upstream side is invalid") from None
        if state.get("side") != side.value:
            raise ProtocolError("UPSTREAM_INVALID", "state and base side do not match")
        if side is not expected_side:
            raise ProtocolError(
                "SIDE_MISMATCH",
                f"upstream side is {side.value}; expected {expected_side.value}",
            )
        if base_info.get("control_mode") != "remote" or state.get("control_mode") != "remote":
            raise ProtocolError("MODE_MISMATCH", "upstream must be in remote mode")
        if state.get("state") != "RUNNING" or state.get("emergency_stopped") is not False or state.get("control_ready") is not True:
            raise ProtocolError("UPSTREAM_UNSAFE", "upstream state is unavailable or unsafe")
        actual_joints = _finite_six(state.get("ActualQ"), "ActualQ")
        actual_gripper = state.get("ActualGripper")
        if isinstance(actual_gripper, bool) or not isinstance(actual_gripper, (int, float)) or not isfinite(actual_gripper):
            raise ProtocolError("UPSTREAM_INVALID", "ActualGripper must be finite")
        joint_speeds = base_info.get("arm", {}).get("joint_speed_deg_s", {})
        joint_speed = _finite_six(joint_speeds.get("max"), "joint maximum speed")
        joint_default_speed = _finite_six(joint_speeds.get("default"), "joint default speed")
        if any(value <= 0 for value in (*joint_speed, *joint_default_speed)):
            raise ProtocolError("UPSTREAM_INVALID", "joint speeds must be positive")
        if any(default > maximum for default, maximum in zip(joint_default_speed, joint_speed, strict=True)):
            raise ProtocolError("UPSTREAM_INVALID", "joint default speed exceeds maximum")
        units = base_info.get("units", {})
        gripper = base_info.get("arm", {}).get("gripper", {})
        backend = gripper.get("backend")
        if backend != "piper":
            raise ProtocolError("UPSTREAM_INVALID", "unsupported gripper backend")
        position_unit, speed_unit = "meter", "meter/second"
        speed_limits = gripper.get("speed_m_s", {})
        maximum = speed_limits.get("max")
        default = speed_limits.get("default")
        if units.get("gripper_position") != position_unit or units.get("gripper_speed") != speed_unit:
            raise ProtocolError("UPSTREAM_INVALID", "gripper units do not match backend")
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not isfinite(maximum) or maximum <= 0:
            raise ProtocolError("UPSTREAM_INVALID", "gripper maximum speed is invalid")
        if isinstance(default, bool) or not isinstance(default, (int, float)) or not isfinite(default) or default <= 0:
            raise ProtocolError("UPSTREAM_INVALID", "gripper default speed is invalid")
        if default > maximum:
            raise ProtocolError("UPSTREAM_INVALID", "gripper default speed exceeds maximum")
        return cls(
            state=_freeze(state),
            base_info=_freeze(base_info),
            received_at=float(received_at),
            side=side,
            joint_max_speed=joint_speed,
            joint_default_speed=joint_default_speed,
            gripper_max_speed=float(maximum),
            gripper_default_speed=float(default),
            actual_joints=actual_joints,
            actual_gripper=float(actual_gripper),
            gripper_backend=backend,
            gripper_position_unit=position_unit,
            gripper_speed_unit=speed_unit,
        )


class PiperControlClient:
    def __init__(self, url: str, request_timeout_s: float = 1.0, connector: Callable[..., Any] = websockets.connect) -> None:
        self.url = url
        self.request_timeout_s = request_timeout_s
        self.connector = connector
        self.socket: Any | None = None
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()

    async def connect(self) -> "PiperControlClient":
        if self.socket is not None:
            return self
        self.socket = await self.connector(self.url)
        try:
            hello = await self._request("hello", identity="bridge")
            if hello != {"identity": "bridge"}:
                raise UpstreamError("PROTOCOL_MISMATCH", "bridge identity was not confirmed")
        except BaseException:
            await self.close()
            raise
        return self

    async def close(self) -> None:
        socket, self.socket = self.socket, None
        if socket is not None:
            await socket.close()

    async def __aenter__(self) -> "PiperControlClient":
        return await self.connect()

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def _request(self, command: str, **fields: object) -> Any:
        async with self._lock:
            if self.socket is None:
                raise UpstreamError("UPSTREAM_UNAVAILABLE", "upstream is not connected")
            request_id = str(next(self._ids))
            try:
                await self.socket.send(json.dumps({"id": request_id, "cmd": command, **fields}))
                raw = await asyncio.wait_for(self.socket.recv(decode=True), self.request_timeout_s)
                response = json.loads(raw)
            except asyncio.CancelledError:
                await self.close()
                raise
            except asyncio.TimeoutError:
                await self.close()
                raise UpstreamError("UPSTREAM_TIMEOUT", f"{command} timed out") from None
            except Exception as exc:
                await self.close()
                raise UpstreamError("UPSTREAM_PROTOCOL", str(exc)) from exc
            if not isinstance(response, dict) or response.get("id") != request_id or response.get("cmd") != command:
                await self.close()
                raise UpstreamError("PROTOCOL_MISMATCH", "upstream response correlation mismatch")
            if response.get("ok") is not True:
                error = response.get("error")
                if not isinstance(error, dict) or not isinstance(error.get("code"), str) or not isinstance(error.get("message"), str):
                    raise UpstreamError("UPSTREAM_PROTOCOL", "malformed upstream error")
                raise UpstreamError(error["code"], error["message"])
            if "data" not in response:
                raise UpstreamError("UPSTREAM_PROTOCOL", "successful upstream response has no data")
            return response["data"]

    async def get_state(self) -> dict[str, Any]:
        return await self._request("get_state")

    async def get_base_info(self) -> dict[str, Any]:
        return await self._request("get_base_info")

    async def servo_j(self, target_joints: tuple[float, ...], speed: tuple[float, ...]) -> dict[str, Any]:
        return await self._request("servoJ", target_joints=list(target_joints), speed=list(speed))

    async def move_gripper(self, target_position: float, speed: float) -> dict[str, Any]:
        return await self._request("move_gripper", target_position=target_position, speed=speed)
