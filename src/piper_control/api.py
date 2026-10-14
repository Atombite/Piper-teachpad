# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""FastAPI REST and single-client WebSocket boundary for Piper control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import mimetypes
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from piper_control.config import ArmConfig, ArmSide
from piper_control.controller import StateConflict
from piper_control.protocol import (
    GetBaseInfoRequest,
    GetStateRequest,
    MoveGripperRequest,
    ProtocolError,
    SUPPORTED_COMMANDS,
    ServoJRequest,
    error_response,
    parse_json_text,
    parse_hello,
    parse_request,
    success_response,
)
from piper_control.types import ControlMode, ServiceState
from piper_control.vectors import (
    CommandValidationError,
    validate_gripper,
    validate_servo,
)


LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "3.0"
HANDSHAKE_TIMEOUT_S = 2.0
WEB_DIRECTORY = Path(__file__).with_name("web")


def _log_event(level: int, event: str, **fields: object) -> None:
    LOGGER.log(
        level,
        "event=%s %s",
        event,
        " ".join(f"{key}={value}" for key, value in fields.items()),
        extra={"event": event, **fields},
    )


class ChangeStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["recover", "homing", "hold"]


class ChangeControlModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ControlMode


@dataclass(slots=True)
class ControlSession:
    token: object
    websocket: WebSocket
    task: asyncio.Task[Any]
    identity: str | None = None
    close_requested: bool = False
    released: asyncio.Event = field(default_factory=asyncio.Event)


def _installed_version(distribution: str, fallback: str = "unknown") -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return fallback


def _base_info(side: ArmSide, config: ArmConfig) -> dict[str, Any]:
    gripper_config = config.gripper
    gripper = {
        "backend": gripper_config.backend,
        "enabled": gripper_config.enabled,
        "position_m": {
            "min": gripper_config.min_position_m,
            "max": gripper_config.max_position_m,
        },
        "physical_position_m": {
            "min": gripper_config.min_position_m,
            "max": gripper_config.max_position_m,
        },
        "speed_m_s": {
            "default": gripper_config.default_speed_m_s,
            "max": gripper_config.max_speed_m_s,
        },
        "effort_nm": gripper_config.effort_nm,
    }
    arm = {
        "name": config.name,
        "can_port": config.can_port,
        "server": config.server.model_dump(mode="json"),
        "joint_limits_deg": {
            "min": list(config.min_joints_deg),
            "max": list(config.max_joints_deg),
        },
        "joint_speed_deg_s": {
            "default": list(config.default_joints_speed_deg_s),
            "max": list(config.max_joints_speed_deg_s),
        },
        "gripper": gripper,
    }
    return {
        "service_version": _installed_version("piper-control", "0.1.0"),
        "protocol_version": PROTOCOL_VERSION,
        "piper_sdk_version": _installed_version("piper-sdk"),
        "side": side.value,
        "supported_commands": list(SUPPORTED_COMMANDS),
        "units": {
            "joint_position": "degree",
            "joint_speed": "degree/second",
            "tcp_position": "millimeter",
            "tcp_orientation": "degree",
            "gripper_position": config.gripper.position_unit,
            "gripper_speed": config.gripper.speed_unit.replace("_per_", "/"),
            "gripper_effort": "newton_meter",
            "timestamp": "unix_second",
        },
        "arm": arm,
        "config": config.model_dump(mode="json"),
    }


def _rest_error(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error": {"code": code, "message": message}},
    )


def create_app(
    controller: Any,
    side: ArmSide,
    config: ArmConfig,
) -> FastAPI:
    """Create an application around only the controller's public safe interface."""

    app = FastAPI()
    lease_lock = asyncio.Lock()
    command_lock = asyncio.Lock()
    active_session: ControlSession | None = None
    static_base_info = _base_info(side, config)
    dashboard_html = (WEB_DIRECTORY / "index.html").read_text(encoding="utf-8")
    dashboard_script = (WEB_DIRECTORY / "app.js").read_text(encoding="utf-8")
    dashboard_style = (WEB_DIRECTORY / "style.css").read_text(encoding="utf-8")
    teaching_html = (
        (WEB_DIRECTORY / "teach.html")
        .read_text(encoding="utf-8")
        .replace("__PIPER_ARM_SIDE__", side.value)
        .replace("__PIPER_SERVER_PORT__", str(config.server.port))
    )

    def state_payload() -> dict[str, Any]:
        payload = controller.snapshot().to_json()
        payload["ws_controller_connected"] = active_session is not None
        payload["ws_controller_identity"] = (
            active_session.identity if active_session is not None else None
        )
        return payload

    def base_info_payload() -> dict[str, Any]:
        payload = dict(static_base_info)
        payload["control_mode"] = controller.snapshot().control_mode.value
        payload["ws_controller_connected"] = active_session is not None
        payload["ws_controller_identity"] = (
            active_session.identity if active_session is not None else None
        )
        return payload

    async def close_active_control(reason: str) -> None:
        async with lease_lock:
            session = active_session
            if session is not None:
                session.close_requested = True
        if session is None:
            return
        try:
            await session.websocket.close(code=4002, reason=reason)
        except RuntimeError:
            pass
        if session.task is not asyncio.current_task() and not session.task.done():
            session.task.cancel()
        await session.released.wait()

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return _rest_error("INVALID_REQUEST", "invalid request body", 422)

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        try:
            return JSONResponse(state_payload())
        except Exception:
            LOGGER.exception("state request failed")
            return _rest_error("INTERNAL_ERROR", "internal server error", 500)

    @app.get("/api/base_info")
    def get_base_info() -> JSONResponse:
        return JSONResponse(base_info_payload())

    @app.get("/api/emergency_stop")
    def emergency_stop() -> JSONResponse:
        headers = {"Cache-Control": "no-store"}
        try:
            controller.emergency_stop("HTTP emergency stop")
        except Exception:
            LOGGER.exception("HTTP emergency stop failed")
            response = _rest_error("INTERNAL_ERROR", "emergency stop failed", 500)
            response.headers.update(headers)
            return response
        return JSONResponse(
            {"ok": True, "data": {"accepted": True}}, headers=headers
        )

    @app.post("/change_state")
    async def change_state(change: ChangeStateRequest) -> JSONResponse:
        try:
            if change.state == "recover":
                await asyncio.to_thread(controller.request_recovery)
            elif change.state == "homing":
                await asyncio.to_thread(controller.request_homing)
            else:
                async with command_lock:
                    await asyncio.to_thread(controller.request_hold)
                    await close_active_control("service entered HOLD")
        except StateConflict as exc:
            return _rest_error("STATE_CONFLICT", str(exc), 409)
        except Exception:
            LOGGER.exception("state transition request failed")
            return _rest_error("INTERNAL_ERROR", "internal server error", 500)
        return JSONResponse(
            {"ok": True, "data": {"accepted": True, "state": change.state}}
        )

    @app.post("/control_mode")
    async def change_control_mode(change: ChangeControlModeRequest) -> JSONResponse:
        try:
            async with command_lock:
                # Keep lease admission and the mode transition atomic. Otherwise a
                # socket can complete an old-identity hello after the no-client
                # check but before the controller publishes its new mode.
                async with lease_lock:
                    if active_session is not None:
                        return _rest_error(
                            "STATE_CONFLICT",
                            "control mode change requires the old control connection to close",
                            409,
                        )
                    await asyncio.to_thread(
                        controller.change_control_mode, change.mode
                    )
        except StateConflict as exc:
            return _rest_error("STATE_CONFLICT", str(exc), 409)
        except Exception:
            LOGGER.exception("control mode transition request failed")
            return _rest_error("INTERNAL_ERROR", "internal server error", 500)
        return JSONResponse(
            {
                "ok": True,
                "data": {
                    "accepted": True,
                    "mode": change.mode.value,
                    "state": controller.snapshot().state.value,
                },
            }
        )

    async def send_conflict(
        websocket: WebSocket, *, already_accepted: bool = False
    ) -> None:
        if not already_accepted:
            await websocket.accept()
        await websocket.send_json(
            error_response(
                None,
                "connect",
                "CLIENT_CONFLICT",
                "controller already connected",
            )
        )
        await websocket.close(code=4409, reason="controller already connected")

    async def execute_ws(payload: object) -> dict[str, object]:
        request_id = payload.get("id") if isinstance(payload, dict) else None
        command = payload.get("cmd") if isinstance(payload, dict) else "unknown"
        if not isinstance(command, str):
            command = "unknown"
        try:
            parsed = parse_request(payload)
            if isinstance(parsed, ServoJRequest):
                command_value = validate_servo(
                    config, parsed.target_joints, parsed.speed
                )
                async with command_lock:
                    await asyncio.to_thread(
                        controller.submit_servo,
                        command_value,
                    )
                data: object = {"accepted": True}
            elif isinstance(parsed, MoveGripperRequest):
                command_value = validate_gripper(
                    config, parsed.target_position, parsed.speed
                )
                async with command_lock:
                    await asyncio.to_thread(
                        controller.submit_gripper,
                        command_value,
                    )
                data = {"accepted": True}
            elif isinstance(parsed, GetStateRequest):
                data = state_payload()
            elif isinstance(parsed, GetBaseInfoRequest):
                data = base_info_payload()
            else:  # pragma: no cover - exhaustive discriminator guard
                raise RuntimeError("unsupported parsed request")
            return success_response(parsed.id, parsed.cmd, data)
        except ProtocolError as exc:
            _log_event(
                logging.WARNING,
                "command_rejected",
                cmd=command,
                code=exc.code,
            )
            return error_response(request_id, command, exc.code, str(exc))
        except CommandValidationError as exc:
            _log_event(
                logging.WARNING,
                "command_rejected",
                cmd=command,
                code="OUT_OF_RANGE",
            )
            return error_response(request_id, command, "OUT_OF_RANGE", str(exc))
        except StateConflict as exc:
            _log_event(
                logging.WARNING,
                "command_rejected",
                cmd=command,
                code="STATE_CONFLICT",
            )
            return error_response(request_id, command, "STATE_CONFLICT", str(exc))
        except Exception:
            LOGGER.exception("WebSocket command failed")
            return error_response(
                request_id, command, "INTERNAL_ERROR", "internal server error"
            )

    @app.websocket("/ws/control")
    async def websocket_control(websocket: WebSocket) -> None:
        nonlocal active_session
        token = object()
        task = asyncio.current_task()
        assert task is not None
        session = ControlSession(token=token, websocket=websocket, task=task)
        blocked_reason: str | None = None
        async with command_lock:
            async with lease_lock:
                snapshot = controller.snapshot()
                if snapshot.control_block_reason is not None:
                    blocked_reason = snapshot.control_block_reason
        if blocked_reason is not None:
            await websocket.accept()
            await websocket.send_json(
                error_response(
                    None,
                    "connect",
                    "STATE_CONFLICT",
                    blocked_reason,
                )
            )
            await websocket.close(code=4403, reason="control is unavailable")
            return

        handshake_complete = False
        try:
            await websocket.accept()
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=HANDSHAKE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                await websocket.send_json(
                    error_response(
                        None,
                        "hello",
                        "HANDSHAKE_TIMEOUT",
                        "hello was not received before the handshake timeout",
                    )
                )
                await websocket.close(code=4408, reason="handshake timeout")
                return
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(
                    code=message.get("code", 1000),
                    reason=message.get("reason"),
                )
            request_id: object | None = None
            command = "hello"
            try:
                if message.get("bytes") is not None:
                    raise ProtocolError(
                        "INVALID_REQUEST", "request must be a text JSON frame"
                    )
                payload = parse_json_text(message.get("text"))
                if isinstance(payload, dict):
                    request_id = payload.get("id")
                    if isinstance(payload.get("cmd"), str):
                        command = payload["cmd"]
                hello = parse_hello(payload)
            except ProtocolError as exc:
                await websocket.send_json(
                    error_response(request_id, command, exc.code, str(exc))
                )
                close_code = 4401 if exc.code == "HANDSHAKE_REQUIRED" else 4400
                await websocket.close(code=close_code, reason=exc.code.lower())
                return
            async with command_lock:
                async with lease_lock:
                    snapshot = controller.snapshot()
                    blocked_reason = snapshot.control_block_reason
                    expected_identity = (
                        "local"
                        if snapshot.control_mode is ControlMode.LOCAL
                        else "bridge"
                    )
                    identity_mismatch = hello.identity != expected_identity
                    conflict = active_session is not None
                    if (
                        blocked_reason is None
                        and not identity_mismatch
                        and not conflict
                    ):
                        active_session = session
                        session.identity = hello.identity
            if blocked_reason is not None:
                await websocket.send_json(
                    error_response(
                        hello.id,
                        hello.cmd,
                        "STATE_CONFLICT",
                        blocked_reason,
                    )
                )
                await websocket.close(code=4403, reason="control is unavailable")
                return
            if identity_mismatch:
                await websocket.send_json(
                    error_response(
                        hello.id,
                        hello.cmd,
                        "IDENTITY_MISMATCH",
                        f"{expected_identity} identity is required in the current mode",
                    )
                )
                await websocket.close(code=4403, reason="identity mismatch")
                return
            if conflict:
                _log_event(logging.INFO, "ws_lease_conflict")
                await send_conflict(websocket, already_accepted=True)
                return
            _log_event(logging.INFO, "ws_lease_acquire")
            handshake_complete = True
            await websocket.send_json(
                success_response(
                    hello.id,
                    hello.cmd,
                    {"identity": hello.identity},
                )
            )
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(
                        code=message.get("code", 1000),
                        reason=message.get("reason"),
                    )
                try:
                    if message.get("bytes") is not None:
                        raise ProtocolError(
                            "INVALID_REQUEST", "request must be a text JSON frame"
                        )
                    payload = parse_json_text(message.get("text"))
                except ProtocolError as exc:
                    _log_event(
                        logging.WARNING,
                        "command_rejected",
                        cmd="unknown",
                        code=exc.code,
                    )
                    await websocket.send_json(
                        error_response(
                            None,
                            "unknown",
                            exc.code,
                            str(exc),
                        )
                    )
                    continue
                if isinstance(payload, dict) and payload.get("cmd") == "hello":
                    await websocket.send_json(
                        error_response(
                            payload.get("id"),
                            "hello",
                            "HANDSHAKE_ALREADY_COMPLETED",
                            "identity is fixed for the lifetime of the connection",
                        )
                    )
                    continue
                await websocket.send_json(await execute_ws(payload))
        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            if not session.close_requested:
                raise
        finally:
            async def hold_then_release() -> None:
                nonlocal active_session
                try:
                    if handshake_complete:
                        try:
                            await asyncio.to_thread(controller.client_disconnected)
                        except Exception:
                            LOGGER.exception("disconnect HOLD failed")
                finally:
                    async with lease_lock:
                        if active_session is session:
                            active_session = None
                            _log_event(logging.INFO, "ws_lease_release")
                    session.released.set()

            cleanup = asyncio.create_task(hold_then_release())
            cancellation: asyncio.CancelledError | None = None
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as exc:
                    cancellation = exc
            await cleanup
            if cancellation is not None:
                raise cancellation

    @app.get("/", include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(dashboard_html)

    @app.get("/static/app.js", include_in_schema=False)
    async def dashboard_javascript() -> Response:
        return Response(dashboard_script, media_type="text/javascript")

    @app.get("/static/style.css", include_in_schema=False)
    async def dashboard_stylesheet() -> Response:
        return Response(dashboard_style, media_type="text/css")

    @app.get("/teach", include_in_schema=False)
    async def teaching_page() -> HTMLResponse:
        return HTMLResponse(teaching_html)

    @app.get("/static/teach/{asset_path:path}", include_in_schema=False)
    def teaching_asset(asset_path: str) -> Response:
        web_root = WEB_DIRECTORY.resolve()
        candidate = (web_root / asset_path).resolve()
        if not candidate.is_relative_to(web_root) or not candidate.is_file():
            raise HTTPException(status_code=404)
        media_type, _encoding = mimetypes.guess_type(candidate.name)
        return Response(
            candidate.read_bytes(),
            media_type=media_type or "application/octet-stream",
        )

    return app
