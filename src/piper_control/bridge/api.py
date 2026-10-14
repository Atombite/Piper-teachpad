# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Multi-client WebSocket boundary for one arm-local Bridge."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from piper_control.bridge.protocol import (
    BridgeGripperRequest,
    BridgeServoRequest,
    ReleaseRequest,
    accepted_response_data,
    parse_bridge_request,
)
from piper_control.protocol import (
    GetBaseInfoRequest,
    GetStateRequest,
    ProtocolError,
    error_response,
    parse_json_text,
    success_response,
)


LOGGER = logging.getLogger(__name__)


def create_bridge_app(service: Any) -> FastAPI:
    """Expose a Bridge coordinator without a downstream singleton lease."""

    app = FastAPI()

    async def execute(connection_id: str, payload: object) -> dict[str, object]:
        request_id = payload.get("id") if isinstance(payload, dict) else None
        command = payload.get("cmd") if isinstance(payload, dict) else "unknown"
        if not isinstance(command, str):
            command = "unknown"
        try:
            parsed = parse_bridge_request(payload)
            if isinstance(parsed, (BridgeServoRequest, BridgeGripperRequest)):
                result = await service.submit(connection_id, parsed)
                if not result.accepted:
                    raise ProtocolError(
                        result.code or "COMMAND_REJECTED",
                        "command was not accepted",
                    )
                data: object = accepted_response_data(parsed.priority)
            elif isinstance(parsed, ReleaseRequest):
                await service.release(connection_id, parsed.priority)
                data = {"released": True, "priority": parsed.priority}
            elif isinstance(parsed, GetStateRequest):
                data = await service.state()
            elif isinstance(parsed, GetBaseInfoRequest):
                data = await service.base_info()
            else:  # pragma: no cover - exhaustive discriminator guard
                raise RuntimeError("unsupported parsed request")
            return success_response(parsed.id, parsed.cmd, data)
        except ProtocolError as exc:
            LOGGER.warning(
                "event=bridge_command_rejected connection_id=%s cmd=%s code=%s",
                connection_id,
                command,
                exc.code,
            )
            return error_response(request_id, command, exc.code, str(exc))
        except Exception:
            LOGGER.exception(
                "event=bridge_command_failure connection_id=%s cmd=%s",
                connection_id,
                command,
            )
            return error_response(
                request_id, command, "INTERNAL_ERROR", "internal server error"
            )

    @app.websocket("/ws/control")
    async def websocket_control(websocket: WebSocket) -> None:
        connection_id = uuid.uuid4().hex
        await websocket.accept()
        try:
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
                    response = await execute(connection_id, payload)
                except ProtocolError as exc:
                    response = error_response(
                        None, "unknown", exc.code, str(exc)
                    )
                await websocket.send_json(response)
        except WebSocketDisconnect:
            pass
        finally:
            await asyncio.shield(service.disconnect(connection_id))

    return app
