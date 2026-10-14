# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Process entry point for one independent Piper Bridge instance."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Event
from typing import Any, Callable, Sequence

import uvicorn

from piper_control.bridge.api import create_bridge_app
from piper_control.bridge.config import BridgeConfig, BridgeServiceConfig, load_bridge_config
from piper_control.bridge.service import BridgeService
from piper_control.config import ArmSide
from piper_control.lifecycle import install_signal_handlers


LOGGER = logging.getLogger(__name__)
SHUTDOWN_TIMEOUT_S = 2.0


def _make_server(app: Any, host: str, port: int) -> uvicorn.Server:
    return uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="info")
    )


@dataclass(frozen=True)
class BridgeRuntimeDependencies:
    service_factory: Callable[..., Any] = BridgeService
    app_factory: Callable[[Any], Any] = create_bridge_app
    server_factory: Callable[[Any, str, int], Any] = _make_server


async def _run_bridge(
    selected: BridgeConfig,
    side: ArmSide,
    shutdown_event: Event,
    dependencies: BridgeRuntimeDependencies,
) -> int:
    service = dependencies.service_factory(selected, side=side)
    app = dependencies.app_factory(service)
    server = dependencies.server_factory(
        app, selected.listen.host, selected.listen.port
    )
    server_task: asyncio.Task[Any] | None = None
    failed = False
    try:
        await service.start()
        server_task = asyncio.create_task(server.serve(), name="piper-bridge-web")
        while not shutdown_event.is_set() and not server_task.done():
            await asyncio.sleep(0.01)
        if server_task.done():
            exception = server_task.exception()
            if exception is not None:
                raise exception
            if not shutdown_event.is_set():
                raise RuntimeError("Bridge web server exited unexpectedly")
    except Exception:
        failed = True
        LOGGER.exception("event=bridge_runtime_failure")
    finally:
        server.should_exit = True
        if server_task is not None:
            try:
                await asyncio.wait_for(server_task, timeout=SHUTDOWN_TIMEOUT_S)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                failed = True
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            except Exception:
                failed = True
                LOGGER.exception("event=bridge_web_shutdown_failure")
        try:
            await asyncio.wait_for(service.stop(), timeout=SHUTDOWN_TIMEOUT_S)
        except asyncio.TimeoutError:
            failed = True
            LOGGER.error(
                "event=bridge_service_shutdown_timeout timeout_s=%s",
                SHUTDOWN_TIMEOUT_S,
            )
        except Exception:
            failed = True
            LOGGER.exception("event=bridge_service_shutdown_failure")
    return 1 if failed else 0


def run_bridge(
    config: BridgeServiceConfig,
    side: ArmSide,
    shutdown_event: Event,
    *,
    dependencies: BridgeRuntimeDependencies = BridgeRuntimeDependencies(),
) -> int:
    try:
        selected = config.bridges[side]
    except KeyError:
        LOGGER.error("event=bridge_side_not_configured side=%s", side.value)
        return 1
    return asyncio.run(_run_bridge(selected, side, shutdown_event, dependencies))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Piper control Bridge")
    parser.add_argument("--config", required=True)
    parser.add_argument("--side", type=ArmSide, choices=tuple(ArmSide), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        config = load_bridge_config(Path(args.config))
        shutdown_event = Event()
        install_signal_handlers(shutdown_event)
        return run_bridge(config, args.side, shutdown_event)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("event=bridge_startup_failure error=%r", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
