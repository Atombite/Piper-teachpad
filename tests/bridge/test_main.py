# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from threading import Event, Thread
import time

from piper_control.bridge.config import BridgeServiceConfig
from piper_control.bridge.main import BridgeRuntimeDependencies, build_parser, run_bridge
from piper_control.config import ArmSide


def config() -> BridgeServiceConfig:
    return BridgeServiceConfig.model_validate(
        {
            "bridges": {
                "left": {
                    "listen": {"host": "127.0.0.1", "port": 18765},
                    "upstream_url": "ws://127.0.0.1:8765/ws/control",
                    "control_hz": 100,
                },
                "right": {
                    "listen": {"host": "127.0.0.1", "port": 19876},
                    "upstream_url": "ws://127.0.0.1:9876/ws/control",
                    "control_hz": 100,
                },
            }
        }
    )


class Service:
    def __init__(self, selected) -> None:
        self.selected = selected
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")


class Server:
    def __init__(self) -> None:
        self.should_exit = False
        self.started = False

    async def serve(self) -> None:
        self.started = True
        while not self.should_exit:
            await asyncio.sleep(0.001)


def test_cli_requires_config_and_side() -> None:
    parser = build_parser()
    args = parser.parse_args(["--config", "bridge.yaml", "--side", "left"])
    assert args.config == "bridge.yaml"
    assert args.side is ArmSide.LEFT


def test_run_bridge_selects_one_side_and_stops_both_workers() -> None:
    services: list[Service] = []
    server = Server()

    def service_factory(selected, *, side):
        assert side is ArmSide.RIGHT
        service = Service(selected)
        services.append(service)
        return service

    deps = BridgeRuntimeDependencies(
        service_factory=service_factory,
        app_factory=lambda service: ("app", service),
        server_factory=lambda app, host, port: server,
    )
    shutdown = Event()
    thread = Thread(
        target=lambda: (time.sleep(0.02), shutdown.set()), daemon=True
    )
    thread.start()
    started = time.monotonic()
    result = run_bridge(config(), ArmSide.RIGHT, shutdown, dependencies=deps)
    elapsed = time.monotonic() - started
    thread.join()

    assert result == 0
    assert len(services) == 1
    assert services[0].selected.listen.port == 19876
    assert services[0].events == ["start", "stop"]
    assert server.started is True
    assert server.should_exit is True
    assert elapsed < 1.0


def test_shutdown_is_bounded_when_service_stop_stalls(monkeypatch) -> None:
    class StallingService(Service):
        async def stop(self) -> None:
            await asyncio.sleep(0.2)

    server = Server()
    shutdown = Event()
    shutdown.set()
    monkeypatch.setattr("piper_control.bridge.main.SHUTDOWN_TIMEOUT_S", 0.01)
    deps = BridgeRuntimeDependencies(
        service_factory=lambda selected, *, side: StallingService(selected),
        app_factory=lambda service: service,
        server_factory=lambda app, host, port: server,
    )

    started = time.monotonic()
    result = run_bridge(config(), ArmSide.LEFT, shutdown, dependencies=deps)

    assert result == 1
    assert time.monotonic() - started < 0.1
