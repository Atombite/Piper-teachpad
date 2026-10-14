# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the camera bridge service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import socket
from threading import Event
from typing import Any, Callable, Mapping, Sequence

from piper_control.camera_bridge.config import (
    CameraBridgeConfig,
    CameraConfig,
    load_camera_config,
)
from piper_control.camera_bridge.publisher import CameraPublisher
from piper_control.camera_bridge.service import CameraBridgeService
from piper_control.camera_bridge.sources import create_source
from piper_control.lifecycle import install_signal_handlers


LOGGER = logging.getLogger(__name__)


def notify_systemd_ready(
    *,
    environ: Mapping[str, str] = os.environ,
    socket_factory: Callable[..., Any] = socket.socket,
) -> None:
    """Tell systemd that camera capture and publication are ready."""
    address = environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    with socket_factory(socket.AF_UNIX, socket.SOCK_DGRAM) as notify_socket:
        notify_socket.connect(address)
        notify_socket.sendall(b"READY=1")


def _make_source(config: CameraConfig, jpeg_quality: int) -> Any:
    return create_source(config, jpeg_quality=jpeg_quality)


@dataclass(frozen=True)
class CameraRuntimeDependencies:
    source_factory: Callable[[CameraConfig, int], Any] = _make_source
    publisher_factory: Callable[[str], Any] = CameraPublisher
    service_factory: Callable[..., Any] = CameraBridgeService
    readiness_notifier: Callable[[], None] = notify_systemd_ready


def run_camera_bridge(
    config: CameraBridgeConfig,
    shutdown_event: Event,
    *,
    dependencies: CameraRuntimeDependencies = CameraRuntimeDependencies(),
) -> int:
    sources = {
        name: dependencies.source_factory(camera, config.jpeg_quality)
        for name, camera in config.camera_items()
    }
    publisher = dependencies.publisher_factory(config.bind)
    service = dependencies.service_factory(
        config,
        sources,
        publisher,
        on_ready=dependencies.readiness_notifier,
    )
    return int(service.run(shutdown_event))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Broadcast the robot's three camera streams over ZeroMQ"
    )
    parser.add_argument("--config", required=True, help="camera YAML configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        config = load_camera_config(Path(args.config))
        shutdown_event = Event()
        install_signal_handlers(shutdown_event)
        return run_camera_bridge(config, shutdown_event)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("event=camera_bridge_startup_failure error=%r", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
