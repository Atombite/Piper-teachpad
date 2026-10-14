# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from piper_control.camera_bridge.config import CameraName, load_camera_config


def test_parser_requires_config() -> None:
    from piper_control.camera_bridge.main import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_production_config_has_approved_hardware_and_profiles() -> None:
    config = load_camera_config(
        Path(__file__).parents[2] / "config" / "camera.yaml"
    )

    assert config.bind == "tcp://0.0.0.0:15555"
    assert config.queue_size == 2
    assert config.jpeg_quality == 90
    assert config.recovery.frame_timeout_ms == 200
    assert config.recovery.startup_timeout_ms == 10000
    assert config.recovery.max_consecutive_frame_failures == 3
    assert config.recovery.reconnect_timeout_ms == 1000
    assert config.recovery.max_reconnect_attempts == 1
    assert config.cameras.right_wrist.serial == (
        "REPLACE_WITH_RIGHT_WRIST_REALSENSE_SERIAL"
    )
    assert config.cameras.head.serial == "REPLACE_WITH_HEAD_REALSENSE_SERIAL"
    assert config.cameras.left_wrist.device.startswith("/dev/v4l/by-id/")
    assert [name for name, _camera in config.camera_items()] == list(CameraName)
    assert all(
        stream.width == 640 and stream.height == 480 and stream.fps == 30
        for _name, camera in config.camera_items()
        for stream in camera.streams.values()
    )


def test_main_logs_concise_startup_error_and_returns_one(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from piper_control.camera_bridge import main as main_module

    monkeypatch.setattr(
        main_module,
        "load_camera_config",
        lambda _path: (_ for _ in ()).throw(ValueError("bad camera config")),
    )

    assert main_module.main(["--config", "missing.yaml"]) == 1
    assert "event=camera_bridge_startup_failure" in caplog.text
    assert "bad camera config" in caplog.text


def test_run_builds_fixed_sources_and_propagates_clean_or_fault_exit() -> None:
    from piper_control.camera_bridge.main import (
        CameraRuntimeDependencies,
        run_camera_bridge,
    )

    config = load_camera_config(
        Path(__file__).parents[2] / "config" / "camera.yaml"
    )
    created: list[object] = []
    ready: list[str] = []

    class FakeService:
        def __init__(self, service_config, sources, publisher, *, on_ready) -> None:
            assert service_config is config
            assert set(sources) == set(CameraName)
            created.extend([*sources.values(), publisher])
            self.on_ready = on_ready

        def run(self, shutdown_event: Event) -> int:
            self.on_ready()
            return 0 if shutdown_event.is_set() else 1

    dependencies = CameraRuntimeDependencies(
        source_factory=lambda camera, jpeg_quality: (camera, jpeg_quality),
        publisher_factory=lambda endpoint: endpoint,
        service_factory=FakeService,
        readiness_notifier=lambda: ready.append("ready"),
    )

    shutdown = Event()
    shutdown.set()
    assert run_camera_bridge(config, shutdown, dependencies=dependencies) == 0
    assert len(created) == 4

    shutdown.clear()
    assert run_camera_bridge(config, shutdown, dependencies=dependencies) == 1
    assert ready == ["ready", "ready"]


def test_systemd_readiness_notification_uses_notify_socket() -> None:
    from piper_control.camera_bridge.main import notify_systemd_ready

    events: list[tuple[str, object]] = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def connect(self, address) -> None:
            events.append(("connect", address))

        def sendall(self, payload: bytes) -> None:
            events.append(("send", payload))

    notify_systemd_ready(
        environ={"NOTIFY_SOCKET": "@camera-ready"},
        socket_factory=lambda *_args: FakeSocket(),
    )

    assert events == [
        ("connect", "\0camera-ready"),
        ("send", b"READY=1"),
    ]


def test_main_installs_signal_handlers_and_clean_signal_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piper_control.camera_bridge import main as main_module

    config = load_camera_config(
        Path(__file__).parents[2] / "config" / "camera.yaml"
    )
    monkeypatch.setattr(main_module, "load_camera_config", lambda _path: config)
    monkeypatch.setattr(
        main_module,
        "install_signal_handlers",
        lambda event: event.set(),
    )
    monkeypatch.setattr(
        main_module,
        "run_camera_bridge",
        lambda _config, event: 0 if event.is_set() else 1,
    )

    assert main_module.main(["--config", "camera.yaml"]) == 0
