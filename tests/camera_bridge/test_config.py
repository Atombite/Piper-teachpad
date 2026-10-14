# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from piper_control.camera_bridge.config import (
    CameraBridgeConfig,
    CameraName,
    StreamName,
    load_camera_config,
)


LEFT_DEVICE = (
    "/dev/v4l/by-id/"
    "REPLACE_WITH_LEFT_WRIST_CAMERA-video-index0"
)
RIGHT_SERIAL = "REPLACE_WITH_RIGHT_WRIST_REALSENSE_SERIAL"
HEAD_SERIAL = "REPLACE_WITH_HEAD_REALSENSE_SERIAL"


def stream(capture_format: str, encoding: str) -> dict[str, object]:
    return {
        "width": 640,
        "height": 480,
        "fps": 30,
        "capture_format": capture_format,
        "encoding": encoding,
    }


def valid_config() -> dict[str, object]:
    return {
        "bind": "tcp://0.0.0.0:15555",
        "queue_size": 2,
        "jpeg_quality": 90,
        "recovery": {
            "frame_timeout_ms": 200,
                "startup_timeout_ms": 10000,
            "max_consecutive_frame_failures": 3,
            "reconnect_timeout_ms": 1000,
            "max_reconnect_attempts": 1,
        },
        "cameras": {
            "left_wrist": {
                "driver": "v4l2",
                "device": LEFT_DEVICE,
                "streams": {"color": stream("mjpeg", "jpeg")},
            },
            "right_wrist": {
                "driver": "realsense",
                "serial": RIGHT_SERIAL,
                "streams": {
                    "color": stream("bgr8", "jpeg"),
                    "depth": stream("z16", "raw"),
                },
            },
            "head": {
                "driver": "realsense",
                "serial": HEAD_SERIAL,
                "streams": {
                    "color": stream("bgr8", "jpeg"),
                    "depth": stream("z16", "raw"),
                    "infrared_1": stream("y8", "raw"),
                    "infrared_2": stream("y8", "raw"),
                },
            },
        },
    }


def write_yaml(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_loads_fixed_hardware_roles_and_stream_profiles(tmp_path: Path) -> None:
    config = load_camera_config(write_yaml(tmp_path, valid_config()))

    assert config.bind == "tcp://0.0.0.0:15555"
    assert config.queue_size == 2
    assert config.jpeg_quality == 90
    assert config.recovery.frame_timeout_ms == 200
    assert config.recovery.max_consecutive_frame_failures == 3
    assert config.recovery.reconnect_timeout_ms == 1000
    assert config.recovery.max_reconnect_attempts == 1
    assert config.cameras.left_wrist.device == LEFT_DEVICE
    assert config.cameras.right_wrist.serial == RIGHT_SERIAL
    assert config.cameras.head.serial == HEAD_SERIAL
    assert tuple(config.cameras.left_wrist.streams) == (StreamName.COLOR,)
    assert set(config.cameras.right_wrist.streams) == {
        StreamName.COLOR,
        StreamName.DEPTH,
    }
    assert set(config.cameras.head.streams) == {
        StreamName.COLOR,
        StreamName.DEPTH,
        StreamName.INFRARED_1,
        StreamName.INFRARED_2,
    }
    assert [name for name, _camera in config.camera_items()] == [
        CameraName.LEFT_WRIST,
        CameraName.RIGHT_WRIST,
        CameraName.HEAD,
    ]
    with pytest.raises(ValidationError, match="frozen"):
        config.jpeg_quality = 80


def test_stream_resolution_is_configurable_and_positive() -> None:
    raw = valid_config()
    raw["cameras"]["head"]["streams"]["color"]["width"] = 1280  # type: ignore[index]
    config = CameraBridgeConfig.model_validate(raw)
    assert config.cameras.head.streams[StreamName.COLOR].width == 1280


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update({"unknown": True}), "extra"),
        (lambda raw: raw.update({"queue_size": 3}), "queue_size"),
        (
            lambda raw: raw["recovery"].update({"max_reconnect_attempts": 2}),
            "max_reconnect_attempts",
        ),
        (lambda raw: raw.update({"jpeg_quality": "90"}), "valid integer"),
        (
            lambda raw: raw["cameras"]["left_wrist"].update(
                {"device": "/dev/video0"}
            ),
            "/dev/v4l/by-id",
        ),
        (
            lambda raw: raw["cameras"]["right_wrist"].update(
                {"serial": HEAD_SERIAL}
            ),
            "unique",
        ),
        (
            lambda raw: raw["cameras"]["head"]["streams"].pop("infrared_2"),
            "head streams",
        ),
        (
            lambda raw: raw["cameras"]["right_wrist"]["streams"].update(
                {"infrared_1": stream("y8", "raw")}
            ),
            "right_wrist streams",
        ),
        (
            lambda raw: raw["cameras"]["head"]["streams"]["depth"].update(
                {"encoding": "jpeg"}
            ),
            "z16",
        ),
        (
            lambda raw: raw["cameras"]["head"]["streams"]["color"].update(
                {"width": 0}
            ),
            "greater than 0",
        ),
        (
            lambda raw: raw["recovery"].update({"frame_timeout_ms": 0}),
            "greater than or equal to 1",
        ),
    ],
)
def test_rejects_unsafe_or_mismatched_configuration(mutation, message: str) -> None:
    raw = deepcopy(valid_config())
    mutation(raw)
    with pytest.raises(ValidationError, match=message):
        CameraBridgeConfig.model_validate(raw)


def test_requires_all_three_camera_roles() -> None:
    raw = valid_config()
    raw["cameras"].pop("head")  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="head"):
        CameraBridgeConfig.model_validate(raw)


def test_example_uses_parseable_placeholders_with_discovery_guidance() -> None:
    path = Path(__file__).parents[2] / "config/camera.yaml"
    text = path.read_text(encoding="utf-8")
    config = load_camera_config(path)

    assert config.cameras.left_wrist.device == LEFT_DEVICE
    assert config.cameras.right_wrist.serial == RIGHT_SERIAL
    assert config.cameras.head.serial == HEAD_SERIAL
    assert "ls -l /dev/v4l/by-id/" in text
    assert "rs-enumerate-devices" in text


def test_rejects_non_mapping_or_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_camera_config(write_yaml(tmp_path, ["not", "a", "mapping"]))

    path = tmp_path / "broken.yaml"
    path.write_text("cameras: [", encoding="utf-8")
    with pytest.raises(ValueError, match="failed to parse YAML"):
        load_camera_config(path)
