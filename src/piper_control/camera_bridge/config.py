# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict configuration for the three-camera bridge."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import urlsplit

import yaml
from pydantic import Field, StrictInt, field_validator, model_validator

from piper_control.config import FrozenModel


class CameraName(str, Enum):
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    HEAD = "head"


class StreamName(str, Enum):
    COLOR = "color"
    DEPTH = "depth"
    INFRARED_1 = "infrared_1"
    INFRARED_2 = "infrared_2"


class StreamConfig(FrozenModel):
    width: StrictInt = Field(gt=0)
    height: StrictInt = Field(gt=0)
    fps: StrictInt = Field(gt=0)
    capture_format: Literal["mjpeg", "bgr8", "z16", "y8"]
    encoding: Literal["jpeg", "raw"]

    @model_validator(mode="after")
    def validate_format_pair(self) -> "StreamConfig":
        expected = {
            "mjpeg": "jpeg",
            "bgr8": "jpeg",
            "z16": "raw",
            "y8": "raw",
        }[self.capture_format]
        if self.encoding != expected:
            raise ValueError(
                f"{self.capture_format} streams require {expected} encoding"
            )
        return self


class RecoveryConfig(FrozenModel):
    frame_timeout_ms: StrictInt = Field(default=200, ge=1, le=1000)
    startup_timeout_ms: StrictInt = Field(default=10000, ge=1000, le=30000)
    max_consecutive_frame_failures: StrictInt = Field(default=3, ge=1, le=10)
    reconnect_timeout_ms: StrictInt = Field(default=1000, ge=1, le=5000)
    max_reconnect_attempts: Literal[1] = 1


class V4L2CameraConfig(FrozenModel):
    driver: Literal["v4l2"]
    device: str = Field(min_length=1)
    streams: dict[StreamName, StreamConfig]

    @field_validator("device")
    @classmethod
    def validate_stable_device(cls, value: str) -> str:
        if not value.startswith("/dev/v4l/by-id/") or not value.endswith(
            "-video-index0"
        ):
            raise ValueError(
                "V4L2 device must use a stable /dev/v4l/by-id/...-video-index0 path"
            )
        return value

    @model_validator(mode="after")
    def validate_streams(self) -> "V4L2CameraConfig":
        if set(self.streams) != {StreamName.COLOR}:
            raise ValueError("left_wrist streams must contain only color")
        stream = self.streams[StreamName.COLOR]
        if (stream.capture_format, stream.encoding) != ("mjpeg", "jpeg"):
            raise ValueError("left_wrist color must use mjpeg/jpeg")
        object.__setattr__(self, "streams", MappingProxyType(dict(self.streams)))
        return self


class RealSenseCameraConfig(FrozenModel):
    driver: Literal["realsense"]
    serial: str = Field(min_length=1)
    streams: dict[StreamName, StreamConfig]

    @model_validator(mode="after")
    def freeze_streams(self) -> "RealSenseCameraConfig":
        object.__setattr__(self, "streams", MappingProxyType(dict(self.streams)))
        return self


CameraConfig: TypeAlias = Annotated[
    V4L2CameraConfig | RealSenseCameraConfig,
    Field(discriminator="driver"),
]


class CameraSetConfig(FrozenModel):
    left_wrist: V4L2CameraConfig
    right_wrist: RealSenseCameraConfig
    head: RealSenseCameraConfig

    @model_validator(mode="after")
    def validate_roles(self) -> "CameraSetConfig":
        if self.right_wrist.serial == self.head.serial:
            raise ValueError("RealSense serials must be unique")
        self._validate_realsense_streams(
            "right_wrist",
            self.right_wrist,
            {StreamName.COLOR, StreamName.DEPTH},
        )
        self._validate_realsense_streams(
            "head",
            self.head,
            {
                StreamName.COLOR,
                StreamName.DEPTH,
                StreamName.INFRARED_1,
                StreamName.INFRARED_2,
            },
        )
        return self

    @staticmethod
    def _validate_realsense_streams(
        role: str,
        camera: RealSenseCameraConfig,
        expected: set[StreamName],
    ) -> None:
        if set(camera.streams) != expected:
            raise ValueError(
                f"{role} streams must be exactly "
                f"{sorted(stream.value for stream in expected)}"
            )
        for name, stream in camera.streams.items():
            expected_format = {
                StreamName.COLOR: ("bgr8", "jpeg"),
                StreamName.DEPTH: ("z16", "raw"),
                StreamName.INFRARED_1: ("y8", "raw"),
                StreamName.INFRARED_2: ("y8", "raw"),
            }[name]
            actual = (stream.capture_format, stream.encoding)
            if actual != expected_format:
                raise ValueError(
                    f"{role} {name.value} must use "
                    f"{expected_format[0]}/{expected_format[1]}"
                )


class CameraBridgeConfig(FrozenModel):
    bind: str = Field(min_length=1)
    queue_size: Literal[2] = 2
    jpeg_quality: StrictInt = Field(default=90, ge=1, le=100)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    cameras: CameraSetConfig

    @field_validator("bind")
    @classmethod
    def validate_bind(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "tcp" or not parsed.hostname:
            raise ValueError("bind must be an absolute tcp:// endpoint")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("bind contains an invalid port") from exc
        if port is None or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("bind must include host and port only")
        return value

    def camera_items(self) -> tuple[tuple[CameraName, CameraConfig], ...]:
        return (
            (CameraName.LEFT_WRIST, self.cameras.left_wrist),
            (CameraName.RIGHT_WRIST, self.cameras.right_wrist),
            (CameraName.HEAD, self.cameras.head),
        )


def load_camera_config(path: Path) -> CameraBridgeConfig:
    """Load a camera YAML file without coercing unsafe scalar types."""

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse YAML configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"YAML configuration {path} must contain a mapping")
    return CameraBridgeConfig.model_validate(raw)
