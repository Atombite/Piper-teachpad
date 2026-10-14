# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""OpenCV V4L2 adapter for the left-wrist MJPEG camera."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time
from typing import Any

import numpy as np

from piper_control.camera_bridge.config import (
    CameraName,
    StreamName,
    V4L2CameraConfig,
)
from piper_control.camera_bridge.frames import CameraFrameSet, EncodedFrame
from piper_control.camera_bridge.sources.base import (
    DeviceDisconnected,
    FatalFrameError,
    FrameTimeout,
)


class V4L2Source:
    def __init__(
        self,
        camera: CameraName,
        config: V4L2CameraConfig,
        *,
        jpeg_quality: int,
        cv2_module: Any | None = None,
        path_exists: Callable[[str], bool] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.camera = camera
        self.config = config
        self.jpeg_quality = jpeg_quality
        self._cv2 = cv2_module
        self._path_exists = path_exists or (lambda value: Path(value).exists())
        self._monotonic_ns = monotonic_ns
        self._unix_ns = unix_ns
        self._capture: Any | None = None
        self._sequence = 0

    @property
    def identity(self) -> str:
        return self.config.device

    def is_present(self) -> bool:
        return self._path_exists(self.config.device)

    def open(self, first_frame_timeout_ms: int) -> CameraFrameSet:
        if self._capture is not None:
            return self.read(first_frame_timeout_ms)
        if not self._path_exists(self.config.device):
            raise DeviceDisconnected(
                f"{self.camera.value} device is absent: {self.config.device}"
            )
        cv2 = self._cv2
        if cv2 is None:
            import cv2 as loaded_cv2

            cv2 = loaded_cv2
            self._cv2 = cv2
        capture = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise DeviceDisconnected(
                f"{self.camera.value} could not open {self.config.device}"
            )
        color = self.config.streams[StreamName.COLOR]
        capture.set(
            cv2.CAP_PROP_FOURCC,
            float(cv2.VideoWriter_fourcc("M", "J", "P", "G")),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(color.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(color.height))
        capture.set(cv2.CAP_PROP_FPS, float(color.fps))
        self._capture = capture
        self._sequence = 0
        try:
            return self.read(first_frame_timeout_ms)
        except Exception:
            self.close()
            raise

    def read(self, timeout_ms: int) -> CameraFrameSet:
        capture = self._capture
        if capture is None:
            raise DeviceDisconnected(f"{self.camera.value} camera is not open")
        ok, image = capture.read()
        if not ok or image is None:
            if not self._path_exists(self.config.device) or not capture.isOpened():
                raise DeviceDisconnected(
                    f"{self.camera.value} disconnected: {self.config.device}"
                )
            raise FrameTimeout(
                f"{self.camera.value} returned no frame within {timeout_ms} ms"
            )
        color = self.config.streams[StreamName.COLOR]
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.shape != (color.height, color.width, 3)
        ):
            actual = getattr(image, "shape", None)
            raise FatalFrameError(
                f"{self.camera.value} frame dimensions/format mismatch: {actual}"
            )
        cv2 = self._cv2
        assert cv2 is not None
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded_ok:
            raise FatalFrameError(f"{self.camera.value} JPEG encoding failed")
        sequence = self._sequence
        self._sequence += 1
        frame = EncodedFrame(
            stream=StreamName.COLOR,
            payload=bytes(encoded),
            width=color.width,
            height=color.height,
            encoding="jpeg",
            pixel_format="bgr8",
        )
        return CameraFrameSet(
            camera=self.camera,
            frameset_sequence=sequence,
            host_monotonic_ns=self._monotonic_ns(),
            host_unix_ns=self._unix_ns(),
            device_timestamp_ms=None,
            frames=(frame,),
        )

    def read_startup(self, timeout_ms: int) -> CameraFrameSet:
        return self.read(timeout_ms)

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
