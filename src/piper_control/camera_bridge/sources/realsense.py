# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Intel RealSense D405/D455 source bound by serial number."""

from __future__ import annotations

from collections.abc import Callable
import logging
import time
from typing import Any

import numpy as np

from piper_control.camera_bridge.config import (
    CameraName,
    RealSenseCameraConfig,
    StreamName,
)
from piper_control.camera_bridge.frames import CameraFrameSet, EncodedFrame
from piper_control.camera_bridge.sources.base import (
    DeviceDisconnected,
    FatalFrameError,
    FrameTimeout,
)


LOGGER = logging.getLogger(__name__)


class RealSenseSource:
    def __init__(
        self,
        camera: CameraName,
        config: RealSenseCameraConfig,
        *,
        jpeg_quality: int,
        rs_module: Any | None = None,
        cv2_module: Any | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.camera = camera
        self.config = config
        self.jpeg_quality = jpeg_quality
        self._rs = rs_module
        self._cv2 = cv2_module
        self._monotonic_ns = monotonic_ns
        self._unix_ns = unix_ns
        self._pipeline: Any | None = None
        self._depth_scale: float | None = None

    @property
    def identity(self) -> str:
        return self.config.serial

    def is_present(self) -> bool:
        return self._device_present()

    def open(self, first_frame_timeout_ms: int) -> CameraFrameSet:
        if self._pipeline is not None:
            return self.read(first_frame_timeout_ms)
        rs = self._load_rs()
        if not self.is_present():
            raise DeviceDisconnected(
                f"{self.camera.value} RealSense serial {self.config.serial} is absent"
            )
        sdk_config = rs.config()
        sdk_config.enable_device(self.config.serial)
        for name in (
            StreamName.COLOR,
            StreamName.DEPTH,
            StreamName.INFRARED_1,
            StreamName.INFRARED_2,
        ):
            if name not in self.config.streams:
                continue
            stream = self.config.streams[name]
            if name is StreamName.COLOR:
                sdk_config.enable_stream(
                    rs.stream.color,
                    stream.width,
                    stream.height,
                    rs.format.bgr8,
                    stream.fps,
                )
            elif name is StreamName.DEPTH:
                sdk_config.enable_stream(
                    rs.stream.depth,
                    stream.width,
                    stream.height,
                    rs.format.z16,
                    stream.fps,
                )
            else:
                index = 1 if name is StreamName.INFRARED_1 else 2
                sdk_config.enable_stream(
                    rs.stream.infrared,
                    index,
                    stream.width,
                    stream.height,
                    rs.format.y8,
                    stream.fps,
                )
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(sdk_config)
        except Exception as exc:
            raise DeviceDisconnected(
                f"{self.camera.value} failed to open RealSense "
                f"{self.config.serial}: {exc}"
            ) from exc
        self._pipeline = pipeline
        self._depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        if self._depth_scale <= 0:
            self.close()
            raise FatalFrameError(f"{self.camera.value} invalid depth scale")
        try:
            return self.read(first_frame_timeout_ms)
        except Exception:
            self.close()
            raise

    def read(self, timeout_ms: int) -> CameraFrameSet:
        pipeline = self._pipeline
        if pipeline is None:
            raise DeviceDisconnected(f"{self.camera.value} camera is not open")
        try:
            frameset = pipeline.wait_for_frames(timeout_ms)
        except Exception as exc:
            if not self.is_present():
                raise DeviceDisconnected(
                    f"{self.camera.value} RealSense {self.config.serial} disconnected: {exc}"
                ) from exc
            raise FrameTimeout(
                f"{self.camera.value} returned no frameset within {timeout_ms} ms: {exc}"
            ) from exc
        host_monotonic_ns = self._monotonic_ns()
        host_unix_ns = self._unix_ns()
        encoded: list[EncodedFrame] = []
        device_timestamp_ms: float | None = None
        for name in (
            StreamName.COLOR,
            StreamName.DEPTH,
            StreamName.INFRARED_1,
            StreamName.INFRARED_2,
        ):
            if name not in self.config.streams:
                continue
            sdk_frame = self._sdk_frame(frameset, name)
            if not sdk_frame:
                raise FrameTimeout(
                    f"{self.camera.value} frameset is missing {name.value}"
                )
            timestamp = float(sdk_frame.get_timestamp())
            if device_timestamp_ms is None:
                device_timestamp_ms = timestamp
            stream = self.config.streams[name]
            data = np.asanyarray(sdk_frame.get_data())
            if name is StreamName.COLOR:
                self._validate_array(data, stream.height, stream.width, 3, np.uint8, "color")
                cv2 = self._load_cv2()
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    data,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    raise FatalFrameError(
                        f"{self.camera.value} color JPEG encoding failed"
                    )
                encoded.append(
                    EncodedFrame(
                        stream=name,
                        payload=bytes(jpeg),
                        width=stream.width,
                        height=stream.height,
                        encoding="jpeg",
                        pixel_format="bgr8",
                        device_timestamp_ms=timestamp,
                    )
                )
            elif name is StreamName.DEPTH:
                self._validate_array(data, stream.height, stream.width, None, np.uint16, "depth")
                depth = np.asarray(data, dtype="<u2", order="C")
                encoded.append(
                    EncodedFrame(
                        stream=name,
                        payload=depth.tobytes(order="C"),
                        width=stream.width,
                        height=stream.height,
                        encoding="raw",
                        pixel_format="z16",
                        device_timestamp_ms=timestamp,
                        dtype="uint16",
                        byte_order="little",
                        stride_bytes=stream.width * 2,
                        depth_scale=self._depth_scale,
                    )
                )
            else:
                self._validate_array(data, stream.height, stream.width, None, np.uint8, name.value)
                infrared = np.asarray(data, dtype=np.uint8, order="C")
                encoded.append(
                    EncodedFrame(
                        stream=name,
                        payload=infrared.tobytes(order="C"),
                        width=stream.width,
                        height=stream.height,
                        encoding="raw",
                        pixel_format="y8",
                        device_timestamp_ms=timestamp,
                        dtype="uint8",
                        stride_bytes=stream.width,
                    )
                )
        return CameraFrameSet(
            camera=self.camera,
            frameset_sequence=int(frameset.get_frame_number()),
            host_monotonic_ns=host_monotonic_ns,
            host_unix_ns=host_unix_ns,
            device_timestamp_ms=device_timestamp_ms,
            frames=tuple(encoded),
        )

    def read_startup(self, timeout_ms: int) -> CameraFrameSet:
        return self.read(timeout_ms)

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        self._depth_scale = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                LOGGER.warning(
                    "camera=%s identity=%s RealSense pipeline stop failed: %r",
                    self.camera.value,
                    self.identity,
                    exc,
                )

    def _load_rs(self) -> Any:
        if self._rs is None:
            import pyrealsense2 as rs

            self._rs = rs
        return self._rs

    def _load_cv2(self) -> Any:
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        return self._cv2

    def _device_present(self) -> bool:
        rs = self._load_rs()
        return any(
            device.get_info(rs.camera_info.serial_number) == self.config.serial
            for device in rs.context().query_devices()
        )

    @staticmethod
    def _sdk_frame(frameset: Any, name: StreamName) -> Any:
        if name is StreamName.COLOR:
            return frameset.get_color_frame()
        if name is StreamName.DEPTH:
            return frameset.get_depth_frame()
        return frameset.get_infrared_frame(
            1 if name is StreamName.INFRARED_1 else 2
        )

    def _validate_array(
        self,
        data: np.ndarray,
        height: int,
        width: int,
        channels: int | None,
        dtype: Any,
        label: str,
    ) -> None:
        expected_shape = (height, width, channels) if channels else (height, width)
        if data.shape != expected_shape:
            raise FatalFrameError(
                f"{self.camera.value} {label} dimensions mismatch: {data.shape}"
            )
        if data.dtype != dtype:
            raise FatalFrameError(
                f"{self.camera.value} {label} format mismatch: {data.dtype}"
            )
