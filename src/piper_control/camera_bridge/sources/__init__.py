# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Physical camera source adapters."""

from piper_control.camera_bridge.sources.base import (
    CameraSource,
    DeviceDisconnected,
    FatalFrameError,
    FrameTimeout,
)
from piper_control.camera_bridge.config import (
    CameraConfig,
    CameraName,
    StreamName,
    V4L2CameraConfig,
)


def create_source(camera_config: CameraConfig, *, jpeg_quality: int = 90) -> CameraSource:
    """Create the fixed-role source implied by a validated camera config."""

    if isinstance(camera_config, V4L2CameraConfig):
        from piper_control.camera_bridge.sources.v4l2 import V4L2Source

        return V4L2Source(
            CameraName.LEFT_WRIST,
            camera_config,
            jpeg_quality=jpeg_quality,
        )
    from piper_control.camera_bridge.sources.realsense import RealSenseSource

    camera = (
        CameraName.HEAD
        if StreamName.INFRARED_1 in camera_config.streams
        else CameraName.RIGHT_WRIST
    )
    return RealSenseSource(camera, camera_config, jpeg_quality=jpeg_quality)

__all__ = [
    "CameraSource",
    "DeviceDisconnected",
    "FatalFrameError",
    "FrameTimeout",
    "create_source",
]
