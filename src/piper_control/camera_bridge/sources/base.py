# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Common camera source boundary and typed failures."""

from __future__ import annotations

from typing import Protocol

from piper_control.camera_bridge.frames import CameraFrameSet


class CameraSourceError(RuntimeError):
    """Base error for camera acquisition."""


class FrameTimeout(CameraSourceError):
    """The device remains present but did not yield a complete frame in time."""


class DeviceDisconnected(CameraSourceError):
    """The configured physical device or driver connection disappeared."""


class FatalFrameError(CameraSourceError):
    """A deterministic profile, format, or encoding invariant failed."""


class CameraSource(Protocol):
    @property
    def identity(self) -> str: ...

    def open(self, first_frame_timeout_ms: int) -> CameraFrameSet: ...

    def read(self, timeout_ms: int) -> CameraFrameSet: ...

    def read_startup(self, timeout_ms: int) -> CameraFrameSet: ...

    def is_present(self) -> bool: ...

    def close(self) -> None: ...
