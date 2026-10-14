# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from piper_control.camera_bridge.config import (
    CameraName,
    StreamName,
    V4L2CameraConfig,
)
from piper_control.camera_bridge.sources.base import (
    DeviceDisconnected,
    FatalFrameError,
    FrameTimeout,
)
from piper_control.camera_bridge.sources import create_source
from piper_control.camera_bridge.sources.v4l2 import V4L2Source


DEVICE = "/dev/v4l/by-id/usb-DECXIN_CAMERA-video-index0"


def config() -> V4L2CameraConfig:
    return V4L2CameraConfig.model_validate(
        {
            "driver": "v4l2",
            "device": DEVICE,
            "streams": {
                "color": {
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "capture_format": "mjpeg",
                    "encoding": "jpeg",
                }
            },
        }
    )


class FakeCapture:
    def __init__(self, reads: list[tuple[bool, object]], *, opened: bool = True) -> None:
        self.reads = deque(reads)
        self.opened = opened
        self.set_calls: list[tuple[int, float]] = []
        self.release_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def read(self) -> tuple[bool, object]:
        return self.reads.popleft() if self.reads else (False, None)

    def release(self) -> None:
        self.release_count += 1


class FakeCV2:
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 6
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    IMWRITE_JPEG_QUALITY = 1

    def __init__(self, capture: FakeCapture, *, encode_ok: bool = True) -> None:
        self.capture = capture
        self.encode_ok = encode_ok
        self.open_calls: list[tuple[str, int]] = []
        self.encode_calls: list[tuple[str, tuple[int, ...], list[int]]] = []

    def VideoCapture(self, device: str, backend: int) -> FakeCapture:
        self.open_calls.append((device, backend))
        return self.capture

    @staticmethod
    def VideoWriter_fourcc(*characters: str) -> int:
        assert characters == ("M", "J", "P", "G")
        return 1196444237

    def imencode(self, extension: str, frame: np.ndarray, params: list[int]):
        self.encode_calls.append((extension, frame.shape, params))
        return self.encode_ok, np.frombuffer(b"encoded-jpeg", dtype=np.uint8)


def source(
    capture: FakeCapture,
    *,
    path_exists=lambda _path: True,
    encode_ok: bool = True,
) -> tuple[V4L2Source, FakeCV2]:
    cv2 = FakeCV2(capture, encode_ok=encode_ok)
    adapter = V4L2Source(
        CameraName.LEFT_WRIST,
        config(),
        jpeg_quality=91,
        cv2_module=cv2,
        path_exists=path_exists,
        monotonic_ns=lambda: 100,
        unix_ns=lambda: 200,
    )
    return adapter, cv2


def test_open_requests_mjpeg_profile_and_returns_encoded_first_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame)])
    adapter, cv2 = source(capture)

    first = adapter.open(first_frame_timeout_ms=200)

    assert adapter.identity == DEVICE
    assert cv2.open_calls == [(DEVICE, cv2.CAP_V4L2)]
    assert capture.set_calls == [
        (cv2.CAP_PROP_FOURCC, 1196444237.0),
        (cv2.CAP_PROP_FRAME_WIDTH, 640.0),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480.0),
        (cv2.CAP_PROP_FPS, 30.0),
    ]
    assert first.camera is CameraName.LEFT_WRIST
    assert first.frameset_sequence == 0
    assert first.host_monotonic_ns == 100
    assert first.host_unix_ns == 200
    assert first.device_timestamp_ms is None
    assert first.frames[0].stream is StreamName.COLOR
    assert first.frames[0].payload == b"encoded-jpeg"
    assert cv2.encode_calls == [(".jpg", frame.shape, [cv2.IMWRITE_JPEG_QUALITY, 91])]


def test_read_advances_sequence_and_close_is_idempotent() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame), (True, frame)])
    adapter, _cv2 = source(capture)
    adapter.open(200)
    assert adapter.read(200).frameset_sequence == 1
    adapter.close()
    adapter.close()
    assert capture.release_count == 1


def test_missing_path_or_closed_capture_is_device_disconnected() -> None:
    capture = FakeCapture([], opened=False)
    adapter, _cv2 = source(capture, path_exists=lambda _path: False)
    with pytest.raises(DeviceDisconnected, match="left_wrist"):
        adapter.open(200)


def test_presence_probe_tracks_stable_device_path() -> None:
    present = [True]
    adapter, _cv2 = source(
        FakeCapture([]), path_exists=lambda _path: present[0]
    )

    assert adapter.is_present() is True
    present[0] = False
    assert adapter.is_present() is False


def test_empty_read_is_timeout_while_device_still_exists() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame), (False, None)])
    adapter, _cv2 = source(capture)
    adapter.open(200)
    with pytest.raises(FrameTimeout, match="200"):
        adapter.read(200)


@pytest.mark.parametrize(
    ("frame", "encode_ok", "message"),
    [
        (np.zeros((240, 320, 3), dtype=np.uint8), True, "dimensions"),
        (np.zeros((480, 640, 3), dtype=np.uint8), False, "JPEG"),
    ],
)
def test_wrong_dimensions_or_jpeg_failure_is_fatal(
    frame: np.ndarray, encode_ok: bool, message: str
) -> None:
    capture = FakeCapture([(True, frame)])
    adapter, _cv2 = source(capture, encode_ok=encode_ok)
    with pytest.raises(FatalFrameError, match=message):
        adapter.open(200)


def test_source_factory_maps_v4l2_config_to_left_wrist() -> None:
    adapter = create_source(config(), jpeg_quality=87)
    assert isinstance(adapter, V4L2Source)
    assert adapter.camera is CameraName.LEFT_WRIST
    assert adapter.jpeg_quality == 87
