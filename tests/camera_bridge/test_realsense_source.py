# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from piper_control.camera_bridge.config import (
    CameraName,
    RealSenseCameraConfig,
    StreamName,
)
from piper_control.camera_bridge.sources.base import (
    DeviceDisconnected,
    FatalFrameError,
    FrameTimeout,
)
from piper_control.camera_bridge.sources import create_source
from piper_control.camera_bridge.sources.realsense import RealSenseSource
from piper_control.camera_bridge.sources.v4l2 import V4L2Source


RIGHT_SERIAL = "RIGHT_WRIST_SERIAL_EXAMPLE"
HEAD_SERIAL = "HEAD_SERIAL_EXAMPLE"


def rs_config(serial: str, *, head: bool) -> RealSenseCameraConfig:
    streams: dict[str, object] = {
        "color": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "capture_format": "bgr8",
            "encoding": "jpeg",
        },
        "depth": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "capture_format": "z16",
            "encoding": "raw",
        },
    }
    if head:
        streams.update(
            {
                "infrared_1": {
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "capture_format": "y8",
                    "encoding": "raw",
                },
                "infrared_2": {
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "capture_format": "y8",
                    "encoding": "raw",
                },
            }
        )
    return RealSenseCameraConfig.model_validate(
        {"driver": "realsense", "serial": serial, "streams": streams}
    )


class FakeFrame:
    def __init__(self, data: np.ndarray, timestamp: float = 12.5) -> None:
        self.data = data
        self.timestamp = timestamp

    def get_data(self) -> np.ndarray:
        return self.data

    def get_timestamp(self) -> float:
        return self.timestamp


class FakeFrameSet:
    def __init__(
        self,
        *,
        color: FakeFrame | None,
        depth: FakeFrame | None,
        infrared_1: FakeFrame | None = None,
        infrared_2: FakeFrame | None = None,
        number: int = 9,
    ) -> None:
        self.color = color
        self.depth = depth
        self.infrared = {1: infrared_1, 2: infrared_2}
        self.number = number

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth

    def get_infrared_frame(self, index: int):
        return self.infrared[index]

    def get_frame_number(self) -> int:
        return self.number


class FakeDevice:
    def __init__(self, serial: str, depth_scale: float = 0.001) -> None:
        self.serial = serial
        self.depth_scale = depth_scale

    def get_info(self, _kind: object) -> str:
        return self.serial

    def first_depth_sensor(self):
        return SimpleNamespace(get_depth_scale=lambda: self.depth_scale)


class FakeConfig:
    def __init__(self) -> None:
        self.device: str | None = None
        self.streams: list[tuple[object, ...]] = []

    def enable_device(self, serial: str) -> None:
        self.device = serial

    def enable_stream(self, *args: object) -> None:
        self.streams.append(args)


class FakePipeline:
    def __init__(self, module: "FakeRS") -> None:
        self.module = module
        self.start_config: FakeConfig | None = None
        self.stop_count = 0
        self.stop_error: Exception | None = None

    def start(self, config: FakeConfig):
        self.start_config = config
        return SimpleNamespace(get_device=lambda: self.module.devices[0])

    def wait_for_frames(self, timeout_ms: int):
        self.module.wait_timeouts.append(timeout_ms)
        value = self.module.framesets.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    def stop(self) -> None:
        self.stop_count += 1
        if self.stop_error is not None:
            raise self.stop_error


class FakeRS:
    stream = SimpleNamespace(color="color", depth="depth", infrared="infrared")
    format = SimpleNamespace(bgr8="bgr8", z16="z16", y8="y8")
    camera_info = SimpleNamespace(serial_number="serial_number")

    def __init__(self, serial: str, framesets: list[object]) -> None:
        self.devices = [FakeDevice(serial)]
        self.framesets = deque(framesets)
        self.configs: list[FakeConfig] = []
        self.pipelines: list[FakePipeline] = []
        self.wait_timeouts: list[int] = []

    def context(self):
        return SimpleNamespace(query_devices=lambda: list(self.devices))

    def config(self) -> FakeConfig:
        config = FakeConfig()
        self.configs.append(config)
        return config

    def pipeline(self) -> FakePipeline:
        pipeline = FakePipeline(self)
        self.pipelines.append(pipeline)
        return pipeline


class FakeCV2:
    IMWRITE_JPEG_QUALITY = 1

    def imencode(self, _extension: str, _frame: np.ndarray, _params: list[int]):
        return True, np.frombuffer(b"jpeg", dtype=np.uint8)


def complete_frameset(*, head: bool, number: int = 9) -> FakeFrameSet:
    depth = np.zeros((480, 640), dtype="<u2")
    depth[0, :2] = (1, 513)
    infrared_1 = np.zeros((480, 640), dtype=np.uint8)
    infrared_1[0, :2] = (3, 4)
    infrared_2 = np.zeros((480, 640), dtype=np.uint8)
    infrared_2[0, :2] = (5, 6)
    return FakeFrameSet(
        color=FakeFrame(np.zeros((480, 640, 3), dtype=np.uint8)),
        depth=FakeFrame(depth),
        infrared_1=FakeFrame(infrared_1) if head else None,
        infrared_2=FakeFrame(infrared_2) if head else None,
        number=number,
    )


def make_source(
    camera: CameraName,
    config: RealSenseCameraConfig,
    rs: FakeRS,
) -> RealSenseSource:
    return RealSenseSource(
        camera,
        config,
        jpeg_quality=90,
        rs_module=rs,
        cv2_module=FakeCV2(),
        monotonic_ns=lambda: 100,
        unix_ns=lambda: 200,
    )


def test_d405_binds_serial_and_enables_only_color_and_depth() -> None:
    config = rs_config(RIGHT_SERIAL, head=False)
    rs = FakeRS(config.serial, [complete_frameset(head=False)])
    source = make_source(CameraName.RIGHT_WRIST, config, rs)

    first = source.open(200)

    sdk_config = rs.configs[0]
    assert sdk_config.device == RIGHT_SERIAL
    assert sdk_config.streams == [
        (rs.stream.color, 640, 480, rs.format.bgr8, 30),
        (rs.stream.depth, 640, 480, rs.format.z16, 30),
    ]
    assert [frame.stream for frame in first.frames] == [
        StreamName.COLOR,
        StreamName.DEPTH,
    ]


def test_d455_outputs_complete_correlated_frameset_and_raw_bytes() -> None:
    config = rs_config(HEAD_SERIAL, head=True)
    rs = FakeRS(config.serial, [complete_frameset(head=True)])
    source = make_source(CameraName.HEAD, config, rs)

    first = source.open(200)

    assert rs.configs[0].streams == [
        (rs.stream.color, 640, 480, rs.format.bgr8, 30),
        (rs.stream.depth, 640, 480, rs.format.z16, 30),
        (rs.stream.infrared, 1, 640, 480, rs.format.y8, 30),
        (rs.stream.infrared, 2, 640, 480, rs.format.y8, 30),
    ]
    assert first.frameset_sequence == 9
    assert first.host_monotonic_ns == 100
    assert first.host_unix_ns == 200
    assert first.device_timestamp_ms == 12.5
    assert [frame.stream for frame in first.frames] == [
        StreamName.COLOR,
        StreamName.DEPTH,
        StreamName.INFRARED_1,
        StreamName.INFRARED_2,
    ]
    assert first.frames[0].payload == b"jpeg"
    assert first.frames[1].payload[:4] == b"\x01\x00\x01\x02"
    assert first.frames[1].depth_scale == 0.001
    assert first.frames[2].payload[:2] == b"\x03\x04"
    assert first.frames[3].payload[:2] == b"\x05\x06"


def test_sdk_wait_uses_requested_timeout_and_close_is_idempotent() -> None:
    config = rs_config(RIGHT_SERIAL, head=False)
    rs = FakeRS(
        config.serial,
        [complete_frameset(head=False), complete_frameset(head=False, number=10)],
    )
    source = make_source(CameraName.RIGHT_WRIST, config, rs)
    source.open(250)
    source.read(300)
    source.close()
    source.close()
    assert rs.wait_timeouts == [250, 300]
    assert rs.pipelines[0].stop_count == 1


def test_close_does_not_replace_camera_fault_with_sdk_stop_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = rs_config(RIGHT_SERIAL, head=False)
    rs = FakeRS(config.serial, [complete_frameset(head=False)])
    source = make_source(CameraName.RIGHT_WRIST, config, rs)
    source.open(200)
    rs.pipelines[0].stop_error = RuntimeError(
        "stop() cannot be called before start()"
    )

    source.close()
    source.close()

    assert rs.pipelines[0].stop_count == 1
    assert "RealSense pipeline stop failed" in caplog.text


def test_absent_serial_and_runtime_disconnect_are_classified() -> None:
    config = rs_config(RIGHT_SERIAL, head=False)
    rs = FakeRS("different", [])
    source = make_source(CameraName.RIGHT_WRIST, config, rs)
    with pytest.raises(DeviceDisconnected, match=RIGHT_SERIAL):
        source.open(200)

    rs = FakeRS(config.serial, [complete_frameset(head=False), RuntimeError("USB")])
    source = make_source(CameraName.RIGHT_WRIST, config, rs)
    source.open(200)
    rs.devices.clear()
    with pytest.raises(DeviceDisconnected, match=RIGHT_SERIAL):
        source.read(200)


def test_presence_probe_uses_configured_sdk_serial() -> None:
    config = rs_config(RIGHT_SERIAL, head=False)
    rs = FakeRS(config.serial, [])
    source = make_source(CameraName.RIGHT_WRIST, config, rs)

    assert source.is_present() is True
    rs.devices.clear()
    assert source.is_present() is False


def test_sdk_timeout_while_serial_is_present_is_frame_timeout() -> None:
    config = rs_config(RIGHT_SERIAL, head=False)
    rs = FakeRS(
        config.serial,
        [complete_frameset(head=False), RuntimeError("Frame didn't arrive")],
    )
    source = make_source(CameraName.RIGHT_WRIST, config, rs)
    source.open(200)
    with pytest.raises(FrameTimeout, match="200"):
        source.read(200)


@pytest.mark.parametrize(
    ("frameset", "message"),
    [
        (
            FakeFrameSet(
                color=FakeFrame(np.zeros((240, 320, 3), dtype=np.uint8)),
                depth=FakeFrame(np.zeros((480, 640), dtype="<u2")),
            ),
            "color dimensions",
        ),
        (
            FakeFrameSet(
                color=FakeFrame(np.zeros((480, 640, 3), dtype=np.uint8)),
                depth=FakeFrame(np.zeros((480, 640), dtype=np.uint8)),
            ),
            "depth format",
        ),
    ],
)
def test_wrong_frame_format_is_fatal(
    frameset: FakeFrameSet, message: str
) -> None:
    config = rs_config(RIGHT_SERIAL, head=False)
    source = make_source(
        CameraName.RIGHT_WRIST,
        config,
        FakeRS(config.serial, [frameset]),
    )
    with pytest.raises(FatalFrameError, match=message):
        source.open(200)


def test_missing_runtime_stream_is_transient_and_valid_frames_resume() -> None:
    config = rs_config(RIGHT_SERIAL, head=False)
    incomplete = FakeFrameSet(
        color=FakeFrame(np.zeros((480, 640, 3), dtype=np.uint8)),
        depth=None,
    )
    rs = FakeRS(
        config.serial,
        [
            complete_frameset(head=False),
            incomplete,
            complete_frameset(head=False, number=10),
            incomplete,
        ],
    )
    source = make_source(CameraName.RIGHT_WRIST, config, rs)
    source.open(200)

    with pytest.raises(FrameTimeout, match="missing depth"):
        source.read(200)
    assert source.read(200).frameset_sequence == 10
    with pytest.raises(FrameTimeout, match="missing depth"):
        source.read(200)


def test_source_factory_maps_profiles_to_roles_without_using_device_serials() -> None:
    right = create_source(rs_config(RIGHT_SERIAL, head=False), jpeg_quality=88)
    head = create_source(rs_config(HEAD_SERIAL, head=True), jpeg_quality=89)

    assert isinstance(right, RealSenseSource)
    assert right.camera is CameraName.RIGHT_WRIST
    assert right.jpeg_quality == 88
    assert isinstance(head, RealSenseSource)
    assert head.camera is CameraName.HEAD
    assert head.jpeg_quality == 89
