# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import pytest

from piper_control.camera_bridge.config import CameraName, StreamName
from piper_control.camera_bridge.protocol import ReceivedFrame


SCRIPT = Path(__file__).parents[2] / "examples" / "camera_preview_gui.py"
SESSION = UUID("12345678-1234-5678-1234-567812345678")


def load_script():
    spec = importlib.util.spec_from_file_location("camera_preview_gui", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame(
    *,
    stream: StreamName,
    payload: bytes,
    width: int,
    height: int,
    encoding: str,
    pixel_format: str,
    stride_bytes: int | None = None,
    depth_scale: float | None = None,
    sequence: int = 1,
) -> ReceivedFrame:
    return ReceivedFrame(
        topic=f"camera/head/{stream.value}",
        session_id=SESSION,
        camera=CameraName.HEAD,
        stream=stream,
        sequence=sequence,
        frameset_sequence=sequence,
        host_monotonic_ns=1,
        host_unix_ns=1,
        device_timestamp_ms=1.0,
        width=width,
        height=height,
        encoding=encoding,
        pixel_format=pixel_format,
        payload=payload,
        dtype="uint16" if pixel_format == "z16" else (
            "uint8" if pixel_format == "y8" else None
        ),
        byte_order="little" if pixel_format == "z16" else None,
        stride_bytes=stride_bytes,
        depth_scale=depth_scale,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("127.0.0.1:15555", "tcp://127.0.0.1:15555"),
        (" tcp://robot.local:6000 ", "tcp://robot.local:6000"),
    ],
)
def test_normalize_endpoint_accepts_ip_port_or_tcp_url(text: str, expected: str) -> None:
    module = load_script()

    assert module.normalize_endpoint(text) == expected


@pytest.mark.parametrize("text", ["", "robot", "robot:not-a-port", "robot:70000"])
def test_normalize_endpoint_rejects_invalid_address(text: str) -> None:
    module = load_script()

    with pytest.raises(ValueError):
        module.normalize_endpoint(text)


def test_cli_defaults_to_local_camera_bridge() -> None:
    module = load_script()

    assert module.build_parser().parse_args([]).endpoint == "127.0.0.1:15555"


def test_frame_to_rgb_decodes_jpeg_and_swaps_bgr_channels() -> None:
    module = load_script()
    bgr = np.full((8, 8, 3), (1, 20, 240), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok

    image = module.frame_to_rgb(
        frame(
            stream=StreamName.COLOR,
            payload=bytes(encoded),
            width=8,
            height=8,
            encoding="jpeg",
            pixel_format="bgr8",
        )
    )

    assert image.shape == (8, 8, 3)
    assert image.mean(axis=(0, 1)).tolist() == pytest.approx([240, 20, 1], abs=3)


def test_frame_to_rgb_maps_z16_depth_and_hides_zero_pixels() -> None:
    module = load_script()
    depth_with_padding = np.array([[0, 1000, 65535]], dtype="<u2")

    image = module.frame_to_rgb(
        frame(
            stream=StreamName.DEPTH,
            payload=depth_with_padding.tobytes(),
            width=2,
            height=1,
            encoding="raw",
            pixel_format="z16",
            stride_bytes=6,
            depth_scale=0.001,
        )
    )

    assert image.shape == (1, 2, 3)
    assert image[0, 0].tolist() == [0, 0, 0]
    assert image[0, 1].any()


def test_frame_to_rgb_expands_y8_with_padded_stride() -> None:
    module = load_script()
    infrared_with_padding = bytes([10, 200, 99])

    image = module.frame_to_rgb(
        frame(
            stream=StreamName.INFRARED_1,
            payload=infrared_with_padding,
            width=2,
            height=1,
            encoding="raw",
            pixel_format="y8",
            stride_bytes=3,
        )
    )

    assert image.tolist() == [[[10, 10, 10], [200, 200, 200]]]


def test_fps_tracker_uses_recent_sample_window() -> None:
    module = load_script()
    tracker = module.FpsTracker(window_s=2.0)

    tracker.record("camera/head/color", 0)
    tracker.record("camera/head/color", 500_000_000)
    tracker.record("camera/head/color", 1_000_000_000)
    assert tracker.fps("camera/head/color", 1_000_000_000) == pytest.approx(2.0)

    tracker.record("camera/head/color", 3_000_000_000)
    assert tracker.fps("camera/head/color", 3_000_000_000) == pytest.approx(0.5)
    assert tracker.fps("camera/head/color", 5_100_000_000) == 0.0


def test_preview_inbox_counts_received_frames_even_when_latest_overwrites() -> None:
    module = load_script()
    inbox = module.PreviewInbox()
    generation = inbox.begin()
    topic = "camera/head/color"

    for timestamp_ns, sequence in (
        (0, 1),
        (500_000_000, 2),
        (1_000_000_000, 5),
    ):
        inbox.put_frames(
            generation,
            {
                topic: frame(
                    stream=StreamName.COLOR,
                    payload=b"unused",
                    width=1,
                    height=1,
                    encoding="jpeg",
                    pixel_format="bgr8",
                    sequence=sequence,
                )
            },
            timestamp_ns,
        )

    frames, _messages = inbox.drain(generation)
    assert set(frames) == {topic}
    assert inbox.fps(topic, 1_000_000_000) == pytest.approx(4.0)
    assert inbox.fps(topic, 3_100_000_000) == 0.0


def test_preview_inbox_ignores_events_from_retired_generation() -> None:
    module = load_script()
    inbox = module.PreviewInbox()
    old_generation = inbox.begin()
    current_generation = inbox.begin()

    inbox.put_message(old_generation, "connected", "tcp://old:1")
    inbox.put_frames(old_generation, {"camera/head/color": object()}, 1)

    assert inbox.drain(current_generation) == ({}, [])


@pytest.mark.parametrize(
    ("statuses", "expected_kind", "expected_text"),
    [
        (
            {"camera/status/right_wrist": {"state": "reconnecting"}},
            "warning",
            "right_wrist reconnecting",
        ),
        (
            {
                "camera/status/head": {
                    "state": "fault",
                    "error_category": "frame_timeout",
                    "error": "no frames",
                }
            },
            "error",
            "head fault: frame_timeout: no frames",
        ),
        (
            {
                "camera/status/left_wrist": {"state": "online"},
                "camera/status/right_wrist": {"state": "online"},
                "camera/status/head": {"state": "online"},
            },
            "online",
            "All cameras online",
        ),
    ],
)
def test_summarize_camera_statuses_surfaces_non_online_state(
    statuses, expected_kind: str, expected_text: str
) -> None:
    module = load_script()

    assert module.summarize_camera_statuses(statuses) == (
        expected_kind,
        expected_text,
    )
