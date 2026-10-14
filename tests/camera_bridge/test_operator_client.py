# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from piper_control.camera_bridge.config import CameraName, StreamName
from piper_control.camera_bridge.protocol import ReceivedFrame


SCRIPT = Path(__file__).parents[2] / "examples" / "camera_zmq_client.py"
_JPEG_OK, _JPEG_ARRAY = cv2.imencode(
    ".jpg", np.zeros((480, 640, 3), dtype=np.uint8)
)
assert _JPEG_OK
JPEG_PAYLOAD = bytes(_JPEG_ARRAY)


def load_script():
    spec = importlib.util.spec_from_file_location("camera_zmq_client", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StepClock:
    def __init__(self, start_ns: int, step_ns: int = 10_000_000) -> None:
        self.value = start_ns - step_ns
        self.step = step_ns

    def __call__(self) -> int:
        self.value += self.step
        return self.value


class FakeSubscriber:
    def __init__(self, batches: list[dict[str, ReceivedFrame]]) -> None:
        self.batches = iter(batches)
        self.statuses: dict[str, dict[str, object]] = {}

    def recv_latest(self, _timeout_ms: int) -> dict[str, ReceivedFrame]:
        return next(self.batches, {})


def received(
    topic: str,
    *,
    session: UUID,
    host_ns: int = 980_000_000,
    sequence: int = 1,
) -> ReceivedFrame:
    _, camera_text, stream_text = topic.split("/")
    stream = StreamName(stream_text)
    raw = stream is not StreamName.COLOR
    infrared = "infrared" in topic
    payload = (
        bytes(640 * 480 * (1 if infrared else 2))
        if raw
        else JPEG_PAYLOAD
    )
    return ReceivedFrame(
        topic=topic,
        session_id=session,
        camera=CameraName(camera_text),
        stream=stream,
        sequence=sequence,
        frameset_sequence=sequence,
        host_monotonic_ns=host_ns,
        host_unix_ns=1,
        device_timestamp_ms=1.0,
        width=640,
        height=480,
        encoding="raw" if raw else "jpeg",
        pixel_format="y8" if "infrared" in topic else ("z16" if raw else "bgr8"),
        payload=payload,
        dtype="uint8" if infrared else ("uint16" if raw else None),
        byte_order="little" if stream is StreamName.DEPTH else None,
        stride_bytes=(640 if infrared else 1280) if raw else None,
        depth_scale=0.001 if stream is StreamName.DEPTH else None,
    )


def test_monitor_reports_all_seven_streams_fps_age_and_dimensions(capsys) -> None:
    module = load_script()
    session = UUID("12345678-1234-5678-1234-567812345678")
    first = {
        topic: received(topic, session=session)
        for topic in module.EXPECTED_TOPICS
    }
    second = {
        topic: received(topic, session=session, sequence=2, host_ns=1_020_000_000)
        for topic in module.EXPECTED_TOPICS
    }

    subscriber = FakeSubscriber([first, second])
    for camera in CameraName:
        subscriber.statuses[f"camera/status/{camera.value}"] = {"state": "online"}
    result = module.monitor_subscriber(
        subscriber,
        expected_topics=module.EXPECTED_TOPICS,
        duration_s=0.08,
        max_age_ms=100,
        poll_timeout_ms=1,
        monotonic_ns=StepClock(1_000_000_000),
    )

    output = capsys.readouterr().out
    assert result == 0
    assert all(topic in output for topic in module.EXPECTED_TOPICS)
    assert "fps=" in output
    assert "age_ms=" in output
    assert "640x480" in output
    assert "encoding=jpeg" in output


def test_monitor_warns_on_stale_frame_and_session_change(capsys) -> None:
    module = load_script()
    topic = "camera/left_wrist/color"
    old_session = UUID("12345678-1234-5678-1234-567812345678")
    new_session = UUID("22345678-1234-5678-1234-567812345678")
    subscriber = FakeSubscriber(
        [
            {topic: received(topic, session=old_session, host_ns=1)},
            {topic: received(topic, session=new_session, host_ns=1)},
        ]
    )

    result = module.monitor_subscriber(
        subscriber,
        expected_topics=(topic,),
        duration_s=0.05,
        max_age_ms=10,
        poll_timeout_ms=1,
        monotonic_ns=StepClock(1_000_000_000),
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "WARNING stale" in output
    assert "WARNING session_changed" in output


def test_monitor_returns_nonzero_for_missing_topic_or_fault_status(capsys) -> None:
    module = load_script()
    topic = "camera/left_wrist/color"
    missing = "camera/head/depth"
    session = UUID("12345678-1234-5678-1234-567812345678")
    subscriber = FakeSubscriber([{topic: received(topic, session=session)}])
    subscriber.statuses["camera/status/head"] = {"state": "fault"}

    result = module.monitor_subscriber(
        subscriber,
        expected_topics=(topic, missing),
        duration_s=0.03,
        max_age_ms=100,
        poll_timeout_ms=1,
        monotonic_ns=StepClock(1_000_000_000),
    )

    output = capsys.readouterr().out
    assert result == 1
    assert f"MISSING {missing}" in output
    assert "STATUS camera/status/head state=fault" in output


def test_parser_accepts_endpoint_and_repeatable_topic_filter() -> None:
    module = load_script()

    args = module.build_parser().parse_args(
        [
            "--endpoint",
            "tcp://robot:15555",
            "--topic",
            "camera/head/color",
            "--topic",
            "camera/head/depth",
            "--max-age-ms",
            "150",
            "--duration",
            "2",
        ]
    )

    assert args.endpoint == "tcp://robot:15555"
    assert args.topic == ["camera/head/color", "camera/head/depth"]
    assert args.max_age_ms == 150
    assert args.duration == 2


def test_monitor_rejects_invalid_jpeg_and_missing_status(capsys) -> None:
    module = load_script()
    topic = "camera/left_wrist/color"
    session = UUID("12345678-1234-5678-1234-567812345678")
    invalid = received(topic, session=session)
    invalid = replace(invalid, payload=b"not-a-jpeg")
    subscriber = FakeSubscriber([{topic: invalid}])

    result = module.monitor_subscriber(
        subscriber,
        expected_topics=(topic,),
        duration_s=0.03,
        max_age_ms=100,
        poll_timeout_ms=1,
        monotonic_ns=StepClock(1_000_000_000),
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "invalid payload" in output
    assert "MISSING STATUS camera/status/left_wrist" in output


def test_main_subscribes_to_image_and_status_topics(monkeypatch) -> None:
    module = load_script()
    captured: list[tuple[str, ...]] = []

    class CapturingSubscriber(FakeSubscriber):
        def __init__(self, _endpoint: str, *, topics) -> None:
            super().__init__([])
            captured.append(tuple(topics))

        def open(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(module, "CameraSubscriber", CapturingSubscriber)

    assert module.main(["--duration", "0.01", "--poll-timeout-ms", "1"]) == 1
    assert set(module.EXPECTED_TOPICS) < set(captured[0])
    assert {
        "camera/status/left_wrist",
        "camera/status/right_wrist",
        "camera/status/head",
    } <= set(captured[0])
