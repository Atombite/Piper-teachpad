#!/usr/bin/env python3
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Read-only health and freshness validator for camera-bridge streams."""

from __future__ import annotations

import argparse
from collections import defaultdict
import time
from typing import Callable, Sequence

import cv2
import numpy as np

from piper_control.camera_bridge.client import CameraSubscriber
from piper_control.camera_bridge.protocol import ReceivedFrame


EXPECTED_TOPICS = (
    "camera/left_wrist/color",
    "camera/right_wrist/color",
    "camera/right_wrist/depth",
    "camera/head/color",
    "camera/head/depth",
    "camera/head/infrared_1",
    "camera/head/infrared_2",
)


def validate_payload(frame: ReceivedFrame) -> None:
    if frame.encoding == "jpeg":
        decoded = cv2.imdecode(
            np.frombuffer(frame.payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None or decoded.shape != (frame.height, frame.width, 3):
            raise ValueError("JPEG decode or dimensions failed")
        return
    bytes_per_pixel = 2 if frame.pixel_format == "z16" else 1
    minimum_stride = frame.width * bytes_per_pixel
    if frame.stride_bytes is None or frame.stride_bytes < minimum_stride:
        raise ValueError("raw stride is smaller than one image row")
    if len(frame.payload) != frame.stride_bytes * frame.height:
        raise ValueError("raw payload size does not match geometry")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate camera-bridge topics, rates, and frame age"
    )
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:15555")
    parser.add_argument(
        "--topic",
        action="append",
        help="exact expected image topic; repeat to select multiple (default: all seven)",
    )
    parser.add_argument("--max-age-ms", type=_positive_int, default=500)
    parser.add_argument("--duration", type=_positive_float, default=5.0)
    parser.add_argument("--poll-timeout-ms", type=_positive_int, default=100)
    return parser


def monitor_subscriber(
    subscriber: CameraSubscriber,
    *,
    expected_topics: Sequence[str],
    duration_s: float,
    max_age_ms: int,
    poll_timeout_ms: int,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> int:
    """Observe current frames and return nonzero for missing, stale, or faulted data."""

    expected = tuple(dict.fromkeys(expected_topics))
    counts: defaultdict[str, int] = defaultdict(int)
    first_received_ns: dict[str, int] = {}
    last_received_ns: dict[str, int] = {}
    latest: dict[str, ReceivedFrame] = {}
    sessions: dict[str, object] = {}
    stale_seen = False
    invalid_seen = False
    deadline_ns = monotonic_ns() + int(duration_s * 1_000_000_000)

    while monotonic_ns() < deadline_ns:
        frames = subscriber.recv_latest(poll_timeout_ms)
        received_ns = monotonic_ns()
        for topic, frame in frames.items():
            if topic not in expected:
                continue
            try:
                validate_payload(frame)
            except ValueError as exc:
                invalid_seen = True
                print(f"WARNING invalid payload topic={topic} reason={exc}")
                continue
            previous_session = sessions.get(topic)
            if previous_session is not None and previous_session != frame.session_id:
                print(
                    f"WARNING session_changed topic={topic} "
                    f"old={previous_session} new={frame.session_id}"
                )
            sessions[topic] = frame.session_id
            age_ms = (received_ns - frame.host_monotonic_ns) / 1_000_000
            if age_ms > max_age_ms:
                stale_seen = True
                print(
                    f"WARNING stale topic={topic} age_ms={age_ms:.1f} "
                    f"limit_ms={max_age_ms}"
                )
            counts[topic] += 1
            first_received_ns.setdefault(topic, received_ns)
            last_received_ns[topic] = received_ns
            latest[topic] = frame

    failed = stale_seen or invalid_seen
    summary_ns = monotonic_ns()
    for topic in expected:
        frame = latest.get(topic)
        if frame is None:
            failed = True
            print(f"MISSING {topic}")
            continue
        span_ns = last_received_ns[topic] - first_received_ns[topic]
        fps = (
            (counts[topic] - 1) * 1_000_000_000 / span_ns
            if counts[topic] > 1 and span_ns > 0
            else 0.0
        )
        age_ms = (summary_ns - frame.host_monotonic_ns) / 1_000_000
        if age_ms > max_age_ms:
            failed = True
        print(
            f"{topic} fps={fps:.1f} age_ms={age_ms:.1f} "
            f"session={frame.session_id} sequence={frame.sequence} "
            f"size={frame.width}x{frame.height} encoding={frame.encoding} "
            f"pixel_format={frame.pixel_format}"
        )

    expected_status_topics = {
        f"camera/status/{topic.split('/')[1]}" for topic in expected
    }
    for topic in sorted(expected_status_topics):
        status = subscriber.statuses.get(topic)
        if status is None:
            failed = True
            print(f"MISSING STATUS {topic}")
            continue
        state = status.get("state", "unknown")
        print(f"STATUS {topic} state={state}")
        if state != "online":
            failed = True
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_topics = tuple(args.topic or EXPECTED_TOPICS)
    status_topics = tuple(
        sorted({f"camera/status/{topic.split('/')[1]}" for topic in expected_topics})
    )
    subscriber = CameraSubscriber(
        args.endpoint, topics=(*expected_topics, *status_topics)
    )
    subscriber.open()
    try:
        return monitor_subscriber(
            subscriber,
            expected_topics=expected_topics,
            duration_s=args.duration,
            max_age_ms=args.max_age_ms,
            poll_timeout_ms=args.poll_timeout_ms,
        )
    finally:
        subscriber.close()


if __name__ == "__main__":
    raise SystemExit(main())
