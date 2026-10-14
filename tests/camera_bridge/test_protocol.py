# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import json
from uuid import UUID

import pytest

from piper_control.camera_bridge.config import CameraName, StreamName
from piper_control.camera_bridge.frames import CameraFrameSet, EncodedFrame
from piper_control.camera_bridge.protocol import (
    PROTOCOL_VERSION,
    CameraProtocolError,
    decode_frame_message,
    encode_frame_message,
    frame_topic,
)


SESSION = UUID("12345678-1234-5678-1234-567812345678")


def frameset(frame: EncodedFrame) -> CameraFrameSet:
    return CameraFrameSet(
        camera=CameraName.HEAD,
        frameset_sequence=41,
        host_monotonic_ns=123456789,
        host_unix_ns=1791512345678900000,
        device_timestamp_ms=1234.5,
        frames=(frame,),
    )


def jpeg_frame() -> EncodedFrame:
    return EncodedFrame(
        stream=StreamName.COLOR,
        payload=b"\xff\xd8jpeg\xff\xd9",
        width=640,
        height=480,
        encoding="jpeg",
        pixel_format="bgr8",
        device_timestamp_ms=1234.5,
    )


def depth_frame() -> EncodedFrame:
    return EncodedFrame(
        stream=StreamName.DEPTH,
        payload=b"\x01\x00\x02\x00",
        width=2,
        height=1,
        encoding="raw",
        pixel_format="z16",
        device_timestamp_ms=1234.5,
        dtype="uint16",
        byte_order="little",
        stride_bytes=4,
        depth_scale=0.001,
    )


def test_frame_topic_is_exact_ascii_path() -> None:
    assert frame_topic(CameraName.HEAD, StreamName.DEPTH) == b"camera/head/depth"


def test_jpeg_message_has_required_fields_and_omits_raw_only_fields() -> None:
    frame = jpeg_frame()
    parts = encode_frame_message(SESSION, 7, frameset(frame), frame)

    assert parts[0] == b"camera/head/color"
    assert parts[2] == frame.payload
    assert b" " not in parts[1]
    metadata = json.loads(parts[1])
    assert metadata == {
        "protocol_version": 1,
        "session_id": str(SESSION),
        "camera": "head",
        "stream": "color",
        "sequence": 7,
        "frameset_sequence": 41,
        "host_monotonic_ns": 123456789,
        "host_unix_ns": 1791512345678900000,
        "device_timestamp_ms": 1234.5,
        "width": 640,
        "height": 480,
        "encoding": "jpeg",
        "pixel_format": "bgr8",
        "payload_size": len(frame.payload),
    }


def test_z16_and_y8_metadata_describe_raw_memory_without_inference() -> None:
    depth = depth_frame()
    depth_metadata = json.loads(
        encode_frame_message(SESSION, 2, frameset(depth), depth)[1]
    )
    assert depth_metadata["dtype"] == "uint16"
    assert depth_metadata["byte_order"] == "little"
    assert depth_metadata["stride_bytes"] == 4
    assert depth_metadata["depth_scale"] == 0.001

    infrared = EncodedFrame(
        stream=StreamName.INFRARED_1,
        payload=b"\x01\x02",
        width=2,
        height=1,
        encoding="raw",
        pixel_format="y8",
        dtype="uint8",
        stride_bytes=2,
    )
    infrared_metadata = json.loads(
        encode_frame_message(SESSION, 3, frameset(infrared), infrared)[1]
    )
    assert infrared_metadata["dtype"] == "uint8"
    assert "byte_order" not in infrared_metadata
    assert "depth_scale" not in infrared_metadata


def test_round_trip_returns_validated_immutable_received_frame() -> None:
    original = depth_frame()
    received = decode_frame_message(
        encode_frame_message(SESSION, 9, frameset(original), original)
    )
    assert received.topic == "camera/head/depth"
    assert received.session_id == SESSION
    assert received.camera is CameraName.HEAD
    assert received.stream is StreamName.DEPTH
    assert received.sequence == 9
    assert received.payload == original.payload
    assert received.depth_scale == 0.001


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda parts: parts[:2], "three parts"),
        (
            lambda parts: [parts[0], b"not-json", parts[2]],
            "metadata JSON",
        ),
        (
            lambda parts: replace_metadata(parts, protocol_version=2),
            "protocol_version",
        ),
        (
            lambda parts: [b"camera/right_wrist/depth", parts[1], parts[2]],
            "topic",
        ),
        (
            lambda parts: replace_metadata(parts, payload_size=999),
            "payload_size",
        ),
        (
            lambda parts: remove_metadata(parts, "width"),
            "width",
        ),
    ],
)
def test_rejects_malformed_or_inconsistent_messages(mutate, message: str) -> None:
    original = list(
        encode_frame_message(SESSION, 1, frameset(depth_frame()), depth_frame())
    )
    with pytest.raises(CameraProtocolError, match=message):
        decode_frame_message(mutate(original))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"stride_bytes": 1}, "stride"),
        ({"payload_size": 2}, "raw payload"),
        ({"depth_scale": float("inf")}, "finite"),
        ({"device_timestamp_ms": float("nan")}, "finite"),
        ({"stream": "color"}, "stream"),
    ],
)
def test_rejects_impossible_raw_geometry_and_nonfinite_metadata(
    updates: dict[str, object], message: str
) -> None:
    original = list(
        encode_frame_message(SESSION, 1, frameset(depth_frame()), depth_frame())
    )
    mutated = replace_metadata(original, **updates)
    if updates.get("payload_size") == 2:
        mutated[2] = mutated[2][:2]
    if updates.get("stream") == "color":
        mutated[0] = b"camera/head/color"

    with pytest.raises(CameraProtocolError, match=message):
        decode_frame_message(mutated)


def replace_metadata(parts: list[bytes], **updates: object) -> list[bytes]:
    metadata = json.loads(parts[1])
    metadata.update(updates)
    return [parts[0], json.dumps(metadata).encode(), parts[2]]


def remove_metadata(parts: list[bytes], key: str) -> list[bytes]:
    metadata = json.loads(parts[1])
    metadata.pop(key)
    return [parts[0], json.dumps(metadata).encode(), parts[2]]


def test_protocol_version_is_one() -> None:
    assert PROTOCOL_VERSION == 1
