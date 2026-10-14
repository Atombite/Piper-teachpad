# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Versioned, language-neutral camera frame wire protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import Field, StrictInt, ValidationError, model_validator

from piper_control.camera_bridge.config import CameraName, StreamName
from piper_control.camera_bridge.frames import CameraFrameSet, EncodedFrame
from piper_control.config import FrozenModel


PROTOCOL_VERSION = 1
IMAGE_TOPIC_COUNT = 7
# HWM is per ZMQ peer connection, not per topic. Two complete seven-topic
# cycles prevent deterministic starvation of the streams published last.
NETWORK_HWM = IMAGE_TOPIC_COUNT * 2


class CameraProtocolError(ValueError):
    """A received camera message does not satisfy protocol version 1."""


class _FrameMetadata(FrozenModel):
    protocol_version: Literal[1]
    session_id: UUID
    camera: CameraName
    stream: StreamName
    sequence: StrictInt = Field(ge=0)
    frameset_sequence: StrictInt = Field(ge=0)
    host_monotonic_ns: StrictInt = Field(ge=0)
    host_unix_ns: StrictInt = Field(ge=0)
    device_timestamp_ms: float | None = None
    width: StrictInt = Field(gt=0)
    height: StrictInt = Field(gt=0)
    encoding: Literal["jpeg", "raw"]
    pixel_format: Literal["bgr8", "z16", "y8"]
    payload_size: StrictInt = Field(gt=0)
    dtype: Literal["uint8", "uint16"] | None = None
    byte_order: Literal["little"] | None = None
    stride_bytes: StrictInt | None = Field(default=None, gt=0)
    depth_scale: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_encoding_fields(self) -> "_FrameMetadata":
        if self.device_timestamp_ms is not None and not math.isfinite(
            self.device_timestamp_ms
        ):
            raise ValueError("device timestamp must be finite")
        raw_fields = (self.dtype, self.byte_order, self.stride_bytes, self.depth_scale)
        if self.encoding == "jpeg":
            if self.stream is not StreamName.COLOR:
                raise ValueError("jpeg stream must be color")
            if self.pixel_format != "bgr8" or any(
                value is not None for value in raw_fields
            ):
                raise ValueError("jpeg metadata cannot contain raw-only fields")
            return self
        if self.pixel_format == "z16":
            if self.stream is not StreamName.DEPTH:
                raise ValueError("z16 stream must be depth")
            if (
                self.dtype != "uint16"
                or self.byte_order != "little"
                or self.stride_bytes is None
                or self.depth_scale is None
            ):
                raise ValueError("z16 metadata requires uint16/little/stride/depth_scale")
            if not math.isfinite(self.depth_scale):
                raise ValueError("depth scale must be finite")
            row_bytes = self.width * 2
        elif self.pixel_format == "y8":
            if self.stream not in (StreamName.INFRARED_1, StreamName.INFRARED_2):
                raise ValueError("y8 stream must be infrared")
            if (
                self.dtype != "uint8"
                or self.byte_order is not None
                or self.stride_bytes is None
                or self.depth_scale is not None
            ):
                raise ValueError("y8 metadata requires uint8/stride")
            row_bytes = self.width
        else:
            raise ValueError("raw metadata requires z16 or y8 pixel format")
        assert self.stride_bytes is not None
        if self.stride_bytes < row_bytes:
            raise ValueError("raw stride is smaller than one image row")
        if self.payload_size != self.stride_bytes * self.height:
            raise ValueError("raw payload_size does not match stride and height")
        return self


@dataclass(frozen=True, slots=True)
class ReceivedFrame:
    topic: str
    session_id: UUID
    camera: CameraName
    stream: StreamName
    sequence: int
    frameset_sequence: int
    host_monotonic_ns: int
    host_unix_ns: int
    device_timestamp_ms: float | None
    width: int
    height: int
    encoding: str
    pixel_format: str
    payload: bytes
    dtype: str | None = None
    byte_order: str | None = None
    stride_bytes: int | None = None
    depth_scale: float | None = None


def frame_topic(camera: CameraName, stream: StreamName) -> bytes:
    return f"camera/{camera.value}/{stream.value}".encode("ascii")


def encode_frame_message(
    session_id: UUID,
    topic_sequence: int,
    frameset: CameraFrameSet,
    frame: EncodedFrame,
) -> tuple[bytes, bytes, bytes]:
    if frame not in frameset.frames:
        raise CameraProtocolError("frame does not belong to frameset")
    metadata = _FrameMetadata(
        protocol_version=PROTOCOL_VERSION,
        session_id=session_id,
        camera=frameset.camera,
        stream=frame.stream,
        sequence=topic_sequence,
        frameset_sequence=frameset.frameset_sequence,
        host_monotonic_ns=frameset.host_monotonic_ns,
        host_unix_ns=frameset.host_unix_ns,
        device_timestamp_ms=frameset.device_timestamp_ms,
        width=frame.width,
        height=frame.height,
        encoding=frame.encoding,
        pixel_format=frame.pixel_format,
        payload_size=len(frame.payload),
        dtype=frame.dtype,
        byte_order=frame.byte_order,
        stride_bytes=frame.stride_bytes,
        depth_scale=frame.depth_scale,
    )
    payload = _metadata_dict(metadata)
    return (
        frame_topic(frameset.camera, frame.stream),
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        frame.payload,
    )


def decode_frame_message(parts: Sequence[bytes]) -> ReceivedFrame:
    if len(parts) != 3:
        raise CameraProtocolError("frame message must contain exactly three parts")
    topic_bytes, metadata_bytes, payload = parts
    try:
        topic = topic_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CameraProtocolError("topic must be ASCII") from exc
    try:
        raw: Any = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CameraProtocolError("invalid metadata JSON") from exc
    if not isinstance(raw, dict):
        raise CameraProtocolError("metadata JSON must be an object")
    try:
        metadata = _FrameMetadata.model_validate(raw)
    except (ValidationError, TypeError, ValueError) as exc:
        raise CameraProtocolError(f"invalid frame metadata: {exc}") from None
    expected_topic = frame_topic(metadata.camera, metadata.stream).decode("ascii")
    if topic != expected_topic:
        raise CameraProtocolError(
            f"topic {topic!r} does not match metadata topic {expected_topic!r}"
        )
    if len(payload) != metadata.payload_size:
        raise CameraProtocolError(
            f"payload_size {metadata.payload_size} does not match {len(payload)} bytes"
        )
    return ReceivedFrame(
        topic=topic,
        session_id=metadata.session_id,
        camera=metadata.camera,
        stream=metadata.stream,
        sequence=metadata.sequence,
        frameset_sequence=metadata.frameset_sequence,
        host_monotonic_ns=metadata.host_monotonic_ns,
        host_unix_ns=metadata.host_unix_ns,
        device_timestamp_ms=metadata.device_timestamp_ms,
        width=metadata.width,
        height=metadata.height,
        encoding=metadata.encoding,
        pixel_format=metadata.pixel_format,
        payload=payload,
        dtype=metadata.dtype,
        byte_order=metadata.byte_order,
        stride_bytes=metadata.stride_bytes,
        depth_scale=metadata.depth_scale,
    )


def _metadata_dict(metadata: _FrameMetadata) -> dict[str, object]:
    return metadata.model_dump(mode="json", exclude_none=True)
