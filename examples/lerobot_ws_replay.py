#!/usr/bin/env python3
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Replay bimanual LeRobot actions or state through Piper WebSockets."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Any, Literal

from websockets.exceptions import WebSocketException

if __package__:
    from examples.ws_client import PiperControlClient
else:  # Support ``python examples/lerobot_ws_replay.py``.
    from ws_client import PiperControlClient


ReplaySource = Literal["actions", "state"]
MOTION_CONFIRMATION = "I_UNDERSTAND"
LEFT_URL = "ws://127.0.0.1:8765/ws/control"
RIGHT_URL = "ws://127.0.0.1:9876/ws/control"
_DUPLICATE_GRIPPER_TOLERANCE_M = 5e-4


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    timestamp_s: float
    left_joints_deg: tuple[float, float, float, float, float, float]
    right_joints_deg: tuple[float, float, float, float, float, float]
    left_gripper_m: float
    right_gripper_m: float


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _finite_vector(value: object, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain exactly {length} values")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{label} must contain exactly {length} values") from exc
    if len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return tuple(
        _finite_float(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def decode_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source: ReplaySource,
    episode_index: int,
) -> tuple[ReplayFrame, ...]:
    """Validate selected LeRobot rows and convert Piper targets to API units."""

    if source not in {"actions", "state"}:
        raise ValueError("source must be 'actions' or 'state'")
    selected: list[Mapping[str, object]] = []
    for row in rows:
        raw_episode = row.get("episode_index")
        if isinstance(raw_episode, bool) or not isinstance(raw_episode, int):
            raise ValueError("episode_index must be an integer")
        if raw_episode == episode_index:
            selected.append(row)
    if not selected:
        raise ValueError(f"episode {episode_index} has no frames")

    selected.sort(key=lambda row: int(row.get("frame_index", -1)))
    frames: list[ReplayFrame] = []
    first_timestamp: float | None = None
    previous_timestamp: float | None = None
    previous_frame_index: int | None = None
    for row in selected:
        frame_index = row.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise ValueError("frame_index must be an integer")
        if previous_frame_index is not None and frame_index <= previous_frame_index:
            raise ValueError("frame_index values must be strictly increasing")
        timestamp = _finite_float(row.get("timestamp"), "timestamp")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        values = _finite_vector(row.get(source), 16, f"{source} frame {frame_index}")
        if abs(values[14] - values[6]) > _DUPLICATE_GRIPPER_TOLERANCE_M:
            raise ValueError(f"{source} frame {frame_index} duplicate left gripper differs")
        if abs(values[15] - values[13]) > _DUPLICATE_GRIPPER_TOLERANCE_M:
            raise ValueError(f"{source} frame {frame_index} duplicate right gripper differs")
        if first_timestamp is None:
            first_timestamp = timestamp
        frames.append(
            ReplayFrame(
                timestamp_s=timestamp - first_timestamp,
                left_joints_deg=tuple(math.degrees(value) for value in values[0:6]),  # type: ignore[arg-type]
                right_joints_deg=tuple(math.degrees(value) for value in values[7:13]),  # type: ignore[arg-type]
                left_gripper_m=values[6],
                right_gripper_m=values[13],
            )
        )
        previous_timestamp = timestamp
        previous_frame_index = frame_index
    return tuple(frames)


def load_trajectory(
    dataset: Path,
    *,
    source: ReplaySource,
    episode_index: int,
) -> tuple[ReplayFrame, ...]:
    """Read the selected columns from all Parquet shards in a LeRobot dataset."""

    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"LeRobot metadata is missing: {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read LeRobot metadata: {exc}") from exc
    metadata = _mapping(info, "metadata")
    if metadata.get("robot_type") != "piper":
        raise ValueError("metadata robot_type must be 'piper'")
    if metadata.get("export_schema_version") != "v2":
        raise ValueError("metadata export_schema_version must be 'v2'")
    features = _mapping(metadata.get("features"), "features")
    feature = _mapping(features.get(source), f"features.{source}")
    if feature.get("dtype") != "float32":
        raise ValueError(f"features.{source}.dtype must be 'float32'")
    shape = feature.get("shape")
    if shape != [16]:
        raise ValueError(f"features.{source}.shape must be [16]")

    files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise ValueError(f"LeRobot data contains no Parquet shards: {dataset / 'data'}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot replay requires pyarrow; install .[dataset]"
        ) from exc
    rows: list[Mapping[str, object]] = []
    columns = [source, "timestamp", "frame_index", "episode_index"]
    for path in files:
        try:
            rows.extend(parquet.read_table(path, columns=columns).to_pylist())
        except Exception as exc:
            raise ValueError(f"could not read Parquet shard {path}: {exc}") from exc
    return decode_rows(rows, source=source, episode_index=episode_index)


def _validate_side(
    side: str,
    frames: Sequence[ReplayFrame],
    info_value: object,
    state_value: object,
    *,
    joint_speed_deg_s: float,
    gripper_speed_m_s: float,
    rate: float,
) -> None:
    info = _mapping(info_value, f"{side} base info")
    state = _mapping(state_value, f"{side} state")
    if info.get("side") != side:
        raise ValueError(f"{side} WebSocket reported side {info.get('side')!r}")
    if state.get("state") != "RUNNING":
        raise ValueError(f"{side} service must be RUNNING")
    arm = _mapping(info.get("arm"), f"{side} arm")
    joint_limits = _mapping(arm.get("joint_limits_deg"), f"{side} joint limits")
    minimums = _finite_vector(joint_limits.get("min"), 6, f"{side} joint minimums")
    maximums = _finite_vector(joint_limits.get("max"), 6, f"{side} joint maximums")
    speed_info = _mapping(arm.get("joint_speed_deg_s"), f"{side} joint speeds")
    maximum_speeds = _finite_vector(speed_info.get("max"), 6, f"{side} maximum speeds")
    if any(joint_speed_deg_s > maximum for maximum in maximum_speeds):
        raise ValueError(f"{side} joint speed exceeds the advertised maximum")
    gripper = _mapping(arm.get("gripper"), f"{side} gripper")
    if gripper.get("enabled") is not True:
        raise ValueError(f"{side} gripper must be enabled")
    position = _mapping(gripper.get("position_m"), f"{side} gripper position")
    gripper_min = _finite_float(position.get("min"), f"{side} gripper minimum")
    gripper_max = _finite_float(position.get("max"), f"{side} gripper maximum")
    gripper_speed = _mapping(gripper.get("speed_m_s"), f"{side} gripper speed")
    gripper_speed_max = _finite_float(
        gripper_speed.get("max"), f"{side} gripper maximum speed"
    )
    if gripper_speed_m_s > gripper_speed_max:
        raise ValueError(f"{side} gripper speed exceeds the advertised maximum")

    for frame_index, frame in enumerate(frames):
        joints = frame.left_joints_deg if side == "left" else frame.right_joints_deg
        gripper_target = frame.left_gripper_m if side == "left" else frame.right_gripper_m
        for joint_index, (target, minimum, maximum) in enumerate(
            zip(joints, minimums, maximums, strict=True), 1
        ):
            if not minimum <= target <= maximum:
                raise ValueError(
                    f"{side} frame {frame_index} joint {joint_index} target "
                    f"{target:g} is outside [{minimum:g}, {maximum:g}] degrees"
                )
        if not gripper_min <= gripper_target <= gripper_max:
            raise ValueError(
                f"{side} frame {frame_index} gripper target {gripper_target:g} "
                f"is outside [{gripper_min:g}, {gripper_max:g}] metres"
            )

    for frame_index, (previous, current) in enumerate(
        zip(frames, frames[1:]), 1
    ):
        delta_t = (current.timestamp_s - previous.timestamp_s) / rate
        previous_joints = (
            previous.left_joints_deg if side == "left" else previous.right_joints_deg
        )
        current_joints = (
            current.left_joints_deg if side == "left" else current.right_joints_deg
        )
        for joint_index, (before, after) in enumerate(
            zip(previous_joints, current_joints, strict=True), 1
        ):
            required_speed = abs(after - before) / delta_t
            if required_speed > joint_speed_deg_s + 1e-9:
                raise ValueError(
                    f"{side} frame {frame_index} joint {joint_index} requires "
                    f"{required_speed:g} deg/s but command speed is "
                    f"{joint_speed_deg_s:g} deg/s; lower --rate or raise "
                    "--joint-speed-deg-s"
                )
        previous_gripper = (
            previous.left_gripper_m
            if side == "left"
            else previous.right_gripper_m
        )
        current_gripper = (
            current.left_gripper_m if side == "left" else current.right_gripper_m
        )
        required_gripper_speed = abs(current_gripper - previous_gripper) / delta_t
        if required_gripper_speed > gripper_speed_m_s + 1e-9:
            raise ValueError(
                f"{side} frame {frame_index} gripper requires "
                f"{required_gripper_speed:g} m/s but command speed is "
                f"{gripper_speed_m_s:g} m/s; lower --rate or raise "
                "--gripper-speed-m-s"
            )


def validate_services(
    frames: Sequence[ReplayFrame],
    left_info: object,
    right_info: object,
    left_state: object,
    right_state: object,
    *,
    joint_speed_deg_s: float,
    gripper_speed_m_s: float,
    rate: float,
) -> None:
    if not frames:
        raise ValueError("trajectory must contain at least one frame")
    _validate_side(
        "left",
        frames,
        left_info,
        left_state,
        joint_speed_deg_s=joint_speed_deg_s,
        gripper_speed_m_s=gripper_speed_m_s,
        rate=rate,
    )
    _validate_side(
        "right",
        frames,
        right_info,
        right_state,
        joint_speed_deg_s=joint_speed_deg_s,
        gripper_speed_m_s=gripper_speed_m_s,
        rate=rate,
    )


async def _send_side(
    client: Any,
    joints: Sequence[float],
    gripper_m: float,
    joint_speed_deg_s: float,
    gripper_speed_m_s: float,
) -> None:
    await client.servo_j(joints, speed=(joint_speed_deg_s,) * 6)
    await client.move_gripper(gripper_m, speed=gripper_speed_m_s)


async def _send_frame(
    frame: ReplayFrame,
    left: Any,
    right: Any,
    joint_speed_deg_s: float,
    gripper_speed_m_s: float,
) -> None:
    await asyncio.gather(
        _send_side(
            left,
            frame.left_joints_deg,
            frame.left_gripper_m,
            joint_speed_deg_s,
            gripper_speed_m_s,
        ),
        _send_side(
            right,
            frame.right_joints_deg,
            frame.right_gripper_m,
            joint_speed_deg_s,
            gripper_speed_m_s,
        ),
    )


async def stream_frames(
    frames: Sequence[ReplayFrame],
    left: Any,
    right: Any,
    *,
    rate: float,
    joint_speed_deg_s: float,
    gripper_speed_m_s: float,
    io_timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> None:
    started_at = clock()
    for index, frame in enumerate(frames):
        if index:
            deadline = started_at + frame.timestamp_s / rate
            await sleep(max(0.0, deadline - clock()))
        await _wait_for_io(
            _send_frame(
                frame,
                left,
                right,
                joint_speed_deg_s,
                gripper_speed_m_s,
            ),
            io_timeout_s,
            f"frame {index} command",
        )


def _at_first_frame(
    frame: ReplayFrame,
    left_state: Mapping[str, object],
    right_state: Mapping[str, object],
    joint_tolerance_deg: float,
    gripper_tolerance_m: float,
) -> bool:
    arrived = True
    for side, state, target_joints, target_gripper in (
        ("left", left_state, frame.left_joints_deg, frame.left_gripper_m),
        ("right", right_state, frame.right_joints_deg, frame.right_gripper_m),
    ):
        if state.get("state") != "RUNNING":
            raise RuntimeError(f"{side} service left RUNNING state")
        actual = _finite_vector(state.get("ActualQ"), 6, f"{side} ActualQ")
        actual_gripper = _finite_float(
            state.get("ActualGripper"), f"{side} ActualGripper"
        )
        if any(
            abs(a - b) > joint_tolerance_deg
            for a, b in zip(actual, target_joints, strict=True)
        ):
            arrived = False
        if abs(actual_gripper - target_gripper) > gripper_tolerance_m:
            arrived = False
    return arrived


async def _wait_for_io(awaitable: Awaitable[Any], timeout_s: float, label: str) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout_s)
    except asyncio.TimeoutError:
        raise TimeoutError(f"{label} timed out after {timeout_s:g} seconds") from None


@asynccontextmanager
async def _client_pair(
    left_url: str,
    right_url: str,
    client_factory: Callable[[str], Any],
    io_timeout_s: float,
) -> AsyncIterator[tuple[Any, Any]]:
    entered: list[Any] = []
    try:
        left_context = client_factory(left_url)
        left = await _wait_for_io(
            left_context.__aenter__(), io_timeout_s, "left WebSocket connect"
        )
        entered.append(left_context)
        right_context = client_factory(right_url)
        right = await _wait_for_io(
            right_context.__aenter__(), io_timeout_s, "right WebSocket connect"
        )
        entered.append(right_context)
        yield left, right
    finally:
        if entered:
            await _wait_for_io(
                asyncio.gather(
                    *(context.__aexit__(None, None, None) for context in entered)
                ),
                io_timeout_s,
                "WebSocket disconnect",
            )


async def preposition(
    frame: ReplayFrame,
    left: Any,
    right: Any,
    *,
    joint_speed_deg_s: float,
    gripper_speed_m_s: float,
    timeout_s: float,
    io_timeout_s: float,
    joint_tolerance_deg: float,
    gripper_tolerance_m: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    await _wait_for_io(
        _send_frame(frame, left, right, joint_speed_deg_s, gripper_speed_m_s),
        min(io_timeout_s, max(0.0, deadline - time.monotonic())),
        "preposition command",
    )
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        left_state, right_state = await _wait_for_io(
            asyncio.gather(left.get_state(), right.get_state()),
            min(io_timeout_s, remaining),
            "preposition state read",
        )
        if _at_first_frame(
            frame,
            _mapping(left_state, "left state"),
            _mapping(right_state, "right state"),
            joint_tolerance_deg,
            gripper_tolerance_m,
        ):
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"arms did not reach the first frame within {timeout_s:g} seconds")


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay bimanual LeRobot actions or state to Piper WebSockets."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--source", choices=("actions", "state"), default="actions")
    parser.add_argument("--episode-index", type=_non_negative_int, default=0)
    parser.add_argument("--rate", type=_positive_float, default=1.0)
    parser.add_argument("--left-url", default=LEFT_URL)
    parser.add_argument("--right-url", default=RIGHT_URL)
    parser.add_argument("--joint-speed-deg-s", type=_positive_float, default=10.0)
    parser.add_argument("--gripper-speed-m-s", type=_positive_float, default=0.02)
    parser.add_argument("--preposition-timeout-s", type=_positive_float, default=30.0)
    parser.add_argument("--joint-tolerance-deg", type=_positive_float, default=2.0)
    parser.add_argument("--gripper-tolerance-m", type=_positive_float, default=0.005)
    parser.add_argument("--io-timeout-s", type=_positive_float, default=2.0)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--confirm-motion", metavar=MOTION_CONFIRMATION)
    return parser


async def async_main(
    args: argparse.Namespace,
    *,
    trajectory_loader: Callable[..., tuple[ReplayFrame, ...]] = load_trajectory,
    client_factory: Callable[[str], Any] = PiperControlClient,
) -> None:
    if args.allow_motion and args.confirm_motion != MOTION_CONFIRMATION:
        raise ValueError(
            f"motion requires --allow-motion --confirm-motion {MOTION_CONFIRMATION}"
        )
    frames = trajectory_loader(
        args.dataset,
        source=args.source,
        episode_index=args.episode_index,
    )
    duration = frames[-1].timestamp_s if frames else 0.0
    mode = "motion-enabled" if args.allow_motion else "dry-run"
    print(
        f"dataset={args.dataset} source={args.source} episode={args.episode_index} "
        f"frames={len(frames)} duration_s={duration:.3f} rate={args.rate:g} mode={mode}"
    )
    if not args.allow_motion:
        return

    async with _client_pair(
        args.left_url, args.right_url, client_factory, args.io_timeout_s
    ) as (left, right):
        left_info, right_info, left_state, right_state = await _wait_for_io(
            asyncio.gather(
                left.get_base_info(),
                right.get_base_info(),
                left.get_state(),
                right.get_state(),
            ),
            args.io_timeout_s,
            "startup state read",
        )
        validate_services(
            frames,
            left_info,
            right_info,
            left_state,
            right_state,
            joint_speed_deg_s=args.joint_speed_deg_s,
            gripper_speed_m_s=args.gripper_speed_m_s,
            rate=args.rate,
        )
        print("prepositioning both arms to the first frame")
        await preposition(
            frames[0],
            left,
            right,
            joint_speed_deg_s=args.joint_speed_deg_s,
            gripper_speed_m_s=args.gripper_speed_m_s,
            timeout_s=args.preposition_timeout_s,
            io_timeout_s=args.io_timeout_s,
            joint_tolerance_deg=args.joint_tolerance_deg,
            gripper_tolerance_m=args.gripper_tolerance_m,
        )
        print("replaying trajectory")
        await stream_frames(
            frames,
            left,
            right,
            rate=args.rate,
            joint_speed_deg_s=args.joint_speed_deg_s,
            gripper_speed_m_s=args.gripper_speed_m_s,
            io_timeout_s=args.io_timeout_s,
        )
        print("replay complete")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(async_main(args))
    except (OSError, RuntimeError, TimeoutError, ValueError, WebSocketException) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
