# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path

import pytest

from examples.lerobot_ws_replay import (
    MOTION_CONFIRMATION,
    ReplayFrame,
    _at_first_frame,
    async_main,
    build_parser,
    decode_rows,
    load_trajectory,
    stream_frames,
    validate_services,
)


def _vector(offset: float) -> list[float]:
    return [
        offset + 0.1,
        offset + 0.2,
        offset + 0.3,
        offset + 0.4,
        offset + 0.5,
        offset + 0.6,
        0.031,
        offset - 0.1,
        offset - 0.2,
        offset - 0.3,
        offset - 0.4,
        offset - 0.5,
        offset - 0.6,
        0.072,
        0.031,
        0.072,
    ]


def _rows() -> list[dict[str, object]]:
    return [
        {
            "timestamp": 2.0,
            "frame_index": 10,
            "episode_index": 3,
            "actions": _vector(0.0),
            "state": _vector(1.0),
        },
        {
            "timestamp": 2.1,
            "frame_index": 11,
            "episode_index": 3,
            "actions": _vector(0.01),
            "state": _vector(1.01),
        },
        {
            "timestamp": 0.0,
            "frame_index": 0,
            "episode_index": 4,
            "actions": _vector(2.0),
            "state": _vector(3.0),
        },
    ]


@pytest.mark.parametrize(("source", "offset"), [("actions", 0.0), ("state", 1.0)])
def test_decode_rows_selects_source_and_maps_bimanual_targets(source, offset):
    frames = decode_rows(_rows(), source=source, episode_index=3)

    assert len(frames) == 2
    assert frames[0].timestamp_s == pytest.approx(0.0)
    assert frames[1].timestamp_s == pytest.approx(0.1)
    assert frames[0].left_joints_deg == pytest.approx(
        tuple(math.degrees(offset + value) for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    )
    assert frames[0].right_joints_deg == pytest.approx(
        tuple(math.degrees(offset - value) for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    )
    assert frames[0].left_gripper_m == pytest.approx(0.031)
    assert frames[0].right_gripper_m == pytest.approx(0.072)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["actions"].__setitem__(15, 0.04), "duplicate right gripper"),
        (lambda row: row["actions"].__setitem__(0, float("nan")), "finite"),
        (lambda row: row.__setitem__("actions", [0.0] * 15), "16 values"),
    ],
)
def test_decode_rows_rejects_malformed_selected_source(mutation, message):
    row = _rows()[0]
    mutation(row)

    with pytest.raises(ValueError, match=message):
        decode_rows([row], source="actions", episode_index=3)


def test_decode_rows_rejects_missing_episode_and_non_increasing_timestamps():
    with pytest.raises(ValueError, match="episode 99"):
        decode_rows(_rows(), source="actions", episode_index=99)

    rows = _rows()[:2]
    rows[1]["timestamp"] = 2.0
    with pytest.raises(ValueError, match="strictly increasing"):
        decode_rows(rows, source="actions", episode_index=3)


def test_decode_rows_accepts_submillimetre_duplicate_state_sampling_skew():
    row = _rows()[0]
    row["state"][14] += 0.0003
    row["state"][15] -= 0.0003

    frame = decode_rows([row], source="state", episode_index=3)[0]

    assert frame.left_gripper_m == pytest.approx(0.031)
    assert frame.right_gripper_m == pytest.approx(0.072)


def _service_info(side: str) -> dict[str, object]:
    return {
        "side": side,
        "arm": {
            "joint_limits_deg": {"min": [-180.0] * 6, "max": [180.0] * 6},
            "joint_speed_deg_s": {"max": [30.0] * 6},
            "gripper": {
                "enabled": True,
                "position_m": {"min": 0.0, "max": 0.1},
                "speed_m_s": {"max": 0.05},
            },
        },
    }


def test_validate_services_checks_every_frame_before_motion():
    frames = (
        ReplayFrame(0.0, (0.0,) * 6, (0.0,) * 6, 0.02, 0.03),
        ReplayFrame(0.1, (181.0,) + (0.0,) * 5, (0.0,) * 6, 0.02, 0.03),
    )

    with pytest.raises(ValueError, match=r"left frame 1 joint 1.*outside"):
        validate_services(
            frames,
            _service_info("left"),
            _service_info("right"),
            {"state": "RUNNING"},
            {"state": "RUNNING"},
            joint_speed_deg_s=10.0,
            gripper_speed_m_s=0.02,
            rate=1.0,
        )


def test_validate_services_rejects_trajectory_faster_than_command_speed():
    frames = (
        ReplayFrame(0.0, (0.0,) * 6, (0.0,) * 6, 0.02, 0.03),
        ReplayFrame(0.1, (2.0,) + (0.0,) * 5, (0.0,) * 6, 0.023, 0.03),
    )

    with pytest.raises(ValueError, match=r"left frame 1 joint 1 requires 20.*10"):
        validate_services(
            frames,
            _service_info("left"),
            _service_info("right"),
            {"state": "RUNNING"},
            {"state": "RUNNING"},
            joint_speed_deg_s=10.0,
            gripper_speed_m_s=0.02,
            rate=1.0,
        )

    validate_services(
        frames,
        _service_info("left"),
        _service_info("right"),
        {"state": "RUNNING"},
        {"state": "RUNNING"},
        joint_speed_deg_s=10.0,
        gripper_speed_m_s=0.02,
        rate=0.5,
    )


def test_load_trajectory_rejects_unproven_robot_layout(tmp_path):
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "info.json").write_text(
        '{"robot_type":"other","export_schema_version":"v2",'
        '"features":{"actions":{"dtype":"float32","shape":[16]}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="robot_type must be 'piper'"):
        load_trajectory(tmp_path, source="actions", episode_index=0)


def test_first_frame_check_does_not_hide_other_arm_fault():
    frame = ReplayFrame(0.0, (10.0,) * 6, (0.0,) * 6, 0.01, 0.02)

    with pytest.raises(RuntimeError, match="right service left RUNNING"):
        _at_first_frame(
            frame,
            {"state": "RUNNING", "ActualQ": [0.0] * 6, "ActualGripper": 0.01},
            {"state": "EMERGENCY_STOP", "ActualQ": [0.0] * 6, "ActualGripper": 0.02},
            1.0,
            0.001,
        )


class _RecordingClient:
    def __init__(self, side: str) -> None:
        self.side = side
        self.calls: list[tuple[str, object, object]] = []

    async def servo_j(self, joints, speed=None):
        self.calls.append(("joints", tuple(joints), tuple(speed or ())))

    async def move_gripper(self, position, speed=None):
        self.calls.append(("gripper", position, speed))


@pytest.mark.asyncio
async def test_stream_frames_preserves_timestamps_and_sends_grippers_separately():
    frames = (
        ReplayFrame(0.0, (1.0,) * 6, (2.0,) * 6, 0.01, 0.02),
        ReplayFrame(0.1, (3.0,) * 6, (4.0,) * 6, 0.03, 0.04),
    )
    left = _RecordingClient("left")
    right = _RecordingClient("right")
    now = [50.0]
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    await stream_frames(
        frames,
        left,
        right,
        rate=2.0,
        joint_speed_deg_s=12.0,
        gripper_speed_m_s=0.03,
        io_timeout_s=1.0,
        clock=lambda: now[0],
        sleep=sleep,
    )

    assert sleeps == pytest.approx([0.05])
    assert left.calls == [
        ("joints", (1.0,) * 6, (12.0,) * 6),
        ("gripper", 0.01, 0.03),
        ("joints", (3.0,) * 6, (12.0,) * 6),
        ("gripper", 0.03, 0.03),
    ]
    assert right.calls == [
        ("joints", (2.0,) * 6, (12.0,) * 6),
        ("gripper", 0.02, 0.03),
        ("joints", (4.0,) * 6, (12.0,) * 6),
        ("gripper", 0.04, 0.03),
    ]


@pytest.mark.asyncio
async def test_stream_frames_times_out_stalled_websocket_command():
    class StalledClient(_RecordingClient):
        async def servo_j(self, joints, speed=None):
            await asyncio.Event().wait()

    frame = ReplayFrame(0.0, (0.0,) * 6, (0.0,) * 6, 0.01, 0.02)

    with pytest.raises(TimeoutError, match="frame 0 command timed out"):
        await stream_frames(
            (frame,),
            StalledClient("left"),
            _RecordingClient("right"),
            rate=1.0,
            joint_speed_deg_s=10.0,
            gripper_speed_m_s=0.02,
            io_timeout_s=0.01,
        )


def test_parser_defaults_to_actions_and_accepts_state():
    parser = build_parser()
    actions = parser.parse_args(["dataset"])
    state = parser.parse_args(["dataset", "--source", "state"])

    assert actions.source == "actions"
    assert state.source == "state"
    with pytest.raises(SystemExit):
        parser.parse_args(["dataset", "--source", "gello"])


@pytest.mark.asyncio
async def test_dry_run_loads_and_reports_without_opening_websockets(capsys):
    frames = (ReplayFrame(0.0, (0.0,) * 6, (0.0,) * 6, 0.01, 0.02),)
    args = argparse.Namespace(
        dataset=Path("dataset"),
        source="state",
        episode_index=0,
        rate=1.0,
        left_url="ws://left",
        right_url="ws://right",
        joint_speed_deg_s=10.0,
        gripper_speed_m_s=0.02,
        preposition_timeout_s=30.0,
        joint_tolerance_deg=2.0,
        gripper_tolerance_m=0.005,
        io_timeout_s=2.0,
        allow_motion=False,
        confirm_motion=None,
    )

    def reject_client(_url: str):
        raise AssertionError("dry-run must not open a WebSocket")

    await async_main(
        args,
        trajectory_loader=lambda *_args, **_kwargs: frames,
        client_factory=reject_client,
    )

    output = capsys.readouterr().out
    assert "source=state" in output
    assert "frames=1" in output
    assert "dry-run" in output


@pytest.mark.asyncio
async def test_motion_requires_exact_confirmation_before_websocket_connection():
    args = build_parser().parse_args(
        ["dataset", "--allow-motion", "--confirm-motion", "NO"]
    )

    with pytest.raises(ValueError, match=MOTION_CONFIRMATION):
        await async_main(
            args,
            trajectory_loader=lambda *_args, **_kwargs: (),
            client_factory=lambda _url: pytest.fail("must not connect"),
        )


@pytest.mark.asyncio
async def test_motion_times_out_stalled_websocket_connection():
    class StalledContext:
        async def __aenter__(self):
            await asyncio.Event().wait()

        async def __aexit__(self, *_exc):
            return None

    args = build_parser().parse_args(
        [
            "dataset",
            "--allow-motion",
            "--confirm-motion",
            MOTION_CONFIRMATION,
            "--io-timeout-s",
            "0.01",
        ]
    )
    frame = ReplayFrame(0.0, (0.0,) * 6, (0.0,) * 6, 0.01, 0.02)

    with pytest.raises(TimeoutError, match="left WebSocket connect timed out"):
        await asyncio.wait_for(
            async_main(
                args,
                trajectory_loader=lambda *_args, **_kwargs: (frame,),
                client_factory=lambda _url: StalledContext(),
            ),
            0.1,
        )
