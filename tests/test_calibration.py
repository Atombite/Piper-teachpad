# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from io import StringIO

import pytest
import yaml

import piper_control.calibration_cli as calibration_cli

from piper_control.calibration_cli import (
    CalibrationError,
    calibration_mode_mdeg,
    render_calibration_yaml,
    run_calibration,
)
from piper_control.config import (
    ArmConfig,
    ArmSide,
    GripperConfig,
    ServerConfig,
    ServiceConfig,
)
from piper_control.types import RawJointFeedback


JOINTS = (1000, 2000, -3000, 4000, -5000, 6000)


def make_arm(side: ArmSide = ArmSide.RIGHT) -> ArmConfig:
    return ArmConfig(
        server=ServerConfig(host="127.0.0.1", port=8765 if side is ArmSide.RIGHT else 9876),
        control_hz=100,
        feedback_timeout_s=0.5,
        shutdown_home_timeout_s=10,
        name=f"{side.value}_arm",
        can_port="can0" if side is ArmSide.RIGHT else "can1",
        auto_initialize=True,
        calibration_joints_pos_deg=(0, 0, 0, 0, 0, 0),
        reset_joints_pos_deg=(0, 0, 0, 0, 0, 0),
        min_joints_deg=(-100, 0, -100, -50, -50, -50),
        max_joints_deg=(100, 100, 0, 50, 50, 50),
        max_joints_speed_deg_s=(30, 30, 30, 30, 30, 30),
        default_joints_speed_deg_s=(10, 10, 10, 10, 10, 10),
        gripper=GripperConfig(
            enabled=False,
            min_position_m=0,
            max_position_m=0.07,
            max_speed_m_s=0.05,
            default_speed_m_s=0.02,
            effort_nm=1,
        ),
    )


def make_config(*arms: ArmConfig, timeout: float = 0.5) -> ServiceConfig:
    selected = arms or (make_arm(),)
    sides = (ArmSide.RIGHT, ArmSide.LEFT)[: len(selected)]
    return ServiceConfig(arms={side: arm.model_copy(update={"feedback_timeout_s": timeout}) for side, arm in zip(sides, selected, strict=True)})


def frame(
    timestamp: float,
    joints: tuple[int, int, int, int, int, int] = JOINTS,
) -> RawJointFeedback:
    return RawJointFeedback(
        joints_mdeg=joints,
        connected=True,
        fault_code=0,
        stale=False,
        received_at_s=timestamp,
    )


class FakeRawDriver:
    def __init__(
        self,
        frames: list[RawJointFeedback],
        transform: Callable[[RawJointFeedback], RawJointFeedback] | None = None,
    ) -> None:
        self.frames = frames
        self.transform = transform or (lambda value: value)
        self.read_count = 0
        self.connect_count = 0

    def connect(self) -> None:
        self.connect_count += 1

    def read_raw_joint_feedback(self) -> RawJointFeedback:
        index = min(self.read_count, len(self.frames) - 1)
        self.read_count += 1
        return self.transform(self.frames[index])

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"calibration accessed unexpected driver member {name}")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


def test_calibration_mode_uses_per_joint_mode_and_tied_median() -> None:
    samples = [JOINTS] * 24
    samples += [(1002, 2000, -3000, 4000, -5000, 6000)] * 25
    samples += [(1000, 2000, -3000, 4000, -5000, 6000)]

    assert calibration_mode_mdeg(samples) == (1001, 2000, -3000, 4000, -5000, 6000)


def test_single_arm_session_uses_50_zero_frames_then_records_logical_range() -> None:
    zero_frames = [frame(float(index), JOINTS) for index in range(1, 51)]
    range_frames = [
        frame(51.0, (0, 1000, -4000, 3000, -6000, 5000)),
        frame(52.0, (2500, 3500, -2000, 5000, -4000, 8000)),
    ]
    driver = FakeRawDriver(zero_frames + range_frames)

    report = run_calibration(
        make_config(make_arm()),
        {"right": driver},
        should_stop=lambda: driver.read_count >= 52,
        sleep=lambda _: None,
    )

    result = report.arms["right"]
    assert driver.connect_count == 1
    assert result.calibration_mdeg == JOINTS
    assert result.min_joints_mdeg == (-1000, -1000, -1000, -1000, -1000, -1000)
    assert result.max_joints_mdeg == (1500, 1500, 1000, 1000, 1000, 2000)
    assert result.range_frame_count == 2


def test_two_arm_calibration_waits_for_both_zero_sets_and_keeps_results_separate() -> None:
    right = FakeRawDriver([frame(float(index), JOINTS) for index in range(1, 52)])
    left_zero = tuple(value + 100 for value in JOINTS)
    left = FakeRawDriver(
        [frame(float(index), left_zero) for index in range(1, 51)]
        + [frame(51.0, tuple(value + 200 for value in left_zero))]
    )
    config = make_config(make_arm(), make_arm(ArmSide.LEFT))

    report = run_calibration(
        config,
        {"right": right, "left": left},
        should_stop=lambda: right.read_count >= 51 and left.read_count >= 51,
        sleep=lambda _: None,
    )

    assert list(report.arms) == ["right", "left"]
    assert report.arms["right"].calibration_mdeg == JOINTS
    assert report.arms["left"].calibration_mdeg == left_zero
    assert all(arm.range_frame_count == 1 for arm in report.arms.values())


def test_duplicate_joint_frames_timeout_instead_of_counting_as_samples() -> None:
    clock = FakeClock()
    driver = FakeRawDriver([frame(1.0)])

    with pytest.raises(CalibrationError, match="new joint frame"):
        run_calibration(
            make_config(make_arm(), timeout=0.03),
            {"right": driver},
            should_stop=lambda: False,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_initial_stale_sdk_frames_are_ignored_until_first_valid_frame() -> None:
    initial_stale = replace(frame(1.0), stale=True, received_at_s=0.0)
    valid_frames = [frame(float(index), JOINTS) for index in range(1, 52)]
    driver = FakeRawDriver([initial_stale] + valid_frames)

    report = run_calibration(
        make_config(make_arm()),
        {"right": driver},
        should_stop=lambda: driver.read_count >= 52,
        sleep=lambda _: None,
    )

    assert report.arms["right"].calibration_mdeg == JOINTS
    assert report.arms["right"].range_frame_count == 1


def test_initial_stale_sdk_frames_still_fail_at_feedback_timeout() -> None:
    clock = FakeClock()
    initial_stale = replace(frame(1.0), stale=True, received_at_s=0.0)
    driver = FakeRawDriver([initial_stale])

    with pytest.raises(CalibrationError, match="new joint frame"):
        run_calibration(
            make_config(make_arm(), timeout=0.03),
            {"right": driver},
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_stale_feedback_after_first_valid_frame_fails_immediately() -> None:
    stale = replace(frame(2.0), stale=True)
    driver = FakeRawDriver([frame(1.0), stale])

    with pytest.raises(CalibrationError, match="stale"):
        run_calibration(
            make_config(make_arm()),
            {"right": driver},
            sleep=lambda _: None,
        )


def test_new_frame_returned_after_wall_clock_deadline_is_rejected() -> None:
    clock = FakeClock()

    class BlockingDriver(FakeRawDriver):
        def read_raw_joint_feedback(self) -> RawJointFeedback:
            clock.now += 0.04
            return super().read_raw_joint_feedback()

    driver = BlockingDriver([frame(float(index)) for index in range(1, 51)])

    with pytest.raises(CalibrationError, match="new joint frame"):
        run_calibration(
            make_config(make_arm(), timeout=0.03),
            {"right": driver},
            should_stop=lambda: False,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
    assert driver.read_count == 1


@pytest.mark.parametrize(
    ("stuck_side", "timeout", "expected_reads"),
    [("right", 0.2, 21), ("left", 0.4, 20)],
)
def test_each_arm_uses_its_own_rate_and_feedback_timeout(
    stuck_side: str, timeout: float, expected_reads: int
) -> None:
    clock = FakeClock()
    right_arm = make_arm().model_copy(
        update={"control_hz": 100, "feedback_timeout_s": 0.2}
    )
    left_arm = make_arm(ArmSide.LEFT).model_copy(
        update={"control_hz": 50, "feedback_timeout_s": 0.4}
    )
    config = ServiceConfig(
        arms={ArmSide.RIGHT: right_arm, ArmSide.LEFT: left_arm}
    )
    advancing = [frame(float(index)) for index in range(1, 100)]
    drivers = {
        "right": FakeRawDriver([frame(1.0)] if stuck_side == "right" else advancing),
        "left": FakeRawDriver([frame(1.0)] if stuck_side == "left" else advancing),
    }

    with pytest.raises(CalibrationError, match=f"{stuck_side}_arm.*new joint frame"):
        run_calibration(
            config,
            drivers,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert drivers[stuck_side].read_count == expected_reads


def test_ctrl_c_during_range_collection_finishes_with_collected_result() -> None:
    driver = FakeRawDriver(
        [frame(float(index), JOINTS) for index in range(1, 51)]
        + [frame(51.0, (0, 2000, -3000, 4000, -5000, 6000))]
    )

    def interrupt_after_range(_: float) -> None:
        if driver.read_count >= 51:
            raise KeyboardInterrupt

    report = run_calibration(
        make_config(make_arm()),
        {"right": driver},
        sleep=interrupt_after_range,
    )

    assert report.arms["right"].range_frame_count == 1
    assert report.arms["right"].min_joints_mdeg[0] == -1000


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: replace(value, connected=False), "disconnected"),
        (lambda value: replace(value, fault_code=7), "fault"),
    ],
)
def test_unhealthy_feedback_aborts_without_result(
    change: Callable[[RawJointFeedback], RawJointFeedback], message: str
) -> None:
    driver = FakeRawDriver([frame(1.0)], change)

    with pytest.raises(CalibrationError, match=message):
        run_calibration(
            make_config(make_arm()),
            {"right": driver},
            should_stop=lambda: False,
            sleep=lambda _: None,
        )


def test_yaml_contains_copyable_degree_vectors_and_raw_metadata() -> None:
    driver = FakeRawDriver(
        [frame(float(index), JOINTS) for index in range(1, 51)]
        + [frame(51.0, (0, 2000, -3000, 4000, -5000, 6000))]
    )
    report = run_calibration(
        make_config(make_arm()),
        {"right": driver},
        should_stop=lambda: driver.read_count >= 51,
        sleep=lambda _: None,
    )

    data = yaml.safe_load(render_calibration_yaml(report))

    assert data["arms"]["right"]["name"] == "right_arm"
    assert "side" not in data["arms"]["right"]
    assert data["arms"]["right"]["can_port"] == "can0"
    assert data["arms"]["right"]["calibration_joints_pos_deg"] == [1, 2, -3, 4, -5, 6]
    assert data["arms"]["right"]["min_joints_deg"] == [-1, 0, 0, 0, 0, 0]
    assert data["arms"]["right"]["max_joints_deg"] == [0, 0, 0, 0, 0, 0]
    assert data["calibration_metadata"]["right"]["range_frame_count"] == 1
    assert data["calibration_metadata"]["right"]["raw_min_joints_mdeg"] == [0, 2000, -3000, 4000, -5000, 6000]


def test_cli_failure_returns_nonzero_and_keeps_stdout_empty(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config = make_config(make_arm())
    config_path.write_text(
        yaml.safe_dump(
            {
                "arms": {
                    side.value: arm.model_dump(mode="json")
                    for side, arm in config.items()
                }
            }
        ),
        encoding="utf-8",
    )
    driver = FakeRawDriver(
        [frame(1.0)], lambda value: replace(value, fault_code=9)
    )
    monkeypatch.setattr(calibration_cli, "PiperArmDriver", lambda _arm: driver)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = calibration_cli.main(
        ["--config", str(config_path)], stdout=stdout, stderr=stderr
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "fault 9" in stderr.getvalue()
