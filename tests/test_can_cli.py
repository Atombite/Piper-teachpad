# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest

from piper_control.can_cli import CanCommandError, can_down, can_status, can_up
from piper_control.config import ServiceConfig


def _config(*sides: str) -> ServiceConfig:
    selected = sides or ("right",)

    def arm(side: str) -> dict[str, object]:
        return {
        "server": {"host": "127.0.0.1", "port": 8765 if side == "right" else 9876},
        "control_hz": 100,
        "feedback_timeout_s": 0.5,
        "shutdown_home_timeout_s": 3,
        "name": f"{side}_arm",
        "can_port": "can0" if side == "right" else "can1",
        "auto_initialize": False,
        "calibration_joints_pos_deg": [0] * 6,
        "reset_joints_pos_deg": [0] * 6,
        "min_joints_deg": [-150, 0, -170, -100, -70, -120],
        "max_joints_deg": [150, 180, 0, 100, 70, 120],
        "max_joints_speed_deg_s": [30] * 6,
        "default_joints_speed_deg_s": [10] * 6,
        "gripper": {
            "enabled": True,
            "min_position_m": 0,
            "max_position_m": 0.07,
            "max_speed_m_s": 0.05,
            "default_speed_m_s": 0.02,
            "effort_nm": 1,
        },
        }
    return ServiceConfig.model_validate(
        {"arms": {side: arm(side) for side in selected}}
    )


@dataclass
class Result:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeRunner:
    def __init__(self, *, up: bool, bitrate: int | None) -> None:
        self.up = up
        self.bitrate = bitrate
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv):  # type: ignore[no-untyped-def]
        args = tuple(argv)
        self.calls.append(args)
        if args[:5] == ("ip", "-details", "-statistics", "link", "show"):
            flags = "UP,LOWER_UP" if self.up else "NOARP"
            bitrate = "" if self.bitrate is None else f" bitrate {self.bitrate}"
            return Result(stdout=f"2: can0: <{flags}> state {'UP' if self.up else 'DOWN'}\n    can state ERROR-ACTIVE{bitrate}\n")
        if args[-1] == "down":
            self.up = False
        elif args[-1] == "up":
            self.up = True
        elif "bitrate" in args:
            self.bitrate = int(args[-1])
        return Result()


def test_can_up_does_not_mutate_an_already_correct_interface() -> None:
    runner = FakeRunner(up=True, bitrate=1_000_000)

    assert can_up(_config(), runner) == 0

    assert runner.calls == [
        ("ip", "-details", "-statistics", "link", "show", "dev", "can0")
    ]


def test_can_up_reconfigures_wrong_bitrate_in_safe_order() -> None:
    runner = FakeRunner(up=True, bitrate=500_000)

    assert can_up(_config(), runner) == 0

    assert runner.calls[1:] == [
        ("ip", "link", "set", "dev", "can0", "down"),
        ("ip", "link", "set", "dev", "can0", "type", "can", "bitrate", "1000000"),
        ("ip", "link", "set", "dev", "can0", "up"),
    ]


def test_can_status_is_read_only_and_returns_kernel_details(capsys) -> None:
    runner = FakeRunner(up=False, bitrate=1_000_000)

    assert can_status(_config(), runner) == 0

    assert len(runner.calls) == 1
    assert "bitrate 1000000" in capsys.readouterr().out


def test_can_down_is_explicit_and_idempotent() -> None:
    runner = FakeRunner(up=True, bitrate=1_000_000)
    assert can_down(_config(), runner) == 0
    assert runner.calls[-1] == ("ip", "link", "set", "dev", "can0", "down")

    runner.calls.clear()
    assert can_down(_config(), runner) == 0
    assert len(runner.calls) == 1


def test_missing_interface_fails_without_mutation() -> None:
    class MissingRunner(FakeRunner):
        def run(self, argv):  # type: ignore[no-untyped-def]
            self.calls.append(tuple(argv))
            return Result(returncode=1, stderr="Device does not exist")

    runner = MissingRunner(up=False, bitrate=None)
    with pytest.raises(CanCommandError, match="can0"):
        can_up(_config(), runner)
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    ("sides", "interfaces"),
    [
        (("right",), ["can0"]),
        (("left",), ["can1"]),
        (("right", "left"), ["can0", "can1"]),
    ],
)
def test_can_iteration_uses_every_configured_side_mapping(
    sides: tuple[str, ...], interfaces: list[str]
) -> None:
    runner = FakeRunner(up=True, bitrate=1_000_000)

    assert can_status(_config(*sides), runner) == 0

    assert [call[-1] for call in runner.calls] == interfaces
