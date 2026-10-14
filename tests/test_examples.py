# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest
from examples.ws_client import PiperControlClient, PiperControlError, ProtocolMismatchError
from piper_control.config import ArmSide, ServiceConfig, load_config
from piper_control.drivers.simulated import SimulatedArmDriver
from piper_control.types import ArmFeedback

from test_config import two_arm_config, valid_config, write_yaml


def build_parser():  # type: ignore[no-untyped-def]
    from tools.hardware_smoke import build_parser as implementation

    return implementation()


def run_probe(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tools.hardware_smoke import run_probe as implementation

    return implementation(*args, **kwargs)


def validate_motion_targets(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tools.hardware_smoke import validate_motion_targets as implementation

    return implementation(*args, **kwargs)


@pytest.mark.asyncio
async def test_client_sends_all_commands_and_matches_response_ids(monkeypatch) -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.received: list[dict[str, object]] = []

        async def send(self, raw: str) -> None:
            request = json.loads(raw)
            self.received.append(request)

        async def recv(self, **_kwargs: object) -> str:
            request = self.received[-1]
            data: object = {"ActualQ": [0.0] * 6}
            if request["cmd"] == "hello":
                data = {"identity": "local"}
            elif request["cmd"] == "get_base_info":
                data = {"mode": "single_arm"}
            elif request["cmd"] in {"servoJ", "move_gripper"}:
                data = {"accepted": True}
            return json.dumps(
                {
                    "id": request["id"],
                    "cmd": request["cmd"],
                    "ok": True,
                    "data": data,
                }
            )

        async def close(self) -> None:
            pass

    websocket = FakeWebSocket()

    async def connect(_url: str) -> FakeWebSocket:
        return websocket

    monkeypatch.setattr("examples.ws_client.websockets.connect", connect)
    async with PiperControlClient("ws://test") as client:
        assert len((await client.get_state())["ActualQ"]) == 6
        assert (await client.get_base_info())["mode"] == "single_arm"
        await client.servo_j([0.0] * 6, speed=[10.0] * 6)
        await client.move_gripper(0.05, speed=0.02)

    assert websocket.received == [
        {"id": "1", "cmd": "hello", "identity": "local"},
        {"id": "2", "cmd": "get_state"},
        {"id": "3", "cmd": "get_base_info"},
        {
            "id": "4",
            "cmd": "servoJ",
            "target_joints": [0.0] * 6,
            "speed": [10.0] * 6,
        },
        {
            "id": "5",
            "cmd": "move_gripper",
            "target_position": 0.05,
            "speed": 0.02,
        },
    ]


@pytest.mark.asyncio
async def test_client_raises_typed_server_error(monkeypatch) -> None:
    class FakeWebSocket:
        async def send(self, raw: str) -> None:
            self.request = json.loads(raw)

        async def recv(self, **_kwargs: object) -> str:
            if self.request["cmd"] == "hello":
                return json.dumps(
                    {
                        "id": self.request["id"],
                        "cmd": "hello",
                        "ok": True,
                        "data": {"identity": "local"},
                    }
                )
            return json.dumps(
                {
                    "id": self.request["id"],
                    "cmd": self.request["cmd"],
                    "ok": False,
                    "error": {"code": "STATE_CONFLICT", "message": "not running"},
                }
            )

        async def close(self) -> None:
            pass

    async def connect(_url: str) -> FakeWebSocket:
        return FakeWebSocket()

    monkeypatch.setattr("examples.ws_client.websockets.connect", connect)
    async with PiperControlClient("ws://test") as client:
        with pytest.raises(PiperControlError) as caught:
            await client.servo_j([0.0] * 6)

    assert caught.value.code == "STATE_CONFLICT"
    assert caught.value.command == "servoJ"


@pytest.mark.asyncio
async def test_client_rejects_mismatched_response_id(monkeypatch) -> None:
    class FakeWebSocket:
        async def send(self, raw: str) -> None:
            self.request = json.loads(raw)

        async def recv(self, **_kwargs: object) -> str:
            if self.request["cmd"] == "hello":
                return json.dumps(
                    {
                        "id": self.request["id"],
                        "cmd": "hello",
                        "ok": True,
                        "data": {"identity": "local"},
                    }
                )
            return json.dumps(
                {
                    "id": f"wrong-{self.request['id']}",
                    "cmd": self.request["cmd"],
                    "ok": True,
                    "data": {},
                }
            )

        async def close(self) -> None:
            pass

    async def connect(_url: str) -> FakeWebSocket:
        return FakeWebSocket()

    monkeypatch.setattr("examples.ws_client.websockets.connect", connect)
    async with PiperControlClient("ws://test") as client:
        with pytest.raises(ProtocolMismatchError, match="response id"):
            await client.get_state()


@pytest.mark.asyncio
async def test_client_rejects_mismatched_response_command(monkeypatch) -> None:
    class FakeWebSocket:
        async def send(self, raw: str) -> None:
            self.request = json.loads(raw)

        async def recv(self, **_kwargs: object) -> str:
            if self.request["cmd"] == "hello":
                return json.dumps(
                    {
                        "id": self.request["id"],
                        "cmd": "hello",
                        "ok": True,
                        "data": {"identity": "local"},
                    }
                )
            return json.dumps(
                {
                    "id": self.request["id"],
                    "cmd": "get_base_info",
                    "ok": True,
                    "data": {},
                }
            )

        async def close(self) -> None:
            pass

    async def connect(_url: str) -> FakeWebSocket:
        return FakeWebSocket()

    monkeypatch.setattr("examples.ws_client.websockets.connect", connect)
    async with PiperControlClient("ws://test") as client:
        with pytest.raises(ProtocolMismatchError, match="response command"):
            await client.get_state()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"id": "2", "cmd": "get_state", "ok": True}, "has no data"),
        (
            {"id": "2", "cmd": "get_state", "ok": False, "error": "broken"},
            "no structured error",
        ),
    ],
)
async def test_client_rejects_malformed_response_envelopes(
    monkeypatch, response: dict[str, object], message: str
) -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.request: dict[str, object] = {}

        async def send(self, raw: str) -> None:
            self.request = json.loads(raw)

        async def recv(self, **_kwargs: object) -> str:
            if self.request["cmd"] == "hello":
                return json.dumps(
                    {
                        "id": self.request["id"],
                        "cmd": "hello",
                        "ok": True,
                        "data": {"identity": "local"},
                    }
                )
            return json.dumps(response)

        async def close(self) -> None:
            pass

    async def connect(_url: str) -> FakeWebSocket:
        return FakeWebSocket()

    monkeypatch.setattr("examples.ws_client.websockets.connect", connect)
    async with PiperControlClient("ws://test") as client:
        with pytest.raises(ProtocolMismatchError, match=message):
            await client.get_state()


@pytest.mark.asyncio
async def test_client_context_manager_closes_connection(monkeypatch) -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self.request: dict[str, object] = {}

        async def send(self, raw: str) -> None:
            self.request = json.loads(raw)

        async def recv(self, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "id": self.request["id"],
                    "cmd": "hello",
                    "ok": True,
                    "data": {"identity": "local"},
                }
            )

        async def close(self) -> None:
            self.closed = True

    websocket = FakeWebSocket()

    async def connect(_url: str) -> FakeWebSocket:
        return websocket

    monkeypatch.setattr("examples.ws_client.websockets.connect", connect)
    async with PiperControlClient("ws://test"):
        assert websocket.closed is False

    assert websocket.closed is True


@pytest.mark.asyncio
async def test_client_serializes_concurrent_request_transactions(monkeypatch) -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, object]] = []
            self.receiving = False

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

        async def recv(self, **_kwargs: object) -> str:
            if self.receiving:
                raise RuntimeError("concurrent recv")
            self.receiving = True
            request = self.sent[-1]
            await asyncio.sleep(0)
            self.receiving = False
            return json.dumps(
                {
                    "id": request["id"],
                    "cmd": request["cmd"],
                    "ok": True,
                    "data": (
                        {"identity": "local"}
                        if request["cmd"] == "hello"
                        else {"command": request["cmd"]}
                    ),
                }
            )

        async def close(self) -> None:
            pass

    websocket = FakeWebSocket()

    async def connect(_url: str) -> FakeWebSocket:
        return websocket

    monkeypatch.setattr("examples.ws_client.websockets.connect", connect)
    async with PiperControlClient("ws://test") as client:
        state, info = await asyncio.gather(
            client.get_state(), client.get_base_info()
        )

    assert state == {"command": "get_state"}
    assert info == {"command": "get_base_info"}


def test_smoke_defaults_to_read_only_and_requires_exact_confirmation() -> None:
    parser = build_parser()
    defaults = parser.parse_args(["--config", "config.yaml", "--side", "right"])
    assert defaults.allow_motion is False
    assert defaults.confirm_motion is None

    incomplete = parser.parse_args(
        ["--config", "config.yaml", "--side", "right", "--allow-motion", "--target-joints", *(["0"] * 6)]
    )
    assert incomplete.allow_motion is True
    assert incomplete.confirm_motion is None

    timed = parser.parse_args(
        ["--config", "config.yaml", "--side", "left", "--motion-timeout", "2.5"]
    )
    assert timed.motion_timeout == 2.5

    for invalid in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--config", "config.yaml", "--side", "right", "--motion-timeout", invalid]
            )


def test_smoke_motion_target_reuses_configured_limits(tmp_path: Path) -> None:
    config = load_config(write_yaml(tmp_path, valid_config()))

    validate_motion_targets(config, ArmSide.RIGHT, [0.0] * 6, 0.01)
    with pytest.raises(ValueError, match="joint 1"):
        validate_motion_targets(config, ArmSide.RIGHT, [151.0] + [0.0] * 5, 0.01)


def test_smoke_source_never_calls_power_off_or_reset() -> None:
    source = (Path(__file__).parents[1] / "tools" / "hardware_smoke.py").read_text(
        encoding="utf-8"
    )
    forbidden = "Disable" + "Arm", "Reset" + "Piper"
    assert all(name not in source for name in forbidden)


def _probe_config(*, dual: bool = False) -> ServiceConfig:
    data = two_arm_config() if dual else valid_config()
    if not dual:
        del data["arms"]["left"]
    for arm in data["arms"].values():
        arm["control_hz"] = 10
        arm["default_joints_speed_deg_s"] = [1.0] * 6
        arm["gripper"]["default_speed_m_s"] = 0.02
    return ServiceConfig.model_validate(data)


class ProbeDriver(SimulatedArmDriver):
    def __init__(self, *, fail_command: bool = False) -> None:
        super().__init__(ArmFeedback(received_at_s=1.0))
        self.fail_command = fail_command

    def command_joints(self, joints_deg) -> None:  # type: ignore[no-untyped-def]
        if self.fail_command:
            raise RuntimeError("command failed")
        super().command_joints(joints_deg)


def test_probe_read_only_never_enables_moves_or_stops() -> None:
    config = _probe_config()
    driver = ProbeDriver()

    run_probe(
        config,
        side=ArmSide.RIGHT,
        allow_motion=False,
        target_joints=None,
        target_gripper=None,
        settle_seconds=0.0,
        drivers={"right": driver},
        output=lambda _text: None,
    )

    assert driver.connect_calls == 1
    assert driver.enable_calls == 0
    assert driver.emergency_stop_calls == 0
    assert driver.hold_calls == 0
    assert driver.joint_commands == []
    assert driver.gripper_commands == []


def test_probe_interpolates_at_default_speed_and_holds_on_success() -> None:
    config = _probe_config()
    driver = ProbeDriver()

    run_probe(
        config,
        side=ArmSide.RIGHT,
        allow_motion=True,
        target_joints=[0.25, 0, 0, 0, 0, 0],
        target_gripper=0.005,
        settle_seconds=0.0,
        drivers={"right": driver},
        wait=lambda _seconds: None,
        output=lambda _text: None,
    )

    joint_positions = [command[0] for command in driver.joint_commands[:-1]]
    assert joint_positions[-1] == pytest.approx(0.25)
    assert all(
        later - earlier <= 0.1 + 1e-12
        for earlier, later in zip([0.0] + joint_positions, joint_positions)
    )
    assert driver.gripper_commands[-1] == pytest.approx(0.005)
    assert all(
        later - earlier <= 0.002 + 1e-12
        for earlier, later in zip(
            [0.0] + driver.gripper_commands, driver.gripper_commands
        )
    )
    assert driver.hold_calls == 1
    assert driver.emergency_stop_calls == 0


def test_probe_constructs_and_cleans_up_only_selected_side() -> None:
    config = _probe_config(dual=True)
    right = ProbeDriver()
    left = ProbeDriver(fail_command=True)

    with pytest.raises(RuntimeError, match="command failed"):
        run_probe(
            config,
            side=ArmSide.LEFT,
            allow_motion=True,
            target_joints=[0.2, 0, 0, 0, 0, 0],
            target_gripper=None,
            settle_seconds=0.0,
            drivers={"right": right, "left": left},
            wait=lambda _seconds: None,
            output=lambda _text: None,
        )

    assert left.emergency_stop_calls == left.hold_calls == 1
    assert right.connect_calls == right.emergency_stop_calls == right.hold_calls == 0


def test_probe_keyboard_interrupt_emergency_stops_then_holds() -> None:
    config = _probe_config()
    driver = ProbeDriver()

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_probe(
            config,
            side=ArmSide.RIGHT,
            allow_motion=True,
            target_joints=[0.2, 0, 0, 0, 0, 0],
            target_gripper=None,
            settle_seconds=0.0,
            drivers={"right": driver},
            wait=interrupt,
            output=lambda _text: None,
        )

    assert driver.emergency_stop_calls == 1
    assert driver.hold_calls == 1


def test_probe_accepts_three_stable_noisy_arrival_samples() -> None:
    class NoisyDriver(ProbeDriver):
        def command_joints(self, joints_deg) -> None:  # type: ignore[no-untyped-def]
            super().command_joints(joints_deg)
            sample = super().read_feedback()
            self.set_feedback(
                replace(
                    sample,
                    joints_deg=(sample.joints_deg[0] + 0.005,) + sample.joints_deg[1:],
                )
            )

    driver = NoisyDriver()
    run_probe(
        _probe_config(),
        side=ArmSide.RIGHT,
        allow_motion=True,
        target_joints=[0.25, 0, 0, 0, 0, 0],
        target_gripper=None,
        settle_seconds=0.0,
        motion_timeout_s=1.0,
        drivers={"right": driver},
        wait=lambda _seconds: None,
        output=lambda _text: None,
    )

    assert driver.hold_calls == 1
    assert driver.emergency_stop_calls == 0
    assert driver.read_feedback().joints_deg[0] == pytest.approx(0.255)


def test_probe_no_progress_times_out_then_emergency_stops_and_holds() -> None:
    class StuckDriver(ProbeDriver):
        def command_joints(self, joints_deg) -> None:  # type: ignore[no-untyped-def]
            self.joint_commands.append(tuple(joints_deg))

    driver = StuckDriver()
    with pytest.raises(TimeoutError, match="motion timeout"):
        run_probe(
            _probe_config(),
            side=ArmSide.RIGHT,
            allow_motion=True,
            target_joints=[0.25, 0, 0, 0, 0, 0],
            target_gripper=None,
            settle_seconds=0.0,
            motion_timeout_s=0.25,
            drivers={"right": driver},
            wait=lambda _seconds: None,
            output=lambda _text: None,
        )

    assert driver.emergency_stop_calls == 1
    assert driver.hold_calls == 1


@pytest.mark.parametrize(
    "unhealthy",
    [
        {"fault_code": 7},
        {"stale": True},
        {"connected": False},
        {"enabled": False},
        {"joints_deg": (float("nan"), 0, 0, 0, 0, 0)},
        {"gripper_position": float("nan")},
        {"tcp_pose": (float("nan"), 0, 0, 0, 0, 0)},
        {"received_at_s": float("inf")},
    ],
)
def test_probe_rejects_unhealthy_motion_feedback_and_cleans_up(unhealthy) -> None:
    class UnhealthyAfterEnableDriver(ProbeDriver):
        def read_feedback(self):  # type: ignore[no-untyped-def]
            sample = super().read_feedback()
            return replace(sample, **unhealthy) if self.enable_calls else sample

    driver = UnhealthyAfterEnableDriver()
    with pytest.raises(RuntimeError, match="unhealthy feedback"):
        run_probe(
            _probe_config(),
            side=ArmSide.RIGHT,
            allow_motion=True,
            target_joints=[0.1, 0, 0, 0, 0, 0],
            target_gripper=None,
            settle_seconds=0.0,
            motion_timeout_s=1.0,
            drivers={"right": driver},
            wait=lambda _seconds: None,
            output=lambda _text: None,
        )

    assert driver.emergency_stop_calls == 1
    assert driver.hold_calls == 1


def test_probe_final_hold_failure_never_reports_motion_success() -> None:
    class HoldFailureDriver(ProbeDriver):
        def hold(self, feedback) -> None:  # type: ignore[no-untyped-def]
            self.hold_calls += 1
            raise RuntimeError("hold failed")

    driver = HoldFailureDriver()
    outputs: list[str] = []
    with pytest.raises(RuntimeError, match="final HOLD failed"):
        run_probe(
            _probe_config(),
            side=ArmSide.RIGHT,
            allow_motion=True,
            target_joints=[0.1, 0, 0, 0, 0, 0],
            target_gripper=None,
            settle_seconds=0.0,
            motion_timeout_s=1.0,
            drivers={"right": driver},
            wait=lambda _seconds: None,
            output=outputs.append,
        )

    assert driver.emergency_stop_calls == 1
    assert driver.hold_calls == 2
    assert not any("motion_confirmed" in output for output in outputs)
