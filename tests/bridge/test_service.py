# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from piper_control.bridge.config import BridgeConfig
from piper_control.bridge.protocol import parse_bridge_request
from piper_control.bridge.service import BridgeService
from piper_control.bridge.upstream import UpstreamError
from piper_control.config import ArmSide
from piper_control.protocol import ProtocolError


def bridge_config(**overrides) -> BridgeConfig:
    values = {
        "listen": {"host": "127.0.0.1", "port": 8766},
        "upstream_url": "ws://127.0.0.1:8765/ws/control",
        "control_hz": 200.0,
    }
    values.update(overrides)
    return BridgeConfig.model_validate(
        values
    )


class FakeUpstream:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.fail = False
        self.q = [0.0] * 6
        self.gripper = 0.025
        self.calls: list[tuple[str, object]] = []
        self.active_sends = 0
        self.maximum_active_sends = 0

    async def connect(self) -> FakeUpstream:
        self.calls.append(("connect", None))
        self.connected = True
        return self

    async def close(self) -> None:
        self.closed = True
        self.connected = False

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("upstream failed")

    async def get_state(self) -> dict:
        self._check()
        self.calls.append(("get_state", None))
        return {
            "side": "left",
            "control_mode": "remote",
            "state": "RUNNING",
            "emergency_stopped": False,
            "control_ready": True,
            "ActualQ": list(self.q),
            "ActualGripper": self.gripper,
        }

    async def get_base_info(self) -> dict:
        self._check()
        self.calls.append(("get_base_info", None))
        return {
            "protocol_version": "3.0",
            "side": "left",
            "control_mode": "remote",
            "units": {
                "gripper_position": "meter",
                "gripper_speed": "meter/second",
            },
            "arm": {
                "joint_speed_deg_s": {
                    "default": [10.0] * 6,
                    "max": [30.0] * 6,
                },
                "gripper": {
                    "backend": "piper",
                    "speed_m_s": {"default": 0.02, "max": 0.05},
                },
            },
        }

    async def servo_j(self, target_joints, speed) -> dict:
        self._check()
        self.active_sends += 1
        self.maximum_active_sends = max(self.maximum_active_sends, self.active_sends)
        await asyncio.sleep(0)
        self.q = list(target_joints)
        self.calls.append(("servoJ", (tuple(target_joints), tuple(speed))))
        self.active_sends -= 1
        return {"accepted": True}

    async def move_gripper(self, target_position, speed) -> dict:
        self._check()
        self.active_sends += 1
        self.maximum_active_sends = max(self.maximum_active_sends, self.active_sends)
        await asyncio.sleep(0)
        self.gripper = target_position
        self.calls.append(("move_gripper", (target_position, speed)))
        self.active_sends -= 1
        return {"accepted": True}


async def eventually(predicate, *, timeout: float = 0.5) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.001)


def servo(marker: float, priority: int = 0):
    return parse_bridge_request(
        {
            "id": f"servo-{marker}",
            "cmd": "servoJ",
            "target_joints": [marker] * 6,
            "speed": [10.0] * 6,
            "priority": priority,
        }
    )


def gripper(marker: float, priority: int = 0):
    return parse_bridge_request(
        {
            "id": f"gripper-{marker}",
            "cmd": "move_gripper",
            "target_position": marker,
            "speed": 0.05,
            "priority": priority,
        }
    )


@pytest.mark.asyncio
async def test_piper_internal_hold_sends_both_axes() -> None:
    upstream = FakeUpstream()
    service = BridgeService(bridge_config(), side=ArmSide.LEFT, upstream_factory=lambda _url: upstream)

    await service.start()
    try:
        await eventually(
            lambda: {name for name, _value in upstream.calls}
            >= {"connect", "get_base_info", "get_state", "servoJ", "move_gripper"}
        )
        first_joint = next(value for name, value in upstream.calls if name == "servoJ")
        first_gripper = next(value for name, value in upstream.calls if name == "move_gripper")
        assert first_joint == ((0.0,) * 6, (10.0,) * 6)
        assert first_gripper == (0.025, 0.02)
        assert upstream.maximum_active_sends == 1
        assert set((await service.base_info())["supported_commands"]) == {
            "servoJ",
            "move_gripper",
            "release",
            "get_state",
            "get_base_info",
        }
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_unsupported_gripper_backend_fails_closed_without_motion() -> None:
    class UnsupportedGripperUpstream(FakeUpstream):
        async def get_base_info(self) -> dict:
            payload = await super().get_base_info()
            payload["units"] = {
                "gripper_position": "radian", "gripper_speed": "radian/second"
            }
            payload["arm"]["gripper"] = {
                "backend": "unsupported_backend",
                "speed_rad_s": {"default": 1.0, "max": 2.0},
            }
            return payload

    upstream = UnsupportedGripperUpstream()
    service = BridgeService(
        bridge_config(reconnect_initial_s=0.01, reconnect_max_s=0.01),
        side=ArmSide.LEFT, upstream_factory=lambda _url: upstream
    )

    await service.start()
    try:
        await eventually(lambda: any(name == "get_state" for name, _ in upstream.calls))
        assert (await service.state())["bridge"]["upstream_available"] is False
        assert not any(name in {"servoJ", "move_gripper"} for name, _ in upstream.calls)
        with pytest.raises(ProtocolError, match="unavailable"):
            await service.submit("client", gripper(0.05))
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_higher_priority_gripper_preempts_joint_interpolation_globally() -> None:
    upstream = FakeUpstream()
    service = BridgeService(bridge_config(), side=ArmSide.LEFT, upstream_factory=lambda _url: upstream)
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in upstream.calls))
        upstream.calls.clear()
        assert (await service.submit("low", servo(100.0, priority=1))).accepted
        await eventually(
            lambda: sum(name == "servoJ" for name, _ in upstream.calls) >= 2
        )

        result = await service.submit("high", gripper(0.05, priority=100))
        assert result.preempted is not None
        await eventually(
            lambda: any(
                name == "move_gripper" and value[0] > 0.025
                for name, value in upstream.calls
            )
        )
        first_high = next(
            index
            for index, (name, value) in enumerate(upstream.calls)
            if name == "move_gripper" and value[0] > 0.025
        )
        await asyncio.sleep(0.03)
        q_after_preemption = [
            value[0][0]
            for name, value in upstream.calls[first_high:]
            if name == "servoJ"
        ]
        assert not q_after_preemption or max(q_after_preemption) == min(q_after_preemption)
        assert upstream.maximum_active_sends == 1
        assert (await service.state())["bridge"]["owner"] == {
            "connection_id": "high",
            "priority": 100,
        }
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_owner_lease_blocks_lower_priority_until_release_then_hands_off() -> None:
    upstream = FakeUpstream()
    service = BridgeService(bridge_config(), side=ArmSide.LEFT, upstream_factory=lambda _url: upstream)
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in upstream.calls))
        await service.submit("high", servo(0.2, priority=10))
        await eventually(lambda: upstream.q[0] == pytest.approx(0.2))
        await service.submit("low", gripper(0.06, priority=0))
        await asyncio.sleep(0.03)
        assert upstream.gripper == pytest.approx(0.025)

        await service.release("high", 10)
        await eventually(lambda: upstream.gripper > 0.025)
        assert (await service.state())["bridge"]["owner"] == {
            "connection_id": "low",
            "priority": 0,
        }
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_disconnect_cancels_owner_and_immediately_hands_off() -> None:
    upstream = FakeUpstream()
    service = BridgeService(bridge_config(), side=ArmSide.LEFT, upstream_factory=lambda _url: upstream)
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in upstream.calls))
        await service.submit("high", servo(20.0, priority=50))
        await eventually(lambda: upstream.q[0] > 0.0)
        await service.submit("waiting", gripper(0.06, priority=5))

        await service.disconnect("high")
        await eventually(lambda: upstream.gripper > 0.025)
        assert (await service.state())["bridge"]["owner"] == {
            "connection_id": "waiting",
            "priority": 5,
        }
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("unrelated_action", ["disconnect", "release"])
async def test_unrelated_client_cleanup_does_not_cancel_owner_motion(
    unrelated_action: str,
) -> None:
    upstream = FakeUpstream()
    service = BridgeService(
        bridge_config(), side=ArmSide.LEFT, upstream_factory=lambda _url: upstream
    )
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in upstream.calls))
        await service.submit("owner", servo(100.0, priority=10))
        await eventually(lambda: upstream.q[0] > 0.1)
        before = upstream.q[0]

        if unrelated_action == "disconnect":
            await service.disconnect("observer")
        else:
            await service.release("observer", 999)

        await eventually(lambda: upstream.q[0] > before + 0.1)
        assert (await service.state())["bridge"]["owner"] == {
            "connection_id": "owner",
            "priority": 10,
        }
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_expired_waiter_is_dropped_and_large_motion_stops_at_deadline() -> None:
    upstream = FakeUpstream()
    service = BridgeService(
        bridge_config(command_ttl_s=0.05, owner_timeout_s=0.15),
        side=ArmSide.LEFT,
        upstream_factory=lambda _url: upstream,
    )
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in upstream.calls))
        await service.submit("owner", gripper(0.03, priority=10))
        await service.submit("expired", servo(50.0, priority=0))
        await asyncio.sleep(0.07)
        await service.release("owner", 10)
        await asyncio.sleep(0.03)
        assert upstream.q == pytest.approx([0.0] * 6)

        await service.submit("bounded", servo(50.0, priority=20))
        await asyncio.sleep(0.08)
        assert 0.0 < upstream.q[0] <= 0.55
        stopped = upstream.q[0]
        await asyncio.sleep(0.03)
        assert upstream.q[0] == pytest.approx(stopped)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joint_interpolation_advances_setpoint_when_feedback_lags() -> None:
    class LaggingFeedbackUpstream(FakeUpstream):
        async def servo_j(self, target_joints, speed) -> dict:
            self.calls.append(("servoJ", (tuple(target_joints), tuple(speed))))
            return {"accepted": True}

    upstream = LaggingFeedbackUpstream()
    service = BridgeService(
        bridge_config(command_ttl_s=0.05, owner_timeout_s=0.15),
        side=ArmSide.LEFT,
        upstream_factory=lambda _url: upstream,
    )
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in upstream.calls))
        upstream.calls.clear()
        await service.submit("operator", servo(10.0, priority=10))
        await eventually(
            lambda: sum(name == "servoJ" for name, _ in upstream.calls) >= 3
        )
        targets = [
            value[0][0] for name, value in upstream.calls if name == "servoJ"
        ]
        assert targets[1] > targets[0]
        assert targets[2] > targets[1]
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joint_plan_holds_final_setpoint_until_deadline_when_feedback_lags() -> None:
    class LaggingFeedbackUpstream(FakeUpstream):
        async def servo_j(self, target_joints, speed) -> dict:
            self.calls.append(("servoJ", (tuple(target_joints), tuple(speed))))
            return {"accepted": True}

    upstream = LaggingFeedbackUpstream()
    service = BridgeService(
        bridge_config(command_ttl_s=0.05, owner_timeout_s=0.15),
        side=ArmSide.LEFT,
        upstream_factory=lambda _url: upstream,
    )
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in upstream.calls))
        upstream.calls.clear()
        await service.submit("operator", servo(0.1, priority=10))
        await eventually(
            lambda: sum(name == "servoJ" for name, _ in upstream.calls) >= 3
        )
        targets = [
            value[0][0] for name, value in upstream.calls if name == "servoJ"
        ]
        assert targets[:3] == pytest.approx([0.05, 0.1, 0.1])
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_reached_joint_plan_finishes_before_next_owner_command_expires() -> None:
    upstream = FakeUpstream()
    service = BridgeService(
        bridge_config(command_ttl_s=0.05, owner_timeout_s=0.15),
        side=ArmSide.LEFT,
        upstream_factory=lambda _url: upstream,
    )
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in upstream.calls))
        await service.submit("operator", servo(0.1, priority=10))
        await service.submit("operator", servo(0.2, priority=10))
        await eventually(lambda: upstream.q[0] == pytest.approx(0.2))
    finally:
        await service.stop()


class GatedUpstream(FakeUpstream):
    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__()
        self.gate = gate
        self.q = [7.0] * 6
        self.gripper = 0.06

    async def connect(self) -> FakeUpstream:
        self.calls.append(("connect", None))
        await self.gate.wait()
        self.connected = True
        return self


@pytest.mark.asyncio
async def test_reconnect_uses_independent_bounded_exponential_backoff() -> None:
    class UnavailableUpstream:
        async def connect(self):
            raise ConnectionError("local mode")

        async def close(self) -> None:
            pass

    delays: list[float] = []
    service: BridgeService

    async def controlled_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 4:
            service._stopping = True
        await asyncio.sleep(0)

    service = BridgeService(
        bridge_config(reconnect_initial_s=0.2, reconnect_max_s=0.5),
        side=ArmSide.LEFT,
        upstream_factory=lambda _url: UnavailableUpstream(),
        sleep=controlled_sleep,
    )
    await service.start()
    await eventually(lambda: service._task is not None and service._task.done())
    await service.stop()

    assert delays == pytest.approx([0.2, 0.4, 0.5, 0.5])


@pytest.mark.asyncio
async def test_local_mode_uses_read_only_probe_until_remote_before_reconnecting() -> None:
    class LocalModeUpstream:
        async def connect(self):
            raise UpstreamError("IDENTITY_MISMATCH", "local identity is required")

        async def close(self) -> None:
            pass

    remote = asyncio.Event()
    probe_calls = 0
    clients: list[object] = []
    connected = FakeUpstream()

    def upstream_factory(_url: str):
        client = LocalModeUpstream() if not clients else connected
        clients.append(client)
        return client

    async def mode_probe(_url: str, _timeout_s: float) -> str:
        nonlocal probe_calls
        probe_calls += 1
        return "remote" if remote.is_set() else "local"

    service = BridgeService(
        bridge_config(reconnect_initial_s=0.01, reconnect_max_s=0.02),
        side=ArmSide.LEFT,
        upstream_factory=upstream_factory,
        mode_probe=mode_probe,
        mode_probe_interval_s=0.01,
    )
    await service.start()
    try:
        await eventually(lambda: probe_calls >= 2)
        assert len(clients) == 1, "Bridge kept opening WebSockets while local mode was active"

        remote.set()
        await eventually(lambda: any(name == "connect" for name, _ in connected.calls))
        assert len(clients) == 2
        assert (await service.state())["bridge"]["upstream_available"] is True
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_internal_hold_does_not_send_gripper_after_external_preemption() -> None:
    class GatedSecondHoldRefresh(FakeUpstream):
        def __init__(self) -> None:
            super().__init__()
            self.state_reads = 0
            self.refresh_blocked = asyncio.Event()
            self.allow_refresh = asyncio.Event()

        async def get_state(self) -> dict:
            self.state_reads += 1
            if self.state_reads == 3:
                self.refresh_blocked.set()
                await self.allow_refresh.wait()
            return await super().get_state()

    upstream = GatedSecondHoldRefresh()
    service = BridgeService(
        bridge_config(), side=ArmSide.LEFT, upstream_factory=lambda _url: upstream
    )
    await service.start()
    try:
        await asyncio.wait_for(upstream.refresh_blocked.wait(), 0.5)
        await service.submit("external", gripper(0.05, priority=0))
        upstream.allow_refresh.set()
        await eventually(lambda: any(name == "move_gripper" for name, _ in upstream.calls))
        first_gripper = next(
            value for name, value in upstream.calls if name == "move_gripper"
        )
        assert first_gripper[0] > 0.025
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_failure_clears_everything_and_reconnect_never_replays_old_target() -> None:
    first = FakeUpstream()
    gate = asyncio.Event()
    recovered = GatedUpstream(gate)
    clients = iter((first, recovered))
    service = BridgeService(bridge_config(), side=ArmSide.LEFT, upstream_factory=lambda _url: next(clients))
    await service.start()
    try:
        await eventually(lambda: any(name == "servoJ" for name, _ in first.calls))
        await service.submit("old", servo(100.0, priority=50))
        await eventually(lambda: first.q[0] > 0.0)
        first.fail = True
        await eventually(lambda: any(name == "connect" for name, _ in recovered.calls))

        with pytest.raises(Exception) as error:
            await service.submit("during-outage", servo(1.0))
        assert getattr(error.value, "code", None) == "UPSTREAM_UNAVAILABLE"
        assert (await service.state())["bridge"]["owner"] is None

        gate.set()
        await eventually(
            lambda: any(name == "move_gripper" for name, _ in recovered.calls)
        )
        motion_targets = [
            value[0]
            for name, value in recovered.calls
            if name == "servoJ"
        ]
        assert motion_targets and all(target == (7.0,) * 6 for target in motion_targets)
        gripper_targets = [value for name, value in recovered.calls if name == "move_gripper"]
        assert gripper_targets and all(target == (0.06, 0.02) for target in gripper_targets)
        assert recovered.gripper == pytest.approx(0.06)
        assert (await service.state())["bridge"]["upstream_available"] is True
        assert (await service.state())["bridge"]["owner"] is None
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_wrong_side_upstream_fails_closed_without_motion() -> None:
    upstream = FakeUpstream()
    service = BridgeService(
        bridge_config(reconnect_initial_s=0.05, reconnect_max_s=0.05),
        side=ArmSide.RIGHT,
        upstream_factory=lambda _url: upstream,
    )
    await service.start()
    try:
        await eventually(lambda: sum(name == "connect" for name, _ in upstream.calls) >= 2)
        assert (await service.state())["bridge"]["upstream_available"] is False
        assert not any(
            name in {"servoJ", "move_gripper"} for name, _ in upstream.calls
        )
        with pytest.raises(ProtocolError, match="unavailable"):
            await service.submit("client", servo(1.0))
    finally:
        await service.stop()
