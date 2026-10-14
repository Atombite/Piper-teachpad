# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from threading import Event, Thread
from typing import get_type_hints

import pytest

from piper_control.config import ArmConfig, ArmSide
from piper_control.controller import ArmController, StateConflict
from piper_control.drivers.simulated import SimulatedArmDriver
from piper_control.types import (
    ArmFeedback,
    ArmStateSnapshot,
    ControlMode,
    ServiceState,
)
from piper_control.vectors import GripperCommand, ServoCommand


def test_snapshot_has_only_one_arm_shape() -> None:
    snapshot = ArmStateSnapshot(side=ArmSide.LEFT)

    payload = snapshot.to_json()

    assert payload["side"] == "left"
    assert len(payload["ActualQ"]) == 6
    assert len(payload["ActualTCPPose"]) == 6
    assert payload["ActualGripper"] == 0.0
    assert payload["robot_timestamp"] == 0.0
    assert "arms" not in payload


def test_snapshot_runtime_type_hints_resolve_arm_side() -> None:
    assert get_type_hints(ArmStateSnapshot)["side"] is ArmSide


def test_replacing_one_arm_snapshot_does_not_change_another() -> None:
    left = ArmStateSnapshot(side=ArmSide.LEFT)
    right = ArmStateSnapshot(side=ArmSide.RIGHT)

    changed_left = replace(left, actual_q=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    assert changed_left.actual_q[0] == 1.0
    assert right.actual_q == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _arm(*, auto_initialize: bool, side: ArmSide = ArmSide.RIGHT) -> dict[str, object]:
    return {
        "server": {"host": "127.0.0.1", "port": 8765 if side is ArmSide.RIGHT else 9876},
        "control_hz": 10,
        "feedback_timeout_s": 0.5,
        "shutdown_home_timeout_s": 30,
        "name": f"{side.value}_arm",
        "can_port": "can0" if side is ArmSide.RIGHT else "can1",
        "auto_initialize": auto_initialize,
        "receive_latency_s": 0.1,
        "calibration_joints_pos_deg": [0, 0, 0, 0, 0, 0],
        "reset_joints_pos_deg": [0, 0, 0, 0, 0, 0],
        "min_joints_deg": [-150, 0, -170, -100, -70, -120],
        "max_joints_deg": [150, 180, 0, 100, 70, 120],
        "max_joints_speed_deg_s": [30, 30, 30, 30, 30, 30],
        "default_joints_speed_deg_s": [10, 10, 10, 10, 10, 10],
        "gripper": {
            "enabled": True,
            "min_position_m": 0.0,
            "max_position_m": 0.07,
            "max_speed_m_s": 0.05,
            "default_speed_m_s": 0.02,
            "effort_nm": 1.0,
        },
    }


def _config(*, auto_initialize: bool = False, side: ArmSide = ArmSide.RIGHT) -> ArmConfig:
    return ArmConfig.model_validate(_arm(auto_initialize=auto_initialize, side=side))


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _controller(
    *, auto_initialize: bool = False, clock=None
) -> tuple[ArmController, SimulatedArmDriver]:
    config = _config(auto_initialize=auto_initialize)
    driver = SimulatedArmDriver(ArmFeedback(received_at_s=100.0))
    kwargs = {"clock": clock} if clock is not None else {}
    return ArmController(ArmSide.RIGHT, config, driver, **kwargs), driver


def test_mujoco_robot_timestamp_overrides_hardware_latency_derivation() -> None:
    config = _config()
    driver = SimulatedArmDriver(
        ArmFeedback(received_at_s=500.0, robot_timestamp_s=1.25)
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)

    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.robot_receive_timestamp == 500.0
    assert snapshot.robot_timestamp == 1.25
    assert snapshot.arm.receive_timestamp == 500.0
    assert snapshot.arm.robot_timestamp == 1.25


def test_hardware_robot_timestamp_still_derives_from_receive_latency() -> None:
    config = _config()
    driver = SimulatedArmDriver(ArmFeedback(received_at_s=500.0))
    controller = ArmController(ArmSide.RIGHT, config, driver)

    controller.tick(dt=0.1)

    assert controller.snapshot().robot_timestamp == pytest.approx(499.9)


def test_service_state_exposes_only_active_lifecycle_states() -> None:
    assert {state.value for state in ServiceState} == {
        "STARTING",
        "HOMING",
        "RUNNING",
        "HOLD",
        "EMERGENCY_STOP",
        "SHUTTING_DOWN",
    }


def test_snapshot_defaults_to_local_control_mode() -> None:
    snapshot = ArmStateSnapshot(side=ArmSide.RIGHT)

    assert snapshot.control_mode is ControlMode.LOCAL
    assert snapshot.to_json()["control_mode"] == "local"


def test_request_hold_discards_pending_motion_and_holds_current_feedback() -> None:
    feedback = ArmFeedback(
        joints_deg=(5.0, 10.0, -20.0, 1.0, -2.0, 3.0),
        gripper_position=0.04,
        received_at_s=100.0,
    )
    config = _config()
    driver = SimulatedArmDriver(feedback)
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.submit_servo(
        ServoCommand((10.0, 10.0, -20.0, 1.0, -2.0, 3.0), (1.0,) * 6)
    )
    controller.submit_gripper(GripperCommand(0.01, 0.01))

    controller.request_hold()
    held = controller.snapshot()
    controller.tick(dt=0.1)

    assert held.state is ServiceState.HOLD
    assert held.hold is True
    assert held.target_q == feedback.joints_deg
    assert held.target_qd == (0.0,) * 6
    assert held.target_gripper == pytest.approx(0.04)
    assert held.target_gripper_speed == 0.0
    assert driver.hold_calls >= 2
    assert controller.snapshot().state is ServiceState.HOLD
    assert controller.snapshot().target_q == feedback.joints_deg
    assert controller.snapshot().target_gripper == pytest.approx(0.04)


def test_hold_rejects_motion_homing_and_recovery_requests() -> None:
    controller, _ = _controller()
    controller.tick(dt=0.1)
    controller.request_hold()

    with pytest.raises(StateConflict, match="RUNNING"):
        controller.submit_servo(ServoCommand((0.0,) * 6, (1.0,) * 6))
    with pytest.raises(StateConflict, match="RUNNING"):
        controller.submit_gripper(GripperCommand(0.01, 0.01))
    with pytest.raises(StateConflict, match="RUNNING"):
        controller.request_homing()
    with pytest.raises(StateConflict, match="EMERGENCY_STOP"):
        controller.request_recovery()


def test_change_control_mode_requires_hold_and_returns_to_running() -> None:
    controller, _ = _controller()
    controller.tick(dt=0.1)

    with pytest.raises(StateConflict, match="HOLD"):
        controller.change_control_mode(ControlMode.REMOTE)

    controller.request_hold()
    controller.change_control_mode(ControlMode.REMOTE)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.control_mode is ControlMode.REMOTE
    assert snapshot.hold is True


def test_repeated_hold_refreshes_driver_hold_without_leaving_hold() -> None:
    controller, driver = _controller()
    controller.tick(dt=0.1)
    controller.request_hold()
    first_calls = driver.hold_calls

    controller.request_hold()

    assert controller.snapshot().state is ServiceState.HOLD
    assert driver.hold_calls == first_calls + 1


def test_switching_to_current_mode_completes_hold_transaction() -> None:
    controller, _ = _controller()
    controller.tick(dt=0.1)
    controller.request_hold()

    controller.change_control_mode(ControlMode.LOCAL)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert controller.snapshot().control_mode is ControlMode.LOCAL


def test_emergency_latch_prevents_mode_change_from_hold() -> None:
    controller, _ = _controller()
    controller.tick(dt=0.1)
    controller.request_hold()
    controller.emergency_stop("operator stop")

    with pytest.raises(StateConflict, match="emergency stop"):
        controller.change_control_mode(ControlMode.REMOTE)

    assert controller.snapshot().control_mode is ControlMode.LOCAL


def test_hold_driver_failure_enters_emergency_stop_without_changing_mode() -> None:
    class FailOnDemandDriver(SimulatedArmDriver):
        fail_hold = False

        def hold(self, feedback: ArmFeedback) -> None:
            if self.fail_hold:
                raise RuntimeError("hold bus failure")
            super().hold(feedback)

    driver = FailOnDemandDriver(ArmFeedback(received_at_s=100.0))
    controller = ArmController(ArmSide.RIGHT, _config(), driver)
    controller.tick(dt=0.1)
    driver.fail_hold = True

    with pytest.raises(StateConflict, match="mode-switch HOLD failed"):
        controller.request_hold()

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert snapshot.control_mode is ControlMode.LOCAL
    assert snapshot.emergency_stopped is True


def test_start_without_auto_initialize_holds_current_pose_and_runs() -> None:
    controller, driver = _controller()
    assert controller.snapshot().state is ServiceState.STARTING

    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.target_q == snapshot.actual_q
    assert driver.connect_calls == 1
    assert driver.enable_calls == 1
    command = ServoCommand(target=(0.0,) * 6, speed=(10.0,) * 6)
    controller.submit_servo(command)


def test_published_snapshot_expires_without_a_new_control_tick() -> None:
    clock = ManualClock()
    controller, _ = _controller(clock=clock)
    controller.tick(dt=0.1)

    assert controller.snapshot().control_ready is True

    clock.advance(0.5)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.arm.feedback_fresh is False
    assert snapshot.control_ready is False
    assert snapshot.control_block_reason == "Robot feedback is stale."


def test_expired_snapshot_rejects_arm_and_gripper_commands() -> None:
    clock = ManualClock()
    controller, _ = _controller(clock=clock)
    controller.tick(dt=0.1)
    clock.advance(0.5)

    with pytest.raises(StateConflict, match="fresh feedback"):
        controller.submit_servo(ServoCommand((0.0,) * 6, (1.0,) * 6))
    with pytest.raises(StateConflict, match="fresh feedback"):
        controller.submit_gripper(GripperCommand(0.01, 0.01))


def test_start_waits_for_first_fresh_feedback_before_health_check() -> None:
    clock = ManualClock()

    class DelayedFeedbackDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(
                ArmFeedback(stale=True, received_at_s=0.0),
            )
            self._reads = 0
            self.read_at: list[float] = []

        def read_feedback(self) -> ArmFeedback:
            self._reads += 1
            self.read_at.append(clock())
            if self._reads == 2:
                self.set_feedback(
                    ArmFeedback(
                        connected=True,
                        enabled=True,
                        stale=False,
                        received_at_s=101.0,
                    )
                )
            return super().read_feedback()

    driver = DelayedFeedbackDriver()
    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        driver,
        clock=clock,
        sleep=clock.advance,
    )

    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.emergency_stopped is False
    assert snapshot.arm.feedback_fresh is True
    assert snapshot.robot_receive_timestamp == 101.0
    assert driver.read_at == pytest.approx([0.0, 0.1])


def test_startup_stale_feedback_faults_after_feedback_timeout() -> None:
    clock = ManualClock()
    driver = SimulatedArmDriver(
        ArmFeedback(stale=True, received_at_s=0.0),
    )
    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        driver,
        clock=clock,
        sleep=clock.advance,
    )

    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert clock.now == pytest.approx(0.5)
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert snapshot.fault_reason == "right arm feedback stale"


@pytest.mark.parametrize("delayed_field", ["connected", "enabled"])
def test_startup_waits_for_stale_connection_state_to_synchronize(
    delayed_field: str,
) -> None:
    clock = ManualClock()

    class SynchronizingStartupDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(ArmFeedback(stale=True, received_at_s=1.0))
            self._reads = 0

        def read_feedback(self) -> ArmFeedback:
            self._reads += 1
            feedback = super().read_feedback()
            if self._reads == 1:
                return replace(feedback, stale=True, **{delayed_field: False})
            return replace(
                feedback,
                connected=True,
                enabled=True,
                stale=False,
                received_at_s=2.0,
            )

    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        SynchronizingStartupDriver(),
        clock=clock,
        sleep=clock.advance,
    )

    controller.tick(dt=0.1)

    assert clock.now == pytest.approx(0.1)
    assert controller.snapshot().state is ServiceState.RUNNING
    assert controller.snapshot().emergency_stopped is False


@pytest.mark.parametrize(
    "stale_update",
    [
        {"fault_code": 42},
        {"joints_deg": (5.0, 0.0, 1.424, 0.0, 0.0, 0.0)},
    ],
)
def test_startup_waits_for_stale_fault_state_to_synchronize(
    stale_update: dict[str, object],
) -> None:
    clock = ManualClock()

    class SynchronizingStartupDriver(SimulatedArmDriver):
        def __init__(self) -> None:
            super().__init__(ArmFeedback(stale=True, received_at_s=1.0))
            self._reads = 0

        def read_feedback(self) -> ArmFeedback:
            self._reads += 1
            feedback = super().read_feedback()
            if self._reads == 1:
                return replace(feedback, stale=True, **stale_update)
            return replace(
                feedback,
                connected=True,
                enabled=True,
                stale=False,
                fault_code=0,
                joints_deg=(5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                received_at_s=2.0,
            )

    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        SynchronizingStartupDriver(),
        clock=clock,
        sleep=clock.advance,
    )

    controller.tick(dt=0.1)

    assert clock.now == pytest.approx(0.1)
    assert controller.snapshot().state is ServiceState.RUNNING
    assert controller.snapshot().emergency_stopped is False


@pytest.mark.parametrize(
    ("feedback_update", "expected_fault"),
    [
        ({"connected": False}, "right arm disconnected"),
        ({"enabled": False}, "right arm disabled"),
    ],
)
def test_startup_persistent_stale_connection_state_faults_after_timeout(
    feedback_update: dict[str, object], expected_fault: str
) -> None:
    class UnhealthyStartupDriver(SimulatedArmDriver):
        def read_feedback(self) -> ArmFeedback:
            return replace(
                super().read_feedback(),
                stale=True,
                **feedback_update,
            )

    clock = ManualClock()
    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        UnhealthyStartupDriver(ArmFeedback(received_at_s=1.0)),
        clock=clock,
        sleep=clock.advance,
    )

    controller.tick(dt=0.1)

    assert clock.now == pytest.approx(0.5)
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == expected_fault


@pytest.mark.parametrize(
    ("feedback_update", "expected_fault"),
    [
        ({"fault_code": 42}, "right arm feedback stale"),
        (
            {"joints_deg": (float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0)},
            "right arm feedback stale",
        ),
    ],
)
def test_startup_persistent_stale_hard_fault_waits_for_feedback_timeout(
    feedback_update: dict[str, object],
    expected_fault: str,
) -> None:
    class UnsafeStartupDriver(SimulatedArmDriver):
        def read_feedback(self) -> ArmFeedback:
            return replace(
                super().read_feedback(),
                stale=True,
                **feedback_update,
            )

    clock = ManualClock()
    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        UnsafeStartupDriver(ArmFeedback(received_at_s=1.0)),
        clock=clock,
        sleep=clock.advance,
    )

    controller.tick(dt=0.1)

    assert clock.now == pytest.approx(0.5)
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == expected_fault


def test_manual_homing_requires_stable_arrival_before_running() -> None:
    controller, _ = _controller()
    controller.tick(dt=0.1)

    controller.request_homing()
    assert controller.snapshot().state is ServiceState.HOMING
    controller.tick(dt=0.1)
    controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.HOMING

    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING


def test_manual_homing_timeout_runs_from_current_healthy_feedback() -> None:
    class StaticFeedbackDriver(SimulatedArmDriver):
        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            self.joint_commands.append(tuple(joints_deg))

    clock = ManualClock()
    config = _config()
    driver = StaticFeedbackDriver(
        ArmFeedback(joints_deg=(5.0, 0, 0, 0, 0, 0), received_at_s=100.0),
        auto_timestamp=True,
    )
    controller = ArmController(ArmSide.RIGHT, config, driver, clock=clock)
    controller.tick(dt=0.1)
    controller.request_homing()

    clock.advance(9.999)
    controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.HOMING
    holds_before_timeout = driver.hold_calls

    clock.advance(0.001)
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.target_q == (5.0, 0, 0, 0, 0, 0)
    assert snapshot.target_qd == (0.0,) * 6
    assert driver.hold_calls == holds_before_timeout + 1
    assert driver.joint_commands[-1] == (5.0, 0, 0, 0, 0, 0)


def test_auto_homing_timeout_runs_from_current_healthy_feedback() -> None:
    clock = ManualClock()
    config = _config(auto_initialize=True)
    driver = SimulatedArmDriver(
        ArmFeedback(joints_deg=(5.0, 0, 0, 0, 0, 0), received_at_s=100.0),
        auto_timestamp=True,
    )
    controller = ArmController(ArmSide.RIGHT, config, driver, clock=clock)

    controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.HOMING
    clock.advance(10.0)
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.target_q == snapshot.actual_q
    assert snapshot.target_qd == (0.0,) * 6


def test_stable_homing_completion_wins_over_timeout(caplog) -> None:  # type: ignore[no-untyped-def]
    clock = ManualClock()
    config = _config()
    driver = SimulatedArmDriver(
        ArmFeedback(joints_deg=(0, 0, 0, 0, 0, 0), received_at_s=100.0),
        auto_timestamp=True,
    )
    controller = ArmController(ArmSide.RIGHT, config, driver, clock=clock)
    controller.tick(dt=0.1)
    controller.request_homing()

    controller.tick(dt=0.1)
    controller.tick(dt=0.1)
    clock.advance(10.0)
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert "event=homing_timeout" not in caplog.text


@pytest.mark.parametrize(
    ("feedback_update", "expected_fault"),
    [
        ({"connected": False}, "right arm disconnected"),
        ({"enabled": False}, "right arm disabled"),
        ({"stale": True}, "right arm feedback stale"),
        ({"fault_code": 42}, "right arm fault code 42"),
    ],
)
def test_homing_timeout_never_promotes_unhealthy_feedback_to_running(
    feedback_update: dict[str, object], expected_fault: str
) -> None:
    clock = ManualClock()
    config = _config()
    driver = SimulatedArmDriver(
        ArmFeedback(joints_deg=(5.0, 0, 0, 0, 0, 0), received_at_s=100.0),
        auto_timestamp=True,
    )
    controller = ArmController(ArmSide.RIGHT, config, driver, clock=clock)
    controller.tick(dt=0.1)
    controller.request_homing()
    driver.set_feedback(replace(driver.read_feedback(), **feedback_update))

    clock.advance(10.0)
    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert snapshot.fault_reason == expected_fault


def test_running_arm_can_home_without_replaying_pending_motion() -> None:
    controller, _ = _controller()
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(3):
        controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.RUNNING

    controller.submit_servo(ServoCommand((10.0, 0, 0, 0, 0, 0), (10.0,) * 6))
    controller.submit_gripper(GripperCommand(0.04, 0.02))
    controller.tick(dt=0.1)
    gripper_before_homing = controller.snapshot().actual_gripper

    controller.submit_servo(ServoCommand((20.0, 0, 0, 0, 0, 0), (10.0,) * 6))
    controller.submit_gripper(GripperCommand(0.06, 0.02))
    controller.request_homing()

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.HOMING
    assert snapshot.target_q == (0.0,) * 6
    assert snapshot.target_qd == (0.0,) * 6
    assert snapshot.moving is False
    assert snapshot.hold is True
    assert snapshot.target_gripper == pytest.approx(gripper_before_homing)

    for _ in range(4):
        controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.RUNNING
    assert snapshot.target_q == (0.0,) * 6
    assert snapshot.target_gripper == pytest.approx(gripper_before_homing)


def test_auto_initialize_enters_homing_without_manual_request() -> None:
    controller, _ = _controller(auto_initialize=True)

    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.HOMING


@pytest.mark.parametrize("offset", [0.01, -0.01])
def test_homing_accepts_feedback_at_quantization_tolerance(offset: float) -> None:
    class StaticFeedbackDriver(SimulatedArmDriver):
        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            self.joint_commands.append(tuple(joints_deg))

    config = _config()
    driver = StaticFeedbackDriver(
        ArmFeedback(
            joints_deg=(offset, 0, 0, 0, 0, 0),
            received_at_s=100.0,
        )
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.request_homing()

    for _ in range(3):
        controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING


@pytest.mark.parametrize("offset", [0.0101, -0.0101])
def test_homing_rejects_feedback_outside_quantization_tolerance(
    offset: float,
) -> None:
    class StaticFeedbackDriver(SimulatedArmDriver):
        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            self.joint_commands.append(tuple(joints_deg))

    config = _config()
    driver = StaticFeedbackDriver(
        ArmFeedback(
            joints_deg=(offset, 0, 0, 0, 0, 0),
            received_at_s=100.0,
        )
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.request_homing()

    for _ in range(4):
        controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.HOMING


def test_unhealthy_feedback_enters_emergency_stop() -> None:
    controller, driver = _controller()
    controller.tick(dt=0.1)
    driver.set_fault(42)

    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert snapshot.emergency_stopped
    assert snapshot.fault_reason == "right arm fault code 42"
    assert driver.emergency_stop_calls == 1


@pytest.mark.parametrize(
    ("make_unhealthy", "reason"),
    [
        (lambda driver: driver.set_feedback_stale(True), "feedback stale"),
        (lambda driver: driver.set_enabled(False), "arm disabled"),
    ],
)
def test_stale_or_disabled_feedback_uses_controller_fault_entry(
    make_unhealthy, reason: str
) -> None:
    controller, driver = _controller()
    controller.tick(dt=0.1)
    make_unhealthy(driver)

    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert reason in (controller.snapshot().fault_reason or "")
    assert driver.emergency_stop_calls == 1


def test_healthy_stale_flag_cannot_mask_frozen_feedback_timestamp() -> None:
    clock = ManualClock()
    controller, _ = _controller(clock=clock)
    controller.tick(dt=0.1)
    clock.advance(0.49)
    controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.RUNNING

    clock.advance(0.02)
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == "right arm feedback timeout"
    assert not controller.snapshot().arm.feedback_fresh


def test_strictly_advancing_feedback_timestamp_keeps_resetting_watchdog() -> None:
    clock = ManualClock()
    controller, driver = _controller(clock=clock)
    controller.tick(dt=0.1)
    for received_at in (101.0, 102.0, 103.0):
        clock.advance(0.4)
        driver.set_feedback(
            ArmFeedback(
                connected=True,
                enabled=True,
                received_at_s=received_at,
            )
        )
        controller.tick(dt=0.1)

        assert controller.snapshot().state is ServiceState.RUNNING


def test_continuously_decreasing_feedback_timestamp_cannot_refresh_watchdog() -> None:
    clock = ManualClock()
    controller, driver = _controller(clock=clock)
    controller.tick(dt=0.1)
    for received_at, elapsed in ((99.0, 0.2), (98.0, 0.2), (97.0, 0.11)):
        clock.advance(elapsed)
        driver.set_feedback(
            ArmFeedback(
                connected=True,
                enabled=True,
                received_at_s=received_at,
            )
        )
        controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == "right arm feedback timeout"


def test_oscillating_old_feedback_timestamps_cannot_refresh_watchdog() -> None:
    clock = ManualClock()
    controller, driver = _controller(clock=clock)
    controller.tick(dt=0.1)
    for received_at, elapsed in ((99.0, 0.2), (100.0, 0.2), (99.5, 0.11)):
        clock.advance(elapsed)
        driver.set_feedback(
            ArmFeedback(
                connected=True,
                enabled=True,
                received_at_s=received_at,
            )
        )
        controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == "right arm feedback timeout"


def test_each_arm_has_independent_feedback_watchdog() -> None:
    clock = ManualClock()
    right = SimulatedArmDriver(ArmFeedback(received_at_s=100.0))
    left_driver = SimulatedArmDriver(ArmFeedback(received_at_s=200.0))
    right_controller = ArmController(
        ArmSide.RIGHT, _config(), right, clock=clock
    )
    left_controller = ArmController(
        ArmSide.LEFT,
        _config(side=ArmSide.LEFT),
        left_driver,
        clock=clock,
    )
    right_controller.tick(dt=0.1)
    left_controller.tick(dt=0.1)
    clock.advance(0.4)
    right.set_feedback(
        ArmFeedback(connected=True, enabled=True, received_at_s=101.0)
    )
    right_controller.tick(dt=0.1)
    left_controller.tick(dt=0.1)
    clock.advance(0.2)

    right_controller.tick(dt=0.1)
    left_controller.tick(dt=0.1)

    assert right_controller.snapshot().state is ServiceState.RUNNING
    assert left_controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert left_controller.snapshot().fault_reason == "left arm feedback timeout"


def test_shutdown_request_exposes_shutting_down_and_rejects_homing() -> None:
    controller, _ = _controller()
    controller.tick(dt=0.1)

    controller.request_shutdown()

    assert controller.snapshot().state is ServiceState.SHUTTING_DOWN
    with pytest.raises(StateConflict, match="RUNNING"):
        controller.request_homing()


def test_homing_request_waits_for_running_tick_boundary() -> None:
    entered = Event()
    release = Event()

    class BlockingCommandDriver(SimulatedArmDriver):
        block_command = False

        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            if self.block_command:
                entered.set()
                assert release.wait(1.0)
            super().command_joints(joints_deg)

    config = _config()
    driver = BlockingCommandDriver(
        ArmFeedback(joints_deg=(1, 0, 0, 0, 0, 0), received_at_s=100.0)
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    driver.block_command = True
    tick_thread = Thread(target=controller.tick, kwargs={"dt": 0.1})
    tick_thread.start()
    assert entered.wait(1.0)

    attempted = Event()
    returned = Event()

    def request_homing() -> None:
        attempted.set()
        controller.request_homing()
        returned.set()

    request_thread = Thread(target=request_homing)
    request_thread.start()
    assert attempted.wait(1.0)
    returned_before_release = returned.wait(0.05)
    release.set()
    tick_thread.join(1.0)
    request_thread.join(1.0)

    assert not returned_before_release
    assert returned.is_set()
    assert controller.snapshot().state is ServiceState.HOMING
    assert controller.snapshot().target_q == (0.0,) * 6


def test_snapshot_is_frozen() -> None:
    controller, _ = _controller()
    controller.tick(dt=0.1)
    with pytest.raises(FrozenInstanceError):
        controller.snapshot().moving = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        controller.snapshot().arm.connected = True  # type: ignore[misc]


def test_joint_command_exception_uses_controller_fault_entry() -> None:
    class FailingCommandDriver(SimulatedArmDriver):
        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            raise RuntimeError("CAN transmit failed")

    config = _config(auto_initialize=True)
    driver = FailingCommandDriver(ArmFeedback(received_at_s=100.0))
    controller = ArmController(ArmSide.RIGHT, config, driver)

    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == "arm I/O error: CAN transmit failed"
    assert driver.emergency_stop_calls == 1
