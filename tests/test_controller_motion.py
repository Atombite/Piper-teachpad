# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from threading import Event, Thread

import pytest

from piper_control.config import ArmConfig, ArmSide
from piper_control.controller import ArmController
from piper_control.drivers.simulated import SimulatedArmDriver
from piper_control.mailbox import LatestValue
from piper_control.types import ArmFeedback, GripperDiagnostics, ServiceState
from piper_control.vectors import GripperCommand, ServoCommand


ZERO6 = (0.0,) * 6


def _arm(
    *,
    name: str,
    side: ArmSide,
    can_port: str,
    latency: float,
    gripper: bool = True,
) -> dict[str, object]:
    return {
        "server": {"host": "127.0.0.1", "port": 8765 if side is ArmSide.RIGHT else 9876},
        "control_hz": 10,
        "feedback_timeout_s": 0.5,
        "shutdown_home_timeout_s": 30,
        "name": name,
        "can_port": can_port,
        "auto_initialize": False,
        "receive_latency_s": latency,
        "calibration_joints_pos_deg": [0, 0, 0, 0, 0, 0],
        "reset_joints_pos_deg": [0, 0, 0, 0, 0, 0],
        "min_joints_deg": [-150, 0, -170, -100, -70, -120],
        "max_joints_deg": [150, 180, 0, 100, 70, 120],
        "max_joints_speed_deg_s": [30, 30, 30, 30, 30, 30],
        "default_joints_speed_deg_s": [10, 10, 10, 10, 10, 10],
        "gripper": {
            "enabled": gripper,
            "min_position_m": 0.0,
            "max_position_m": 0.07,
            "max_speed_m_s": 0.05,
            "default_speed_m_s": 0.02,
            "effort_nm": 1.0,
        },
    }


def _config(
    *, side: ArmSide = ArmSide.RIGHT, gripper: bool = True
) -> ArmConfig:
    return ArmConfig.model_validate(
        _arm(
            name=f"{side.value}_arm",
            side=side,
            can_port="can0" if side is ArmSide.RIGHT else "can1",
            latency=0.1 if side is ArmSide.RIGHT else 0.2,
            gripper=gripper,
        )
    )


def test_piper_controller_uses_meter_speed_lead_and_diagnostics() -> None:
    class LaggingGripperDriver(SimulatedArmDriver):
        def command_gripper(self, position: float) -> None:
            self.gripper_commands.append(float(position))

    diagnostics = GripperDiagnostics(
        backend="piper",
        enabled=True,
        communication_ok=True,
        fault_code=0,
        warning_code=0,
        feedback_age_s=0.02,
    )
    driver = LaggingGripperDriver(
        ArmFeedback(
            gripper_position=0.05,
            gripper=diagnostics,
            received_at_s=100.0,
        )
    )
    controller = ArmController(ArmSide.LEFT, _config(side=ArmSide.LEFT), driver)
    controller.tick(dt=0.1)
    controller.submit_gripper(GripperCommand(0.0, 0.02))
    controller.tick(dt=0.1)

    assert driver.gripper_commands[-1] == pytest.approx(0.048)
    payload = controller.snapshot().to_json()
    assert payload["ActualGripper"] == pytest.approx(0.05)
    assert payload["TargetGripper"] == pytest.approx(0.0)
    assert payload["gripper"] == {
        "backend": "piper",
        "enabled": True,
        "communication_ok": True,
        "fault_code": 0,
        "warning_code": 0,
        "feedback_age_s": 0.02,
    }


def test_disconnect_holds_latest_gripper_scalar_and_discards_pending_command() -> None:
    class RecordingHoldDriver(SimulatedArmDriver):
        def __init__(self, feedback: ArmFeedback) -> None:
            super().__init__(feedback)
            self.held_grippers: list[float] = []

        def hold(self, feedback: ArmFeedback) -> None:
            self.held_grippers.append(feedback.gripper_position)
            super().hold(feedback)

    driver = RecordingHoldDriver(
        ArmFeedback(gripper_position=0.042, received_at_s=100.0)
    )
    controller = ArmController(ArmSide.LEFT, _config(side=ArmSide.LEFT), driver)
    controller.tick(dt=0.1)
    controller.submit_gripper(GripperCommand(0.01, 0.02))
    controller.client_disconnected()
    controller.tick(dt=0.1)

    driver.set_feedback(
        ArmFeedback(
            gripper_position=0.030,
            connected=True,
            enabled=True,
            received_at_s=101.0,
        )
    )
    controller.tick(dt=0.1)

    assert driver.held_grippers[-1] == pytest.approx(0.042)
    assert driver.gripper_commands == []
    assert controller.snapshot().target_gripper == pytest.approx(0.042)


def test_explicit_hold_keeps_latched_gripper_target_across_feedback_drift() -> None:
    class RecordingHoldDriver(SimulatedArmDriver):
        def __init__(self, feedback: ArmFeedback) -> None:
            super().__init__(feedback)
            self.held_grippers: list[float] = []

        def hold(self, feedback: ArmFeedback) -> None:
            self.held_grippers.append(feedback.gripper_position)
            super().hold(feedback)

    driver = RecordingHoldDriver(
        ArmFeedback(gripper_position=0.042, received_at_s=100.0)
    )
    controller = ArmController(ArmSide.LEFT, _config(side=ArmSide.LEFT), driver)
    controller.tick(dt=0.1)
    controller.request_hold()

    driver.set_feedback(
        ArmFeedback(
            gripper_position=0.030,
            connected=True,
            enabled=True,
            received_at_s=101.0,
        )
    )
    controller.tick(dt=0.1)
    controller.request_hold()
    controller.client_disconnected()

    assert driver.held_grippers[-1] == pytest.approx(0.042)
    assert controller.snapshot().target_gripper == pytest.approx(0.042)


def test_focus_hold_clears_stale_piper_gripper_speed() -> None:
    driver = SimulatedArmDriver(
        ArmFeedback(gripper_position=0.05, received_at_s=100.0)
    )
    config = _config(side=ArmSide.LEFT)
    controller = ArmController(ArmSide.LEFT, config, driver)
    controller.tick(dt=0.1)
    controller.submit_gripper(GripperCommand(0.01, 0.02))
    controller.tick(dt=0.1)
    driver.set_feedback(
        ArmFeedback(
            gripper_position=0.06,
            connected=True,
            enabled=True,
            received_at_s=101.0,
        )
    )

    controller.client_disconnected()
    controller.tick(dt=0.1)
    commands_after_hold = len(driver.gripper_commands)
    controller.submit_servo(
        ServoCommand(ZERO6, config.default_joints_speed_deg_s)
    )
    controller.tick(dt=0.1)

    assert len(driver.gripper_commands) == commands_after_hold + 1
    assert driver.gripper_commands[-1] == pytest.approx(0.06)
    assert controller.snapshot().target_gripper_speed == 0.0


def test_recovery_hold_clears_stale_piper_gripper_speed() -> None:
    driver = SimulatedArmDriver(
        ArmFeedback(gripper_position=0.05), auto_timestamp=True
    )
    config = _config(side=ArmSide.LEFT)
    controller = ArmController(ArmSide.LEFT, config, driver)
    controller.tick(dt=0.1)
    controller.submit_gripper(GripperCommand(0.01, 0.02))
    controller.tick(dt=0.1)
    current = driver.read_feedback()
    driver.set_feedback(
        ArmFeedback(
            gripper_position=0.06,
            connected=True,
            enabled=True,
            received_at_s=current.received_at_s,
        )
    )

    controller.emergency_stop("test")
    controller.tick(dt=0.1)
    controller.request_recovery()
    controller.tick(dt=0.1)
    commands_after_hold = len(driver.gripper_commands)
    controller.submit_servo(
        ServoCommand(ZERO6, config.default_joints_speed_deg_s)
    )
    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert len(driver.gripper_commands) == commands_after_hold + 1
    assert driver.gripper_commands[-1] == pytest.approx(0.06)
    assert controller.snapshot().target_gripper_speed == 0.0


def _running(
    *, side: ArmSide = ArmSide.RIGHT
) -> tuple[ArmController, ArmConfig, dict[str, SimulatedArmDriver]]:
    config = _config(side=side)
    driver = SimulatedArmDriver(
        ArmFeedback(received_at_s=100.0 if side is ArmSide.RIGHT else 99.0)
    )
    drivers = {side.value: driver}
    controller = ArmController(side, config, driver)
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(3):
        controller.tick(dt=0.1)
    assert controller.snapshot().state is ServiceState.RUNNING
    return controller, config, drivers


def test_new_servo_target_overwrites_old_and_reports_signed_step_velocity() -> None:
    controller, config, _ = _running()
    first = (30.0, 0, 0, 0, 0, 0)
    latest = (-10.0, 0, 0, 0, 0, 0)
    controller.submit_servo(ServoCommand(first, (10.0,) * 6))
    controller.submit_servo(ServoCommand(latest, (5.0,) * 6))

    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.target_q == latest
    assert snapshot.target_qd == (-5.0, 0, 0, 0, 0, 0)


def test_lagging_feedback_allows_command_to_accumulate_to_five_degree_lead() -> None:
    class LaggingDriver(SimulatedArmDriver):
        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            self.joint_commands.append(tuple(joints_deg))

    config = _config()
    driver = LaggingDriver(ArmFeedback(received_at_s=100.0))
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.submit_servo(
        ServoCommand((30.0, 0, 0, 0, 0, 0), (20.0,) * 6)
    )

    for _ in range(4):
        controller.tick(dt=0.1)

    commanded_joint_1 = [command[0] for command in driver.joint_commands[-4:]]
    assert commanded_joint_1 == pytest.approx([2.0, 4.0, 5.0, 5.0])


def test_lagging_gripper_feedback_allows_command_to_accumulate_to_five_mm_lead() -> None:
    class LaggingGripperDriver(SimulatedArmDriver):
        def command_gripper(self, position_m: float) -> None:
            self.gripper_commands.append(float(position_m))

    config = _config()
    driver = LaggingGripperDriver(
        ArmFeedback(gripper_position=0.07, received_at_s=100.0)
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.submit_gripper(GripperCommand(0.0, 0.02))

    for _ in range(4):
        controller.tick(dt=0.1)

    assert driver.gripper_commands[-4:] == pytest.approx(
        [0.068, 0.066, 0.065, 0.065]
    )


def test_auto_initialization_holds_the_feedback_gripper_position() -> None:
    class LaggingGripperDriver(SimulatedArmDriver):
        def command_gripper(self, position_m: float) -> None:
            self.gripper_commands.append(float(position_m))

    config = _config().model_copy(update={"auto_initialize": True})
    driver = LaggingGripperDriver(
        ArmFeedback(gripper_position=0.07, received_at_s=100.0)
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)

    controller.tick(dt=0.1)

    assert driver.gripper_commands[-1] == pytest.approx(0.07)
    assert controller.snapshot().target_gripper == pytest.approx(0.07)
    assert controller.snapshot().target_gripper_speed == 0.0


def test_disconnect_hold_resynchronizes_the_gripper_command_accumulator() -> None:
    class LaggingGripperDriver(SimulatedArmDriver):
        def command_gripper(self, position_m: float) -> None:
            self.gripper_commands.append(float(position_m))

    config = _config()
    driver = LaggingGripperDriver(
        ArmFeedback(gripper_position=0.07, received_at_s=100.0)
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.submit_gripper(GripperCommand(0.0, 0.02))
    for _ in range(3):
        controller.tick(dt=0.1)
    assert driver.gripper_commands[-1] == pytest.approx(0.065)

    controller.client_disconnected()
    controller.submit_gripper(GripperCommand(0.0, 0.02))
    controller.tick(dt=0.1)

    assert driver.gripper_commands[-1] == pytest.approx(0.068)


def test_homing_resynchronizes_and_holds_the_feedback_gripper_position() -> None:
    class LaggingGripperDriver(SimulatedArmDriver):
        def command_gripper(self, position_m: float) -> None:
            self.gripper_commands.append(float(position_m))

    config = _config()
    driver = LaggingGripperDriver(
        ArmFeedback(gripper_position=0.07, received_at_s=100.0)
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.submit_gripper(GripperCommand(0.0, 0.02))
    for _ in range(3):
        controller.tick(dt=0.1)
    assert driver.gripper_commands[-1] == pytest.approx(0.065)

    controller.request_homing()
    controller.tick(dt=0.1)

    assert driver.gripper_commands[-1] == pytest.approx(0.07)


def test_gripper_command_does_not_change_joint_target() -> None:
    controller, config, _ = _running()
    joint_target = (20.0, 0, 0, 0, 0, 0)
    controller.submit_servo(ServoCommand(joint_target, config.default_joints_speed_deg_s))
    controller.tick(dt=0.1)
    before = controller.snapshot().target_q

    controller.submit_gripper(GripperCommand(0.02, config.gripper.default_speed_m_s))
    controller.tick(dt=0.1)

    assert controller.snapshot().target_q == before


def test_each_snapshot_has_single_arm_shape_and_timestamp() -> None:
    right_controller, _, right_drivers = _running()
    left_controller, _, left_drivers = _running(side=ArmSide.LEFT)
    right_drivers["right"].set_feedback(
        ArmFeedback(
            joints_deg=(1, 2, -3, 4, -5, 6),
            gripper_position=0.01,
            tcp_pose=(1, 2, 3, 4, 5, 6),
            connected=True,
            enabled=True,
            received_at_s=100.0,
        )
    )
    left_drivers["left"].set_feedback(
        ArmFeedback(
            joints_deg=(-1, 2, -3, -4, 5, -6),
            gripper_position=0.02,
            tcp_pose=(6, 5, 4, 3, 2, 1),
            connected=True,
            enabled=True,
            received_at_s=99.0,
        )
    )

    right_controller.tick(dt=0.1)
    left_controller.tick(dt=0.1)

    right_snapshot = right_controller.snapshot()
    left_snapshot = left_controller.snapshot()
    assert right_snapshot.actual_q == (1, 2, -3, 4, -5, 6)
    assert left_snapshot.actual_q == (-1, 2, -3, -4, 5, -6)
    assert right_snapshot.actual_tcp_pose == (1, 2, 3, 4, 5, 6)
    assert left_snapshot.actual_tcp_pose == (6, 5, 4, 3, 2, 1)
    assert right_snapshot.actual_tcp_speed == ZERO6
    assert right_snapshot.actual_gripper == 0.01
    assert left_snapshot.actual_gripper == 0.02
    assert right_snapshot.robot_receive_timestamp == 100.0
    assert left_snapshot.robot_timestamp == pytest.approx(98.8)
    assert right_snapshot.arm.robot_timestamp == pytest.approx(99.9)


def test_independent_arm_servo_sends_only_to_its_driver() -> None:
    right_controller, _, right_drivers = _running()
    left_controller, _, left_drivers = _running(side=ArmSide.LEFT)
    right_before = len(right_drivers["right"].joint_commands)
    left_before = len(left_drivers["left"].joint_commands)
    right_controller.submit_servo(
        ServoCommand((10.0, 0, 0, 0, 0, 0), (10.0,) * 6)
    )

    right_controller.tick(dt=0.1)

    assert len(right_drivers["right"].joint_commands) == right_before + 1
    assert len(left_drivers["left"].joint_commands) == left_before
    assert right_drivers["right"].joint_commands[-1][0] == pytest.approx(1.0)


def test_arm_send_failure_emergency_stops_only_its_controller() -> None:
    class FailingLeftDriver(SimulatedArmDriver):
        fail_command = False

        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            if self.fail_command:
                raise RuntimeError("left send failed")
            super().command_joints(joints_deg)

    config = _config(side=ArmSide.LEFT)
    right = SimulatedArmDriver(ArmFeedback(received_at_s=100.0))
    left = FailingLeftDriver(ArmFeedback(received_at_s=99.0))
    controller = ArmController(ArmSide.LEFT, config, left)
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(3):
        controller.tick(dt=0.1)
    left.fail_command = True

    controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == "arm I/O error: left send failed"
    assert right.emergency_stop_calls == 0
    assert left.emergency_stop_calls == 1


def test_single_arm_snapshot_has_no_missing_side_placeholders() -> None:
    controller, _, _ = _running()

    snapshot = controller.snapshot()

    assert len(snapshot.actual_q) == 6
    assert len(snapshot.actual_qd) == 6
    assert len(snapshot.actual_tcp_pose) == 6
    assert len(snapshot.target_q) == 6
    assert snapshot.actual_gripper == 0.0
    assert snapshot.target_gripper == 0.0


def test_disabled_gripper_is_zero_filled_even_if_driver_reports_a_value() -> None:
    config = _config(gripper=False)
    driver = SimulatedArmDriver(
        ArmFeedback(gripper_position=0.04, received_at_s=100.0)
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(3):
        controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.actual_gripper == 0.0
    assert snapshot.actual_gripper_speed == 0.0
    assert snapshot.target_gripper == 0.0
    assert snapshot.target_gripper_speed == 0.0


def test_actual_velocities_use_consecutive_feedback_and_capped_dt() -> None:
    controller, _, drivers = _running()
    drivers["right"].set_feedback(
        ArmFeedback(
            joints_deg=(2, 0, 0, 0, 0, 0),
            gripper_position=0.01,
            connected=True,
            enabled=True,
            received_at_s=101.0,
        )
    )

    controller.tick(dt=1.0)

    snapshot = controller.snapshot()
    assert snapshot.actual_qd == pytest.approx((10, 0, 0, 0, 0, 0))
    assert snapshot.actual_gripper_speed == pytest.approx(0.05)


def test_first_feedback_sample_has_zero_actual_velocity() -> None:
    config = _config()
    driver = SimulatedArmDriver(
        ArmFeedback(
            joints_deg=(12, 0, -3, 4, -5, 6),
            gripper_position=0.03,
            received_at_s=100.0,
        )
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)

    controller.tick(dt=0.1)

    assert controller.snapshot().actual_qd == ZERO6
    assert controller.snapshot().actual_gripper_speed == 0.0


def test_interpolation_caps_dt_and_repeated_overrun_uses_fault_entry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="piper_control.controller")
    controller, config, drivers = _running()
    target = (30.0, 0, 0, 0, 0, 0)
    controller.submit_servo(ServoCommand(target, (10.0,) * 6))

    controller.tick(dt=0.3)

    assert drivers["right"].joint_commands[-1][0] == pytest.approx(2.0)
    controller.tick(dt=0.3)
    controller.tick(dt=0.3)
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert controller.snapshot().fault_reason == "control loop overrun"
    events = [getattr(record, "event", None) for record in caplog.records]
    assert events.count("control_overrun") == 1
    assert events.count("control_overrun_fault") == 1


def test_feedback_timeout_sized_cycle_delay_faults_on_first_occurrence() -> None:
    class ManualClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    class DelayedFeedbackDriver(SimulatedArmDriver):
        delay_s = 0.0

        def read_feedback(self) -> ArmFeedback:
            clock.advance(self.delay_s)
            return super().read_feedback()

    clock = ManualClock()
    driver = DelayedFeedbackDriver(ArmFeedback(received_at_s=100.0))
    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        driver,
        clock=clock,
    )
    controller.tick(dt=0.1)
    driver.delay_s = 0.5

    controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert snapshot.fault_reason == "control loop overrun"
    assert driver.emergency_stop_calls == 1


def test_scheduler_rebases_after_a_blocked_tick_instead_of_catching_up() -> None:
    class StopLoop(Exception):
        pass

    class ManualClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    class AdvancingWakeEvent:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def wait(self, delay: float) -> bool:
            self.waits.append(delay)
            clock.advance(delay)
            return False

        def clear(self) -> None:
            pass

    clock = ManualClock()
    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        SimulatedArmDriver(ArmFeedback(received_at_s=100.0)),
        clock=clock,
    )
    wake_event = AdvancingWakeEvent()
    controller._wake_event = wake_event  # type: ignore[assignment]
    starts: list[float] = []
    elapsed: list[float] = []

    def blocking_tick(*, dt: float) -> ArmStateSnapshot:
        starts.append(clock())
        elapsed.append(dt)
        if len(starts) == 1:
            clock.advance(2.0)
            return controller.snapshot()
        raise StopLoop

    controller.tick = blocking_tick  # type: ignore[method-assign]

    with pytest.raises(StopLoop):
        controller._run_until_shutdown()

    assert starts == pytest.approx([0.0, 2.1])
    assert elapsed == pytest.approx([0.1, 0.1])
    assert wake_event.waits == pytest.approx([0.1])


def test_scheduler_bounds_wait_after_clock_moves_backwards() -> None:
    class StopLoop(Exception):
        pass

    class ManualClock:
        now = 100.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    class AdvancingWakeEvent:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def wait(self, delay: float) -> bool:
            self.waits.append(delay)
            clock.advance(delay)
            return False

        def clear(self) -> None:
            pass

    clock = ManualClock()
    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        SimulatedArmDriver(ArmFeedback(received_at_s=100.0)),
        clock=clock,
    )
    wake_event = AdvancingWakeEvent()
    controller._wake_event = wake_event  # type: ignore[assignment]
    ticks = 0

    def jumping_tick(*, dt: float) -> ArmStateSnapshot:
        nonlocal ticks
        ticks += 1
        if ticks == 1:
            clock.now = -100.0
            return controller.snapshot()
        raise StopLoop

    controller.tick = jumping_tick  # type: ignore[method-assign]

    with pytest.raises(StopLoop):
        controller._run_until_shutdown()

    assert wake_event.waits == pytest.approx([0.1])


def test_scheduler_rebases_after_repeated_early_wakeups() -> None:
    class StopLoop(Exception):
        pass

    class ManualClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    class EarlyWakeEvent:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def wait(self, delay: float) -> bool:
            self.waits.append(delay)
            return True

        def clear(self) -> None:
            pass

    clock = ManualClock()
    controller = ArmController(
        ArmSide.RIGHT,
        _config(),
        SimulatedArmDriver(ArmFeedback(received_at_s=100.0)),
        clock=clock,
    )
    wake_event = EarlyWakeEvent()
    controller._wake_event = wake_event  # type: ignore[assignment]
    ticks = 0

    def interrupt_after_three_ticks(*, dt: float) -> ArmStateSnapshot:
        nonlocal ticks
        ticks += 1
        if ticks == 3:
            raise StopLoop
        return controller.snapshot()

    controller.tick = interrupt_after_three_ticks  # type: ignore[method-assign]

    with pytest.raises(StopLoop):
        controller._run_until_shutdown()

    assert wake_event.waits == pytest.approx([0.1, 0.1])


def test_shutdown_return_waits_until_inflight_motion_send_finishes() -> None:
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
    driver = BlockingCommandDriver(ArmFeedback(received_at_s=100.0))
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(3):
        controller.tick(dt=0.1)
    controller.submit_servo(
        ServoCommand((30.0, 0, 0, 0, 0, 0), (10.0,) * 6)
    )
    driver.block_command = True
    tick_thread = Thread(target=controller.tick, kwargs={"dt": 0.1})
    tick_thread.start()
    assert entered.wait(1.0)

    attempted = Event()
    returned = Event()
    commands_at_return: list[int] = []

    def shutdown() -> None:
        attempted.set()
        controller.request_shutdown()
        commands_at_return.append(len(driver.joint_commands))
        returned.set()

    shutdown_thread = Thread(target=shutdown)
    shutdown_thread.start()
    assert attempted.wait(1.0)
    returned_before_release = returned.wait(0.05)
    release.set()
    tick_thread.join(1.0)
    shutdown_thread.join(1.0)

    assert not returned_before_release
    assert commands_at_return == [len(driver.joint_commands)]
    assert controller.snapshot().state is ServiceState.SHUTTING_DOWN


def test_fault_clear_cannot_race_late_submit_into_recovered_motion() -> None:
    put_entered = Event()
    put_release = Event()
    emergency_called = Event()

    class BlockingPutMailbox(LatestValue):
        def put(self, value):  # type: ignore[no-untyped-def]
            put_entered.set()
            assert put_release.wait(1.0)
            super().put(value)

    class ControlledFailureDriver(SimulatedArmDriver):
        fail_command = False

        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            if self.fail_command:
                raise RuntimeError("send failed")
            super().command_joints(joints_deg)

        def emergency_stop(self) -> None:
            emergency_called.set()
            super().emergency_stop()

    config = _config()
    driver = ControlledFailureDriver(
        ArmFeedback(received_at_s=100.0), auto_timestamp=True
    )
    controller = ArmController(ArmSide.RIGHT, config, driver)
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(3):
        controller.tick(dt=0.1)

    controller._servo_mailbox = BlockingPutMailbox()  # type: ignore[attr-defined]
    command = ServoCommand((30.0, 0, 0, 0, 0, 0), (10.0,) * 6)
    submit_thread = Thread(target=controller.submit_servo, args=(command,))
    submit_thread.start()
    assert put_entered.wait(1.0)
    driver.fail_command = True
    tick_thread = Thread(target=controller.tick, kwargs={"dt": 0.1})
    tick_thread.start()
    fault_before_submit_release = emergency_called.wait(0.05)
    put_release.set()
    submit_thread.join(1.0)
    tick_thread.join(1.0)

    assert not fault_before_submit_release
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    driver.fail_command = False
    controller.request_recovery()
    controller.tick(dt=0.1)
    controller.request_homing()
    for _ in range(4):
        controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.RUNNING
    assert controller.snapshot().target_q == ZERO6
