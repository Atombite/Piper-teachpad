# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest

from piper_control.config import ArmConfig, ArmSide
from piper_control.controller import ArmController
from piper_control.drivers.simulated import SimulatedArmDriver
from piper_control.lifecycle import SingletonLock, install_signal_handlers
from piper_control.types import ArmFeedback, ServiceState
from piper_control.vectors import GripperCommand, ServoCommand


ZERO6 = (0.0,) * 6


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _config(*, timeout: float = 1.0) -> ArmConfig:
    return ArmConfig.model_validate(
        {
            "server": {"host": "127.0.0.1", "port": 8765},
            "control_hz": 10,
            "feedback_timeout_s": 10.0,
            "shutdown_home_timeout_s": timeout,
            "name": "right_arm",
            "can_port": "can0",
            "auto_initialize": False,
            "receive_latency_s": 0.0,
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
    )


def _controller(
    *, timeout: float = 1.0, driver: SimulatedArmDriver | None = None
) -> tuple[ArmController, SimulatedArmDriver, ManualClock]:
    clock = ManualClock()
    actual_driver = driver or SimulatedArmDriver(
        ArmFeedback(joints_deg=(2, 0, 0, 0, 0, 0), received_at_s=1.0)
    )
    controller = ArmController(
        ArmSide.RIGHT, _config(timeout=timeout), actual_driver, clock=clock
    )
    controller.tick(dt=0.1)
    return controller, actual_driver, clock


def test_healthy_shutdown_homes_at_default_speed_then_holds() -> None:
    controller, driver, clock = _controller()
    commands_before = len(driver.joint_commands)

    result = controller.shutdown_gracefully(wait=clock.advance)

    assert result.homed
    assert result.exit_code == 0
    assert controller.snapshot().state is ServiceState.SHUTTING_DOWN
    assert driver.joint_commands[commands_before][0] == pytest.approx(1.0)
    assert driver.joint_commands[-1] == ZERO6
    assert driver.hold_calls >= 1
    assert driver.disable_calls == driver.reset_calls == 0


def test_shutdown_commands_advance_while_feedback_lags() -> None:
    class StaticFeedbackDriver(SimulatedArmDriver):
        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            self.joint_commands.append(tuple(joints_deg))

    driver = StaticFeedbackDriver(
        ArmFeedback(joints_deg=(10, 0, 0, 0, 0, 0), received_at_s=1.0)
    )
    controller, _, clock = _controller(timeout=5.0, driver=driver)
    controller.request_shutdown()

    for _ in range(4):
        clock.advance(0.1)
        controller.tick(dt=0.1)

    assert [command[0] for command in driver.joint_commands[-4:]] == [
        9.0,
        8.0,
        7.0,
        6.0,
    ]


def test_faulted_shutdown_holds_without_homing() -> None:
    controller, driver, clock = _controller()
    controller.emergency_stop("operator button")
    controller.tick(dt=0.1)
    commands_before = len(driver.joint_commands)

    result = controller.shutdown_gracefully(wait=clock.advance)

    assert not result.homed
    assert result.exit_code != 0
    assert len(driver.joint_commands) == commands_before + 1
    assert driver.hold_calls >= 1
    assert driver.disable_calls == driver.reset_calls == 0


def test_shutdown_timeout_holds_and_returns_nonzero() -> None:
    class StaticDriver(SimulatedArmDriver):
        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            self.joint_commands.append(tuple(joints_deg))

    driver = StaticDriver(
        ArmFeedback(joints_deg=(2, 0, 0, 0, 0, 0), received_at_s=1.0)
    )
    controller, _, clock = _controller(timeout=0.25, driver=driver)

    result = controller.shutdown_gracefully(wait=clock.advance)

    assert not result.homed
    assert result.timed_out
    assert result.exit_code != 0
    assert driver.hold_calls >= 1
    assert driver.joint_commands[-1] == (2.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_request_shutdown_discards_old_motion_and_homes_without_old_gripper() -> None:
    controller, driver, _ = _controller()
    controller.request_homing()
    for _ in range(5):
        controller.tick(dt=0.1)
    config = _config()
    controller.submit_servo(
        ServoCommand((20.0, 0, 0, 0, 0, 0), config.default_joints_speed_deg_s)
    )
    controller.submit_gripper(
        GripperCommand(0.05, config.gripper.default_speed_m_s)
    )
    joints_before = len(driver.joint_commands)
    grippers_before = len(driver.gripper_commands)

    controller.request_shutdown()
    for _ in range(5):
        controller.tick(dt=0.1)

    assert controller.snapshot().state is ServiceState.SHUTTING_DOWN
    assert controller.snapshot().target_q == ZERO6
    assert controller.snapshot().target_gripper == driver.read_feedback().gripper_position
    assert all(command[0] <= 1.0 for command in driver.joint_commands[joints_before:])
    assert len(driver.gripper_commands) == grippers_before
    assert driver.hold_calls >= 1


@pytest.mark.parametrize("disconnect_after_shutdown", [True, False])
def test_disconnect_cleanup_cannot_cancel_shutdown_homing(
    disconnect_after_shutdown: bool,
) -> None:
    controller, driver, clock = _controller(timeout=1.0)

    if disconnect_after_shutdown:
        controller.request_shutdown()
        controller.client_disconnected()
    else:
        controller.client_disconnected()
        controller.request_shutdown()

    assert controller.snapshot().state is ServiceState.SHUTTING_DOWN
    assert controller.snapshot().target_q == ZERO6
    for _ in range(6):
        clock.advance(0.1)
        controller.tick(dt=0.1)

    snapshot = controller.snapshot()
    assert snapshot.state is ServiceState.SHUTTING_DOWN
    assert snapshot.fault_reason is None
    assert snapshot.actual_q == ZERO6
    assert snapshot.target_q == ZERO6
    assert snapshot.hold
    assert driver.joint_commands[-1] == ZERO6


def test_run_does_not_exit_until_requested_shutdown_is_homed() -> None:
    controller, driver, clock = _controller()

    class AdvancingWakeEvent:
        def set(self) -> None:
            pass

        def clear(self) -> None:
            pass

        def wait(self, timeout: float) -> bool:
            clock.advance(timeout)
            return False

    controller._wake_event = AdvancingWakeEvent()  # type: ignore[attr-defined]
    commands_before = len(driver.joint_commands)
    controller.request_shutdown()

    controller.run()

    shutdown_commands = driver.joint_commands[commands_before:]
    assert shutdown_commands[0][0] == pytest.approx(1.0)
    assert shutdown_commands[-1] == ZERO6
    assert driver.hold_calls >= 1
    assert driver.shutdown_calls == 1


def test_run_records_driver_shutdown_failure() -> None:
    class FailingShutdownDriver(SimulatedArmDriver):
        def shutdown(self) -> None:
            super().shutdown()
            raise RuntimeError("serial close failed")

    driver = FailingShutdownDriver(
        ArmFeedback(joints_deg=(2, 0, 0, 0, 0, 0), received_at_s=1.0)
    )
    controller, _, clock = _controller(driver=driver)

    class AdvancingWakeEvent:
        def set(self) -> None:
            pass

        def clear(self) -> None:
            pass

        def wait(self, timeout: float) -> bool:
            clock.advance(timeout)
            return False

    controller._wake_event = AdvancingWakeEvent()  # type: ignore[attr-defined]
    controller.request_shutdown()
    controller.run()

    result = controller.shutdown_result()
    assert result is not None
    assert result.faulted
    assert "right driver shutdown failed: serial close failed" in result.errors
    assert driver.shutdown_calls == 1


def test_nonfinite_first_feedback_never_holds_fabricated_zero() -> None:
    driver = SimulatedArmDriver(
        ArmFeedback(joints_deg=(float("nan"), 0, 0, 0, 0, 0), received_at_s=1.0)
    )
    controller, _, clock = _controller(driver=driver)

    result = controller.shutdown_gracefully(wait=clock.advance)

    assert not result.homed and result.exit_code != 0
    assert driver.hold_calls == 0
    assert result.errors
    assert "no valid feedback" in " ".join(result.errors)


def test_nonfinite_feedback_holds_only_last_valid_position() -> None:
    controller, driver, clock = _controller()
    valid = driver.read_feedback()
    driver.set_feedback(
        replace(valid, joints_deg=(float("nan"), 0, 0, 0, 0, 0))
    )
    controller.tick(dt=0.1)
    commands_before = len(driver.joint_commands)

    result = controller.shutdown_gracefully(wait=clock.advance)

    assert not result.homed
    assert driver.hold_calls >= 1
    assert driver.joint_commands[commands_before:] == [valid.joints_deg]


def test_hold_failure_fails_controller_shutdown() -> None:
    class FailingHoldDriver(SimulatedArmDriver):
        def hold(self, feedback):  # type: ignore[no-untyped-def]
            self.hold_calls += 1
            raise RuntimeError("hold failed")

    clock = ManualClock()
    right = FailingHoldDriver(ArmFeedback(received_at_s=1.0))
    controller = ArmController(ArmSide.RIGHT, _config(), right, clock=clock)
    controller.tick(dt=0.1)
    controller.emergency_stop("operator")
    controller.tick(dt=0.1)

    result = controller.shutdown_gracefully(wait=clock.advance)

    assert result.exit_code != 0 and result.errors
    assert right.hold_calls >= 1


def test_final_hold_failure_publishes_controller_emergency() -> None:
    class FailingHoldDriver(SimulatedArmDriver):
        fail_hold = False
        def hold(self, feedback):  # type: ignore[no-untyped-def]
            self.hold_calls += 1
            if self.fail_hold:
                raise RuntimeError("final hold failed")
            super().hold(feedback)

    driver = FailingHoldDriver(ArmFeedback(received_at_s=1.0))
    controller, _, clock = _controller(driver=driver)
    driver.fail_hold = True

    result = controller.shutdown_gracefully(wait=clock.advance)

    snapshot = controller.snapshot()
    assert result.exit_code != 0
    assert snapshot.state is ServiceState.EMERGENCY_STOP
    assert "final shutdown HOLD failed" in (snapshot.fault_reason or "")
    assert driver.emergency_stop_calls >= 1


def test_blocked_shutdown_send_returns_at_timeout_and_cannot_continue_motion() -> None:
    command_entered = Event()
    release_command = Event()
    command_returned = Event()

    class BlockingCommandDriver(SimulatedArmDriver):
        block = False
        motion_attempts = 0

        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            self.motion_attempts += 1
            with self._lock:  # type: ignore[attr-defined]
                if self.block:
                    command_entered.set()
                    assert release_command.wait(1.0)
                super().command_joints(joints_deg)
            command_returned.set()

    driver = BlockingCommandDriver(
        ArmFeedback(joints_deg=(2, 0, 0, 0, 0, 0), received_at_s=1.0)
    )
    controller, _, clock = _controller(timeout=0.02, driver=driver)
    driver.block = True

    result = controller.shutdown_gracefully(wait=clock.advance)

    assert command_entered.is_set()
    assert result.timed_out and result.exit_code != 0
    release_command.set()
    assert command_returned.wait(1.0)
    controller.tick(dt=0.1)
    assert driver.motion_attempts == 1
    assert not driver.gripper_commands


def test_shutdown_deadline_applies_before_waiting_for_state_lock() -> None:
    command_entered = Event()
    release_command = Event()
    shutdown_returned = Event()
    result_box = []

    class StateLockBlockingDriver(SimulatedArmDriver):
        block = False
        motion_attempts = 0

        def command_joints(self, joints_deg):  # type: ignore[no-untyped-def]
            self.motion_attempts += 1
            with self._lock:  # type: ignore[attr-defined]
                if self.block:
                    self.block = False
                    command_entered.set()
                    assert release_command.wait(1.0)
                super().command_joints(joints_deg)

    driver = StateLockBlockingDriver(ArmFeedback(received_at_s=1.0))
    controller, _, _ = _controller(timeout=0.02, driver=driver)
    controller.request_homing()
    for _ in range(3):
        controller.tick(dt=0.1)
    attempts_before_block = driver.motion_attempts
    driver.block = True
    tick_thread = Thread(target=controller.tick, kwargs={"dt": 0.1})
    tick_thread.start()
    assert command_entered.wait(1.0)

    def shutdown() -> None:
        result_box.append(controller.shutdown_gracefully())
        shutdown_returned.set()

    shutdown_thread = Thread(target=shutdown)
    shutdown_thread.start()
    returned_before_sdk_release = shutdown_returned.wait(0.2)
    release_command.set()
    tick_thread.join(1.0)
    shutdown_thread.join(1.0)
    controller.tick(dt=0.1)

    assert returned_before_sdk_release
    assert result_box[0].timed_out and result_box[0].exit_code != 0
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert driver.motion_attempts == attempts_before_block + 1


def test_shutdown_fault_during_homing_stops_homing_and_holds() -> None:
    controller, driver, clock = _controller()
    waits = 0

    def wait_and_fault(seconds: float) -> None:
        nonlocal waits
        waits += 1
        clock.advance(seconds)
        if waits == 1:
            current = driver.read_feedback()
            driver.set_feedback(replace(current, fault_code=9, received_at_s=2.0))

    result = controller.shutdown_gracefully(wait=wait_and_fault)

    assert not result.homed
    assert result.exit_code != 0
    assert controller.snapshot().state is ServiceState.EMERGENCY_STOP
    assert driver.emergency_stop_calls == 1
    assert driver.hold_calls >= 1


def test_signal_handlers_only_set_the_supplied_event(monkeypatch) -> None:
    installed: dict[int, object] = {}
    monkeypatch.setattr(signal, "signal", lambda number, handler: installed.setdefault(number, handler))
    event = __import__("threading").Event()

    install_signal_handlers(event)
    installed[signal.SIGTERM](signal.SIGTERM, None)  # type: ignore[operator]

    assert event.is_set()
    assert set(installed) == {signal.SIGINT, signal.SIGTERM}


def test_singleton_lock_rejects_contention_and_releases_on_close(tmp_path: Path) -> None:
    path = tmp_path / "piper-control.lock"
    first = SingletonLock(path)
    second = SingletonLock(path)
    first.acquire()

    with pytest.raises(RuntimeError, match="already running"):
        second.acquire()

    first.release()
    second.acquire()
    second.release()
    assert path.read_text(encoding="ascii") == str(os.getpid())
